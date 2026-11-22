"""Freeze the validation Pareto set, train freshly, and test official challenging."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sota.FRED_LLM_GE.phase1_detection import map_reporter
from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn_scratch import FasterRCNNScratchAdapter
from sota.FRED_LLM_GE.phase1_detection.candidate_runner import DEFAULT_PROTOCOL, FRED_DIR, load_protocol
from sota.FRED_LLM_GE.phase1_detection.export_split import dataset_parts
from sota.FRED_LLM_GE.phase1_detection.faster_rcnn_baseline import identity, summarize
from sota.FRED_LLM_GE.phase1_detection.faster_rcnn_search_space import built_architecture
from sota.FRED_LLM_GE.phase1_detection.final_eval import predict_test_split
from sota.FRED_LLM_GE.phase1_detection.fred_splits import challenging_split
from sota.FRED_LLM_GE.phase1_detection.proxy_data import REVISION, load_exported_split
from sota.FRED_LLM_GE.phase1_detection.result_contract import validate_gene_id
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import (
    file_sha256, freeze_manifest, load_generation, pareto_front)
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.train_eval import (
    set_random_seed, write_json)


def freeze_finalists(run_root):
    run_root = Path(run_root).resolve()
    fingerprint = identity(run_root)
    summarize(run_root, expected_fingerprint=fingerprint)
    checkpoint, archive, records = load_generation(run_root / "checkpoints", 20)
    status = json.loads((run_root / "status.json").read_text())
    if status.get("status") != "complete" or status.get("last_completed_generation") != 20:
        raise ValueError("Freeze requires a successfully completed generation-20 run")
    if not (run_root / "checkpoints/checkpoint_gen_20.pkl").is_file():
        raise FileNotFoundError("Completed generation 20 requires both checkpoint files")
    archived = archive.get("protocol_fingerprint", {})
    if {k: v for k, v in archived.items() if k != "evolution_settings"} != fingerprint:
        raise ValueError("Generation-20 archive has incompatible provenance")
    front = pareto_front(records)
    selected = list(front)
    if not any(g == "seed" for g, _ in selected):
        stock = summarize(run_root)["runs"][0]
        selected.append(("seed", (stock["mAP50"], stock["mAP50_95"], stock["params"])))
    finalists = []
    output = run_root / "final"
    for gene_id, fitness in selected:
        validate_gene_id(gene_id)
        source = (FRED_DIR / "seeds/seed_faster_rcnn_scratch.py" if gene_id == "seed"
                  else run_root / "candidates/models" / f"network_{gene_id}.py")
        architecture, params = built_architecture(source)
        if params != int(fitness[2]):
            raise ValueError(f"Archived parameter count differs for {gene_id}")
        # Equal architectures share one final run, but retain every archive ID.
        alias = next((f for f in finalists if f["architecture_sha256"] == architecture), None)
        if alias is not None:
            alias["aliases"].append(gene_id)
            if gene_id == "seed":
                alias["stock_control"] = True
            continue
        target = output / "sources" / f"network_{gene_id}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != file_sha256(source):
            raise ValueError(f"Frozen source changed: {target}")
        if not target.exists():
            # Exclusive creation prevents accidental replacement of a frozen source.
            with target.open("x") as handle:
                handle.write(source.read_text())
        finalists.append({"gene_id": gene_id, "aliases": [gene_id],
                          "stock_control": gene_id == "seed",
                          "pareto_front": any(g == gene_id for g, _ in front),
                          "validation_fitness": list(fitness), "params": params,
                          "architecture_sha256": architecture,
                          "source_sha256": file_sha256(source),
                          "source": str(target.relative_to(run_root)),
                          "ancestry": archive.get("GLOBAL_DATA_ANCESTRY", {}).get(gene_id, {})})
    protocol = load_protocol(DEFAULT_PROTOCOL)
    training = dict(protocol["families"]["faster_rcnn_scratch"]["final"])
    selection = {"schema_version": "fred-faster-rcnn-pareto-v1", "generation": 20,
                 "rule": ["maximize_mAP50", "maximize_mAP50_95", "minimize_parameters"],
                 "initialization": "fresh_framework_random", "identity": fingerprint,
                 "archive_sha256": file_sha256(checkpoint), "training": training,
                 "population_checkpoint_sha256": file_sha256(run_root / "checkpoints/checkpoint_gen_20.pkl"),
                 "final_sampling": protocol["challenging"]["final"],
                 "finalists": finalists}
    freeze_manifest(output / "finalists.json", selection)
    return selection


def checkpoint_matches(saved, expected):
    if saved.get("provenance") != expected:
        raise ValueError("Final checkpoint differs from frozen training protocol")


def evaluate_finalist(run_root, selection, finalist, device):
    run_root = Path(run_root)
    output = run_root / "final" / finalist["gene_id"]
    output.mkdir(parents=True, exist_ok=True)
    source = run_root / finalist["source"]
    if file_sha256(source) != finalist["source_sha256"]:
        raise ValueError("Frozen finalist source changed")
    protocol = load_protocol(DEFAULT_PROTOCOL)
    family = protocol["families"]["faster_rcnn_scratch"]
    settings = dict(protocol["proxy"], **selection["training"])
    family = dict(family, search=selection["training"])
    settings.update(train_frame_step=protocol["challenging"]["final"]["train_frame_step"],
                    test_frame_step=protocol["challenging"]["final"]["test_frame_step"],
                    events=protocol["challenging"]["events"])
    provenance = {"selection_sha256": file_sha256(run_root / "final/finalists.json"),
                  "architecture_sha256": finalist["architecture_sha256"],
                  "source_sha256": finalist["source_sha256"], "settings": settings,
                  "family": family}
    weights = output / "model.pt"
    adapter = FasterRCNNScratchAdapter(family, FRED_DIR)
    set_random_seed(settings["seed"])
    model = adapter.build(source, 1, device)
    data_root = FRED_DIR / protocol["challenging"]["data_root"]
    (train_root, train, step), = dataset_parts(protocol["challenging"], data_root, "final")
    if not (train_root / "data.yaml").is_file():
        raise FileNotFoundError("Prepare and finalize the official final training export first")
    if len(train) != 172:
        raise ValueError("Expected 172 official training sequences")
    markers = {s: file_sha256(train_root / "done" / f"{s}.json") for s, _ in train}
    provenance["final_export_sha256"] = hashlib.sha256(
        json.dumps(markers, sort_keys=True).encode()).hexdigest()
    if weights.exists():
        saved = torch.load(weights, map_location="cpu", weights_only=False)
        checkpoint_matches(saved, provenance)
        model.load_state_dict(saved["state_dict"], strict=True)
        if saved["initialization"] != adapter.initialization:
            raise ValueError("Fresh initialization identity changed")
    else:
        data = load_exported_split(train_root, train, expected_step=step)
        progress = output / "training.pt"
        resume_state = None
        if progress.exists():
            resume_state = torch.load(progress, map_location="cpu", weights_only=False)
            checkpoint_matches(resume_state, provenance)
            if resume_state["initialization"] != adapter.initialization:
                raise ValueError("Training resume initialization differs")
            model.load_state_dict(resume_state["state_dict"], strict=True)

        def save_epoch(completed_epochs, optimizer):
            temporary = progress.with_suffix(f".{os.getpid()}.tmp")
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "completed_epochs": completed_epochs, "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "provenance": provenance, "initialization": adapter.initialization}, temporary)
            os.replace(temporary, progress)

        adapter.train(model, data, settings, device, resume_state=resume_state, on_epoch=save_epoch)
        temporary = weights.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"state_dict": model.state_dict(), "provenance": provenance,
                    "initialization": adapter.initialization}, temporary)
        os.replace(temporary, weights)
    _, test = challenging_split(data_root)
    if len(test) != 59:
        raise ValueError("Expected 59 official test sequences")
    frames, expected = [], []
    checkpoint_hash = file_sha256(weights)
    for sequence, folder in test:
        chunk = output / "predictions" / f"{sequence}.json"
        chunk_identity = {"checkpoint_sha256": checkpoint_hash, "provenance": provenance,
                          "sequence": sequence, "folder": folder}
        if chunk.exists():
            record = json.loads(chunk.read_text())
            if record["identity"] != chunk_identity:
                raise ValueError("Cached test predictions have incompatible provenance")
        else:
            sequence_frames, ids = predict_test_split(
                adapter, model, [(sequence, folder)], settings, device, output)
            if not ids or len(sequence_frames) != len(ids):
                raise ValueError(f"Incomplete test sequence: {sequence}")
            record = {"identity": chunk_identity, "frames": sequence_frames, "expected": ids}
            write_json(chunk, record)
        frames.extend(record["frames"])
        expected.extend(record["expected"])
    metadata = {
        "dataset_revision": REVISION, "manifest_identity": selection["identity"]["sha256"],
        "split_identity": "official_challenging_train:172;official_challenging_test:59",
        "annotation_policy": "fred_coordinates_txt_clipped_to_image",
        "model_identity": finalist["source_sha256"], "checkpoint_identity": checkpoint_hash,
        "code_revision": selection["identity"]["git_commit"],
        "postprocessing": adapter.postprocessing(settings),
        "training_config_identity": hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        "modality": "event", "benchmark_split": "challenging",
        "project_split": "held_out_test", "official_split": "challenging_test",
        "initialization_policy": "fresh_framework_random",
        "initial_state_sha256": adapter.initialization["initial_state_sha256"],
        "finalists_manifest_sha256": provenance["selection_sha256"]}
    report_root = run_root / "final"
    report_path = report_root / "held_out_reports" / f"{finalist['gene_id']}_evaluation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["metadata"] != metadata or report["num_frames"] != len(expected):
            raise ValueError("Existing held-out report differs from the frozen run")
        return report
    _, report_path = map_reporter.report(
        {"metadata": metadata, "frames": frames, "expected_sample_ids": expected,
         "params": finalist["params"]}, sota_root=report_root,
        gene_id=finalist["gene_id"], mode="held-out")
    return json.loads(report_path.read_text())


def run(run_root, device="cuda", freeze_only=False):
    run_root = Path(run_root).resolve()
    output = run_root / "final"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = freeze_finalists(run_root)
        if freeze_only:
            return selection
        rows = []
        for finalist in selection["finalists"]:
            report = evaluate_finalist(run_root, selection, finalist, device)
            rows.append({"gene_id": finalist["gene_id"], "aliases": ",".join(finalist["aliases"]),
                         "pareto_front": finalist["pareto_front"],
                         "stock_control": finalist["stock_control"], **report["metrics"],
                         "test_frames": report["num_frames"]})
        temporary = output / "comparison.csv.tmp"
        with temporary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "comparison.csv")
        write_json(output / "status.json", {"status": "complete",
                   "last_completed_generation": 20, "finalists": len(rows),
                   "selection_sha256": file_sha256(output / "finalists.json")})
        return rows


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    run(args.run_root, args.device, args.freeze_only)


if __name__ == "__main__":
    cli()
