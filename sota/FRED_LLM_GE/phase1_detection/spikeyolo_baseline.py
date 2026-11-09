"""Independent SpikeYOLO scratch baseline: run seeds 0/1/2, then summarize."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .protocol_fingerprint import compute_fingerprint

FRED = Path(__file__).resolve().parents[1]
PROTOCOL = FRED / "configs/phase1/protocol.yaml"
SEED = FRED / "seeds/seed_spikeyolo.yaml"
DEFAULT_OUTPUT = FRED / "baseline_validation/phase1_detection/spikeyolo_stock"
ARTIFACTS = (
    "results/seed_results.csv", "results/seed_evaluation.json", "trained_models/seed.pt",
    "training_args/seed/args.yaml", "training_args/seed/official_args.yaml",
    "training_args/seed/model.json", "training_args/seed/runtime.json", "training_args/seed/run.json",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    return compute_fingerprint(FRED.parents[1], "spikeyolo", "stock_spikeyolo", SEED)


def validate(root, seed):
    results_path = root / "results/seed_results.csv"
    if results_path.is_file():
        with results_path.open() as stream:
            rows = list(csv.reader(stream))
        if len(rows) != 1:
            raise ValueError(f"invalid result rows: {root}")
        if rows[0][:3] == ["-inf", "-inf", "inf"]:
            reason = rows[0][3] if len(rows[0]) > 3 else "missing failure reason"
            raise ValueError(f"failed SpikeYOLO baseline: {reason}")

    report_path = root / "results/seed_evaluation.json"
    report = json.loads(report_path.read_text())
    metadata = report["metadata"]
    args_path = root / "training_args/seed/args.yaml"
    args = yaml.safe_load(args_path.read_text())
    protocol = yaml.safe_load(PROTOCOL.read_text())
    family = protocol["families"]["spikeyolo"]
    if (report["mode"] != "validation" or report["gene_id"] != "seed"
            or metadata["model_identity"] != digest(SEED)
            or metadata["official_split"] != "challenging_train"
            or metadata["input_representation"] != "released_event_frame_MS_GetT_T1"
            or metadata["upstream"] != family["upstream"]
            or args["initialization"] != "scratch" or args["proxy"]["seed"] != seed
            or args["family_settings"] != family
            or args["proxy"]["epochs"] != protocol["challenging"]["search"]["epochs"]
            or metadata["checkpoint_policy"] != family["checkpoint_policy"]
            or metadata["genome_version"] != family["genome_version"]):
        raise ValueError(f"SpikeYOLO baseline provenance mismatch: {root}")
    with results_path.open() as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 1:
        raise ValueError(f"invalid result rows: {root}")
    values = list(map(float, rows[0][:3]))
    if (not all(math.isfinite(value) for value in values)
            or not 0 <= values[1] <= values[0] <= 1
            or values[2] <= 0 or not values[2].is_integer()):
        raise ValueError(f"failed SpikeYOLO baseline: {root}")
    official_path = root / "training_args/seed/official_args.yaml"
    official = yaml.safe_load(official_path.read_text())
    expected = dict(family["training"], pretrained=False, seed=seed,
                    epochs=protocol["challenging"]["search"]["epochs"],
                    imgsz=protocol["proxy"]["img_width"], batch=protocol["proxy"]["batch_size"])
    if any(official.get(key) != value for key, value in expected.items()):
        raise ValueError("SpikeYOLO official trainer arguments changed")
    from .spikeyolo_search_space import model_config
    built = json.loads((root / "training_args/seed/model.json").read_text())
    if built != model_config(SEED):
        raise ValueError("SpikeYOLO baseline architecture differs from the official stock seed")
    runtime_path = root / "training_args/seed/runtime.json"
    runtime = json.loads(runtime_path.read_text())
    if (runtime["actual_batch_shape"] != [protocol["proxy"]["img_width"]] * 2
            or runtime["actual_train_frames"] != 10053):
        raise ValueError("SpikeYOLO actual batch resolution or training coverage differs from protocol")
    run = json.loads((root / "training_args/seed/run.json").read_text())
    if not math.isfinite(run["wall_time_seconds"]) or run["wall_time_seconds"] <= 0:
        raise ValueError("missing SpikeYOLO baseline duration")
    if metadata["checkpoint_identity"] != digest(root / "trained_models/seed.pt"):
        raise ValueError("SpikeYOLO baseline weights changed")
    metrics = report["metrics"]
    if (not all(math.isfinite(metrics[key]) for key in ("mAP50", "mAP50_95"))
            or not 0 <= metrics["mAP50_95"] <= metrics["mAP50"] <= 1):
        raise ValueError("invalid SpikeYOLO report metrics")
    # result_contract.write_results serializes mAP to six decimal places;
    # the JSON reporter deliberately retains the original precision.
    serialized = [float(f"{metrics[key]:.6f}") for key in ("mAP50", "mAP50_95")]
    if serialized != values[:2] or metrics["params"] != int(values[2]):
        raise ValueError("SpikeYOLO report/CSV metrics disagree after six-decimal serialization")
    if report["num_frames"] != 6733:
        raise ValueError("SpikeYOLO validation coverage differs from 6733 frames")
    return dict(seed=seed, mAP50=values[0], mAP50_95=values[1], params=int(values[2]),
                wall_time_seconds=run["wall_time_seconds"], report_sha256=digest(report_path),
                args_sha256=digest(args_path), official_args_sha256=digest(official_path),
                model_sha256=digest(root / "training_args/seed/model.json"),
                runtime_sha256=digest(runtime_path))


def verify_export():
    """Require current checklist counts and complete per-sequence export markers."""
    from .candidate_runner import challenging_search, load_protocol
    from .export_split import dataset_parts
    protocol = load_protocol(PROTOCOL)
    if protocol["challenging"]["events"]["source"] != "released":
        raise ValueError("SpikeYOLO baseline requires released Event frames")
    settings = protocol["challenging"]
    parts = dataset_parts(settings, FRED / settings["data_root"], "search")
    for export, sequences, step in parts:
        for sequence, folder in sequences:
            marker = json.loads((export / "done" / f"{sequence}.json").read_text())
            if (str(marker["sequence"]) != str(sequence) or marker["hf_folder"] != folder
                    or marker["frame_step"] != step
                    or marker["events"] != settings["events"]
                    or marker["frames"] != len(marker["records"])):
                raise ValueError(f"export marker differs from frozen FRED protocol: {sequence}")
    if (len(parts[0][1]), len(parts[1][1])) != (152, 20):
        raise ValueError("checklist search sequence counts changed")
    _, train, load_val = challenging_search(protocol, "spikeyolo")
    val = load_val()
    if (train["frames"], len(val["images"])) != (10053, 6733):
        raise ValueError("checklist search frame counts changed; reverify the protocol")
    # Read markers for training too, without decoding images.
    from .proxy_data import load_exported_split
    exported = load_exported_split(parts[0][0], parts[0][1])
    if len(exported["images"]) != train["frames"]:
        raise ValueError("training manifest and completed export markers disagree")


def run_seed(seed, output, device):
    verify_export()
    before = identity()
    root = output / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "baseline_start.json").write_text(json.dumps(dict(
        schema_version="fred-spikeyolo-baseline-start-v1", seed=seed,
        started_at=datetime.now(timezone.utc).isoformat(), identity=before), indent=2) + "\n")
    subprocess.run(
        [sys.executable, "-m", "sota.FRED_LLM_GE.phase1_detection.candidate_runner",
         "--family", "spikeyolo", "--model", "network", "--seed", str(seed),
         "--device", device, "--protocol", str(PROTOCOL), "--sota_root", str(root)],
        cwd=FRED.parents[1], check=True)
    result = validate(root, seed)
    if identity() != before:
        raise ValueError("SpikeYOLO protocol/code/data changed during baseline")
    (root / "baseline.json").write_text(json.dumps(dict(result, identity=before), indent=2) + "\n")
    print(result)


def same_execution_identity(left, right):
    """Recovery may change the completion validator or commit, never execution inputs.

    Both full identities are retained for review. The per-file hashes still bind
    model, trainer, scorer, protocol, data manifests, dependencies, and prompts.
    """
    def execution(record):
        result = {key: value for key, value in record.items() if key not in ("sha256", "git_commit")}
        result["files"] = {key: value for key, value in record["files"].items()
                           if key != "spikeyolo_baseline.py"}
        return result
    return execution(left) == execution(right)


def verify_observation(root, name, seed, current):
    if name not in ("baseline_start.json", "baseline_recovery_observation.json"):
        raise ValueError("unknown baseline provenance observation")
    observed = json.loads((root / name).read_text())
    expected_schema = ("fred-spikeyolo-baseline-start-v1" if name == "baseline_start.json"
                       else "fred-spikeyolo-post-training-observation-v1")
    if (observed["schema_version"] != expected_schema or observed["seed"] != seed
            or not same_execution_identity(observed["identity"], current)):
        raise ValueError("SpikeYOLO execution inputs changed since provenance observation")
    report = json.loads((root / "results/seed_evaluation.json").read_text())
    if report["metadata"]["code_revision"] != observed["identity"]["git_commit"]:
        raise ValueError("SpikeYOLO training revision disagrees with provenance observation")
    if name == "baseline_recovery_observation.json":
        # Legacy runs lack a pre-training fingerprint. Require a separately
        # recorded, explicit post-training attestation and frozen artifact hashes.
        if not observed["basis"] or observed["artifacts"] != {p: digest(root / p) for p in ARTIFACTS}:
            raise ValueError("SpikeYOLO post-training observation or artifacts changed")


def recover_seed(seed, output, observation):
    """Finalize a saved run without training; refuse missing provenance or overwrites."""
    root = output / f"seed_{seed}"
    if (root / "baseline.json").exists():
        raise FileExistsError(root / "baseline.json")
    verify_export()
    current = identity()
    result = validate(root, seed)
    verify_observation(root, observation, seed, current)
    recovery = dict(observation_file=observation, observation_sha256=digest(root / observation),
                    recovered_at=datetime.now(timezone.utc).isoformat())
    if identity() != current:
        raise ValueError("SpikeYOLO protocol/code/data changed during recovery")
    # Exclusive creation preserves any concurrent completion record.
    with (root / "baseline.json").open("x") as stream:
        stream.write(json.dumps(dict(result, identity=current, recovery=recovery), indent=2) + "\n")
    print(result)


def summarize(output):
    verify_export()
    current = identity()
    runs = []
    for seed in (0, 1, 2):
        root = output / f"seed_{seed}"
        recorded = json.loads((root / "baseline.json").read_text())
        actual = validate(root, seed)
        recovery = recorded.get("recovery")
        if recovery is not None:
            verify_observation(root, recovery["observation_file"], seed, current)
            if digest(root / recovery["observation_file"]) != recovery["observation_sha256"]:
                raise ValueError("SpikeYOLO recovery observation changed")
            if not same_execution_identity(recorded["identity"], current):
                raise ValueError("SpikeYOLO recovered execution inputs changed")
            expected = dict(actual, identity=recorded["identity"], recovery=recovery)
        else:
            expected = dict(actual, identity=current)
        if recorded != expected:
            raise ValueError(f"stale, incomplete, or changed SpikeYOLO baseline: {root}")
        if recovery is not None:
            actual = dict(actual, identity=recorded["identity"], recovery=recovery)
        runs.append(actual)
    if len({run["params"] for run in runs}) != 1:
        raise ValueError("SpikeYOLO seed parameter counts disagree")
    aggregate = {key: dict(mean=statistics.mean(run[key] for run in runs),
                           sample_sd=statistics.stdev(run[key] for run in runs))
                 for key in ("mAP50", "mAP50_95", "wall_time_seconds")}
    summary = dict(
        schema_version="fred-stock-detector-baseline-v1", family="spikeyolo",
        initialization="scratch", identity=current, seeds=[0, 1, 2], params=runs[0]["params"],
        input_representation="released_event_frame_MS_GetT_T1",
        aggregate=aggregate, runs=runs)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    print(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-seed")
    run.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    run.add_argument("--device", default="cuda")
    recover = commands.add_parser("recover-seed", help="validate saved artifacts without training")
    recover.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    recover.add_argument("--observation", default="baseline_start.json",
                         choices=("baseline_start.json", "baseline_recovery_observation.json"))
    commands.add_parser("summarize")
    commands.add_parser("verify-data")
    args = parser.parse_args()
    if args.command == "run-seed":
        run_seed(args.seed, args.output.resolve(), args.device)
    elif args.command == "recover-seed":
        recover_seed(args.seed, args.output.resolve(), args.observation)
    elif args.command == "summarize":
        summarize(args.output.resolve())
    else:
        verify_export()
        print("Verified 152/10053 train and 20/6733 validation sequences/frames")


if __name__ == "__main__":
    main()
