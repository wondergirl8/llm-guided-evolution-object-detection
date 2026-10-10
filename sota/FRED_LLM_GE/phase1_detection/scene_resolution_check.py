"""Bounded 640 -> 960 engineering comparison on the existing five-scene export."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path

from . import submit_scene_overnight as scheduler

ROOT = Path("sota/FRED_LLM_GE/phase1_detection/seeds/yolo11")
JOB = Path("sota/FRED_LLM_GE/phase1_detection/jobs/check_scene_resolution.sbatch")
GPU_SECONDS = 2 * 3600
IMGSZ = 960
PURPOSE = "960-pixel resolution engineering diagnostic; not a formal baseline or evolution fitness"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, record):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


def training_options(reference, data, weights, run_id):
    expected = {"epochs": 50, "fraction": 1.0, "batch": 2, "imgsz": 640,
                "seed": 0, "deterministic": True, "gene_id": "seed", "amp": False,
                "optimizer": "AdamW", "lr0": 0.0002, "momentum": 0.9, "warmup_bias_lr": 0.0}
    if any(reference.get(key) != value for key, value in expected.items()):
        raise ValueError("original run differs from the validated 640-pixel engineering settings")
    return {"model_name": "network", "variant_dir": ROOT / "models", "data": data,
            "weights": str(weights), "epochs": 50, "batch": 2, "imgsz": IMGSZ,
            "device": "0", "run_id": run_id, "fraction": 1.0, "amp": False,
            "adamw_lr": 0.0002}


def reference_inputs(directory, reference_run):
    from .seeds.yolo11 import train_eval as trainer
    from sota.FRED_LLM_GE.data.scene_diagnostic_export import verify_export
    if re.fullmatch(r"scene_check_\d+", reference_run) is None:
        raise ValueError("expected a completed scene_check_JOBID reference")
    source = verify_export(directory)
    report_path = ROOT / "results" / f"{reference_run}_evaluation.json"
    report = json.loads(report_path.read_text())
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    metadata = report["metadata"]
    weights = Path(metadata["training_config"]["weights"]).resolve()
    if (summary["status"] != "completed" or summary["run_id"] != reference_run
            or summary["validation"] != report["metrics"]
            or report["num_frames"] != source["counts"]["validation"]):
        raise ValueError("reference data/run summary is inconsistent")
    training_options(metadata["training_config"], directory / "export/data.yaml", weights, "unused")
    if (sha256(directory / "export/source.json") != metadata["source_record_sha256"]
            or sha256(directory / "manifest.sqlite") != metadata["manifest_identity"]
            or sha256(ROOT / "trained_models" / f"{reference_run}.pt") != metadata["checkpoint_identity"]
            or trainer.validate_training_history(ROOT / "runs" / reference_run / "results.csv") !=
            metadata["training_history"]):
        raise ValueError("original provenance, checkpoint or history changed")
    for name, digest in metadata["code_files_sha256"].items():
        path = (ROOT / name if name in ("train_eval.py", "network.py", "validator.py") else
                ROOT.parent.parent / "adapters" / name if name == "yolo11.py" else ROOT.parent.parent / name)
        if sha256(path) != digest:
            raise ValueError(f"original model/evaluator code changed: {name}")
    # Snapshot the current starting weights; the old report did not hash their bytes.
    identities = {str(p): sha256(p) for p in (
        report_path, summary_path, weights, directory / "export/source.json",
        directory / "export/data.yaml", directory / "manifest.sqlite")}
    return {"directory": str(directory), "reference_run": reference_run,
            "reference_metrics": report["metrics"], "starting_weights": str(weights),
            "input_sha256": identities, "training_config": metadata["training_config"],
            "purpose": PURPOSE}


def assert_unchanged(identities):
    if any(sha256(path) != digest for path, digest in identities.items()):
        raise ValueError("diagnostic source or starting weights changed")


def submit(directory, reference_run):
    if scheduler.run(["git", "branch", "--show-current"]) != "fred-yolo11-infrastructure":
        raise ValueError("wrong branch")
    scheduler.run(["git", "diff", "--exit-code"])
    scheduler.run(["git", "diff", "--cached", "--exit-code"])
    if Path(scheduler.run(["git", "rev-parse", "--git-path", "MERGE_HEAD"])).exists():
        raise ValueError("merge in progress")
    inputs = reference_inputs(directory.resolve(), reference_run)

    def budget():
        jobs = scheduler.queue_snapshot()
        if any("JobName=fred-resolution" in record for _, record in jobs):
            raise ValueError("a resolution check is already queued/running")
        return scheduler.check_budget(jobs, new_gpu_seconds=GPU_SECONDS, new_jobs=1)

    budget()
    revision = scheduler.run(["git", "rev-parse", "HEAD"])
    parent = Path("data/fred_resolution_checks")
    parent.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="img960_", dir=parent)).resolve()
    args = ["--partition=coe-gpu", "--account=isye", "--qos=coe-ice", str(JOB), str(output), revision]
    scheduler.run(["sbatch", "--test-only", *args])
    resources = budget()
    write_json(output / "request.json", {**inputs, "code_revision": revision,
               "imgsz": IMGSZ, "budget": resources,
               "limit_source": "Object Detection VIP (5).pdf, page 3, 2026-09-28 notes"})
    dependencies = resources["wait_for_gpu_jobs"]
    flags = ([f"--dependency=afterany:{':'.join(dependencies)}", "--kill-on-invalid-dep=yes"]
             if dependencies else [])
    job_id = scheduler.run(["sbatch", "--parsable", *flags, *args]).split(";")[0]
    if not job_id.isdecimal():
        raise ValueError("unrecognized GPU submission response")
    write_json(output / "job.json", {"job_id": job_id, "output": str(output),
                                    "code_revision": revision, "budget": resources})
    print(json.dumps({"job_id": job_id, "output": str(output), "budget": resources}, indent=2))
    print(f"Log: data/logs/fred-resolution-{job_id}.out")
    print("RESOLUTION CHECK SUBMITTED: one GPU, maximum 2h; same data, 960 pixels")


def execute(output):
    from .seeds.yolo11 import train_eval as trainer
    from .fitness import evaluate_frames
    from .review_scene_checkpoint import group_images, grouped_metrics, check_reproduction, preview
    from sota.FRED_LLM_GE.data.scene_diagnostic_export import verify_export
    import torch
    from ultralytics import YOLO
    request = json.loads((output / "request.json").read_text())
    if scheduler.run(["git", "rev-parse", "HEAD"]) != request["code_revision"]:
        raise ValueError("submitted revision changed")
    if request["imgsz"] != IMGSZ:
        raise ValueError("requested resolution changed")
    assert_unchanged(request["input_sha256"])
    directory = Path(request["directory"])
    current = reference_inputs(directory, request["reference_run"])
    if any(current[key] != request[key] for key in current):
        raise ValueError("reference inputs changed since submission")
    if importlib.metadata.version("ultralytics") != "8.4.165":
        raise ValueError("pinned Ultralytics changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one allocated CUDA GPU required")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal():
        raise ValueError("Slurm job identity required")
    run_id = f"resolution_960_{job_id}"
    data = directory / "export/data.yaml"
    trainer.validate_data(data)
    paths = [p for split in ("train", "val") for kind, suffix in (("images", "*.png"), ("labels", "*.txt"))
             for p in sorted((directory / "export" / kind / split).glob(suffix))]
    content = {str(p): sha256(p) for p in paths}
    write_json(output / "source_bytes.json", content)
    weights = Path(request["starting_weights"])
    print("Allocated GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__, flush=True)
    print(f"Starting fresh pretrained seed at {IMGSZ}px; reference checkpoint is preserved", flush=True)
    options = training_options(request["training_config"], data, weights, run_id)
    trainer.run(**options)
    report_path = ROOT / "results" / f"{run_id}_evaluation.json"
    report = json.loads(report_path.read_text())
    history = trainer.validate_training_history(ROOT / "runs" / run_id / "results.csv")
    if history["epochs_completed"] != 50 or history != report["metadata"]["training_history"]:
        raise ValueError("50 finite training epochs required")
    config = report["metadata"]["training_config"]
    if (config["imgsz"] != IMGSZ or config["amp"] is not False or config["lr0"] != 0.0002
            or report["metrics"]["params"] != request["reference_metrics"]["params"]
            or report["num_frames"] != verify_export(directory)["counts"]["validation"]):
        raise ValueError("resolution run configuration or evaluation coverage differs")
    images, labels = trainer.val_images(data)
    selected = verify_export(directory)["selected_sequences"]
    groups = group_images(images, selected)
    checkpoint = ROOT / "trained_models" / f"{run_id}.pt"
    if sha256(checkpoint) != report["metadata"]["checkpoint_identity"]:
        raise ValueError("new best checkpoint differs from its evaluation report")
    frames = trainer.predict_frames(YOLO(str(checkpoint)), images, labels, IMGSZ, 2, "0")
    write_json(output / "predictions.json", {"frames": frames, "evaluation_report": report})
    per_scene = grouped_metrics(frames, groups, report["metrics"]["params"], evaluate_frames)
    write_json(output / "scene_metrics.json", per_scene)
    check_reproduction(per_scene["combined"], report["metrics"])
    for sequence, items in groups.items():
        preview(items, frames, output / f"sequence_{sequence}_preview.png")
    assert_unchanged(request["input_sha256"])
    assert_unchanged(content)
    write_json(output / "summary.json", {"status": "completed", "purpose": PURPOSE,
               "run_id": run_id, "reference_run": request["reference_run"],
               "reference_metrics": request["reference_metrics"], "validation": report["metrics"],
               "per_scene_validation": per_scene,
               "history": history, "imgsz": IMGSZ, "code_revision": request["code_revision"],
               "starting_weights_sha256": request["input_sha256"][str(weights)],
               "evaluation_report": str(report_path),
               "best_checkpoint": str(ROOT / "trained_models" / f"{run_id}.pt")})
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2))
    print("RESOLUTION CHECK COMPLETE; engineering diagnostic only", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--reference-run")
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    elif args.directory and args.reference_run:
        submit(args.directory, args.reference_run)
    else:
        parser.error("provide --directory and --reference-run, or --execute")


if __name__ == "__main__":
    main()
