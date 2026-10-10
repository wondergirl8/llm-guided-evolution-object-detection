"""Evaluate a saved 960px checkpoint after the known post-training OOM; never train."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import tempfile
from pathlib import Path

import yaml

from . import scene_resolution_check as resolution
from . import submit_scene_overnight as scheduler
from .checkpoint_code import LEGACY_REVISION

ROOT = resolution.ROOT
JOB = Path("sota/FRED_LLM_GE/phase1_detection/jobs/recover_scene_resolution.sbatch")
GPU_SECONDS = 15 * 60
PURPOSE = "saved 960-pixel checkpoint evaluation recovery; engineering diagnostic only"


def failed_inputs(failed_output):
    from .seeds.yolo11 import train_eval as trainer
    failed_output = failed_output.resolve()
    request_path, job_path = failed_output / "request.json", failed_output / "job.json"
    request, job = (json.loads(p.read_text()) for p in (request_path, job_path))
    job_id = job["job_id"]
    if (not isinstance(job_id, str) or re.fullmatch(r"\d+", job_id) is None or
            request["code_revision"] != LEGACY_REVISION or job["code_revision"] != LEGACY_REVISION or
            Path(job["output"]).resolve() != failed_output or request["imgsz"] != 960):
        raise ValueError("expected the known failed resolution job and original receipt")
    state = scheduler.run(["sacct", "--noheader", "--parsable2", "--jobs", job_id,
                           "--format=JobIDRaw,State,ExitCode"])
    main = [line.strip().split("|") for line in state.splitlines() if line.strip().split("|")[0] == job_id]
    if len(main) != 1 or main[0][1:3] != ["FAILED", "1:0"]:
        raise ValueError("original job must be finished FAILED with exit 1:0")
    if (failed_output / "summary.json").exists():
        raise ValueError("original diagnostic already has a completion summary")
    resolution.assert_unchanged(request["input_sha256"])
    directory = Path(request["directory"])
    reference = resolution.reference_inputs(directory, request["reference_run"])
    if any(reference[k] != request[k] for k in reference if k != "reference_code_changes"):
        raise ValueError("original reference inputs changed")
    data = directory / "export/data.yaml"
    trainer.validate_data(data)
    source_bytes_path = failed_output / "source_bytes.json"
    source_bytes = json.loads(source_bytes_path.read_text())
    expected = {str(p) for split in ("train", "val") for kind, suffix in (("images", "*.png"), ("labels", "*.txt"))
                for p in (directory / "export" / kind / split).glob(suffix)}
    if len(expected) != 5120 or set(source_bytes) != expected:
        raise ValueError("original pre-training source snapshot lacks full 1536/1024 image/label coverage")
    resolution.assert_unchanged(source_bytes)
    run_id = f"resolution_960_{job_id}"
    run = ROOT / "runs" / run_id
    history = trainer.validate_training_history(run / "results.csv")
    if history["epochs_completed"] != 50:
        raise ValueError("50 complete finite training epochs required")
    config = {**request["training_config"], "data": str(data.resolve()), "imgsz": 960, "device": "0"}
    args_path = run / "args.yaml"
    args = yaml.safe_load(args_path.read_text())
    keys = ("epochs", "fraction", "batch", "imgsz", "seed", "deterministic", "amp",
            "optimizer", "lr0", "momentum", "warmup_bias_lr", "data")
    if any(args.get(k) != config[k] for k in keys):
        raise ValueError("saved training settings differ from submitted 960-pixel protocol")
    best = run / "weights/best.pt"
    checkpoint = ROOT / "trained_models" / f"{run_id}.pt"
    checkpoint_sha = resolution.sha256(checkpoint)
    if best.stat().st_size == 0 or resolution.sha256(best) != checkpoint_sha:
        raise ValueError("saved checkpoint copies are empty or differ")
    for path in (ROOT / "results" / f"{run_id}_evaluation.json",
                 ROOT / "results" / f"{run_id}_results.csv"):
        if path.exists():
            raise ValueError("original run already has evaluator outputs; inspect before recovery")
    log = Path("data/logs") / f"fred-resolution-{job_id}.out"
    text = log.read_text(errors="replace")
    if "torch.OutOfMemoryError: CUDA out of memory" not in text or "predict_frames" not in text:
        raise ValueError("original log does not document the known prediction OOM")
    snapshots = {str(p.resolve()): resolution.sha256(p) for p in (
        request_path, job_path, source_bytes_path, run / "results.csv", args_path, best, checkpoint, log)}
    snapshots.update(request["input_sha256"])
    return {"failed_output": str(failed_output), "original_job_id": job_id,
            "original_job_status": "FAILED", "training_code_revision": request["code_revision"],
            "run_id": run_id, "directory": str(directory), "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha, "history": history, "training_config": config,
            "reference_metrics": request["reference_metrics"], "input_sha256": snapshots,
            "reference_code_changes": reference["reference_code_changes"],
            "purpose": PURPOSE}


def submit(failed_output):
    if scheduler.run(["git", "branch", "--show-current"]) != "fred-yolo11-infrastructure":
        raise ValueError("wrong branch")
    scheduler.run(["git", "diff", "--exit-code"])
    scheduler.run(["git", "diff", "--cached", "--exit-code"])
    if Path(scheduler.run(["git", "rev-parse", "--git-path", "MERGE_HEAD"])).exists():
        raise ValueError("merge in progress")
    inputs = failed_inputs(failed_output)

    def budget():
        jobs = scheduler.queue_snapshot()
        if any("JobName=fred-resolution" in record for _, record in jobs):
            raise ValueError("a resolution training or recovery job is already queued/running")
        return scheduler.check_budget(jobs, new_gpu_seconds=GPU_SECONDS, new_jobs=1)

    budget()
    revision = scheduler.run(["git", "rev-parse", "HEAD"])
    parent = Path("data/fred_resolution_recoveries")
    parent.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{inputs['run_id']}_", dir=parent)).resolve()
    args = ["--partition=coe-gpu", "--account=isye", "--qos=coe-ice", str(JOB), str(output), revision]
    scheduler.run(["sbatch", "--test-only", *args])
    resources = budget()
    resolution.write_json(output / "request.json", {**inputs, "code_revision": revision, "budget": resources,
        "limit_source": "Object Detection VIP (5).pdf, page 3, 2026-09-28 notes"})
    dependencies = resources["wait_for_gpu_jobs"]
    flags = ([f"--dependency=afterany:{':'.join(dependencies)}", "--kill-on-invalid-dep=yes"]
             if dependencies else [])
    job_id = scheduler.run(["sbatch", "--parsable", *flags, *args]).split(";")[0]
    if not job_id.isdecimal():
        raise ValueError("unrecognized GPU submission response")
    resolution.write_json(output / "job.json", {"job_id": job_id, "output": str(output), "budget": resources})
    print(json.dumps({"job_id": job_id, "output": str(output), "budget": resources}, indent=2))
    print(f"Log: data/logs/fred-resolution-recovery-{job_id}.out")
    print("RESOLUTION RECOVERY SUBMITTED: one GPU, maximum 15 minutes; no training", flush=True)


def execute(output):
    from .seeds.yolo11 import train_eval as trainer
    from .fitness import evaluate_frames
    from .map_reporter import report as publish_report
    from .review_scene_checkpoint import group_images, grouped_metrics, check_reproduction, preview
    from sota.FRED_LLM_GE.data.scene_diagnostic_export import verify_export
    import torch
    from ultralytics import YOLO
    request = json.loads((output / "request.json").read_text())
    if scheduler.run(["git", "rev-parse", "HEAD"]) != request["code_revision"]:
        raise ValueError("recovery revision changed")
    current = failed_inputs(Path(request["failed_output"]))
    if any(current[k] != request[k] for k in current):
        raise ValueError("saved run changed since recovery submission")
    if importlib.metadata.version("ultralytics") != "8.4.165":
        raise ValueError("pinned Ultralytics environment changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one allocated CUDA GPU required")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    data = Path(request["directory"]) / "export/data.yaml"
    images, labels = trainer.val_images(data)
    source = verify_export(Path(request["directory"]))
    groups = group_images(images, source["selected_sequences"])
    model = YOLO(request["checkpoint"])
    params = sum(p.numel() for p in model.model.parameters())
    if params != request["reference_metrics"]["params"]:
        raise ValueError("saved model parameter count differs from protected seed")
    print("Evaluating saved checkpoint at 960 pixels, at most two images per call", flush=True)
    frames = trainer.predict_frames(model, images, labels, 960, 2, "0")
    resolution.write_json(output / "predictions.json", {"frames": frames, "recovery_request": request})
    metrics = grouped_metrics(frames, groups, params, evaluate_frames)
    resolution.write_json(output / "scene_metrics.json", metrics)
    for sequence, items in groups.items():
        preview(items, frames, output / f"sequence_{sequence}_preview.png")
    resolution.assert_unchanged(request["input_sha256"])
    content = json.loads((Path(request["failed_output"]) / "source_bytes.json").read_text())
    resolution.assert_unchanged(content)
    metadata = trainer.report_metadata(data, ROOT / "network.py", Path(request["checkpoint"]),
                                       request["training_config"])
    metadata.update(training_config=request["training_config"], training_history=request["history"],
                    training_code_revision=request["training_code_revision"], recovery=request,
                    checkpoint_identity_scope="captured after failed job; copies verified equal at recovery",
                    inference_batching="explicit source lists of at most 2 images")
    # Recovery reports live here, never overwrite failed-run artifacts or publish into the search results directory.
    _, report_path = publish_report({"metadata": metadata, "frames": frames, "params": params,
        "expected_sample_ids": [p.stem for p in images]}, sota_root=output, gene_id=request["run_id"])
    report = json.loads(report_path.read_text())
    check_reproduction(metrics["combined"], report["metrics"])
    resolution.write_json(output / "summary.json", {"status": "completed", "purpose": PURPOSE,
        "original_job_status": "FAILED", "original_job_id": request["original_job_id"],
        "run_id": request["run_id"], "history": request["history"], "imgsz": 960,
        "reference_metrics": request["reference_metrics"], "per_scene_validation": metrics,
        "validation": report["metrics"], "evaluation_report": str(report_path),
        "checkpoint_sha256": request["checkpoint_sha256"], "code_revision": request["code_revision"],
        "training_code_revision": request["training_code_revision"], "training_repeated": False})
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2))
    print("RESOLUTION RECOVERY COMPLETE; saved checkpoint evaluated; original job remains FAILED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-output", type=Path)
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    elif args.failed_output:
        submit(args.failed_output)
    else:
        parser.error("provide --failed-output or --execute")


if __name__ == "__main__":
    main()
