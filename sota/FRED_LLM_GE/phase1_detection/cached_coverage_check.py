"""CPU preparation then one bounded GPU run on eleven cached recordings."""
import argparse
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

from . import submit_scene_overnight as scheduler
from .scene_resolution_check import ROOT, sha256, write_json, assert_unchanged, reference_inputs, training_options
from sota.FRED_LLM_GE.data.scene_coverage_export import COHORT, PURPOSE, verify_export
from sota.FRED_LLM_GE.phase0_data.prepare_cached_coverage import (
    CONFIG, INVENTORY, OFFICIAL, DEFAULT_PROJECT, cached_readiness, prepare,
)

JOBS = Path("sota/FRED_LLM_GE/phase1_detection/jobs")


def submit(directory, reference_run):
    if scheduler.run(["git", "branch", "--show-current"]) != "fred-yolo11-infrastructure":
        raise ValueError("wrong branch")
    scheduler.run(["git", "diff", "--exit-code"])
    scheduler.run(["git", "diff", "--cached", "--exit-code"])
    if Path(scheduler.run(["git", "rev-parse", "--git-path", "MERGE_HEAD"])).exists():
        raise ValueError("merge in progress")
    inputs = reference_inputs(directory.resolve(), reference_run)
    readiness = cached_readiness()
    for path in (CONFIG, INVENTORY, OFFICIAL, DEFAULT_PROJECT):
        inputs["input_sha256"][str(path.resolve())] = sha256(path)

    def budget():
        jobs = scheduler.queue_snapshot()
        if any("JobName=fred-coverage-" in record for _, record in jobs):
            raise ValueError("a cached coverage check is already queued/running")
        return scheduler.check_budget(jobs)

    budget()
    revision = scheduler.run(["git", "rev-parse", "HEAD"])
    parent = Path("data/fred_coverage_checks")
    parent.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cached640_", dir=parent)).resolve()
    common = ["--account=isye", "--qos=coe-ice"]
    cpu = ["--partition=pace-cpu", *common, str(JOBS / "prepare_cached_coverage.sbatch"), str(output), revision]
    gpu = ["--partition=coe-gpu", *common, str(JOBS / "train_cached_coverage.sbatch"), str(output), revision]
    scheduler.run(["sbatch", "--test-only", *cpu])
    scheduler.run(["sbatch", "--test-only", *gpu])
    resources = budget()
    write_json(output / "request.json", {**inputs, "purpose": PURPOSE, "cohort": COHORT,
               "cached_readiness": readiness, "code_revision": revision, "budget": resources,
               "limit_source": "Object Detection VIP (5).pdf, page 3, 2026-09-28 notes",
               "limit_source_sha256": "b460cff1e8cc3e9ae77c690780d06a0b10a7180366337751bf27e43dea591a14"})
    cpu_id = scheduler.run(["sbatch", "--parsable", *cpu]).split(";")[0]
    try:
        if not cpu_id.isdecimal():
            raise ValueError("unrecognized CPU submission response")
        dependency = f"afterok:{cpu_id}"
        if resources["wait_for_gpu_jobs"]:
            dependency += ",afterany:" + ":".join(resources["wait_for_gpu_jobs"])
        gpu_id = scheduler.run(["sbatch", "--parsable", f"--dependency={dependency}",
                                "--kill-on-invalid-dep=yes", *gpu]).split(";")[0]
        if not gpu_id.isdecimal():
            raise ValueError("unrecognized GPU submission response")
    except BaseException:
        if cpu_id.isdecimal():
            scheduler.run(["scancel", cpu_id])
        raise
    receipt = {"cpu_job": cpu_id, "gpu_job": gpu_id, "output": str(output),
               "code_revision": revision, "budget": resources, "dependency": dependency}
    write_json(output / "jobs.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print(f"CPU log: data/logs/fred-coverage-data-{cpu_id}.out")
    print(f"GPU log: data/logs/fred-coverage-gpu-{gpu_id}.out")
    print("CACHED COVERAGE SUBMITTED: no downloads; up to 2h CPU then 2h on one GPU", flush=True)


def request_record(output):
    request = json.loads((output / "request.json").read_text())
    if (scheduler.run(["git", "rev-parse", "HEAD"]) != request["code_revision"]
            or request["purpose"] != PURPOSE
            or request["cohort"] != {k: list(v) for k, v in COHORT.items()}):
        raise ValueError("coverage request or code revision changed")
    assert_unchanged(request["input_sha256"])
    return request


def prepare_job(output):
    request = request_record(output)
    if cached_readiness() != request["cached_readiness"]:
        raise ValueError("selected cache entry identities changed")
    prepare(output)
    assert_unchanged(request["input_sha256"])


def execute(output):
    from .seeds.yolo11 import train_eval as trainer
    from .fitness import evaluate_frames
    from .review_scene_checkpoint import group_images, grouped_metrics, check_reproduction, preview
    import torch
    from ultralytics import YOLO
    request = request_record(output)
    if importlib.metadata.version("ultralytics") != "8.4.165":
        raise ValueError("pinned Ultralytics changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one allocated CUDA GPU required")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal():
        raise ValueError("Slurm job identity required")
    source = verify_export(output)
    if source["selected_sequences"] != request["cached_readiness"]["selected_sequences"]:
        raise ValueError("prepared cohort differs from submission")
    data = output / "export/data.yaml"
    trainer.validate_data(data)
    paths = [p for side in ("train", "val") for kind, suffix in (("images", "*.png"), ("labels", "*.txt"))
             for p in sorted((output / "export" / kind / side).glob(suffix))]
    paths += [output / name for name in ("manifest.sqlite", "manifest_validation.json", "preparation.json",
                                        "export/source.json", "export/data.yaml", "export/label_provenance.jsonl")]
    content = {str(p): sha256(p) for p in paths}
    write_json(output / "source_bytes.json", content)
    run_id = f"cached_coverage_640_{job_id}"
    options = training_options(request["training_config"], data, Path(request["starting_weights"]), run_id)
    options["imgsz"] = 640
    print("Allocated GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__, flush=True)
    print("Fresh pretrained seed; 11 cached recordings; 640px, batch 2, 50 epochs", flush=True)
    trainer.run(**options)
    report_path = ROOT / "results" / f"{run_id}_evaluation.json"
    report = json.loads(report_path.read_text())
    history = trainer.validate_training_history(ROOT / "runs" / run_id / "results.csv")
    config = report["metadata"]["training_config"]
    if (history["epochs_completed"] != 50 or history != report["metadata"]["training_history"]
            or any(config.get(k) != v for k, v in {"imgsz": 640, "batch": 2, "amp": False,
                    "optimizer": "AdamW", "lr0": 0.0002, "epochs": 50}.items())
            or report["num_frames"] != source["counts"]["validation"]
            or report["metrics"]["params"] != request["reference_metrics"]["params"]):
        raise ValueError("coverage run configuration, history or evaluation coverage differs")
    checkpoint = ROOT / "trained_models" / f"{run_id}.pt"
    if sha256(checkpoint) != report["metadata"]["checkpoint_identity"]:
        raise ValueError("new checkpoint differs from evaluation report")
    evaluation_identities = {str(p): sha256(p) for p in (checkpoint, report_path)}
    images, labels = trainer.val_images(data)
    groups = group_images(images, source["selected_sequences"])
    frames = trainer.predict_frames(YOLO(str(checkpoint)), images, labels, 640, 2, "0")
    write_json(output / "predictions.json", {"frames": frames, "evaluation_report": report})
    per_recording = grouped_metrics(frames, groups, report["metrics"]["params"], evaluate_frames)
    check_reproduction(per_recording["combined"], report["metrics"])
    scenes = {name: [p for sequence in sequences for p in groups.get(sequence, [])]
              for name, sequences in COHORT.items() if any(s in groups for s in sequences)}
    per_scene = grouped_metrics(frames, scenes, report["metrics"]["params"], evaluate_frames)
    for sequence, items in groups.items():
        preview(items, frames, output / f"sequence_{sequence}_preview.png")
    assert_unchanged(request["input_sha256"])
    assert_unchanged(content)
    assert_unchanged(evaluation_identities)
    verify_export(output)
    write_json(output / "summary.json", {"status": "completed", "purpose": PURPOSE,
               "run_id": run_id, "imgsz": 640, "cohort": COHORT, "source": source,
               "history": history, "validation": report["metrics"],
               "per_recording_validation": per_recording, "per_scene_validation": per_scene,
               "previous_five_recording_metrics": request["reference_metrics"],
               "comparison_scope": "cohort and training data changed; aggregate scores are not paired resolution evidence",
               "code_revision": request["code_revision"],
               "starting_weights_sha256": request["input_sha256"][str(Path(request["starting_weights"]).resolve())],
               "checkpoint_sha256": sha256(checkpoint), "evaluation_report": str(report_path),
               "best_checkpoint": str(checkpoint)})
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2), flush=True)
    print("CACHED COVERAGE CHECK COMPLETE; engineering diagnostic only", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--reference-run")
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare_job(args.prepare)
    elif args.execute:
        execute(args.execute)
    elif args.directory and args.reference_run:
        submit(args.directory, args.reference_run)
    else:
        parser.error("provide --directory and --reference-run, --prepare, or --execute")


if __name__ == "__main__":
    main()
