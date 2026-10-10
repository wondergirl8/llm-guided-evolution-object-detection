"""Re-evaluate a completed engineering checkpoint by scene, without training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import tempfile
from pathlib import Path

from . import submit_scene_overnight as scheduler

ROOT = Path("sota/FRED_LLM_GE/phase1_detection/seeds/yolo11")
JOB = Path("sota/FRED_LLM_GE/phase1_detection/jobs/review_scene_checkpoint.sbatch")
GPU_SECONDS = 15 * 60
PREVIEW_CONF = 0.5  # Display only; AP retains the original 0.001 prediction threshold.


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, record):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


def group_images(images, selected):
    expected = {s for s, side in selected.items() if side == "validation"}
    groups = {s: [] for s in sorted(expected, key=int)}
    seen = set()
    for image in images:
        match = re.fullmatch(r"(\d+)_(\d+)", image.stem)
        if match is None or match[1] not in groups or image.stem in seen:
            raise ValueError("unexpected or duplicate validation frame")
        seen.add(image.stem)
        groups[match[1]].append(image)
    if not groups or any(len(items) != 512 for items in groups.values()):
        raise ValueError("expected 512 validation frames per scene representative")
    return groups


def grouped_metrics(frames, groups, params, evaluator):
    by_id = {f["sample_id"]: f for f in frames}
    expected = {p.stem for images in groups.values() for p in images}
    if len(by_id) != len(frames) or set(by_id) != expected:
        raise ValueError("missing, extra or duplicate predictions; no-detection frames are required")
    reports = {}
    # Preserve original frame order for AP tie handling when reproducing combined AP.
    partitions = {"combined": [f["sample_id"] for f in frames],
                  **{s: [p.stem for p in images] for s, images in groups.items()}}
    for name, ids in partitions.items():
        result = evaluator([by_id[s] for s in ids], params=params, expected_sample_ids=ids)
        reports[name] = {"num_frames": len(ids), "mAP50": result.map50,
                         "mAP50_95": result.map50_95, "params": result.params}
    return reports


def check_reproduction(actual, original):
    for key in ("mAP50", "mAP50_95"):
        if not math.isclose(actual[key], original[key], rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"combined {key} differs from original report: {actual[key]} vs {original[key]}")
    if actual["params"] != original["params"]:
        raise ValueError("parameter count differs from original report")


def preview(images, frames, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from sota.FRED_LLM_GE.data.scene_diagnostic_export import evenly_spaced
    by_id = {f["sample_id"]: f for f in frames}
    chosen = evenly_spaced(images, 6)
    fig, axes = plt.subplots(6, 2, figsize=(14, 22))
    for row, image in enumerate(chosen):
        frame = by_id[image.stem]
        with Image.open(image) as source:
            pixels = source.copy()
        for column, field in enumerate(("target", "prediction")):
            ax = axes[row, column]
            ax.imshow(pixels)
            record = frame[field]
            scores = record.get("scores", [None] * len(record["boxes"]))
            for box, score in zip(record["boxes"], scores):
                if score is not None and score < PREVIEW_CONF:
                    continue
                x1, y1, x2, y2 = box
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                       edgecolor="lime", linewidth=1.5))
                if score is not None:
                    ax.text(x1, y1, f"{score:.2f}", color="lime", fontsize=8)
            ax.set_title(f"{image.stem}: {field}")
            ax.axis("off")
    fig.suptitle(f"Labels and predictions; display confidence >= {PREVIEW_CONF}; AP uses conf=0.001")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def execute(output):
    from .seeds.yolo11 import train_eval as trainer
    from .fitness import evaluate_frames
    from sota.FRED_LLM_GE.data.scene_diagnostic_export import verify_export
    import torch
    from ultralytics import YOLO
    request = json.loads((output / "request.json").read_text())
    directory, run = Path(request["directory"]), request["run_id"]
    report_path = ROOT / "results" / f"{run}_evaluation.json"
    checkpoint = ROOT / "trained_models" / f"{run}.pt"
    report = json.loads(report_path.read_text())
    if (sha256(report_path) != request["original_report_sha256"]
            or sha256(checkpoint) != report["metadata"]["checkpoint_identity"]):
        raise ValueError("original evaluation report or checkpoint changed")
    if scheduler.run(["git", "rev-parse", "HEAD"]) != request["code_revision"]:
        raise ValueError("review code revision changed")
    from .checkpoint_code import verify_code_files
    code_changes = verify_code_files(ROOT, report["metadata"])
    if importlib.metadata.version("ultralytics") != "8.4.165":
        raise ValueError("pinned Ultralytics environment changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one allocated CUDA GPU required")
    source = verify_export(directory)
    data = directory / "export/data.yaml"
    if (sha256(data.parent / "source.json") != report["metadata"]["source_record_sha256"]
            or trainer.validate_training_history(ROOT / "runs" / run / "results.csv") !=
            report["metadata"]["training_history"]):
        raise ValueError("original data provenance or training history changed")
    trainer.validate_data(data)
    images, labels = trainer.val_images(data)
    groups = group_images(images, source["selected_sequences"])
    content = {str(p): sha256(p) for image in images for p in (image, labels / f"{image.stem}.txt")}
    model = YOLO(str(checkpoint))
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = report["metadata"]["training_config"]
    frames = trainer.predict_frames(model, images, labels, config["imgsz"], config["batch"], "0")
    write_json(output / "predictions.json", {"frames": frames, "source_file_sha256": content,
                                            "original_report": report})
    metrics = grouped_metrics(frames, groups, report["metrics"]["params"], evaluate_frames)
    write_json(output / "metrics.json", metrics)  # Preserve evidence even if reproduction fails.
    check_reproduction(metrics["combined"], report["metrics"])
    if any(sha256(path) != digest for path, digest in content.items()):
        raise ValueError("validation inputs changed during review")
    selection = json.loads((directory / "selection.json").read_text())
    scene_names = {sequence: name for name, sequence in selection.items()}
    for sequence, items in groups.items():
        preview(items, frames, output / f"sequence_{sequence}_preview.png")
        print(scene_names[sequence], sequence, json.dumps(metrics[sequence]), flush=True)
    write_json(output / "summary.json", {"status": "completed", "purpose": source["purpose"],
               "original_run": run, "original_report_sha256": sha256(report_path),
               "checkpoint_sha256": sha256(checkpoint), "metrics": metrics,
               "scene_names": scene_names, "code_revision": request["code_revision"],
               "preview_confidence": PREVIEW_CONF, "combined_matches_original_abs_tolerance": 1e-5,
               "code_changes": code_changes})
    print("SCENE CHECKPOINT REVIEW COMPLETE; no training or evolution", flush=True)


def submit(directory, run):
    if scheduler.run(["git", "branch", "--show-current"]) != "fred-yolo11-infrastructure":
        raise ValueError("wrong branch")
    scheduler.run(["git", "diff", "--exit-code"])
    scheduler.run(["git", "diff", "--cached", "--exit-code"])
    if Path(scheduler.run(["git", "rev-parse", "--git-path", "MERGE_HEAD"])).exists():
        raise ValueError("merge in progress")
    if re.fullmatch(r"scene_check_\d+", run) is None:
        raise ValueError("expected a completed scene_check_JOBID run")
    directory = directory.resolve()
    report_path = ROOT / "results" / f"{run}_evaluation.json"
    original = json.loads(report_path.read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "completed" or summary["run_id"] != run or summary["validation"] != original["metrics"]:
        raise ValueError("directory is not the completed original run")
    if sha256(ROOT / "trained_models" / f"{run}.pt") != original["metadata"]["checkpoint_identity"]:
        raise ValueError("checkpoint changed")

    def budget():
        jobs = scheduler.queue_snapshot()
        if any("JobName=fred-scene-review" in record for _, record in jobs):
            raise ValueError("a scene review is already queued/running")
        return scheduler.check_budget(jobs, new_gpu_seconds=GPU_SECONDS, new_jobs=1)

    budget()
    revision = scheduler.run(["git", "rev-parse", "HEAD"])
    directory_out = Path("data/fred_checkpoint_reviews")
    directory_out.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{run}_", dir=directory_out)).resolve()
    args = ["--partition=coe-gpu", "--account=isye", "--qos=coe-ice", str(JOB), str(output), revision]
    scheduler.run(["sbatch", "--test-only", *args])
    resources = budget()
    request = {"directory": str(directory), "run_id": run, "code_revision": revision,
               "original_report_sha256": sha256(report_path), "budget": resources}
    write_json(output / "request.json", request)
    dependencies = resources["wait_for_gpu_jobs"]
    flags = [f"--dependency=afterany:{':'.join(dependencies)}"] if dependencies else []
    job_id = scheduler.run(["sbatch", "--parsable", *flags, *args]).split(";")[0]
    if not job_id.isdecimal():
        raise ValueError("unrecognized GPU submission response")
    write_json(output / "job.json", {"job_id": job_id, **request})
    print(json.dumps({"job_id": job_id, "output": str(output), "budget": resources}, indent=2))
    print(f"Log: data/logs/fred-scene-review-{job_id}.out")
    print("SCENE REVIEW SUBMITTED: one GPU, maximum 15 minutes; no training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    elif args.directory and args.run_id:
        submit(args.directory, args.run_id)
    else:
        parser.error("provide --directory and --run-id, or --execute")


if __name__ == "__main__":
    main()
