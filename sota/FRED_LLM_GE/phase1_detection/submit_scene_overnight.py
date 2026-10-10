"""Submit a bounded CPU -> one-GPU diagnostic under the VIP queue envelope.

Object Detection VIP (5).pdf, page 3 (2026-09-28 notes): 32 GPU-hours
across jobs; max 50 queued jobs. Count all existing jobs' full requested
GPU walltimes conservatively, including running and dependent jobs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

GPU_SECONDS = 2 * 3600
MAX_GPU_SECONDS = 32 * 3600
MAX_JOBS = 50
JOBS = Path("sota/FRED_LLM_GE/phase1_detection/jobs")


def walltime_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError(f"cannot safely budget Slurm TimeLimit={value}")
    days, hours, minutes, seconds = (int(v or 0) for v in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError("invalid Slurm time limit")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def reservation(job: str) -> tuple[int, int]:
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", job))
    tres = fields.get("ReqTRES")
    if tres is None:
        raise ValueError("Slurm job has no ReqTRES; refusing an unknown GPU reservation")
    gpu_values = {}
    for item in tres.split(","):
        key, value = item.split("=", 1)
        if key == "gres/gpu" or key.startswith("gres/gpu:"):
            if not value.isdecimal():
                raise ValueError("unknown GPU TRES count")
            gpu_values[key] = int(value)
    # Generic and model-specific GPU TRES describe the same devices.
    count = gpu_values.get("gres/gpu", sum(gpu_values.values()))
    if sum(value for key, value in gpu_values.items() if key != "gres/gpu") > count:
        raise ValueError("contradictory GPU TRES counts")
    seconds = walltime_seconds(fields["TimeLimit"]) if count else 0
    return count, count * seconds


def check_budget(jobs: list[tuple[str, str]], *, new_gpu_seconds: int = GPU_SECONDS,
                 new_jobs: int = 2) -> dict:
    if (type(new_gpu_seconds) is not int or new_gpu_seconds <= 0
            or type(new_jobs) is not int or new_jobs <= 0):
        raise ValueError("new resource reservation must be positive integers")
    reservations = {job_id: reservation(record) for job_id, record in jobs}
    if len(reservations) != len(jobs):
        raise ValueError("duplicate Slurm job identities")
    existing_seconds = sum(amount for _, amount in reservations.values())
    if len(jobs) + new_jobs > MAX_JOBS:
        raise ValueError("adding jobs would exceed the documented 50-job limit")
    if existing_seconds + new_gpu_seconds > MAX_GPU_SECONDS:
        raise ValueError("adding this job would exceed the documented 32 GPU-hour envelope")
    return {"existing_jobs": len(jobs), "existing_reserved_gpu_hours": existing_seconds / 3600,
            "new_gpu_hours": new_gpu_seconds / 3600,
            "total_reserved_gpu_hours": (existing_seconds + new_gpu_seconds) / 3600,
            "wait_for_gpu_jobs": [job_id for job_id, (count, _) in reservations.items() if count]}


def run(command: list[str]) -> str:
    # SBATCH_* variables can override file directives; this workflow fixes its envelope.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    return subprocess.check_output(command, text=True, env=env).strip()


def queue_snapshot() -> list[tuple[str, str]]:
    output = run(["squeue", "--noheader", "--user", str(os.getuid()), "--format=%i"])
    ids = list(dict.fromkeys(output.split()))
    records = []
    for job_id in ids:
        if not job_id.isdecimal():
            raise ValueError("expand or finish existing job arrays before this bounded submission")
        record = run(["scontrol", "show", "job", "-o", job_id])
        if "JobName=fred-night-" in record:
            raise ValueError("an overnight scene workflow is already queued/running")
        records.append((job_id, record))
    return records


def main() -> None:
    if run(["git", "branch", "--show-current"]) != "fred-yolo11-infrastructure":
        raise ValueError("wrong branch")
    run(["git", "diff", "--exit-code"])
    run(["git", "diff", "--cached", "--exit-code"])
    if Path(run(["git", "rev-parse", "--git-path", "MERGE_HEAD"])).exists():
        raise ValueError("merge in progress")
    revision = run(["git", "rev-parse", "HEAD"])
    for path in ("data/.venv-yolo11/bin/python", "yolo11m.pt",
                 "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/learning_rate_check_6135264_evaluation.json"):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    budget = check_budget(queue_snapshot())
    print("VIP resource envelope:", json.dumps(budget), flush=True)
    parent = Path("data/fred_overnight")
    parent.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="scene_", dir=parent))
    common = ["--account=isye", "--qos=coe-ice"]
    cpu = ["--partition=pace-cpu", *common, str(JOBS / "prepare_scene_overnight.sbatch"), str(directory), revision]
    gpu = ["--partition=coe-gpu", *common, str(JOBS / "train_scene_overnight.sbatch"), str(directory), revision]
    run(["sbatch", "--test-only", *cpu])
    run(["sbatch", "--test-only", *gpu])
    # Refresh before mutations; Slurm also applies the live account/QOS rules.
    budget = check_budget(queue_snapshot())
    with (directory / "submission.json").open("x") as handle:
        json.dump({"code_revision": revision, "budget": budget, "directory": str(directory),
                   "limit_source": "Object Detection VIP (5).pdf, page 3, 2026-09-28 notes",
                   "limit_source_sha256": "b460cff1e8cc3e9ae77c690780d06a0b10a7180366337751bf27e43dea591a14"},
                  handle, indent=2)
    cpu_id = run(["sbatch", "--parsable", *cpu]).split(";")[0]
    gpu_id = None
    try:
        if not cpu_id.isdecimal():
            raise ValueError("unrecognized CPU submission response")
        dependency = f"afterok:{cpu_id}"
        if budget["wait_for_gpu_jobs"]:
            dependency += ",afterany:" + ":".join(budget["wait_for_gpu_jobs"])
        gpu_id = run(["sbatch", "--parsable", f"--dependency={dependency}",
                      "--kill-on-invalid-dep=yes", *gpu]).split(";")[0]
        if not gpu_id.isdecimal():
            raise ValueError("unrecognized GPU submission response")
    except BaseException:
        if cpu_id.isdecimal():
            run(["scancel", cpu_id])
        raise
    receipt = {"cpu_job": cpu_id, "gpu_job": gpu_id, "directory": str(directory),
               "code_revision": revision, "budget": budget, "dependency": dependency}
    with (directory / "jobs.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)
    print(f"CPU log: data/logs/fred-night-data-{cpu_id}.out")
    print(f"GPU log: data/logs/fred-night-gpu-{gpu_id}.out")
    print("OVERNIGHT WORKFLOW SUBMITTED: up to 2h CPU then 2h on one GPU; queue waits extra")


if __name__ == "__main__":
    main()
