"""Plan or submit an isolated official-split Faster R-CNN workflow."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sota.FRED_LLM_GE.phase1_detection.protocol_fingerprint import scratch_code_identity
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn_scratch.selection import freeze_manifest


def workflow(search_complete=False):
    if search_complete:
        return [("finalize", [], []), ("report", ["finalize"], [])]
    return [
        ("smoke", [], []),
        ("export", [], ["--array=0-7"]),
        ("export_finalize", ["export"], []),
        ("baseline", ["smoke", "export_finalize"], ["--array=0-2"]),
        ("server", ["baseline"], []),
        ("evolve", ["baseline", "server"], []),
        ("finalize", ["evolve"], []),
        ("report", ["finalize"], []),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=os.getenv("FRED_RUN_TAG"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Submit jobs; default prints the workflow")
    parser.add_argument("--port", type=int, default=int(os.getenv("LLMGE_PORT", "8295")))
    args = parser.parse_args()
    if not args.run_tag or not re.fullmatch(r"faster_rcnn_scratch_[0-9]{8}_eot0_[a-z0-9][a-z0-9_-]*", args.run_tag):
        parser.error("Use a fresh faster_rcnn_scratch_YYYYMMDD_eot0_<name> run tag")
    root = ROOT / "fred_runs" / args.run_tag
    status_path = root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    search_complete = (status.get("status") == "complete"
                       and status.get("last_completed_generation") == 20
                       and (root / "checkpoints/global_gen_20.pkl").is_file()
                       and (root / "checkpoints/checkpoint_gen_20.pkl").is_file())
    if not args.submit:
        print(json.dumps({"run_root": str(root), "port": args.port,
                          "stages": workflow(search_complete), "submitting": False}, indent=2))
        return
    os.chdir(ROOT)
    record = root / "launch.json"
    if root.exists() and not args.resume:
        parser.error("Run directory already exists; use --resume or a fresh tag")
    root.mkdir(parents=True, exist_ok=True)
    freeze_manifest(root / "code_snapshot.json", scratch_code_identity(ROOT))
    if record.exists():
        old = json.loads(record.read_text())
        ids = [v for v in old.values() if isinstance(v, str) and v.isdigit()]
        current_jobs = subprocess.check_output(
            ["squeue", "--me", "--noheader", "--format=%i"], text=True).split()
        active = sorted(set(ids) & {job.split("_")[0] for job in current_jobs})
        if active:
            raise SystemExit(f"Run still has active jobs: {active}")
        record.rename(root / f"launch-{time.time_ns()}.json")
    for name in ("evaluation", "islands", "evolution", "server", "export"):
        (ROOT / "run_job_outputs" / name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LLMGE_DETECTOR="faster_rcnn_scratch",
               FRED_SEED_FAMILY="faster_rcnn_scratch", FRED_RUN_TAG=args.run_tag,
               FRED_NUM_GENERATIONS="21", LLMGE_PORT=str(args.port), SUBMIT_ISLAND_CONTROLLER="0")
    if env.get("FRED_EVENTS"):
        raise ValueError("This experiment requires released event frames; unset FRED_EVENTS")
    submitted = {}
    for name, dependencies, extra in workflow(search_complete):
        command = ["sbatch", "--parsable", "--export=ALL"] + extra
        if dependencies:
            # The controller waits for HTTP readiness after the server starts.
            terms = [("after:" if dep == "server" else "afterok:") + submitted[dep]
                     for dep in dependencies]
            command += ["--dependency=" + ",".join(terms), "--kill-on-invalid-dep=yes"]
        if name == "evolve":
            env["LLMGE_SERVER_JOB_ID"] = submitted["server"]
        command.append(str(ROOT / "scripts/faster_rcnn_scratch" / f"{name}.sbatch"))
        job = subprocess.check_output(command, env=env, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected SLURM job ID: {job}")
        submitted[name] = job
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(submitted, indent=2) + "\n")
        temporary.replace(record)
        print(f"{name}: {job}", flush=True)


if __name__ == "__main__":
    main()
