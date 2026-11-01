"""Train the fixed seed model on each event setting and compare (no LLM-GE).

    # 1. export every setting once (CPU jobs; see export_split.py)
    uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --sweep --shard 0 --num_shards 4
    uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --sweep --finalize
    # 2. one protocol per (setting, training seed), plus jobs.txt and a Slurm array script
    uv run python sota/FRED_LLM_GE/phase1_detection/event_sweep.py setup --seeds 0 1 2
    # 3. once the jobs finish
    uv run python sota/FRED_LLM_GE/phase1_detection/event_sweep.py summary

Settings are the released frames (the control) plus challenging.event_sweep in
protocol.yaml. Each run is candidate_runner.py with --model network, a copy of
the protocol with that setting as challenging.events, and its own --sota_root
under <data_root>/event_sweep/<setting>/seed<k>/, so nothing else is touched.
The model and image size are the same for every setting, so inference time is
too; settings differ in mAP and in the window_us of events each frame waits for.
"""

import argparse
import copy
import statistics
from pathlib import Path

import yaml

try:
    from .export_split import DEFAULT_PROTOCOL, FRED_DIR, events_suffix
except ImportError:
    from export_split import DEFAULT_PROTOCOL, FRED_DIR, events_suffix

REPO_ROOT = FRED_DIR.parents[1]
CONTROL = {"source": "released", "window_us": 33333, "encoding": "palette"}
# Same resources as an LLM-GE evaluation job (slurm-config/slurm_config.yaml).
SBATCH = """#!/bin/bash
#SBATCH --job-name=event_sweep
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
# GPUs this PyTorch build supports (not Blackwell, not AMD).
#SBATCH -C "H100|H200|A100-80GB|A100-40GB|L40S|A40"
#SBATCH --mem 32G
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --output={out}/slurm-%A_%a.out
# Submit from the repository root: sbatch --array=0-{last} {script}
module load cuda
module load uv
unset VIRTUAL_ENV
export UV_CACHE_DIR="${{TMPDIR:-/tmp}}/uv-cache-${{SLURM_JOB_ID}}"
cd {repo}
hostname
eval "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {jobs})"
"""


def tag(events):
    return events_suffix(events).lstrip("_") or "released"


def settings(protocol, only=None):
    """The released control plus challenging.event_sweep, or just the tags in only."""
    found = [CONTROL] + protocol["challenging"]["event_sweep"]
    if only:
        unknown = set(only) - {tag(events) for events in found}
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)}; choose from {[tag(e) for e in found]}")
        found = [events for events in found if tag(events) in only]
    return found


def setup(protocol, sweep_dir, seeds, family, only=None, no_hsv=False):
    commands = []
    for events in settings(protocol, only):
        for seed in seeds:
            run_dir = sweep_dir / tag(events) / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_protocol = copy.deepcopy(protocol)
            run_protocol["challenging"]["events"] = events
            run_protocol["proxy"]["seed"] = seed
            if no_hsv:
                # Hue/saturation/value jitter recolours pixels, and colour is
                # what tells event polarities (and the background) apart.
                run_protocol["families"][family]["training"]["augmentation"].update(
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
            (run_dir / "protocol.yaml").write_text(yaml.safe_dump(run_protocol, sort_keys=False))
            commands.append(
                f"uv run python {FRED_DIR / 'phase1_detection' / 'candidate_runner.py'} "
                f"--family {family} --model network --protocol {run_dir / 'protocol.yaml'} "
                f"--sota_root {run_dir}")
    jobs = sweep_dir / "jobs.txt"
    jobs.write_text("\n".join(commands) + "\n")
    script = sweep_dir / "train_sweep.sbatch"
    out = REPO_ROOT / "run_job_outputs" / "event_sweep"
    out.mkdir(parents=True, exist_ok=True)
    script.write_text(SBATCH.format(repo=REPO_ROOT, out=out, last=len(commands) - 1, script=script, jobs=jobs))
    print(f"{len(commands)} runs ({len(settings(protocol, only))} settings x {len(seeds)} seeds)")
    print(f"from {REPO_ROOT}:\n  sbatch --array=0-{len(commands) - 1}%4 {script}")


def read_result(path):
    """(mAP50, mAP50:95, params) from a results CSV, or None if it is missing or failed.

    Failures were once written as 0,0,999999999 and are now -inf,-inf,inf
    (result_contract.py, which also appends a reason column).
    """
    if not Path(path).is_file():
        return None
    map50, map50_95, params = (float(v) for v in Path(path).read_text().split(",")[:3])
    if params == 999999999 or params == float("inf"):
        return None
    return map50, map50_95, int(params)


def mean_sd(values):
    """(mean, sample sd) of a list of scores; (None, None) if empty."""
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summary(protocol, sweep_dir, only=None):
    lines = ["setting,window_ms,encoding,runs,mAP50_mean,mAP50_sd,mAP50_95_mean,mAP50_95_sd"]
    print(f"{'setting':<24}{'window ms':>10}{'runs':>6}{'mAP50':>18}{'mAP50:95':>18}")
    for events in settings(protocol, only):
        scores = []
        for csv in sorted((sweep_dir / tag(events)).glob("seed*/results/seed_results.csv")):
            if score := read_result(csv):
                scores.append(score[:2])
        stats = [*mean_sd([s[0] for s in scores]), *mean_sd([s[1] for s in scores])]
        shown = [f"{stats[i]:.4f} ± {stats[i + 1]:.4f}" if scores else "-" for i in (0, 2)]
        window_ms = events["window_us"] / 1000
        print(f"{tag(events):<24}{window_ms:>10.1f}{len(scores):>6}{shown[0]:>18}{shown[1]:>18}")
        lines.append(",".join([tag(events), f"{window_ms}", events["encoding"], str(len(scores)),
                               *("" if v is None else f"{v:.6f}" for v in stats)]))
    (sweep_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    print(f"-> {sweep_dir / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Compare event settings with the fixed seed model")
    parser.add_argument("command", choices=["setup", "summary"])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--family", default="yolo11")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Setting tags to run, e.g. released raw_w10000_polarity (default: all)")
    parser.add_argument("--no_hsv", action="store_true",
                        help="Train without hue/saturation/value augmentation (setup only)")
    parser.add_argument("--sweep_dir", default="event_sweep",
                        help="Folder under the data root for this sweep's runs and summary")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    sweep_dir = FRED_DIR / protocol["challenging"]["data_root"] / args.sweep_dir
    if args.command == "setup":
        setup(protocol, sweep_dir, args.seeds, args.family, args.only, args.no_hsv)
    else:
        summary(protocol, sweep_dir, args.only)


if __name__ == "__main__":
    main()
