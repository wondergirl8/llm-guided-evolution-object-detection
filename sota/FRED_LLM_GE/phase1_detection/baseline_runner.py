"""Run the stock YOLO11n seed under the candidate protocol, once per seed.

Run ``run-seed --seed N`` inside a GPU allocation for N=0,1,2, then run
``summarize`` after all three jobs finish. No SLURM job is submitted here.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from .protocol_fingerprint import compute_fingerprint
except ImportError:
    from protocol_fingerprint import compute_fingerprint


FRED_DIR = Path(__file__).resolve().parents[1]
ROOT = FRED_DIR.parents[1]
PROTOCOL = FRED_DIR / "configs/phase1/protocol.yaml"
SEED_YAML = FRED_DIR / "seeds/seed_yolo11n.yaml"
OUTPUT = FRED_DIR / "baseline_validation/phase1_detection/yolo11_stock"
SEEDS = (0, 1, 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    """Include the frozen data/protocol identity and local evaluation code."""
    record = compute_fingerprint(ROOT, "yolo11", "stock_yolo11", SEED_YAML)
    code = ["candidate_runner.py", "adapters/ultralytics_yolo11.py",
            "adapters/yolo11.py", "yolo11_search_space.py", "map_reporter.py",
            "fitness.py", "result_contract.py", "baseline_runner.py"]
    record["baseline_code"] = {name: digest(Path(__file__).parent / name) for name in code}
    record["baseline_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def paths(seed, output=OUTPUT):
    root = Path(output) / f"seed_{seed}"
    return root, root / "results/seed_results.csv", root / "results/seed_evaluation.json", root / "training_args/seed/args.yaml"


def validate_seed(seed, output=OUTPUT):
    """Reject failure sentinels, stale stock files, or a different recipe."""
    root, csv_path, report_path, args_path = paths(seed, output)
    with csv_path.open(newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError(f"invalid baseline result: {csv_path}")
    map50, map95, params = map(float, rows[0][:3])
    if not (math.isfinite(map50) and math.isfinite(map95) and
            math.isfinite(params) and 0 <= map95 <= map50 <= 1 and
            params > 0 and params.is_integer()):
        raise ValueError(f"failed or invalid baseline result: {csv_path}: {rows[0]}")
    report = json.loads(report_path.read_text())
    meta = report["metadata"]
    if (report.get("gene_id") != "seed" or report.get("mode") != "validation" or
            meta.get("model_identity") != digest(SEED_YAML) or
            meta.get("modality") != "event" or
            meta.get("official_split") != "challenging_train"):
        raise ValueError(f"baseline report is not the stock event search run: {report_path}")
    metrics = report["metrics"]
    for name, value in (("mAP50", map50), ("mAP50_95", map95), ("params", params)):
        if not math.isclose(float(metrics[name]), value, abs_tol=1e-6):
            raise ValueError(f"baseline report and CSV disagree on {name}: {root}")
    args = yaml.safe_load(args_path.read_text())
    protocol = yaml.safe_load(PROTOCOL.read_text())
    if (args.get("pretrained") is not False or args.get("seed") != seed or
            args.get("epochs") != protocol["challenging"]["search"]["epochs"] or
            args.get("optimizer") != protocol["families"]["yolo11"]["training"]["optimizer"] or
            args.get("imgsz") != protocol["proxy"]["img_width"] or
            args.get("batch") != protocol["proxy"]["batch_size"]):
        raise ValueError(f"baseline args do not match scratch search protocol: {args_path}")
    return {"seed": seed, "mAP50": map50, "mAP50_95": map95,
            "params": int(params), "report_sha256": digest(report_path),
            "args_sha256": digest(args_path)}


def run_seed(seed, output=OUTPUT):
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    before = identity()  # Fails early if the export, dependency, or split is missing.
    root, csv_path, report_path, args_path = paths(seed, output)
    if root.exists():
        raise FileExistsError(f"baseline seed directory already exists; review it before retrying: {root}")
    root.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).parent / "candidate_runner.py"),
               "--family", "yolo11", "--model", "network", "--seed", str(seed),
               "--protocol", str(PROTOCOL), "--sota_root", str(root), "--device", "cuda"]
    start = time.monotonic()
    completed = subprocess.run(command, check=False)
    elapsed = time.monotonic() - start
    if completed.returncode:
        raise RuntimeError(f"baseline seed {seed} exited {completed.returncode}; inspect {root}")
    result = validate_seed(seed, output)
    after = identity()
    if before != after:
        raise RuntimeError("protocol, split, environment, or evaluation code changed during training")
    result.update({"wall_time_seconds": elapsed, "identity": before,
                   "result_csv": str(csv_path.relative_to(FRED_DIR)),
                   "report": str(report_path.relative_to(FRED_DIR)),
                   "args": str(args_path.relative_to(FRED_DIR))})
    (root / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Seed {seed}: mAP50={result['mAP50']:.6f}, mAP50:95={result['mAP50_95']:.6f}, "
          f"params={result['params']}, wall={elapsed:.1f}s")


def summarize(output=OUTPUT):
    current = identity()
    runs = []
    for seed in SEEDS:
        root, _, report_path, args_path = paths(seed, output)
        recorded = json.loads((root / "run.json").read_text())
        validated = validate_seed(seed, output)
        if (recorded.get("identity") != current or
                any(recorded.get(k) != v for k, v in validated.items()) or
                recorded.get("wall_time_seconds", 0) <= 0):
            raise ValueError(f"baseline seed {seed} is stale, altered, or incomplete")
        runs.append(recorded)
    if len({r["params"] for r in runs}) != 1:
        raise ValueError("stock seed parameter counts differ across runs")
    aggregate = {key: {"mean": statistics.mean(r[key] for r in runs),
                       "sample_sd": statistics.stdev(r[key] for r in runs)}
                 for key in ("mAP50", "mAP50_95", "wall_time_seconds")}
    protocol = yaml.safe_load(PROTOCOL.read_text())
    summary = {"schema_version": "fred-stock-detector-baseline-v1", "family": "yolo11",
               "modality": protocol["proxy"]["modality"],
               "official_split": protocol["proxy"]["official_split"],
               "initialization": "scratch",
               "identity": current, "seed_yaml": str(SEED_YAML.relative_to(FRED_DIR)),
               "seeds": list(SEEDS), "params": runs[0]["params"],
               "aggregate": aggregate, "runs": runs}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    print(f"Saved {output / 'summary.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-seed")
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "run-seed":
        run_seed(args.seed)
    else:
        summarize()


if __name__ == "__main__":
    main()
