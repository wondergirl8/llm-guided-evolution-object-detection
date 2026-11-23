"""Independent three-seed Faster R-CNN stock baseline; no LLM required."""

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from .candidate_runner import DEFAULT_PROTOCOL, FRED_DIR, load_protocol, run
from .protocol_fingerprint import compute_fingerprint
from .seeds.faster_rcnn_scratch.selection import freeze_manifest

ROOT = FRED_DIR.parents[1]
SEED = FRED_DIR / "seeds/seed_faster_rcnn_scratch.py"
SEEDS = (0, 1, 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(run_root):
    return compute_fingerprint(ROOT, "faster_rcnn_scratch", Path(run_root).name, SEED)


def validate_seed(run_root, seed):
    root = Path(run_root) / "baseline" / f"seed_{seed}"
    report_path = root / "results/seed_evaluation.json"
    args_path = root / "training_args/seed/args.json"
    weights = root / "trained_models/seed.pt"
    with (root / "results/seed_results.csv").open() as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError("Incomplete baseline result")
    map50, map95, params = map(float, rows[0][:3])
    if not (0 <= map95 <= map50 <= 1 and 0 < params < 999_999_999 and params.is_integer()):
        raise ValueError("Invalid or failed baseline fitness")
    report = json.loads(report_path.read_text())
    args = json.loads(args_path.read_text())
    proxy = args["proxy"]
    initialization = args["initialization"]
    config = {k: v for k, v in args.items() if k != "initialization"}
    if report["metadata"]["training_config_identity"] != hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError("Baseline recipe identity differs from its report")
    expected_family = copy.deepcopy(load_protocol(DEFAULT_PROTOCOL)["families"]["faster_rcnn_scratch"])
    expected_family["search"]["seed"] = seed
    if args["family_settings"] != expected_family:
        raise ValueError("Baseline optimizer or backbone settings differ from the frozen recipe")
    if (report["mode"] != "validation" or report["gene_id"] != "seed"
            or report["metadata"]["model_identity"] != digest(SEED)
            or report["metadata"]["official_split"] != "challenging_train"
            or initialization["pretrained"] is not False
            or initialization["checkpoint"] is not None or initialization["seed"] != seed
            or proxy["seed"] != seed or proxy["epochs"] != 3
            or proxy["batch_size"] != 2 or proxy["img_width"] != 640
            or proxy["frame_step"] != 50 or proxy["val_frame_step"] != 10
            or len(proxy["train_sequences"]) != 152 or len(proxy["val_sequences"]) != 20):
        raise ValueError("Baseline does not match the official scratch search protocol")
    for name, value in (("mAP50", map50), ("mAP50_95", map95), ("params", params)):
        if abs(report["metrics"][name] - value) > 1e-6:
            raise ValueError("Baseline report and fitness disagree")
    if report["metadata"]["checkpoint_identity"] != digest(weights):
        raise ValueError("Baseline checkpoint identity changed")
    return {"seed": seed, "mAP50": map50, "mAP50_95": map95, "params": int(params),
            "report_sha256": digest(report_path), "args_sha256": digest(args_path),
            "checkpoint_sha256": digest(weights)}


def run_seed(run_root, seed, device="cuda", resume=False):
    if seed not in SEEDS:
        raise ValueError("Baseline seed must be 0, 1, or 2")
    run_root = Path(run_root).resolve()
    before = identity(run_root)
    root = run_root / "baseline" / f"seed_{seed}"
    if resume and (root / "run.json").exists():
        recorded = json.loads((root / "run.json").read_text())
        validated = validate_seed(run_root, seed)
        if recorded.get("identity") != before or any(recorded.get(k) != v for k, v in validated.items()):
            raise ValueError("Existing baseline differs from frozen run")
        return validated
    root.mkdir(parents=True, exist_ok=False)
    protocol = copy.deepcopy(load_protocol(DEFAULT_PROTOCOL))
    protocol["proxy"]["seed"] = seed
    start = time.monotonic()
    run("seed", "network", root / "models", "faster_rcnn_scratch", protocol, root, device)
    result = validate_seed(run_root, seed)
    if before != identity(run_root):
        raise RuntimeError("Code, dependencies, or data changed during baseline training")
    freeze_manifest(root / "run.json", dict(result, identity=before,
                    wall_time_seconds=time.monotonic() - start))
    return result


def verify_summary(recorded, computed):
    """Keep provenance exact; tolerate only roundoff in derived aggregates."""
    if (set(recorded) != set(computed)
            or {k: v for k, v in recorded.items() if k != "aggregate"}
            != {k: v for k, v in computed.items() if k != "aggregate"}):
        raise ValueError("Frozen baseline summary provenance or seed records differ")
    actual, expected = recorded["aggregate"], computed["aggregate"]
    if set(actual) != set(expected):
        raise ValueError("Frozen baseline summary aggregate fields differ")
    for metric, statistics_ in expected.items():
        if set(actual[metric]) != set(statistics_):
            raise ValueError(f"Frozen baseline summary statistics differ: {metric}")
        for name, value in statistics_.items():
            saved = actual[metric][name]
            if (type(saved) not in (int, float) or not math.isfinite(saved)
                    or not math.isfinite(value)
                    or not math.isclose(saved, value, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError(f"Frozen baseline summary statistic differs: {metric}.{name}")


def summarize(run_root, expected_fingerprint=None):
    current = identity(run_root)
    if expected_fingerprint is not None:
        expected = {k: v for k, v in expected_fingerprint.items() if k != "evolution_settings"}
        if expected != current:
            raise ValueError("Baseline and evolutionary run fingerprints differ")
    runs = []
    for seed in SEEDS:
        record = json.loads((Path(run_root) / "baseline" / f"seed_{seed}" / "run.json").read_text())
        validated = validate_seed(run_root, seed)
        if (record.get("identity") != current or record.get("wall_time_seconds", 0) <= 0
                or any(record.get(k) != v for k, v in validated.items())):
            raise ValueError(f"Baseline seed {seed} is stale or incomplete")
        runs.append(record)
    if len({r["params"] for r in runs}) != 1:
        raise ValueError("Stock baseline parameter counts differ")
    summary = {"family": "faster_rcnn_scratch", "initialization": "scratch",
               "identity": current, "runs": runs, "aggregate": {
                   key: {"mean": statistics.mean(r[key] for r in runs),
                         "sample_sd": statistics.stdev(r[key] for r in runs)}
                   for key in ("mAP50", "mAP50_95", "wall_time_seconds")}}
    path = Path(run_root) / "baseline/summary.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        verify_summary(recorded, summary)
        return recorded
    freeze_manifest(path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-seed", "summarize"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "run-seed":
        if args.seed is None:
            parser.error("run-seed requires --seed")
        print(run_seed(args.run_root, args.seed, args.device, args.resume))
    else:
        print(json.dumps(summarize(args.run_root), indent=2))


if __name__ == "__main__":
    main()
