"""Independent EMS-YOLO scratch baseline: run seeds 0/1/2, then summarize."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from .protocol_fingerprint import compute_fingerprint

FRED = Path(__file__).resolve().parents[1]
PROTOCOL = FRED / "configs/phase1/protocol.yaml"
SEED = FRED / "seeds/seed_ems_yolo.py"
DEFAULT_OUTPUT = FRED / "baseline_validation/phase1_detection/ems_yolo_stock"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    return compute_fingerprint(FRED.parents[1], "ems_yolo", "stock_ems_yolo", SEED)


def validate(root, seed):
    report_path = root / "results/seed_evaluation.json"
    report = json.loads(report_path.read_text())
    metadata = report["metadata"]
    args_path = root / "training_args/seed/args.yaml"
    args = yaml.safe_load(args_path.read_text())
    protocol = yaml.safe_load(PROTOCOL.read_text())
    family = protocol["families"]["ems_yolo"]
    if (report["mode"] != "validation" or report["gene_id"] != "seed"
            or metadata["model_identity"] != digest(SEED)
            or metadata["official_split"] != "challenging_train"
            or metadata["input_representation"] != "released_event_frame_repeated_over_time"
            or metadata["upstream"] != family["upstream"]
            or args["initialization"] != "scratch" or args["proxy"]["seed"] != seed
            or args["family_settings"] != family
            or args["proxy"]["epochs"] != family["training"]["epochs"]):
        raise ValueError(f"EMS-YOLO baseline provenance mismatch: {root}")
    with (root / "results/seed_results.csv").open() as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 1:
        raise ValueError(f"invalid result rows: {root}")
    values = list(map(float, rows[0][:3]))
    if (not all(math.isfinite(value) for value in values)
            or not 0 <= values[1] <= values[0] <= 1
            or values[2] <= 0 or not values[2].is_integer()):
        raise ValueError(f"failed EMS-YOLO baseline: {root}")
    run = json.loads((root / "training_args/seed/run.json").read_text())
    if run["wall_time_seconds"] <= 0:
        raise ValueError("missing EMS-YOLO baseline duration")
    if metadata["checkpoint_identity"] != digest(root / "trained_models/seed.pt"):
        raise ValueError("EMS-YOLO baseline weights changed")
    return dict(seed=seed, mAP50=values[0], mAP50_95=values[1], params=int(values[2]),
                wall_time_seconds=run["wall_time_seconds"], report_sha256=digest(report_path),
                args_sha256=digest(args_path))


def run_seed(seed, output, device):
    before = identity()
    root = output / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [sys.executable, "-m", "sota.FRED_LLM_GE.phase1_detection.candidate_runner",
         "--family", "ems_yolo", "--model", "network", "--seed", str(seed),
         "--device", device, "--protocol", str(PROTOCOL), "--sota_root", str(root)],
        cwd=FRED.parents[1], check=True)
    result = validate(root, seed)
    if identity() != before:
        raise ValueError("EMS-YOLO protocol/code/data changed during baseline")
    (root / "baseline.json").write_text(json.dumps(dict(result, identity=before), indent=2) + "\n")
    print(result)


def summarize(output):
    current = identity()
    runs = []
    for seed in (0, 1, 2):
        root = output / f"seed_{seed}"
        recorded = json.loads((root / "baseline.json").read_text())
        actual = validate(root, seed)
        if recorded != dict(actual, identity=current):
            raise ValueError(f"stale, incomplete, or changed EMS-YOLO baseline: {root}")
        runs.append(actual)
    if len({run["params"] for run in runs}) != 1:
        raise ValueError("EMS-YOLO seed parameter counts disagree")
    aggregate = {key: dict(mean=statistics.mean(run[key] for run in runs),
                           sample_sd=statistics.stdev(run[key] for run in runs))
                 for key in ("mAP50", "mAP50_95", "wall_time_seconds")}
    summary = dict(
        schema_version="fred-stock-detector-baseline-v1", family="ems_yolo",
        initialization="scratch", identity=current, seeds=[0, 1, 2], params=runs[0]["params"],
        input_representation="released_event_frame_repeated_over_time",
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
    commands.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "run-seed":
        run_seed(args.seed, args.output.resolve(), args.device)
    else:
        summarize(args.output.resolve())


if __name__ == "__main__":
    main()
