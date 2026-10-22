"""Compare paired YOLO26 training ablations on the search-validation split."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

try:
    from . import candidate_runner, yolo26_ablation
except ImportError:
    import candidate_runner
    import yolo26_ablation


METRICS = ("mAP50", "mAP50_95", "mAP75", "mAP_small", "recall_small",
           "training_seconds")
SAME_FIELDS = (
    "dataset_revision", "manifest_identity", "split_identity", "annotation_policy",
    "model_identity", "code_revision", "postprocessing", "modality",
    "benchmark_split", "project_split", "official_split", "source_model_sha256",
    "ablation_code_sha256", "ablation_plan_sha256", "ablation_protocol_sha256",
    "training_export_sha256", "training_marker_sha256", "training_label_sha256",
    "validation_marker_sha256", "validation_label_sha256",
    "validation_sample_ids_sha256", "ultralytics_version",
    "runtime_environment",
)


def _selected_cases(plan, experiments):
    cases = plan["experiments"]
    requested = experiments or [name for name, case in cases.items()
                                if case.get("compare_to") and "edit" not in case]
    selected = set()
    for name in requested:
        if name not in cases or "edit" in cases[name]:
            raise ValueError(f"{name}: choose a listed training experiment")
        chain = set()
        while name is not None:
            if name in chain:
                raise ValueError("ablation comparisons must not form a cycle")
            chain.add(name)
            if "edit" in cases[name]:
                raise ValueError("training comparison cannot include an architecture edit")
            selected.add(name)
            name = cases[name].get("compare_to")
    if not any(cases[name].get("compare_to") for name in selected):
        raise ValueError("choose at least one experiment with a comparison")
    return [name for name in cases if name in selected]


def _metric(value, name, path):
    if value is None and name in ("mAP_small", "recall_small"):
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (name != "training_seconds" and value > 1)
            or (name == "training_seconds" and value == 0)):
        raise ValueError(f"{path}: invalid {name}")
    return float(value)


def compare(root, plan, protocol, experiments=None):
    """Require all planned seeds and matching model/data/runtime before deltas."""
    names = _selected_cases(plan, experiments)
    seeds = plan["repeat_seeds"]
    runs = {}
    shared = None
    reference = None
    sequence_counts = None
    default_epochs = protocol["challenging"]["search"]["epochs"]
    for name in names:
        case = plan["experiments"][name]
        for seed in seeds:
            path = Path(root) / name / f"seed_{seed}" / "results/ablation_evaluation.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing paired ablation report: {path}")
            report = json.loads(path.read_text())
            metadata = report["metadata"]
            details = report.get("diagnostics")
            if (report.get("mode") != "validation" or report.get("gene_id") != "ablation"
                    or not isinstance(details, dict)
                    or metadata.get("ablation_experiment") != name
                    or metadata.get("ablation_comparison") != case.get("compare_to")
                    or metadata.get("ablation_seed") != seed
                    or metadata.get("completed_epochs") != case.get("epochs", default_epochs)):
                raise ValueError(f"{path}: incomplete or mismatched experiment report")
            fields = {key: metadata.get(key) for key in SAME_FIELDS}
            if any(value is None for value in fields.values()):
                raise ValueError(f"{path}: missing comparison identity")
            counts = {key: (item["frames"], item["ground_truth"])
                      for key, item in details["by_sequence"].items()}
            identity = (report.get("schema_version"), report.get("metric_units"),
                        report.get("num_frames"), report.get("evaluator"),
                        report["metrics"].get("params"), details.get("definition"), fields)
            if reference is None:
                reference, shared, sequence_counts = identity, fields, counts
            elif identity != reference or counts != sequence_counts:
                raise ValueError(f"{path}: model, data, evaluator, or runtime differs")
            options = case.get("train", {})
            if ("optimizer" in options and metadata.get("resolved_optimizer") != options["optimizer"]
                    or "lr0" in options and metadata.get("resolved_lr0") != options["lr0"]
                    or "momentum" in options
                    and metadata.get("resolved_momentum") != options["momentum"]
                    or "warmup_bias_lr" in options
                    and metadata.get("resolved_warmup_bias_lr") != options["warmup_bias_lr"]):
                raise ValueError(f"{path}: actual optimizer settings differ from the plan")
            if not metadata.get("initial_weights_sha256"):
                raise ValueError(f"{path}: missing initial model identity")
            values = {}
            for key in METRICS:
                if key == "training_seconds":
                    value = metadata.get(key)
                elif key in report["metrics"]:
                    value = report["metrics"][key]
                else:
                    value = details["overall"].get(key)
                values[key] = _metric(value, key, path)
            runs[name, seed] = {
                "experiment": name, "seed": seed, "report": str(path),
                "initial_weights_sha256": metadata["initial_weights_sha256"],
                "resolved_optimizer": metadata.get("resolved_optimizer"),
                "resolved_lr0": metadata.get("resolved_lr0"),
                "resolved_momentum": metadata.get("resolved_momentum"),
                "resolved_warmup_bias_lr": metadata.get("resolved_warmup_bias_lr"),
                **values,
            }

    comparisons = []
    roots = [name for name in names if plan["experiments"][name].get("compare_to") is None]
    for name in roots:
        if len({runs[name, seed]["initial_weights_sha256"] for seed in seeds}) != len(seeds):
            raise ValueError(f"{name}: independent seeds reused the same initial weights")
    for name in names:
        baseline = plan["experiments"][name].get("compare_to")
        if baseline is None:
            continue
        for seed in seeds:
            if runs[name, seed]["initial_weights_sha256"] != runs[baseline, seed][
                    "initial_weights_sha256"]:
                raise ValueError(f"{name}, seed {seed}: initial model weights differ from {baseline}")
        deltas = [{"seed": seed, **{
            key: (runs[name, seed][key] - runs[baseline, seed][key]
                  if runs[name, seed][key] is not None and runs[baseline, seed][key] is not None
                  else None) for key in METRICS}} for seed in seeds]
        mean_deltas = {key: (mean(row[key] for row in deltas)
                             if all(row[key] is not None for row in deltas) else None)
                       for key in METRICS}
        baseline_spread = (max(runs[baseline, seed]["mAP50_95"] for seed in seeds)
                           - min(runs[baseline, seed]["mAP50_95"] for seed in seeds))
        comparisons.append({
            "experiment": name, "compare_to": baseline,
            "paired_deltas": deltas, "mean_deltas": mean_deltas,
            "baseline_mAP50_95_seed_range": baseline_spread,
            "consistent_mAP50_95_gain_for_review": (
                all(row["mAP50_95"] > 0 for row in deltas)
                and mean_deltas["mAP50_95"] > baseline_spread),
        })
    return {
        "scope": "search-validation training-recipe screen; no held-out or architecture claim",
        "shared_identity": shared, "seeds": seeds,
        "runs": [{key: value for key, value in runs[name, seed].items()
                  if key != "initial_weights_sha256"} for name in names for seed in seeds],
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=yolo26_ablation.DEFAULT_OUTPUT)
    parser.add_argument("--plan", type=Path, default=yolo26_ablation.DEFAULT_PLAN)
    parser.add_argument("--protocol", type=Path, default=candidate_runner.DEFAULT_PROTOCOL)
    parser.add_argument("--experiment", action="append",
                        help="compare this training arm and its reference; omit for all training arms")
    parser.add_argument("--output", type=Path, help="new JSON comparison file")
    args = parser.parse_args()
    protocol = candidate_runner.load_protocol(args.protocol)
    plan = yolo26_ablation.load_plan(args.plan, protocol)
    result = json.dumps(compare(args.root, plan, protocol, args.experiment), indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(result)
        print(args.output)


if __name__ == "__main__":
    main()
