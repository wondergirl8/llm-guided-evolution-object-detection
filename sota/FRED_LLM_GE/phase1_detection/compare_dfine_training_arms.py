"""Compare paired D-FINE training recipes using search-validation reports only."""

import argparse
import copy
import json
from pathlib import Path
from statistics import mean


ARMS = ("baseline", "longer_budget", "cosine_lr")
SAME_FIELDS = (
    "dataset_revision", "manifest_identity", "split_identity", "annotation_policy",
    "model_identity", "code_revision", "postprocessing", "modality",
    "benchmark_split", "project_split", "official_split",
    "validation_sample_ids_sha256", "training_sample_ids_sha256", "runtime_environment",
)
METRICS = ("mAP50", "mAP50_95", "mAP75", "training_seconds")


def compare(paths):
    """Require complete, comparable seed triplets before summarizing recipes."""
    runs = {}
    shared = None
    reference = None
    shared_recipe = None
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text())
        metadata = report["metadata"]
        arm, seed = metadata.get("training_arm"), metadata.get("training_seed")
        if arm not in ARMS or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{path}: expected a D-FINE training arm and nonnegative seed")
        key = (seed, arm)
        if key in runs:
            raise ValueError(f"{path}: duplicate seed {seed} and arm {arm}")
        if report.get("mode") != "validation" or "diagnostics" not in report:
            raise ValueError(f"{path}: expected a search-validation diagnostic report")
        fields = {name: metadata.get(name) for name in SAME_FIELDS}
        if any(value is None for value in fields.values()):
            raise ValueError(f"{path}: missing comparison metadata")
        recipe = copy.deepcopy(metadata.get("training_recipe"))
        expected_epochs = 40 if arm == "longer_budget" else 20
        expected_schedule = "cosine" if arm == "cosine_lr" else "constant"
        if (not isinstance(recipe, dict) or recipe.get("family") != "dfine"
                or not isinstance(recipe.get("proxy"), dict)
                or not isinstance(recipe.get("family_settings"), dict)
                or recipe["proxy"].get("epochs") != expected_epochs
                or recipe["proxy"].get("seed") != seed
                or recipe["family_settings"].get("lr_schedule") != expected_schedule):
            raise ValueError(f"{path}: training recipe does not match {arm}")
        recipe["proxy"].pop("seed")
        recipe["proxy"]["epochs"] = 20
        recipe["family_settings"]["lr_schedule"] = "constant"
        if shared_recipe is None:
            shared_recipe = recipe
        elif recipe != shared_recipe:
            raise ValueError(f"{path}: training settings differ beyond epochs or schedule")
        protocol = (report.get("gene_id"), report.get("schema_version"),
                    report.get("metric_units"), report.get("num_frames"),
                    report.get("evaluator"), report["metrics"].get("params"), fields,
                    report["diagnostics"].get("definition"))
        if reference is None:
            reference, shared = protocol, fields
        elif protocol != reference:
            raise ValueError(f"{path}: model, data, evaluator, environment, or policy differs")
        history = report.get("training_history", [])
        if len(history) != expected_epochs or [row.get("epoch") for row in history] != list(
                range(1, expected_epochs + 1)):
            raise ValueError(f"{path}: incomplete {arm} training history")
        losses = ("loss_cls", "loss_bbox", "loss_giou", "loss_fgl", "loss_go_lsd")
        if any(any(name not in row for name in losses) for row in history):
            raise ValueError(f"{path}: missing individual loss terms")
        rates = [row["lr"] for row in history]
        if arm == "cosine_lr":
            valid_rates = all(a > b for a, b in zip(rates, rates[1:]))
        else:
            valid_rates = all(rate == rates[0] for rate in rates)
        if not valid_rates:
            raise ValueError(f"{path}: learning-rate history does not match {arm}")
        if metadata.get("training_seconds", 0) <= 0:
            raise ValueError(f"{path}: missing training cost")
        runs[key] = {
            "seed": seed, "arm": arm, "report": str(path),
            "initial_weights_sha256": metadata.get("initial_weights_sha256"),
            "order": [row.get("order_sha256") for row in history],
            "mAP50": report["metrics"]["mAP50"],
            "mAP50_95": report["metrics"]["mAP50_95"],
            "mAP75": report["diagnostics"]["overall"]["mAP75"],
            "training_seconds": metadata["training_seconds"],
            "final_losses": {name: history[-1][name] for name in losses},
        }
    seeds = sorted({seed for seed, _ in runs})
    if len(seeds) < 2 or set(runs) != {(seed, arm) for seed in seeds for arm in ARMS}:
        raise ValueError("provide all three training arms for at least two paired seeds")
    for seed in seeds:
        baseline = runs[seed, "baseline"]
        if not baseline["initial_weights_sha256"] or any(not value for value in baseline["order"]):
            raise ValueError(f"seed {seed}: missing initialization or training order identity")
        for arm in ARMS[1:]:
            other = runs[seed, arm]
            if (other["initial_weights_sha256"] != baseline["initial_weights_sha256"]
                    or other["order"][:20] != baseline["order"]):
                raise ValueError(f"seed {seed}: arms did not share initialization and frame order")
    summary = {arm: {name: mean(runs[seed, arm][name] for seed in seeds)
                     for name in METRICS} for arm in ARMS}
    paired_deltas = [{"seed": seed, "arm": arm,
                      **{name: runs[seed, arm][name] - runs[seed, "baseline"][name]
                         for name in METRICS}}
                     for seed in seeds for arm in ARMS[1:]]
    visible_runs = [{key: value for key, value in runs[seed, arm].items()
                     if key not in ("order", "initial_weights_sha256")}
                    for seed in seeds for arm in ARMS]
    return {"shared_protocol": shared, "seeds": seeds, "runs": visible_runs,
            "arm_means": summary, "paired_deltas_vs_baseline": paired_deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reports), indent=2))


if __name__ == "__main__":
    main()
