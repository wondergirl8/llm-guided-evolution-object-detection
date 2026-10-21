"""Compare D-FINE search-validation diagnostic reports.

Only reads reports. It does not train, evaluate, or use the held-out split.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


_SAME_RUN_FIELDS = ("dataset_revision", "manifest_identity", "split_identity",
                    "annotation_policy", "postprocessing", "modality",
                    "benchmark_split", "project_split", "official_split",
                    "validation_sample_ids_sha256", "training_recipe_identity")
_DIAGNOSTIC_FIELDS = ("mAP75", "mAP_small", "recall_small",
                      "false_negatives_iou50", "poorly_localized_iou50_to75")
_DELTA_FIELDS = ("mAP50", "mAP50_95", *_DIAGNOSTIC_FIELDS)
_GAIN_FIELDS = ("mAP50_95", "mAP75", "mAP_small", "recall_small")


def _confirm_gains(reports, runs):
    """Screen paired gains against observed seed-to-seed range; do not promote."""
    by_key = {(row["model"], row["seed"]): row for row in runs}
    if "seed" not in {row["model"] for row in runs} or len(by_key) == len({row["seed"] for row in runs}):
        raise ValueError("gain confirmation needs a seed and at least one candidate")
    metadata = {(report["gene_id"], report["metadata"]["training_seed"]): report["metadata"]
                for _, report in reports}
    required = ("code_revision", "runtime_code_sha256", "training_sample_ids_sha256",
                "training_targets_sha256", "validation_targets_sha256",
                "runtime_environment", "inference_device")
    reference = metadata[next(iter(metadata))]
    for (model, seed), item in metadata.items():
        if any(item.get(field) is None or item[field] != reference.get(field)
               for field in required):
            raise ValueError(f"{model} seed {seed}: code, data, labels, or hardware differs")
        latency = item.get("preloaded_inference_ms_per_frame")
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"{model} seed {seed}: missing preloaded inference time")
        if not item.get("training_order_sha256"):
            raise ValueError(f"{model} seed {seed}: missing training-order identity")
        if model != "seed" and item["training_order_sha256"] != metadata["seed", seed].get(
                "training_order_sha256"):
            raise ValueError(f"{model} seed {seed}: training frame order differs from seed")
    seeds = sorted(row["seed"] for row in runs if row["model"] == "seed")
    if len({by_key["seed", seed]["params"] for seed in seeds}) != 1:
        raise ValueError("seed parameter count changed across training seeds")
    candidates = {}
    for model in sorted({row["model"] for row in runs} - {"seed"}):
        if by_key[model, seeds[0]]["model_identity"] == by_key["seed", seeds[0]]["model_identity"]:
            raise ValueError(f"{model}: source is unchanged from the seed")
        if len({by_key[model, seed]["params"] for seed in seeds}) != 1:
            raise ValueError(f"{model}: parameter count changed across seeds")
        gains = {}
        for field in _GAIN_FIELDS:
            baseline = [by_key["seed", seed][field] for seed in seeds]
            candidate = [by_key[model, seed][field] for seed in seeds]
            if any(value is None or not math.isfinite(value) for value in baseline + candidate):
                gains[field] = {"passes_variation_screen": False, "reason": "metric unavailable"}
                continue
            paired = [new - old for old, new in zip(baseline, candidate)]
            variation = max(max(baseline) - min(baseline),
                            max(candidate) - min(candidate))
            gains[field] = {
                "seed_mean": mean(baseline), "candidate_mean": mean(candidate),
                "paired_deltas": paired, "mean_gain": mean(paired),
                "observed_variation": variation,
                "passes_variation_screen": all(delta > 0 for delta in paired)
                and mean(paired) > variation,
            }
        passed = [field for field, detail in gains.items()
                  if detail["passes_variation_screen"]]
        seed_latency = mean(metadata["seed", seed]["preloaded_inference_ms_per_frame"]
                            for seed in seeds)
        candidate_latency = mean(metadata[model, seed]["preloaded_inference_ms_per_frame"]
                                 for seed in seeds)
        seed_params = by_key["seed", seeds[0]]["params"]
        candidate_params = by_key[model, seeds[0]]["params"]
        candidates[model] = {
            "metrics": gains, "metrics_exceeding_variation": passed,
            "gain_confirmed_for_review": bool(passed),
            "seed_params": seed_params, "params": candidate_params,
            "delta_params": candidate_params - seed_params,
            "seed_mean_inference_ms_per_frame": seed_latency,
            "mean_inference_ms_per_frame": candidate_latency,
            "delta_inference_ms_per_frame": candidate_latency - seed_latency,
            "paired_inference_deltas_ms": [
                metadata[model, seed]["preloaded_inference_ms_per_frame"]
                - metadata["seed", seed]["preloaded_inference_ms_per_frame"]
                for seed in seeds],
        }
    return {"criterion": "Every paired gain is positive and its mean exceeds the larger "
                         "seed-to-seed range of seed and candidate runs; resource costs "
                         "require human review before promotion.",
            "training_seeds": seeds, "candidates": candidates}


def compare(paths, *, checkpoint_review=False, confirm_gains=False):
    """Refuse unlike runs, then expose per-run, per-sequence, and seed spread."""
    if checkpoint_review and confirm_gains:
        raise ValueError("checkpoint review cannot confirm architecture gains")
    reports = [(Path(path), json.loads(Path(path).read_text())) for path in paths]
    if not reports:
        raise ValueError("provide at least one diagnostic report")
    reference = reports[0][1]
    if "diagnostics" not in reference:
        raise ValueError(f"{reports[0][0]}: missing diagnostics")
    shared = {key: reference["metadata"].get(key) for key in _SAME_RUN_FIELDS}
    if any(value is None for value in shared.values()):
        raise ValueError("reports need D-FINE confirmation metadata")
    reference_split = {sequence: (data["frames"], data["ground_truth"])
                       for sequence, data in reference["diagnostics"]["by_sequence"].items()}
    runs, sequences = [], []
    identities, seen_seeds = {}, defaultdict(set)
    for path, report in reports:
        metadata = report["metadata"]
        gene = report["gene_id"]
        seed = metadata.get("training_seed")
        if (report.get("mode") != "validation" or
                report.get("schema_version") != reference.get("schema_version") or
                report.get("metric_units") != reference.get("metric_units") or
                report.get("num_frames") != reference["num_frames"] or
                report.get("evaluator") != reference["evaluator"] or
                any(metadata.get(key) != value for key, value in shared.items())):
            raise ValueError(f"{path}: different evaluation or training protocol")
        details = report.get("diagnostics")
        if details is None or details.get("definition") != reference["diagnostics"]["definition"]:
            raise ValueError(f"{path}: missing or incompatible diagnostics")
        if {name: (data["frames"], data["ground_truth"])
                for name, data in details["by_sequence"].items()} != reference_split:
            raise ValueError(f"{path}: sequence ground truth differs")
        if not isinstance(seed, int) or seed < 0 or seed in seen_seeds[gene]:
            raise ValueError(f"{path}: missing or duplicate training seed for {gene}")
        seen_seeds[gene].add(seed)
        model_identity = metadata["model_identity"]
        if gene in identities and identities[gene] != model_identity:
            raise ValueError(f"{path}: model source changed across seeds")
        identities[gene] = model_identity
        runs.append(dict(model=gene, seed=seed, report=str(path),
                         model_identity=model_identity,
                         **report["metrics"],
                         **{key: details["overall"][key] for key in _DIAGNOSTIC_FIELDS}))
        for sequence, data in details["by_sequence"].items():
            sequences.append(dict(model=gene, seed=seed, sequence=sequence, **data))

    if not checkpoint_review and any(len(seeds) < 2 for seeds in seen_seeds.values()):
        raise ValueError("each model needs at least two independent training seeds")
    if any(seeds != next(iter(seen_seeds.values())) for seeds in seen_seeds.values()):
        raise ValueError("compare every model using the same training seeds")
    runs.sort(key=lambda row: (row["model"], row["seed"]))
    sequences.sort(key=lambda row: (row["sequence"], row["model"], row["seed"]))
    seed_source = identities.get("seed")
    for row in runs:
        row["same_model_source_as_seed"] = (
            row["model_identity"] == seed_source if seed_source is not None else None)
    spread = {}
    for gene in sorted(seen_seeds):
        values = [row["mAP50_95"] for row in runs if row["model"] == gene]
        spread[gene] = {"seeds": len(values), "mean_mAP50_95": mean(values),
                        "min_mAP50_95": min(values), "max_mAP50_95": max(values)}
    seed_runs = {row["seed"]: row for row in runs if row["model"] == "seed"}
    deltas = [{"model": row["model"], "seed": row["seed"],
               **{key: (row[key] - seed_runs[row["seed"]][key]
                        if row[key] is not None and seed_runs[row["seed"]][key] is not None
                        else None) for key in _DELTA_FIELDS}}
              for row in runs if row["model"] != "seed" and row["seed"] in seed_runs]
    output = {"shared_protocol": shared,
              "comparison_scope": ("exploratory checkpoint review; confirm across independent "
                                   "training seeds before ranking" if checkpoint_review
                                   else "independent training-seed comparison"),
              "runs": runs,
              "deltas_vs_seed": deltas, "mAP50_95_seed_spread": spread,
              "by_sequence": sequences}
    if confirm_gains:
        output["gain_confirmation"] = _confirm_gains(reports, runs)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-review", action="store_true",
                        help="compare selected checkpoints before independent-seed confirmation")
    parser.add_argument("--confirm-gains", action="store_true",
                        help="screen paired candidate gains against measured seed variation")
    parser.add_argument("--output", type=Path,
                        help="write a new JSON review file instead of printing it")
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    result = json.dumps(compare(args.reports, checkpoint_review=args.checkpoint_review,
                                confirm_gains=args.confirm_gains), indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(result)
        print(args.output)


if __name__ == "__main__":
    main()
