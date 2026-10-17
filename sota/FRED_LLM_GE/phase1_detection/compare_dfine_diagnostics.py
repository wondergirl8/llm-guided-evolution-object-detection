"""Compare repeated D-FINE search-validation diagnostic reports.

Only reads reports. It does not train, evaluate, or use the held-out split.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


_SAME_RUN_FIELDS = ("dataset_revision", "manifest_identity", "split_identity",
                    "annotation_policy", "postprocessing", "modality",
                    "benchmark_split", "project_split", "official_split",
                    "validation_sample_ids_sha256", "training_recipe_identity")
_DIAGNOSTIC_FIELDS = ("mAP75", "mAP_small", "recall_small",
                      "false_negatives_iou50", "poorly_localized_iou50_to75")


def compare(paths):
    """Refuse unlike runs, then expose per-run, per-sequence, and seed spread."""
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
                         **report["metrics"],
                         **{key: details["overall"][key] for key in _DIAGNOSTIC_FIELDS}))
        for sequence, data in details["by_sequence"].items():
            sequences.append(dict(model=gene, seed=seed, sequence=sequence, **data))

    if any(len(seeds) < 2 for seeds in seen_seeds.values()):
        raise ValueError("each model needs at least two independent training seeds")
    if any(seeds != next(iter(seen_seeds.values())) for seeds in seen_seeds.values()):
        raise ValueError("compare every model using the same training seeds")
    runs.sort(key=lambda row: (row["model"], row["seed"]))
    sequences.sort(key=lambda row: (row["sequence"], row["model"], row["seed"]))
    spread = {}
    for gene in sorted(seen_seeds):
        values = [row["mAP50_95"] for row in runs if row["model"] == gene]
        spread[gene] = {"seeds": len(values), "mean_mAP50_95": mean(values),
                        "min_mAP50_95": min(values), "max_mAP50_95": max(values)}
    return {"shared_protocol": shared, "runs": runs,
            "mAP50_95_seed_spread": spread, "by_sequence": sequences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reports), indent=2))


if __name__ == "__main__":
    main()
