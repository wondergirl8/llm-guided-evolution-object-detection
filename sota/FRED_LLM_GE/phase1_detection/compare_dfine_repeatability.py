"""Compare D-FINE checkpoint re-scoring and unchanged-seed training repeats."""

import argparse
import json
from pathlib import Path


_SHARED = (
    "model_identity", "code_revision", "runtime_code_sha256",
    "dataset_revision", "manifest_identity", "split_identity",
    "annotation_policy", "training_recipe_identity",
    "training_sample_ids_sha256", "training_targets_sha256",
    "validation_sample_ids_sha256", "validation_targets_sha256",
    "postprocessing", "runtime_environment",
)
_METRICS = ("mAP50", "mAP50_95")


def _read(path):
    return json.loads(Path(path).read_text())


def _validate_training(report):
    if (report.get("gene_id") != "seed" or report.get("mode") != "validation"
            or report.get("metadata", {}).get("official_split") != "challenging_train"):
        raise ValueError("expected an unchanged D-FINE seed search-validation report")
    metadata = report["metadata"]
    required = (*_SHARED, "training_config_identity", "training_seed", "initial_weights_sha256",
                "training_order_sha256", "checkpoint_identity")
    missing = [name for name in required if metadata.get(name) is None]
    if missing or not report.get("training_history"):
        raise ValueError(f"repeat report lacks step-1 evidence: {missing or 'training_history'}")
    if any(name not in report.get("metrics", {}) for name in _METRICS):
        raise ValueError("repeat report lacks mAP50 or mAP50:95")
    return metadata


def _first_history_difference(first, second):
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return {"epoch": a.get("epoch", index + 1),
                    "fields": sorted(key for key in a.keys() | b.keys()
                                     if a.get(key) != b.get(key))}
    if len(first) != len(second):
        return {"epoch": min(len(first), len(second)) + 1,
                "fields": ["missing_epoch"]}
    return None


def compare(rescore_path, repeat_paths, independent_paths=()):
    """Return evidence and a diagnosis; never infer an architecture gain."""
    if len(repeat_paths) != 2:
        raise ValueError("provide exactly two same-seed training reports")
    if independent_paths and len(independent_paths) < 2:
        raise ValueError("provide at least two independent-seed reports")
    rescore = _read(rescore_path)
    if (rescore.get("schema_version") != "fred-dfine-rescore-v1"
            or rescore.get("gene_id") != "seed" or len(rescore.get("passes", [])) != 2):
        raise ValueError("expected a two-pass D-FINE seed rescore report")
    repeats = [_read(path) for path in repeat_paths]
    independent = [_read(path) for path in independent_paths]
    metadata = [_validate_training(report) for report in repeats + independent]
    first, second = metadata[:2]
    if first["training_seed"] != second["training_seed"]:
        raise ValueError("the two training repeats must use the same seed")
    seeds = [item["training_seed"] for item in metadata[2:]]
    if len(set(seeds)) != len(seeds) or first["training_seed"] in seeds:
        raise ValueError("independent training seeds must be distinct from each other and the repeat seed")

    shared_differences = sorted(
        name for name in _SHARED if any(item[name] != first[name] for item in metadata[1:]))
    if first["training_config_identity"] != second["training_config_identity"]:
        shared_differences.append("training_config_identity")
    reference_differences = sorted(
        name for name in ("model_identity", "validation_sample_ids_sha256", "postprocessing")
        if rescore.get("metadata", {}).get(name) != first[name])
    same_initial_weights = first["initial_weights_sha256"] == second["initial_weights_sha256"]
    same_training_order = first["training_order_sha256"] == second["training_order_sha256"]
    history_difference = _first_history_difference(
        repeats[0]["training_history"], repeats[1]["training_history"])
    same_checkpoint = first["checkpoint_identity"] == second["checkpoint_identity"]
    metric_differences = {
        name: repeats[1]["metrics"][name] - repeats[0]["metrics"][name]
        for name in _METRICS
    }
    scoring_repeats_match = (rescore["passes"][0] == rescore["passes"][1]
                             and rescore.get("repeated_scoring_matches") is True)
    scoring_reference_matches = rescore.get("reference_scoring_matches") is True

    if not scoring_repeats_match:
        finding = "same_checkpoint_scoring_changed"
    elif not scoring_reference_matches:
        finding = "scoring_differs_from_reference"
    elif shared_differences or reference_differences:
        finding = "protocol_or_data_identity_changed"
    elif not same_initial_weights or not same_training_order:
        finding = "initialization_or_training_order_changed"
    elif history_difference is not None:
        finding = "training_history_diverged"
    elif not same_checkpoint or any(metric_differences.values()):
        finding = "checkpoint_or_metric_diverged_after_matching_history"
    else:
        finding = "same_seed_repeated"

    by_seed = [repeats[0], *independent]
    independent_weights_vary = len({item["initial_weights_sha256"] for item in
                                    [first, *metadata[2:]]}) == len(by_seed)
    independent_order_varies = len({item["training_order_sha256"] for item in
                                   [first, *metadata[2:]]}) == len(by_seed)
    if finding == "same_seed_repeated" and len(independent) >= 2 and not (
            independent_weights_vary and independent_order_varies):
        finding = "independent_seeds_reused_initialization_or_order"
    spread = {
        name: {"min": min(report["metrics"][name] for report in by_seed),
               "max": max(report["metrics"][name] for report in by_seed)}
        for name in _METRICS
    }
    for value in spread.values():
        value["range"] = value["max"] - value["min"]
    return {
        "schema_version": "fred-dfine-repeatability-v1",
        "finding": finding,
        "complete": finding == "same_seed_repeated" and len(independent) >= 2,
        "same_checkpoint_scoring": {
            "passes_match": scoring_repeats_match,
            "reference_matches": scoring_reference_matches,
            "checkpoint_identity": rescore.get("checkpoint_identity"),
        },
        "same_seed_training": {
            "seed": first["training_seed"],
            "shared_identity_differences": shared_differences,
            "reference_identity_differences": reference_differences,
            "initial_weights_match": same_initial_weights,
            "training_order_matches": same_training_order,
            "first_history_difference": history_difference,
            "checkpoint_matches": same_checkpoint,
            "metric_differences": metric_differences,
        },
        "independent_seeds": [first["training_seed"], *seeds],
        "independent_initial_weights_vary": independent_weights_vary,
        "independent_training_orders_vary": independent_order_varies,
        "independent_seed_spread": spread if len(independent) >= 2 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", required=True, type=Path)
    parser.add_argument("--repeat", required=True, nargs=2, type=Path,
                        metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--independent", nargs="*", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = compare(args.rescore, args.repeat, args.independent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
