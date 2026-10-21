"""Compare D-FINE confidence and NMS policies on one fixed validation checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

try:
    from .candidate_runner import (DEFAULT_PROTOCOL, predict_frames, runtime_environment)
    from .fitness import evaluate_frames
    from .rescore_dfine import prepare_checkpoint
except ImportError:
    from candidate_runner import DEFAULT_PROTOCOL, predict_frames, runtime_environment
    from fitness import evaluate_frames
    from rescore_dfine import prepare_checkpoint


def score_policy(adapter, model, val_data, proxy, device):
    frames = predict_frames(adapter, model, val_data, proxy, device)
    result, details = evaluate_frames(
        frames, params=adapter.param_count(model),
        expected_sample_ids=val_data["sample_ids"], diagnostics=True)
    prediction_hash = hashlib.sha256(json.dumps(
        [(frame["sample_id"], frame["prediction"]) for frame in frames],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "mAP50": result.map50,
        "mAP50_95": result.map50_95,
        "mAP75": details["overall"]["mAP75"],
        "mAP_small": details["overall"]["mAP_small"],
        "recall_small": details["overall"]["recall_small"],
        "predictions": sum(len(frame["prediction"]["scores"]) for frame in frames),
        "predictions_sha256": prediction_hash,
    }


def compare(checkpoint, reference_report, model_name="network", variant_dir=None,
            protocol_path=DEFAULT_PROTOCOL, device="cpu"):
    checkpoint, reference, adapter, source, proxy, val_data, sample_hash = prepare_checkpoint(
        checkpoint, reference_report, model_name, variant_dir, protocol_path)
    model = adapter.build(source, num_classes=1, device=device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True),
                          strict=True)
    baseline_conf, baseline_nms = proxy["eval_conf"], proxy["nms_iou"]
    if baseline_nms is None:
        raise ValueError("reference policy already has NMS disabled")
    confidences = sorted({0.01, baseline_conf, 0.10})
    policies = []
    for confidence in confidences:
        for nms_iou in (baseline_nms, None):
            setting = dict(proxy, eval_conf=confidence, nms_iou=nms_iou)
            policy = {"confidence": confidence, "nms_iou": nms_iou,
                      "postprocessing": adapter.postprocessing(setting),
                      "metrics": score_policy(adapter, model, val_data, setting, device)}
            policies.append(policy)
    baseline = next(policy for policy in policies
                    if policy["confidence"] == baseline_conf
                    and policy["nms_iou"] == baseline_nms)
    reference_metrics = reference["metrics"]
    baseline_matches = all(
        abs(baseline["metrics"][key] - reference_metrics[key]) <= 1e-9
        for key in ("mAP50", "mAP50_95"))
    for policy in policies:
        policy["delta_vs_baseline"] = {
            key: (policy["metrics"][key] - baseline["metrics"][key]
                  if policy["metrics"][key] is not None and baseline["metrics"][key] is not None
                  else None)
            for key in ("mAP50", "mAP50_95", "mAP75", "mAP_small", "recall_small")}
    repeat = score_policy(adapter, model, val_data, proxy, device)
    return {
        "schema_version": "fred-dfine-postprocessing-comparison-v1",
        "gene_id": reference["gene_id"],
        "checkpoint_identity": reference["metadata"]["checkpoint_identity"],
        "model_identity": reference["metadata"]["model_identity"],
        "validation_sample_ids_sha256": sample_hash,
        "reference_report": str(Path(reference_report).resolve()),
        "runtime_environment": runtime_environment(device),
        "baseline_postprocessing": baseline["postprocessing"],
        "baseline_reference_matches": baseline_matches,
        "baseline_repeat_matches": baseline["metrics"] == repeat,
        "policies": policies,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--model", default="network")
    parser.add_argument("--variant-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = compare(args.checkpoint, args.reference_report, args.model,
                     args.variant_dir, args.protocol, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["baseline_reference_matches"] or not result["baseline_repeat_matches"]:
        raise SystemExit("baseline scoring did not reproduce; do not choose a policy")


if __name__ == "__main__":
    main()
