"""Measure whether selected D-FINE query locations cover validation drones."""

import argparse
import json
from pathlib import Path

import torch

try:
    from .candidate_runner import DEFAULT_PROTOCOL
    from .rescore_dfine import prepare_checkpoint
except ImportError:
    from candidate_runner import DEFAULT_PROTOCOL
    from rescore_dfine import prepare_checkpoint


GROUPS = ("all", "small", "missed", "missed_small")


def _best_iou(box, predictions):
    if len(predictions) == 0:
        return 0.0
    low = torch.maximum(box[:2], predictions[:, :2])
    high = torch.minimum(box[2:], predictions[:, 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    box_area = (box[2:] - box[:2]).prod()
    pred_area = (predictions[:, 2:] - predictions[:, :2]).prod(-1)
    return float((intersection / (box_area + pred_area - intersection).clamp_min(1e-8)).max())


def frame_coverage(coordinates, selected_indices, boxes, original_boxes, predictions,
                   small_area=32 * 32):
    """Count GT boxes containing any token center or selected query center.

    Coverage means a query's original spatial center lies inside the GT box;
    it measures initial query selection, not whether the predicted box is accurate.
    """
    if len(boxes) != len(original_boxes):
        raise ValueError("resized and original GT boxes must correspond")
    selected = coordinates[selected_indices]
    counts = {group: {"objects": 0, "available": 0, "selected": 0}
              for group in GROUPS}
    examples = []
    for box, original in zip(boxes, original_boxes):
        small = float((original[2:] - original[:2]).prod()) < small_area
        missed = _best_iou(box, predictions) < 0.5
        available = bool(((coordinates >= box[:2]) & (coordinates <= box[2:])).all(-1).any())
        covered = bool(((selected >= box[:2]) & (selected <= box[2:])).all(-1).any())
        categories = ("all",) + (("small",) if small else ()) + (
            ("missed",) if missed else ()) + (("missed_small",) if missed and small else ())
        for category in categories:
            counts[category]["objects"] += 1
            counts[category]["available"] += int(available)
            counts[category]["selected"] += int(covered)
        if missed and small:
            examples.append({"box": [float(value) for value in original],
                             "available": available, "selected": covered})
    return counts, examples


def analyze(checkpoint, reference_report, model_name="network", variant_dir=None,
            protocol_path=DEFAULT_PROTOCOL, device="cpu"):
    checkpoint, reference, adapter, source, proxy, data, sample_hash = prepare_checkpoint(
        checkpoint, reference_report, model_name, variant_dir, protocol_path)
    model = adapter.build(source, num_classes=1, device=device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True),
                          strict=True)
    model.eval()
    totals = {group: {"objects": 0, "available": 0, "selected": 0}
              for group in GROUPS}
    level_counts = [0, 0, 0]
    examples = []
    with torch.no_grad():
        for index, sample_id in enumerate(data["sample_ids"]):
            raw = data["images"][index:index + 1]
            height, width = raw.shape[-2:]
            predictions, _ = adapter.predict(model, raw, proxy, device)[0]
            event = raw.to(device=device, dtype=torch.float32).div_(255)
            features = model.encoder(model.backbone(event))
            tokens, coordinates = model.decoder._tokens(features)
            scores = model.decoder.objectness(tokens).squeeze(-1)
            selected = scores.topk(model.num_queries, dim=1).indices[0].cpu()
            coordinates = coordinates[0].cpu() * torch.tensor([width, height])
            first = features[0].shape[-2] * features[0].shape[-1]
            second = first + features[1].shape[-2] * features[1].shape[-1]
            level_counts[0] += int((selected < first).sum())
            level_counts[1] += int(((selected >= first) & (selected < second)).sum())
            level_counts[2] += int((selected >= second).sum())
            frame_counts, frame_examples = frame_coverage(
                coordinates, selected, data["boxes"][index], data["orig_boxes"][index],
                predictions)
            for group in GROUPS:
                for name in totals[group]:
                    totals[group][name] += frame_counts[group][name]
            for example in frame_examples:
                if len(examples) < 20:
                    examples.append({"sample_id": sample_id, **example})
    coverage = {group: {**counts,
                        "available_fraction": (counts["available"] / counts["objects"]
                                               if counts["objects"] else None),
                        "selected_fraction": (counts["selected"] / counts["objects"]
                                              if counts["objects"] else None)}
                for group, counts in totals.items()}
    return {
        "reference_report": str(Path(reference_report).resolve()),
        "checkpoint_identity": reference["metadata"]["checkpoint_identity"],
        "model_identity": reference["metadata"]["model_identity"],
        "postprocessing": reference["metadata"].get("postprocessing"),
        "validation_sample_ids_sha256": sample_hash,
        "device": str(device),
        "num_frames": len(data["sample_ids"]), "num_queries": model.num_queries,
        "selected_query_levels": dict(zip(("stride4", "stride8", "stride16"), level_counts)),
        "coverage": coverage, "missed_small_examples": examples,
        "definition": ("small: original box area < 1024 px2; missed: no postprocessed "
                       "prediction IoU >= 0.5; available/selected: token/query spatial "
                       "center inside the resized GT box"),
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
        parser.error(f"output already exists: {args.output}")
    result = analyze(args.checkpoint, args.reference_report, args.model,
                     args.variant_dir, args.protocol, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
