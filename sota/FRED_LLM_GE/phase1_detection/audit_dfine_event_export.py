"""Audit D-FINE search-validation Event frames, boxes, and prediction overlays."""

import argparse
import hashlib
import json
import math
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw

try:
    from .candidate_runner import (DEFAULT_PROTOCOL, FRED_DIR, predict_frames,
                                   target_boxes_sha256, load_protocol)
    from .fitness import evaluate_frames
    from .proxy_data import STEP_US
    from .rescore_dfine import prepare_checkpoint
    from ..phase0_data.coco_adapter import parse_fred_annotation
    from ..data.fred_stream import _select_frames
except ImportError:
    from candidate_runner import (DEFAULT_PROTOCOL, FRED_DIR, predict_frames,
                                  target_boxes_sha256, load_protocol)
    from fitness import evaluate_frames
    from proxy_data import STEP_US
    from rescore_dfine import prepare_checkpoint
    from coco_adapter import parse_fred_annotation
    from fred_stream import _select_frames


def _source_boxes(zip_path, sequence, width, height):
    """Reproduce the export's current clipping policy from an original local ZIP."""
    with zipfile.ZipFile(zip_path) as archive:
        lines = archive.read(f"{sequence}/coordinates.txt").decode().splitlines()
    boxes, counts = {}, {"annotations": 0, "clipped": 0, "dropped": 0,
                         "malformed": 0}
    for line in lines:
        if not line.strip():
            continue
        annotation = parse_fred_annotation(line)
        if annotation is None:
            counts["malformed"] += 1
            continue
        counts["annotations"] += 1
        raw = [annotation[key] for key in ("x1", "y1", "x2", "y2")]
        if not all(math.isfinite(value) for value in [annotation["time"], *raw]):
            counts["malformed"] += 1
            continue
        clipped = [max(0.0, raw[0]), max(0.0, raw[1]),
                   min(float(width), raw[2]), min(float(height), raw[3])]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            counts["dropped"] += 1
            continue
        if clipped != raw:
            counts["clipped"] += 1
        frame_index = round(annotation["time"] * 1_000_000 / STEP_US)
        boxes.setdefault(frame_index, []).append(clipped)
    return boxes, counts


def _label_boxes(path, width, height):
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"invalid single-class YOLO label: {path}")
        cx, cy, w, h = map(float, parts[1:])
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (cx, cy, w, h)):
            raise ValueError(f"invalid normalized label coordinates: {path}")
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes


def _boxes_match(first, second, tolerance=0.002):
    return (len(first) == len(second) and all(
        abs(a - b) <= tolerance for x, y in zip(first, second)
        for a, b in zip(x, y)))


def audit_export(root, val_data, report, source_root=None, expected_step=None):
    """Check each exported validation frame without changing the dataset."""
    root = Path(root)
    source_root = Path(source_root) if source_root else None
    expected_ids = val_data["sample_ids"]
    expected_paths = val_data["images"].paths
    width, height = val_data["orig_size"]
    marker_entries, issues = {}, []
    source_counts = {"annotations": 0, "clipped": 0, "dropped": 0,
                     "malformed": 0}
    source_checked = 0
    source_pixel_samples_checked = 0
    sequence_ids = report["metadata"]["split_identity"].partition(";val=")[2].split(",")
    for sequence in sequence_ids:
        marker = json.loads((root / "done" / f"{sequence}.json").read_text())
        folder = marker["hf_folder"]
        if (str(marker["sequence"]) != sequence or tuple(marker["size"]) != (width, height)
                or marker["frames"] != len(marker["records"])):
            issues.append(f"sequence {sequence}: marker identity, size, or frame count differs")
        if expected_step is not None and marker["frame_step"] != expected_step:
            issues.append(f"sequence {sequence}: frame step differs from protocol")
        if not isinstance(marker["frame_step"], int) or marker["frame_step"] < 1:
            raise ValueError(f"sequence {sequence}: invalid frame step")
        source = None
        if source_root is not None:
            archive = source_root / folder / f"{sequence}.zip"
            if not archive.is_file():
                issues.append(f"sequence {sequence}: original archive missing: {archive}")
            else:
                source, counts = _source_boxes(archive, sequence, width, height)
                source_checked += 1
                for key, value in counts.items():
                    source_counts[key] += value
                with zipfile.ZipFile(archive) as original:
                    source_frames, _ = _select_frames(int(sequence), original.namelist())
                    selected = source_frames[::marker["frame_step"]]
                    indices = [counter // STEP_US for counter, _ in selected]
                    if indices != [item["frame_index"] for item in marker["records"]]:
                        issues.append(f"sequence {sequence}: Event frame indices differ from source ZIP")
                    if selected:
                        _, member = selected[0]
                        exported = root / "images" / f"{marker['records'][0]['name']}.png"
                        with Image.open(BytesIO(original.read(member))) as source_image:
                            with Image.open(exported) as exported_image:
                                if (source_image.convert("RGB").tobytes() !=
                                        exported_image.convert("RGB").tobytes()):
                                    issues.append(f"sequence {sequence}: first Event PNG pixels differ")
                        source_pixel_samples_checked += 1
        for record in marker["records"]:
            frame_index = record["frame_index"]
            name = f"{sequence}_{frame_index:06d}"
            sample_id = f"{folder}/{sequence}:{frame_index}"
            if record["name"] != name or sample_id in marker_entries:
                issues.append(f"{sample_id}: name or sample identity differs")
            marker_entries[sample_id] = (root / "images" / f"{name}.png", record["boxes"])
            if source is not None and not _boxes_match(
                    record["boxes"], source.get(frame_index, []), tolerance=1e-4):
                issues.append(f"{sample_id}: exported boxes differ from original annotations")

    if list(marker_entries) != expected_ids:
        issues.append("marker sample order/identities differ from the evaluation loader")
    if len(expected_ids) != report["num_frames"]:
        issues.append("export frame count differs from evaluation report")
    sample_hash = hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
    if sample_hash != report["metadata"].get("validation_sample_ids_sha256"):
        issues.append("validation sample ID hash differs from evaluation report")
    target_hash = target_boxes_sha256(val_data)
    recorded_target_hash = report["metadata"].get("validation_targets_sha256")
    if recorded_target_hash is not None and target_hash != recorded_target_hash:
        issues.append("validation target hash differs from evaluation report")

    small, empty, annotated = [], [], []
    for sample_id, path, target in zip(expected_ids, expected_paths, val_data["orig_boxes"]):
        if sample_id not in marker_entries:
            continue
        marker_path, marker_boxes = marker_entries[sample_id]
        boxes = target.tolist()
        if path != marker_path or not _boxes_match(boxes, marker_boxes, tolerance=1e-4):
            issues.append(f"{sample_id}: loader path or boxes differ from marker")
        label_path = root / "labels" / f"{marker_path.stem}.txt"
        if not label_path.is_file() or not _boxes_match(
                boxes, _label_boxes(label_path, width, height)):
            issues.append(f"{sample_id}: YOLO labels differ from pixel boxes")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (width, height):
                issues.append(f"{sample_id}: Event PNG mode or dimensions differ")
        for box in boxes:
            if (not all(math.isfinite(x) for x in box) or not
                    (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                issues.append(f"{sample_id}: invalid pixel target box")
        if not boxes:
            empty.append(sample_id)
        else:
            annotated.append(sample_id)
            if any((box[2] - box[0]) * (box[3] - box[1]) < 1024 for box in boxes):
                small.append(sample_id)
    return {
        "sample_ids_sha256": sample_hash, "target_boxes_sha256": target_hash,
        "recorded_target_hash_available": recorded_target_hash is not None,
        "frames": len(expected_ids), "small_object_frames": len(small),
        "empty_frames": len(empty), "small_sample_ids": small,
        "empty_sample_ids": empty, "annotated_sample_ids": annotated,
        "source_sequences_checked": source_checked,
        "source_pixel_samples_checked": source_pixel_samples_checked,
        "source_sequences_expected": len(sequence_ids), "source_annotations": source_counts,
        "issues": issues,
    }


def select_examples(audit, diagnostics, limit):
    """Select a few small, empty, missed, and poorly localized frames."""
    examples = diagnostics.get("examples", {})
    high_miss = sorted(diagnostics.get("by_sequence", {}),
                       key=lambda sequence: diagnostics["by_sequence"][sequence]
                       ["false_negatives_iou50"], reverse=True)
    groups = {
        "high_miss_sequence": [next((sample_id for sample_id in audit["annotated_sample_ids"]
                                     if sample_id.startswith(sequence + ":")), "")
                               for sequence in high_miss],
        "small": audit["small_sample_ids"], "empty": audit["empty_sample_ids"],
        "miss": [item["sample_id"] for item in examples.get("false_negatives", [])],
        "localization": [item["sample_id"] for item in examples.get("poorly_localized", [])],
    }
    selected = []
    for category, sample_ids in groups.items():
        count, seen = 0, set()
        for sample_id in sample_ids:
            if sample_id and sample_id not in seen:
                selected.append((category, sample_id))
                seen.add(sample_id)
                count += 1
            if count == limit:
                break
    return selected


def render_overlays(selected, frames, val_data, output_dir):
    by_id = {frame["sample_id"]: (frame, path) for frame, path in
             zip(frames, val_data["images"].paths)}
    manifest = []
    for category, sample_id in selected:
        frame, path = by_id[sample_id]
        with Image.open(path) as source:
            image = source.copy()
        draw = ImageDraw.Draw(image)
        for box in frame["prediction"]["boxes"]:
            draw.rectangle(box, outline="orange", width=2)
        for box in frame["target"]["boxes"]:
            draw.rectangle(box, outline="lime", width=2)
        name = f"{category}_{path.stem}.png"
        image.save(output_dir / name)
        manifest.append({"category": category, "sample_id": sample_id,
                         "image": name, "green": "target", "orange": "prediction"})
    return manifest


def run(checkpoint, reference_report, output_dir, source_root=None,
        model_name="network", variant_dir=None, protocol_path=DEFAULT_PROTOCOL,
        device="cpu", examples_per_category=2):
    if examples_per_category < 1:
        raise ValueError("examples_per_category must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    _, report, adapter, source, proxy, val_data, _ = prepare_checkpoint(
        checkpoint, reference_report, model_name, variant_dir, protocol_path)
    challenging = load_protocol(protocol_path)["challenging"]
    root = FRED_DIR / challenging["data_root"] / f"search_val_step{challenging['search']['val_frame_step']}"
    audit = audit_export(root, val_data, report, source_root, proxy["val_frame_step"])
    model = adapter.build(source, num_classes=1, device=device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True),
                          strict=True)
    frames = predict_frames(adapter, model, val_data, proxy, device)
    score, diagnostics = evaluate_frames(
        frames, params=adapter.param_count(model),
        expected_sample_ids=val_data["sample_ids"], diagnostics=True)
    score_matches_reference = all(
        abs(value - report["metrics"][name]) <= 1e-9
        for name, value in (("mAP50", score.map50), ("mAP50_95", score.map50_95)))
    if not score_matches_reference:
        audit["issues"].append("saved checkpoint score differs from reference report")
    for frame, sample_id, boxes in zip(frames, val_data["sample_ids"], val_data["orig_boxes"]):
        if (frame["sample_id"] != sample_id or frame["width"] != val_data["orig_size"][0]
                or frame["height"] != val_data["orig_size"][1]
                or not _boxes_match(frame["target"]["boxes"], boxes.tolist(), 1e-4)):
            audit["issues"].append(f"{sample_id}: evaluator frame or targets differ from export")
        for box in frame["prediction"]["boxes"]:
            if (not all(math.isfinite(x) for x in box) or not
                    (0 <= box[0] < box[2] <= frame["width"]
                     and 0 <= box[1] < box[3] <= frame["height"])):
                audit["issues"].append(f"{sample_id}: invalid original-pixel prediction")
    selected = select_examples(audit, diagnostics, examples_per_category)
    output_dir.mkdir(parents=True)
    overlays = render_overlays(selected, frames, val_data, output_dir)
    result = {key: value for key, value in audit.items()
              if key not in ("small_sample_ids", "empty_sample_ids",
                             "annotated_sample_ids")}
    counts = audit["source_annotations"]
    policy_review = any(counts[key] for key in ("clipped", "dropped", "malformed"))
    source_verified = (audit["source_sequences_checked"] ==
                       audit["source_sequences_expected"] and
                       audit["source_pixel_samples_checked"] ==
                       audit["source_sequences_expected"])
    result.update({
        "overlays": overlays,
        "score_matches_reference": score_matches_reference,
        "diagnostic_misses": diagnostics["overall"]["false_negatives_iou50"],
        "diagnostic_localization_errors": diagnostics["overall"][
            "poorly_localized_iou50_to75"],
        "source_policy_review_required": policy_review,
        "audit_complete": (not audit["issues"] and
                           audit["recorded_target_hash_available"] and
                           source_verified and not policy_review),
    })
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--source-zips", type=Path,
                        help="local <folder>/<sequence>.zip archives with coordinates.txt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="network")
    parser.add_argument("--variant-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-category", type=int, default=2)
    args = parser.parse_args()
    result = run(args.checkpoint, args.reference_report, args.output_dir,
                 args.source_zips, args.model, args.variant_dir, args.protocol,
                 args.device, args.examples_per_category)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
