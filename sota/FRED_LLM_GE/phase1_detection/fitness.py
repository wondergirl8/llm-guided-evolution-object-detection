"""Detector-independent single-class COCO bbox evaluation.

Input: one record per evaluation frame, including frames with no detections.
Boxes are original-image pixel xyxy; labels must already be mapped to drone=0.
No confidence filtering, NMS (removes overlapping duplicate predictions;), resizing, annotation parsing or split construction
is performed here. Those choices belong to the frozen data/model adapters.
Input box: [left, top, right, bottom], class 0

fitness.py takes a candidate model’s predictions and the correct FRED annotations,
computes object-detection accuracy, and returns three values inside a
DetectionResult (mAp50, Map50_95, param_count)
"""

import contextlib
import io
import math
from collections import defaultdict
from numbers import Integral, Real

#  need to import DetectionResult, since this is where we will be stroing results
try:
    from .result_contract import DetectionResult
except ImportError:  # standalone module execution
    from result_contract import DetectionResult


# Parameter counting is performed by each model adapter or runner because model
# frameworks expose parameters differently. This common evaluator receives only
# the final integer through evaluate_frames(..., params=...).


# helper function to varify the output is the number
def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("coordinates and scores must be finite numbers")
    return float(value)


# define the boxes for either groud-truth params or predictions
# the sample inout:
# record = {
#     "boxes": [
#         [100, 200, 140, 240],  # First predicted drone
#         [500, 300, 560, 350],  # Second predicted drone
#     ],
#     "labels": [0, 0],         # 0 means drone, bc later COCO evaluator expects each box to have a category
#     "scores": [0.92, 0.78],   # Confidence for each prediction
# }

# _boxes(record, width=1280, height=720, prediction=True)
def _boxes(record, width, height, prediction):
    if not isinstance(record, dict):
        raise TypeError("box record must be a dictionary")
    boxes, labels = record.get("boxes"), record.get("labels")
    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise ValueError("boxes and labels must be lists")
    if len(boxes) != len(labels):
        raise ValueError("boxes and labels must have equal lengths")
    scores = record.get("scores") if prediction else [None] * len(boxes)
    if prediction and not isinstance(scores, list):
        raise ValueError("prediction scores must be a list")
    if len(scores) != len(boxes):
        raise ValueError("boxes and scores must have equal lengths")

    output = []
    for box, label, score in zip(boxes, labels, scores):
        if isinstance(label, bool) or not isinstance(label, Integral) or label != 0:
            raise ValueError("single-class evaluation requires integer drone label 0")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("each xyxy box must contain four coordinates")
        x1, y1, x2, y2 = map(_number, box)
        # [left, top, right, bottom], class 0
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("box must have positive area within original image bounds")

        # converting to COCO format
        item = {"category_id": 1, "bbox": [x1, y1, x2-x1, y2-y1],
                "area": (x2-x1)*(y2-y1), "iscrowd": 0}
        # if working with a prediction box append score, if ground-truth dont
        if prediction:
            score = _number(score)
            if not 0 <= score <= 1:
                raise ValueError("confidence score must be in [0, 1]")
            item["score"] = score
        output.append(item)
    return output


# frames evaluator
# * requires the remaining arguments to be passed by name:
# evaluate_frames(frames, params=123, expected_sample_ids=sample_ids)
# expected_sample_ids is the list of every frame that should be evaluated.
# expected_sample_ids = ["sequence_1:frame_0", "sequence_1:frame_2"]
def _defined_metric(value):
    """COCO uses -1 when an area or sequence has no ground truth."""
    return None if value < 0 else float(value)


def _bbox_iou(a, b):
    """IoU of two COCO xywh boxes, used only to describe unmatched targets."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    return overlap / (a[2] * a[3] + b[2] * b[3] - overlap)


def _xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _diagnostics(evaluator, images, annotations, gt, dt):
    """Use the same COCO inputs for sequence metrics and missed boxes."""
    from pycocotools.cocoeval import COCOeval

    overall_stats = tuple(evaluator.stats)
    image_ids = defaultdict(list)
    sequence_for_image = {}
    ground_truth_counts = defaultdict(int)
    for image in images:
        sequence = image["sample_id"].rsplit(":", 1)[0]
        image_ids[sequence].append(image["id"])
        sequence_for_image[image["id"]] = sequence
    for ann in annotations:
        ground_truth_counts[sequence_for_image[ann["image_id"]]] += 1

    misses, localization = [], []
    miss_counts = defaultdict(int)
    localization_counts = defaultdict(int)
    # One category, first COCO area range ('all'), maxDets=100.
    for entry in evaluator.evalImgs[:len(images)]:
        if entry is None:
            continue
        image_id = entry["image_id"]
        sequence = sequence_for_image[image_id]
        for index, target_id in enumerate(entry["gtIds"]):
            if entry["gtIgnore"][index]:
                continue
            target = gt.anns[target_id]
            match50 = entry["gtMatches"][0, index]
            if match50:
                if not entry["gtMatches"][5, index]:  # matched at .50, missed at .75
                    localization_counts[sequence] += 1
                    prediction = dt.anns[int(match50)]
                    localization.append({
                        "sample_id": images[image_id - 1]["sample_id"],
                        "target_xyxy": _xyxy(target["bbox"]),
                        "best_prediction_xyxy": _xyxy(prediction["bbox"]),
                        "best_score": prediction["score"],
                        "best_iou": _bbox_iou(target["bbox"], prediction["bbox"]),
                    })
                continue
            miss_counts[sequence] += 1
            best_id = max(entry["dtIds"],
                          key=lambda pred_id: _bbox_iou(target["bbox"], dt.anns[pred_id]["bbox"]),
                          default=None)
            best = dt.anns[best_id] if best_id is not None else None
            best_iou = _bbox_iou(target["bbox"], best["bbox"]) if best else 0.0
            misses.append({
                "sample_id": images[image_id - 1]["sample_id"],
                "target_xyxy": _xyxy(target["bbox"]),
                "best_prediction_xyxy": _xyxy(best["bbox"]) if best_iou else None,
                "best_score": best["score"] if best_iou else None,
                "best_iou": best_iou,
            })

    per_sequence = {}
    for sequence, ids in sorted(image_ids.items()):
        score = {"frames": len(ids), "ground_truth": ground_truth_counts[sequence],
                 "false_negatives_iou50": miss_counts[sequence],
                 "poorly_localized_iou50_to75": localization_counts[sequence],
                 "mAP50": None, "mAP75": None, "mAP50_95": None}
        if ground_truth_counts[sequence]:
            subset = COCOeval(gt, dt, "bbox")
            subset.params.imgIds = ids
            with contextlib.redirect_stdout(io.StringIO()):
                subset.evaluate()
                subset.accumulate()
                subset.summarize()
            score.update(mAP50=_defined_metric(subset.stats[1]),
                         mAP75=_defined_metric(subset.stats[2]),
                         mAP50_95=_defined_metric(subset.stats[0]))
        per_sequence[sequence] = score

    return {
        "definition": "COCO bbox, original pixels, maxDets=100; small area <32x32 px; "
                      "poorly localized = matched at IoU .50 but missed at .75",
        "overall": {"mAP75": _defined_metric(overall_stats[2]),
                    "mAP_small": _defined_metric(overall_stats[3]),
                    "recall_small": _defined_metric(overall_stats[9]),
                    "false_negatives_iou50": len(misses),
                    "poorly_localized_iou50_to75": len(localization)},
        "by_sequence": per_sequence,
        "examples": {
            "false_negatives": sorted(misses, key=lambda item: (item["best_iou"],
                                      item["sample_id"]))[:10],
            "poorly_localized": sorted(localization, key=lambda item: (
                -item["best_score"], item["sample_id"]))[:10],
        },
    }


def evaluate_frames(frames, *, params, expected_sample_ids, diagnostics=False):
    """Compute AP once across the complete declared evaluation partition.

    expected_sample_ids comes from the frozen manifest, not model predictions.
    This prevents silently dropping frames on which the detector found nothing.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    expected = list(expected_sample_ids)
    if not expected or any(not isinstance(x, str) or not x for x in expected):
        raise ValueError("expected_sample_ids must be nonempty strings")
    # we want to evaluate each frame once
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate manifest sample IDs")
    if isinstance(params, bool) or not isinstance(params, Integral) or params < 0:
        raise ValueError("params must be a nonnegative integer")

    expected_set = set(expected)
    images, annotations, predictions, seen = [], [], [], set()

    # Loop over frames and assigns temporary COCO image IDs starting at 1.
    for image_id, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict):
            raise TypeError("each frame must be a dictionary")
        sample_id = frame.get("sample_id")
        if sample_id not in expected_set or sample_id in seen:
            raise ValueError("unexpected or duplicate frame sample_id")
        seen.add(sample_id)
        width, height = frame.get("width"), frame.get("height")
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0
               for x in (width, height)):
            raise ValueError("width and height must be positive integers")
        # adding image to the coco dataset
        images.append({"id": image_id, "width": width, "height": height,
                       "sample_id": sample_id})

        # collect boxes and check completeness
        for ann in _boxes(frame.get("target"), width, height, False):
            annotations.append(dict(ann, id=len(annotations)+1, image_id=image_id))
        for pred in _boxes(frame.get("prediction"), width, height, True):
            predictions.append(dict(pred, id=len(predictions)+1, image_id=image_id))

    if seen != expected_set:
        missing = sorted(expected_set - seen)
        raise ValueError(f"missing evaluation frames: {missing}")
    if not annotations:
        raise ValueError("AP is undefined for a partition without ground-truth drones")

    # Construct both COCO objects directly: loadRes([]) fails on empty predictions.
    with contextlib.redirect_stdout(io.StringIO()):
        # ground-truth (gt) and detection (dt)
        gt, dt = COCO(), COCO()
        # define shared images and the single drone category
        common = {"images": images, "categories": [{"id": 1, "name": "drone"}]}
        gt.dataset = dict(common, annotations=annotations)
        dt.dataset = dict(common, annotations=predictions)
        gt.createIndex()
        dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        # Returns mAP50 from stats[1], mAP50:95 from stats[0], and the supplied parameter count.
    result = DetectionResult(
        map50=float(evaluator.stats[1]),
        map50_95=float(evaluator.stats[0]),
        params=int(params),
    )
    if diagnostics:
        details = _diagnostics(evaluator, images, annotations, gt, dt)
        return result, details
    return result
