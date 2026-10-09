"""Convert Ultralytics YOLO11 results to the shared detection output contract."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


def _tolist(value: Any, field: str) -> list:
    """Move a tensor-like value to CPU and convert it to ordinary Python lists."""
    if value is None:
        raise ValueError(f"YOLO11 result is missing {field}")
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if not callable(getattr(value, "tolist", None)):
        raise TypeError(f"YOLO11 {field} must be tensor-like and provide tolist()")
    return value.tolist()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"YOLO11 {field} values must be finite numbers")
    return float(value)


def yolo11_prediction(result: Any) -> dict[str, list]:
    """Return one model-neutral prediction record from an Ultralytics Result.

    Ultralytics `boxes.xyxy` is already expressed in original-image pixels after
    prediction postprocessing. The caller remains responsible for using the
    frozen confidence and NMS configuration when inference is run.
    """
    boxes_object = getattr(result, "boxes", None)
    if boxes_object is None:
        return {"boxes": [], "labels": [], "scores": []}

    boxes = _tolist(getattr(boxes_object, "xyxy", None), "boxes.xyxy")
    labels = _tolist(getattr(boxes_object, "cls", None), "boxes.cls")
    scores = _tolist(getattr(boxes_object, "conf", None), "boxes.conf")
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError("YOLO11 boxes, labels, and scores must have equal lengths")

    converted_boxes = []
    converted_labels = []
    converted_scores = []
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"YOLO11 box {index} must contain four xyxy coordinates")
        converted_boxes.append([
            _finite_number(coordinate, f"box {index}") for coordinate in box
        ])
        numeric_label = _finite_number(label, f"label {index}")
        if not numeric_label.is_integer():
            raise ValueError(f"YOLO11 label {index} must be an integer")
        integer_label = int(numeric_label)
        if integer_label != 0:
            raise ValueError("FRED Phase 1 requires every YOLO11 label to be drone class 0")
        converted_labels.append(integer_label)
        numeric_score = _finite_number(score, f"score {index}")
        if not 0 <= numeric_score <= 1:
            raise ValueError(f"YOLO11 score {index} must be in [0, 1]")
        converted_scores.append(numeric_score)

    return {
        "boxes": converted_boxes,
        "labels": converted_labels,
        "scores": converted_scores,
    }


def yolo11_frame(result: Any, *, sample_id: str, target: dict) -> dict:
    """Join a converted prediction to caller-supplied trusted FRED ground truth."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty manifest identity")
    shape = getattr(result, "orig_shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("YOLO11 result.orig_shape must be (height, width)")
    height, width = shape
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, Integral) or not isinstance(height, Integral)
            or width <= 0 or height <= 0):
        raise ValueError("YOLO11 original width and height must be positive integers")
    if not isinstance(target, dict) or "boxes" not in target or "labels" not in target:
        raise ValueError("target must be a trusted record containing boxes and labels")
    return {
        "sample_id": sample_id,
        "width": int(width),
        "height": int(height),
        "target": target,
        "prediction": yolo11_prediction(result),
    }
