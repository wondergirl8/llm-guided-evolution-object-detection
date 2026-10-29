"""Convert rescaled MMDetection predictions into the FRED detection contract."""
import math
from numbers import Integral, Real


def _list(value):
    if value is None:
        raise ValueError("Missing MMDetection prediction field")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist()


def faster_rcnn_prediction(result):
    instances = getattr(result, "pred_instances", None)
    if instances is None:
        raise ValueError("Missing pred_instances")
    boxes = _list(getattr(instances, "bboxes", None))
    labels = _list(getattr(instances, "labels", None))
    scores = _list(getattr(instances, "scores", None))
    if not len(boxes) == len(labels) == len(scores):
        raise ValueError("Prediction field lengths differ")
    for box, label, score in zip(boxes, labels, scores):
        if len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, Real)
                               or not math.isfinite(x) for x in box):
            raise ValueError("Expected finite xyxy box")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Degenerate box")
        if isinstance(label, bool) or not isinstance(label, Integral) or label != 0:
            raise ValueError("Expected drone label 0")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid confidence score")
    return {"boxes": boxes, "labels": labels, "scores": scores}


def faster_rcnn_frame(result, *, sample_id, target):
    height, width = result.metainfo["ori_shape"][:2]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Expected nonempty sample identity")
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in (width, height)):
        raise ValueError("Expected positive original dimensions")
    if not isinstance(target, dict) or not {"boxes", "labels"} <= target.keys():
        raise ValueError("Expected trusted target")
    prediction = faster_rcnn_prediction(result)
    for x1, y1, x2, y2 in prediction["boxes"]:
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Prediction is outside original image")
    return {"sample_id": sample_id, "width": int(width), "height": int(height),
            "target": target, "prediction": prediction}
