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
from numbers import Integral, Real

#  need to import DetectionResult, since this is where we will be stroing results
try:
    from .result_contract import DetectionResult
except ImportError:  # standalone module execution
    from result_contract import DetectionResult


# NEEDS MODIFICATIONS
# currently do not understand where to get the params count for the model
# bc dont indestand where is it will be living and how represented
#  assuming this is the structure params = count_parameters([
#     model.layer1_weights,
#     model.layer1_bias,
#     model.layer2_weights,
#     model.layer2_bias,
# ])
def count_parameters(parameter_arrays):
    total = 0
    seen = set()

    for array in parameter_arrays:
        # Count a shared array only once
        if id(array) in seen:
            continue
        seen.add(id(array))

        total += int(array.size)

    return total


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
    boxes, labels = record["boxes"], record["labels"]
    if len(boxes) != len(labels):
        raise ValueError("boxes and labels must have equal lengths")
    scores = record["scores"] if prediction else [None] * len(boxes)
    if len(scores) != len(boxes):
        raise ValueError("boxes and scores must have equal lengths")

    output = []
    for box, label, score in zip(boxes, labels, scores):
    if isinstance(label, bool) or not isinstance(label, Integral) or label != 0:
        raise ValueError("single-class evaluation requires integer drone label 0")
    if len(box) != 4:
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
def evaluate_frames(frames, *, params, expected_sample_ids):
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
    
    expected_set = set(expected)
    images, annotations, predictions, seen = [], [], [], set()


    # Loop over frames and assigns temporary COCO image IDs starting at 1.
    for image_id, frame in enumerate(frames, start=1):
    sample_id = frame["sample_id"]
    if sample_id not in expected_set or sample_id in seen:
        raise ValueError("unexpected or duplicate frame sample_id")
    seen.add(sample_id)
    width, height = frame["width"], frame["height"]
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0
           for x in (width, height)):
        raise ValueError("width and height must be positive integers")
    # adding image to the coco dataset
    images.append({"id": image_id, "width": width, "height": height})


    # collect boxes and check completeness
    for ann in _boxes(frame["target"], width, height, False):
    annotations.append(dict(ann, id=len(annotations)+1, image_id=image_id))
    for pred in _boxes(frame["prediction"], width, height, True):
        predictions.append(dict(pred, id=len(predictions)+1, image_id=image_id))
    if seen != expected_set:
        raise ValueError("missing evaluation frames")
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
    return DetectionResult(float(evaluator.stats[1]), float(evaluator.stats[0]), params)