# Detection Model Output Standard

## Purpose

YOLO11, RT-DETR, Faster R-CNN, and future evolved detectors must be evaluated by
the same `evaluate_frames()` function. Each model adapter therefore converts its
framework-specific output into the format defined here. Adapters translate data;
they do not calculate mAP, choose the evaluation split, or load ground truth.

## Coordinate and class rules

- Boxes use original-image pixel coordinates in `xyxy` order:
  `[x_left, y_top, x_right, y_bottom]`.
- A valid box satisfies `0 <= x_left < x_right <= width` and
  `0 <= y_top < y_bottom <= height`.
- Primary Phase 1 detection is single-class. Integer label `0` means `drone`.
- Each predicted box has one confidence score in the inclusive range `[0, 1]`.
- Resize, padding, and letterbox transforms must be reversed before evaluation.
- Confidence filtering and NMS happen before conversion and must use the frozen
  evaluation configuration.

## Prediction record

Every model adapter must return this structure for one image:

```python
{
    "boxes": [
        [100.0, 200.0, 140.0, 240.0],
        [500.0, 300.0, 560.0, 350.0],
    ],
    "labels": [0, 0],
    "scores": [0.92, 0.78],
}
```

The three lists have equal length and corresponding positions describe the same
detection. A model that detects nothing must return all three lists as empty:

```python
{"boxes": [], "labels": [], "scores": []}
```

## Target record

Ground truth comes from the trusted FRED data interface, never from a detector or
prediction adapter:

```python
{
    "boxes": [[102.0, 199.0, 141.0, 242.0]],
    "labels": [0],
}
```

Targets do not have confidence scores.

## Complete evaluation frame

The training/evaluation harness joins one converted prediction with its trusted
target and manifest identity:

```python
{
    "sample_id": "101:45",
    "width": 1280,
    "height": 720,
    "target": {
        "boxes": [[102.0, 199.0, 141.0, 242.0]],
        "labels": [0],
    },
    "prediction": {
        "boxes": [[100.0, 200.0, 140.0, 240.0]],
        "labels": [0],
        "scores": [0.92],
    },
}
```

There must be exactly one frame record for every ID in the frozen evaluation
manifest, including images with no predictions and images with no ground-truth
drones. `expected_sample_ids` must be built from that manifest independently of
model output.

## Adapter boundary

Each model family owns only its conversion step:

```text
framework output -> family adapter -> prediction record
                                      + trusted FRED target
                                      -> complete frame
                                      -> evaluate_frames()
```

The adapter must fail on missing fields, malformed shapes, non-finite values,
unsupported labels, or output that cannot be mapped back to the original image.
It must not silently clip, invent, or discard correctness-critical values.

## YOLO11 usage

Ultralytics produces one `Results` object per image. Convert each result with:

```python
from sota.FRED_LLM_GE.phase1_detection.adapters.yolo11 import yolo11_prediction

prediction = yolo11_prediction(result)
```

Then the harness creates the complete frame using the corresponding FRED target.
Pass the accumulated frames to the shared reporter. Model parameter count is
supplied separately; for the provided Ultralytics wrapper, the actual module is
`trained.model`.
