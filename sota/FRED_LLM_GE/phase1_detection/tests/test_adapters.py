import math
import pytest

from sota.FRED_LLM_GE.phase1_detection.adapters.yolo11 import (
    yolo11_frame,
    yolo11_prediction,
)


class TensorLike:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class Boxes:
    def __init__(self):
        self.xyxy = TensorLike([[10.0, 20.0, 30.0, 40.0]])
        self.cls = TensorLike([0.0])
        self.conf = TensorLike([0.9])


class Result:
    def __init__(self):
        self.boxes = Boxes()
        self.orig_shape = (720, 1280)


def test_converts_yolo11_result():
    assert yolo11_prediction(Result()) == {
        "boxes": [[10.0, 20.0, 30.0, 40.0]],
        "labels": [0],
        "scores": [0.9],
    }


def test_builds_common_frame_with_trusted_target():
    target = {"boxes": [[11.0, 21.0, 31.0, 41.0]], "labels": [0]}
    frame = yolo11_frame(Result(), sample_id="101:45", target=target)
    assert frame["width"] == 1280
    assert frame["height"] == 720
    assert frame["target"] is target


def test_empty_detection_is_preserved():
    result = Result()
    result.boxes = None
    assert yolo11_prediction(result) == {"boxes": [], "labels": [], "scores": []}


@pytest.mark.parametrize("label", [1.0, 0.5])
def test_rejects_unsupported_labels(label):
    result = Result()
    result.boxes = Boxes()
    result.boxes.cls = TensorLike([label])
    with pytest.raises(ValueError):
        yolo11_prediction(result)


def test_rejects_nonfinite_scores():
    result = Result()
    result.boxes = Boxes()
    result.boxes.conf = TensorLike([math.nan])
    with pytest.raises(ValueError):
        yolo11_prediction(result)
