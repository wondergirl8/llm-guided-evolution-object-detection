import pytest

pytest.importorskip("pycocotools")

from sota.FRED_LLM_GE.phase1_detection.fitness import evaluate_frames


def _frame(prediction):
    return {
        "sample_id": "0:0",
        "width": 100,
        "height": 100,
        "target": {"boxes": [[10, 10, 30, 30]], "labels": [0]},
        "prediction": prediction,
    }


def test_perfect_prediction_has_perfect_ap():
    result = evaluate_frames(
        [_frame({"boxes": [[10, 10, 30, 30]], "labels": [0], "scores": [0.9]})],
        params=321,
        expected_sample_ids=["0:0"],
    )
    assert result.map50 == pytest.approx(1.0)
    assert result.map50_95 == pytest.approx(1.0)
    assert result.params == 321


def test_empty_predictions_score_zero():
    result = evaluate_frames(
        [_frame({"boxes": [], "labels": [], "scores": []})],
        params=321,
        expected_sample_ids=["0:0"],
    )
    assert result.map50 == pytest.approx(0.0)
    assert result.map50_95 == pytest.approx(0.0)


def test_missing_manifest_frame_is_rejected():
    with pytest.raises(ValueError, match="missing evaluation frames"):
        evaluate_frames(
            [_frame({"boxes": [], "labels": [], "scores": []})],
            params=321,
            expected_sample_ids=["0:0", "0:1"],
        )


def test_invalid_parameter_count_is_rejected():
    with pytest.raises(ValueError, match="params"):
        evaluate_frames(
            [_frame({"boxes": [], "labels": [], "scores": []})],
            params=-1,
            expected_sample_ids=["0:0"],
        )
