import json

import pytest

pytest.importorskip("pycocotools")

from sota.FRED_LLM_GE.phase1_detection.fitness import evaluate_frames
from sota.FRED_LLM_GE.phase1_detection import map_reporter


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


def test_diagnostics_reuse_coco_matches_and_split_by_sequence():
    frames = [
        {**_frame({"boxes": [[10, 10, 20, 20]], "labels": [0], "scores": [0.9]}),
         "sample_id": "train/1:0", "target": {"boxes": [[10, 10, 20, 20]], "labels": [0]}},
        {**_frame({"boxes": [], "labels": [], "scores": []}),
         "sample_id": "train/2:0"},
        {**_frame({"boxes": [[60, 50, 80, 70]], "labels": [0], "scores": [0.8]}),
         "sample_id": "train/2:1", "target": {"boxes": [[50, 50, 70, 70]], "labels": [0]}},
        {**_frame({"boxes": [], "labels": [], "scores": []}),
         "sample_id": "train/3:0", "target": {"boxes": [], "labels": []}},
    ]
    result, details = evaluate_frames(
        frames, params=321, expected_sample_ids=[f["sample_id"] for f in frames],
        diagnostics=True,
    )

    assert result.map50_95 > 0
    assert details["overall"]["mAP75"] == pytest.approx(result.map50_95)
    assert details["overall"]["mAP_small"] == pytest.approx(result.map50_95)
    assert details["overall"]["recall_small"] > 0
    assert details["overall"]["false_negatives_iou50"] == 2
    assert details["overall"]["poorly_localized_iou50_to75"] == 0
    assert details["by_sequence"]["train/1"]["mAP50_95"] == pytest.approx(1.0)
    assert details["by_sequence"]["train/2"]["mAP50_95"] == pytest.approx(0.0)
    assert details["by_sequence"]["train/3"]["mAP50_95"] is None
    assert {item["sample_id"] for item in details["examples"]["false_negatives"]} == {
        "train/2:0", "train/2:1",
    }
    assert details["examples"]["poorly_localized"] == []


def test_poor_localization_is_an_ap75_miss_after_ap50_match():
    frame = _frame({"boxes": [[10, 10, 26, 26]], "labels": [0], "scores": [0.9]})
    result, details = evaluate_frames([frame], params=321,
                                      expected_sample_ids=["0:0"], diagnostics=True)
    assert result.map50 == pytest.approx(1.0)
    assert details["overall"]["mAP75"] == pytest.approx(0.0)
    assert details["overall"]["false_negatives_iou50"] == 0
    assert details["overall"]["poorly_localized_iou50_to75"] == 1
    assert details["by_sequence"]["0"]["poorly_localized_iou50_to75"] == 1
    assert details["examples"]["false_negatives"] == []
    assert details["examples"]["poorly_localized"][0]["best_iou"] == pytest.approx(0.64)


def test_diagnostics_marks_missing_small_objects_undefined():
    frame = _frame({"boxes": [[10, 10, 60, 60]], "labels": [0], "scores": [0.9]})
    frame["target"]["boxes"] = [[10, 10, 60, 60]]
    _, details = evaluate_frames([frame], params=321,
                                 expected_sample_ids=["0:0"], diagnostics=True)
    assert details["overall"]["mAP_small"] is None
    assert details["overall"]["recall_small"] is None


def test_optional_report_adds_diagnostics_without_changing_fitness_csv(tmp_path):
    metadata = {key: "test" for key in (
        "dataset_revision", "manifest_identity", "split_identity", "annotation_policy",
        "model_identity", "checkpoint_identity", "code_revision", "postprocessing",
        "training_config_identity",
    )}
    metadata.update(modality="event", benchmark_split="challenging",
                    project_split="validation", official_split="challenging_train")
    payload = {"metadata": metadata,
               "frames": [_frame({"boxes": [[10, 10, 30, 30]],
                                   "labels": [0], "scores": [0.9]})],
               "params": 321, "expected_sample_ids": ["0:0"]}

    result, output = map_reporter.report(payload, sota_root=tmp_path,
                                         gene_id="with_details", diagnostics=True)
    details = json.loads(output.read_text())["diagnostics"]
    assert details["overall"]["mAP75"] == pytest.approx(1.0)
    assert details["by_sequence"]["0"]["mAP50_95"] == pytest.approx(1.0)
    assert (tmp_path / "results" / "with_details_results.csv").read_text() == (
        f"{result.map50:.6f},{result.map50_95:.6f},321\n"
    )

    _, ordinary = map_reporter.report(payload, sota_root=tmp_path, gene_id="ordinary")
    assert "diagnostics" not in json.loads(ordinary.read_text())
