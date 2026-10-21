"""Tests for the D-FINE repeatability report, without a FRED download."""

import json

import pytest

from sota.FRED_LLM_GE.phase1_detection.compare_dfine_repeatability import compare


def _training(path, seed, *, checkpoint=None, score=None):
    score = 0.20 + seed * 0.01 if score is None else score
    metadata = {name: "same" for name in (
        "model_identity", "code_revision", "runtime_code_sha256", "dataset_revision",
        "manifest_identity", "split_identity", "annotation_policy",
        "training_recipe_identity", "training_sample_ids_sha256",
        "training_targets_sha256", "validation_sample_ids_sha256",
        "validation_targets_sha256", "postprocessing",
    )}
    metadata.update(official_split="challenging_train", training_seed=seed,
                    training_config_identity=f"config-{seed}",
                    initial_weights_sha256=f"initial-{seed}",
                    training_order_sha256=f"order-{seed}",
                    checkpoint_identity=checkpoint or f"checkpoint-{seed}",
                    runtime_environment={"hardware": "gpu", "pytorch": "2.x"})
    path.write_text(json.dumps({
        "gene_id": "seed", "mode": "validation", "metadata": metadata,
        "metrics": {"mAP50": score, "mAP50_95": score / 2},
        "training_history": [{"epoch": 1, "train_loss": 2.0,
                              "order_sha256": f"order-{seed}"}],
    }))
    return path


def _rescore(path, *, matches=True):
    passes = [{"mAP50": 0.2, "mAP50_95": 0.1, "predictions_sha256": "prediction"},
              {"mAP50": 0.2, "mAP50_95": 0.1,
               "predictions_sha256": "prediction" if matches else "changed"}]
    path.write_text(json.dumps({
        "schema_version": "fred-dfine-rescore-v1", "gene_id": "seed",
        "metadata": {"model_identity": "same",
                     "validation_sample_ids_sha256": "same", "postprocessing": "same"},
        "checkpoint_identity": "saved-seed", "passes": passes,
        "repeated_scoring_matches": matches, "reference_scoring_matches": True,
    }))
    return path


def test_repeatability_review_requires_real_repeats_and_independent_seeds(tmp_path):
    rescore = _rescore(tmp_path / "rescore.json")
    a = _training(tmp_path / "a.json", 0)
    b = _training(tmp_path / "b.json", 0)
    c = _training(tmp_path / "c.json", 1)
    d = _training(tmp_path / "d.json", 2)

    partial = compare(rescore, [a, b])
    assert partial["finding"] == "same_seed_repeated"
    assert partial["complete"] is False

    result = compare(rescore, [a, b], [c, d])
    assert result["complete"] is True
    assert result["independent_seeds"] == [0, 1, 2]
    assert result["independent_seed_spread"]["mAP50"]["range"] == pytest.approx(0.02)
    assert result["same_seed_training"]["first_history_difference"] is None

    reused = json.loads(d.read_text())
    reused["metadata"]["training_order_sha256"] = "order-1"
    d.write_text(json.dumps(reused))
    reused_result = compare(rescore, [a, b], [c, d])
    assert reused_result["finding"] == "independent_seeds_reused_initialization_or_order"
    assert reused_result["complete"] is False


def test_repeatability_review_locates_first_divergence(tmp_path):
    rescore = _rescore(tmp_path / "rescore.json")
    a = _training(tmp_path / "a.json", 0)
    b = _training(tmp_path / "b.json", 0)
    changed = json.loads(b.read_text())
    changed["training_history"][0]["train_loss"] = 2.5
    b.write_text(json.dumps(changed))
    result = compare(rescore, [a, b])
    assert result["finding"] == "training_history_diverged"
    assert result["same_seed_training"]["first_history_difference"] == {
        "epoch": 1, "fields": ["train_loss"]}

    changed["metadata"]["initial_weights_sha256"] = "different"
    b.write_text(json.dumps(changed))
    assert compare(rescore, [a, b])["finding"] == "initialization_or_training_order_changed"


def test_repeatability_review_rejects_incomplete_evidence_and_flags_scoring(tmp_path):
    rescore = _rescore(tmp_path / "rescore.json", matches=False)
    a = _training(tmp_path / "a.json", 0)
    b = _training(tmp_path / "b.json", 0)
    assert compare(rescore, [a, b])["finding"] == "same_checkpoint_scoring_changed"

    changed = json.loads(b.read_text())
    del changed["metadata"]["initial_weights_sha256"]
    b.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="step-1 evidence"):
        compare(rescore, [a, b])
