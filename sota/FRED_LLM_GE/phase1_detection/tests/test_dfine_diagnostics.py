import json
import sys
from unittest.mock import patch

import pytest

from sota.FRED_LLM_GE.phase1_detection import candidate_runner
from sota.FRED_LLM_GE.phase1_detection.compare_dfine_diagnostics import compare
from sota.FRED_LLM_GE.phase1_detection.compare_dfine_training_arms import (
    compare as compare_training_arms,
)


def _report(path, gene, seed, score):
    report = {
        "schema_version": "fred-detection-report-v1", "metric_units": "fraction",
        "gene_id": gene, "mode": "validation", "num_frames": 2,
        "evaluator": {"backend": "pycocotools"},
        "metadata": {
            "training_seed": seed, "model_identity": gene,
            **{key: "shared" for key in (
                "dataset_revision", "manifest_identity", "split_identity",
                "annotation_policy", "postprocessing", "modality",
                "benchmark_split", "project_split", "official_split",
                "validation_sample_ids_sha256", "training_recipe_identity")},
        },
        "metrics": {"mAP50": score + 0.1, "mAP50_95": score, "params": 850000},
        "diagnostics": {
            "definition": "COCO bbox diagnostic",
            "overall": {"mAP75": score - 0.1, "mAP_small": score - 0.2,
                        "recall_small": 0.3, "false_negatives_iou50": 2,
                        "poorly_localized_iou50_to75": 1},
            "by_sequence": {"train/1": {"frames": 2, "ground_truth": 3,
                                        "false_negatives_iou50": 2,
                                        "poorly_localized_iou50_to75": 1,
                                        "mAP50": score + 0.1,
                                        "mAP75": score - 0.1,
                                        "mAP50_95": score}},
        },
    }
    path.write_text(json.dumps(report))
    return path


def _arm_report(path, arm, seed, score):
    _report(path, "seed", seed, score)
    report = json.loads(path.read_text())
    metadata = report["metadata"]
    metadata.update(training_arm=arm, training_seconds=100 if arm != "longer_budget" else 200,
                    initial_weights_sha256=f"initial-{seed}",
                    training_sample_ids_sha256="train-ids", code_revision="revision",
                    runtime_environment={"device": "same-gpu"})
    epochs = 40 if arm == "longer_budget" else 20
    metadata["training_recipe"] = {
        "family": "dfine", "proxy": {"seed": seed, "epochs": epochs},
        "family_settings": {"lr_schedule": (
            "cosine" if arm == "cosine_lr" else "constant")},
    }
    report["training_history"] = [
        {"epoch": epoch, "lr": 0.001 if arm != "cosine_lr" else 0.001 / epoch,
         "order_sha256": f"order-{seed}-{epoch}",
         **{name: 0.1 for name in (
             "loss_cls", "loss_bbox", "loss_giou", "loss_fgl", "loss_go_lsd")}}
        for epoch in range(1, epochs + 1)]
    path.write_text(json.dumps(report))
    return path


def test_training_arm_comparison_uses_complete_paired_seeds(tmp_path):
    paths = [_arm_report(tmp_path / f"{arm}_{seed}.json", arm, seed,
                         0.2 + 0.02 * seed + 0.01 * index)
             for seed in (0, 1)
             for index, arm in enumerate(("baseline", "longer_budget", "cosine_lr"))]
    result = compare_training_arms(paths)
    assert result["seeds"] == [0, 1]
    assert result["arm_means"]["cosine_lr"]["mAP50_95"] == pytest.approx(0.23)
    assert result["arm_means"]["longer_budget"]["training_seconds"] == 200
    assert result["paired_deltas_vs_baseline"][0]["mAP75"] == pytest.approx(0.01)

    with pytest.raises(ValueError, match="all three training arms"):
        compare_training_arms(paths[:-1])
    altered = json.loads(paths[1].read_text())
    altered["metadata"]["training_recipe"]["family_settings"]["lr"] = 0.01
    paths[1].write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="settings differ"):
        compare_training_arms(paths)
    altered["metadata"]["training_recipe"]["family_settings"].pop("lr")
    altered["metadata"]["initial_weights_sha256"] = "different"
    paths[1].write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="initialization and frame order"):
        compare_training_arms(paths)


def test_confirmation_seed_uses_separate_root_and_enables_diagnostics(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    arguments = ["candidate_runner.py", "--family", "dfine", "--model", "network_gene",
                 "--sota_root", str(tmp_path), "--training-seed", "7", "--device", "cpu"]
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    assert run.call_args.args[4]["proxy"]["seed"] == 7
    assert run.call_args.args[5] == str(tmp_path / "diagnostic_repeats" / "seed_7")
    assert run.call_args.args[2] == str(tmp_path / "models")
    assert run.call_args.kwargs["diagnostics"] is True

    result = tmp_path / "diagnostic_repeats" / "seed_7" / "results"
    result.mkdir(parents=True)
    (result / "gene_evaluation.json").write_text("{}")
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        with pytest.raises(SystemExit):
            candidate_runner.main()
    run.assert_not_called()


@pytest.mark.parametrize("arm,epochs,schedule", [
    ("baseline", 20, "constant"),
    ("longer_budget", 40, "constant"),
    ("cosine_lr", 20, "cosine"),
])
def test_training_arms_change_one_factor_and_isolate_outputs(
    tmp_path, monkeypatch, arm, epochs, schedule,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    arguments = ["candidate_runner.py", "--family", "dfine", "--model", "network_gene",
                 "--sota_root", str(tmp_path), "--training-seed", "3",
                 "--dfine-training-arm", arm, "--device", "cpu"]
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    protocol = run.call_args.args[4]
    assert protocol["challenging"]["search"]["epochs"] == epochs
    assert protocol["families"]["dfine"]["lr_schedule"] == schedule
    assert run.call_args.args[5] == str(tmp_path / "training_ablations" / arm / "seed_3")
    assert run.call_args.kwargs == {
        "diagnostics": True, "training_arm": arm, "refinement_bins": None}


@pytest.mark.parametrize("bins", [16, 32])
def test_refinement_ablation_changes_only_bins_and_isolates_outputs(
    tmp_path, monkeypatch, bins,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    arguments = ["candidate_runner.py", "--family", "dfine", "--model", "network",
                 "--sota_root", str(tmp_path), "--training-seed", "3",
                 "--dfine-reg-max", str(bins), "--device", "cpu"]
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        candidate_runner.main()
    protocol = run.call_args.args[4]
    original = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    assert protocol["families"]["dfine"]["model"] == {
        **original["families"]["dfine"]["model"], "reg_max": bins}
    assert protocol["challenging"] == original["challenging"]
    assert run.call_args.args[5] == str(
        tmp_path / "model_ablations" / f"reg_max_{bins}" / "seed_3")
    assert run.call_args.kwargs == {
        "diagnostics": True, "training_arm": None, "refinement_bins": bins}


def test_refinement_ablation_rejects_training_arm(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    arguments = ["candidate_runner.py", "--family", "dfine", "--model", "network",
                 "--sota_root", str(tmp_path), "--training-seed", "3",
                 "--dfine-reg-max", "32", "--dfine-training-arm", "cosine_lr",
                 "--device", "cpu"]
    with patch.object(sys, "argv", arguments), patch.object(candidate_runner, "run") as run:
        with pytest.raises(SystemExit):
            candidate_runner.main()
    run.assert_not_called()


def test_comparison_checks_protocol_and_exposes_seed_spread(tmp_path):
    paths = [_report(tmp_path / f"{gene}_{seed}.json", gene, seed, score)
             for gene, scores in (("seed", (0.1, 0.2)), ("candidate", (0.3, 0.4)))
             for seed, score in enumerate(scores)]
    result = compare(paths)
    assert result["mAP50_95_seed_spread"]["candidate"] == {
        "seeds": 2, "mean_mAP50_95": pytest.approx(0.35),
        "min_mAP50_95": 0.3, "max_mAP50_95": 0.4,
    }
    assert len(result["by_sequence"]) == 4
    assert result["runs"][0]["poorly_localized_iou50_to75"] == 1

    changed = json.loads(paths[-1].read_text())
    changed["metadata"]["training_recipe_identity"] = "different"
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different evaluation or training protocol"):
        compare(paths)

    changed["metadata"]["training_recipe_identity"] = "shared"
    changed["metadata"]["training_seed"] = 2
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="same training seeds"):
        compare(paths)


def test_checkpoint_review_compares_single_seed_without_ranking(tmp_path):
    paths = [_report(tmp_path / f"{gene}.json", gene, 0, score)
             for gene, score in (("seed", 0.1), ("candidate", 0.3))]

    result = compare(paths, checkpoint_review=True)

    assert len(result["runs"]) == 2
    assert result["runs"][0]["mAP75"] == pytest.approx(0.2)
    assert result["runs"][0]["same_model_source_as_seed"] is False
    assert result["deltas_vs_seed"][0]["mAP75"] == pytest.approx(0.2)
    assert result["deltas_vs_seed"][0]["false_negatives_iou50"] == 0
    assert "before ranking" in result["comparison_scope"]


def _confirmation_report(path, gene, seed, score):
    _report(path, gene, seed, score)
    report = json.loads(path.read_text())
    report["metadata"].update({
        "code_revision": "same-revision",
        "runtime_code_sha256": "same-runtime-code",
        "training_sample_ids_sha256": "same-training-frames",
        "training_targets_sha256": "same-training-labels",
        "validation_targets_sha256": "same-validation-labels",
        "training_order_sha256": f"order-{seed}",
        "runtime_environment": {"hardware": "same-gpu", "pytorch": "same-version"},
        "inference_device": "cuda",
        "preloaded_inference_ms_per_frame": 2.0 if gene == "seed" else 2.5,
    })
    if gene != "seed":
        report["metrics"]["params"] = 900000
    path.write_text(json.dumps(report))
    return path


def test_gain_confirmation_requires_paired_improvement_beyond_variation(tmp_path):
    paths = [_confirmation_report(tmp_path / f"{gene}_{seed}.json", gene, seed,
                                  0.30 + 0.005 * seed + gain)
             for gene, gain in (("seed", 0), ("candidate", 0.04))
             for seed in (0, 1, 2)]
    result = compare(paths, confirm_gains=True)["gain_confirmation"]
    candidate = result["candidates"]["candidate"]
    assert candidate["gain_confirmed_for_review"] is True
    assert candidate["metrics"]["mAP50_95"]["mean_gain"] == pytest.approx(0.04)
    assert candidate["metrics"]["mAP50_95"]["observed_variation"] == pytest.approx(0.01)
    assert candidate["delta_params"] == 50000
    assert candidate["delta_inference_ms_per_frame"] == pytest.approx(0.5)

    weak_paths = [_confirmation_report(tmp_path / f"weak_{gene}_{seed}.json", gene, seed,
                                       0.30 + 0.005 * seed + gain)
                  for gene, gain in (("seed", 0), ("candidate", 0.002))
                  for seed in (0, 1, 2)]
    weak = compare(weak_paths, confirm_gains=True)["gain_confirmation"]
    assert weak["candidates"]["candidate"]["gain_confirmed_for_review"] is False

    changed = json.loads(paths[-1].read_text())
    changed["metadata"]["runtime_environment"]["hardware"] = "other-gpu"
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hardware differs"):
        compare(paths, confirm_gains=True)
    changed["metadata"]["runtime_environment"]["hardware"] = "same-gpu"
    changed["metadata"]["validation_targets_sha256"] = "different-labels"
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="labels"):
        compare(paths, confirm_gains=True)
