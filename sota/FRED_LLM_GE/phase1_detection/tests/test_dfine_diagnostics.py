import json
import sys
from unittest.mock import patch

import pytest

from sota.FRED_LLM_GE.phase1_detection import candidate_runner
from sota.FRED_LLM_GE.phase1_detection.compare_dfine_diagnostics import compare


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
