import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ultralytics = pytest.importorskip("ultralytics")

from sota.FRED_LLM_GE.phase1_detection import (
    candidate_runner, compare_yolo26_ablations, yolo26_ablation,
)


def test_ablation_check_builds_each_architecture_once(capsys):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    plan = yolo26_ablation.load_plan(yolo26_ablation.DEFAULT_PLAN, protocol)
    with patch.object(ultralytics, "YOLO", wraps=ultralytics.YOLO) as build:
        yolo26_ablation.check_architectures(plan, protocol)
    assert build.call_count == 3
    output = capsys.readouterr().out
    assert all(f"{name}: " in output for name in plan["experiments"])
    assert "seed_p4_half: 2353288 parameters" in output
    assert "seed_p5_half: " in output


def test_ablation_plan_rejects_confounded_comparison(tmp_path):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    plan = yaml.safe_load(yolo26_ablation.DEFAULT_PLAN.read_text())
    plan["experiments"]["seed_longer"]["train"] = {"mosaic": 0.0}
    path = tmp_path / "bad_plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match="exactly one factor"):
        yolo26_ablation.load_plan(path, protocol)


def test_ablation_plan_rejects_ignored_learning_rate(tmp_path):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    plan = yaml.safe_load(yolo26_ablation.DEFAULT_PLAN.read_text())
    plan["experiments"]["seed"]["train"] = {"lr0": 0.001}
    path = tmp_path / "bad_plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match="optimizer=auto ignores lr0"):
        yolo26_ablation.load_plan(path, protocol)


@pytest.mark.parametrize("experiment, seed, expected_args, expected_params", [
    ("seed_no_mosaic", 0, {"mosaic": 0.0}, 2515976),
    ("seed_sgd_low_lr", 1, {"optimizer": "SGD", "lr0": 0.001}, 2515976),
    ("seed_p4_half", 2, {}, 2353288),
])
def test_ablation_smoke_saves_resolved_args_curves_and_report(
        tmp_path, monkeypatch, experiment, seed, expected_args, expected_params):
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        images = dataset / split / "images"
        labels = dataset / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.fromarray(np.full((64, 96, 3), 80, dtype=np.uint8)).save(images / "0.png")
        (labels / "0.txt").write_text("0 0.2 0.2 0.1 0.1\n")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(dataset), "train": "train/images",
                                         "val": "val/images", "names": {0: "drone"}}))
    validation = {"images": torch.zeros(1, 3, 64, 96, dtype=torch.uint8),
                  "orig_boxes": [torch.tensor([[10.0, 10.0, 20.0, 20.0]])],
                  "sample_ids": ["train/1:0"], "orig_size": (96, 64)}
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    protocol["challenging"]["search"]["epochs"] = 1
    protocol["families"]["yolo26_p2"]["workers"] = 0
    proxy = dict(protocol["proxy"], img_width=96, batch_size=1, epochs=1,
                 train_sequences=["0"], val_sequences=["1"], frame_step=1,
                 val_frame_step=1)
    monkeypatch.setattr(candidate_runner, "challenging_search", lambda *_: (
        proxy, {"data_yaml": data_yaml, "frames": 1}, lambda: validation))
    audit = {"train": {"issue_count": 0, "marker_sha256": "a" * 64,
                       "label_sha256": "b" * 64},
             "search_val": {"issue_count": 0, "marker_sha256": "c" * 64,
                            "label_sha256": "d" * 64}}
    monkeypatch.setattr(yolo26_ablation.audit_yolo26_data_path, "audit_search",
                        lambda *_: audit)
    plan = yolo26_ablation.load_plan(yolo26_ablation.DEFAULT_PLAN, protocol)
    output = tmp_path / "ablations"
    run_dir = yolo26_ablation.run_experiment(
        plan, yolo26_ablation.DEFAULT_PLAN, protocol, candidate_runner.DEFAULT_PROTOCOL,
        experiment, seed, output, "cpu")

    args = yaml.safe_load((run_dir / "ultralytics/train/args.yaml").read_text())
    for key, value in expected_args.items():
        assert args[key] == value
    assert args["epochs"] == 1
    assert args["seed"] == seed
    assert len((run_dir / "ultralytics/train/results.csv").read_text().splitlines()) >= 2
    report = json.loads((run_dir / "results/ablation_evaluation.json").read_text())
    assert "diagnostics" in report
    assert report["num_frames"] == 1
    assert report["metrics"]["params"] == expected_params
    assert report["metadata"]["ablation_experiment"] == experiment
    assert report["metadata"]["ablation_comparison"] == plan["experiments"][experiment]["compare_to"]
    assert report["metadata"]["resolved_optimizer"]
    assert report["metadata"]["resolved_lr0"] > 0
    requested = plan["experiments"][experiment].get("train", {})
    if "optimizer" in requested:
        assert report["metadata"]["resolved_optimizer"] == requested["optimizer"]
    if "lr0" in requested:
        assert report["metadata"]["resolved_lr0"] == requested["lr0"]
    if "momentum" in requested:
        assert report["metadata"]["resolved_momentum"] == requested["momentum"]
    if "warmup_bias_lr" in requested:
        assert report["metadata"]["resolved_warmup_bias_lr"] == requested["warmup_bias_lr"]
    assert report["metadata"]["completed_epochs"] == 1
    assert report["metadata"]["training_seconds"] > 0
    assert len(report["metadata"]["initial_weights_sha256"]) == 64
    assert len(report["metadata"]["validation_sample_ids_sha256"]) == 64
    assert len(report["metadata"]["ablation_code_sha256"]) == 64
    assert report["metadata"]["resolved_train_args_sha256"] == (
        candidate_runner.sha256_file(run_dir / "ultralytics/train/args.yaml"))
    assert json.loads((run_dir / "data_path_audit.json").read_text()) == audit
    assert report["metadata"]["training_label_sha256"] == "b" * 64
    assert report["metadata"]["validation_label_sha256"] == "d" * 64
    if experiment == "seed_no_mosaic":
        anchor_dir = yolo26_ablation.run_experiment(
            plan, yolo26_ablation.DEFAULT_PLAN, protocol, candidate_runner.DEFAULT_PROTOCOL,
            "seed_fixed_adamw", seed, output, "cpu")
        anchor = json.loads((anchor_dir / "results/ablation_evaluation.json").read_text())
        assert (report["metadata"]["initial_weights_sha256"]
                == anchor["metadata"]["initial_weights_sha256"])
    with pytest.raises(FileExistsError, match="fresh output root"):
        yolo26_ablation.run_experiment(
            plan, yolo26_ablation.DEFAULT_PLAN, protocol, candidate_runner.DEFAULT_PROTOCOL,
            experiment, seed, output, "cpu")


def test_ablation_refuses_data_path_mismatch_before_training(tmp_path, monkeypatch):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    plan = yolo26_ablation.load_plan(yolo26_ablation.DEFAULT_PLAN, protocol)
    monkeypatch.setattr(yolo26_ablation.audit_yolo26_data_path, "audit_search",
                        lambda *_: {"train": {"issue_count": 1},
                                    "search_val": {"issue_count": 0}})
    with pytest.raises(ValueError, match="data-path audit found mismatches"):
        yolo26_ablation.run_experiment(
            plan, yolo26_ablation.DEFAULT_PLAN, protocol, candidate_runner.DEFAULT_PROTOCOL,
            "seed_fixed_adamw", 0, tmp_path / "ablations", "cpu")
    assert not (tmp_path / "ablations").exists()


def test_training_comparison_requires_paired_seeds_and_shared_initialization(tmp_path):
    protocol = candidate_runner.load_protocol(candidate_runner.DEFAULT_PROTOCOL)
    plan = yolo26_ablation.load_plan(yolo26_ablation.DEFAULT_PLAN, protocol)
    shared = {key: "same" for key in compare_yolo26_ablations.SAME_FIELDS}
    shared["runtime_environment"] = {"device": "cuda:0"}
    for arm, ap in (("seed_fixed_adamw", 0.4), ("seed_no_mosaic", 0.43)):
        for seed in plan["repeat_seeds"]:
            directory = tmp_path / arm / f"seed_{seed}" / "results"
            directory.mkdir(parents=True)
            report = {
                "schema_version": "fred-detection-report-v1", "gene_id": "ablation",
                "mode": "validation", "metric_units": "fraction", "num_frames": 10,
                "evaluator": {"backend": "pycocotools"},
                "metrics": {"mAP50": ap + 0.3, "mAP50_95": ap + seed * 0.001,
                            "params": 2515976},
                "diagnostics": {"definition": "COCO bbox", "overall": {
                    "mAP75": ap + 0.1, "mAP_small": 0.2, "recall_small": 0.3},
                    "by_sequence": {"train/4": {"frames": 10, "ground_truth": 5}}},
                "metadata": dict(shared, ablation_experiment=arm,
                                 ablation_comparison=plan["experiments"][arm].get("compare_to"),
                                 ablation_seed=seed, completed_epochs=20,
                                 initial_weights_sha256=f"weights-{seed}",
                                 resolved_optimizer="AdamW", resolved_lr0=0.002,
                                 resolved_momentum=0.9, resolved_warmup_bias_lr=0.0,
                                 training_seconds=10.0),
            }
            (directory / "ablation_evaluation.json").write_text(json.dumps(report))

    result = compare_yolo26_ablations.compare(
        tmp_path, plan, protocol, ["seed_no_mosaic"])
    assert result["seeds"] == [0, 1, 2]
    comparison = result["comparisons"][0]
    assert comparison["experiment"] == "seed_no_mosaic"
    assert comparison["mean_deltas"]["mAP50_95"] == pytest.approx(0.03)
    assert comparison["consistent_mAP50_95_gain_for_review"] is True

    report_path = (tmp_path / "seed_no_mosaic" / "seed_1" /
                   "results/ablation_evaluation.json")
    report = json.loads(report_path.read_text())
    report["metadata"]["initial_weights_sha256"] = "different"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="initial model weights differ"):
        compare_yolo26_ablations.compare(tmp_path, plan, protocol, ["seed_no_mosaic"])

    report["metadata"]["initial_weights_sha256"] = "weights-1"
    report["metadata"]["validation_sample_ids_sha256"] = "different"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="model, data, evaluator, or runtime differs"):
        compare_yolo26_ablations.compare(tmp_path, plan, protocol, ["seed_no_mosaic"])

    report["metadata"]["validation_sample_ids_sha256"] = "same"
    report["metadata"]["training_label_sha256"] = "changed"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="model, data, evaluator, or runtime differs"):
        compare_yolo26_ablations.compare(tmp_path, plan, protocol, ["seed_no_mosaic"])

    report_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing paired ablation report"):
        compare_yolo26_ablations.compare(tmp_path, plan, protocol, ["seed_no_mosaic"])
