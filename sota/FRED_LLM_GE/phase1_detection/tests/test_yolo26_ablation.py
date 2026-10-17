import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ultralytics = pytest.importorskip("ultralytics")

from sota.FRED_LLM_GE.phase1_detection import candidate_runner, yolo26_ablation


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
    assert len(report["metadata"]["ablation_code_sha256"]) == 64
    assert report["metadata"]["resolved_train_args_sha256"] == (
        candidate_runner.sha256_file(run_dir / "ultralytics/train/args.yaml"))
    with pytest.raises(FileExistsError, match="fresh output root"):
        yolo26_ablation.run_experiment(
            plan, yolo26_ablation.DEFAULT_PLAN, protocol, candidate_runner.DEFAULT_PROTOCOL,
            experiment, seed, output, "cpu")
