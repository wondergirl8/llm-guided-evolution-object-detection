"""Offline integration evidence: synthetic model outputs, real COCO evaluation."""

import ast
import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as trainer
from sota.FRED_LLM_GE.phase1_detection.tests.test_candidate_runner import fake_yolo11_config


@pytest.fixture
def training_fixture(tmp_path, monkeypatch):
    data = tmp_path / "export"
    for split in ("train", "val"):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
        stem = "1_00000001" if split == "train" else "2_00000001"
        Image.new("RGB", (32, 32)).save(data / "images" / split / f"{stem}.png")
        (data / "labels" / split / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (data / "data.yaml").write_text(yaml.safe_dump({
        "path": str(data), "train": "images/train", "val": "images/val", "names": {0: "drone"}}))
    (data / "source.json").write_text(json.dumps({
        "dataset_revision": "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd",
        "manifest_sha256": "synthetic_manifest", "project_split_approval_reference": "synthetic_split",
        "purpose": "synthetic offline integration fixture"}))
    output = tmp_path / "output"
    monkeypatch.setattr(trainer, "SCRIPT_DIR", output)
    calls = []

    class FakeYOLO:
        epoch_losses = {}

        def __init__(self, path):
            self.model = types.SimpleNamespace(
                yaml=fake_yolo11_config(),
                parameters=lambda: [types.SimpleNamespace(numel=lambda: 1234)])
            self.path = Path(path)
            self.callbacks = {}
        def add_callback(self, event, callback):
            self.callbacks.setdefault(event, []).append(callback)
        def load(self, weights):
            assert self.path.name == "yolo11m.yaml"
            return self
        def train(self, **kwargs):
            calls.append(kwargs)
            best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"synthetic checkpoint")
            history = best.parent.parent / "results.csv"
            self.trainer = types.SimpleNamespace(best=best, csv=history)
            with history.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", *trainer.LOSS_COLUMNS])
                writer.writeheader()
            for epoch in range(1, kwargs["epochs"] + 1):
                with history.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["epoch", *trainer.LOSS_COLUMNS])
                    losses = {key: 1.0 for key in trainer.LOSS_COLUMNS}
                    losses.update(self.epoch_losses.get(epoch, {}))
                    writer.writerow({"epoch": epoch, **losses})
                for callback in self.callbacks.get("on_fit_epoch_end", []):
                    callback(self.trainer)
        def predict(self, source, **kwargs):
            if isinstance(source, np.ndarray):
                return [types.SimpleNamespace(orig_shape=source.shape[:2])]
            return [types.SimpleNamespace(path=p, orig_shape=(32, 32), boxes=types.SimpleNamespace(
                xyxy=np.array([[8, 8, 24, 24]]), cls=np.array([0]), conf=np.array([0.9]))) for p in source]

    monkeypatch.setitem(__import__('sys').modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return data / "data.yaml", output, calls, FakeYOLO


def test_seed_training_reaches_shared_evaluator_and_refuses_rerun(training_fixture, capsys):
    data, output, calls, _ = training_fixture
    path = trainer.run("network", output / "models", data, "yolo11m.pt", 1, 2, 64, "cpu", run_id="seed_check")
    values = [float(v) for v in path.read_text().strip().split(',')]
    assert values == pytest.approx([1, 1, 1234])
    assert "job done" in capsys.readouterr().out
    assert calls[0]["fraction"] == 1 and calls[0]["exist_ok"] is False
    assert calls[0]["amp"] is True
    assert not {"optimizer", "lr0", "momentum", "warmup_bias_lr"} & calls[0].keys()
    report = json.loads((output / "results" / "seed_check_evaluation.json").read_text())
    assert report["metadata"]["modality"] == "event"
    assert report["metadata"]["experiment_purpose"] == "synthetic offline integration fixture"
    assert report["metadata"]["candidate_gene_id"] == "seed"
    assert report["metadata"]["code_files_sha256"]
    assert report["metadata"]["training_config"]["amp"] is True
    assert report["metadata"]["training_config"]["optimizer"] == "auto"
    history = output / "runs" / "seed_check" / "results.csv"
    assert report["metadata"]["training_history"]["sha256"] == trainer.sha256_file(history)
    assert report["metadata"]["training_history"]["epochs_completed"] == 1
    assert (output / "trained_models" / "seed_check.pt").is_file()
    with pytest.raises(FileExistsError):
        trainer.run("network", output / "models", data, "yolo11m.pt", 1, 2, 64, "cpu", run_id="seed_check")


def test_model_failure_never_publishes_success_or_fitness(training_fixture, monkeypatch, capsys):
    data, output, _, model = training_fixture
    def fail(*a, **k):
        raise RuntimeError("synthetic infrastructure failure")
    monkeypatch.setattr(model, "train", fail)
    assert trainer.main(["--model", "network", "--data", str(data), "--weights", "yolo11m.pt",
                         "--device", "cpu", "--run-id", "failed_check"]) == 1
    text = capsys.readouterr().out
    assert "FRED_RUN_FAILED" in text and "job done" not in text
    assert not (output / "results" / "failed_check_results.csv").exists()
    assert json.loads((output / "failures" / "failed_check_failure.json").read_text())["status"] == "failed"


def test_candidate_protected_code_is_rejected_before_training(training_fixture, capsys):
    data, output, calls, _ = training_fixture
    variants = output / "models"
    variants.mkdir(parents=True)
    source = Path(trainer.__file__).with_name("network.py").read_text()
    (variants / "network_ABC.py").write_text(source + "\nraise RuntimeError('must not execute')\n")
    assert trainer.main(["--model", "network_ABC", "--variant_dir", str(variants), "--data", str(data)]) == 1
    assert not calls
    assert "job done" not in capsys.readouterr().out


def test_valid_mutation_uses_candidate_identity_and_shared_evaluator(training_fixture):
    data, output, calls, _ = training_fixture
    variants = output / "models"
    variants.mkdir(parents=True)
    source = Path(trainer.__file__).with_name("network.py").read_text()
    source = source.rsplit("GENOME =", 1)[0] + "GENOME = {'backbone_p3': 2, 'neck_p3': 2, 'attention': 3}\n"
    (variants / "network_ABC.py").write_text(source)
    path = trainer.run("network_ABC", variants, data, "yolo11m.pt", 1, 2, 64, "cpu", run_id="candidate_check")
    report = json.loads((output / "results" / "candidate_check_evaluation.json").read_text())
    assert report["metadata"]["candidate_gene_id"] == "ABC"
    assert [float(v) for v in path.read_text().split(',')] == pytest.approx([1, 1, 1234])
    assert len(calls) == 1


@pytest.mark.parametrize('label', ['0 nan 0.5 0.2 0.2', '1 0.5 0.5 0.2 0.2',
                                   '0 0.01 0.5 0.5 0.5', '0 0.5 0.5 0 0.2'])
def test_invalid_labels_are_not_silently_clamped(tmp_path, label):
    p = tmp_path / "label.txt"
    p.write_text(label)
    with pytest.raises(ValueError):
        trainer.read_targets(p, 32, 32)


def test_empty_labels_differ_from_missing_labels(tmp_path):
    p = tmp_path / "label.txt"
    with pytest.raises(FileNotFoundError):
        trainer.read_targets(p, 32, 32)
    p.touch()
    assert trainer.read_targets(p, 32, 32) == {"boxes": [], "labels": []}


def test_truncated_predictions_are_rejected(training_fixture):
    data, _, _, _ = training_fixture
    images, labels = trainer.val_images(data)
    model = types.SimpleNamespace(predict=lambda **kwargs: [])
    with pytest.raises(ValueError, match="fewer predictions"):
        trainer.predict_frames(model, images, labels, 64, 2, "cpu")


def test_list_source_cannot_turn_full_validation_set_into_one_gpu_batch(monkeypatch):
    images = [Path(f"52_{i:08d}.png") for i in range(1025)]
    calls = []

    def predict(**kwargs):
        # Emulate the Ultralytics list loader: its actual batch is len(source),
        # not the caller's batch option. Empty detections still count as frames.
        source = kwargs["source"]
        assert len(source) <= 2
        assert kwargs["conf"] == .001 and kwargs["iou"] == .7
        assert kwargs["max_det"] == 100 and kwargs["imgsz"] == 960
        calls.append(source)
        return iter(types.SimpleNamespace(path=p, orig_shape=(720, 1280)) for p in source)

    monkeypatch.setattr(trainer, "read_targets", lambda *a: {"boxes": [], "labels": []})
    monkeypatch.setattr(trainer, "yolo11_frame", lambda **kw: {"sample_id": kw["sample_id"],
                         "target": kw["target"], "prediction": {"boxes": [], "labels": [], "scores": []}})
    frames = trainer.predict_frames(types.SimpleNamespace(predict=predict), images, Path("labels"), 960, 2, "0")
    assert [f["sample_id"] for f in frames] == [p.stem for p in images]
    assert len(calls) == 513 and len(calls[-1]) == 1
    assert [p for call in calls for p in call] == [str(p) for p in images]


@pytest.mark.parametrize("failure", ["missing", "extra", "wrong_order"])
def test_chunk_predictions_fail_on_missing_extra_or_wrong_identity(monkeypatch, failure):
    images = [Path(f"52_{i:08d}.png") for i in range(5)]
    calls = []

    def predict(**kwargs):
        paths = kwargs["source"][:]
        calls.append(paths[:])
        if len(calls) == 2:
            paths = paths[:-1] if failure == "missing" else paths + paths[:1] if failure == "extra" else paths[::-1]
        return iter(types.SimpleNamespace(path=p, orig_shape=(720, 1280)) for p in paths)

    monkeypatch.setattr(trainer, "read_targets", lambda *a: {})
    monkeypatch.setattr(trainer, "yolo11_frame", lambda **kw: {})
    with pytest.raises(ValueError, match={"missing": "fewer", "extra": "extra", "wrong_order": "order"}[failure]):
        trainer.predict_frames(types.SimpleNamespace(predict=predict), images, Path("labels"), 960, 2, "0")
    assert len(calls) == 2


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_prediction_batch_is_rejected(batch):
    with pytest.raises(ValueError, match="positive integer"):
        trainer.predict_frames(None, [], Path("labels"), 960, batch, "0")


def test_run_loop_recognizes_failed_trainer_sentinel():
    source = (Path(__file__).resolve().parents[4] / "run_improved.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "check_contents_for_error")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "run_improved.py", "exec"), namespace)
    assert namespace[node.name]("FRED_RUN_FAILED [data]: missing labels") is False


@pytest.mark.parametrize("flag,amp", [("--amp", True), ("--no-amp", False)])
def test_precision_flag_reaches_training_and_report(training_fixture, flag, amp):
    data, output, calls, _ = training_fixture
    assert trainer.main(["--data", str(data), "--device", "cpu", "--epochs", "2",
                         "--run-id", "precision_check", flag]) == 0
    assert calls[0]["amp"] is amp
    report = json.loads((output / "results" / "precision_check_evaluation.json").read_text())
    assert report["metadata"]["training_config"]["amp"] is amp
    assert report["metadata"]["training_history"]["epochs_completed"] == 2


@pytest.mark.parametrize("column", trainer.LOSS_COLUMNS)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_loss_stops_at_bad_epoch_without_fitness(
        training_fixture, monkeypatch, capsys, column, value):
    data, output, _, model = training_fixture
    monkeypatch.setattr(model, "epoch_losses", {2: {column: value}})
    assert trainer.main(["--data", str(data), "--device", "cpu", "--epochs", "3",
                         "--run-id", "nonfinite_check", "--no-amp"]) == 1
    failure = json.loads((output / "failures" / "nonfinite_check_failure.json").read_text())
    assert failure["category"] == "training_history"
    assert f"non-finite {column} at epoch 2" in failure["message"]
    history = output / "runs" / "nonfinite_check" / "results.csv"
    with history.open() as handle:
        assert len(list(csv.DictReader(handle))) == 2  # Epoch 3 never runs.
    assert not (output / "trained_models" / "nonfinite_check.pt").exists()
    assert not (output / "results" / "nonfinite_check_results.csv").exists()
    assert not (output / "results" / "nonfinite_check_evaluation.json").exists()
    text = capsys.readouterr().out
    assert "FRED_RUN_FAILED" in text and "job done" not in text


def test_final_guard_rejects_recovered_history_when_callback_did_not_run(
        training_fixture, monkeypatch):
    data, output, _, model = training_fixture
    monkeypatch.setattr(model, "add_callback", lambda *args: None)
    monkeypatch.setattr(model, "epoch_losses", {1: {"val/cls_loss": "nan"}})
    # The last epoch is healthy, but the earlier invalid loss still rejects the run.
    assert trainer.main(["--data", str(data), "--device", "cpu", "--epochs", "2",
                         "--run-id", "recovered_check"]) == 1
    assert not (output / "results" / "recovered_check_results.csv").exists()


@pytest.mark.parametrize("contents", [
    None, "", "epoch,train/box_loss\n1,1\n",
    "epoch," + ",".join(trainer.LOSS_COLUMNS) + "\n",
    "epoch," + ",".join(trainer.LOSS_COLUMNS) + "\n1,1,1,1,1,1\n",
    "epoch," + ",".join(trainer.LOSS_COLUMNS) + "\n2,1,1,1,1,1,1\n",
    "epoch," + ",".join(trainer.LOSS_COLUMNS) + "\ninvalid,1,1,1,1,1,1\n",
])
def test_missing_or_malformed_training_history_is_rejected(tmp_path, contents):
    history = tmp_path / "results.csv"
    if contents is not None:
        history.write_text(contents)
    with pytest.raises(trainer.TrainingHistoryError):
        trainer.validate_training_history(history)


def test_explicit_adamw_rate_preserves_auto_beta_and_bias_warmup(training_fixture):
    data, output, calls, _ = training_fixture
    assert trainer.main(["--data", str(data), "--device", "cpu", "--epochs", "2",
                         "--run-id", "lr_check", "--no-amp", "--adamw-lr", "0.0002"]) == 0
    expected = {"optimizer": "AdamW", "lr0": 0.0002,
                "momentum": 0.9, "warmup_bias_lr": 0.0}
    for key, value in expected.items():
        assert calls[0][key] == value
    report = json.loads((output / "results" / "lr_check_evaluation.json").read_text())
    for key, value in expected.items():
        assert report["metadata"]["training_config"][key] == value
    assert calls[0]["amp"] is False


@pytest.mark.parametrize("rate", ["0", "-0.002", "nan", "inf", "-inf"])
def test_invalid_adamw_rate_fails_before_training(training_fixture, rate):
    data, output, calls, _ = training_fixture
    assert trainer.main(["--data", str(data), "--device", "cpu", "--run-id", "bad_lr",
                         f"--adamw-lr={rate}"]) == 1
    assert not calls
    assert not (output / "results" / "bad_lr_results.csv").exists()
    failure = json.loads((output / "failures" / "bad_lr_failure.json").read_text())
    assert "positive and finite" in failure["message"]
