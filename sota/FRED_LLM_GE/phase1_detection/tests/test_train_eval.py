"""Offline integration evidence: synthetic model outputs, real COCO evaluation."""

import ast
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
        def __init__(self, path):
            self.model = types.SimpleNamespace(
                yaml=fake_yolo11_config(),
                parameters=lambda: [types.SimpleNamespace(numel=lambda: 1234)])
            self.path = Path(path)
        def load(self, weights):
            assert self.path.name == "yolo11m.yaml"
            return self
        def train(self, **kwargs):
            calls.append(kwargs)
            best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"synthetic checkpoint")
            self.trainer = types.SimpleNamespace(best=best)
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
    report = json.loads((output / "results" / "seed_check_evaluation.json").read_text())
    assert report["metadata"]["modality"] == "event"
    assert report["metadata"]["experiment_purpose"] == "synthetic offline integration fixture"
    assert report["metadata"]["candidate_gene_id"] == "seed"
    assert report["metadata"]["code_files_sha256"]
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


def test_run_loop_recognizes_failed_trainer_sentinel():
    source = (Path(__file__).resolve().parents[4] / "run_improved.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "check_contents_for_error")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "run_improved.py", "exec"), namespace)
    assert namespace[node.name]("FRED_RUN_FAILED [data]: missing labels") is False
