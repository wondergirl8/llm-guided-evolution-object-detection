"""Offline acceptance checks for the bounded resolution comparison."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sota.FRED_LLM_GE.phase1_detection import scene_resolution_check as resolution


def config():
    return {"epochs": 50, "fraction": 1.0, "batch": 2, "imgsz": 640, "seed": 0,
            "deterministic": True, "gene_id": "seed", "amp": False, "optimizer": "AdamW",
            "lr0": 0.0002, "momentum": 0.9, "warmup_bias_lr": 0.0}


def test_resolution_run_preserves_other_training_settings_and_starts_pretrained():
    original = config()
    weights = Path("original_pretrained.pt")
    options = resolution.training_options(original, Path("original/data.yaml"), weights, "resolution_960_123")
    assert options["imgsz"] == 960 and original["imgsz"] == 640
    assert options["epochs"] == 50 and options["batch"] == 2 and options["fraction"] == 1
    assert options["amp"] is False and options["adamw_lr"] == 0.0002
    assert options["weights"] == str(weights) and options["model_name"] == "network"
    for key, value in (("amp", True), ("lr0", 0.002), ("optimizer", "auto"), ("imgsz", 960),
                       ("epochs", 20), ("seed", 1)):
        with pytest.raises(ValueError, match="original run differs"):
            resolution.training_options({**original, key: value}, Path("data.yaml"), weights, "new")


def test_changed_inputs_and_overwrite_are_rejected(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"unchanged")
    identities = {str(path): resolution.sha256(path)}
    resolution.assert_unchanged(identities)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        resolution.assert_unchanged(identities)
    output = tmp_path / "summary.json"
    resolution.write_json(output, {"status": "completed"})
    with pytest.raises(FileExistsError):
        resolution.write_json(output, {"status": "overwrite"})


@pytest.mark.parametrize("duplicate", [False, True])
def test_submission_counts_one_two_hour_gpu_job_and_waits_for_existing_gpus(tmp_path, monkeypatch, duplicate):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resolution, "reference_inputs", lambda *a: {
        "reference_run": "scene_check_6141846", "directory": str(tmp_path), "purpose": resolution.PURPOSE})
    calls = []

    def run(command):
        calls.append(command)
        if command[:3] == ["git", "branch", "--show-current"]:
            return "fred-yolo11-infrastructure"
        if command[:2] == ["git", "rev-parse"]:
            return ".git/MERGE_HEAD" if "--git-path" in command else "a" * 40
        if command[0] == "squeue":
            return "123"
        if command[0] == "scontrol":
            name = "fred-resolution" if duplicate else "other"
            return f"JobName={name} ReqTRES=gres/gpu=1,gres/gpu:h100=1 TimeLimit=02:00:00"
        if command[:2] == ["sbatch", "--parsable"]:
            return "2001"
        return ""

    monkeypatch.setattr(resolution.scheduler, "run", run)
    if duplicate:
        with pytest.raises(ValueError, match="already queued"):
            resolution.submit(tmp_path, "scene_check_6141846")
        assert not any(c[0] == "sbatch" for c in calls)
    else:
        resolution.submit(tmp_path, "scene_check_6141846")
        submit = next(c for c in calls if c[:2] == ["sbatch", "--parsable"])
        assert "--dependency=afterany:123" in submit and "--kill-on-invalid-dep=yes" in submit
        assert len([c for c in calls if c[:2] == ["sbatch", "--test-only"]]) == 1
        request = json.loads(next((tmp_path / "data/fred_resolution_checks").glob("*/request.json")).read_text())
        assert request["budget"]["existing_jobs"] == 1
        assert request["budget"]["total_reserved_gpu_hours"] == 4
        assert request["imgsz"] == 960


def test_job_envelope_matches_submission_budget():
    text = resolution.JOB.read_text()
    assert "#SBATCH --time=02:00:00" in text and resolution.GPU_SECONDS == 7200
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --ntasks=1" in text and "#SBATCH --nodes=1" in text
    assert "OMP_NUM_THREADS" in text


@pytest.mark.parametrize("failure", [None, "timeout", "wrong_resolution", "input_changed", "scene_reproduction"])
def test_execution_publishes_success_only_for_complete_unchanged_run(tmp_path, monkeypatch, failure):
    from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as trainer
    from sota.FRED_LLM_GE.phase1_detection import fitness, review_scene_checkpoint as reviewer
    from sota.FRED_LLM_GE.data import scene_diagnostic_export as exporter
    monkeypatch.setattr(resolution, "ROOT", tmp_path / "seed")
    monkeypatch.setenv("SLURM_JOB_ID", "2001")
    output, source = tmp_path / "output", tmp_path / "source"
    output.mkdir()
    (source / "export/images/train").mkdir(parents=True)
    image = source / "export/images/train/3_00000001.png"
    image.write_bytes(b"synthetic bytes; actual PNG validation is mocked")
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"synthetic initial weights")
    original_config = config()
    metrics = {"mAP50": 0.88, "mAP50_95": 0.48, "params": 10}
    inputs = {"directory": str(source), "reference_run": "scene_check_123", "reference_metrics": metrics,
              "starting_weights": str(weights), "input_sha256": {str(weights): resolution.sha256(weights)},
              "training_config": original_config, "purpose": resolution.PURPOSE}
    resolution.write_json(output / "request.json", {**inputs, "imgsz": 960, "code_revision": "original"})
    monkeypatch.setattr(resolution.scheduler, "run", lambda _: "original")
    monkeypatch.setattr(resolution, "reference_inputs", lambda *a: dict(inputs))
    monkeypatch.setattr(resolution.importlib.metadata, "version", lambda _: "8.4.165")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="synthetic", cuda=SimpleNamespace(is_available=lambda: True,
        device_count=lambda: 1, get_device_name=lambda _: "synthetic GPU")))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: object()))
    monkeypatch.setattr(trainer, "validate_data", lambda _: None)
    history = {"status": "finite", "epochs_completed": 50}
    monkeypatch.setattr(trainer, "validate_training_history", lambda _: history)
    monkeypatch.setattr(exporter, "verify_export", lambda _: {"counts": {"validation": 1024},
                       "selected_sequences": {"52": "validation", "180": "validation", "3": "train"}})
    images = [Path(f"{s}_{i:08d}.png") for s in ("180", "52") for i in range(512)]
    monkeypatch.setattr(trainer, "val_images", lambda _: (images, source / "export/labels/val"))
    monkeypatch.setattr(trainer, "predict_frames", lambda *a: [{"sample_id": p.stem} for p in images])
    monkeypatch.setattr(fitness, "evaluate_frames", lambda *a, **kw: SimpleNamespace(
                       map50=.9 if failure == "scene_reproduction" else .88, map50_95=.48, params=10))
    monkeypatch.setattr(reviewer, "preview", lambda images, frames, destination: destination.write_text("synthetic preview"))
    calls = []

    def train(**options):
        calls.append(options)
        if failure == "timeout":
            raise RuntimeError("synthetic timeout")
        if failure == "input_changed":
            image.write_bytes(b"changed source")
        folder = resolution.ROOT / "results"
        folder.mkdir(parents=True)
        checkpoint = resolution.ROOT / "trained_models/resolution_960_2001.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic best checkpoint")
        resolution.write_json(folder / "resolution_960_2001_evaluation.json", {
            "metrics": metrics, "num_frames": 1024, "metadata": {
            "checkpoint_identity": resolution.sha256(checkpoint),
            "training_history": history, "training_config": {"imgsz": 640 if failure == "wrong_resolution" else 960,
                                                           "amp": False, "lr0": 0.0002}}})

    monkeypatch.setattr(trainer, "run", train)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            resolution.execute(output)
        assert not (output / "summary.json").exists()
    else:
        resolution.execute(output)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["imgsz"] == 960
        assert summary["reference_metrics"] == metrics
        assert summary["per_scene_validation"]["52"]["num_frames"] == 512
        assert len(list(output.glob("sequence_*_preview.png"))) == 2
        assert summary["starting_weights_sha256"] == resolution.sha256(weights)
    assert calls[0]["weights"] == str(weights) and calls[0]["imgsz"] == 960
    assert weights.read_bytes() == b"synthetic initial weights"


def test_reference_capture_checks_previous_code_checkpoint_and_provenance(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as trainer
    from sota.FRED_LLM_GE.data import scene_diagnostic_export as exporter
    seed = tmp_path / "phase1/seeds/yolo11"
    monkeypatch.setattr(resolution, "ROOT", seed)
    (seed / "results").mkdir(parents=True)
    (seed / "trained_models").mkdir()
    source = tmp_path / "original"
    (source / "export").mkdir(parents=True)
    for path in (source / "manifest.sqlite", source / "export/source.json", source / "export/data.yaml",
                 seed / "network.py", seed / "trained_models/scene_check_123.pt"):
        path.write_bytes(b"synthetic immutable artifact")
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"original pretrained bytes")
    metrics = {"mAP50": .88, "mAP50_95": .48, "params": 10}
    history = {"epochs_completed": 50, "status": "finite"}
    report = {"num_frames": 1024, "metrics": metrics, "metadata": {
        "training_config": {**config(), "weights": str(weights)}, "training_history": history,
        "checkpoint_identity": resolution.sha256(seed / "trained_models/scene_check_123.pt"),
        "source_record_sha256": resolution.sha256(source / "export/source.json"),
        "manifest_identity": resolution.sha256(source / "manifest.sqlite"),
        "code_files_sha256": {"network.py": resolution.sha256(seed / "network.py")}}}
    resolution.write_json(seed / "results/scene_check_123_evaluation.json", report)
    resolution.write_json(source / "summary.json", {"status": "completed", "run_id": "scene_check_123", "validation": metrics})
    monkeypatch.setattr(exporter, "verify_export", lambda _: {"counts": {"validation": 1024}})
    monkeypatch.setattr(trainer, "validate_training_history", lambda _: history)
    inputs = resolution.reference_inputs(source, "scene_check_123")
    assert inputs["starting_weights"] == str(weights)
    assert inputs["input_sha256"][str(weights)] == resolution.sha256(weights)
    (seed / "network.py").write_text("changed code")
    with pytest.raises(ValueError, match="model/evaluator code changed"):
        resolution.reference_inputs(source, "scene_check_123")
