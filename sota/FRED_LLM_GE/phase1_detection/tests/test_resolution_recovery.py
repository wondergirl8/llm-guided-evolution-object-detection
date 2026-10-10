"""Offline recovery guards; GPU execution is verified separately on ICE."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sota.FRED_LLM_GE.phase1_detection import checkpoint_code as code
from sota.FRED_LLM_GE.phase1_detection import recover_scene_resolution as recovery
from sota.FRED_LLM_GE.phase1_detection import scene_resolution_check as resolution
from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as trainer
from sota.FRED_LLM_GE.phase1_detection.tests.test_scene_resolution_check import config


def test_only_the_known_prediction_body_change_is_compatible(tmp_path):
    root = tmp_path / "seeds/yolo11"
    root.mkdir(parents=True)
    path = root / "train_eval.py"
    current = Path(trainer.__file__).read_bytes()
    metadata = {"code_files_sha256": {"train_eval.py": code.LEGACY_TRAINER_SHA256}}
    path.write_bytes(current)
    changes = code.verify_code_files(root, metadata)
    assert changes["train_eval.py"]["original_sha256"] == code.LEGACY_TRAINER_SHA256
    path.write_bytes(current.replace(b"PREDICT_CONF = 0.001", b"PREDICT_CONF = 0.1"))
    with pytest.raises(ValueError, match="beyond prediction batching"):
        code.verify_code_files(root, metadata)
    path.write_bytes(current + b"\ndef injected_training(): pass\n")
    with pytest.raises(ValueError, match="beyond prediction batching"):
        code.verify_code_files(root, metadata)
    with pytest.raises(ValueError, match="code changed"):
        code.verify_code_files(root, {"code_files_sha256": {"train_eval.py": "unrecognized"}})


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recovery, "ROOT", tmp_path / "seed")
    failed = tmp_path / "failed"
    failed.mkdir()
    directory = tmp_path / "source"
    content = {}
    for split, count in (("train", 1536), ("val", 1024)):
        for kind, suffix in (("images", ".png"), ("labels", ".txt")):
            folder = directory / "export" / kind / split
            folder.mkdir(parents=True)
            for i in range(count):
                path = folder / f"52_{i:08d}{suffix}"
                path.write_bytes(b"synthetic snapshot; PNG validation mocked")
                content[str(path)] = resolution.sha256(path)
    data = directory / "export/data.yaml"
    data.write_text("synthetic data config")
    metrics = {"mAP50": .88, "mAP50_95": .48, "params": 10}
    reference = {"directory": str(directory), "reference_run": "scene_check_123", "reference_metrics": metrics,
                 "training_config": config(), "input_sha256": {str(data): resolution.sha256(data)},
                 "starting_weights": "synthetic_pretrained.pt", "purpose": resolution.PURPOSE}
    request = {**reference, "imgsz": 960, "code_revision": code.LEGACY_REVISION}
    resolution.write_json(failed / "request.json", request)
    resolution.write_json(failed / "job.json", {"job_id": "2001", "output": str(failed),
                          "code_revision": code.LEGACY_REVISION})
    resolution.write_json(failed / "source_bytes.json", content)
    run = recovery.ROOT / "runs/resolution_960_2001"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best.pt").write_bytes(b"synthetic best checkpoint")
    copied = recovery.ROOT / "trained_models/resolution_960_2001.pt"
    copied.parent.mkdir()
    copied.write_bytes(b"synthetic best checkpoint")
    with (run / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *trainer.LOSS_COLUMNS])
        writer.writeheader()
        for epoch in range(1, 51):
            writer.writerow({"epoch": epoch, **dict.fromkeys(trainer.LOSS_COLUMNS, 1)})
    (run / "args.yaml").write_text(yaml.safe_dump({**config(), "imgsz": 960, "data": str(data)}))
    log = tmp_path / "data/logs/fred-resolution-2001.out"
    log.parent.mkdir(parents=True)
    log.write_text("predict_frames\ntorch.OutOfMemoryError: CUDA out of memory\n")
    monkeypatch.setattr(trainer, "validate_data", lambda _: None)
    monkeypatch.setattr(resolution, "reference_inputs", lambda *a: {**reference, "reference_code_changes": {}})
    monkeypatch.setattr(recovery.scheduler, "run", lambda cmd: "2001|FAILED|1:0" if cmd[0] == "sacct" else "new-code")
    return failed, run, copied, log


@pytest.mark.parametrize("failure", [None, "incomplete", "nonfinite", "checkpoint", "source", "config", "active", "snapshot"])
def test_recovery_requires_completed_finite_history_equal_checkpoint_and_unchanged_data(saved_run, monkeypatch, failure):
    failed, run, copied, log = saved_run
    if failure == "incomplete":
        history = run / "results.csv"
        history.write_text("\n".join(history.read_text().splitlines()[:-1]) + "\n")
    elif failure == "nonfinite":
        history = run / "results.csv"
        history.write_text(history.read_text().replace("50,1", "50,nan", 1))
    elif failure == "checkpoint":
        copied.write_bytes(b"different")
    elif failure == "source":
        path = next(iter(json.loads((failed / "source_bytes.json").read_text())))
        Path(path).write_bytes(b"changed source")
    elif failure == "config":
        args = yaml.safe_load((run / "args.yaml").read_text())
        (run / "args.yaml").write_text(yaml.safe_dump({**args, "amp": True}))
    elif failure == "active":
        monkeypatch.setattr(recovery.scheduler, "run", lambda _: "2001|RUNNING|0:0")
    elif failure == "snapshot":
        (failed / "source_bytes.json").write_text("{}")
    if failure:
        with pytest.raises((ValueError, trainer.TrainingHistoryError)):
            recovery.failed_inputs(failed)
    else:
        result = recovery.failed_inputs(failed)
        assert result["history"]["epochs_completed"] == 50
        assert result["checkpoint_sha256"] == resolution.sha256(copied)
        assert result["original_job_status"] == "FAILED"
        assert str(log.resolve()) in result["input_sha256"]


@pytest.mark.parametrize("failure,real_report", [(None, False), (None, True), ("oom", False),
                                               ("changed_input", False), ("wrong_params", False)])
def test_execute_never_trains_or_overwrites_original_run(saved_run, monkeypatch, failure, real_report):
    from sota.FRED_LLM_GE.phase1_detection import fitness, map_reporter, review_scene_checkpoint as reviewer
    from sota.FRED_LLM_GE.data import scene_diagnostic_export as exporter
    failed, run, copied, log = saved_run
    output = failed.parent / "recovery"
    output.mkdir()
    inputs = recovery.failed_inputs(failed)
    resolution.write_json(output / "request.json", {**inputs, "code_revision": "new-code"})
    def forbidden_training(*a, **kw):
        pytest.fail("recovery must never invoke training")
    monkeypatch.setattr(trainer, "run", forbidden_training)
    images = [Path(f"{s}_{i:08d}.png") for s in ("52", "180") for i in range(512)]
    monkeypatch.setattr(trainer, "val_images", lambda _: (images, Path("labels")))
    monkeypatch.setattr(exporter, "verify_export", lambda _: {"selected_sequences": {"52": "validation", "180": "validation"}})
    monkeypatch.setattr(recovery.importlib.metadata, "version", lambda _: "8.4.165")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _: None,
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True)),
        use_deterministic_algorithms=lambda *a, **kw: None,
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)))
    model = SimpleNamespace(model=SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 9 if failure == "wrong_params" else 10)]))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model))
    def predict(*args):
        assert args[3:6] == (960, 2, "0")
        if failure == "oom":
            raise RuntimeError("synthetic GPU OOM")
        if failure == "changed_input":
            log.write_text("changed log")
        return [{"sample_id": p.stem, "width": 1280, "height": 720,
                 "target": {"boxes": [[10, 10, 20, 20]], "labels": [0]},
                 "prediction": {"boxes": [], "labels": [], "scores": []}} for p in images]
    monkeypatch.setattr(trainer, "predict_frames", predict)
    if real_report:
        directory = Path(inputs["directory"]) / "export"
        resolution.write_json(directory / "source.json", {"dataset_revision": "synthetic revision",
            "manifest_sha256": "synthetic manifest", "project_split_approval_reference": "synthetic split"})
        (recovery.ROOT / "network.py").write_text("synthetic architecture bytes")
    else:
        monkeypatch.setattr(trainer, "report_metadata", lambda *a: {})
        monkeypatch.setattr(fitness, "evaluate_frames", lambda *a, **kw: SimpleNamespace(map50=.9, map50_95=.5, params=10))
    monkeypatch.setattr(reviewer, "preview", lambda *a: None)
    def publish(payload, *, sota_root, gene_id):
        assert sota_root == output
        assert payload["metadata"]["training_code_revision"] == code.LEGACY_REVISION
        report = output / "evaluation.json"
        resolution.write_json(report, {"metrics": {"mAP50": .9, "mAP50_95": .5, "params": 10}})
        return None, report
    if not real_report:
        monkeypatch.setattr(map_reporter, "report", publish)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            recovery.execute(output)
        assert not (output / "summary.json").exists()
    else:
        recovery.execute(output)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["training_repeated"] is False
        assert summary["original_job_status"] == "FAILED"
        assert summary["per_scene_validation"]["52"]["num_frames"] == 512
        if real_report:
            report = json.loads(Path(summary["evaluation_report"]).read_text())
            assert report["num_frames"] == 1024
            assert report["metrics"]["mAP50"] == 0
            assert report["metadata"]["training_code_revision"] == code.LEGACY_REVISION
            assert (output / "results/resolution_960_2001_results.csv").is_file()
    assert copied.read_bytes() == b"synthetic best checkpoint"
    assert not (failed / "summary.json").exists()
    assert not (recovery.ROOT / "results").exists()


@pytest.mark.parametrize("duplicate", [False, True])
def test_submission_budget_dependency_and_duplicate_guard(tmp_path, monkeypatch, duplicate):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recovery, "failed_inputs", lambda _: {"run_id": "resolution_960_2001"})
    calls = []
    def command(args):
        calls.append(args)
        if args[:3] == ["git", "branch", "--show-current"]:
            return "fred-yolo11-infrastructure"
        if args[:2] == ["git", "rev-parse"]:
            return ".git/MERGE_HEAD" if "--git-path" in args else "revision"
        if args[0] == "squeue":
            return "123"
        if args[0] == "scontrol":
            return f"JobName={'fred-resolution-recovery' if duplicate else 'other'} ReqTRES=gres/gpu=1 TimeLimit=02:00:00"
        if args[:2] == ["sbatch", "--parsable"]:
            return "2002"
        return ""
    monkeypatch.setattr(recovery.scheduler, "run", command)
    if duplicate:
        with pytest.raises(ValueError, match="already queued"):
            recovery.submit(tmp_path)
        assert not any(c[0] == "sbatch" for c in calls)
    else:
        recovery.submit(tmp_path)
        actual = next(c for c in calls if c[:2] == ["sbatch", "--parsable"])
        assert "--dependency=afterany:123" in actual and "--kill-on-invalid-dep=yes" in actual
        request = json.loads(next((tmp_path / "data/fred_resolution_recoveries").glob("*/request.json")).read_text())
        assert request["budget"]["total_reserved_gpu_hours"] == 2.25
        assert request["budget"]["new_gpu_hours"] == .25
    text = recovery.JOB.read_text() if recovery.JOB.is_file() else Path(__file__).parents[1].joinpath("jobs/recover_scene_resolution.sbatch").read_text()
    assert "#SBATCH --time=00:15:00" in text and recovery.GPU_SECONDS == 900
    assert "#SBATCH --gres=gpu:1" in text and "#SBATCH --ntasks=1" in text
