"""Offline coverage, reproduction and resource checks; no CUDA or source downloads."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sota.FRED_LLM_GE.phase1_detection import review_scene_checkpoint as review
from sota.FRED_LLM_GE.phase1_detection.submit_scene_overnight import check_budget


def test_review_uses_one_job_and_quarter_gpu_hour():
    jobs = [(str(i), "ReqTRES=cpu=4 TimeLimit=UNLIMITED") for i in range(49)]
    result = check_budget(jobs, new_gpu_seconds=900, new_jobs=1)
    assert result["new_gpu_hours"] == 0.25
    with pytest.raises(ValueError, match="50-job"):
        check_budget(jobs + [("50", "ReqTRES=cpu=4 TimeLimit=UNLIMITED")],
                     new_gpu_seconds=900, new_jobs=1)
    gpu = [("1", "ReqTRES=gres/gpu=1 TimeLimit=31:45:00")]
    assert check_budget(gpu, new_gpu_seconds=900, new_jobs=1)["total_reserved_gpu_hours"] == 32
    with pytest.raises(ValueError, match="32 GPU-hour"):
        check_budget(gpu, new_gpu_seconds=901, new_jobs=1)
    with pytest.raises(ValueError, match="positive integers"):
        check_budget([], new_jobs=0)


def test_only_exact_validation_coverage_is_accepted():
    selected = {"3": "train", "52": "validation", "180": "validation"}
    images = [Path(f"{s}_{i:08d}.png") for s in ("180", "52") for i in range(512)]
    assert set(review.group_images(images, selected)) == {"52", "180"}
    for invalid in (images[:-1], images + [images[0]], images + [Path("3_00000000.png")],
                    images + [Path("bad.png")]):
        with pytest.raises(ValueError):
            review.group_images(invalid, selected)


def test_metrics_include_no_detections_and_preserve_combined_order():
    groups = {"52": [Path("52_00000001.png")], "180": [Path("180_00000001.png")]}
    frames = [{"sample_id": s, "prediction": {"boxes": []}} for s in
              ("180_00000001", "52_00000001")]
    calls = []

    def evaluator(actual, *, params, expected_sample_ids):
        calls.append((actual, expected_sample_ids))
        assert len(actual) == len(expected_sample_ids)
        return SimpleNamespace(map50=0, map50_95=0, params=params)

    report = review.grouped_metrics(frames, groups, 10, evaluator)
    assert calls[0] == (frames, ["180_00000001", "52_00000001"])
    assert report["combined"]["num_frames"] == 2 and report["52"]["num_frames"] == 1
    for invalid in (frames[:-1], frames + [frames[0]], frames + [{"sample_id": "3_1"}]):
        with pytest.raises(ValueError, match="predictions"):
            review.grouped_metrics(invalid, groups, 10, evaluator)


def test_reproduction_differences_and_nonfinite_metrics_fail():
    original = {"mAP50": 0.88, "mAP50_95": 0.48, "params": 10}
    review.check_reproduction(dict(original), original)
    for key, value in (("mAP50", 0.89), ("mAP50_95", float("nan")), ("params", 11)):
        with pytest.raises(ValueError):
            review.check_reproduction({**original, key: value}, original)


def test_real_coco_metrics_expose_a_scene_with_all_missed_drones():
    pytest.importorskip("pycocotools")
    from sota.FRED_LLM_GE.phase1_detection.fitness import evaluate_frames
    groups = {"52": [Path("52_00000001.png")], "180": [Path("180_00000001.png")]}
    frames = []
    for sequence in ("180", "52"):
        prediction = ({"boxes": [[1, 1, 8, 8]], "labels": [0], "scores": [0.9]} if sequence == "180"
                      else {"boxes": [], "labels": [], "scores": []})
        frames.append({"sample_id": f"{sequence}_00000001", "width": 10, "height": 10,
                       "target": {"boxes": [[1, 1, 8, 8]], "labels": [0]}, "prediction": prediction})
    report = review.grouped_metrics(frames, groups, 10, evaluate_frames)
    assert report["180"]["mAP50"] == pytest.approx(1)
    assert report["52"]["mAP50"] == 0
    assert 0.49 < report["combined"]["mAP50"] < 0.51


@pytest.mark.parametrize("existing_review", [False, True])
def test_submission_reuses_checkpoint_waits_for_gpu_and_never_trains(tmp_path, monkeypatch, existing_review):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(review, "ROOT", tmp_path / "seed")
    run = "scene_check_6141846"
    data = tmp_path / "data/original"
    data.mkdir(parents=True)
    checkpoint = review.ROOT / "trained_models" / f"{run}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")
    metrics = {"mAP50": 0.88, "mAP50_95": 0.48, "params": 10}
    original = {"metrics": metrics, "metadata": {"checkpoint_identity": review.sha256(checkpoint)}}
    report = review.ROOT / "results" / f"{run}_evaluation.json"
    report.parent.mkdir()
    review.write_json(report, original)
    review.write_json(data / "summary.json", {"status": "completed", "run_id": run, "validation": metrics})
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[:3] == ["git", "branch", "--show-current"]:
            return "fred-yolo11-infrastructure"
        if command[:2] == ["git", "rev-parse"]:
            return ".git/MERGE_HEAD" if "--git-path" in command else "a" * 40
        if command[0] == "squeue":
            return "123"
        if command[0] == "scontrol":
            return (f"JobName={'fred-scene-review' if existing_review else 'other'} "
                    "ReqTRES=gres/gpu=1,gres/gpu:h100=1 TimeLimit=02:00:00")
        if command[:2] == ["sbatch", "--parsable"]:
            return "2001"
        return ""

    monkeypatch.setattr(review.scheduler, "run", fake_run)
    if existing_review:
        with pytest.raises(ValueError, match="already queued"):
            review.submit(data, run)
        assert not any(c[0] == "sbatch" for c in commands)
    else:
        review.submit(data, run)
        submit = next(c for c in commands if c[:2] == ["sbatch", "--parsable"])
        assert "--dependency=afterany:123" in submit
        assert len([c for c in commands if c[:2] == ["sbatch", "--test-only"]]) == 1
        receipt = json.loads(next((tmp_path / "data/fred_checkpoint_reviews").glob("*/job.json")).read_text())
        assert receipt["budget"]["total_reserved_gpu_hours"] == 2.25
        assert receipt["original_report_sha256"] == review.sha256(report)
        assert not any("train_eval.py" in part for c in commands for part in c)


def test_review_outputs_cannot_overwrite_existing_evidence(tmp_path):
    output = tmp_path / "summary.json"
    review.write_json(output, {"status": "completed"})
    with pytest.raises(FileExistsError):
        review.write_json(output, {"status": "changed"})
    assert json.loads(output.read_text())["status"] == "completed"


def test_actual_preview_renderer_produces_a_readable_png(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image
    images, frames = [], []
    for index in range(6):
        image = tmp_path / f"52_{index:08d}.png"
        Image.new("RGB", (100, 60), (20, 20, 30)).save(image)
        images.append(image)
        frames.append({"sample_id": image.stem,
                       "target": {"boxes": [[10, 10, 30, 30]], "labels": [0]},
                       "prediction": {"boxes": [[10, 10, 30, 30], [50, 30, 60, 40]],
                                      "labels": [0, 0], "scores": [0.9, 0.1]}})
    destination = tmp_path / "preview.png"
    review.preview(images, frames, destination)
    with Image.open(destination) as output:
        assert output.width >= 1000 and output.height >= 2000
        output.verify()


@pytest.mark.parametrize("failure", [None, "metric_difference", "input_changed"])
def test_execution_preserves_originals_and_publishes_success_only_after_checks(tmp_path, monkeypatch, failure):
    import sys
    from sota.FRED_LLM_GE.phase1_detection import fitness
    from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as trainer
    from sota.FRED_LLM_GE.data import scene_diagnostic_export as exporter
    seed = tmp_path / "phase1/seeds/yolo11"
    monkeypatch.setattr(review, "ROOT", seed)
    source = tmp_path / "original"
    output = tmp_path / "review"
    (source / "export/labels/val").mkdir(parents=True)
    output.mkdir()
    (seed / "trained_models").mkdir(parents=True)
    (seed / "results").mkdir()
    checkpoint = seed / "trained_models/scene_check_123.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    original_source = source / "export/source.json"
    original_source.write_text('{}')
    images = []
    for sequence in ("180", "52"):
        for index in range(512):
            image = source / "export" / f"{sequence}_{index:08d}.png"
            image.write_bytes(b"synthetic placeholder; image decoding is mocked")
            (source / "export/labels/val" / f"{image.stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            images.append(image)
    (source / "selection.json").write_text(json.dumps({"arched_hall": "180", "city_skyline": "52"}))
    metrics = {"mAP50": 0.88, "mAP50_95": 0.48, "params": 10}
    report = {"metrics": metrics, "metadata": {
        "checkpoint_identity": review.sha256(checkpoint), "code_files_sha256": {},
        "source_record_sha256": review.sha256(original_source), "training_history": {"status": "finite"},
        "training_config": {"imgsz": 640, "batch": 2}}}
    original_report = seed / "results/scene_check_123_evaluation.json"
    review.write_json(original_report, report)
    review.write_json(output / "request.json", {"directory": str(source), "run_id": "scene_check_123",
                      "original_report_sha256": review.sha256(original_report), "code_revision": "original"})
    original_hashes = {p: review.sha256(p) for p in (checkpoint, original_source, original_report)}
    monkeypatch.setattr(review.scheduler, "run", lambda _: "original")
    monkeypatch.setattr(review.importlib.metadata, "version", lambda _: "8.4.165")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True)),
        manual_seed=lambda _: None, use_deterministic_algorithms=lambda *a, **kw: None))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: object()))
    monkeypatch.setattr(exporter, "verify_export", lambda _: {
        "selected_sequences": {"180": "validation", "52": "validation", "3": "train"},
        "purpose": "synthetic offline test"})
    monkeypatch.setattr(trainer, "validate_data", lambda _: None)
    monkeypatch.setattr(trainer, "validate_training_history", lambda _: {"status": "finite"})
    monkeypatch.setattr(trainer, "val_images", lambda _: (images, source / "export/labels/val"))

    def predict(*args):
        if failure == "input_changed":
            images[0].write_bytes(b"changed during inference")
        return [{"sample_id": image.stem, "prediction": {"boxes": []}} for image in images]

    monkeypatch.setattr(trainer, "predict_frames", predict)
    monkeypatch.setattr(fitness, "evaluate_frames", lambda *a, **kw: SimpleNamespace(
        map50=0.9 if failure == "metric_difference" else 0.88, map50_95=0.48, params=10))
    monkeypatch.setattr(review, "preview", lambda images, frames, destination: destination.write_text("synthetic preview"))
    if failure:
        with pytest.raises(ValueError):
            review.execute(output)
        assert not (output / "summary.json").exists()
        assert (output / "metrics.json").exists() and (output / "predictions.json").exists()
    else:
        review.execute(output)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["metrics"]["combined"]["num_frames"] == 1024
        assert summary["metrics"]["52"]["num_frames"] == 512
        assert len(list(output.glob("*_preview.png"))) == 2
    assert all(review.sha256(path) == digest for path, digest in original_hashes.items())
