"""Offline acceptance: cache immutability, exact cohort, export, jobs and failure gates."""
import json
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sota.FRED_LLM_GE.data import scene_coverage_export as exporter
from sota.FRED_LLM_GE.data.cached_source import CachedOnlyFredSource
from sota.FRED_LLM_GE.archive.phase0_data_old import fred_api, cli
from sota.FRED_LLM_GE.archive.phase0_data_old.annotation_policy import POLICY_VERSION, APPROVAL_REFERENCE
from sota.FRED_LLM_GE.archive.phase0_data_old.cache import BoundedCache
from sota.FRED_LLM_GE.archive.phase0_data_old.schema import RemoteObject
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import load_project_split_manifest
from sota.FRED_LLM_GE.phase1_detection import cached_coverage_check as workflow
from sota.FRED_LLM_GE.phase0_data import prepare_cached_coverage as preparation


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_real_prepared_cache_read_never_downloads_evicts_or_touches_metadata(tmp_path, monkeypatch):
    archive = tmp_path / "0.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("0/RGB/0.jpg", b"rgb")
        z.writestr("0/Event/Frames/0.png", b"event")
        z.writestr("0/coordinates.txt", b"0.033333: 1,2,3,4,1,drone\n")
    cfg = SimpleNamespace(source=SimpleNamespace(provider="huggingface_hub", dataset_id="owner/fred",
                         revision="a" * 40), schema_version="schema-v1")
    root = tmp_path / "cache"
    warm = object.__new__(fred_api.HFFredSource)
    warm.config, warm.cache = cfg, BoundedCache(root, 100_000)
    remote = RemoteObject(f"hf://datasets/owner/fred@{'a' * 40}/train/0.zip", "train/0.zip", archive.stat().st_size)

    def copy(destination):
        path = destination / "0.zip"
        path.write_bytes(archive.read_bytes())
        return path

    entry = warm.cache.get_or_create(warm.cache_identity(remote), expected_size_bytes=remote.size_bytes, builder=copy)
    warm.materialize_object = lambda _: entry
    record = SimpleNamespace(sequence_id="0", archive=remote)
    prepared = warm.prepare_sequence("0", record)
    before = tree_bytes(root)
    monkeypatch.setattr(fred_api, "_import_huggingface_hub", lambda: pytest.fail("provider import"))
    source = CachedOnlyFredSource(cfg, cache=BoundedCache(root, 100_000))
    monkeypatch.setattr(source.cache, "get_or_create", lambda *a, **k: pytest.fail("cache creation"))
    monkeypatch.setattr(source.cache, "reserve", lambda *a, **k: pytest.fail("eviction"))
    assert source.prepare_sequence("0", record).sequence_root == prepared.sequence_root
    with source.activate_prepared_window((("0", record),)):
        with source.open_prepared_sequence("0", record) as handle:
            assert (handle.sequence_root / "Event/Frames/0.png").read_bytes() == b"event"
    for operation in (source.verify_revision, source.list_objects, lambda: source.materialize_object(remote), source.cleanup_cache):
        with pytest.raises(RuntimeError, match="forbidden"):
            operation()
    assert tree_bytes(root) == before
    (prepared.sequence_root / "Event/Frames/0.png").unlink()
    missing = CachedOnlyFredSource(cfg, cache=BoundedCache(root, 100_000))
    with pytest.raises(fred_api.PreparedSequenceUnavailable):
        missing.prepare_sequence("0", record)


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.sqlite"
    project = load_project_split_manifest(exporter.DEFAULT_PROJECT)
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE metadata(key TEXT, value_json TEXT);"
                         "CREATE TABLE samples(sample_id TEXT, sequence_id TEXT, frame_index INT, project_split TEXT);"
                         "CREATE TABLE annotations(sample_id TEXT);")
        metadata = {"project_split_content_hash": project.content_hash,
                    "project_split_version": project.version, "annotation_policy": POLICY_VERSION,
                    "validation_status": "passed_annotation_policy_bringup_split_non_freeze",
                    "dataset_revision": "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd",
                    "fred_repository_revision": "2bf89c5376eda528431b62d6c60f2c13d8f95ab4",
                    "project_split_approval_reference": project.approval_reference,
                    "annotation_policy_approval_reference": APPROVAL_REFERENCE}
        db.executemany("INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v)) for k, v in metadata.items()])
        for sequence, side in exporter.selected_sequences().items():
            db.executemany("INSERT INTO samples VALUES (?,?,?,?)",
                           [(f"{sequence}:{i}", sequence, i, side) for i in range(514)])
            db.executemany("INSERT INTO annotations VALUES (?)", [(f"{sequence}:{i}",) for i in range(1, 514)])
    return path


@pytest.mark.parametrize("mutation", [
    "DELETE FROM samples WHERE sequence_id='187'",
    "UPDATE samples SET project_split='train' WHERE sequence_id='187'",
    "INSERT INTO samples VALUES ('21:0','21',0,'held_out_test')",
    "UPDATE metadata SET value_json='\"stale\"' WHERE key='project_split_content_hash'",
])
def test_fixed_cohort_rejects_missing_wrong_split_test_or_stale_metadata(manifest, mutation):
    assert len(exporter.validate_manifest(manifest)) == 11
    with sqlite3.connect(manifest) as db:
        db.execute(mutation)
    with pytest.raises(ValueError):
        exporter.validate_manifest(manifest)


def test_real_png_export_counts_lineage_and_original_contract(manifest, monkeypatch):
    from sota.FRED_LLM_GE.archive.phase0_data_old import dataset, inventory
    from sota.FRED_LLM_GE.data import cached_source, scene_diagnostic_export
    with pytest.raises(ValueError, match="exactly one"):
        scene_diagnostic_export.validate_manifest(manifest)
    records = {s: SimpleNamespace(sequence_id=s) for s in exporter.selected_sequences()}
    monkeypatch.setattr(inventory, "load_inventory", lambda _: SimpleNamespace(
        dataset_revision="980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd", inventory_hash="synthetic", by_sequence=lambda: records))
    monkeypatch.setattr(fred_api, "HFFredSource", lambda _: pytest.fail("network-capable export source"))
    monkeypatch.setattr(cached_source, "CachedOnlyFredSource", lambda _: SimpleNamespace(
        prepare_sequence=lambda *_: None, activate_prepared_window=lambda *_: nullcontext()))

    class SyntheticDataset:
        def __init__(self, *, project_split, **kwargs):
            with sqlite3.connect(manifest) as db:
                self.rows = db.execute("SELECT sequence_id, frame_index FROM samples WHERE project_split=? "
                                       "ORDER BY CAST(sequence_id AS INTEGER), frame_index", (project_split.value,)).fetchall()

        def __getitem__(self, index):
            sequence, frame = self.rows[index]
            return SimpleNamespace(sequence_id=sequence, frame_index=frame, event=Image.new("RGB", (16, 16)),
                                   annotations=[SimpleNamespace(box_xyxy=(-1, 2, 8, 10))])

        def close(self):
            pass

    monkeypatch.setattr(dataset, "FREDDataset", SyntheticDataset)
    output = manifest.parent / "export"
    exporter.export(manifest, manifest.parent / "inventory.json", output)
    result = exporter.verify_export(manifest.parent)
    assert result["counts"] == {"train": 3584, "validation": 2048}
    lineage = [json.loads(line) for line in (output / "label_provenance.jsonl").read_text().splitlines()]
    assert len(lineage) == len({r["sample_id"] for r in lineage}) == 5632
    assert all(r["targets"][0]["source_box_xyxy"][0] == -1 for r in lineage)
    assert all(r["targets"][0]["label_box_xyxy"][0] == 0 for r in lineage)
    with Image.open(output / "images/val/187_00000513.png") as im:
        assert im.size == (16, 16)
    with pytest.raises(FileExistsError):
        exporter.export(manifest, manifest.parent / "inventory.json", output)
    (output / "images/val/187_00000513.png").unlink()
    with pytest.raises(ValueError, match="coverage"):
        exporter.verify_export(manifest.parent)


def test_cli_explicitly_selects_cached_only_for_inspection_and_manifest(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.data import cached_source
    marker = RuntimeError("cache-only source selected")
    monkeypatch.setattr(cli, "_load_build_inputs", lambda _: (
        SimpleNamespace(manifest_root=tmp_path, validation_root=tmp_path),
        SimpleNamespace(by_sequence=lambda: {"3": object()}), object(), object()))
    monkeypatch.setattr(cli, "validate_policy_source", lambda *a: None)
    monkeypatch.setattr(cli, "_source", lambda _: pytest.fail("default provider source selected"))

    def stop(config):
        raise marker

    monkeypatch.setattr(cached_source, "CachedOnlyFredSource", stop)
    for arguments, command in ((["inspect-sequence", "3"], cli.command_inspect_sequence),
                               (["build-manifest", "--sequence", "3"], cli.command_build_manifest)):
        args = cli.build_parser().parse_args(arguments + ["--cached-only", "--inventory", "i",
                       "--official-split", "o", "--project-split", "p"])
        with pytest.raises(RuntimeError, match="cache-only source selected"):
            command(args)


@pytest.mark.parametrize("failure", [None, "duplicate", "gpu_submission"])
def test_submission_two_job_budget_dependencies_and_cancellation(tmp_path, monkeypatch, failure):
    selected = exporter.selected_sequences()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(workflow, "reference_inputs", lambda *a: {"input_sha256": {}})
    monkeypatch.setattr(workflow, "sha256", lambda _: "hash")
    monkeypatch.setattr(workflow, "cached_readiness", lambda: {"selected_sequences": selected})
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
            name = "fred-coverage-gpu" if failure == "duplicate" else "other"
            return f"JobName={name} ReqTRES=gres/gpu=1,gres/gpu:h100=1 TimeLimit=02:00:00"
        if command[:2] == ["sbatch", "--parsable"]:
            if any("--dependency=" in v for v in command):
                if failure == "gpu_submission":
                    raise subprocess.CalledProcessError(1, command)
                return "2002"
            return "2001"
        return ""

    monkeypatch.setattr(workflow.scheduler, "run", run)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            workflow.submit(tmp_path, "scene_check_6141846")
        if failure == "duplicate":
            assert not any(c[0] == "sbatch" for c in calls)
        else:
            assert ["scancel", "2001"] in calls
    else:
        workflow.submit(tmp_path, "scene_check_6141846")
        gpu = next(c for c in calls if any("--dependency=" in v for v in c))
        assert "--dependency=afterok:2001,afterany:123" in gpu
        assert "--kill-on-invalid-dep=yes" in gpu
        record = json.loads(next((tmp_path / "data/fred_coverage_checks").glob("*/jobs.json")).read_text())
        assert record["budget"]["total_reserved_gpu_hours"] == 4
        assert len([c for c in calls if c[:2] == ["sbatch", "--test-only"]]) == 2


def test_missing_cache_stops_before_any_job_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.scheduler, "run", lambda c: "fred-yolo11-infrastructure" if "branch" in c else ".git/absent")
    monkeypatch.setattr(workflow, "reference_inputs", lambda *a: {})
    monkeypatch.setattr(workflow, "cached_readiness", lambda: (_ for _ in ()).throw(ValueError("missing prepared cache")))
    with pytest.raises(ValueError, match="missing prepared cache"):
        workflow.submit(tmp_path, "scene_check_6141846")


def test_job_envelopes_match_two_hour_single_gpu_reservation():
    cpu = (workflow.JOBS / "prepare_cached_coverage.sbatch").read_text()
    gpu = (workflow.JOBS / "train_cached_coverage.sbatch").read_text()
    assert "#SBATCH --time=02:00:00" in cpu and "#SBATCH --gres=" not in cpu
    assert "#SBATCH --time=02:00:00" in gpu and "#SBATCH --gres=gpu:1" in gpu
    assert all("HF_HUB_OFFLINE=1" in text and "git diff --quiet" in text for text in (cpu, gpu))


@pytest.mark.parametrize("missing", [False, True])
def test_cache_readiness_does_not_treat_archive_entries_as_prepared(tmp_path, monkeypatch, missing):
    selected = exporter.selected_sequences()
    for sequence in selected:
        root = tmp_path / "entries" / sequence
        root.mkdir(parents=True)
        record = {"identity": {"provider": "huggingface_hub" if missing and sequence == "187" else "phase0_prepared_sequence",
                              "dataset_id": "owner/fred", "revision": "a" * 40},
                  "entry_metadata": {"sequence_id": sequence}}
        (root / "entry.json").write_text(json.dumps(record))
    config = SimpleNamespace(cache=SimpleNamespace(root=tmp_path),
                             source=SimpleNamespace(dataset_id="owner/fred", revision="a" * 40))
    monkeypatch.setattr(preparation, "load_config", lambda _: config)
    monkeypatch.setattr(preparation, "load_inventory", lambda _: SimpleNamespace(dataset_id="owner/fred", dataset_revision="a" * 40))
    if missing:
        with pytest.raises(ValueError, match="187.*no downloads"):
            preparation.cached_readiness()
    else:
        assert preparation.cached_readiness()["selected_sequences"] == selected


@pytest.mark.parametrize("failure", [None, "timeout", "short_history", "wrong_config", "mutated_source", "missing_prediction"])
def test_gpu_success_requires_complete_history_all_predictions_and_unchanged_inputs(tmp_path, monkeypatch, failure):
    from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11 import train_eval as trainer
    from sota.FRED_LLM_GE.phase1_detection import fitness, review_scene_checkpoint as reviewer
    selected = exporter.selected_sequences()
    output = tmp_path / "output"
    (output / "export/images/train").mkdir(parents=True)
    image = output / "export/images/train/3_00000001.png"
    image.write_bytes(b"synthetic image bytes")
    for name in ("manifest.sqlite", "manifest_validation.json", "preparation.json",
                 "export/source.json", "export/data.yaml", "export/label_provenance.jsonl"):
        (output / name).write_text("synthetic provenance")
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"pretrained bytes")
    config = {"epochs": 50, "fraction": 1.0, "batch": 2, "imgsz": 640, "seed": 0,
              "deterministic": True, "gene_id": "seed", "amp": False, "optimizer": "AdamW",
              "lr0": 0.0002, "momentum": 0.9, "warmup_bias_lr": 0.0}
    metrics = {"mAP50": .8, "mAP50_95": .4, "params": 10}
    workflow.write_json(output / "request.json", {"purpose": exporter.PURPOSE, "cohort": exporter.COHORT,
            "code_revision": "submitted", "input_sha256": {str(weights): workflow.sha256(weights)},
            "cached_readiness": {"selected_sequences": selected}, "starting_weights": str(weights),
            "training_config": config, "reference_metrics": metrics})
    monkeypatch.setattr(workflow, "ROOT", tmp_path / "seed")
    monkeypatch.setattr(workflow.scheduler, "run", lambda _: "submitted")
    monkeypatch.setenv("SLURM_JOB_ID", "2001")
    monkeypatch.setattr(workflow.importlib.metadata, "version", lambda _: "8.4.165")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="synthetic", cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda _: "synthetic GPU")))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: object()))
    monkeypatch.setattr(workflow, "verify_export", lambda _: {"selected_sequences": selected,
                       "counts": {"train": 3584, "validation": 2048}})
    monkeypatch.setattr(trainer, "validate_data", lambda _: None)
    history = {"status": "finite", "epochs_completed": 49 if failure == "short_history" else 50}
    monkeypatch.setattr(trainer, "validate_training_history", lambda _: history)
    images = [Path(f"{s}_{i:08d}.png") for s in ("52", "180", "186", "187") for i in range(512)]
    monkeypatch.setattr(trainer, "val_images", lambda _: (images, output / "export/labels/val"))
    monkeypatch.setattr(trainer, "predict_frames", lambda *a: [
        {"sample_id": p.stem, "prediction": {"boxes": []}} for p in
        (images[:-1] if failure == "missing_prediction" else images)])
    monkeypatch.setattr(fitness, "evaluate_frames", lambda *a, **kw: SimpleNamespace(map50=.8, map50_95=.4, params=10))
    monkeypatch.setattr(reviewer, "preview", lambda items, frames, destination: destination.write_text("synthetic preview"))
    calls = []

    def train(**options):
        calls.append(options)
        if failure == "timeout":
            raise RuntimeError("synthetic timeout")
        if failure == "mutated_source":
            image.write_bytes(b"changed")
        seed = workflow.ROOT
        (seed / "results").mkdir(parents=True)
        checkpoint = seed / "trained_models/cached_coverage_640_2001.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"new checkpoint")
        workflow.write_json(seed / "results/cached_coverage_640_2001_evaluation.json", {
            "metrics": metrics, "num_frames": 2048, "metadata": {
                "training_history": history, "checkpoint_identity": workflow.sha256(checkpoint),
                "training_config": {**config, "imgsz": 960 if failure == "wrong_config" else 640}}})

    monkeypatch.setattr(trainer, "run", train)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            workflow.execute(output)
        assert not (output / "summary.json").exists()
    else:
        workflow.execute(output)
        report = json.loads((output / "summary.json").read_text())
        assert report["status"] == "completed"
        assert report["per_scene_validation"]["arched_hall"]["num_frames"] == 1536
        assert report["per_scene_validation"]["city_skyline"]["num_frames"] == 512
        assert len(report["per_recording_validation"]) == 5
        assert len(list(output.glob("sequence_*_preview.png"))) == 4
        # All 2048 no-detection frames survive export and evaluation coverage.
        assert len(json.loads((output / "predictions.json").read_text())["frames"]) == 2048
        assert report["starting_weights_sha256"] == workflow.sha256(weights)
        with pytest.raises(FileExistsError):
            workflow.execute(output)
        assert len(calls) == 1
    assert calls[0]["weights"] == str(weights) and calls[0]["imgsz"] == 640
    assert calls[0]["batch"] == 2 and calls[0]["amp"] is False
