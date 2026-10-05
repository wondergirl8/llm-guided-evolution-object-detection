"""Invented scene fixtures validate collection/resume, not real split independence."""

import contextlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sota.FRED_LLM_GE.phase0_data import scene_audit
from sota.FRED_LLM_GE.phase0_data.config import load_config
from .test_validation import CONFIG


def test_snapshot_selection_is_deterministic_bounded_and_uses_later_frames():
    assert scene_audit.snapshot_indexes(100) == (24, 49, 74, 94)
    assert scene_audit.snapshot_indexes(1) == (0,)
    assert len(scene_audit.snapshot_indexes(2)) == 1
    with pytest.raises(ValueError):
        scene_audit.snapshot_indexes(0)
    shards = [tuple(range(172))[i::8] for i in range(8)]
    assert sorted(value for shard in shards for value in shard) == list(range(172))


def test_candidate_ranking_is_cross_split_and_retains_manual_review():
    snapshot = lambda h: {"difference_hash": h, "grayscale_stddev": 20}
    records = [{"sequence_id": "225", "project_split": "train", "snapshots": [snapshot("0000000000000000")]},
               {"sequence_id": "230", "project_split": "validation", "snapshots": [snapshot("0000000000000001")]},
               {"sequence_id": "999", "project_split": "held_out_test", "snapshots": [snapshot("0000000000000001")]}]
    candidates = scene_audit.cross_split_candidates(records)
    assert candidates == [{"train_sequence_id": "225", "validation_sequence_id": "230",
                           "minimum_dhash_distance": 1, "decision": "manual_review_required"}]
    records[0]["snapshots"][0]["grayscale_stddev"] = 0
    assert scene_audit.cross_split_candidates(records) == []


def test_collect_resume_and_complete_summary_preserve_split_and_raw_data(sequence_root, tmp_path, monkeypatch):
    config = load_config(CONFIG, workspace_root=tmp_path)
    roots = {"0": sequence_root, "8": tmp_path / "8"}
    shutil.copytree(sequence_root, roots["8"])
    for path in list((roots["8"] / "RGB").iterdir()) + list((roots["8"] / "Event/Frames").iterdir()):
        path.rename(path.with_name(path.name.replace("Video_0_", "Video_8_")))
    official = SimpleNamespace(challenging_train=("0", "8"), challenging_test=("999",))
    project = SimpleNamespace(train=("0",), validation=("8",))
    inventory = SimpleNamespace(by_sequence=lambda: {"0": object(), "8": object()})
    context = {"test_fixture": "invented"}
    monkeypatch.setattr(scene_audit, "audit_inputs", lambda: (config, inventory, official, project, context))
    prepared = []

    class FakeSource:
        def __init__(self, config):
            pass
        def prepare_sequence(self, sequence, record):
            assert sequence != "999"
            prepared.append(sequence)
        @contextlib.contextmanager
        def activate_prepared_window(self, *args):
            yield
        @contextlib.contextmanager
        def open_prepared_sequence(self, sequence, *args):
            yield SimpleNamespace(sequence_root=roots[sequence])

    monkeypatch.setattr(scene_audit, "HFFredSource", FakeSource)
    output = tmp_path / "scene-audit"
    original = (roots["0"] / "coordinates.txt").read_bytes()
    scene_audit.collect(output, 0, 2)
    partial = scene_audit.summarize(output)
    assert not partial["complete"]
    assert partial["missing_sequence_ids"] == ["8"]
    assert partial["overview_pages"] == []
    scene_audit.collect(output, 1, 2, summarize_when_complete=True)
    assert json.loads((output / "summary.json").read_text())["complete"] is True
    summary = scene_audit.summarize(output)
    assert summary["complete"] and summary["inspected_sequences"] == 2
    assert summary["split_changed"] is False
    assert summary["phase0_frozen"] is False
    assert summary["records"][0]["source_drone_class_counts"] == {"test drone": 2}
    assert (output / summary["overview_pages"][0]).is_file()
    scene_audit.collect(output, 0, 2)
    assert prepared == ["0", "8"]
    assert (roots["0"] / "coordinates.txt").read_bytes() == original
    record = json.loads((output / "sequence_0.json").read_text())
    thumbnail = output / record["snapshots"][0]["thumbnail"]
    thumbnail.write_bytes(b"modified")
    with pytest.raises(ValueError, match="modified"):
        scene_audit.collect(output, 0, 2)
    with pytest.raises(ValueError, match="invalid scene-audit shard"):
        scene_audit.collect(output, 8, 8)


def test_held_out_sequence_cannot_be_collected(tmp_path, monkeypatch):
    official = SimpleNamespace(challenging_train=("999",), challenging_test=("999",))
    project = SimpleNamespace(train=(), validation=())
    monkeypatch.setattr(scene_audit, "audit_inputs", lambda: (object(), object(), official, project, {}))
    monkeypatch.setattr(scene_audit, "HFFredSource", lambda _: object())
    with pytest.raises(ValueError, match="held-out"):
        scene_audit.collect(tmp_path / "output", 0, 1)


def test_input_identity_and_split_hashes_are_pinned(tmp_path, monkeypatch):
    config = load_config(CONFIG, workspace_root=tmp_path)
    inventory = SimpleNamespace(dataset_id=config.source.dataset_id, dataset_revision=config.source.revision)
    official = SimpleNamespace(repository_revision=config.fred_repository.revision,
                               train_source_sha256=config.fred_repository.challenging_train_sha256,
                               test_source_sha256=config.fred_repository.challenging_test_sha256)
    monkeypatch.setattr(scene_audit, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(scene_audit, "load_inputs", lambda *args: (inventory, official, object()))
    monkeypatch.setattr(scene_audit, "sha256_file", lambda *args: "fixture-hash")
    monkeypatch.setattr(scene_audit.subprocess, "check_output", lambda *args, **kwargs: "fixture-revision\n")
    assert scene_audit.audit_inputs()[-1]["dataset_revision"] == config.source.revision
    official.train_source_sha256 = "stale"
    with pytest.raises(ValueError, match="split source hashes"):
        scene_audit.audit_inputs()
    inventory.dataset_id = "wrong-dataset"
    with pytest.raises(ValueError, match="source revisions"):
        scene_audit.audit_inputs()
