"""Synthetic resume evidence; these fixtures establish no real split independence."""

import contextlib
import json
import shutil
from types import SimpleNamespace

import pytest

from sota.FRED_LLM_GE.archive.phase0_data_old import resume_scene_audit as resume
from sota.FRED_LLM_GE.archive.phase0_data_old import scene_audit
from sota.FRED_LLM_GE.archive.phase0_data_old.config import load_config
from sota.FRED_LLM_GE.archive.phase0_data_old.provenance import sha256_file
from .test_validation import CONFIG


@pytest.fixture
def partial_audit(sequence_root, tmp_path, monkeypatch):
    config = load_config(CONFIG, workspace_root=tmp_path)
    roots = {"0": sequence_root, "8": tmp_path / "8"}
    shutil.copytree(sequence_root, roots["8"])
    for path in list((roots["8"] / "RGB").iterdir()) + list((roots["8"] / "Event/Frames").iterdir()):
        path.rename(path.with_name(path.name.replace("Video_0_", "Video_8_")))
    official = SimpleNamespace(challenging_train=("0", "8"), challenging_test=("999",))
    project = SimpleNamespace(train=("0",), validation=("8",))
    inventory = SimpleNamespace(by_sequence=lambda: {"0": object(), "8": object()})
    old = {"code_revision": "a" * 40, "scene_audit_source_sha256": sha256_file(scene_audit.__file__),
           "dataset_revision": "pinned-fixture", "repository_revision": "fixture-repo",
           "config_sha256": "fixture-config", "inventory_sha256": "fixture-inventory",
           "official_sha256": "fixture-official", "project_sha256": "fixture-project",
           "sampling_policy": "paired_frames_quartiles_and_95_percent_v1"}
    current = {**old, "code_revision": "b" * 40}
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
    monkeypatch.setattr(scene_audit, "audit_inputs", lambda: (config, inventory, official, project, old))
    source = tmp_path / "original"
    scene_audit.collect(source, 0, 2)
    monkeypatch.setattr(scene_audit, "audit_inputs", lambda: (config, inventory, official, project, current))
    return source, tmp_path / "resumed", current, prepared


def test_resume_collects_only_missing_and_preserves_original_provenance(partial_audit):
    source, output, current, prepared = partial_audit
    before = {path.name: sha256_file(path) for path in source.iterdir()}
    resume.main(["--source", str(source), "--output", str(output)])
    assert prepared == ["0", "8"]
    assert before == {path.name: sha256_file(path) for path in source.iterdir()}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] and summary["inspected_sequences"] == 2
    assert not summary["phase0_frozen"] and not summary["split_changed"]
    copied = json.loads((output / "sequence_0.json").read_text())
    assert copied["context"] == current
    assert copied["producing_context"]["code_revision"] == "a" * 40
    assert copied["reused_from"]["sha256"] == before["sequence_0.json"]
    fresh = json.loads((output / "sequence_8.json").read_text())
    assert fresh["context"] == current and "reused_from" not in fresh
    with pytest.raises(FileExistsError, match="overwrite"):
        resume.stage_records(source, output)


@pytest.mark.parametrize("key", ["dataset_revision", "repository_revision", "config_sha256",
                                "inventory_sha256", "official_sha256", "project_sha256",
                                "scene_audit_source_sha256", "sampling_policy"])
def test_resume_rejects_changed_scientific_inputs_before_creating_output(partial_audit, key):
    source, output, current, _ = partial_audit
    current[key] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        resume.stage_records(source, output)
    assert not output.exists()


def test_resume_checks_thumbnail_integrity_before_creating_output(partial_audit):
    source, output, _, _ = partial_audit
    record = json.loads((source / "sequence_0.json").read_text())
    (source / record["snapshots"][0]["thumbnail"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="modified"):
        resume.stage_records(source, output)
    assert not output.exists()


def test_resume_rejects_held_out_and_overlapping_directories(partial_audit):
    source, output, _, _ = partial_audit
    with pytest.raises(ValueError, match="separate"):
        resume.stage_records(source, source / "child")
    record = json.loads((source / "sequence_0.json").read_text())
    record["sequence_id"] = "999"
    (source / "sequence_999.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unexpected"):
        resume.stage_records(source, output)
    assert not output.exists()


def test_resume_keeps_original_producing_context_across_multiple_resumes(partial_audit):
    source, output, current, _ = partial_audit
    resume.stage_records(source, output)
    current["code_revision"] = "c" * 40
    destination = output.parent / "resumed-again"
    resume.stage_records(output, destination)
    record = json.loads((destination / "sequence_0.json").read_text())
    assert record["context"]["code_revision"] == "c" * 40
    assert record["producing_context"]["code_revision"] == "a" * 40
    assert record["reused_from"]["context"]["code_revision"] == "b" * 40
