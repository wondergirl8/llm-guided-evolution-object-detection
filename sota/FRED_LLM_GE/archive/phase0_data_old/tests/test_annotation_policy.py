"""Policy acceptance checks use invented fixtures; ICE evidence remains separate."""

import contextlib
import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sota.FRED_LLM_GE.archive.phase0_data_old.annotation_policy import (
    APPROVAL_REFERENCE, POLICY_VERSION, converted_box, validate_policy_source,
)
from sota.FRED_LLM_GE.archive.phase0_data_old.config import load_config
from sota.FRED_LLM_GE.archive.phase0_data_old.manifest import ManifestWriter, read_manifest_metadata
from sota.FRED_LLM_GE.archive.phase0_data_old.policy_review import review_strict_audit, selected_review_sequences
from sota.FRED_LLM_GE.archive.phase0_data_old.provenance import sha256_file
from sota.FRED_LLM_GE.archive.phase0_data_old.schema import RemoteObject
from sota.FRED_LLM_GE.archive.phase0_data_old.fred_api import ArchiveMemberMetadata
from sota.FRED_LLM_GE.archive.phase0_data_old.inventory import SequenceInventoryRecord
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import OfficialSplitManifest, ProjectSplitManifest
from sota.FRED_LLM_GE.archive.phase0_data_old.validation import inspect_sequence, write_validation_report
from sota.FRED_LLM_GE.archive.phase0_data_old.visualization import HeldOutVisualizationError, render_annotation_overlay
from sota.FRED_LLM_GE.data.yolo_export import export_subset, yolo_labels

from .test_validation import CONFIG, prepared_sequence
from .test_development_audit import report


def test_corner_clipping_does_not_clamp_center_and_size_independently():
    source = (-10, 5, 10, 15)
    target = converted_box(source, 100, 50, POLICY_VERSION)
    assert target["source_box_xyxy"] == source
    assert target["label_box_xyxy"] == (0, 5, 10, 15)
    assert target["normalized_xywh"] == (0.05, 0.2, 0.1, 0.2)
    assert target["clipped"] is True
    sample = SimpleNamespace(event=SimpleNamespace(size=(100, 50)),
                             annotations=(SimpleNamespace(box_xyxy=source),))
    assert yolo_labels(sample, POLICY_VERSION) == "0 0.05000000 0.20000000 0.10000000 0.20000000\n"
    with pytest.raises(ValueError, match="outside"):
        yolo_labels(sample)


@pytest.mark.parametrize("box", [(-10, 1, 0, 5), (100, 1, 110, 5), (1, 1, 1, 4),
                                  (float("nan"), 1, 5, 8), (1, 1, 1.000000001, 5)])
def test_policy_keeps_invalid_or_unrepresentable_targets_blocking(box):
    with pytest.raises(ValueError):
        converted_box(box, 100, 50, POLICY_VERSION)


def inspect_fixture(sequence_root, tmp_path, sequence_id="225", timestamp="0.0", source_line=1):
    for path in list((sequence_root / "RGB").iterdir()) + list((sequence_root / "Event" / "Frames").iterdir()):
        path.rename(path.with_name(path.name.replace("Video_0_", f"Video_{sequence_id}_")))
    orphan = f"{timestamp}: -2, 2, 10, 12, 7, drone\n"
    paired = "0.033333: -2, 2, 10, 12, 7, drone\n"
    (sequence_root / "coordinates.txt").write_text(orphan + paired if source_line == 1 else paired + orphan)
    official = OfficialSplitManifest("x", (sequence_id,), ("999",), "a", "b", "c", "now")
    project = ProjectSplitManifest("v1", "sequence", (sequence_id,), (), ("999",), "DG-P0-02")
    return dict(config=load_config(CONFIG, workspace_root=tmp_path),
                inventory_record=SequenceInventoryRecord(sequence_id, "train", RemoteObject("hf://archive", f"train/{sequence_id}.zip")),
                prepared_sequence=prepared_sequence(sequence_root), official_split=official, project_split=project)


@pytest.mark.parametrize("sequence_id", ["225", "230"])
def test_approved_orphans_are_preserved_separately_from_paired_boxes(sequence_root, tmp_path, sequence_id):
    kwargs = inspect_fixture(sequence_root, tmp_path, sequence_id)
    strict = inspect_sequence(**kwargs)
    approved = inspect_sequence(**kwargs, annotation_policy=POLICY_VERSION)
    assert not strict.is_valid
    assert approved.is_valid
    assert approved.samples == strict.samples  # Canonical paired coordinates are unmodified.
    assert len(approved.unpaired_annotations) == 1
    assert approved.samples[0].annotations[0].box_xyxy == (-2, 2, 10, 12)
    path = tmp_path / "manifest.sqlite"
    with ManifestWriter(path, {"annotation_policy": POLICY_VERSION}) as writer:
        for sample in approved.samples:
            writer.add_sample(sample)
        writer.add_unpaired_annotation(sequence_id, approved.unpaired_annotations[0], POLICY_VERSION)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT x1 FROM annotations").fetchone()[0] == -2
        raw, policy = db.execute("SELECT annotation_json, policy_version FROM unpaired_annotations").fetchone()
    assert json.loads(raw)["box_xyxy"] == [-2, 2, 10, 12]
    assert policy == POLICY_VERSION
    assert read_manifest_metadata(path)["unpaired_annotation_count"] == 1
    report_path = tmp_path / "report.json"
    write_validation_report([approved], report_path, context={"annotation_policy": POLICY_VERSION})
    assert json.loads(report_path.read_text())["unpaired_annotations"][0]["sequence_id"] == sequence_id


@pytest.mark.parametrize("sequence_id,timestamp,source_line", [("226", "0.0", 1), ("225", "0.01", 1), ("225", "0.0", 2)])
def test_other_unmatched_records_remain_blocking(sequence_root, tmp_path, sequence_id, timestamp, source_line):
    result = inspect_sequence(**inspect_fixture(sequence_root, tmp_path, sequence_id, timestamp, source_line),
                              annotation_policy=POLICY_VERSION)
    assert not result.is_valid
    assert result.unpaired_annotations == ()
    assert any(f.code == "annotation.unmatched_timestamp" and f.severity == "error" for f in result.findings)


def test_approval_is_pinned_and_cannot_be_combined_with_bringup(sequence_root, tmp_path):
    kwargs = inspect_fixture(sequence_root, tmp_path)
    config = kwargs["config"]
    with pytest.raises(ValueError, match="audited FRED revisions"):
        validate_policy_source(replace(config, source=replace(config.source, revision="a" * 40)), POLICY_VERSION)
    with pytest.raises(ValueError, match="timestamp policy"):
        validate_policy_source(replace(config, timestamp=replace(config.timestamp, frame_index_offset=0)), POLICY_VERSION)
    with pytest.raises(ValueError, match="choose"):
        inspect_sequence(**kwargs, annotation_policy=POLICY_VERSION, allow_partial_out_of_bounds=True)


def test_approved_timestamp_does_not_hide_a_fully_outside_unpaired_box(sequence_root, tmp_path):
    kwargs = inspect_fixture(sequence_root, tmp_path)
    path = sequence_root / "coordinates.txt"
    path.write_text(path.read_text().replace("0.0: -2, 2, 10, 12", "0.0: 40, 2, 50, 12"))
    inspection = inspect_sequence(**kwargs, annotation_policy=POLICY_VERSION)
    assert not inspection.is_valid
    assert any(f.code == "annotation.unpaired_box_no_image_overlap" for f in inspection.findings)


def test_held_out_visual_guard_still_applies_with_approved_policy(tmp_path):
    with pytest.raises(HeldOutVisualizationError):
        render_annotation_overlay(Image.new("RGB", (32, 24)), (), project_split="held_out_test",
                                  destination=tmp_path / "forbidden.png", annotation_policy=POLICY_VERSION)
    assert not (tmp_path / "forbidden.png").exists()


def test_manifest_loader_and_both_export_splits_keep_box_lineage(sequence_root, tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.archive.phase0_data_old import fred_api, inventory as inventory_module

    kwargs = inspect_fixture(sequence_root, tmp_path)
    inspection = inspect_sequence(**kwargs, annotation_policy=POLICY_VERSION)
    config = kwargs["config"]
    metadata = {"dataset_id": config.source.dataset_id, "dataset_revision": config.source.revision,
                "schema_version": config.schema_version, "fred_repository_revision": config.fred_repository.revision,
                "annotation_policy": POLICY_VERSION, "annotation_policy_approval_reference": APPROVAL_REFERENCE,
                "project_split_approval_reference": "BRINGUP test fixture only"}
    manifest = tmp_path / "manifest.sqlite"
    with ManifestWriter(manifest, metadata) as writer:
        writer.add_sample(inspection.samples[0])
        writer.add_sample(replace(inspection.samples[0], sample_id="230:0", sequence_id="230",
                                  project_split=inspection.samples[0].project_split.__class__.VALIDATION))
        writer.add_unpaired_annotation("225", inspection.unpaired_annotations[0], POLICY_VERSION)
    record = kwargs["inventory_record"]
    inventory = SimpleNamespace(dataset_id=config.source.dataset_id, dataset_revision=config.source.revision,
                                inventory_hash="fixture-hash", by_sequence=lambda: {"225": record, "230": record})

    class FakeSource:
        def __init__(self, config):
            self.config = config
        def prepare_sequence(self, *args):
            pass
        @contextlib.contextmanager
        def activate_prepared_window(self, *args):
            yield
        @contextlib.contextmanager
        def open_prepared_sequence(self, *args):
            yield prepared_sequence(sequence_root)

    monkeypatch.setattr(fred_api, "HFFredSource", FakeSource)
    monkeypatch.setattr(inventory_module, "load_inventory", lambda _: inventory)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{}")
    export = tmp_path / "export"
    export_subset(manifest, inventory_path, export, 1, 1)
    train_label = (export / "labels/train/225_00000000.txt").read_text()
    val_label = (export / "labels/val/230_00000000.txt").read_text()
    assert train_label == val_label == "0 0.15625000 0.29166667 0.31250000 0.41666667\n"
    source = json.loads((export / "source.json").read_text())
    assert source["clipped_label_counts"] == {"train": 1, "validation": 1}
    assert source["unpaired_source_annotation_count"] == 1
    rows = [json.loads(line) for line in (export / "label_provenance.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["targets"][0]["source_box_xyxy"] == [-2, 2, 10, 12]
    assert rows[0]["targets"][0]["label_box_xyxy"] == [0, 2, 10, 12]
    with pytest.raises(FileExistsError):
        export_subset(manifest, inventory_path, export, 1, 1)


def test_review_reuses_hash_linked_reports_and_does_not_hide_other_failures(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.archive.phase0_data_old import policy_review

    ids = ("0", "116", "225", "230")
    inventory = SimpleNamespace(dataset_revision="dataset-rev")
    official = SimpleNamespace(challenging_train=ids, repository_revision="repo-rev")
    project = SimpleNamespace(version="split-v1", train=("0", "225", "230"), validation=("116",))
    monkeypatch.setattr(policy_review, "load_inputs", lambda *args: (inventory, official, project))
    input_paths = [tmp_path / name for name in ("inventory", "official", "project")]
    for path in input_paths:
        path.write_text("{}")
    summary = {"schema_version": "fred-development-audit-summary-v1", "complete": True,
               "dataset_revision": "dataset-rev", "inspected_sequence_count": 4,
               **{key: sha256_file(path) for key, path in zip(
                   ("inventory_sha256", "official_split_sha256", "project_split_sha256"), input_paths)},
               "sequences": []}
    for seq in ids:
        bounds = {"severity": "error", "code": "annotation.out_of_bounds",
                  "message": "box (-2.0, 2.0, 10.0, 12.0) is outside 32x24"}
        findings = [bounds] if seq in ("0", "116") else [
            {"severity": "error", "code": "annotation.unmatched_timestamp", "line_number": 1,
             "message": "annotation timestamp 0.0 matched 0 frames within tolerance 0.000001"}]
        path = tmp_path / f"sequence_{seq}.json"
        path.write_text(json.dumps(report(seq, valid=False, findings=findings)))
        summary["sequences"].append({"sequence_id": seq, "report_path": str(path),
                                     "report_sha256": sha256_file(path),
                                     "project_split": "validation" if seq == "116" else "train",
                                     "out_of_bounds_categories": {"partly_visible": 1} if seq in ("0", "116") else {}})
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    original_bytes = {row["report_path"]: Path(row["report_path"]).read_bytes() for row in summary["sequences"]}
    reviewed = review_strict_audit(summary_path, *input_paths)
    assert reviewed["remaining_blockers"] == []
    assert reviewed["counts"]["paired_extended_box_warning"] == 2
    assert reviewed["counts"]["documented_unpaired_zero_requires_source_review"] == 2
    assert selected_review_sequences(reviewed) == ids
    assert all(Path(path).read_bytes() == content for path, content in original_bytes.items())
    path = Path(summary["sequences"][0]["report_path"])
    changed = json.loads(path.read_text())
    changed["findings"].append({"severity": "error", "code": "image.unreadable", "message": "bad"})
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="modified"):
        review_strict_audit(summary_path, *input_paths)
    summary["sequences"][0]["report_sha256"] = sha256_file(path)
    summary_path.write_text(json.dumps(summary))
    assert len(review_strict_audit(summary_path, *input_paths)["remaining_blockers"]) == 1


def test_review_workflow_writes_real_fixture_overlays_and_failed_checks(tmp_path, monkeypatch):
    from sota.FRED_LLM_GE.archive.phase0_data_old import policy_review

    config = load_config(CONFIG, workspace_root=tmp_path)
    ids = ("0", "116", "225", "230")
    records = {seq: SequenceInventoryRecord(seq, "train", RemoteObject("hf://fixture", f"train/{seq}.zip"))
               for seq in ids}
    official = OfficialSplitManifest(config.fred_repository.revision, ids, ("999",), "a", "b", "c", "now")
    project = ProjectSplitManifest("fixture-v1", "sequence", ("0", "225", "230"), ("116",), ("999",), "BRINGUP fixture")
    inventory = SimpleNamespace(dataset_revision=config.source.revision, by_sequence=lambda: records)
    audit = {"source_summary_sha256": "fixture", "sequence_count": 4, "counts": {}, "remaining_blockers": [],
             "sequences": [{"sequence_id": seq, "project_split": "validation" if seq == "116" else "train",
                            "out_of_bounds_categories": {"partly_visible": 1}} for seq in ids]}
    monkeypatch.setattr(policy_review, "load_inputs", lambda *args: (inventory, official, project))
    monkeypatch.setattr(policy_review, "review_strict_audit", lambda *args: audit)
    for seq in ids:
        root = tmp_path / seq
        (root / "RGB").mkdir(parents=True)
        (root / "Event/Frames").mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (32, 24), "black").save(root / "RGB" / f"Video_{seq}_00_00_00.{index*33333:06d}.jpg")
            token = "" if seq == "116" else "frame_"
            Image.new("RGB", (32, 24), "black").save(root / "Event/Frames" / f"Video_{seq}_{token}{(index+1)*33333}.png")
        orphan = "0.0: -2, 2, 10, 12, 7, fixture drone\n" if seq in ("225", "230") else ""
        (root / "coordinates.txt").write_text(orphan + "0.033333: -2, 2, 10, 12, 7, fixture drone\n"
                                              + "0.066666: 4, 5, 15, 16, 7, fixture drone\n")

    class FakeSource:
        def __init__(self, config):
            self.config = config
        def prepare_sequence(self, *args):
            pass
        @contextlib.contextmanager
        def activate_prepared_window(self, *args):
            yield
        @contextlib.contextmanager
        def open_prepared_sequence(self, seq, record):
            yield SimpleNamespace(sequence_root=tmp_path / seq,
                                  raw_hdf5=ArchiveMemberMetadata(f"{seq}/Event/events.hdf5", 32, 24, 100))

    monkeypatch.setattr(policy_review, "HFFredSource", FakeSource)
    output = tmp_path / "review"
    result = policy_review.run_review(tmp_path / "summary.json", output, CONFIG)
    assert result["status"] == "numerical_checks_passed_visual_review_required"
    assert result["phase0_frozen"] is False
    assert result["evolution_fitness_authorized"] is False
    assert len(result["unpaired_source_annotations"]) == 2
    assert result["clipped_visual_label_count"] == 4
    for sample in result["visual_samples"]:
        for filename in sample["images"].values():
            with Image.open(output / filename) as image:
                assert image.size == (32, 24)
    # A completed output cannot be ambiguously overwritten.
    with pytest.raises(FileExistsError):
        policy_review.run_review(tmp_path / "summary.json", output, CONFIG)
    source = tmp_path / "230/coordinates.txt"
    source.write_text(source.read_text().replace("0.0:", "0.01:"))
    with pytest.raises(ValueError, match="fresh source inspection failed"):
        policy_review.run_review(tmp_path / "summary.json", tmp_path / "failed", CONFIG)
    failed = json.loads((tmp_path / "failed/review.json").read_text())
    assert failed["status"] == "failed"
    assert failed["evolution_fitness_authorized"] is False
