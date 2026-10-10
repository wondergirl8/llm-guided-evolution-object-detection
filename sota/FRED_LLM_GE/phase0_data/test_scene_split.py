"""Group-boundary regressions and isolated synthetic scene-evidence fixtures."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sota.FRED_LLM_GE.archive.phase0_data_old.provenance import sha256_file, stable_hash
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import (
    OfficialSplitManifest,
    ProjectSplitManifest,
    load_project_split_manifest,
    validate_project_split,
)
from sota.FRED_LLM_GE.phase0_data.check_scene_split import verify_scene_evidence


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "configs/phase0/project_split_scene_groups_infrastructure_v1.json"
BASE = ROOT / "configs/phase0/project_split_bringup_v1.json"


def reference_official():
    # Offline membership reference only; live official JSON is checked on ICE.
    base = load_project_split_manifest(BASE)
    return OfficialSplitManifest("a" * 40, base.train + base.validation,
                                 base.held_out_test, "train-hash", "test-hash", "hash", "now")


def test_approved_real_membership_has_whole_scene_groups_and_preserves_test():
    project = load_project_split_manifest(PROJECT)
    validate_project_split(project, reference_official())
    groups = dict(project.recording_groups)
    assert len(project.train) == 129 and len(project.validation) == 43
    assert set(project.validation) == set(groups["city_skyline"]) | set(groups["arched_hall"])
    assert "8" in project.train and "230" in project.train
    assert project.held_out_test == load_project_split_manifest(BASE).held_out_test
    raw = json.loads(PROJECT.read_text())
    assert raw["research_split_frozen"] is False
    assert "BRINGUP" in project.approval_reference


def test_reassigning_one_skyline_sequence_is_rejected_even_with_exact_coverage():
    project = load_project_split_manifest(PROJECT)
    changed = replace(project, train=project.train + ("52",),
                      validation=tuple(s for s in project.validation if s != "52"))
    with pytest.raises(ValueError, match="crosses a split"):
        validate_project_split(changed, reference_official())


@pytest.mark.parametrize("groups, message", [
    ((), "requires explicit"),
    ((("a", ("0",)),), "cover development"),
    ((("a", ("0",)), ("b", ("0", "1"))), "disjoint"),
    ((("a", ("0", "2")), ("b", ("1",))), "non-development"),
    ((("a", ("0",)), ("a", ("1",))), "unique non-empty names"),
    ((("a", ("0", "0")), ("b", ("1",))), "duplicate"),
    ((("a", ()), ("b", ("0", "1"))), "non-empty and disjoint"),
])
def test_invalid_group_contract_fails(groups, message):
    project = ProjectSplitManifest("v1", "recording_group", ("0",), ("1",),
                                   ("2",), "approved", groups)
    official = OfficialSplitManifest("a" * 40, ("0", "1"), ("2",), "x", "y", "z", "now")
    with pytest.raises(ValueError, match=message):
        validate_project_split(project, official)


def test_legacy_manifest_hash_is_preserved_and_groups_change_new_identity():
    project = load_project_split_manifest(BASE)
    assert project.content_hash == stable_hash({
        "version": project.version, "grouping_unit": project.grouping_unit,
        "train": project.train, "validation": project.validation,
        "held_out_test": project.held_out_test, "approval_reference": project.approval_reference,
    })
    grouped = load_project_split_manifest(PROJECT)
    changed = replace(grouped, recording_groups=tuple(
        (name + "-different-identity", members) for name, members in grouped.recording_groups))
    assert grouped.content_hash != changed.content_hash


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def synthetic_evidence(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    base = tmp_path / "base.json"
    write_json(base, {"version": "old", "grouping_unit": "sequence", "train": ["0"],
                      "validation": ["1"], "held_out_test": ["2"], "approval_reference": "old"})
    context = {"project_sha256": sha256_file(base)}
    records = []
    for seq, side in (("0", "train"), ("1", "validation")):
        # These bytes test integrity checks only, not real image decoding.
        thumbnail = audit / f"{seq}.jpg"
        thumbnail.write_bytes(f"synthetic fixture {seq}".encode())
        record = {"schema_version": "fred-development-scene-snapshots-v1", "status": "complete",
                  "sequence_id": seq, "project_split": side, "context": context,
                  "paired_frame_count": 1, "snapshots": [{"frame_index": 0,
                  "thumbnail": thumbnail.name, "thumbnail_sha256": sha256_file(thumbnail)}]}
        write_json(audit / f"sequence_{seq}.json", record)
        records.append(record)
    summary = {"context": context, "complete": True, "missing_sequence_ids": [],
               "split_changed": False, "phase0_frozen": False, "expected_sequences": 2,
               "inspected_sequences": 2, "records": records}
    write_json(audit / "summary.json", summary)
    write_json(audit / "resume_receipt.json", {"synthetic": True})
    project = tmp_path / "project.json"
    write_json(project, {"version": "test", "grouping_unit": "recording_group",
        "approval_scope": "infrastructure_branch_working_split", "research_split_frozen": False,
        "approval_reference": "DG-P0-02-INFRASTRUCTURE-BRINGUP-2026-10-10",
        "train": ["0"], "validation": ["1"], "held_out_test": ["2"],
        "recording_groups": {"a": ["0"], "b": ["1"]},
        "counts": {"train_sequences": 1, "validation_sequences": 1,
                   "train_paired_frames": 1, "validation_paired_frames": 1},
        "scene_review": {"context": context, "base_project_split": base.name,
            "summary_sha256": sha256_file(audit / "summary.json"),
            "receipt_sha256": sha256_file(audit / "resume_receipt.json"),
            "group_evidence": {"a": {"sequence_count": 1, "paired_frame_count": 1},
                               "b": {"sequence_count": 1, "paired_frame_count": 1}}}})
    return project, audit, tmp_path


def test_existing_evidence_is_verified_without_rewriting(synthetic_evidence):
    project, audit, workspace = synthetic_evidence
    before = {p.name: p.read_bytes() for p in audit.iterdir()}
    result = verify_scene_evidence(project, audit, workspace=workspace)
    assert result["groups"] == 2 and result["held_out_test_sequences"] == 1
    assert before == {p.name: p.read_bytes() for p in audit.iterdir()}


@pytest.mark.parametrize("filename, message", [
    ("summary.json", "hash mismatch"), ("resume_receipt.json", "hash mismatch"),
    ("0.jpg", "modified scene thumbnail"), ("sequence_0.json", "stale or incomplete"),
])
def test_modified_scene_evidence_is_rejected(synthetic_evidence, filename, message):
    project, audit, workspace = synthetic_evidence
    if filename == "sequence_0.json":
        record = json.loads((audit / filename).read_text())
        record["project_split"] = "validation"
        write_json(audit / filename, record)
    else:
        with (audit / filename).open("ab") as f:
            f.write(b"changed")
    with pytest.raises(ValueError, match=message):
        verify_scene_evidence(project, audit, workspace=workspace)


def test_wrong_scene_group_counts_are_rejected(synthetic_evidence):
    project, audit, workspace = synthetic_evidence
    raw = json.loads(project.read_text())
    raw["scene_review"]["group_evidence"]["a"]["paired_frame_count"] = 10
    write_json(project, raw)
    with pytest.raises(ValueError, match="reviewed counts"):
        verify_scene_evidence(project, audit, workspace=workspace)
