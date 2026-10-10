"""Read-only check of the approved infrastructure split against saved scene evidence.

Uses the archived manifest/scene verifier explicitly for compatibility. This
does not select the team's loader or collect, download or decode source frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sota.FRED_LLM_GE.archive.phase0_data_old.provenance import sha256_file
from sota.FRED_LLM_GE.archive.phase0_data_old.scene_audit import verify_record
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import (
    load_official_split_manifest,
    load_project_split_manifest,
    validate_project_split,
)


DEFAULT_PROJECT = Path("sota/FRED_LLM_GE/configs/phase0/project_split_scene_groups_infrastructure_v1.json")
DEFAULT_AUDIT = Path("data/fred_scene_audit/resume_6140043")
DEFAULT_OFFICIAL = Path("data/fred_phase0/metadata/official_challenging_split.json")


def verify_scene_evidence(project_path: Path, audit: Path, *, workspace: Path = Path(".")) -> dict:
    """Check reviewed bytes and group rollups without changing saved split labels."""
    raw = json.loads(project_path.read_text())
    project = load_project_split_manifest(project_path)
    if (raw.get("approval_scope") != "infrastructure_branch_working_split"
            or raw.get("research_split_frozen") is not False
            or project.approval_reference != "DG-P0-02-INFRASTRUCTURE-BRINGUP-2026-10-10"
            or project.grouping_unit != "recording_group"):
        raise ValueError("expected the approved infrastructure scene-group split")
    evidence = raw["scene_review"]
    for name, key in (("summary.json", "summary_sha256"), ("resume_receipt.json", "receipt_sha256")):
        if sha256_file(audit / name) != evidence[key]:
            raise ValueError(f"scene evidence hash mismatch: {name}")
    summary = json.loads((audit / "summary.json").read_text())
    context = evidence["context"]
    if (summary.get("context") != context or summary.get("complete") is not True
            or summary.get("missing_sequence_ids") or summary.get("split_changed") is not False
            or summary.get("phase0_frozen") is not False):
        raise ValueError("scene summary is incomplete or has a different scientific context")
    base_path = workspace / evidence["base_project_split"]
    if sha256_file(base_path) != context["project_sha256"]:
        raise ValueError("original bring-up split changed since scene collection")
    base = load_project_split_manifest(base_path)
    records = {r["sequence_id"]: r for r in summary["records"]}
    development = set(project.train) | set(project.validation)
    if (len(records) != len(summary["records"]) or set(records) != development
            or development != set(base.train) | set(base.validation)
            or project.held_out_test != base.held_out_test
            or summary.get("expected_sequences") != len(development)
            or summary.get("inspected_sequences") != len(development)):
        raise ValueError("scene evidence must cover development IDs exactly and preserve test IDs")
    for seq, record in records.items():
        # Audit records retain their producing bring-up membership; never relabel
        # them to match this new split or demand their Git revision equal HEAD.
        original_side = "train" if seq in base.train else "validation"
        checked = verify_record(audit / f"sequence_{seq}.json", seq, context, original_side)
        if checked != record:
            raise ValueError(f"scene summary differs from sequence record {seq}")
    if set(evidence["group_evidence"]) != {name for name, _ in project.recording_groups}:
        raise ValueError("group evidence must cover the approved groups exactly")
    for name, members in project.recording_groups:
        group_evidence = evidence["group_evidence"][name]
        if (group_evidence["sequence_count"] != len(members)
                or group_evidence["paired_frame_count"] != sum(records[s]["paired_frame_count"] for s in members)):
            raise ValueError(f"scene group {name} differs from the reviewed counts")
    counts = {
        "train_sequences": len(project.train), "validation_sequences": len(project.validation),
        "train_paired_frames": sum(records[s]["paired_frame_count"] for s in project.train),
        "validation_paired_frames": sum(records[s]["paired_frame_count"] for s in project.validation),
    }
    if counts != raw["counts"]:
        raise ValueError("split counts differ from the reviewed proposal")
    return {**counts, "groups": len(project.recording_groups), "held_out_test_sequences": len(project.held_out_test)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-split", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--scene-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--official-split", type=Path, default=DEFAULT_OFFICIAL)
    args = parser.parse_args(argv)
    raw = json.loads(args.project_split.read_text())
    context = raw["scene_review"]["context"]
    if sha256_file(args.official_split) != context["official_sha256"]:
        raise ValueError("official split differs from the scene-audit input")
    official = load_official_split_manifest(args.official_split)
    if official.repository_revision != context["repository_revision"]:
        raise ValueError("official split repository revision differs from reviewed evidence")
    validate_project_split(load_project_split_manifest(args.project_split), official)
    counts = verify_scene_evidence(args.project_split, args.scene_audit)
    print(json.dumps(counts, sort_keys=True))
    print(f"SCENE SPLIT CHECK COMPLETE: {counts['train_sequences']} train / "
          f"{counts['validation_sequences']} validation; "
          f"{counts['held_out_test_sequences']} held out; no training submitted")


if __name__ == "__main__":
    main()
