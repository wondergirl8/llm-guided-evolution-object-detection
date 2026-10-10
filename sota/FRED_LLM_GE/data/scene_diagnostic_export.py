"""Bounded scene-diverse engineering export; never a frozen research dataset."""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from sota.FRED_LLM_GE.archive.phase0_data_old.annotation_policy import POLICY_VERSION
from sota.FRED_LLM_GE.archive.phase0_data_old.manifest import read_manifest_metadata
from sota.FRED_LLM_GE.archive.phase0_data_old.provenance import sha256_file
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import load_project_split_manifest
from sota.FRED_LLM_GE.phase0_data.check_scene_split import DEFAULT_PROJECT
from .yolo_export import export_subset

FRAMES_PER_SEQUENCE = 512
PURPOSE = "scene-diverse overnight engineering diagnostic; not a formal baseline or evolution fitness"


def evenly_spaced(items: list[int], count: int) -> list[int]:
    if not 1 <= count <= len(items):
        raise ValueError("not enough eligible frames for the fixed diagnostic budget")
    if count == 1:
        return [items[len(items) // 2]]
    return [items[i * (len(items) - 1) // (count - 1)] for i in range(count)]


def validate_manifest(manifest: Path, project_path: Path = DEFAULT_PROJECT) -> dict[str, str]:
    project = load_project_split_manifest(project_path)
    metadata = read_manifest_metadata(manifest)
    if (metadata.get("project_split_content_hash") != project.content_hash
            or metadata.get("project_split_version") != project.version
            or metadata.get("annotation_policy") != POLICY_VERSION
            or "non_freeze" not in str(metadata.get("validation_status"))):
        raise ValueError("diagnostic requires the approved scene split and annotation policy")
    with closing(sqlite3.connect(f"file:{manifest.resolve()}?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT DISTINCT sequence_id, project_split FROM samples").fetchall()
    selected = {}
    for sequence, side in rows:
        expected = "train" if sequence in project.train else "validation" if sequence in project.validation else None
        if side != expected or expected is None:
            raise ValueError("manifest contains test IDs or incorrect project membership")
        if sequence in selected:
            raise ValueError("sequence occurs on both sides")
        selected[sequence] = side
    for name, members in project.recording_groups:
        if len(set(members) & set(selected)) != 1:
            raise ValueError(f"expected exactly one diagnostic sequence in scene group {name}")
    if len(selected) != len(project.recording_groups):
        raise ValueError("manifest has unexpected scene membership")
    return selected


def scene_indices(manifest: Path, split: str, count: int, policy: str | None) -> dict:
    if policy != POLICY_VERSION or not 1 <= count <= FRAMES_PER_SEQUENCE:
        raise ValueError("unexpected diagnostic policy or frame budget")
    with closing(sqlite3.connect(f"file:{manifest.resolve()}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT s.sequence_id, EXISTS(SELECT 1 FROM annotations a WHERE a.sample_id=s.sample_id) "
            "FROM samples s WHERE s.project_split=? "
            "ORDER BY CAST(s.sequence_id AS INTEGER), s.frame_index", (split,)).fetchall()
    eligible = {}
    for index, (sequence, has_labels) in enumerate(rows):
        eligible.setdefault(sequence, [])
        if has_labels:
            eligible[sequence].append(index)
    if not eligible:
        raise ValueError(f"no diagnostic {split} samples")
    return {sequence: evenly_spaced(indices, count) for sequence, indices in eligible.items()}


def export(manifest: Path, inventory: Path, output: Path) -> Path:
    validate_manifest(manifest)
    return export_subset(manifest, inventory, output, FRAMES_PER_SEQUENCE, FRAMES_PER_SEQUENCE,
                         selector=scene_indices,
                         selection_policy="512_evenly_spaced_annotated_frames_per_scene_representative_v1",
                         purpose=PURPOSE)


def verify_export(directory: Path) -> dict:
    manifest = directory / "manifest.sqlite"
    selected = validate_manifest(manifest)
    source = json.loads((directory / "export/source.json").read_text())
    project = load_project_split_manifest(DEFAULT_PROJECT)
    counts = {side: FRAMES_PER_SEQUENCE * list(selected.values()).count(side)
              for side in ("train", "validation")}
    if (source.get("manifest_sha256") != sha256_file(manifest)
            or source.get("project_split_content_hash") != project.content_hash
            or source.get("purpose") != PURPOSE or source.get("counts") != counts):
        raise ValueError("diagnostic export provenance or counts changed")
    expected = {}
    for side, image_side in (("train", "train"), ("validation", "val")):
        image_dir = directory / "export/images" / image_side
        stems = {p.stem for p in image_dir.glob("*.png")}
        labels = {p.stem for p in (directory / "export/labels" / image_side).glob("*.txt")}
        if len(stems) != counts[side] or labels != stems:
            raise ValueError("diagnostic image/label coverage changed")
        per_sequence = {sequence: 0 for sequence, split in selected.items() if split == side}
        for stem in stems:
            sequence, frame = stem.split("_")
            if sequence not in per_sequence or not frame.isdecimal():
                raise ValueError("unexpected image identity")
            per_sequence[sequence] += 1
        if any(n != FRAMES_PER_SEQUENCE for n in per_sequence.values()):
            raise ValueError("diagnostic sequence frame coverage changed")
        expected[side] = stems
    if expected["train"] & expected["validation"]:
        raise ValueError("diagnostic split overlap")
    return {"selected_sequences": selected, "counts": counts, "purpose": PURPOSE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify_only:
        if args.inventory is None:
            parser.error("--inventory is required for export")
        export(args.directory / "manifest.sqlite", args.inventory, args.directory / "export")
    print(json.dumps(verify_export(args.directory), sort_keys=True))
    print("SCENE DIAGNOSTIC EXPORT VERIFIED; no research freeze")


if __name__ == "__main__":
    main()
