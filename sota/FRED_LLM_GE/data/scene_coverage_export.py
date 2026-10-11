"""Fixed, cached recording cohort for broader engineering bring-up at 640px."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from sota.FRED_LLM_GE.archive.phase0_data_old.annotation_policy import POLICY_VERSION
from sota.FRED_LLM_GE.archive.phase0_data_old.manifest import read_manifest_metadata
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import load_project_split_manifest
from sota.FRED_LLM_GE.phase0_data.check_scene_split import DEFAULT_PROJECT
from . import scene_diagnostic_export as original
from .yolo_export import export_subset

COHORT = {"courtyard": ("3", "199", "200"), "outdoor_gardens": ("25", "203", "204"),
          "utility_corridor": ("168",), "city_skyline": ("52",),
          "arched_hall": ("180", "186", "187")}
PURPOSE = "cached recording coverage engineering diagnostic; not a formal baseline or evolution fitness"
SELECTION_POLICY = "512_evenly_spaced_annotated_frames_per_cached_recording_v1"


def selected_sequences():
    project = load_project_split_manifest(DEFAULT_PROJECT)
    groups = dict(project.recording_groups)
    selected = {}
    for group, sequences in COHORT.items():
        if not set(sequences) <= set(groups[group]):
            raise ValueError("fixed cohort differs from approved scene groups")
        for sequence in sequences:
            side = "train" if sequence in project.train else "validation" if sequence in project.validation else None
            if side is None or sequence in selected:
                raise ValueError("test or duplicate sequence in fixed cohort")
            selected[sequence] = side
    return selected


def validate_manifest(manifest):
    project = load_project_split_manifest(DEFAULT_PROJECT)
    metadata = read_manifest_metadata(manifest)
    if (metadata.get("project_split_content_hash") != project.content_hash
            or metadata.get("project_split_version") != project.version
            or metadata.get("annotation_policy") != POLICY_VERSION
            or "non_freeze" not in str(metadata.get("validation_status"))):
        raise ValueError("coverage diagnostic requires approved split/policy and non-freeze status")
    with closing(sqlite3.connect(f"file:{Path(manifest).resolve()}?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT DISTINCT sequence_id, project_split FROM samples").fetchall()
    expected = selected_sequences()
    if len(rows) != len(expected) or dict(rows) != expected:
        raise ValueError("coverage manifest differs from the exact 11-recording cohort")
    return expected


def export(manifest, inventory, output):
    validate_manifest(manifest)
    return export_subset(manifest, inventory, output, 512, 512,
                         selector=original.scene_indices, selection_policy=SELECTION_POLICY,
                         cached_only=True, purpose=PURPOSE)


def verify_export(directory):
    directory = Path(directory)
    source = json.loads((directory / "export/source.json").read_text())
    if source.get("cached_only") is not True or source.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("coverage export was not produced with the fixed cached-only selector")
    return original.verify_export(directory, manifest_validator=validate_manifest, purpose=PURPOSE)
