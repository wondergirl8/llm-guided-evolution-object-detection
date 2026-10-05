"""Bounded FRED Phase 0 sample export for a YOLO11 bring-up run.

The manifest and approved split remain the source of truth. This command
materializes only the requested samples and prepares only their sequences.
"""

import argparse
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from sota.FRED_LLM_GE.phase0_data.annotation_policy import (
    APPROVAL_REFERENCE, POLICY_VERSION, converted_box, validate_policy, validate_policy_source,
)


def yolo_targets(sample, annotation_policy: str | None = None) -> list[dict]:
    image = sample.event
    if image is None or sample.annotations is None:
        raise ValueError("event image and trusted annotations are required")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("invalid image size")
    targets = []
    for annotation in sample.annotations:
        targets.append({
            **converted_box(annotation.box_xyxy, width, height, annotation_policy),
            "timestamp": getattr(annotation, "timestamp", None),
            "source_line": getattr(annotation, "source_line", None),
            "track_id": getattr(annotation, "track_id", None),
            "original_class": getattr(annotation, "original_class", None),
        })
    return targets


def yolo_labels(sample, annotation_policy: str | None = None) -> str:
    lines = ["0 " + " ".join(f"{value:.8f}" for value in target["normalized_xywh"])
             for target in yolo_targets(sample, annotation_policy)]
    return "\n".join(lines) + ("\n" if lines else "")


def selected_indices(manifest_path: Path, split: str, limit: int,
                     annotation_policy: str | None = None):
    validate_policy(annotation_policy)
    if not 0 < limit <= 128:
        raise ValueError("bring-up sample limits must be from 1 to 128")
    with closing(sqlite3.connect(f"file:{manifest_path.resolve()}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT s.sequence_id, "
            "EXISTS(SELECT 1 FROM annotations a WHERE a.sample_id = s.sample_id), "
            "EXISTS(SELECT 1 FROM annotations a WHERE a.sample_id = s.sample_id "
            "AND (a.x1 < 0 OR a.y1 < 0 OR a.x2 > s.event_width "
            "OR a.y2 > s.event_height)) "
            "FROM samples s WHERE s.project_split = ? "
            "ORDER BY CAST(s.sequence_id AS INTEGER), s.frame_index",
            (split,),
        ).fetchall()
    if not rows:
        raise ValueError(f"manifest contains no {split} samples")
    first_sequence = rows[0][0]
    # Prefer annotated frames so a small bring-up set actually exercises labels.
    # Dataset indices still refer to the full split ordering.
    annotated = [index for index, (sequence_id, has_label, out_of_bounds) in enumerate(rows)
                 if sequence_id == first_sequence and has_label
                 and (annotation_policy or not out_of_bounds)]
    if not annotated:
        raise ValueError(f"{split} sequence {first_sequence} has no eligible annotated frames")
    return {first_sequence: annotated[:limit]}


def export_subset(manifest_path: Path, inventory_path: Path, output: Path,
                  train_limit: int, val_limit: int) -> Path:
    """Export an explicit small project-train/validation subset, atomically."""
    from sota.FRED_LLM_GE.phase0_data.config import load_config
    from sota.FRED_LLM_GE.phase0_data.dataset import FREDDataset
    from sota.FRED_LLM_GE.phase0_data.fred_api import HFFredSource
    from sota.FRED_LLM_GE.phase0_data.inventory import load_inventory
    from sota.FRED_LLM_GE.phase0_data.manifest import read_manifest_metadata
    from sota.FRED_LLM_GE.phase0_data.schema import AccessMode, Modality, ProjectSplit

    manifest_path = manifest_path.resolve()
    inventory_path = inventory_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "phase0" / "default.yaml",
                         workspace_root=root.parent.parent)
    inventory = load_inventory(inventory_path)
    metadata = read_manifest_metadata(manifest_path)
    annotation_policy = POLICY_VERSION if metadata.get("annotation_policy") == POLICY_VERSION else None
    validate_policy_source(config, annotation_policy)
    if annotation_policy and metadata.get("annotation_policy_approval_reference") != APPROVAL_REFERENCE:
        raise ValueError("manifest is missing the DG-P0-04 approval reference")
    if annotation_policy and metadata.get("fred_repository_revision") != config.fred_repository.revision:
        raise ValueError("manifest differs from the approved FRED repository revision")
    if not metadata.get("project_split_approval_reference"):
        raise ValueError("manifest has no approved project split reference")
    if metadata.get("dataset_revision") != inventory.dataset_revision:
        raise ValueError("manifest and inventory revisions differ")
    source = HFFredSource(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fred-yolo11-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        counts = {}
        clipped_counts = {}
        label_counts = {}
        lineage = []
        for split, limit, mode in (("train", train_limit, AccessMode.TRAIN),
                                   ("validation", val_limit, AccessMode.TRUSTED_EVALUATOR)):
            indices_by_sequence = selected_indices(manifest_path, split, limit, annotation_policy)
            dataset = FREDDataset(
                manifest_path=manifest_path, inventory=inventory, source=source,
                project_split=ProjectSplit(split), modality=Modality.EVENT,
                access_mode=mode,
            )
            target_split = "train" if split == "train" else "val"
            (temp / "images" / target_split).mkdir(parents=True)
            (temp / "labels" / target_split).mkdir(parents=True)
            counts[split] = 0
            clipped_counts[split] = 0
            label_counts[split] = 0
            try:
                for sequence_id, indices in indices_by_sequence.items():
                    record = inventory.by_sequence()[sequence_id]
                    source.prepare_sequence(sequence_id, record)
                    with source.activate_prepared_window(((sequence_id, record),)):
                        for index in indices:
                            sample = dataset[index]
                            stem = f"{sample.sequence_id}_{sample.frame_index:08d}"
                            sample.event.convert("RGB").save(
                                temp / "images" / target_split / f"{stem}.png")
                            targets = yolo_targets(sample, annotation_policy)
                            lines = ["0 " + " ".join(f"{v:.8f}" for v in t["normalized_xywh"])
                                     for t in targets]
                            (temp / "labels" / target_split / f"{stem}.txt").write_text(
                                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                            lineage.append({"sample_id": f"{sample.sequence_id}:{sample.frame_index}",
                                            "project_split": split, "targets": targets})
                            clipped_counts[split] += sum(t["clipped"] for t in targets)
                            label_counts[split] += len(targets)
                            counts[split] += 1
            finally:
                dataset.close()
        data_yaml = (f"path: {output}\ntrain: images/train\nval: images/val\n"
                     "names:\n  0: drone\n")
        (temp / "data.yaml").write_text(data_yaml, encoding="utf-8")
        (temp / "source.json").write_text(json.dumps({
            "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "inventory": str(inventory_path),
            "inventory_hash": inventory.inventory_hash,
            "dataset_revision": inventory.dataset_revision,
            "project_split_approval_reference": metadata["project_split_approval_reference"],
            "annotation_policy": metadata["annotation_policy"],
            "selection_policy": ("first_annotated_frames_approved_bounds_per_split_v1"
                                 if annotation_policy else "first_in_bounds_annotated_frames_per_split_v1"),
            "annotation_policy_approval_reference": metadata.get("annotation_policy_approval_reference"),
            "label_counts": label_counts,
            "clipped_label_counts": clipped_counts,
            "unpaired_source_annotation_count": metadata.get("unpaired_annotation_count", 0),
            "counts": counts,
            "purpose": "bounded bring-up only; not a formal FRED baseline",
        }, indent=2) + "\n", encoding="utf-8")
        (temp / "label_provenance.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in lineage), encoding="utf-8")
        temp.rename(output)
    return output / "data.yaml"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--val-limit", type=int, default=32)
    args = parser.parse_args(argv)
    print(export_subset(args.manifest, args.inventory, args.output,
                        args.train_limit, args.val_limit))


if __name__ == "__main__":
    main()
