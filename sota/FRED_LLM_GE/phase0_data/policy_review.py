"""Reuse immutable strict audit evidence and verify DG-P0-04 on real development data.

This produces review overlays and numerical evidence, never evolution fitness or
a Phase 0 freeze. No held-out test archive or annotation is opened.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .annotation_policy import (
    APPROVAL_REFERENCE, POLICY_VERSION, permits_unpaired_annotation, validate_policy_source,
)
from .cli import DEFAULT_CONFIG
from .config import load_config
from .development_audit import (
    DEFAULT_INVENTORY, DEFAULT_OFFICIAL, DEFAULT_PROJECT,
    classify_bounds_finding, load_inputs, validate_inspection,
)
from .fred_api import HFFredSource
from .provenance import atomic_write_json, sha256_file
from .validation import inspect_sequence, write_validation_report
from .visualization import render_annotation_overlay
from ..phase1_detection.adapters.yolo11 import yolo_targets


def review_strict_audit(summary_path, inventory_path, official_path, project_path):
    """Reclassify only approved findings in a separate, hash-linked review artifact."""
    inventory, official, project = load_inputs(inventory_path, official_path, project_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = set(official.challenging_train)
    rows = summary.get("sequences", [])
    if (summary.get("schema_version") != "fred-development-audit-summary-v1"
            or summary.get("complete") is not True
            or summary.get("dataset_revision") != inventory.dataset_revision
            or summary.get("inspected_sequence_count") != len(expected)
            or len(rows) != len(expected)
            or {row["sequence_id"] for row in rows} != expected):
        raise ValueError("strict development audit is incomplete, stale, or has duplicate sequences")
    for name, path in (("inventory_sha256", inventory_path),
                       ("official_split_sha256", official_path),
                       ("project_split_sha256", project_path)):
        if summary.get(name) != sha256_file(path):
            raise ValueError(f"strict audit has stale {name}")
    counts, blockers, reviewed = Counter(), [], []
    for row in rows:
        path = Path(row["report_path"])
        if sha256_file(path) != row["report_sha256"]:
            raise ValueError(f"strict report was modified: {path}")
        report = validate_inspection(
            json.loads(path.read_text(encoding="utf-8")), row["sequence_id"],
            dataset_revision=inventory.dataset_revision,
            repository_revision=official.repository_revision,
            project_split_version=project.version,
        )
        expected_split = "train" if row["sequence_id"] in project.train else "validation"
        if row.get("project_split") != expected_split:
            raise ValueError("strict audit row has stale project split membership")
        changes = []
        for finding in report["findings"]:
            if finding.get("severity") != "error":
                continue
            disposition = None
            if classify_bounds_finding(finding) == "partly_visible":
                disposition = "paired_extended_box_warning"
            elif finding["code"] == "annotation.unmatched_timestamp":
                match = re.fullmatch(
                    r"annotation timestamp (\d+(?:\.\d+)?) matched 0 frames within tolerance 0.000001",
                    finding.get("message", ""),
                )
                if match and permits_unpaired_annotation(
                    row["sequence_id"],
                    SimpleNamespace(timestamp=match[1], source_line=finding.get("line_number")),
                    "0.033333",
                ):
                    disposition = "documented_unpaired_zero_requires_source_review"
            if disposition:
                counts[disposition] += 1
                changes.append({"original_finding": finding, "disposition": disposition})
            else:
                blockers.append({"sequence_id": row["sequence_id"], "finding": finding})
        reviewed.append({**row, "policy_dispositions": changes})
    return {"source_summary": str(summary_path), "source_summary_sha256": sha256_file(summary_path),
            "sequence_count": len(rows), "counts": dict(counts),
            "remaining_blockers": blockers, "sequences": reviewed}


def selected_review_sequences(audit: dict) -> tuple[str, ...]:
    rows = audit["sequences"]
    selected = []
    for split in ("train", "validation"):
        candidates = [row["sequence_id"] for row in rows
                      if row["project_split"] == split
                      and row["out_of_bounds_categories"].get("partly_visible", 0)]
        if not candidates:
            raise ValueError(f"audit has no {split} boundary cases for visual review")
        selected.append(min(candidates, key=int))
    # Sequence 116 is the documented second filename form; 225/230 are the
    # only approved unmatched records. Membership is checked before fetching.
    selected.extend(("116", "225", "230"))
    return tuple(dict.fromkeys(selected))


def run_review(summary_path: Path, output: Path, config_path: Path = DEFAULT_CONFIG,
               inventory_path: Path = DEFAULT_INVENTORY, official_path: Path = DEFAULT_OFFICIAL,
               project_path: Path = DEFAULT_PROJECT) -> dict:
    config = load_config(config_path, workspace_root=Path.cwd())
    validate_policy_source(config, POLICY_VERSION)
    audit = review_strict_audit(summary_path, inventory_path, official_path, project_path)
    inventory, official, project = load_inputs(inventory_path, official_path, project_path)
    if (inventory.dataset_revision != config.source.revision
            or official.repository_revision != config.fred_repository.revision):
        raise ValueError("audit inputs differ from the approved source revisions")
    sequences = selected_review_sequences(audit)
    if not set(sequences) <= set(official.challenging_train):
        raise ValueError("visual-review selection includes non-development sequences")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": "fred-annotation-policy-review-v1", "status": "incomplete",
              "annotation_policy": POLICY_VERSION, "approval_reference": APPROVAL_REFERENCE,
              "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "policy_source_sha256": sha256_file(Path(__file__).with_name("annotation_policy.py")),
              "config_sha256": sha256_file(config_path),
              "dataset_revision": config.source.revision,
              "repository_revision": config.fred_repository.revision,
              "audit": audit, "review_sequence_ids": sequences, "visual_samples": [],
              "unpaired_source_annotations": [], "phase0_frozen": False,
              "evolution_fitness_authorized": False,
              "remaining_gates": ["visual_review", "recording_session_leakage_and_split_freeze",
                                  "controlled_training_and_metric_protocol", "baseline_and_evolution_integration"]}
    source = HFFredSource(config)
    try:
        if audit["remaining_blockers"]:
            raise ValueError("unapproved blocking findings remain in the strict audit")
        for sequence_id in sequences:
            record = inventory.by_sequence()[sequence_id]
            source.prepare_sequence(sequence_id, record)
            with source.activate_prepared_window(((sequence_id, record),)):
                with source.open_prepared_sequence(sequence_id, record) as prepared:
                    inspection = inspect_sequence(
                        config=config, inventory_record=record, prepared_sequence=prepared,
                        official_split=official, project_split=project, annotation_policy=POLICY_VERSION,
                    )
                    write_validation_report([inspection], output / f"sequence_{sequence_id}.json",
                                            context={"annotation_policy": POLICY_VERSION,
                                                     "approval_reference": APPROVAL_REFERENCE},
                                            expected_sequence_ids=(sequence_id,))
                    if not inspection.is_valid:
                        raise ValueError(f"fresh source inspection failed for sequence {sequence_id}")
                    result["unpaired_source_annotations"].extend(
                        {"sequence_id": sequence_id, **asdict(a)} for a in inspection.unpaired_annotations)
                    boundary = {f.sample_id for f in inspection.findings
                                if f.code == "annotation.partial_out_of_bounds_approved"}
                    chosen = [sample for sample in inspection.samples if sample.sample_id in boundary][:2]
                    annotated = [sample for sample in inspection.samples if sample.annotations]
                    if annotated:
                        chosen.extend((annotated[0], annotated[-1]))
                    for sample in {s.sample_id: s for s in chosen}.values():
                        files = {}
                        for modality in ("rgb", "event"):
                            ref = getattr(sample, modality)
                            with Image.open(prepared.sequence_root / ref.archive_member) as image:
                                image.load()
                                filename = f"{sequence_id}_{sample.frame_index:08d}_{modality}.png"
                                render_annotation_overlay(
                                    image, sample.annotations, project_split=sample.project_split,
                                    destination=output / filename, annotation_policy=POLICY_VERSION)
                                files[modality] = filename
                                if modality == "event":
                                    targets = yolo_targets(SimpleNamespace(event=image, annotations=sample.annotations),
                                                           POLICY_VERSION)
                        result["visual_samples"].append(
                            {"sample_id": sample.sample_id, "project_split": sample.project_split.value,
                             "event_archive_member": sample.event.archive_member,
                             "images": files, "targets": targets})
        omitted = {(a["sequence_id"], a["source_line"]) for a in result["unpaired_source_annotations"]}
        if omitted != {("225", 1), ("230", 1)}:
            raise ValueError("fresh inspection did not recover both documented unpaired records")
        samples = result["visual_samples"]
        if not {"train", "validation"} <= {s["project_split"] for s in samples}:
            raise ValueError("visual samples do not cover both development splits")
        members = [s["event_archive_member"] for s in samples]
        if not (any("_frame_" in m for m in members) and any("_frame_" not in m for m in members)):
            raise ValueError("visual samples do not cover both released event filename forms")
        result["clipped_visual_label_count"] = sum(t["clipped"] for s in samples for t in s["targets"])
        if not result["clipped_visual_label_count"]:
            raise ValueError("visual set contains no clipped boundary label")
        result["status"] = "numerical_checks_passed_visual_review_required"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_write_json(output / "review.json", result)
    print(f"Policy review: {output / 'review.json'}")
    print(f"Strict reports reused: {audit['sequence_count']}; dispositions: {audit['counts']}")
    print(f"Visual pairs: {len(result['visual_samples'])}; clipped labels: {result['clipped_visual_label_count']}")
    print("Numerical checks passed. Inspect overlays before freezing Phase 0; no fitness assigned.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args(argv)
    run_review(args.summary, args.output, args.config, args.inventory, args.official, args.project)


if __name__ == "__main__":
    main()
