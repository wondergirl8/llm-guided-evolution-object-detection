"""Summarize strict Phase 0 inspections of FRED challenging-train sequences.

This is evidence collection for DG-P0-02 and annotation policy review. It does
not approve a split or turn a data-quality failure into a valid manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .inventory import load_inventory
from .provenance import atomic_write_json, sha256_file, utc_now
from .splits import (
    load_official_split_manifest,
    load_project_split_manifest,
    validate_official_split_manifest,
    validate_project_split,
)


DEFAULT_INVENTORY = Path("data/fred_phase0/metadata/source_inventory.json")
DEFAULT_OFFICIAL = Path("data/fred_phase0/metadata/official_challenging_split.json")
DEFAULT_PROJECT = Path("sota/FRED_LLM_GE/configs/phase0/project_split_bringup_v1.json")
BOUNDS_MESSAGE = re.compile(r"^box \(([^)]+)\) is outside (\d+)x(\d+)$")


def classify_bounds_finding(finding: dict) -> str | None:
    """Classify an existing raw-box finding without altering its severity."""
    if finding.get("code") != "annotation.out_of_bounds":
        return None
    message = finding.get("message")
    match = BOUNDS_MESSAGE.fullmatch(message) if isinstance(message, str) else None
    if match is None:
        return "unclassified"
    try:
        box = tuple(float(part.strip()) for part in match.group(1).split(","))
        width, height = int(match.group(2)), int(match.group(3))
    except ValueError:
        return "unclassified"
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        return "unclassified"
    x1, y1, x2, y2 = box
    if width <= 0 or height <= 0 or x1 >= x2 or y1 >= y2:
        return "unclassified"
    overlaps = max(0, x1) < min(width, x2) and max(0, y1) < min(height, y2)
    return "partly_visible" if overlaps else "fully_outside"


def load_inputs(inventory_path: Path, official_path: Path, project_path: Path):
    inventory = load_inventory(inventory_path)
    official = load_official_split_manifest(official_path)
    project = load_project_split_manifest(project_path)
    validate_official_split_manifest(official, inventory)
    validate_project_split(project, official)
    return inventory, official, project


def validate_inspection(report: dict, sequence_id: str, *,
                        dataset_revision: str, repository_revision: str,
                        project_split_version: str) -> dict:
    """Require a complete inspection, while preserving data-quality failures."""
    if report.get("schema_version") != "phase0-validation-report-v2":
        raise ValueError(f"sequence {sequence_id}: unsupported inspection schema")
    context = report.get("context", {})
    expected_context = {
        "dataset_revision": dataset_revision,
        "repository_revision": repository_revision,
        "project_split_version": project_split_version,
    }
    for name, expected in expected_context.items():
        if context.get(name) != expected:
            raise ValueError(f"sequence {sequence_id}: stale {name} in inspection")
    if context.get("partial_out_of_bounds_bringup_policy") is not False:
        raise ValueError(f"sequence {sequence_id}: inspection used the bring-up exception")
    scope = report.get("scope", {})
    if (report.get("execution", {}).get("status") != "completed"
            or not scope.get("complete")
            or scope.get("expected_sequence_ids") != [sequence_id]
            or scope.get("inspected_sequence_ids") != [sequence_id]
            or report.get("sequence_count") != 1
            or len(report.get("sequences", [])) != 1
            or report["sequences"][0].get("sequence_id") != sequence_id):
        raise ValueError(f"sequence {sequence_id}: inspection did not complete its scope")
    if report.get("status") not in ("passed", "failed"):
        raise ValueError(f"sequence {sequence_id}: invalid inspection status")
    if report.get("valid") != (report["status"] == "passed"):
        raise ValueError(f"sequence {sequence_id}: inconsistent inspection validity")
    if not isinstance(report.get("findings"), list):
        raise ValueError(f"sequence {sequence_id}: findings are missing")
    has_errors = any(finding.get("severity") == "error" for finding in report["findings"])
    if report["valid"] == has_errors:
        raise ValueError(f"sequence {sequence_id}: findings disagree with inspection status")
    return report


def affected_filename_indexes(summary_path: Path, inventory_path: Path,
                              official_path: Path, project_path: Path) -> tuple[int, ...]:
    """Locate the original audit's filename failures for a bounded recheck."""
    inventory, official, _ = load_inputs(inventory_path, official_path, project_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("schema_version") != "fred-development-audit-summary-v1"
            or summary.get("complete") is not True
            or summary.get("inspected_sequence_count") != len(official.challenging_train)
            or summary.get("dataset_revision") != inventory.dataset_revision):
        raise ValueError("original development audit is incomplete or stale")
    for name, path in (("inventory_sha256", inventory_path),
                       ("official_split_sha256", official_path),
                       ("project_split_sha256", project_path)):
        if summary.get(name) != sha256_file(path):
            raise ValueError(f"original audit has a stale {name}")
    sequences = summary.get("sequences", [])
    if (len(sequences) != len(official.challenging_train)
            or {item["sequence_id"] for item in sequences} != set(official.challenging_train)):
        raise ValueError("original audit does not cover each development sequence exactly once")
    affected = {
        item["sequence_id"] for item in sequences
        if item["finding_counts"].get("error:pairing.event_filename_unrecognized", 0)
    }
    return tuple(index for index, sequence_id in enumerate(official.challenging_train)
                 if sequence_id in affected)


def summarize(report_dir: Path, inventory_path: Path, official_path: Path,
              project_path: Path, *, reuse_unaffected_from: Path | None = None) -> dict:
    if reuse_unaffected_from is not None and report_dir.resolve() == reuse_unaffected_from.resolve():
        raise ValueError("new and previous report directories must differ")
    inventory, official, project = load_inputs(
        inventory_path, official_path, project_path,
    )
    expected = official.challenging_train
    train = set(project.train)
    counters = {"train": Counter(), "validation": Counter()}
    inspected = []
    invalid = []
    missing = []
    reused = []
    sequence_summaries = []
    findings = Counter()
    bounds = Counter()
    totals = Counter()
    for sequence_id in expected:
        path = report_dir / f"sequence_{sequence_id}.json"
        from_previous = False
        if not path.is_file() and reuse_unaffected_from is not None:
            path = reuse_unaffected_from / f"sequence_{sequence_id}.json"
            from_previous = True
        if not path.is_file():
            missing.append(sequence_id)
            continue
        report = validate_inspection(
            json.loads(path.read_text(encoding="utf-8")), sequence_id,
            dataset_revision=inventory.dataset_revision,
            repository_revision=official.repository_revision,
            project_split_version=project.version,
        )
        if from_previous and any(
            finding.get("code") == "pairing.event_filename_unrecognized"
            for finding in report["findings"]
        ):
            missing.append(sequence_id)
            continue
        inspected.append(sequence_id)
        if from_previous:
            reused.append(sequence_id)
        if not report["valid"]:
            invalid.append(sequence_id)
        split = "train" if sequence_id in train else "validation"
        counters[split]["sequences"] += 1
        for name in ("sample_count", "annotation_count"):
            value = report[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"sequence {sequence_id}: invalid {name}")
            counters[split][name] += value
            totals[name] += value
        sequence = report["sequences"][0]
        sequence_counts = {}
        for name in ("rgb_count", "event_count"):
            value = sequence[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"sequence {sequence_id}: invalid {name}")
            counters[split][name] += value
            sequence_counts[name] = value
        sequence_findings = Counter()
        sequence_bounds = Counter()
        for finding in report["findings"]:
            if finding.get("severity") not in ("error", "warning") or not finding.get("code"):
                raise ValueError(f"sequence {sequence_id}: malformed finding")
            finding_key = f"{finding['severity']}:{finding['code']}"
            findings[finding_key] += 1
            sequence_findings[finding_key] += 1
            bounds_category = classify_bounds_finding(finding)
            if bounds_category is not None:
                bounds[bounds_category] += 1
                sequence_bounds[bounds_category] += 1
        sequence_summaries.append({
            "sequence_id": sequence_id,
            "report_path": path.as_posix(),
            "report_sha256": sha256_file(path),
            "project_split": split,
            "data_status": report["status"],
            "sample_count": report["sample_count"],
            "annotation_count": report["annotation_count"],
            **sequence_counts,
            "first_annotation_timestamp": sequence.get("first_annotation_timestamp"),
            "last_annotation_timestamp": sequence.get("last_annotation_timestamp"),
            "finding_counts": dict(sorted(sequence_findings.items())),
            "out_of_bounds_categories": dict(sorted(sequence_bounds.items())),
        })
    return {
        "schema_version": "fred-development-audit-summary-v1",
        "created_at": utc_now(),
        "purpose": "strict data-quality evidence only; no approved split or fitness",
        "complete": not missing,
        "expected_sequence_count": len(expected),
        "inspected_sequence_count": len(inspected),
        "missing_sequence_ids": missing,
        "reused_previous_sequence_ids": reused,
        "invalid_sequence_ids": invalid,
        "sequences": sequence_summaries,
        "dataset_revision": inventory.dataset_revision,
        "inventory_sha256": sha256_file(inventory_path),
        "official_split_sha256": sha256_file(official_path),
        "project_split_sha256": sha256_file(project_path),
        "by_project_split": {name: dict(counters[name]) for name in counters},
        "totals": dict(totals),
        "finding_counts": dict(sorted(findings.items())),
        "out_of_bounds_categories": dict(sorted(bounds.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sequence = commands.add_parser("sequence-id")
    sequence.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    sequence.add_argument("--index", type=int, required=True)
    verify = commands.add_parser("verify-report")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--sequence", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--report-dir", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    summary.add_argument("--reuse-unaffected-from", type=Path)
    affected = commands.add_parser("affected-filename-indexes")
    affected.add_argument("--summary", type=Path, required=True)
    affected.add_argument("--expected-count", type=int, required=True)
    for command in (verify, summary, affected):
        command.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
        command.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
        command.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args(argv)
    if args.command == "sequence-id":
        official = load_official_split_manifest(args.official)
        if not 0 <= args.index < len(official.challenging_train):
            parser.error(f"index must be from 0 to {len(official.challenging_train) - 1}")
        print(official.challenging_train[args.index])
        return
    if args.command == "verify-report":
        inventory, official, project = load_inputs(
            args.inventory, args.official, args.project,
        )
        if args.sequence not in official.challenging_train:
            raise ValueError("inspection is not a challenging-train sequence")
        report = validate_inspection(
            json.loads(args.report.read_text(encoding="utf-8")), args.sequence,
            dataset_revision=inventory.dataset_revision,
            repository_revision=official.repository_revision,
            project_split_version=project.version,
        )
        counts = Counter(f"{item['severity']}:{item['code']}" for item in report["findings"])
        print(f"sequence {args.sequence}: data_status={report['status']}; "
              f"samples={report['sample_count']}; findings={dict(counts)}")
        return
    if args.command == "affected-filename-indexes":
        indexes = affected_filename_indexes(
            args.summary, args.inventory, args.official, args.project,
        )
        if len(indexes) != args.expected_count:
            raise ValueError(f"expected {args.expected_count} affected sequences; found {len(indexes)}")
        print(*indexes, sep="\n")
        return
    result = summarize(
        args.report_dir, args.inventory, args.official, args.project,
        reuse_unaffected_from=args.reuse_unaffected_from,
    )
    output = args.output or args.report_dir / "summary.json"
    atomic_write_json(output, result)
    print(f"audit summary: {output}; inspected "
          f"{result['inspected_sequence_count']}/{result['expected_sequence_count']}; "
          f"invalid data sequences={len(result['invalid_sequence_ids'])}; "
          f"complete={result['complete']}; "
          f"reused={len(result['reused_previous_sequence_ids'])}; "
          f"bounds={result['out_of_bounds_categories']}")


if __name__ == "__main__":
    main()
