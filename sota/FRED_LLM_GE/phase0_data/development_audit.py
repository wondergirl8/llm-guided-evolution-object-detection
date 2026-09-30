"""Summarize strict Phase 0 inspections of FRED challenging-train sequences.

This is evidence collection for DG-P0-02 and annotation policy review. It does
not approve a split or turn a data-quality failure into a valid manifest.
"""

from __future__ import annotations

import argparse
import json
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


def summarize(report_dir: Path, inventory_path: Path, official_path: Path,
              project_path: Path) -> dict:
    inventory, official, project = load_inputs(
        inventory_path, official_path, project_path,
    )
    expected = official.challenging_train
    train = set(project.train)
    counters = {"train": Counter(), "validation": Counter()}
    inspected = []
    invalid = []
    missing = []
    sequence_summaries = []
    findings = Counter()
    totals = Counter()
    for sequence_id in expected:
        path = report_dir / f"sequence_{sequence_id}.json"
        if not path.is_file():
            missing.append(sequence_id)
            continue
        report = validate_inspection(
            json.loads(path.read_text(encoding="utf-8")), sequence_id,
            dataset_revision=inventory.dataset_revision,
            repository_revision=official.repository_revision,
            project_split_version=project.version,
        )
        inspected.append(sequence_id)
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
        for finding in report["findings"]:
            if finding.get("severity") not in ("error", "warning") or not finding.get("code"):
                raise ValueError(f"sequence {sequence_id}: malformed finding")
            finding_key = f"{finding['severity']}:{finding['code']}"
            findings[finding_key] += 1
            sequence_findings[finding_key] += 1
        sequence_summaries.append({
            "sequence_id": sequence_id,
            "project_split": split,
            "data_status": report["status"],
            "sample_count": report["sample_count"],
            "annotation_count": report["annotation_count"],
            **sequence_counts,
            "first_annotation_timestamp": sequence.get("first_annotation_timestamp"),
            "last_annotation_timestamp": sequence.get("last_annotation_timestamp"),
            "finding_counts": dict(sorted(sequence_findings.items())),
        })
    return {
        "schema_version": "fred-development-audit-summary-v1",
        "created_at": utc_now(),
        "purpose": "strict data-quality evidence only; no approved split or fitness",
        "complete": not missing,
        "expected_sequence_count": len(expected),
        "inspected_sequence_count": len(inspected),
        "missing_sequence_ids": missing,
        "invalid_sequence_ids": invalid,
        "sequences": sequence_summaries,
        "dataset_revision": inventory.dataset_revision,
        "inventory_sha256": sha256_file(inventory_path),
        "official_split_sha256": sha256_file(official_path),
        "project_split_sha256": sha256_file(project_path),
        "by_project_split": {name: dict(counters[name]) for name in counters},
        "totals": dict(totals),
        "finding_counts": dict(sorted(findings.items())),
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
    for command in (verify, summary):
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
    result = summarize(
        args.report_dir, args.inventory, args.official, args.project,
    )
    output = args.output or args.report_dir / "summary.json"
    atomic_write_json(output, result)
    print(f"audit summary: {output}; inspected "
          f"{result['inspected_sequence_count']}/{result['expected_sequence_count']}; "
          f"invalid data sequences={len(result['invalid_sequence_ids'])}; "
          f"complete={result['complete']}")


if __name__ == "__main__":
    main()
