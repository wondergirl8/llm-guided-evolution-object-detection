"""Strict audit reports must keep quality failures and incomplete coverage visible."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sota.FRED_LLM_GE.archive.phase0_data_old.development_audit import (
    affected_filename_indexes, classify_bounds_finding, summarize, validate_inspection,
)


def report(sequence_id, *, valid, findings=None):
    return {
        "schema_version": "phase0-validation-report-v2",
        "context": {
            "dataset_revision": "dataset-rev",
            "repository_revision": "repo-rev",
            "project_split_version": "split-v1",
            "partial_out_of_bounds_bringup_policy": False,
        },
        "status": "passed" if valid else "failed",
        "valid": valid,
        "execution": {"status": "completed", "error": None},
        "scope": {
            "complete": True,
            "expected_sequence_ids": [sequence_id],
            "inspected_sequence_ids": [sequence_id],
        },
        "sequence_count": 1,
        "sample_count": 12,
        "annotation_count": 4,
        "sequences": [{"sequence_id": sequence_id, "rgb_count": 12,
                       "event_count": 12}],
        "findings": findings if findings is not None else [],
    }


class DevelopmentAuditTest(unittest.TestCase):
    def test_affected_filename_indexes_checks_original_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            summary = {
                "schema_version": "fred-development-audit-summary-v1",
                "complete": True,
                "inspected_sequence_count": 2,
                "dataset_revision": "dataset-rev",
                "inventory_sha256": "hash",
                "official_split_sha256": "hash",
                "project_split_sha256": "hash",
                "sequences": [
                    {"sequence_id": "3", "finding_counts": {}},
                    {"sequence_id": "8", "finding_counts": {
                        "error:pairing.event_filename_unrecognized": 1,
                    }},
                ],
            }
            summary_path.write_text(json.dumps(summary))
            inventory = SimpleNamespace(dataset_revision="dataset-rev")
            official = SimpleNamespace(challenging_train=("3", "8"))
            with patch(
                "sota.FRED_LLM_GE.archive.phase0_data_old.development_audit.load_inputs",
                return_value=(inventory, official, object()),
            ), patch(
                "sota.FRED_LLM_GE.archive.phase0_data_old.development_audit.sha256_file",
                return_value="hash",
            ):
                args = (summary_path, root / "inventory", root / "official", root / "project")
                self.assertEqual(affected_filename_indexes(*args), (1,))
                summary["project_split_sha256"] = "stale"
                summary_path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError, "stale project_split_sha256"):
                    affected_filename_indexes(*args)

    def test_existing_bounds_messages_are_classified_without_changing_severity(self):
        finding = {"code": "annotation.out_of_bounds",
                   "message": "box (949.2, 507.64, 1120.8, 731.16) is outside 1280x720"}
        self.assertEqual(classify_bounds_finding(finding), "partly_visible")
        finding["message"] = "box (949.2, 721.0, 1120.8, 731.16) is outside 1280x720"
        self.assertEqual(classify_bounds_finding(finding), "fully_outside")
        finding["message"] = "unrecognized report format"
        self.assertEqual(classify_bounds_finding(finding), "unclassified")

    def test_complete_data_failure_is_retained_as_evidence(self):
        finding = {"severity": "error", "code": "annotation.out_of_bounds"}
        failed = report("3", valid=False, findings=[finding])
        self.assertIs(validate_inspection(
            failed, "3", dataset_revision="dataset-rev",
            repository_revision="repo-rev", project_split_version="split-v1",
        ), failed)
        failed["context"]["partial_out_of_bounds_bringup_policy"] = True
        with self.assertRaisesRegex(ValueError, "bring-up exception"):
            validate_inspection(
                failed, "3", dataset_revision="dataset-rev",
                repository_revision="repo-rev", project_split_version="split-v1",
            )

    def test_partial_summary_reports_missing_and_invalid_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sequence_3.json").write_text(json.dumps(report(
                "3", valid=False, findings=[
                    {"severity": "error", "code": "annotation.out_of_bounds",
                     "message": "box (10.0, 5.0, 31.0, 20.0) is outside 30x30"},
                ],
            )), encoding="utf-8")
            inventory = SimpleNamespace(dataset_revision="dataset-rev")
            official = SimpleNamespace(
                repository_revision="repo-rev", challenging_train=("3", "8"),
            )
            project = SimpleNamespace(version="split-v1", train=("3",), validation=("8",))
            with patch(
                "sota.FRED_LLM_GE.archive.phase0_data_old.development_audit.load_inputs",
                return_value=(inventory, official, project),
            ), patch(
                "sota.FRED_LLM_GE.archive.phase0_data_old.development_audit.sha256_file",
                return_value="hash",
            ):
                summary = summarize(root, root / "inventory", root / "official",
                                    root / "project")
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["missing_sequence_ids"], ["8"])
            self.assertEqual(summary["invalid_sequence_ids"], ["3"])
            self.assertEqual(summary["finding_counts"],
                             {"error:annotation.out_of_bounds": 1})
            self.assertEqual(summary["by_project_split"]["train"]["sample_count"], 12)
            self.assertEqual(summary["sequences"][0]["sequence_id"], "3")
            self.assertEqual(summary["sequences"][0]["data_status"], "failed")
            self.assertEqual(summary["sequences"][0]["finding_counts"],
                             {"error:annotation.out_of_bounds": 1})
            self.assertEqual(summary["out_of_bounds_categories"], {"partly_visible": 1})

    def test_reuse_keeps_unaffected_reports_but_requires_pairing_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / "previous", root / "current"
            previous.mkdir()
            current.mkdir()
            (previous / "sequence_3.json").write_text(json.dumps(report("3", valid=True)))
            stale = report("8", valid=False, findings=[
                {"severity": "error", "code": "pairing.event_filename_unrecognized"},
            ])
            stale["sample_count"] = 0
            (previous / "sequence_8.json").write_text(json.dumps(stale))
            inventory = SimpleNamespace(dataset_revision="dataset-rev")
            official = SimpleNamespace(
                repository_revision="repo-rev", challenging_train=("3", "8"),
            )
            project = SimpleNamespace(version="split-v1", train=("3",), validation=("8",))
            with patch(
                "sota.FRED_LLM_GE.archive.phase0_data_old.development_audit.load_inputs",
                return_value=(inventory, official, project),
            ), patch(
                "sota.FRED_LLM_GE.archive.phase0_data_old.development_audit.sha256_file",
                return_value="hash",
            ):
                first = summarize(current, root / "inventory", root / "official",
                                  root / "project", reuse_unaffected_from=previous)
                self.assertEqual(first["reused_previous_sequence_ids"], ["3"])
                self.assertEqual(first["missing_sequence_ids"], ["8"])
                (current / "sequence_8.json").write_text(json.dumps(report("8", valid=True)))
                final = summarize(current, root / "inventory", root / "official",
                                  root / "project", reuse_unaffected_from=previous)
            self.assertTrue(final["complete"])
            self.assertEqual(final["inspected_sequence_count"], 2)
            self.assertEqual(final["reused_previous_sequence_ids"], ["3"])
            self.assertEqual(final["sequences"][1]["report_path"],
                             (current / "sequence_8.json").as_posix())


if __name__ == "__main__":
    unittest.main()
