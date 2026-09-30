"""Strict audit reports must keep quality failures and incomplete coverage visible."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sota.FRED_LLM_GE.phase0_data.development_audit import (
    summarize, validate_inspection,
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
                    {"severity": "error", "code": "annotation.out_of_bounds"},
                ],
            )), encoding="utf-8")
            inventory = SimpleNamespace(dataset_revision="dataset-rev")
            official = SimpleNamespace(
                repository_revision="repo-rev", challenging_train=("3", "8"),
            )
            project = SimpleNamespace(version="split-v1", train=("3",), validation=("8",))
            with patch(
                "sota.FRED_LLM_GE.phase0_data.development_audit.load_inputs",
                return_value=(inventory, official, project),
            ), patch(
                "sota.FRED_LLM_GE.phase0_data.development_audit.sha256_file",
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


if __name__ == "__main__":
    unittest.main()
