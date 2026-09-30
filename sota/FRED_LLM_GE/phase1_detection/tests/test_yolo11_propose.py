"""Check the boundary between untrusted LLM text and a YOLO11 candidate."""

import json
import tempfile
import unittest
from pathlib import Path

from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11.propose import (
    parse_genome, propose,
)
from sota.FRED_LLM_GE.phase1_detection.seeds.yolo11.validator import validate_source


ORIGINAL = {"backbone_p3": 2, "attention": 2, "neck_p3": 2}
VALID = 'GENOME = {"backbone_p3": 3, "attention": 2, "neck_p3": 2}'


class ProposeTest(unittest.TestCase):
    def test_only_one_approved_step_is_accepted(self):
        self.assertEqual(parse_genome(f"```python\n{VALID}\n```", ORIGINAL)["backbone_p3"], 3)
        rejected = (
            'GENOME = {"backbone_p3": 4, "attention": 2, "neck_p3": 2}',
            'GENOME = {"backbone_p3": 3, "attention": 3, "neck_p3": 2}',
            'GENOME = {"backbone_p3": 3, "attention": 2, "neck_p3": 2, "nc": 2}',
            'GENOME = {"backbone_p3": 3, "backbone_p3": 2, "attention": 2, "neck_p3": 2}',
            'GENOME = {[]: 3, "attention": 2, "neck_p3": 2}',
            'GENOME = {"backbone_p3": True, "attention": 2, "neck_p3": 2}',
            'GENOME = {"backbone_p3": 3, "attention": 2, "neck_p3": 2}\nprint("unsafe")',
        )
        for response in rejected:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_genome(response, ORIGINAL)

    def test_valid_candidate_and_failed_attempt_are_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "proposal"
            source = propose(
                "P123", target, "test-model", responses=iter((
                    ("revision", 1, "not Python"),
                    ("revision", 2, VALID),
                )),
            )
            self.assertEqual(validate_source(source)["backbone_p3"], 3)
            record = json.loads((target / "proposal.json").read_text())
            self.assertEqual(record["status"], "candidate_generated_unscored")
            self.assertIn("error", record["attempts"][0])
            self.assertEqual(record["model_revision"], "revision")
            self.assertTrue((target / "prompt.txt").is_file())
            with self.assertRaises(FileExistsError):
                propose("P123", target, "test-model", responses=iter(()))

    def test_invalid_responses_are_preserved_without_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "proposal"
            with self.assertRaises(RuntimeError):
                propose("P124", target, "test-model", responses=iter((
                    ("revision", 1, "I cannot produce the assignment"),
                )))
            self.assertFalse((target / "network_P124.py").exists())
            record = json.loads((target / "proposal.json").read_text())
            self.assertEqual(record["status"], "invalid_llm_responses")
            self.assertEqual(len(record["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
