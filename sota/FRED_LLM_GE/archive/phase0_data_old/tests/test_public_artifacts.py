from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISSUE_REGISTRY = PROJECT_ROOT / "artifacts" / "phase0" / "known_data_issues.yaml"
PROTECTED_LABEL_KEYS = {
    "box_xyxy",
    "class_name",
    "example_box_xyxy",
    "example_timestamp",
    "timestamp",
    "track_id",
    "verified_sequence_ids",
}


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def test_public_issue_registry_redacts_held_out_label_evidence():
    registry = yaml.safe_load(ISSUE_REGISTRY.read_text(encoding="utf-8"))
    held_out = [
        issue
        for issue in registry["issues"]
        if issue.get("affected_scope", {}).get("official_split") == "challenging_test"
    ]
    assert held_out, "expected the protected held-out issue classification"
    for issue in held_out:
        assert PROTECTED_LABEL_KEYS.isdisjoint(set(_keys(issue)))
        evidence = issue["evidence"]
        assert evidence["access_class"] == "evaluator_protected"
        assert evidence["label_level_evidence_committed"] is False
        assert len(evidence["protected_evidence_fingerprint_sha256"]) == 64
