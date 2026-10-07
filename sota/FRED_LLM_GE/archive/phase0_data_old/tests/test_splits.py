import pytest

from sota.FRED_LLM_GE.phase0_data.splits import (
    OfficialSplitManifest,
    ProjectSplitManifest,
    parse_split_text,
    validate_project_split,
)


def official() -> OfficialSplitManifest:
    return OfficialSplitManifest("a" * 40, ("0", "1"), ("2",), "x", "y", "z", "now")


def test_split_parser_normalizes_slashes_and_preserves_order():
    assert parse_split_text("0/\n 2/ \n", source="test") == ("0", "2")


def test_split_parser_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        parse_split_text("0/\n0/\n", source="test")


def test_project_split_must_exactly_partition_official_membership():
    valid = ProjectSplitManifest("v1", "sequence", ("0",), ("1",), ("2",), "DG-P0-02")
    validate_project_split(valid, official())
    invalid = ProjectSplitManifest("v1", "sequence", ("0",), (), ("2",), "DG-P0-02")
    with pytest.raises(ValueError, match="non-empty|partition"):
        validate_project_split(invalid, official())


def test_project_split_requires_approval_reference():
    value = ProjectSplitManifest("v1", "sequence", ("0",), ("1",), ("2",), "")
    with pytest.raises(ValueError, match="approval"):
        validate_project_split(value, official())


def test_project_split_rejects_duplicate_membership_entries():
    value = ProjectSplitManifest(
        "v1", "sequence", ("0", "0"), ("1",), ("2",), "DG-P0-02"
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_project_split(value, official())
