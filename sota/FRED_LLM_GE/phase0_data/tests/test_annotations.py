import pytest

from sota.FRED_LLM_GE.phase0_data.annotations import (
    AnnotationParseError,
    parse_coordinates_text,
)


def test_parser_preserves_required_fields():
    result = parse_coordinates_text(
        "1.33332: 490.0, 413.0, 539.0, 448.0, 1, DJI Mini 2\n"
    )
    annotation = result.annotations[0]
    assert annotation.timestamp == "1.33332"
    assert annotation.box_xyxy == (490.0, 413.0, 539.0, 448.0)
    assert annotation.track_id == 1
    assert annotation.original_class == "DJI Mini 2"


@pytest.mark.parametrize(
    "line",
    [
        "not an annotation",
        "0.1: 1, 2, 1, 4, 1, drone",
        "0.1: nan, 2, 3, 4, 1, drone",
        "0.1: 1, 2, 3, 4, 1.5, drone",
        "",
    ],
)
def test_parser_never_silently_drops_malformed_lines(line):
    with pytest.raises(AnnotationParseError):
        parse_coordinates_text(line)


def test_audit_mode_returns_findings_and_valid_records():
    result = parse_coordinates_text(
        "0.1: 1, 2, 3, 4, 1, drone\nbad\n", strict=False
    )
    assert len(result.annotations) == 1
    assert result.findings[0].line_number == 2
