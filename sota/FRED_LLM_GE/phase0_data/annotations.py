"""Strict parser for the released FRED ``coordinates.txt`` format."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .schema import Annotation, ValidationFinding


class AnnotationParseError(ValueError):
    def __init__(self, findings: tuple[ValidationFinding, ...]):
        super().__init__(f"annotation parsing produced {len(findings)} finding(s)")
        self.findings = findings


@dataclass(frozen=True, slots=True)
class AnnotationParseResult:
    annotations: tuple[Annotation, ...]
    findings: tuple[ValidationFinding, ...]

    @property
    def by_timestamp(self) -> dict[str, tuple[Annotation, ...]]:
        grouped: dict[str, list[Annotation]] = {}
        for annotation in self.annotations:
            grouped.setdefault(annotation.timestamp, []).append(annotation)
        return {key: tuple(value) for key, value in grouped.items()}


def _finding(
    code: str,
    message: str,
    *,
    source_reference: str,
    line_number: int,
    content: str,
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        severity="error",
        message=message,
        source_reference=source_reference,
        line_number=line_number,
        details={"content": content.rstrip("\n")},
    )


def parse_coordinates_text(
    content: str,
    *,
    source_reference: str = "coordinates.txt",
    strict: bool = True,
) -> AnnotationParseResult:
    annotations: list[Annotation] = []
    findings: list[ValidationFinding] = []

    if not content.splitlines():
        findings.append(
            ValidationFinding(
                code="annotation.empty_file",
                severity="error",
                message="annotation file contains no formal records",
                source_reference=source_reference,
            )
        )

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            findings.append(
                _finding(
                    "annotation.blank_line",
                    "blank annotation lines are not an approved formal record",
                    source_reference=source_reference,
                    line_number=line_number,
                    content=raw_line,
                )
            )
            continue
        try:
            timestamp_text, fields_text = line.split(":", maxsplit=1)
            timestamp = Decimal(timestamp_text.strip())
            if not timestamp.is_finite() or timestamp < 0:
                raise ValueError("timestamp must be finite and non-negative")
            fields = [part.strip() for part in fields_text.split(",", maxsplit=5)]
            if len(fields) != 6 or not all(fields):
                raise ValueError("expected x1, y1, x2, y2, track_id, class")
            x1, y1, x2, y2 = (float(value) for value in fields[:4])
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                raise ValueError("box coordinates must be finite")
            if x2 <= x1 or y2 <= y1:
                raise ValueError("box must satisfy x2 > x1 and y2 > y1")
            track_decimal = Decimal(fields[4])
            if not track_decimal.is_finite() or track_decimal != track_decimal.to_integral_value():
                raise ValueError("track_id must be an integer")
            track_id = int(track_decimal)
            if track_id < 0:
                raise ValueError("track_id must be non-negative")
            annotations.append(
                Annotation(
                    timestamp=timestamp_text.strip(),
                    box_xyxy=(x1, y1, x2, y2),
                    track_id=track_id,
                    original_class=fields[5],
                    source_line=line_number,
                )
            )
        except (InvalidOperation, ValueError) as error:
            findings.append(
                _finding(
                    "annotation.malformed_line",
                    str(error),
                    source_reference=source_reference,
                    line_number=line_number,
                    content=raw_line,
                )
            )

    result = AnnotationParseResult(tuple(annotations), tuple(findings))
    if strict and findings:
        raise AnnotationParseError(result.findings)
    return result


def parse_coordinates_file(path: str | Path, *, strict: bool = True) -> AnnotationParseResult:
    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except OSError as error:
        finding = ValidationFinding(
            code="annotation.unreadable",
            severity="error",
            message=str(error),
            source_reference=str(source),
        )
        raise AnnotationParseError((finding,)) from error
    return parse_coordinates_text(content, source_reference=str(source), strict=strict)
