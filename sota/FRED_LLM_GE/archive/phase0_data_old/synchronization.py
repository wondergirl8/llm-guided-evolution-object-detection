"""Evidence-oriented RGB/event ordering and timestamp association."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .config import TimestampConfig
from .schema import ValidationFinding


_NATURAL_PARTS = re.compile(r"(\d+)")
_EVENT_FRAME_NAME = re.compile(r"^Video_(\d+)_frame_(\d+)\.png$")
_RGB_FRAME_NAME = re.compile(
    r"^Video_(\d+)_(\d{2})_(\d{2})_(\d{2})\.(\d+)\.jpg$"
)


def _natural_key(path: Path) -> tuple[object, ...]:
    parts: list[object] = []
    for part in _NATURAL_PARTS.split(path.name):
        parts.append(int(part) if part.isdigit() else part.lower())
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class FramePair:
    frame_index: int
    timestamp: str
    rgb_relative_path: str
    event_relative_path: str


@dataclass(frozen=True, slots=True)
class PairingResult:
    pairs: tuple[FramePair, ...]
    findings: tuple[ValidationFinding, ...]
    rgb_count: int
    event_count: int


def timestamp_for_frame(frame_index: int, policy: TimestampConfig) -> str:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if policy.policy != "fred_upstream_30hz_v1":
        raise ValueError(f"unsupported timestamp policy {policy.policy!r}")
    value = (Decimal(frame_index) + policy.frame_index_offset) * Decimal(
        policy.frame_period_seconds
    )
    # This deliberately reproduces the pinned upstream code's six-decimal then
    # float-string conversion. Dataset-level verification remains a freeze gate.
    return str(float(f"{value:.6f}"))


def pair_sequence_frames(
    sequence_root: str | Path,
    *,
    sequence_id: str,
    timestamp_policy: TimestampConfig,
) -> PairingResult:
    root = Path(sequence_root).resolve()
    rgb_directory = root / "RGB"
    event_directory = root / "Event" / "Frames"
    rgb_files = sorted(
        (path for path in rgb_directory.iterdir() if path.is_file()), key=_natural_key
    ) if rgb_directory.is_dir() else []
    event_files = sorted(
        (path for path in event_directory.iterdir() if path.is_file()), key=_natural_key
    ) if event_directory.is_dir() else []
    rgb = [path for path in rgb_files if path.suffix.lower() == ".jpg"]
    event = [path for path in event_files if path.suffix.lower() == ".png"]
    findings: list[ValidationFinding] = []
    _record_unsupported_files(
        findings,
        sequence_id=sequence_id,
        modality="RGB",
        unsupported=[path for path in rgb_files if path.suffix.lower() != ".jpg"],
    )
    _record_unsupported_files(
        findings,
        sequence_id=sequence_id,
        modality="event",
        unsupported=[path for path in event_files if path.suffix.lower() != ".png"],
    )
    if len(rgb) != len(event):
        findings.append(
            ValidationFinding(
                code="pairing.count_mismatch",
                severity="error",
                message=f"RGB count {len(rgb)} differs from event-frame count {len(event)}",
                sequence_id=sequence_id,
            )
        )
        return PairingResult((), tuple(findings), len(rgb), len(event))
    if not rgb:
        findings.append(
            ValidationFinding(
                code="pairing.empty_sequence",
                severity="error",
                message="sequence contains no RGB/event frame pairs",
                sequence_id=sequence_id,
            )
        )
        return PairingResult((), tuple(findings), 0, 0)

    findings.extend(_validate_rgb_names(rgb, sequence_id=sequence_id))
    findings.extend(
        _validate_event_names(
            event,
            sequence_id=sequence_id,
            timestamp_policy=timestamp_policy,
        )
    )
    if findings:
        return PairingResult((), tuple(findings), len(rgb), len(event))

    pairs = tuple(
        FramePair(
            frame_index=index,
            timestamp=timestamp_for_frame(index, timestamp_policy),
            rgb_relative_path=rgb_path.relative_to(root).as_posix(),
            event_relative_path=event_path.relative_to(root).as_posix(),
        )
        for index, (rgb_path, event_path) in enumerate(zip(rgb, event, strict=True))
    )
    return PairingResult(pairs, tuple(findings), len(rgb), len(event))


def _record_unsupported_files(
    findings: list[ValidationFinding],
    *,
    sequence_id: str,
    modality: str,
    unsupported: list[Path],
) -> None:
    if unsupported:
        findings.append(
            ValidationFinding(
                code="pairing.unsupported_frame_files",
                severity="error",
                message=f"{modality} directory contains unsupported frame files",
                sequence_id=sequence_id,
                details={"files": [path.name for path in unsupported]},
            )
        )


def _validate_rgb_names(
    paths: list[Path], *, sequence_id: str
) -> tuple[ValidationFinding, ...]:
    findings: list[ValidationFinding] = []
    timestamps: list[Decimal] = []
    malformed: list[str] = []
    wrong_sequence: list[str] = []
    for path in paths:
        match = _RGB_FRAME_NAME.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        member_sequence, hour, minute, second, fraction = match.groups()
        if member_sequence != sequence_id:
            wrong_sequence.append(path.name)
        timestamps.append(
            Decimal(hour) * 3600
            + Decimal(minute) * 60
            + Decimal(second)
            + Decimal(f"0.{fraction}")
        )
    if malformed:
        findings.append(
            _name_finding(
                "pairing.rgb_filename_unrecognized", malformed, sequence_id=sequence_id
            )
        )
    if wrong_sequence:
        findings.append(
            _name_finding(
                "pairing.rgb_sequence_mismatch", wrong_sequence, sequence_id=sequence_id
            )
        )
    if len(timestamps) != len(set(timestamps)):
        findings.append(
            ValidationFinding(
                code="pairing.rgb_duplicate_timestamp",
                severity="error",
                message="RGB filenames contain duplicate source timestamps",
                sequence_id=sequence_id,
            )
        )
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        findings.append(
            ValidationFinding(
                code="pairing.rgb_non_monotonic_timestamp",
                severity="error",
                message="naturally ordered RGB filenames are not timestamp-monotonic",
                sequence_id=sequence_id,
            )
        )
    return tuple(findings)


def _validate_event_names(
    paths: list[Path], *, sequence_id: str, timestamp_policy: TimestampConfig
) -> tuple[ValidationFinding, ...]:
    findings: list[ValidationFinding] = []
    counters: list[int] = []
    malformed: list[str] = []
    wrong_sequence: list[str] = []
    for path in paths:
        match = _EVENT_FRAME_NAME.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        member_sequence, counter = match.groups()
        if member_sequence != sequence_id:
            wrong_sequence.append(path.name)
        counters.append(int(counter))
    if malformed:
        findings.append(
            _name_finding(
                "pairing.event_filename_unrecognized", malformed, sequence_id=sequence_id
            )
        )
    if wrong_sequence:
        findings.append(
            _name_finding(
                "pairing.event_sequence_mismatch", wrong_sequence, sequence_id=sequence_id
            )
        )
    if len(counters) != len(set(counters)):
        findings.append(
            ValidationFinding(
                code="pairing.event_duplicate_counter",
                severity="error",
                message="event filenames contain duplicate frame counters",
                sequence_id=sequence_id,
            )
        )
    period_microseconds = Decimal(timestamp_policy.frame_period_seconds) * 1_000_000
    if period_microseconds != period_microseconds.to_integral_value():
        raise ValueError("event filename validation requires an integral microsecond frame period")
    mismatches = []
    for index, counter in enumerate(counters):
        expected = int(
            (Decimal(index) + timestamp_policy.frame_index_offset) * period_microseconds
        )
        if counter != expected:
            mismatches.append(
                {"ordered_index": index, "expected_counter": expected, "actual_counter": counter}
            )
    if mismatches:
        findings.append(
            ValidationFinding(
                code="pairing.event_counter_gap_or_offset",
                severity="error",
                message=(
                    f"{len(mismatches)} event filenames violate the verified counter sequence"
                ),
                sequence_id=sequence_id,
                details={"first_mismatches": mismatches[:20]},
            )
        )
    return tuple(findings)


def _name_finding(
    code: str, paths: list[str], *, sequence_id: str
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        severity="error",
        message=f"{len(paths)} frame filenames do not follow the verified FRED convention",
        sequence_id=sequence_id,
        details={"first_files": paths[:20]},
    )
