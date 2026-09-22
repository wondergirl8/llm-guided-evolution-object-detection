"""Sequence inspection and canonical sample validation."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .annotations import AnnotationParseError, parse_coordinates_file
from .config import Phase0Config
from .fred_api import PreparedSequence
from .inventory import SequenceInventoryRecord
from .provenance import atomic_write_json, utc_now
from .schema import (
    FREDSample,
    ModalityReference,
    OfficialSplit,
    ProjectSplit,
    SampleProvenance,
    ValidationFinding,
    stable_sample_id,
)
from .splits import OfficialSplitManifest, ProjectSplitManifest
from .synchronization import pair_sequence_frames


@dataclass(frozen=True, slots=True)
class SequenceInspection:
    sequence_id: str
    samples: tuple[FREDSample, ...]
    findings: tuple[ValidationFinding, ...]
    rgb_count: int
    event_count: int
    annotation_count: int
    raw_hdf5_size_bytes: int | None

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)


def _split_for_sequence(
    sequence_id: str,
    official: OfficialSplitManifest,
    project: ProjectSplitManifest,
) -> tuple[OfficialSplit, ProjectSplit]:
    if sequence_id in official.challenging_test:
        official_value = OfficialSplit.CHALLENGING_TEST
    elif sequence_id in official.challenging_train:
        official_value = OfficialSplit.CHALLENGING_TRAIN
    else:
        raise ValueError(f"sequence {sequence_id} is absent from official challenging split")
    if sequence_id in project.train:
        project_value = ProjectSplit.TRAIN
    elif sequence_id in project.validation:
        project_value = ProjectSplit.VALIDATION
    elif sequence_id in project.held_out_test:
        project_value = ProjectSplit.HELD_OUT_TEST
    else:
        raise ValueError(f"sequence {sequence_id} is absent from approved project split")
    return official_value, project_value


def inspect_sequence(
    *,
    config: Phase0Config,
    inventory_record: SequenceInventoryRecord,
    prepared_sequence: PreparedSequence,
    official_split: OfficialSplitManifest,
    project_split: ProjectSplitManifest,
) -> SequenceInspection:
    sequence_id = inventory_record.sequence_id
    root = prepared_sequence.sequence_root
    findings: list[ValidationFinding] = []
    pairing = pair_sequence_frames(
        root, sequence_id=sequence_id, timestamp_policy=config.timestamp
    )
    findings.extend(pairing.findings)
    raw_hdf5_size = _validate_raw_hdf5(
        prepared_sequence,
        sequence_id,
        inventory_record.archive.logical_reference,
        findings,
    )
    try:
        parsed = parse_coordinates_file(root / "coordinates.txt", strict=False)
        findings.extend(parsed.findings)
    except AnnotationParseError as error:
        findings.extend(error.findings)
        parsed = None
    if parsed is None:
        return SequenceInspection(
            sequence_id,
            (),
            tuple(findings),
            pairing.rgb_count,
            pairing.event_count,
            0,
            raw_hdf5_size,
        )

    official_value, project_value = _split_for_sequence(
        sequence_id, official_split, project_split
    )
    associated = _associate_annotations(
        parsed.annotations,
        pairing.pairs,
        tolerance=Decimal(config.timestamp.tolerance_seconds),
        sequence_id=sequence_id,
        findings=findings,
    )
    samples: list[FREDSample] = []
    archive_ref = inventory_record.archive.logical_reference
    for pair in pairing.pairs:
        annotations = associated.get(pair.frame_index, ())
        rgb_size = _image_size(root / pair.rgb_relative_path, sequence_id, findings)
        event_size = _image_size(root / pair.event_relative_path, sequence_id, findings)
        if rgb_size and event_size and rgb_size != event_size:
            findings.append(
                ValidationFinding(
                    code="pairing.dimension_mismatch",
                    severity="error",
                    message=f"RGB dimensions {rgb_size} differ from event dimensions {event_size}",
                    sequence_id=sequence_id,
                    sample_id=stable_sample_id(sequence_id, pair.frame_index),
                )
            )
        if rgb_size:
            width, height = rgb_size
            for annotation in annotations:
                x1, y1, x2, y2 = annotation.box_xyxy
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    findings.append(
                        ValidationFinding(
                            code="annotation.out_of_bounds",
                            severity="error",
                            message=f"box {annotation.box_xyxy} is outside {width}x{height}",
                            sequence_id=sequence_id,
                            sample_id=stable_sample_id(sequence_id, pair.frame_index),
                            line_number=annotation.source_line,
                        )
                    )
        samples.append(
            FREDSample(
                sample_id=stable_sample_id(sequence_id, pair.frame_index),
                sequence_id=sequence_id,
                frame_index=pair.frame_index,
                timestamp=pair.timestamp,
                rgb=ModalityReference(
                    remote_logical_reference=archive_ref,
                    archive_member=pair.rgb_relative_path,
                    width=rgb_size[0] if rgb_size else None,
                    height=rgb_size[1] if rgb_size else None,
                ),
                event=ModalityReference(
                    remote_logical_reference=archive_ref,
                    archive_member=pair.event_relative_path,
                    width=event_size[0] if event_size else None,
                    height=event_size[1] if event_size else None,
                ),
                annotations=annotations,
                official_split=official_value,
                project_split=project_value,
                provenance=SampleProvenance(
                    fred_dataset_identity=config.source.dataset_id,
                    fred_dataset_revision=config.source.revision,
                    fred_repository_revision=config.fred_repository.revision,
                    phase0_schema_version=config.schema_version,
                    manifest_version=config.manifest_version,
                ),
            )
        )
    return SequenceInspection(
        sequence_id=sequence_id,
        samples=tuple(samples),
        findings=tuple(findings),
        rgb_count=pairing.rgb_count,
        event_count=pairing.event_count,
        annotation_count=len(parsed.annotations),
        raw_hdf5_size_bytes=raw_hdf5_size,
    )


def _associate_annotations(
    annotations: tuple[Any, ...],
    pairs: tuple[Any, ...],
    *,
    tolerance: Decimal,
    sequence_id: str,
    findings: list[ValidationFinding],
) -> dict[int, tuple[Any, ...]]:
    if tolerance < 0:
        raise ValueError("timestamp tolerance must be non-negative")
    frame_timestamps = [Decimal(pair.timestamp) for pair in pairs]
    associated: dict[int, list[Any]] = {}
    previous: Decimal | None = None
    for annotation in annotations:
        try:
            timestamp = Decimal(annotation.timestamp)
        except InvalidOperation:
            continue  # The strict parser already emitted a blocking finding.
        if previous is not None and timestamp < previous:
            findings.append(
                ValidationFinding(
                    code="annotation.non_monotonic_timestamp",
                    severity="error",
                    message=(
                        f"annotation timestamp {annotation.timestamp} follows later "
                        f"timestamp {previous}"
                    ),
                    sequence_id=sequence_id,
                    source_reference="coordinates.txt",
                    line_number=annotation.source_line,
                )
            )
        previous = timestamp
        position = bisect_left(frame_timestamps, timestamp)
        candidate_indexes = {
            index for index in (position - 1, position) if 0 <= index < len(frame_timestamps)
        }
        matches = [
            index
            for index in candidate_indexes
            if abs(frame_timestamps[index] - timestamp) <= tolerance
        ]
        if len(matches) != 1:
            code = (
                "annotation.unmatched_timestamp"
                if not matches
                else "annotation.ambiguous_timestamp"
            )
            findings.append(
                ValidationFinding(
                    code=code,
                    severity="error",
                    message=(
                        f"annotation timestamp {annotation.timestamp} matched "
                        f"{len(matches)} frames within tolerance {tolerance}"
                    ),
                    sequence_id=sequence_id,
                    source_reference="coordinates.txt",
                    line_number=annotation.source_line,
                )
            )
            continue
        associated.setdefault(pairs[matches[0]].frame_index, []).append(annotation)
    return {frame_index: tuple(items) for frame_index, items in associated.items()}


def _validate_raw_hdf5(
    prepared_sequence: PreparedSequence,
    sequence_id: str,
    archive_logical_reference: str,
    findings: list[ValidationFinding],
) -> int | None:
    metadata = prepared_sequence.raw_hdf5
    if metadata is None:
        findings.append(
            ValidationFinding(
                code="raw_hdf5.missing_from_archive",
                severity="error",
                message="pinned sequence archive has no Event/events.hdf5 member",
                sequence_id=sequence_id,
                source_reference=archive_logical_reference,
            )
        )
        return None
    if metadata.size_bytes <= 0:
        findings.append(
            ValidationFinding(
                code="raw_hdf5.empty_archive_member",
                severity="error",
                message="expected a non-empty Event/events.hdf5 archive member",
                sequence_id=sequence_id,
                source_reference=metadata.archive_member,
            )
        )
        return None
    return metadata.size_bytes


def _image_size(
    path: Path, sequence_id: str, findings: list[ValidationFinding]
) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as error:
        findings.append(
            ValidationFinding(
                code="image.unreadable",
                severity="error",
                message=str(error),
                sequence_id=sequence_id,
                source_reference=str(path),
            )
        )
        return None


def write_validation_report(
    inspections: list[SequenceInspection],
    path: str | Path,
    *,
    context: dict[str, Any],
    expected_sequence_ids: tuple[str, ...] | None = None,
    execution_error: dict[str, str] | None = None,
) -> None:
    findings = [finding.to_dict() for item in inspections for finding in item.findings]
    inspected_ids = [item.sequence_id for item in inspections]
    expected_ids = list(
        dict.fromkeys(expected_sequence_ids if expected_sequence_ids is not None else inspected_ids)
    )
    expected_set = set(expected_ids)
    inspected_set = set(inspected_ids)
    missing_ids = [sequence_id for sequence_id in expected_ids if sequence_id not in inspected_set]
    unexpected_ids = sorted(inspected_set - expected_set)
    duplicate_ids = sorted(
        sequence_id
        for sequence_id, count in Counter(inspected_ids).items()
        if count > 1
    )
    scope_complete = bool(inspections) and not (
        missing_ids or unexpected_ids or duplicate_ids
    )
    data_valid = not any(item["severity"] == "error" for item in findings)
    valid = execution_error is None and scope_complete and data_valid
    atomic_write_json(
        path,
        {
            "schema_version": "phase0-validation-report-v2",
            "created_at": utc_now(),
            "context": context,
            "status": "passed" if valid else "failed",
            "valid": valid,
            "execution": {
                "status": "completed" if execution_error is None else "failed",
                "error": execution_error,
            },
            "scope": {
                "complete": scope_complete,
                "expected_sequence_ids": expected_ids,
                "inspected_sequence_ids": inspected_ids,
                "missing_sequence_ids": missing_ids,
                "unexpected_sequence_ids": unexpected_ids,
                "duplicate_sequence_ids": duplicate_ids,
            },
            "sequence_count": len(inspections),
            "sample_count": sum(len(item.samples) for item in inspections),
            "annotation_count": sum(item.annotation_count for item in inspections),
            "sequences": [
                {
                    "sequence_id": item.sequence_id,
                    "valid": item.is_valid,
                    "rgb_count": item.rgb_count,
                    "event_count": item.event_count,
                    "annotation_count": item.annotation_count,
                    "raw_hdf5_size_bytes": item.raw_hdf5_size_bytes,
                    "first_sample_id": item.samples[0].sample_id if item.samples else None,
                    "last_sample_id": item.samples[-1].sample_id if item.samples else None,
                    "first_annotation_timestamp": _first_annotation_timestamp(item),
                    "last_annotation_timestamp": _last_annotation_timestamp(item),
                }
                for item in inspections
            ],
            "findings": findings,
        },
    )


def _first_annotation_timestamp(inspection: SequenceInspection) -> str | None:
    for sample in inspection.samples:
        if sample.annotations:
            return sample.annotations[0].timestamp
    return None


def _last_annotation_timestamp(inspection: SequenceInspection) -> str | None:
    for sample in reversed(inspection.samples):
        if sample.annotations:
            return sample.annotations[-1].timestamp
    return None
