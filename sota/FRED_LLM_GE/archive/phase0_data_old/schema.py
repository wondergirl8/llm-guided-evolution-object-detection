"""Versioned, model-neutral semantic types for the FRED Phase 0 boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


SCHEMA_VERSION = "P0-CONTRACT-FRED-SAMPLE-v1"


class OfficialSplit(StrEnum):
    CHALLENGING_TRAIN = "challenging_train"
    CHALLENGING_TEST = "challenging_test"


class ProjectSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELD_OUT_TEST = "held_out_test"
    UNASSIGNED = "unassigned"


class Modality(StrEnum):
    RGB = "rgb"
    EVENT = "event"
    RGB_EVENT = "rgb_event"


class AccessMode(StrEnum):
    TRAIN = "train"
    EVALUATION_INPUT = "evaluation_input"
    TRUSTED_EVALUATOR = "trusted_evaluator"


@dataclass(frozen=True, slots=True)
class RemoteObject:
    logical_reference: str
    repo_path: str
    size_bytes: int | None = None
    sha256: str | None = None
    blob_id: str | None = None


@dataclass(frozen=True, slots=True)
class Annotation:
    timestamp: str
    box_xyxy: tuple[float, float, float, float]
    track_id: int
    original_class: str
    source_line: int | None = None


@dataclass(frozen=True, slots=True)
class ModalityReference:
    remote_logical_reference: str
    archive_member: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class SampleProvenance:
    fred_dataset_identity: str
    fred_dataset_revision: str
    fred_repository_revision: str
    phase0_schema_version: str
    manifest_version: str


@dataclass(frozen=True, slots=True)
class FREDSample:
    sample_id: str
    sequence_id: str
    frame_index: int
    timestamp: str
    rgb: ModalityReference
    event: ModalityReference
    annotations: tuple[Annotation, ...]
    official_split: OfficialSplit
    project_split: ProjectSplit
    provenance: SampleProvenance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    severity: str
    message: str
    sequence_id: str | None = None
    sample_id: str | None = None
    source_reference: str | None = None
    line_number: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_sample_id(sequence_id: str, frame_index: int) -> str:
    if not sequence_id or ":" in sequence_id:
        raise ValueError("sequence_id must be non-empty and must not contain ':'")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return f"{sequence_id}:{frame_index}"
