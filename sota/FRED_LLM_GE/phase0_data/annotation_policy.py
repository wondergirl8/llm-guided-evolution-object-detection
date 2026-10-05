"""Explicit, versioned DG-P0-04 handling; source annotations are never edited."""

from __future__ import annotations

import math
from decimal import Decimal


POLICY_VERSION = "dg-p0-04-v1"
DATASET_REVISION = "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd"
REPOSITORY_REVISION = "2bf89c5376eda528431b62d6c60f2c13d8f95ab4"
APPROVAL_REFERENCE = "DG-P0-04; Bill approval in Codex on 2026-10-05"


def validate_policy(policy: str | None) -> None:
    if policy not in (None, POLICY_VERSION):
        raise ValueError(f"unsupported annotation policy: {policy}")


def validate_policy_source(config, policy: str | None) -> None:
    validate_policy(policy)
    if policy is not None and (
        config.source.revision != DATASET_REVISION
        or config.fred_repository.revision != REPOSITORY_REVISION
    ):
        raise ValueError("DG-P0-04 approval applies only to the audited FRED revisions")
    if policy is not None and (
        config.timestamp.policy != "fred_upstream_30hz_v1"
        or config.timestamp.frame_period_seconds != "0.033333"
        or config.timestamp.frame_index_offset != 1
        or Decimal(config.timestamp.tolerance_seconds) != Decimal("0.000001")
    ):
        raise ValueError("DG-P0-04 approval requires the audited timestamp policy")


def permits_unpaired_annotation(sequence_id, annotation, first_frame_timestamp) -> bool:
    """Only the audited first-line records in sequences 225 and 230 may be omitted."""
    return (
        sequence_id in ("225", "230")
        and annotation.source_line == 1
        and Decimal(annotation.timestamp) == 0
        and Decimal(first_frame_timestamp) == Decimal("0.033333")
    )


def converted_box(box, width: int, height: int, policy: str | None = None) -> dict:
    """Clip corners before normalization, retaining both boxes for export lineage."""
    validate_policy(policy)
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("FRED box must have four finite coordinates")
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError("FRED box must have positive area")
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("FRED box has no positive-area image overlap")
    was_clipped = tuple(box) != clipped
    if was_clipped and policy is None:
        raise ValueError("FRED box is outside canonical image coordinates")
    a, b, c, d = clipped
    normalized = ((a+c)/(2*width), (b+d)/(2*height), (c-a)/width, (d-b)/height)
    # A rounded zero-area YOLO label would silently lose a valid source object.
    if float(f"{normalized[2]:.8f}") <= 0 or float(f"{normalized[3]:.8f}") <= 0:
        raise ValueError("FRED box becomes zero area at YOLO label precision")
    return {
        "source_box_xyxy": tuple(box),
        "label_box_xyxy": clipped,
        "image_size": (width, height),
        "normalized_xywh": normalized,
        "clipped": was_clipped,
        "policy_version": policy or "strict_in_bounds_v1",
    }
