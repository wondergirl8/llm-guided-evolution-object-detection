"""Deterministic remote inventory for FRED sequence archives."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fred_api import HFFredSource
from .provenance import atomic_write_json, stable_hash, utc_now
from .schema import RemoteObject


INVENTORY_SCHEMA_VERSION = "fred-remote-inventory-v1"
_ARCHIVE_PATTERN = re.compile(r"^(train|test)/(\d+)\.zip$")


@dataclass(frozen=True, slots=True)
class SequenceInventoryRecord:
    sequence_id: str
    canonical_storage_split: str
    archive: RemoteObject
    rgb_status: str = "pending_content_verification"
    event_frames_status: str = "pending_content_verification"
    raw_hdf5_status: str = "pending_content_verification"
    annotations_status: str = "pending_content_verification"
    validation_status: str = "metadata_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceInventory:
    dataset_id: str
    dataset_revision: str
    schema_version: str
    records: tuple[SequenceInventoryRecord, ...]
    inventory_hash: str
    created_at: str

    def by_sequence(self) -> dict[str, SequenceInventoryRecord]:
        return {record.sequence_id: record for record in self.records}

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "schema_version": self.schema_version,
            "inventory_hash": self.inventory_hash,
            "created_at": self.created_at,
            "records": [record.to_dict() for record in self.records],
        }


def build_inventory(source: HFFredSource) -> SourceInventory:
    source.verify_revision()
    records: list[SequenceInventoryRecord] = []
    seen: set[str] = set()
    for remote_object in source.list_objects():
        match = _ARCHIVE_PATTERN.fullmatch(remote_object.repo_path)
        if not match:
            continue
        storage_split, sequence_id = match.groups()
        if sequence_id in seen:
            raise ValueError(f"duplicate sequence archive for sequence {sequence_id}")
        seen.add(sequence_id)
        records.append(
            SequenceInventoryRecord(
                sequence_id=sequence_id,
                canonical_storage_split=storage_split,
                archive=remote_object,
            )
        )
    records.sort(key=lambda item: int(item.sequence_id))
    if not records:
        raise ValueError("remote inventory did not contain any train/<id>.zip or test/<id>.zip objects")
    hash_payload = [record.to_dict() for record in records]
    return SourceInventory(
        dataset_id=source.config.source.dataset_id,
        dataset_revision=source.config.source.revision,
        schema_version=INVENTORY_SCHEMA_VERSION,
        records=tuple(records),
        inventory_hash=stable_hash(hash_payload),
        created_at=utc_now(),
    )


def write_inventory(inventory: SourceInventory, path: str | Path) -> None:
    atomic_write_json(path, inventory.to_dict())


def load_inventory(path: str | Path) -> SourceInventory:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(
        SequenceInventoryRecord(
            sequence_id=item["sequence_id"],
            canonical_storage_split=item["canonical_storage_split"],
            archive=RemoteObject(**item["archive"]),
            rgb_status=item["rgb_status"],
            event_frames_status=item["event_frames_status"],
            raw_hdf5_status=item["raw_hdf5_status"],
            annotations_status=item["annotations_status"],
            validation_status=item["validation_status"],
        )
        for item in raw["records"]
    )
    calculated = stable_hash([record.to_dict() for record in records])
    if calculated != raw["inventory_hash"]:
        raise ValueError("inventory content hash does not match its records")
    sequence_ids = [record.sequence_id for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("inventory contains duplicate sequence IDs")
    if any(not sequence_id.isdecimal() for sequence_id in sequence_ids):
        raise ValueError("inventory sequence IDs must contain decimal digits only")
    if raw["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise ValueError(f"unsupported inventory schema {raw['schema_version']!r}")
    return SourceInventory(
        dataset_id=raw["dataset_id"],
        dataset_revision=raw["dataset_revision"],
        schema_version=raw["schema_version"],
        records=records,
        inventory_hash=raw["inventory_hash"],
        created_at=raw["created_at"],
    )
