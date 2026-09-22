"""Model-neutral manifest reader with protected label-access modes."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from .fred_api import HFFredSource, PreparedSequenceUnavailable
from .inventory import SourceInventory
from .manifest import read_manifest_metadata
from .schema import AccessMode, Annotation, Modality, ProjectSplit


class DatasetAccessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LoadedSample:
    sample_id: str
    sequence_id: str
    frame_index: int
    timestamp: str
    modality: Modality
    rgb: Image.Image | None
    event: Image.Image | None
    annotations: tuple[Annotation, ...] | None
    metadata: dict[str, Any]


class FREDDataset:
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        inventory: SourceInventory,
        source: HFFredSource,
        project_split: ProjectSplit | str,
        modality: Modality | str,
        access_mode: AccessMode | str,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.inventory = inventory
        self.source = source
        self.project_split = ProjectSplit(project_split)
        self.modality = Modality(modality)
        self.access_mode = AccessMode(access_mode)
        if self.access_mode == AccessMode.TRAIN and self.project_split != ProjectSplit.TRAIN:
            raise DatasetAccessError("training access is allowed only for the project-train split")
        self._connection: sqlite3.Connection | None = None
        self._connection_pid: int | None = None
        self._inventory_by_sequence = self.inventory.by_sequence()
        self._validate_identity()
        self._sample_ids = self._read_sample_ids()

    def _validate_identity(self) -> None:
        metadata = read_manifest_metadata(self.manifest_path)
        required = ("dataset_id", "dataset_revision", "schema_version")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise DatasetAccessError(f"manifest is missing identity metadata: {missing}")
        source_config = getattr(getattr(self.source, "config", None), "source", None)
        if source_config is None:
            raise DatasetAccessError("remote source does not expose its pinned identity")
        expected = (metadata["dataset_id"], metadata["dataset_revision"])
        if expected != (self.inventory.dataset_id, self.inventory.dataset_revision):
            raise DatasetAccessError("manifest and inventory source identities differ")
        if expected != (source_config.dataset_id, source_config.revision):
            raise DatasetAccessError("manifest and remote source identities differ")

    def _connect(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._connection is not None and self._connection_pid == pid:
            return self._connection
        self.close()
        self._connection = sqlite3.connect(f"file:{self.manifest_path}?mode=ro", uri=True)
        self._connection.row_factory = sqlite3.Row
        self._connection_pid = pid
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._connection_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_connection_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _read_sample_ids(self) -> tuple[str, ...]:
        rows = self._connect().execute(
            """
            SELECT sample_id FROM samples WHERE project_split = ?
            ORDER BY CAST(sequence_id AS INTEGER), frame_index
            """,
            (self.project_split.value,),
        ).fetchall()
        return tuple(row["sample_id"] for row in rows)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, index: int) -> LoadedSample:
        sample_id = self._sample_ids[index]
        connection = self._connect()
        row = connection.execute(
            "SELECT * FROM samples WHERE sample_id = ?", (sample_id,)
        ).fetchone()
        if row is None:
            raise DatasetAccessError(f"manifest lost indexed sample {sample_id}")
        annotations: tuple[Annotation, ...] | None = None
        if self.access_mode != AccessMode.EVALUATION_INPUT:
            annotations = tuple(
                Annotation(
                    timestamp=item["timestamp"],
                    box_xyxy=(item["x1"], item["y1"], item["x2"], item["y2"]),
                    track_id=item["track_id"],
                    original_class=item["original_class"],
                    source_line=item["source_line"],
                )
                for item in connection.execute(
                    "SELECT * FROM annotations WHERE sample_id = ? ORDER BY annotation_index",
                    (sample_id,),
                )
            )

        inventory_record = self._inventory_by_sequence.get(row["sequence_id"])
        if inventory_record is None:
            raise DatasetAccessError(f"sequence {row['sequence_id']} is absent from pinned inventory")
        rgb = None
        event = None
        with self._open_prepared_sequence(
            row["sequence_id"], inventory_record
        ) as materialized:
            if self.modality in (Modality.RGB, Modality.RGB_EVENT):
                rgb = self._load_image(
                    materialized.sequence_root, row["rgb_archive_member"]
                )
            if self.modality in (Modality.EVENT, Modality.RGB_EVENT):
                event = self._load_image(
                    materialized.sequence_root, row["event_archive_member"]
                )

        metadata = {"provenance": json.loads(row["provenance_json"])}
        if self.access_mode != AccessMode.EVALUATION_INPUT:
            metadata.update(
                {
                    "official_split": row["official_split"],
                    "project_split": row["project_split"],
                }
            )
        return LoadedSample(
            sample_id=sample_id,
            sequence_id=row["sequence_id"],
            frame_index=row["frame_index"],
            timestamp=row["timestamp"],
            modality=self.modality,
            rgb=rgb,
            event=event,
            annotations=annotations,
            metadata=metadata,
        )

    @contextmanager
    def _open_prepared_sequence(
        self, sequence_id: str, inventory_record: Any
    ) -> Iterator[Any]:
        opener = getattr(self.source, "open_prepared_sequence", None)
        if opener is None:
            raise DatasetAccessError(
                "remote source does not support prepared sequence reads; "
                "prepare the sequence before dataset access"
            )
        stack = ExitStack()
        try:
            prepared = stack.enter_context(opener(sequence_id, inventory_record))
        except PreparedSequenceUnavailable as error:
            stack.close()
            raise DatasetAccessError(
                f"sequence {sequence_id} is unavailable to the active prepared "
                "window; the coordinator must prepare and activate it before "
                "dataset access"
            ) from error
        with stack:
            yield prepared

    @staticmethod
    def _load_image(root: Path, relative_path: str) -> Image.Image:
        candidate = (root / relative_path).resolve()
        if root != candidate and root not in candidate.parents:
            raise DatasetAccessError(f"manifest image path escapes sequence root: {relative_path}")
        if not candidate.is_file():
            raise DatasetAccessError(f"manifest image is missing: {candidate}")
        with Image.open(candidate) as image:
            image.load()
            return image.copy()
