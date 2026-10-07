"""Atomic SQLite backend for the versioned canonical FRED manifest."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path

from .provenance import stable_hash, stable_json_dumps, utc_now
from .schema import FREDSample


MANIFEST_BACKEND = "sqlite-v1"


class ManifestError(RuntimeError):
    pass


class ManifestWriter:
    """Write a complete manifest to a temporary DB and promote it atomically."""

    def __init__(self, destination: str | Path, metadata: dict[str, object]):
        self.destination = Path(destination).resolve()
        self.metadata = dict(metadata)
        self._temporary: Path | None = None
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> "ManifestWriter":
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.destination.name}.", suffix=".tmp", dir=self.destination.parent
        )
        os.close(descriptor)
        self._temporary = Path(name)
        self._connection = sqlite3.connect(self._temporary)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()
        return self

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ManifestError("manifest writer is not open")
        return self._connection

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            CREATE TABLE samples (
                sample_id TEXT PRIMARY KEY,
                sequence_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL CHECK(frame_index >= 0),
                timestamp TEXT NOT NULL,
                rgb_logical_reference TEXT NOT NULL,
                rgb_archive_member TEXT NOT NULL,
                rgb_width INTEGER,
                rgb_height INTEGER,
                event_logical_reference TEXT NOT NULL,
                event_archive_member TEXT NOT NULL,
                event_width INTEGER,
                event_height INTEGER,
                official_split TEXT NOT NULL,
                project_split TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                UNIQUE(sequence_id, frame_index)
            );
            CREATE INDEX idx_samples_project_split ON samples(project_split, sequence_id, frame_index);
            CREATE INDEX idx_samples_official_split ON samples(official_split, sequence_id, frame_index);
            CREATE TABLE annotations (
                sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
                annotation_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                track_id INTEGER NOT NULL,
                original_class TEXT NOT NULL,
                source_line INTEGER,
                PRIMARY KEY(sample_id, annotation_index)
            );
            """
        )

    def add_sample(self, sample: FREDSample) -> None:
        provenance_json = json.dumps(asdict(sample.provenance), sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.sample_id,
                sample.sequence_id,
                sample.frame_index,
                sample.timestamp,
                sample.rgb.remote_logical_reference,
                sample.rgb.archive_member,
                sample.rgb.width,
                sample.rgb.height,
                sample.event.remote_logical_reference,
                sample.event.archive_member,
                sample.event.width,
                sample.event.height,
                sample.official_split.value,
                sample.project_split.value,
                provenance_json,
            ),
        )
        self.connection.executemany(
            """
            INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    sample.sample_id,
                    index,
                    annotation.timestamp,
                    *annotation.box_xyxy,
                    annotation.track_id,
                    annotation.original_class,
                    annotation.source_line,
                )
                for index, annotation in enumerate(sample.annotations)
            ],
        )

    def _records_sha256(self) -> str:
        digest = hashlib.sha256()
        queries = (
            (
                "samples",
                "SELECT * FROM samples ORDER BY CAST(sequence_id AS INTEGER), "
                "frame_index, sample_id",
            ),
            (
                "annotations",
                "SELECT * FROM annotations ORDER BY sample_id, annotation_index",
            ),
        )
        for table, query in queries:
            digest.update(f"{table}\n".encode("utf-8"))
            for row in self.connection.execute(query):
                digest.update(stable_json_dumps(list(row)).encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()

    def __exit__(self, exception_type: object, *_: object) -> None:
        assert self._temporary is not None
        if exception_type is not None:
            if self._connection is not None:
                self._connection.close()
            self._temporary.unlink(missing_ok=True)
            return
        try:
            sample_count = self.connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            if sample_count <= 0:
                raise ManifestError("refusing to publish an empty canonical manifest")
            final_metadata = {
                **self.metadata,
                "backend": MANIFEST_BACKEND,
                "created_at": utc_now(),
                "sample_count": sample_count,
                "records_sha256": self._records_sha256(),
            }
            final_metadata["content_identity"] = stable_hash(
                {key: value for key, value in final_metadata.items() if key != "created_at"}
            )
            self.connection.executemany(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                [
                    (key, json.dumps(value, sort_keys=True))
                    for key, value in sorted(final_metadata.items())
                ],
            )
            self.connection.commit()
            integrity = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ManifestError(f"SQLite integrity check failed: {integrity}")
            self.connection.close()
            os.replace(self._temporary, self.destination)
        except BaseException:
            if self._connection is not None:
                self._connection.close()
            self._temporary.unlink(missing_ok=True)
            raise


def read_manifest_metadata(path: str | Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as connection:
        return {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value_json FROM metadata")
        }
