"""Pinned official FRED remote source and safe per-sequence materialization."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
import zipfile
from contextlib import contextmanager
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .cache import BoundedCache, CacheEntry, CacheIdentity
from .config import Phase0Config
from .provenance import atomic_write_json, atomic_write_text, sha256_file
from .schema import RemoteObject


class FredRemoteError(RuntimeError):
    pass


class FredRevisionMismatch(FredRemoteError):
    pass


class UnsafeArchiveError(FredRemoteError):
    pass


PREPARED_SEQUENCE_VERSION = "fred-prepared-sequence-v1"
PREPARED_METADATA_NAME = "sequence_metadata.json"
PREPARED_SUCCESS_NAME = "_SUCCESS"


class PreparedSequenceUnavailable(FredRemoteError):
    pass


@dataclass(frozen=True, slots=True)
class ArchiveMemberMetadata:
    archive_member: str
    size_bytes: int
    compressed_size_bytes: int
    crc32: int


@dataclass(frozen=True, slots=True)
class PreparedSequence:
    sequence_id: str
    entry: CacheEntry
    sequence_root: Path
    raw_hdf5: ArchiveMemberMetadata | None


@dataclass(frozen=True, slots=True)
class _SequenceArchiveLayout:
    runtime_member_names: tuple[str, ...]
    runtime_size_bytes: int
    raw_hdf5: ArchiveMemberMetadata | None


def _import_huggingface_hub() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.hf_api import RepoFile
    except ImportError as error:
        raise FredRemoteError(
            "huggingface-hub is required; install the project dependencies before remote access"
        ) from error
    return HfApi, RepoFile, hf_hub_download


def _safe_repo_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise FredRemoteError(f"unsafe remote repository path: {path!r}")
    return candidate.as_posix()


def _logical_reference(dataset_id: str, revision: str, repo_path: str) -> str:
    return f"hf://datasets/{dataset_id}@{revision}/{repo_path}"


def _validated_sequence_archive(sequence_id: str, inventory_record: Any) -> RemoteObject:
    if not re.fullmatch(r"\d+", sequence_id):
        raise ValueError("FRED sequence_id must contain decimal digits only")
    if inventory_record.sequence_id != sequence_id:
        raise ValueError("inventory record sequence ID differs from requested sequence")
    return inventory_record.archive


class HFFredSource:
    def __init__(self, config: Phase0Config, cache: BoundedCache | None = None):
        self.config = config
        self.cache = cache or BoundedCache(
            config.cache.root,
            config.cache.max_bytes,
            lock_timeout_seconds=config.cache.lock_timeout_seconds,
        )
        HfApi, _, _ = _import_huggingface_hub()
        self._api = HfApi()
        self._active_prepared_sequences: dict[tuple[str, str], PreparedSequence] = {}

    def _require_inactive_prepared_window(self, operation: str) -> None:
        if getattr(self, "_active_prepared_sequences", {}):
            raise FredRemoteError(
                f"cannot {operation} while a prepared window is active"
            )

    @property
    def token(self) -> str | bool:
        value = os.getenv(self.config.source.token_environment_variable)
        return value if value else False

    def _with_retries(self, operation_name: str, operation: Any) -> Any:
        errors: list[str] = []
        for attempt in range(1, self.config.source.max_retries + 1):
            try:
                return operation()
            except Exception as error:  # library transports several public exception types
                errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                if attempt < self.config.source.max_retries:
                    time.sleep(self.config.source.retry_backoff_seconds * (2 ** (attempt - 1)))
        raise FredRemoteError(f"{operation_name} failed after bounded retries: {'; '.join(errors)}")

    def verify_revision(self) -> dict[str, Any]:
        info = self._with_retries(
            "dataset revision verification",
            lambda: self._api.dataset_info(
                self.config.source.dataset_id,
                revision=self.config.source.revision,
                files_metadata=True,
                token=self.token,
            ),
        )
        if info.sha != self.config.source.revision:
            raise FredRevisionMismatch(
                f"requested dataset revision {self.config.source.revision}, received {info.sha}"
            )
        if info.id != self.config.source.dataset_id:
            raise FredRevisionMismatch(
                f"requested dataset {self.config.source.dataset_id}, received {info.id}"
            )
        return {
            "dataset_id": info.id,
            "revision": info.sha,
            "private": info.private,
            "gated": info.gated,
            "used_storage": info.used_storage,
            "file_count": len(info.siblings or []),
        }

    def list_objects(self) -> tuple[RemoteObject, ...]:
        _, RepoFile, _ = _import_huggingface_hub()
        entries = self._with_retries(
            "dataset tree listing",
            lambda: list(
                self._api.list_repo_tree(
                    self.config.source.dataset_id,
                    recursive=True,
                    expand=False,
                    revision=self.config.source.revision,
                    repo_type=self.config.source.repo_type,
                    token=self.token,
                )
            ),
        )
        objects: list[RemoteObject] = []
        for entry in entries:
            if not isinstance(entry, RepoFile):
                continue
            path = _safe_repo_path(entry.path)
            lfs = getattr(entry, "lfs", None)
            objects.append(
                RemoteObject(
                    logical_reference=_logical_reference(
                        self.config.source.dataset_id, self.config.source.revision, path
                    ),
                    repo_path=path,
                    size_bytes=getattr(entry, "size", None),
                    sha256=getattr(lfs, "sha256", None) if lfs else None,
                    blob_id=getattr(entry, "blob_id", None),
                )
            )
        return tuple(sorted(objects, key=lambda item: item.repo_path))

    def cache_identity(self, remote_object: RemoteObject) -> CacheIdentity:
        return CacheIdentity(
            provider=self.config.source.provider,
            dataset_id=self.config.source.dataset_id,
            revision=self.config.source.revision,
            logical_reference=remote_object.logical_reference,
            schema_version=self.config.schema_version,
        )

    def materialize_object(self, remote_object: RemoteObject) -> CacheEntry:
        self._require_inactive_prepared_window("materialize remote objects")
        repo_path = _safe_repo_path(remote_object.repo_path)
        expected_reference = _logical_reference(
            self.config.source.dataset_id, self.config.source.revision, repo_path
        )
        if remote_object.logical_reference != expected_reference:
            raise FredRemoteError(
                "remote object logical reference does not match the configured dataset revision"
            )
        _, _, hf_hub_download = _import_huggingface_hub()
        identity = self.cache_identity(remote_object)

        def build(temporary_root: Path) -> Path:
            downloaded = self._with_retries(
                f"download {repo_path}",
                lambda: hf_hub_download(
                    repo_id=self.config.source.dataset_id,
                    filename=repo_path,
                    repo_type=self.config.source.repo_type,
                    revision=self.config.source.revision,
                    local_dir=temporary_root,
                    token=self.token,
                    force_download=False,
                ),
            )
            path = Path(downloaded).resolve()
            if remote_object.size_bytes is not None and path.stat().st_size != remote_object.size_bytes:
                raise FredRemoteError(
                    f"size mismatch for {repo_path}: expected {remote_object.size_bytes}, "
                    f"received {path.stat().st_size}"
                )
            if self.config.cache.verify_content_sha256 and remote_object.sha256:
                digest = sha256_file(path)
                if digest != remote_object.sha256:
                    raise FredRemoteError(
                        f"SHA-256 mismatch for {repo_path}: expected {remote_object.sha256}, got {digest}"
                    )
            return path

        return self.cache.get_or_create(
            identity,
            expected_size_bytes=remote_object.size_bytes,
            builder=build,
        )

    def _prepared_identity(self, remote_object: RemoteObject) -> CacheIdentity:
        return CacheIdentity(
            provider="phase0_prepared_sequence",
            dataset_id=self.config.source.dataset_id,
            revision=self.config.source.revision,
            logical_reference=(
                f"{remote_object.logical_reference}#prepared-{PREPARED_SEQUENCE_VERSION}"
            ),
            schema_version=self.config.schema_version,
        )

    def prepare_sequence(
        self, sequence_id: str, inventory_record: Any
    ) -> PreparedSequence:
        """Atomically prepare one pinned sequence for later read-only opening."""

        self._require_inactive_prepared_window("prepare sequences")
        remote_object = _validated_sequence_archive(sequence_id, inventory_record)
        prepared_identity = self._prepared_identity(remote_object)
        existing = self.cache.get(prepared_identity)
        if existing is not None:
            return _prepared_sequence_from_entry(
                entry=existing,
                sequence_id=sequence_id,
                remote_object=remote_object,
                dataset_id=self.config.source.dataset_id,
                dataset_revision=self.config.source.revision,
                schema_version=self.config.schema_version,
            )

        archive_entry = self.materialize_object(remote_object)
        with zipfile.ZipFile(archive_entry.payload_path) as archive:
            members = archive.infolist()
            _validate_zip_members(members)
            layout = _inspect_sequence_archive(members, sequence_id=sequence_id)

        def build_prepared_entry(temporary_root: Path) -> Path:
            prepared_root = temporary_root / "prepared"
            extraction_root = prepared_root / "sequence"
            extraction_root.mkdir(parents=True)
            with zipfile.ZipFile(archive_entry.payload_path) as archive:
                _validate_zip_members(archive.infolist())
                _extract_runtime_members(
                    archive,
                    member_names=layout.runtime_member_names,
                    destination=extraction_root,
                )
            sequence_root = _find_sequence_root(extraction_root)
            runtime_files = tuple(
                path for path in sequence_root.rglob("*") if path.is_file()
            )
            actual_runtime_size = sum(path.stat().st_size for path in runtime_files)
            if len(runtime_files) != len(layout.runtime_member_names):
                raise FredRemoteError(
                    "prepared sequence member count differs from the validated archive"
                )
            if actual_runtime_size != layout.runtime_size_bytes:
                raise FredRemoteError(
                    "prepared sequence size differs from the validated archive"
                )
            metadata = {
                "format_version": PREPARED_SEQUENCE_VERSION,
                "sequence_id": sequence_id,
                "dataset_id": self.config.source.dataset_id,
                "dataset_revision": self.config.source.revision,
                "schema_version": self.config.schema_version,
                "archive": asdict(remote_object),
                "sequence_relative_path": sequence_root.relative_to(
                    prepared_root
                ).as_posix(),
                "runtime_member_count": len(runtime_files),
                "runtime_size_bytes": actual_runtime_size,
                "raw_hdf5": asdict(layout.raw_hdf5) if layout.raw_hdf5 else None,
            }
            atomic_write_json(prepared_root / PREPARED_METADATA_NAME, metadata)
            atomic_write_text(
                prepared_root / PREPARED_SUCCESS_NAME,
                f"{PREPARED_SEQUENCE_VERSION}\n",
            )
            return prepared_root

        prepared_entry = self.cache.get_or_create(
            prepared_identity,
            expected_size_bytes=layout.runtime_size_bytes,
            builder=build_prepared_entry,
            protected_keys={archive_entry.identity.key},
            entry_metadata={
                "prepared_sequence_format": PREPARED_SEQUENCE_VERSION,
                "sequence_id": sequence_id,
            },
        )

        return _prepared_sequence_from_entry(
            entry=prepared_entry,
            sequence_id=sequence_id,
            remote_object=remote_object,
            dataset_id=self.config.source.dataset_id,
            dataset_revision=self.config.source.revision,
            schema_version=self.config.schema_version,
        )

    @contextmanager
    def activate_prepared_window(
        self, sequences: Iterable[tuple[str, Any]]
    ) -> Iterator[tuple[PreparedSequence, ...]]:
        """Activate one coordinator-owned immutable sequence window."""

        if getattr(self, "_active_prepared_sequences", {}):
            raise FredRemoteError("a prepared sequence window is already active")
        active: dict[tuple[str, str], PreparedSequence] = {}
        sequence_ids: set[str] = set()
        for sequence_id, inventory_record in sequences:
            if sequence_id in sequence_ids:
                raise ValueError(
                    f"prepared window contains duplicate sequence {sequence_id}"
                )
            remote_object = _validated_sequence_archive(sequence_id, inventory_record)
            identity = self._prepared_identity(remote_object)
            entry = self.cache.get_read_only(identity)
            if entry is None:
                raise PreparedSequenceUnavailable(
                    f"sequence {sequence_id} is not prepared; run prepare_sequence first"
                )
            key = (sequence_id, remote_object.logical_reference)
            active[key] = _prepared_sequence_from_entry(
                entry=entry,
                sequence_id=sequence_id,
                remote_object=remote_object,
                dataset_id=self.config.source.dataset_id,
                dataset_revision=self.config.source.revision,
                schema_version=self.config.schema_version,
            )
            sequence_ids.add(sequence_id)
        if not active:
            raise ValueError("prepared window must contain at least one sequence")

        self._active_prepared_sequences = active
        try:
            yield tuple(active.values())
        finally:
            self._active_prepared_sequences = {}

    @contextmanager
    def open_prepared_sequence(
        self, sequence_id: str, inventory_record: Any
    ) -> Iterator[PreparedSequence]:
        """Open one immutable handle from the coordinator's active window."""

        remote_object = _validated_sequence_archive(sequence_id, inventory_record)
        prepared = getattr(self, "_active_prepared_sequences", {}).get(
            (sequence_id, remote_object.logical_reference)
        )
        if prepared is None:
            raise PreparedSequenceUnavailable(
                f"sequence {sequence_id} is not in the active prepared window"
            )
        yield prepared

    def cleanup_cache(
        self,
        *,
        required_bytes: int = 0,
        protected_sequences: Iterable[PreparedSequence] = (),
    ) -> None:
        """Coordinator-only whole-entry cleanup for the bounded cache."""

        self._require_inactive_prepared_window("clean the cache")
        protected_keys = {item.entry.identity.key for item in protected_sequences}
        self.cache.reserve(required_bytes, protected_keys=protected_keys)


def _prepared_sequence_from_entry(
    *,
    entry: CacheEntry,
    sequence_id: str,
    remote_object: RemoteObject,
    dataset_id: str,
    dataset_revision: str,
    schema_version: str,
) -> PreparedSequence:
    prepared_root = entry.payload_path.resolve()
    metadata_path = prepared_root / PREPARED_METADATA_NAME
    success_path = prepared_root / PREPARED_SUCCESS_NAME
    try:
        marker = success_path.read_text(encoding="utf-8")
    except OSError as error:
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has no completion marker; prepare it again"
        ) from error
    if marker != f"{PREPARED_SEQUENCE_VERSION}\n":
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has an incompatible completion marker"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has invalid metadata; prepare it again"
        ) from error

    expected_values = {
        "format_version": PREPARED_SEQUENCE_VERSION,
        "sequence_id": sequence_id,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "schema_version": schema_version,
        "archive": asdict(remote_object),
    }
    for name, expected in expected_values.items():
        if metadata.get(name) != expected:
            raise PreparedSequenceUnavailable(
                f"prepared sequence {sequence_id} has incompatible {name} metadata"
            )
    if entry.entry_metadata != {
        "prepared_sequence_format": PREPARED_SEQUENCE_VERSION,
        "sequence_id": sequence_id,
    }:
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has incompatible cache metadata"
        )

    relative_path = metadata.get("sequence_relative_path")
    if not isinstance(relative_path, str):
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has no sequence path metadata"
        )
    sequence_root = (prepared_root / PurePosixPath(relative_path)).resolve()
    if prepared_root != sequence_root and prepared_root not in sequence_root.parents:
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has an unsafe sequence path"
        )
    if not (
        (sequence_root / "coordinates.txt").is_file()
        and (sequence_root / "RGB").is_dir()
        and (sequence_root / "Event" / "Frames").is_dir()
    ):
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has incomplete runtime content"
        )
    if not isinstance(metadata.get("runtime_member_count"), int) or metadata[
        "runtime_member_count"
    ] < 3:
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has invalid runtime member metadata"
        )
    if not isinstance(metadata.get("runtime_size_bytes"), int) or metadata[
        "runtime_size_bytes"
    ] <= 0:
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has invalid runtime size metadata"
        )

    raw = metadata.get("raw_hdf5")
    if raw is not None and not isinstance(raw, dict):
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has invalid raw HDF5 metadata"
        )
    try:
        raw_hdf5 = ArchiveMemberMetadata(**raw) if raw is not None else None
    except (TypeError, ValueError) as error:
        raise PreparedSequenceUnavailable(
            f"prepared sequence {sequence_id} has invalid archive metadata"
        ) from error
    return PreparedSequence(
        sequence_id=sequence_id,
        entry=entry,
        sequence_root=sequence_root,
        raw_hdf5=raw_hdf5,
    )


def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
    seen: set[str] = set()
    for member in members:
        candidate = PurePosixPath(member.filename)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise UnsafeArchiveError(f"unsafe archive member path: {member.filename!r}")
        normalized = candidate.as_posix()
        if normalized in seen:
            raise UnsafeArchiveError(f"duplicate archive member: {normalized!r}")
        seen.add(normalized)
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise UnsafeArchiveError(f"symbolic links are not allowed in FRED archives: {normalized}")


def _inspect_sequence_archive(
    members: list[zipfile.ZipInfo], *, sequence_id: str
) -> _SequenceArchiveLayout:
    files = [member for member in members if not member.is_dir()]
    coordinate_members = [
        member
        for member in files
        if PurePosixPath(member.filename).name == "coordinates.txt"
    ]
    layouts: list[tuple[PurePosixPath, list[zipfile.ZipInfo]]] = []
    for coordinate in coordinate_members:
        prefix = PurePosixPath(coordinate.filename).parent
        if prefix.parts and prefix.name != sequence_id:
            continue
        selected = [
            member
            for member in files
            if _is_runtime_member(
                _relative_to_prefix(PurePosixPath(member.filename), prefix)
            )
        ]
        relative_names = {
            _relative_to_prefix(PurePosixPath(member.filename), prefix).as_posix()
            for member in selected
        }
        has_rgb = any(name.startswith("RGB/") for name in relative_names)
        has_event = any(name.startswith("Event/Frames/") for name in relative_names)
        if "coordinates.txt" in relative_names and has_rgb and has_event:
            layouts.append((prefix, selected))
    if len(layouts) != 1:
        raise FredRemoteError(
            f"expected exactly one valid FRED sequence layout, found {len(layouts)}"
        )

    prefix, selected = layouts[0]
    hdf5_members = [
        member
        for member in files
        if _relative_to_prefix(PurePosixPath(member.filename), prefix)
        == PurePosixPath("Event/events.hdf5")
    ]
    if len(hdf5_members) > 1:
        raise FredRemoteError("expected at most one Event/events.hdf5 archive member")
    raw_hdf5 = None
    if hdf5_members:
        member = hdf5_members[0]
        raw_hdf5 = ArchiveMemberMetadata(
            archive_member=PurePosixPath(member.filename).as_posix(),
            size_bytes=member.file_size,
            compressed_size_bytes=member.compress_size,
            crc32=member.CRC,
        )
    return _SequenceArchiveLayout(
        runtime_member_names=tuple(
            member.filename for member in sorted(selected, key=lambda item: item.filename)
        ),
        runtime_size_bytes=sum(member.file_size for member in selected),
        raw_hdf5=raw_hdf5,
    )


def _relative_to_prefix(path: PurePosixPath, prefix: PurePosixPath) -> PurePosixPath:
    if not prefix.parts:
        return path
    try:
        return path.relative_to(prefix)
    except ValueError:
        return PurePosixPath("__outside_sequence__")


def _is_runtime_member(relative: PurePosixPath) -> bool:
    if relative == PurePosixPath("coordinates.txt"):
        return True
    parts = relative.parts
    return bool(
        (len(parts) == 2 and parts[0] == "RGB" and relative.suffix.lower() == ".jpg")
        or (
            len(parts) == 3
            and parts[:2] == ("Event", "Frames")
            and relative.suffix.lower() == ".png"
        )
    )


def _extract_runtime_members(
    archive: zipfile.ZipFile, *, member_names: tuple[str, ...], destination: Path
) -> None:
    root = destination.resolve()
    for member_name in member_names:
        member = archive.getinfo(member_name)
        target = (root / PurePosixPath(member_name)).resolve()
        if root != target and root not in target.parents:
            raise UnsafeArchiveError(f"archive member escapes extraction root: {member_name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def _find_sequence_root(extracted_root: Path) -> Path:
    candidates = [
        path
        for path in (extracted_root, *sorted(extracted_root.rglob("*")))
        if path.is_dir()
        and (path / "RGB").is_dir()
        and (path / "Event" / "Frames").is_dir()
        and (path / "coordinates.txt").is_file()
    ]
    if len(candidates) != 1:
        raise FredRemoteError(
            f"expected exactly one FRED sequence root, found {len(candidates)} in archive"
        )
    return candidates[0]
