"""Bounded, version-aware, atomically populated cache for remote FRED objects."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, stable_hash, utc_now


class CacheError(RuntimeError):
    pass


class CacheCapacityError(CacheError):
    pass


class CacheLockTimeout(CacheError):
    pass


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    provider: str
    dataset_id: str
    revision: str
    logical_reference: str
    schema_version: str

    @property
    def key(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class CacheEntry:
    identity: CacheIdentity
    payload_path: Path
    size_bytes: int
    created_at: str
    last_accessed_at: str
    entry_metadata: dict[str, Any] = field(default_factory=dict)


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _pid_file_owner_is_alive(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
        pid = int(content.split()[0].removeprefix("pid="))
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, ValueError, IndexError):
        return False
    return True


class _ExclusiveLock:
    def __init__(self, path: Path, timeout_seconds: float, stale_seconds: float = 3600):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self._acquired = False

    def __enter__(self) -> "_ExclusiveLock":
        deadline = time.monotonic() + self.timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(f"pid={os.getpid()} created={time.time()}\n")
                self._acquired = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds and not self._owner_is_alive():
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise CacheLockTimeout(f"timed out waiting for cache lock {self.path}")
                time.sleep(0.1)

    def _owner_is_alive(self) -> bool:
        return _pid_file_owner_is_alive(self.path)

    def __exit__(self, *_: object) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)


class BoundedCache:
    """Cache whole logical entries so eviction cannot leave partial sequences."""

    METADATA_NAME = "entry.json"
    CAPACITY_LOCK_NAME = ".capacity.lock"

    def __init__(self, root: str | Path, max_bytes: int, *, lock_timeout_seconds: float = 60):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root).expanduser().resolve()
        self.max_bytes = max_bytes
        self.lock_timeout_seconds = lock_timeout_seconds
        self.entries_root = self.root / "entries"
        self.locks_root = self.root / "locks"
        self.temporary_root = self.root / "temporary"
        self.entries_root.mkdir(parents=True, exist_ok=True)
        self.locks_root.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self._validated_keys: set[str] = set()
        self._touched_keys: set[str] = set()
        self._state_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_state_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._state_lock = threading.Lock()

    def _entry_root(self, identity: CacheIdentity) -> Path:
        return self.entries_root / identity.key

    def _load_entry(self, identity: CacheIdentity, *, touch: bool) -> CacheEntry | None:
        root = self._entry_root(identity)
        metadata_path = root / self.METADATA_NAME
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["identity"] != asdict(identity):
                return None
            root = root.resolve()
            payload = (root / metadata["payload_relative_path"]).resolve()
            if root != payload and root not in payload.parents:
                return None
            if not payload.exists():
                return None
            with self._state_lock:
                needs_validation = identity.key not in self._validated_keys
            if needs_validation:
                actual_size = _tree_size(root)
                if actual_size != metadata["size_bytes"]:
                    return None
                with self._state_lock:
                    self._validated_keys.add(identity.key)
            else:
                actual_size = int(metadata["size_bytes"])
            with self._state_lock:
                should_touch = touch and identity.key not in self._touched_keys
                if should_touch:
                    self._touched_keys.add(identity.key)
            if should_touch:
                metadata["last_accessed_at"] = utc_now()
                atomic_write_json(metadata_path, metadata)
            return CacheEntry(
                identity=identity,
                payload_path=payload,
                size_bytes=actual_size,
                created_at=metadata["created_at"],
                last_accessed_at=metadata["last_accessed_at"],
                entry_metadata=dict(metadata.get("entry_metadata", {})),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def get(self, identity: CacheIdentity) -> CacheEntry | None:
        return self._load_entry(identity, touch=True)

    def get_read_only(self, identity: CacheIdentity) -> CacheEntry | None:
        return self._load_entry(identity, touch=False)

    def usage_bytes(self) -> int:
        return sum(
            self._recorded_entry_size(path)
            for path in self.entries_root.iterdir()
            if path.is_dir()
        )

    def _recorded_entry_size(self, root: Path) -> int:
        try:
            metadata = json.loads((root / self.METADATA_NAME).read_text(encoding="utf-8"))
            size = int(metadata["size_bytes"])
            if size < 0:
                raise ValueError("negative cache size")
            return size
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return _tree_size(root)

    def _eviction_candidates(self, protected_keys: set[str]) -> list[tuple[str, Path, int]]:
        candidates: list[tuple[str, Path, int]] = []
        for root in self.entries_root.iterdir():
            if not root.is_dir() or root.name in protected_keys:
                continue
            metadata_path = root / self.METADATA_NAME
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                last_accessed = str(metadata["last_accessed_at"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                last_accessed = ""
            candidates.append((last_accessed, root, self._recorded_entry_size(root)))
        return sorted(candidates, key=lambda item: (item[0], item[1].name))

    def _building_entry_keys(self) -> set[str]:
        return {
            path.name.removesuffix(".lock")
            for path in self.locks_root.glob("*.lock")
            if path.name != self.CAPACITY_LOCK_NAME
        }

    def _reserve_unlocked(
        self, required_bytes: int, *, protected_keys: set[str] | None = None
    ) -> None:
        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        if required_bytes > self.max_bytes:
            raise CacheCapacityError(
                f"entry needs {required_bytes} bytes but cache capacity is {self.max_bytes}"
            )
        protected = (protected_keys or set()) | self._building_entry_keys()
        usage = self.usage_bytes()
        for _, path, size in self._eviction_candidates(protected):
            if usage + required_bytes <= self.max_bytes:
                break
            shutil.rmtree(path)
            with self._state_lock:
                self._validated_keys.discard(path.name)
                self._touched_keys.discard(path.name)
            usage -= size
        if usage + required_bytes > self.max_bytes:
            raise CacheCapacityError(
                f"cannot reserve {required_bytes} bytes within {self.max_bytes}-byte cache"
            )

    def reserve(self, required_bytes: int, *, protected_keys: set[str] | None = None) -> None:
        lock = self.locks_root / self.CAPACITY_LOCK_NAME
        with _ExclusiveLock(lock, self.lock_timeout_seconds):
            self._reserve_unlocked(required_bytes, protected_keys=protected_keys)

    def _discard_invalid_entry(self, identity: CacheIdentity) -> None:
        path = self._entry_root(identity)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        with self._state_lock:
            self._validated_keys.discard(identity.key)
            self._touched_keys.discard(identity.key)

    def get_or_create(
        self,
        identity: CacheIdentity,
        *,
        expected_size_bytes: int | None,
        builder: Callable[[Path], str | Path],
        protected_keys: set[str] | None = None,
        entry_metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        existing = self.get(identity)
        if existing is not None:
            return existing
        lock_path = self.locks_root / f"{identity.key}.lock"
        with _ExclusiveLock(lock_path, self.lock_timeout_seconds):
            existing = self.get(identity)
            if existing is not None:
                return existing
            self._discard_invalid_entry(identity)
            if expected_size_bytes is not None:
                self.reserve(expected_size_bytes, protected_keys=protected_keys)

            temporary = Path(tempfile.mkdtemp(prefix=f".{identity.key}.", dir=self.temporary_root))
            final = self._entry_root(identity)
            try:
                payload = Path(builder(temporary)).resolve()
                if temporary != payload and temporary not in payload.parents:
                    raise CacheError("cache builder returned a payload outside its temporary entry")
                if not payload.exists():
                    raise CacheError("cache builder did not create its declared payload")
                created = utc_now()
                metadata = {
                    "identity": asdict(identity),
                    "payload_relative_path": str(payload.relative_to(temporary)),
                    "created_at": created,
                    "last_accessed_at": created,
                    "size_bytes": 0,
                    "entry_metadata": entry_metadata or {},
                }
                atomic_write_json(temporary / self.METADATA_NAME, metadata)
                for _ in range(4):
                    actual_size = _tree_size(temporary)
                    if metadata["size_bytes"] == actual_size:
                        break
                    metadata["size_bytes"] = actual_size
                    atomic_write_json(temporary / self.METADATA_NAME, metadata)
                metadata["size_bytes"] = _tree_size(temporary)
                if metadata["size_bytes"] > self.max_bytes:
                    raise CacheCapacityError(
                        f"built entry is {metadata['size_bytes']} bytes, exceeding cache capacity"
                    )
                capacity_lock = self.locks_root / self.CAPACITY_LOCK_NAME
                with _ExclusiveLock(capacity_lock, self.lock_timeout_seconds):
                    self._reserve_unlocked(
                        metadata["size_bytes"],
                        protected_keys=(protected_keys or set()) | {identity.key},
                    )
                    os.replace(temporary, final)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        entry = self.get(identity)
        if entry is None:
            raise CacheError("cache entry failed validation immediately after creation")
        return entry
