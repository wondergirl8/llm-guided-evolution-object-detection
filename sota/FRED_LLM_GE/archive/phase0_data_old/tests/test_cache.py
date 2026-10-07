from pathlib import Path
import threading

import pytest

from sota.FRED_LLM_GE.phase0_data import cache as cache_module
from sota.FRED_LLM_GE.phase0_data.cache import (
    BoundedCache,
    CacheCapacityError,
    CacheIdentity,
)


def identity(name: str) -> CacheIdentity:
    return CacheIdentity("test", "fred", "a" * 40, name, "v1")


def test_cache_hit_does_not_run_builder_twice(tmp_path: Path):
    cache = BoundedCache(tmp_path / "cache", 10_000)
    calls = []

    def builder(root: Path) -> Path:
        calls.append(1)
        payload = root / "payload.bin"
        payload.write_bytes(b"abc")
        return payload

    first = cache.get_or_create(identity("one"), expected_size_bytes=3, builder=builder)
    second = cache.get_or_create(identity("one"), expected_size_bytes=3, builder=builder)
    assert calls == [1]
    assert first.payload_path.read_bytes() == b"abc"
    assert second.identity == first.identity


def test_failed_builder_never_publishes_partial_entry(tmp_path: Path):
    cache = BoundedCache(tmp_path / "cache", 10_000)

    def builder(root: Path) -> Path:
        (root / "partial").write_bytes(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        cache.get_or_create(identity("broken"), expected_size_bytes=7, builder=builder)
    assert cache.get(identity("broken")) is None


def test_in_progress_entry_is_not_observable_before_atomic_publication(tmp_path: Path):
    cache = BoundedCache(tmp_path / "cache", 10_000)
    value = identity("atomic")
    builder_started = threading.Event()
    allow_completion = threading.Event()
    results = []
    errors = []

    def builder(root: Path) -> Path:
        payload = root / "payload.bin"
        payload.write_bytes(b"partial")
        builder_started.set()
        assert allow_completion.wait(timeout=5)
        payload.write_bytes(b"complete")
        return payload

    def populate() -> None:
        try:
            results.append(
                cache.get_or_create(value, expected_size_bytes=8, builder=builder)
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=populate)
    thread.start()
    assert builder_started.wait(timeout=5)
    assert cache.get(value) is None
    allow_completion.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert results[0].payload_path.read_bytes() == b"complete"


def test_identity_includes_revision(tmp_path: Path):
    cache = BoundedCache(tmp_path / "cache", 10_000)

    def builder(root: Path) -> Path:
        path = root / "payload"
        path.write_text("ok")
        return path

    old = identity("object")
    new = CacheIdentity("test", "fred", "b" * 40, "object", "v1")
    assert cache.get_or_create(old, expected_size_bytes=2, builder=builder).identity != cache.get_or_create(
        new, expected_size_bytes=2, builder=builder
    ).identity


def test_corrupt_entry_is_rebuilt_atomically(tmp_path: Path):
    cache = BoundedCache(tmp_path / "cache", 10_000)
    value = identity("corrupt")
    corrupt_root = cache.entries_root / value.key
    corrupt_root.mkdir()
    (corrupt_root / "entry.json").write_text("not-json", encoding="utf-8")

    def builder(root: Path) -> Path:
        path = root / "payload"
        path.write_text("recovered", encoding="utf-8")
        return path

    entry = cache.get_or_create(value, expected_size_bytes=9, builder=builder)
    assert entry.payload_path.read_text(encoding="utf-8") == "recovered"


def test_repeated_cache_hits_skip_tree_walks_and_metadata_rewrites(
    tmp_path: Path, monkeypatch
):
    cache = BoundedCache(tmp_path / "cache", 10_000)
    value = identity("warm")

    def builder(root: Path) -> Path:
        path = root / "payload"
        path.write_text("ready", encoding="utf-8")
        return path

    cache.get_or_create(value, expected_size_bytes=5, builder=builder)
    tree_walks = []
    metadata_writes = []
    monkeypatch.setattr(cache_module, "_tree_size", lambda path: tree_walks.append(path))
    monkeypatch.setattr(
        cache_module,
        "atomic_write_json",
        lambda path, data: metadata_writes.append((path, data)),
    )

    assert cache.get(value) is not None
    assert cache.get(value) is not None
    assert tree_walks == []
    assert metadata_writes == []


def test_oversized_built_entry_is_rejected_without_publication(tmp_path: Path):
    cache = BoundedCache(tmp_path / "cache", 128)
    value = identity("oversized")

    def builder(root: Path) -> Path:
        payload = root / "payload.bin"
        payload.write_bytes(b"x" * 512)
        return payload

    with pytest.raises(CacheCapacityError, match="exceeding cache capacity"):
        cache.get_or_create(value, expected_size_bytes=None, builder=builder)

    assert cache.get(value) is None
    assert tuple(cache.temporary_root.iterdir()) == ()
