import json
import multiprocessing
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sota.FRED_LLM_GE.phase0_data import fred_api as fred_api_module
from sota.FRED_LLM_GE.phase0_data.cache import BoundedCache
from sota.FRED_LLM_GE.phase0_data.fred_api import (
    FredRemoteError,
    HFFredSource,
    PREPARED_METADATA_NAME,
    PREPARED_SEQUENCE_VERSION,
    PREPARED_SUCCESS_NAME,
    PreparedSequenceUnavailable,
)
from sota.FRED_LLM_GE.phase0_data.schema import RemoteObject


DATASET_ID = "owner/fred"
REVISION = "a" * 40
SCHEMA_VERSION = "schema-v1"
REMOTE = RemoteObject(
    f"hf://datasets/{DATASET_ID}@{REVISION}/train/0.zip",
    "train/0.zip",
)
RECORD = SimpleNamespace(sequence_id="0", archive=REMOTE)


def _source(cache_root: Path, max_bytes: int = 100_000) -> HFFredSource:
    source = object.__new__(HFFredSource)
    source.config = SimpleNamespace(
        source=SimpleNamespace(
            provider="huggingface_hub",
            dataset_id=DATASET_ID,
            revision=REVISION,
        ),
        schema_version=SCHEMA_VERSION,
    )
    source.cache = BoundedCache(cache_root, max_bytes)
    return source


def _write_sequence_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("0/RGB/0.jpg", b"rgb")
        archive.writestr("0/Event/Frames/0.png", b"event")
        archive.writestr("0/coordinates.txt", b"coordinates")
        archive.writestr("0/Event/events.hdf5", b"raw-event-stream")


def _copy(source: Path, destination: Path) -> Path:
    destination.write_bytes(source.read_bytes())
    return destination


def _source_with_cached_archive(tmp_path: Path):
    archive_path = tmp_path / "0.zip"
    _write_sequence_archive(archive_path)
    source = _source(tmp_path / "cache")
    remote = RemoteObject(
        REMOTE.logical_reference,
        REMOTE.repo_path,
        archive_path.stat().st_size,
    )
    record = SimpleNamespace(sequence_id="0", archive=remote)
    archive_entry = source.cache.get_or_create(
        source.cache_identity(remote),
        expected_size_bytes=archive_path.stat().st_size,
        builder=lambda root: _copy(archive_path, root / "0.zip"),
    )
    calls = []

    def materialize(requested):
        calls.append(requested)
        return archive_entry

    source.materialize_object = materialize
    return source, remote, record, archive_entry, calls


def _provider_access_forbidden(_remote_object):
    raise AssertionError("prepared reader attempted provider access")


def _spawned_prepared_reader(
    source: HFFredSource, remote: RemoteObject, result_queue
) -> None:
    record = SimpleNamespace(sequence_id="0", archive=remote)
    try:
        with source.open_prepared_sequence("0", record) as prepared:
            result_queue.put(
                (
                    "ok",
                    (prepared.sequence_root / "coordinates.txt").read_text(
                        encoding="utf-8"
                    ),
                )
            )
    except BaseException as error:
        result_queue.put(("error", f"{type(error).__name__}: {error}"))


def test_prepare_open_and_cleanup_are_explicit_versioned_phases(
    tmp_path: Path, monkeypatch
):
    source, remote, record, archive_entry, calls = _source_with_cached_archive(tmp_path)

    first = source.prepare_sequence("0", record)
    second = source.prepare_sequence("0", record)

    assert calls == [remote]
    assert first.entry.identity == second.entry.identity
    assert first.entry.identity.logical_reference.endswith(
        f"#prepared-{PREPARED_SEQUENCE_VERSION}"
    )
    assert (first.sequence_root / "RGB" / "0.jpg").read_bytes() == b"rgb"
    assert (first.sequence_root / "Event" / "Frames" / "0.png").read_bytes() == b"event"
    assert first.raw_hdf5.archive_member == "0/Event/events.hdf5"
    prepared_root = first.entry.payload_path
    assert (prepared_root / PREPARED_SUCCESS_NAME).read_text(encoding="utf-8") == (
        f"{PREPARED_SEQUENCE_VERSION}\n"
    )
    metadata = json.loads(
        (prepared_root / PREPARED_METADATA_NAME).read_text(encoding="utf-8")
    )
    assert metadata["format_version"] == PREPARED_SEQUENCE_VERSION
    assert metadata["dataset_revision"] == REVISION
    assert metadata["archive"]["logical_reference"] == remote.logical_reference

    source.materialize_object = _provider_access_forbidden
    monkeypatch.setattr(
        zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: pytest.fail("prepared opening reopened the ZIP"),
    )
    with source.activate_prepared_window((("0", record),)) as active:
        assert active[0].entry.identity == first.entry.identity
        for _ in range(2):
            with source.open_prepared_sequence("0", record) as opened:
                assert opened.sequence_root == first.sequence_root
        with pytest.raises(FredRemoteError, match="cannot clean"):
            source.cleanup_cache(required_bytes=1)
        with pytest.raises(FredRemoteError, match="cannot prepare"):
            source.prepare_sequence("0", record)

    source.cache.max_bytes = first.entry.size_bytes + 16
    source.cleanup_cache(required_bytes=1, protected_sequences=(first,))
    assert source.cache.get(first.entry.identity) is not None
    assert source.cache.get(archive_entry.identity) is None


def test_open_prepared_sequence_never_falls_back_to_provider(tmp_path: Path):
    source = _source(tmp_path / "cache")
    source.materialize_object = _provider_access_forbidden

    with pytest.raises(PreparedSequenceUnavailable, match="run prepare_sequence first"):
        with source.activate_prepared_window((("0", RECORD),)):
            pass


def test_incompatible_prepared_marker_is_rejected(tmp_path: Path):
    source, _remote, record, _archive_entry, _calls = _source_with_cached_archive(
        tmp_path
    )
    prepared = source.prepare_sequence("0", record)
    marker = prepared.entry.payload_path / PREPARED_SUCCESS_NAME
    marker.write_text("x" * len(marker.read_text(encoding="utf-8")), encoding="utf-8")

    reopened = _source(source.cache.root, source.cache.max_bytes)
    reopened.materialize_object = _provider_access_forbidden
    with pytest.raises(PreparedSequenceUnavailable, match="completion marker"):
        with reopened.activate_prepared_window((("0", record),)):
            pass


def test_interrupted_preparation_never_publishes_partial_entry(
    tmp_path: Path, monkeypatch
):
    source, remote, record, _archive_entry, _calls = _source_with_cached_archive(
        tmp_path
    )

    def interrupted_extract(_archive, *, member_names, destination):
        assert member_names
        partial = destination / "0" / "RGB" / "partial.jpg"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"partial")
        raise RuntimeError("preparation interrupted")

    monkeypatch.setattr(
        fred_api_module, "_extract_runtime_members", interrupted_extract
    )

    with pytest.raises(RuntimeError, match="preparation interrupted"):
        source.prepare_sequence("0", record)

    assert source.cache.get(source._prepared_identity(remote)) is None
    assert tuple(source.cache.temporary_root.iterdir()) == ()


def test_spawned_processes_open_one_prepared_entry_read_only(tmp_path: Path):
    source, remote, record, _archive_entry, _calls = _source_with_cached_archive(
        tmp_path
    )
    prepared = source.prepare_sequence("0", record)
    source.materialize_object = _provider_access_forbidden
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    with source.activate_prepared_window((("0", record),)):
        processes = [
            context.Process(
                target=_spawned_prepared_reader,
                args=(source, remote, result_queue),
            )
            for _ in range(2)
        ]

        for process in processes:
            process.start()
        with pytest.raises(FredRemoteError, match="cannot clean"):
            source.cleanup_cache(required_bytes=1)
        results = [result_queue.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert source.cache.get_read_only(prepared.entry.identity) is not None

    assert sorted(results) == [("ok", "coordinates"), ("ok", "coordinates")]

    source.cache.max_bytes = 1
    source.cleanup_cache()
    assert source.cache.get_read_only(prepared.entry.identity) is None


def test_interrupted_active_window_releases_coordinator_state(tmp_path: Path):
    source, _remote, record, _archive_entry, _calls = _source_with_cached_archive(
        tmp_path
    )
    source.prepare_sequence("0", record)

    with pytest.raises(RuntimeError, match="worker failure"):
        with source.activate_prepared_window((("0", record),)):
            raise RuntimeError("worker failure")

    assert getattr(source, "_active_prepared_sequences", {}) == {}
    source.cleanup_cache()
