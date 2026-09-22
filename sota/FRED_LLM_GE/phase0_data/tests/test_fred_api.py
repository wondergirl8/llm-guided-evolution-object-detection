import hashlib
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sota.FRED_LLM_GE.phase0_data import fred_api as fred_api_module
from sota.FRED_LLM_GE.phase0_data.cache import BoundedCache, CacheIdentity
from sota.FRED_LLM_GE.phase0_data.fred_api import (
    FredRemoteError,
    HFFredSource,
    PREPARED_SEQUENCE_VERSION,
    UnsafeArchiveError,
    _import_huggingface_hub,
    _validate_zip_members,
)
from sota.FRED_LLM_GE.phase0_data.schema import RemoteObject


DATASET_ID = "owner/fred"
REVISION = "a" * 40
SCHEMA_VERSION = "schema-v1"
REMOTE = RemoteObject(
    f"hf://datasets/{DATASET_ID}@{REVISION}/train/0.zip",
    "train/0.zip",
)


def _source_config(*, verify_content_sha256: bool = False):
    return SimpleNamespace(
        source=SimpleNamespace(
            provider="huggingface_hub",
            dataset_id=DATASET_ID,
            revision=REVISION,
            repo_type="dataset",
            token_environment_variable="FRED_PHASE0_TEST_TOKEN",
            max_retries=1,
            retry_backoff_seconds=0,
        ),
        schema_version=SCHEMA_VERSION,
        cache=SimpleNamespace(verify_content_sha256=verify_content_sha256),
    )


def _source_with_cache(cache_root: Path, max_bytes: int = 100_000):
    source = object.__new__(HFFredSource)
    source.config = _source_config()
    source.cache = BoundedCache(cache_root, max_bytes)
    return source


def _provider_access_forbidden(_remote_object):
    raise AssertionError("read-only worker attempted provider access")


def test_pinned_hugging_face_client_symbols_are_importable():
    api, repo_file, download = _import_huggingface_hub()
    assert api.__name__ == "HfApi"
    assert repo_file.__name__ == "RepoFile"
    assert download.__name__ == "hf_hub_download"


def test_archive_path_traversal_is_rejected():
    with pytest.raises(UnsafeArchiveError, match="unsafe"):
        _validate_zip_members([zipfile.ZipInfo("../escape.txt")])


def test_archive_symlinks_are_rejected():
    member = zipfile.ZipInfo("link")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(UnsafeArchiveError, match="Symbolic|symbolic"):
        _validate_zip_members([member])


def test_duplicate_archive_members_are_rejected():
    with pytest.raises(UnsafeArchiveError, match="duplicate"):
        _validate_zip_members([zipfile.ZipInfo("same"), zipfile.ZipInfo("same")])


def test_materialization_rejects_mismatched_logical_reference_before_download():
    source = object.__new__(HFFredSource)
    source.config = SimpleNamespace(
        source=SimpleNamespace(dataset_id="owner/fred", revision="a" * 40)
    )
    remote = RemoteObject("hf://datasets/other/fred@bad/train/0.zip", "train/0.zip")
    with pytest.raises(FredRemoteError, match="logical reference"):
        source.materialize_object(remote)


def test_materialization_uses_exact_pin_and_verifies_size_and_hash(
    tmp_path: Path, monkeypatch
):
    content = b"pinned-object"
    digest = hashlib.sha256(content).hexdigest()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        return str(path)

    monkeypatch.delenv("FRED_PHASE0_TEST_TOKEN", raising=False)
    monkeypatch.setattr(
        fred_api_module,
        "_import_huggingface_hub",
        lambda: (object, object, download),
    )
    source = _source_with_cache(tmp_path / "cache")
    source.config = _source_config(verify_content_sha256=True)
    remote = RemoteObject(
        REMOTE.logical_reference,
        REMOTE.repo_path,
        len(content),
        digest,
        "pinned-blob",
    )

    entry = source.materialize_object(remote)

    assert entry.payload_path.read_bytes() == content
    assert entry.identity == CacheIdentity(
        "huggingface_hub",
        DATASET_ID,
        REVISION,
        REMOTE.logical_reference,
        SCHEMA_VERSION,
    )
    assert calls == [
        {
            "repo_id": DATASET_ID,
            "filename": "train/0.zip",
            "repo_type": "dataset",
            "revision": REVISION,
            "local_dir": calls[0]["local_dir"],
            "token": False,
            "force_download": False,
        }
    ]


@pytest.mark.parametrize(
    ("expected_size", "expected_sha256", "message"),
    [
        (999, hashlib.sha256(b"pinned-object").hexdigest(), "size mismatch"),
        (len(b"pinned-object"), "0" * 64, "SHA-256 mismatch"),
    ],
)
def test_size_or_hash_mismatch_never_publishes_download(
    tmp_path: Path,
    monkeypatch,
    expected_size: int,
    expected_sha256: str,
    message: str,
):
    content = b"pinned-object"

    def download(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        return str(path)

    monkeypatch.setattr(
        fred_api_module,
        "_import_huggingface_hub",
        lambda: (object, object, download),
    )
    source = _source_with_cache(tmp_path / "cache")
    source.config = _source_config(verify_content_sha256=True)
    remote = RemoteObject(
        REMOTE.logical_reference,
        REMOTE.repo_path,
        expected_size,
        expected_sha256,
    )

    with pytest.raises(FredRemoteError, match=message):
        source.materialize_object(remote)

    assert source.cache.get(source.cache_identity(remote)) is None
    assert tuple(source.cache.temporary_root.iterdir()) == ()


def test_interrupted_download_cleans_partial_temporary_content(tmp_path: Path, monkeypatch):
    def interrupted_download(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"partial")
        raise RuntimeError("connection interrupted")

    monkeypatch.setattr(
        fred_api_module,
        "_import_huggingface_hub",
        lambda: (object, object, interrupted_download),
    )
    source = _source_with_cache(tmp_path / "cache")

    with pytest.raises(FredRemoteError, match="connection interrupted"):
        source.materialize_object(REMOTE)

    assert source.cache.get(source.cache_identity(REMOTE)) is None
    assert tuple(source.cache.temporary_root.iterdir()) == ()


def test_sequence_preparation_excludes_raw_hdf5_from_runtime_cache(
    tmp_path: Path, monkeypatch
):
    archive_path = tmp_path / "0.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("0/RGB/0.jpg", b"rgb")
        archive.writestr("0/Event/Frames/0.png", b"event")
        archive.writestr("0/coordinates.txt", b"0.033333: 1,2,3,4,1,drone\n")
        archive.writestr("0/coordinates_rgb.txt", b"not canonical")
        archive.writestr("0/Event/events.hdf5", b"raw-event-stream")

    source = object.__new__(HFFredSource)
    source.config = SimpleNamespace(
        source=SimpleNamespace(
            provider="huggingface_hub",
            dataset_id="owner/fred",
            revision="a" * 40,
        ),
        schema_version="schema-v1",
    )
    source.cache = BoundedCache(tmp_path / "cache", 100_000)
    remote = RemoteObject(
        f"hf://datasets/owner/fred@{'a' * 40}/train/0.zip",
        "train/0.zip",
        archive_path.stat().st_size,
    )
    archive_entry = source.cache.get_or_create(
        source.cache_identity(remote),
        expected_size_bytes=archive_path.stat().st_size,
        builder=lambda root: _copy_archive(archive_path, root / "0.zip"),
    )
    source.materialize_object = lambda _: archive_entry
    record = SimpleNamespace(sequence_id="0", archive=remote)

    result = source.prepare_sequence("0", record)

    assert (result.sequence_root / "RGB" / "0.jpg").read_bytes() == b"rgb"
    assert (result.sequence_root / "Event" / "Frames" / "0.png").read_bytes() == b"event"
    assert not (result.sequence_root / "Event" / "events.hdf5").exists()
    assert not (result.sequence_root / "coordinates_rgb.txt").exists()
    assert result.raw_hdf5.archive_member == "0/Event/events.hdf5"
    assert result.raw_hdf5.size_bytes == len(b"raw-event-stream")
    assert result.entry.identity.logical_reference.endswith(
        f"#prepared-{PREPARED_SEQUENCE_VERSION}"
    )

    warm_source = object.__new__(HFFredSource)
    warm_source.config = source.config
    warm_source.cache = BoundedCache(tmp_path / "cache", 100_000)
    warm_source.materialize_object = _provider_access_forbidden
    monkeypatch.setattr(
        zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: pytest.fail("warm access reopened source ZIP"),
    )
    with warm_source.activate_prepared_window((("0", record),)):
        for _ in range(2):
            with warm_source.open_prepared_sequence("0", record) as warm_result:
                assert warm_result.raw_hdf5 == result.raw_hdf5
                assert warm_result.sequence_root == result.sequence_root


def _copy_archive(source: Path, destination: Path) -> Path:
    destination.write_bytes(source.read_bytes())
    return destination
