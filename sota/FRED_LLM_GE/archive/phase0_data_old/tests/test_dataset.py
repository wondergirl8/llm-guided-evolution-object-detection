import multiprocessing
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import zipfile

import pytest
from PIL import Image

from sota.FRED_LLM_GE.phase0_data import cache as cache_module
from sota.FRED_LLM_GE.phase0_data import dataset as dataset_module
from sota.FRED_LLM_GE.phase0_data import fred_api as fred_api_module
from sota.FRED_LLM_GE.phase0_data.cache import BoundedCache
from sota.FRED_LLM_GE.phase0_data.dataset import DatasetAccessError, FREDDataset
from sota.FRED_LLM_GE.phase0_data.fred_api import (
    HFFredSource,
    PreparedSequenceUnavailable,
)
from sota.FRED_LLM_GE.phase0_data.inventory import (
    SequenceInventoryRecord,
    SourceInventory,
)
from sota.FRED_LLM_GE.phase0_data.manifest import ManifestWriter
from sota.FRED_LLM_GE.phase0_data.schema import (
    AccessMode,
    Annotation,
    FREDSample,
    Modality,
    ModalityReference,
    OfficialSplit,
    ProjectSplit,
    RemoteObject,
    SampleProvenance,
)


def make_sample() -> FREDSample:
    return FREDSample(
        sample_id="0:0",
        sequence_id="0",
        frame_index=0,
        timestamp="0.033333",
        rgb=ModalityReference(
            "hf://archive", "RGB/Video_0_00_00_00.000000.jpg", 32, 24
        ),
        event=ModalityReference(
            "hf://archive", "Event/Frames/Video_0_frame_33333.png", 32, 24
        ),
        annotations=(Annotation("0.033333", (1, 2, 10, 12), 7, "drone", 1),),
        official_split=OfficialSplit.CHALLENGING_TRAIN,
        project_split=ProjectSplit.TRAIN,
        provenance=SampleProvenance("fred", "a" * 40, "b" * 40, "v1", "m1"),
    )


def make_inventory() -> SourceInventory:
    record = SequenceInventoryRecord(
        "0", "train", RemoteObject("hf://archive", "train/0.zip", 100, "c" * 64, "blob")
    )
    return SourceInventory("fred", "a" * 40, "v1", (record,), "hash", "now")


class FakeSource:
    def __init__(self, root: Path):
        self.root = root
        self.config = SimpleNamespace(
            source=SimpleNamespace(dataset_id="fred", revision="a" * 40)
        )

    @contextmanager
    def open_prepared_sequence(self, sequence_id, record):
        yield SimpleNamespace(sequence_root=self.root)


class MissingPreparedSource(FakeSource):
    @contextmanager
    def open_prepared_sequence(self, sequence_id, record):
        raise PreparedSequenceUnavailable("sequence is not prepared")
        yield


def _provider_access_forbidden(*_args, **_kwargs):
    raise AssertionError("spawned dataset worker attempted provider access")


def _spawned_dataset_reader(dataset: FREDDataset, result_queue) -> None:
    try:
        loaded = dataset[0]
        result_queue.put(
            (
                "ok",
                loaded.sample_id,
                loaded.rgb.size if loaded.rgb is not None else None,
            )
        )
    except BaseException as error:
        result_queue.put(("error", f"{type(error).__name__}: {error}"))
    finally:
        dataset.close()


def write_manifest(path: Path):
    with ManifestWriter(
        path,
        {
            "manifest_version": "v1",
            "schema_version": "v1",
            "dataset_id": "fred",
            "dataset_revision": "a" * 40,
        },
    ) as writer:
        writer.add_sample(make_sample())


def _real_prepared_source(sequence_root: Path, tmp_path: Path) -> HFFredSource:
    archive_path = tmp_path / "0.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sequence_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=f"0/{path.relative_to(sequence_root).as_posix()}",
                )

    source = object.__new__(HFFredSource)
    source.config = SimpleNamespace(
        source=SimpleNamespace(
            provider="test",
            dataset_id="fred",
            revision="a" * 40,
        ),
        schema_version="v1",
    )
    source.cache = BoundedCache(tmp_path / "cache", 2_000_000)
    record = make_inventory().records[0]

    def cache_archive(root: Path) -> Path:
        destination = root / "0.zip"
        destination.write_bytes(archive_path.read_bytes())
        return destination

    archive_entry = source.cache.get_or_create(
        source.cache_identity(record.archive),
        expected_size_bytes=archive_path.stat().st_size,
        builder=cache_archive,
    )
    source.materialize_object = lambda _remote: archive_entry
    source.prepare_sequence("0", record)
    return source


def _entry_snapshot(cache: BoundedCache) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(cache.entries_root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(cache.entries_root.rglob("*"))
        if path.is_file()
    }


def test_candidate_facing_evaluation_never_receives_labels(sequence_root: Path, tmp_path: Path):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=FakeSource(sequence_root),
        project_split="train",
        modality="rgb_event",
        access_mode="evaluation_input",
    )
    loaded = dataset[0]
    assert loaded.annotations is None
    assert loaded.rgb.size == (32, 24)
    assert loaded.event.size == (32, 24)
    assert "official_split" not in loaded.metadata
    assert "project_split" not in loaded.metadata


def test_evaluation_input_does_not_query_annotation_storage(
    sequence_root: Path, tmp_path: Path
):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    with sqlite3.connect(manifest) as connection:
        connection.execute("DROP TABLE annotations")
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=FakeSource(sequence_root),
        project_split="train",
        modality="rgb",
        access_mode="evaluation_input",
    )
    assert dataset[0].annotations is None


def test_training_access_is_restricted_to_project_train(sequence_root: Path, tmp_path: Path):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    with pytest.raises(DatasetAccessError, match="project-train"):
        FREDDataset(
            manifest_path=manifest,
            inventory=make_inventory(),
            source=FakeSource(sequence_root),
            project_split="validation",
            modality=Modality.RGB,
            access_mode=AccessMode.TRAIN,
        )


def test_manifest_source_identity_mismatch_is_rejected(sequence_root: Path, tmp_path: Path):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    mismatched = make_inventory()
    mismatched = SourceInventory(
        mismatched.dataset_id,
        "f" * 40,
        mismatched.schema_version,
        mismatched.records,
        mismatched.inventory_hash,
        mismatched.created_at,
    )
    with pytest.raises(DatasetAccessError, match="identities differ"):
        FREDDataset(
            manifest_path=manifest,
            inventory=mismatched,
            source=FakeSource(sequence_root),
            project_split="train",
            modality="rgb",
            access_mode="train",
        )


def test_loader_preserves_released_event_image_mode(sequence_root: Path, tmp_path: Path):
    Image.new("L", (32, 24), 1).save(
        sequence_root / "Event" / "Frames" / "Video_0_frame_33333.png"
    )
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=FakeSource(sequence_root),
        project_split="train",
        modality="event",
        access_mode="train",
    )
    assert dataset[0].event.mode == "L"


@pytest.mark.parametrize(
    ("modality", "expected_paths", "has_rgb", "has_event"),
    [
        (
            Modality.RGB,
            ["RGB/Video_0_00_00_00.000000.jpg"],
            True,
            False,
        ),
        (
            Modality.EVENT,
            ["Event/Frames/Video_0_frame_33333.png"],
            False,
            True,
        ),
        (
            Modality.RGB_EVENT,
            [
                "RGB/Video_0_00_00_00.000000.jpg",
                "Event/Frames/Video_0_frame_33333.png",
            ],
            True,
            True,
        ),
    ],
)
def test_loader_decodes_exactly_the_requested_modalities(
    sequence_root: Path,
    tmp_path: Path,
    monkeypatch,
    modality: Modality,
    expected_paths: list[str],
    has_rgb: bool,
    has_event: bool,
):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    loaded_paths = []

    def load_image(root: Path, relative_path: str):
        loaded_paths.append(relative_path)
        return Image.new("L", (32, 24))

    monkeypatch.setattr(FREDDataset, "_load_image", staticmethod(load_image))
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=FakeSource(sequence_root),
        project_split="train",
        modality=modality,
        access_mode="train",
    )

    loaded = dataset[0]
    assert (loaded.rgb is not None) is has_rgb
    assert (loaded.event is not None) is has_event
    assert loaded_paths == expected_paths


def test_missing_prepared_modality_fails_without_fallback(
    sequence_root: Path, tmp_path: Path
):
    missing = sequence_root / "RGB" / "Video_0_00_00_00.000000.jpg"
    missing.unlink()
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=FakeSource(sequence_root),
        project_split="train",
        modality="rgb",
        access_mode="train",
    )

    with pytest.raises(DatasetAccessError, match="manifest image is missing"):
        dataset[0]


def test_missing_prepared_sequence_requires_explicit_preparation(
    sequence_root: Path, tmp_path: Path
):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=MissingPreparedSource(sequence_root),
        project_split="train",
        modality="rgb",
        access_mode="train",
    )

    with pytest.raises(
        DatasetAccessError, match="prepare and activate it before dataset access"
    ):
        dataset[0]


def test_loader_reuses_one_manifest_connection_per_process(
    sequence_root: Path, tmp_path: Path, monkeypatch
):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    real_connect = dataset_module.sqlite3.connect
    connections = []

    def counting_connect(*args, **kwargs):
        connections.append(args[0])
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(dataset_module.sqlite3, "connect", counting_connect)
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=FakeSource(sequence_root),
        project_split="train",
        modality="rgb",
        access_mode="train",
    )
    initialization_connections = len(connections)

    dataset[0]
    dataset[0]
    assert len(connections) == initialization_connections


def test_repeated_getitem_has_no_remote_extraction_or_cache_metadata_side_effects(
    sequence_root: Path, tmp_path: Path, monkeypatch
):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    source = _real_prepared_source(sequence_root, tmp_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("dataset sample loading attempted mutable cache or provider work")

    source.materialize_object = forbidden
    source.cleanup_cache = forbidden
    monkeypatch.setattr(fred_api_module, "_extract_runtime_members", forbidden)
    monkeypatch.setattr(cache_module, "atomic_write_json", forbidden)
    monkeypatch.setattr(source.cache, "reserve", forbidden)

    inventory = make_inventory()
    before = _entry_snapshot(source.cache)
    with source.activate_prepared_window((("0", inventory.records[0]),)):
        monkeypatch.setattr(source.cache, "get", forbidden)
        monkeypatch.setattr(source.cache, "get_read_only", forbidden)
        dataset = FREDDataset(
            manifest_path=manifest,
            inventory=inventory,
            source=source,
            project_split="train",
            modality="rgb_event",
            access_mode="train",
        )

        first = dataset[0]
        second = dataset[0]

        assert first.sample_id == second.sample_id == "0:0"
        assert first.rgb is not None and first.event is not None
        assert second.rgb is not None and second.event is not None
    assert _entry_snapshot(source.cache) == before


def test_spawned_process_reopens_manifest_read_only_without_breaking_parent(
    sequence_root: Path, tmp_path: Path
):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    dataset = FREDDataset(
        manifest_path=manifest,
        inventory=make_inventory(),
        source=FakeSource(sequence_root),
        project_split="train",
        modality="rgb",
        access_mode="train",
    )
    assert dataset[0].sample_id == "0:0"

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_spawned_dataset_reader,
        args=(dataset, result_queue),
    )
    process.start()
    result = result_queue.get(timeout=10)
    process.join(timeout=10)

    assert process.exitcode == 0
    assert result == ("ok", "0:0", (32, 24))
    assert dataset[0].sample_id == "0:0"


def test_spawned_process_reads_real_prepared_cache_without_metadata_writes(
    sequence_root: Path, tmp_path: Path
):
    manifest = tmp_path / "manifest.sqlite"
    write_manifest(manifest)
    source = _real_prepared_source(sequence_root, tmp_path)
    source.materialize_object = _provider_access_forbidden
    before = _entry_snapshot(source.cache)
    inventory = make_inventory()
    with source.activate_prepared_window((("0", inventory.records[0]),)):
        dataset = FREDDataset(
            manifest_path=manifest,
            inventory=inventory,
            source=source,
            project_split="train",
            modality="rgb",
            access_mode="train",
        )

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_spawned_dataset_reader,
            args=(dataset, result_queue),
        )
        process.start()
        result = result_queue.get(timeout=10)
        process.join(timeout=10)

    assert process.exitcode == 0
    assert result == ("ok", "0:0", (32, 24))
    assert _entry_snapshot(source.cache) == before
