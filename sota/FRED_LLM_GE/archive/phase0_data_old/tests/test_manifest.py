from pathlib import Path

import pytest

from sota.FRED_LLM_GE.phase0_data.manifest import (
    ManifestError,
    ManifestWriter,
    read_manifest_metadata,
)
from sota.FRED_LLM_GE.phase0_data.schema import (
    Annotation,
    FREDSample,
    ModalityReference,
    OfficialSplit,
    ProjectSplit,
    SampleProvenance,
)


def sample() -> FREDSample:
    provenance = SampleProvenance("fred", "a" * 40, "b" * 40, "schema-v1", "manifest-v1")
    return FREDSample(
        "0:0",
        "0",
        0,
        "0.033333",
        ModalityReference("hf://archive", "RGB/0.jpg", 32, 24),
        ModalityReference("hf://archive", "Event/Frames/0.png", 32, 24),
        (Annotation("0.033333", (1, 2, 10, 12), 7, "drone", 1),),
        OfficialSplit.CHALLENGING_TRAIN,
        ProjectSplit.TRAIN,
        provenance,
    )


def test_manifest_is_atomic_and_records_metadata(tmp_path: Path):
    path = tmp_path / "manifest.sqlite"
    with ManifestWriter(path, {"manifest_version": "v1"}) as writer:
        writer.add_sample(sample())
    metadata = read_manifest_metadata(path)
    assert metadata["sample_count"] == 1
    assert metadata["backend"] == "sqlite-v1"


def test_empty_manifest_is_not_published(tmp_path: Path):
    path = tmp_path / "manifest.sqlite"
    with pytest.raises(ManifestError, match="empty"):
        with ManifestWriter(path, {"manifest_version": "v1"}):
            pass
    assert not path.exists()


def test_duplicate_sample_aborts_without_replacing_existing_manifest(tmp_path: Path):
    path = tmp_path / "manifest.sqlite"
    with ManifestWriter(path, {"manifest_version": "original"}) as writer:
        writer.add_sample(sample())
    original = path.read_bytes()
    with pytest.raises(Exception):
        with ManifestWriter(path, {"manifest_version": "broken"}) as writer:
            writer.add_sample(sample())
            writer.add_sample(sample())
    assert path.read_bytes() == original


def test_manifest_content_identity_hashes_records_and_is_reproducible(tmp_path: Path):
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    with ManifestWriter(first, {"manifest_version": "v1"}) as writer:
        writer.add_sample(sample())
    with ManifestWriter(second, {"manifest_version": "v1"}) as writer:
        writer.add_sample(sample())
    first_metadata = read_manifest_metadata(first)
    second_metadata = read_manifest_metadata(second)
    assert first_metadata["records_sha256"] == second_metadata["records_sha256"]
    assert first_metadata["content_identity"] == second_metadata["content_identity"]

    altered = sample()
    altered = FREDSample(
        altered.sample_id,
        altered.sequence_id,
        altered.frame_index,
        altered.timestamp,
        altered.rgb,
        altered.event,
        (Annotation("0.033333", (2, 2, 10, 12), 7, "drone", 1),),
        altered.official_split,
        altered.project_split,
        altered.provenance,
    )
    third = tmp_path / "third.sqlite"
    with ManifestWriter(third, {"manifest_version": "v1"}) as writer:
        writer.add_sample(altered)
    assert read_manifest_metadata(third)["records_sha256"] != first_metadata["records_sha256"]
