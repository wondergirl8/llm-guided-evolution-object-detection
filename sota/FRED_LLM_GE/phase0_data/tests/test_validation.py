import json
from pathlib import Path
from types import SimpleNamespace

from sota.FRED_LLM_GE.phase0_data.config import load_config
from sota.FRED_LLM_GE.phase0_data.fred_api import ArchiveMemberMetadata
from sota.FRED_LLM_GE.phase0_data.inventory import SequenceInventoryRecord
from sota.FRED_LLM_GE.phase0_data.schema import RemoteObject
from sota.FRED_LLM_GE.phase0_data.splits import (
    OfficialSplitManifest,
    ProjectSplitManifest,
)
from sota.FRED_LLM_GE.phase0_data.validation import (
    SequenceInspection,
    inspect_sequence,
    write_validation_report,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "phase0" / "default.yaml"


def prepared_sequence(sequence_root: Path, *, has_raw_hdf5: bool = True):
    raw_hdf5 = (
        ArchiveMemberMetadata("0/Event/events.hdf5", 32, 24, 1234)
        if has_raw_hdf5
        else None
    )
    return SimpleNamespace(
        sequence_root=sequence_root,
        raw_hdf5=raw_hdf5,
    )


def test_valid_sequence_constructs_model_neutral_samples(sequence_root: Path, tmp_path: Path):
    config = load_config(CONFIG, workspace_root=tmp_path)
    record = SequenceInventoryRecord(
        "0", "train", RemoteObject("hf://archive", "train/0.zip")
    )
    official = OfficialSplitManifest("x", ("0",), ("1",), "a", "b", "c", "now")
    project = ProjectSplitManifest("v1", "sequence", ("0",), (), ("1",), "DG-P0-02")
    result = inspect_sequence(
        config=config,
        inventory_record=record,
        prepared_sequence=prepared_sequence(sequence_root),
        official_split=official,
        project_split=project,
    )
    assert result.is_valid
    assert [sample.sample_id for sample in result.samples] == ["0:0", "0:1"]
    assert result.samples[0].annotations[0].track_id == 7


def test_out_of_bounds_box_is_blocking(sequence_root: Path, tmp_path: Path):
    (sequence_root / "coordinates.txt").write_text(
        "0.033333: 1, 2, 100, 12, 7, drone\n", encoding="utf-8"
    )
    config = load_config(CONFIG, workspace_root=tmp_path)
    result = inspect_sequence(
        config=config,
        inventory_record=SequenceInventoryRecord(
            "0", "train", RemoteObject("hf://archive", "train/0.zip")
        ),
        prepared_sequence=prepared_sequence(sequence_root),
        official_split=OfficialSplitManifest("x", ("0",), ("1",), "a", "b", "c", "now"),
        project_split=ProjectSplitManifest(
            "v1", "sequence", ("0",), (), ("1",), "DG-P0-02"
        ),
    )
    assert not result.is_valid
    assert any(item.code == "annotation.out_of_bounds" for item in result.findings)


def test_annotation_timestamp_within_configured_tolerance_is_associated(
    sequence_root: Path, tmp_path: Path
):
    (sequence_root / "coordinates.txt").write_text(
        "0.0333334: 1, 2, 10, 12, 7, drone\n", encoding="utf-8"
    )
    config = load_config(CONFIG, workspace_root=tmp_path)
    result = inspect_sequence(
        config=config,
        inventory_record=SequenceInventoryRecord(
            "0", "train", RemoteObject("hf://archive", "train/0.zip")
        ),
        prepared_sequence=prepared_sequence(sequence_root),
        official_split=OfficialSplitManifest("x", ("0",), ("1",), "a", "b", "c", "now"),
        project_split=ProjectSplitManifest(
            "v1", "sequence", ("0",), (), ("1",), "DG-P0-02"
        ),
    )
    assert result.is_valid
    assert result.samples[0].annotations[0].timestamp == "0.0333334"


def test_repeated_inspection_is_deterministic_in_samples_findings_and_provenance(
    sequence_root: Path, tmp_path: Path
):
    (sequence_root / "coordinates.txt").write_text(
        "0.033333: 1, 2, 100, 12, 7, drone\n", encoding="utf-8"
    )
    config = load_config(CONFIG, workspace_root=tmp_path)
    record = SequenceInventoryRecord(
        "0", "train", RemoteObject("hf://archive", "train/0.zip")
    )
    official = OfficialSplitManifest("x", ("0",), ("1",), "a", "b", "c", "now")
    project = ProjectSplitManifest("v1", "sequence", ("0",), (), ("1",), "DG-P0-02")
    arguments = {
        "config": config,
        "inventory_record": record,
        "prepared_sequence": prepared_sequence(sequence_root),
        "official_split": official,
        "project_split": project,
    }

    first = inspect_sequence(**arguments)
    second = inspect_sequence(**arguments)

    assert first == second
    assert [finding.code for finding in first.findings] == ["annotation.out_of_bounds"]
    assert first.samples[0].provenance.fred_dataset_identity == config.source.dataset_id
    assert first.samples[0].provenance.fred_dataset_revision == config.source.revision
    assert (
        first.samples[0].provenance.fred_repository_revision
        == config.fred_repository.revision
    )
    assert first.samples[0].provenance.phase0_schema_version == config.schema_version
    assert first.samples[0].provenance.manifest_version == config.manifest_version


def test_missing_raw_hdf5_is_blocking(sequence_root: Path, tmp_path: Path):
    config = load_config(CONFIG, workspace_root=tmp_path)
    result = inspect_sequence(
        config=config,
        inventory_record=SequenceInventoryRecord(
            "0", "train", RemoteObject("hf://archive", "train/0.zip")
        ),
        prepared_sequence=prepared_sequence(sequence_root, has_raw_hdf5=False),
        official_split=OfficialSplitManifest("x", ("0",), ("1",), "a", "b", "c", "now"),
        project_split=ProjectSplitManifest(
            "v1", "sequence", ("0",), (), ("1",), "DG-P0-02"
        ),
    )
    assert not result.is_valid
    assert any(item.code == "raw_hdf5.missing_from_archive" for item in result.findings)


def test_empty_or_interrupted_validation_report_is_never_valid(tmp_path: Path):
    report_path = tmp_path / "report.json"
    write_validation_report(
        [],
        report_path,
        context={"manifest_version": "test"},
        expected_sequence_ids=("0",),
        execution_error={"type": "FredRemoteError", "message": "download failed"},
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["status"] == "failed"
    assert report["execution"]["status"] == "failed"
    assert report["scope"]["complete"] is False
    assert report["scope"]["missing_sequence_ids"] == ["0"]


def test_validation_report_detects_incomplete_expected_scope(tmp_path: Path):
    report_path = tmp_path / "report.json"
    inspection = SequenceInspection("0", (), (), 1, 1, 0, 10)
    write_validation_report(
        [inspection],
        report_path,
        context={},
        expected_sequence_ids=("0", "1"),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["scope"]["missing_sequence_ids"] == ["1"]
