"""Offline fixtures for selection, protected membership, and Slurm budget rules."""
import json
import sqlite3

import pytest

from sota.FRED_LLM_GE.archive.phase0_data_old.annotation_policy import POLICY_VERSION
from sota.FRED_LLM_GE.archive.phase0_data_old.splits import load_project_split_manifest
from sota.FRED_LLM_GE.data.scene_diagnostic_export import evenly_spaced, scene_indices, validate_manifest
from sota.FRED_LLM_GE.phase0_data.check_scene_split import DEFAULT_PROJECT
from sota.FRED_LLM_GE.phase1_detection.submit_scene_overnight import (
    check_budget, reservation, walltime_seconds,
)


def job(gpus=1, time="02:00:00", typed=True):
    tres = f"cpu=8,gres/gpu={gpus}"
    if typed:
        tres += f",gres/gpu:h100={gpus}"
    return f"JobId=1 JobName=example ReqTRES={tres} TimeLimit={time}"


def test_gpu_model_tres_is_not_double_counted():
    assert reservation(job(2, "08:00:00")) == (2, 16 * 3600)
    assert reservation("ReqTRES=cpu=4,mem=48G TimeLimit=UNLIMITED") == (0, 0)
    assert reservation("ReqTRES=cpu=4,gres/gpu:h100=1 TimeLimit=02:00:00") == (1, 7200)
    assert walltime_seconds("1-02:30:00") == 95400


def test_gpu_volume_reserves_full_limits_of_running_and_pending_jobs():
    result = check_budget([("1", job(2, "15:00:00"))])
    assert result["total_reserved_gpu_hours"] == 32
    assert result["wait_for_gpu_jobs"] == ["1"]
    with pytest.raises(ValueError, match="32 GPU-hour"):
        check_budget([("1", job(2, "16:00:00"))])


def test_cpu_jobs_count_toward_queue_but_not_gpu_volume():
    jobs = [(str(i), "ReqTRES=cpu=4 TimeLimit=02:00:00") for i in range(48)]
    assert check_budget(jobs)["total_reserved_gpu_hours"] == 2
    with pytest.raises(ValueError, match="50-job"):
        check_budget(jobs + [("49", "ReqTRES=cpu=4 TimeLimit=02:00:00")])


@pytest.mark.parametrize("record", ["TimeLimit=02:00:00", "ReqTRES=gres/gpu=unknown TimeLimit=02:00:00",
                                   "ReqTRES=gres/gpu=1 TimeLimit=UNLIMITED",
                                   "ReqTRES=gres/gpu=0,gres/gpu:h100=1 TimeLimit=02:00:00"])
def test_unknown_resource_requests_fail_closed(record):
    with pytest.raises(ValueError):
        reservation(record)


def test_temporal_sampling_spans_eligible_frames_and_has_no_duplicates():
    frames = list(range(1, 3500))
    selected = evenly_spaced(frames, 512)
    assert selected[0] == 1 and selected[-1] == 3499
    assert len(set(selected)) == 512
    assert selected == evenly_spaced(frames, 512)
    with pytest.raises(ValueError, match="not enough"):
        evenly_spaced([1, 2], 512)


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.sqlite"
    project = load_project_split_manifest(DEFAULT_PROJECT)
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE metadata(key TEXT, value_json TEXT);"
                         "CREATE TABLE samples(sample_id TEXT, sequence_id TEXT, frame_index INT, project_split TEXT);"
                         "CREATE TABLE annotations(sample_id TEXT);")
        metadata = {"project_split_content_hash": project.content_hash,
                    "project_split_version": project.version, "annotation_policy": POLICY_VERSION,
                    "validation_status": "passed_annotation_policy_bringup_split_non_freeze"}
        db.executemany("INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v)) for k, v in metadata.items()])
        for sequence in ("3", "25", "168", "52", "180"):
            side = "train" if sequence in project.train else "validation"
            db.executemany("INSERT INTO samples VALUES (?,?,?,?)",
                           [(f"{sequence}:{i}", sequence, i, side) for i in range(514)])
            # Frame 0 has no labels and cannot enter the annotated-only selection.
            db.executemany("INSERT INTO annotations VALUES (?)", [(f"{sequence}:{i}",) for i in range(1, 514)])
    return path


def test_selection_uses_dataset_global_order_and_all_representatives(manifest):
    selected = validate_manifest(manifest)
    assert len(selected) == 5
    indices = scene_indices(manifest, "train", 512, POLICY_VERSION)
    assert set(indices) == {"3", "25", "168"}
    assert indices["3"][0] == 1 and indices["3"][-1] == 513
    assert indices["25"][0] == 515 and indices["25"][-1] == 1027
    assert sum(map(len, indices.values())) == 1536
    assert sum(map(len, scene_indices(manifest, "validation", 512, POLICY_VERSION).values())) == 1024


@pytest.mark.parametrize("mutation", [
    "UPDATE samples SET project_split='train' WHERE sequence_id='52'",
    "DELETE FROM samples WHERE sequence_id='168'",
    "INSERT INTO samples VALUES ('4:0','4',0,'train')",
    "INSERT INTO samples VALUES ('21:0','21',0,'held_out_test')",
    "UPDATE metadata SET value_json='\"stale\"' WHERE key='project_split_content_hash'",
])
def test_invalid_or_stale_scene_membership_is_rejected(manifest, mutation):
    with sqlite3.connect(manifest) as db:
        db.execute(mutation)
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_insufficient_annotations_and_wrong_policy_are_rejected(manifest):
    with sqlite3.connect(manifest) as db:
        db.execute("DELETE FROM annotations WHERE sample_id LIKE '3:%'")
    with pytest.raises(ValueError, match="not enough"):
        scene_indices(manifest, "train", 512, POLICY_VERSION)
    with pytest.raises(ValueError, match="policy"):
        scene_indices(manifest, "validation", 512, None)


def test_atomic_export_preserves_selection_and_clipped_label_lineage(manifest, monkeypatch):
    # Synthetic acceptance fixture: real PNGs/labels, no source downloads or GPU.
    from contextlib import nullcontext
    from types import SimpleNamespace
    from PIL import Image
    from sota.FRED_LLM_GE.archive.phase0_data_old import dataset, fred_api, inventory
    from sota.FRED_LLM_GE.data.scene_diagnostic_export import export, verify_export
    directory = manifest.parent
    revision = "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd"
    with sqlite3.connect(manifest) as db:
        for key, value in {"dataset_revision": revision,
                           "fred_repository_revision": "2bf89c5376eda528431b62d6c60f2c13d8f95ab4",
                           "project_split_approval_reference": load_project_split_manifest(DEFAULT_PROJECT).approval_reference,
                           "annotation_policy_approval_reference": "DG-P0-04; Bill approval in Codex on 2026-10-05"}.items():
            db.execute("INSERT INTO metadata VALUES (?,?)", (key, json.dumps(value)))
    records = {s: SimpleNamespace(sequence_id=s) for s in ("3", "25", "168", "52", "180")}
    monkeypatch.setattr(inventory, "load_inventory", lambda _: SimpleNamespace(
        dataset_revision=revision, inventory_hash="synthetic", by_sequence=lambda: records))
    source = SimpleNamespace(prepare_sequence=lambda *_: None,
                             activate_prepared_window=lambda *_: nullcontext())
    monkeypatch.setattr(fred_api, "HFFredSource", lambda _: source)

    class SyntheticDataset:
        def __init__(self, *, project_split, **kwargs):
            with sqlite3.connect(manifest) as db:
                self.rows = db.execute("SELECT sequence_id, frame_index FROM samples WHERE project_split=? "
                                       "ORDER BY CAST(sequence_id AS INTEGER), frame_index", (project_split.value,)).fetchall()

        def __getitem__(self, index):
            sequence, frame = self.rows[index]
            return SimpleNamespace(sequence_id=sequence, frame_index=frame, event=Image.new("RGB", (16, 16)),
                                   annotations=[SimpleNamespace(box_xyxy=(-1, 2, 8, 10))])

        def close(self):
            pass

    monkeypatch.setattr(dataset, "FREDDataset", SyntheticDataset)
    output = directory / "export"
    export(manifest, directory / "synthetic-inventory.json", output)
    assert verify_export(directory)["counts"] == {"train": 1536, "validation": 1024}
    lineage = [json.loads(line) for line in (output / "label_provenance.jsonl").read_text().splitlines()]
    assert len(lineage) == 2560 and len({r["sample_id"] for r in lineage}) == 2560
    assert all(r["targets"][0]["source_box_xyxy"][0] == -1 for r in lineage)
    assert all(r["targets"][0]["label_box_xyxy"][0] == 0 for r in lineage)
    with pytest.raises(FileExistsError):
        export(manifest, directory / "synthetic-inventory.json", output)
    (output / "images/train/3_00000001.png").unlink()
    with pytest.raises(ValueError, match="coverage"):
        verify_export(directory)


@pytest.mark.parametrize("gpu_submission_fails", [False, True])
def test_submission_dependencies_and_failed_gpu_submission_cleanup(tmp_path, monkeypatch, gpu_submission_fails):
    import subprocess
    from sota.FRED_LLM_GE.phase1_detection import submit_scene_overnight as submit
    monkeypatch.chdir(tmp_path)
    for path in ("data/.venv-yolo11/bin/python", "yolo11m.pt",
                 "sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/results/learning_rate_check_6135264_evaluation.json"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[:3] == ["git", "branch", "--show-current"]:
            return "fred-yolo11-infrastructure"
        if command[:2] == ["git", "rev-parse"]:
            return ".git/MERGE_HEAD" if "--git-path" in command else "a" * 40
        if command[0] == "squeue":
            return "123"
        if command[0] == "scontrol":
            return job()
        if command[:2] == ["sbatch", "--parsable"]:
            if any("--dependency=" in v for v in command):
                if gpu_submission_fails:
                    raise subprocess.CalledProcessError(1, command)
                return "2002"
            return "2001"
        return ""

    monkeypatch.setattr(submit, "run", fake_run)
    if gpu_submission_fails:
        with pytest.raises(subprocess.CalledProcessError):
            submit.main()
        assert ["scancel", "2001"] in commands
        assert not list((tmp_path / "data/fred_overnight").glob("*/jobs.json"))
    else:
        submit.main()
        gpu = next(c for c in commands if any("--dependency=" in v for v in c))
        assert "--dependency=afterok:2001,afterany:123" in gpu
        assert "--kill-on-invalid-dep=yes" in gpu
        assert len([c for c in commands if c[:2] == ["sbatch", "--test-only"]]) == 2
        receipt = json.loads(next((tmp_path / "data/fred_overnight").glob("*/jobs.json")).read_text())
        assert receipt["budget"]["total_reserved_gpu_hours"] == 4
