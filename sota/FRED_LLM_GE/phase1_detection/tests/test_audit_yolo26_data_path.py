"""Offline checks for search-export label and image-path auditing."""

import hashlib
import json
import shutil

import pytest
import yaml
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection.audit_yolo26_data_path import (
    audit_search, audit_split,
)
from sota.FRED_LLM_GE.phase1_detection.export_split import yolo_label


def _split(root, step, boxes):
    for subdir in ("done", "images", "labels"):
        (root / subdir).mkdir(parents=True)
    records = []
    for index, frame_boxes in enumerate(boxes, 1):
        name = f"119_{index:06d}"
        records.append({"name": name, "frame_index": index, "boxes": frame_boxes})
        Image.new("RGB", (128, 64), (index, 20, 30)).save(root / "images" / f"{name}.png")
        (root / "labels" / f"{name}.txt").write_text(yolo_label(frame_boxes, 128, 64))
    (root / "done" / "119.json").write_text(json.dumps({
        "sequence": "119", "hf_folder": "train", "frame_step": step,
        "frames": len(records), "size": [128, 64], "records": records,
    }))
    (root / "train.txt").write_text("\n".join(
        str(root / "images" / f"{record['name']}.png") for record in records) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(root), "train": "train.txt", "val": "trainer_val/images",
        "names": {0: "drone"}}, sort_keys=False))
    return records


def _training_export(root):
    exact = [10.0, 10.0, 20.0, 20.0]
    shifted = [10.00001, 10.0, 20.00001, 20.0]
    assert yolo_label([exact], 128, 64) == yolo_label([shifted], 128, 64)
    records = _split(root, 50, [[exact, exact], [exact, shifted], []])
    for subdir in ("images", "labels"):
        (root / "trainer_val" / subdir).mkdir(parents=True)
        for record in records[::2]:
            name = record["name"]
            suffix = ".png" if subdir == "images" else ".txt"
            shutil.copy2(root / subdir / f"{name}{suffix}",
                         root / "trainer_val" / subdir / f"{name}{suffix}")


def test_audit_classifies_duplicates_and_preserves_export(tmp_path):
    train = tmp_path / "search_train_step50"
    val = tmp_path / "search_val_step10"
    _training_export(train)
    _split(val, 10, [[[1.0, 1.0, 5.0, 5.0]]])
    original = hashlib.sha256((train / "labels/119_000001.txt").read_bytes()).hexdigest()
    protocol = {"challenging": {"search": {"train_frame_step": 50, "val_frame_step": 10}}}
    result = audit_search(tmp_path, protocol, sample_images=2)
    audit = result["train"]
    assert audit["frames"] == 3
    assert audit["background_frames"] == 1
    assert audit["duplicate_label_rows"] == 2
    assert audit["marker_duplicate_rows"] == 1
    assert audit["rounding_collision_rows"] == 1
    assert {row["kind"] for row in audit["duplicate_examples"]} == {
        "marker_duplicate", "rounding_collision"}
    assert audit["trainer_val_frames"] == 2
    assert audit["image_probe"]["sample_count"] == 2
    assert audit["issue_count"] == 0
    assert result["search_val"]["issue_count"] == 0
    assert original == hashlib.sha256((train / "labels/119_000001.txt").read_bytes()).hexdigest()


def test_audit_detects_label_drift_and_changed_trainer_copy(tmp_path):
    train = tmp_path / "search_train_step50"
    _training_export(train)
    (train / "labels/119_000002.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (train / "trainer_val/images/119_000001.png").write_bytes(b"changed")
    result = audit_split(train, sample_images=0, trainer_val=True)
    assert result["issue_count"] == 2
    assert any("label differs from marker" in issue for issue in result["issues"])
    assert any("trainer_val differs" in issue for issue in result["issues"])
    assert result["image_probe"]["sample_count"] == 0


def test_audit_detects_stale_training_paths(tmp_path):
    train = tmp_path / "search_train_step50"
    _training_export(train)
    (train / "train.txt").write_text("/old/mount/119_000001.png\n")
    result = audit_split(train, sample_images=0, trainer_val=True, expected_step=50)
    assert any("stale paths" in issue for issue in result["issues"])


def test_audit_requires_original_export(tmp_path):
    with pytest.raises(FileNotFoundError, match="no export markers"):
        audit_split(tmp_path / "search_train_step50")
