"""Small offline Event-export audit fixtures; no FRED download or GPU needed."""

import hashlib
import json
import zipfile
from io import BytesIO

import torch
from PIL import Image

from sota.FRED_LLM_GE.phase1_detection.audit_dfine_event_export import (
    audit_export, render_overlays, select_examples,
)
from sota.FRED_LLM_GE.phase1_detection.candidate_runner import (
    predict_frames, target_boxes_sha256,
)
from sota.FRED_LLM_GE.phase1_detection.export_split import yolo_label
from sota.FRED_LLM_GE.phase1_detection.proxy_data import LazyImages


def _fixture(tmp_path):
    root = tmp_path / "search_val_step10"
    for name in ("done", "images", "labels"):
        (root / name).mkdir(parents=True)
    records = [
        {"name": "4_000001", "frame_index": 1,
         "boxes": [[0.0, 2.0, 8.0, 8.0]]},
        {"name": "4_000011", "frame_index": 11, "boxes": []},
        {"name": "4_000021", "frame_index": 21,
         "boxes": [[10.0, 10.0, 20.0, 20.0]]},
    ]
    archive_path = tmp_path / "source" / "train" / "4.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("4/coordinates.txt",
                         "0.033333: -2.0, 2.0, 8.0, 8.0, 1, drone\n"
                         "0.699993: 10.0, 10.0, 20.0, 20.0, 1, drone\n")
        by_index = {record["frame_index"]: record for record in records}
        for frame_index in range(1, 22):
            image = Image.new("RGB", (32, 32), (frame_index, 20, 30))
            if frame_index in by_index:
                record = by_index[frame_index]
                image.save(root / "images" / f"{record['name']}.png")
                (root / "labels" / f"{record['name']}.txt").write_text(
                    yolo_label(record["boxes"], 32, 32))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            counter = frame_index * 33_333
            archive.writestr(f"4/Event/Frames/Video_4_frame_{counter}.png",
                             buffer.getvalue())
    (root / "done" / "4.json").write_text(json.dumps({
        "sequence": "4", "hf_folder": "train", "frame_step": 10,
        "frames": 3, "size": [32, 32], "records": records,
    }))
    ids = [f"train/4:{record['frame_index']}" for record in records]
    paths = [root / "images" / f"{record['name']}.png" for record in records]
    boxes = [torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
             for record in records]
    data = {"images": LazyImages(paths, (32, 32)), "orig_boxes": boxes,
            "sample_ids": ids, "orig_size": (32, 32)}
    report = {"num_frames": 3, "metadata": {
        "split_identity": "train=5;val=4",
        "validation_sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "validation_targets_sha256": target_boxes_sha256(data),
    }}
    return root, data, report, archive_path


def test_audit_checks_source_identity_clipping_and_label_geometry(tmp_path):
    root, data, report, archive = _fixture(tmp_path)
    result = audit_export(root, data, report, archive.parents[1], expected_step=10)
    assert result["issues"] == []
    assert result["source_sequences_checked"] == 1
    assert result["source_pixel_samples_checked"] == 1
    assert result["source_annotations"]["clipped"] == 1
    assert result["empty_frames"] == 1
    assert result["small_object_frames"] == 2

    (root / "labels" / "4_000001.txt").write_text("0 0.5 0.5 0.1 0.1")
    bad = audit_export(root, data, report, archive.parents[1], expected_step=10)
    assert any("YOLO labels differ" in issue for issue in bad["issues"])

    Image.new("RGB", (32, 32), (255, 0, 0)).save(root / "images/4_000001.png")
    bad_pixels = audit_export(root, data, report, archive.parents[1], expected_step=10)
    assert any("Event PNG pixels differ" in issue for issue in bad_pixels["issues"])


def test_audit_selects_cases_and_draws_original_pixel_boxes(tmp_path):
    root, data, report, _ = _fixture(tmp_path)
    audit = audit_export(root, data, report)
    diagnostics = {"by_sequence": {"train/4": {"false_negatives_iou50": 2}},
                   "examples": {"false_negatives": [{"sample_id": "train/4:21"}],
                                "poorly_localized": []}}
    selected = select_examples(audit, diagnostics, limit=2)
    assert any(category == "small" for category, _ in selected)
    assert any(category == "high_miss_sequence" for category, _ in selected)
    assert ("empty", "train/4:11") in selected
    frames = [{"sample_id": sample_id,
               "target": {"boxes": boxes.tolist()},
               "prediction": {"boxes": [[4, 4, 9, 9]]}}
              for sample_id, boxes in zip(data["sample_ids"], data["orig_boxes"])]
    output = tmp_path / "overlays"
    output.mkdir()
    manifest = render_overlays(selected, frames, data, output)
    assert len(manifest) == len(selected)
    assert all((output / item["image"]).is_file() for item in manifest)


def test_runner_maps_resized_predictions_back_to_original_pixels():
    class Adapter:
        def predict(self, model, images, proxy, device):
            return [(torch.tensor([[5., 5., 15., 15.]]), torch.tensor([0.8]))]

    data = {"images": torch.zeros((1, 3, 32, 32), dtype=torch.uint8),
            "orig_size": (64, 64),
            "orig_boxes": [torch.tensor([[10., 10., 30., 30.]])],
            "sample_ids": ["train/4:1"]}
    frames = predict_frames(Adapter(), None, data, {"batch_size": 1}, "cpu")
    assert frames[0]["prediction"]["boxes"] == [[10., 10., 30., 30.]]
    assert frames[0]["target"]["boxes"] == [[10., 10., 30., 30.]]
