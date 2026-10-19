"""Shared provisional Event-pilot data loading; cache layout retained for YOLO11."""
import os
import zipfile
from pathlib import Path
import numpy as np
import torch
from sota.FRED_LLM_GE.data.fred_stream import REVISION, STEP_US, stream_event_frames
from sota.FRED_LLM_GE.phase0_data.coco_adapter import parse_fred_annotation
SCRIPT_DIR = Path(__file__).resolve().parent / "seeds" / "yolo11"
TRAIN_SEQUENCES = "5,18,30,42,57,66,96,104"
VAL_SEQUENCES = "1,50,72"
FRAME_STEP = 10

def load_annotations(split, sequence_id, width, height):
    """Read <seq>/coordinates.txt; return {frame_index: [[x1, y1, x2, y2], ...]}."""
    from huggingface_hub import HfFileSystem

    archive = f"datasets/GabrieleMagrini/FRED@{REVISION}/{split}/{sequence_id}.zip"
    with HfFileSystem().open(archive, "rb") as handle:
        text = zipfile.ZipFile(handle).read(f"{sequence_id}/coordinates.txt").decode()
    boxes = {}
    for line in text.splitlines():
        annotation = parse_fred_annotation(line)
        if annotation is None:
            continue
        # Boxes may extend past the image edge; keep only the visible part.
        x1, y1 = max(0.0, annotation["x1"]), max(0.0, annotation["y1"])
        x2, y2 = min(float(width), annotation["x2"]), min(float(height), annotation["y2"])
        if x2 > x1 and y2 > y1:
            frame_index = round(annotation["time"] * 1_000_000 / STEP_US)
            boxes.setdefault(frame_index, []).append([x1, y1, x2, y2])
    return boxes


def load_sequence(split, sequence_id, frame_step, img_width):
    """Return cached event frames, resized to img_width, with their boxes.

    images: uint8 [N, 3, H, W]; boxes: list of [K, 4] tensors in resized
    pixels; orig_boxes: the same boxes in original 1280x720 pixels.
    """
    cache_dir = SCRIPT_DIR / "data_cache"
    cache_path = cache_dir / f"{split}_{sequence_id}_step{frame_step}_w{img_width}.pt"
    if cache_path.is_file():
        return torch.load(cache_path)

    images, sample_ids, frame_indices = [], [], []
    report = cache_dir / f"bad_timestamps_{split}_{sequence_id}.csv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for image, timestamp in stream_event_frames(split, sequence_id, report, frame_step):
        width, height = image.size
        img_height = round(height * img_width / width / 4) * 4
        image = image.convert("RGB").resize((img_width, img_height))
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
        frame_indices.append(round(timestamp * 1_000_000 / STEP_US))
        sample_ids.append(f"{split}/{sequence_id}:{frame_indices[-1]}")
    if not images:
        raise ValueError(f"no event frames streamed for {split}/{sequence_id}")

    annotations = load_annotations(split, sequence_id, width, height)
    scale = torch.tensor([img_width / width, img_height / height] * 2)
    orig_boxes = [torch.tensor(annotations.get(i, []), dtype=torch.float32).reshape(-1, 4)
                  for i in frame_indices]
    data = {
        "images": torch.stack(images),
        "boxes": [b * scale for b in orig_boxes],
        "orig_boxes": orig_boxes,
        "sample_ids": sample_ids,
        "orig_size": (width, height),
    }
    # Per-process temp name: parallel candidates may build the same cache.
    temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(data, temp_path)
    os.replace(temp_path, cache_path)
    return data


def load_split(split, sequences, frame_step, img_width):
    return load_archives([(split,s) for s in sequences],frame_step,img_width)

def load_archives(archives, frame_step, img_width):
    """Load sequence IDs from explicit physical HF train/test archive locations."""
    parts = [load_sequence(split,s,frame_step,img_width) for split,s in archives]
    return {
        "images": torch.cat([p["images"] for p in parts]),
        "boxes": [b for p in parts for b in p["boxes"]],
        "orig_boxes": [b for p in parts for b in p["orig_boxes"]],
        "sample_ids": [i for p in parts for i in p["sample_ids"]],
        "orig_size": parts[0]["orig_size"],
    }


