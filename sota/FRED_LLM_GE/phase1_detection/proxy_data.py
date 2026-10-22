"""Load the proxy-protocol FRED event frames shared by every model family.

Frames are streamed from Hugging Face with data/fred_stream.py, labelled from
each sequence's coordinates.txt, resized to the protocol width and cached as
uint8 tensors so every candidate in a run reuses the same data. The cache file
names match seeds/yolo11/train_eval.py, so both read the same cache.
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

FRED_DIR = Path(__file__).resolve().parents[1]
for path in (FRED_DIR / "data", FRED_DIR / "phase0_data"):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from fred_stream import REVISION, STEP_US, stream_event_frames  # noqa: E402
from coco_adapter import parse_fred_annotation  # noqa: E402


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


def load_sequence(cache_dir, split, sequence_id, frame_step, img_width):
    """Return cached event frames, resized to img_width, with their boxes.

    images: uint8 [N, 3, H, W]; boxes: list of [K, 4] tensors in resized
    pixels; orig_boxes: the same boxes in original 1280x720 pixels.
    """
    cache_dir = Path(cache_dir)
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


def load_split(cache_dir, split, sequences, frame_step, img_width):
    parts = [load_sequence(cache_dir, split, s, frame_step, img_width) for s in sequences]
    return {
        "images": torch.cat([p["images"] for p in parts]),
        "boxes": [b for p in parts for b in p["boxes"]],
        "orig_boxes": [b for p in parts for b in p["orig_boxes"]],
        "sample_ids": [i for p in parts for i in p["sample_ids"]],
        "orig_size": parts[0]["orig_size"],
    }


class LazyImages:
    """uint8 [N, 3, H, W] frames read from PNG files when requested."""

    def __init__(self, paths, size):
        self.paths = list(paths)
        width, height = size
        self.shape = (len(self.paths), 3, height, width)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        if isinstance(index, slice):
            paths = self.paths[index]
        elif isinstance(index, torch.Tensor) and index.ndim == 1 and index.dtype in (
                torch.int32, torch.int64):
            paths = [self.paths[int(i)] for i in index]
        else:
            raise TypeError("LazyImages requires a slice or 1D integer tensor index")
        frames = []
        for path in paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            frames.append(torch.from_numpy(pixels).permute(2, 0, 1))
        return torch.stack(frames)


def load_exported_split(root, sequences):
    """Frames written by export_split.py for these sequences, in the load_split format.

    Images stay on disk (LazyImages); boxes are the exact pixel boxes recorded
    at export. Frames are kept at their native size, so boxes and orig_boxes match.
    """
    import json

    root = Path(root)
    paths, boxes, sample_ids, size = [], [], [], None
    for sequence, folder in sequences:
        done = root / "done" / f"{sequence}.json"
        if not done.is_file():
            raise FileNotFoundError(f"{done} missing: run export_split.py --set search")
        marker = json.loads(done.read_text())
        size = size or tuple(marker["size"])
        if tuple(marker["size"]) != size:
            raise ValueError(f"sequence {sequence} has frame size {marker['size']}, not {size}")
        for record in marker["records"]:
            paths.append(root / "images" / f"{record['name']}.png")
            boxes.append(torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4))
            sample_ids.append(f"{folder}/{sequence}:{record['frame_index']}")
    if not paths:
        raise ValueError(f"no exported frames under {root}")
    return {"images": LazyImages(paths, size), "boxes": boxes, "orig_boxes": boxes,
            "sample_ids": sample_ids, "orig_size": size}
