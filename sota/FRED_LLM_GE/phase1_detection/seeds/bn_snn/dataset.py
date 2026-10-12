"""Streaming event-data pipeline for the BN-SNN object-detection baseline.

The module deliberately separates three concerns:

1. Reading a sample from GEN1 or a HuggingFace streaming dataset.
2. Converting asynchronous events into a short sequence of neuromorphic frames.
3. Collating SSD-style targets for a PyTorch training loop.

The public sample representation is intentionally small so that the same model
can consume GEN1 now and FRED later without changing the network interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


@dataclass(frozen=True)
class EventSample:
    """One event window and its object annotations in pixel coordinates."""

    events: np.ndarray  # Structured array or [N, 4] array: x, y, timestamp, polarity.
    boxes_xyxy: np.ndarray  # [M, 4], pixel coordinates: x1, y1, x2, y2.
    labels: np.ndarray  # [M], integer class IDs; zero is reserved for background.
    width: int
    height: int
    sample_id: str = "unknown"


def _field(record: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    """Read the first available field name from a heterogeneous dataset row."""

    for name in names:
        if name in record:
            return record[name]
    return default


def _events_to_array(events: Any) -> np.ndarray:
    """Normalize common event encodings to columns ``x,y,t,p``."""

    if isinstance(events, Mapping):
        x = np.asarray(_field(events, ("x", "xs")))
        y = np.asarray(_field(events, ("y", "ys")))
        t = np.asarray(_field(events, ("t", "timestamp", "timestamps")))
        p = np.asarray(_field(events, ("p", "polarity", "pol")))
        return np.column_stack((x, y, t, p))

    array = np.asarray(events)
    if array.dtype.names:
        names = {name.lower(): name for name in array.dtype.names}
        return np.column_stack(
            (
                array[names["x"]],
                array[names["y"]],
                array[names.get("t", names.get("timestamp"))],
                array[names.get("p", names.get("polarity"))],
            )
        )
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError("events must be a mapping, structured array, or [N,4] array")
    return array[:, :4]


def _annotations_to_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert common GEN1/FRED annotation layouts to SSD target arrays."""

    boxes = _field(row, ("boxes", "bboxes", "bounding_boxes", "bbox"), [])
    labels = _field(row, ("labels", "class_ids", "classes", "category_id"), [])
    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    # A flat [x, y, width, height] box is common in exported event datasets.
    if boxes.size and boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)
    if boxes.size and boxes.shape[-1] == 4:
        boxes = boxes.reshape(-1, 4).copy()
        # The dataset adapter's contract is xyxy; convert xywh when requested.
        if str(row.get("box_format", "xyxy")).lower() in {"xywh", "coco"}:
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
    else:
        boxes = np.empty((0, 4), dtype=np.float32)

    count = min(len(boxes), len(labels))
    return boxes[:count], labels[:count]


def row_to_sample(row: Mapping[str, Any], *, default_width: int, default_height: int) -> EventSample:
    """Adapt one GEN1/FRED row without materializing the complete stream."""

    events = _events_to_array(_field(row, ("events", "event", "event_stream"), row))
    boxes, labels = _annotations_to_arrays(row)
    width = int(_field(row, ("width", "sensor_width"), default_width))
    height = int(_field(row, ("height", "sensor_height"), default_height))
    sample_id = str(_field(row, ("sample_id", "id", "sequence_id"), "unknown"))
    return EventSample(events, boxes, labels, width, height, sample_id)


def iter_gen1(path: str | Path, *, width: int = 304, height: int = 240) -> Iterator[EventSample]:
    """Stream GEN1 samples from ``.npz`` files or an HDF5 event container.

    No full-dataset cache is created. Each file is opened, converted, yielded,
    and closed, which keeps memory proportional to one event window.
    """

    root = Path(path)
    files = sorted(root.glob("*.npz")) if root.is_dir() else [root]
    for file_path in files:
        if file_path.suffix.lower() == ".npz":
            with np.load(file_path, allow_pickle=True) as data:
                row = {key: data[key] for key in data.files}
            yield row_to_sample(row, default_width=width, default_height=height)
            continue

        if file_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"unsupported GEN1 file: {file_path}")
        try:
            import h5py
        except ImportError as error:  # pragma: no cover - depends on optional runtime
            raise ImportError("GEN1 HDF5 streaming requires the optional 'h5py' package") from error

        # GEN1 exports differ slightly in group naming; these are the usual paths.
        with h5py.File(file_path, "r") as handle:
            event_node = handle.get("events")
            if event_node is None:
                event_node = handle.get("event")
            label_node = handle.get("labels")
            if label_node is None:
                label_node = handle.get("annotations")
            if event_node is None:
                raise KeyError(f"no 'events' dataset found in {file_path}")
            events = event_node[()]
            labels = label_node[()] if label_node is not None else []
            yield row_to_sample(
                {"events": events, "boxes": labels, "labels": np.zeros(len(labels), dtype=np.int64)},
                default_width=width,
                default_height=height,
            )


def iter_fred_hf(
    dataset_name: str,
    *,
    split: str = "train",
    width: int = 304,
    height: int = 240,
    **load_dataset_kwargs: Any,
) -> Iterator[EventSample]:
    """Stream FRED rows through HuggingFace ``load_dataset(..., streaming=True)``."""

    from datasets import load_dataset

    stream = load_dataset(dataset_name, split=split, streaming=True, **load_dataset_kwargs)
    for row in stream:
        yield row_to_sample(row, default_width=width, default_height=height)


def events_to_time_steps(
    sample: EventSample,
    *,
    time_steps: int = 10,
    representation: str = "count",
) -> torch.Tensor:
    """Voxelize asynchronous events as ``[T, 2, H, W]`` SCNN input.

    The two channels preserve ON/OFF polarity. ``count`` accumulates events in
    each temporal bin; ``binary`` records whether a pixel fired at least once.
    This temporal discretization lets the recurrent SNN process event history
    one step at a time while retaining polarity information.
    """

    if time_steps < 1:
        raise ValueError("time_steps must be positive")
    events = _events_to_array(sample.events)
    voxel = np.zeros((time_steps, 2, sample.height, sample.width), dtype=np.float32)
    if len(events) == 0:
        return torch.from_numpy(voxel)

    timestamps = events[:, 2].astype(np.float64)
    start, stop = float(timestamps.min()), float(timestamps.max())
    duration = max(stop - start, 1.0)
    bins = np.clip(((timestamps - start) / duration * time_steps).astype(np.int64), 0, time_steps - 1)
    xs = np.clip(events[:, 0].astype(np.int64), 0, sample.width - 1)
    ys = np.clip(events[:, 1].astype(np.int64), 0, sample.height - 1)
    polarities = (events[:, 3] > 0).astype(np.int64)
    for time_bin, channel, x, y in zip(bins, polarities, xs, ys):
        voxel[time_bin, channel, y, x] += 1.0
    if representation == "binary":
        voxel = (voxel > 0).astype(np.float32)
    elif representation != "count":
        raise ValueError("representation must be 'count' or 'binary'")
    return torch.from_numpy(voxel)


def ssd_target(sample: EventSample) -> dict[str, torch.Tensor | str]:
    """Return normalized boxes and labels in the format expected by an SSD head."""

    boxes = torch.as_tensor(sample.boxes_xyxy, dtype=torch.float32).reshape(-1, 4).clone()
    if len(boxes):
        boxes[:, [0, 2]] /= sample.width
        boxes[:, [1, 3]] /= sample.height
        boxes.clamp_(0.0, 1.0)
    return {
        "boxes": boxes,
        "labels": torch.as_tensor(sample.labels, dtype=torch.int64),
        "image_id": sample.sample_id,
    }


class EventStreamDataset(IterableDataset):
    """Iterable dataset that performs event voxelization just before batching."""

    def __init__(self, source: Iterable[EventSample], *, time_steps: int = 10, representation: str = "count"):
        super().__init__()
        self.source = source
        self.time_steps = time_steps
        self.representation = representation

    def __iter__(self) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor | str]]]:
        for sample in self.source:
            yield events_to_time_steps(sample, time_steps=self.time_steps, representation=self.representation), ssd_target(sample)


def detection_collate(
    batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor | str]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor | str]]]:
    """Stack fixed-shape event tensors while retaining variable-length SSD boxes."""

    event_batches, targets = zip(*batch)
    return torch.stack(event_batches, dim=0), list(targets)  # [B,T,2,H,W], not [T,B,...] yet.


def make_event_loader(
    source: Iterable[EventSample],
    *,
    batch_size: int = 8,
    time_steps: int = 10,
    representation: str = "count",
    num_workers: int = 0,
) -> DataLoader:
    """Build a streaming DataLoader; the training loop can transpose B/T as needed."""

    dataset = EventStreamDataset(source, time_steps=time_steps, representation=representation)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=detection_collate)
