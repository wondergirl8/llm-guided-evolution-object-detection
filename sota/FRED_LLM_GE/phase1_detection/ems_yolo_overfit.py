"""Overfit the stock EMS-YOLO seed on a tiny labeled FRED training subset.

This is a pipeline diagnostic, not a baseline or an evolutionary fitness run.
It deliberately evaluates the training images it optimized and preserves the
protocol's confidence threshold and shared COCO AP implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import torch
import yaml
from PIL import Image

from .adapters.ems_yolo import EMSYoloAdapter
from .candidate_runner import code_revision, predict_frames
from .fitness import evaluate_frames
from .proxy_data import LazyImages


FRED = Path(__file__).resolve().parents[1]
PROTOCOL = FRED / "configs/phase1/protocol.yaml"
SEED = FRED / "seeds/seed_ems_yolo.py"
DEFAULT_OUTPUT = FRED.parents[1] / "analysis/diagnostics/ems_yolo_overfit"


def _boxes(label_path, width, height):
    values = []
    for line in label_path.read_text().splitlines():
        fields = list(map(float, line.split()))
        if len(fields) != 5 or fields[0] != 0:
            raise ValueError(f"invalid single-class YOLO label: {label_path}: {line!r}")
        _, center_x, center_y, box_width, box_height = fields
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        x2 = (center_x + box_width / 2) * width
        y2 = (center_y + box_height / 2) * height
        values.append([max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)])
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


def load_labeled_subset(root, count, seed):
    """Choose a deterministic shuffled subset from all labeled export frames."""
    root = Path(root)
    candidates = []
    for image_text in (root / "train.txt").read_text().splitlines():
        image_path = Path(image_text)
        label_path = root / "labels" / f"{image_path.stem}.txt"
        if label_path.is_file() and label_path.stat().st_size:
            candidates.append((image_path, label_path))
    if len(candidates) < count:
        raise ValueError(f"requested {count} labeled frames, found {len(candidates)}")
    random.Random(seed).shuffle(candidates)
    selected = candidates[:count]
    with Image.open(selected[0][0]) as image:
        size = image.size
    boxes = []
    for image_path, label_path in selected:
        with Image.open(image_path) as image:
            if image.size != size:
                raise ValueError(f"inconsistent frame size: {image_path}")
        boxes.append(_boxes(label_path, *size))
    paths = [item[0] for item in selected]
    return {
        "images": LazyImages(paths, size),
        "boxes": boxes,
        "orig_boxes": boxes,
        "sample_ids": [f"tiny-overfit/{path.stem}" for path in paths],
        "orig_size": size,
    }


def atomic_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args):
    if args.output.exists():
        raise FileExistsError(
            f"diagnostic output already exists; preserve it and choose a new --output: {args.output}"
        )
    args.output.mkdir(parents=True)
    protocol = yaml.safe_load(args.protocol.read_text())
    settings = copy.deepcopy(protocol["families"]["ems_yolo"])
    proxy = copy.deepcopy(protocol["proxy"])
    if proxy["eval_conf"] != 0.05:
        raise ValueError("the EMS tiny-overfit diagnostic is frozen to eval_conf=0.05")
    settings["training"]["epochs"] = args.epochs
    proxy.update(epochs=args.epochs, seed=args.seed,
                 batch_size=min(proxy["batch_size"], args.subset_size))
    challenging = protocol["challenging"]
    data_root = FRED / challenging["data_root"] / "search_train_step50"
    data = load_labeled_subset(data_root, args.subset_size, args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    adapter = EMSYoloAdapter(settings, FRED)
    model = adapter.build(SEED, num_classes=1, device=device)
    params = adapter.param_count(model)
    print(f"EMS tiny overfit: frames={args.subset_size}, epochs={args.epochs}, "
          f"eval_conf={proxy['eval_conf']}, params={params}, device={device}", flush=True)
    adapter.train(model, data, proxy, device)
    frames = predict_frames(adapter, model, data, proxy, device)
    result = evaluate_frames(frames, params=params,
                             expected_sample_ids=data["sample_ids"])
    predictions = sum(len(frame["prediction"]["boxes"]) for frame in frames)
    frames_with_predictions = sum(bool(frame["prediction"]["boxes"]) for frame in frames)

    model_path = args.output / "seed.pt"
    adapter.save(model, model_path)
    summary = {
        "schema_version": "ems-yolo-tiny-overfit-v1",
        "diagnostic_only": True,
        "code_revision": code_revision(),
        "seed": args.seed,
        "subset_size": args.subset_size,
        "epochs": args.epochs,
        "eval_conf": proxy["eval_conf"],
        "mAP50": result.map50,
        "mAP50_95": result.map50_95,
        "params": result.params,
        "predictions": predictions,
        "frames_with_predictions": frames_with_predictions,
        "sample_ids": data["sample_ids"],
        "model": str(model_path),
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if result.map50 == 0:
        print("DIAGNOSTIC RESULT: FAIL -- seed could not overfit the tiny subset", flush=True)
    elif result.map50 < 0.5:
        print("DIAGNOSTIC RESULT: WEAK -- nonzero localization, but poor overfit", flush=True)
    else:
        print("DIAGNOSTIC RESULT: PASS -- seed can overfit the tiny subset", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.subset_size < 1 or args.epochs < 1 or args.seed < 0:
        parser.error("subset size and epochs must be positive; seed must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
