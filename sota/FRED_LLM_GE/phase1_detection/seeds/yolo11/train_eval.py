"""
Train and evaluate one LLM-generated variant of the FRED YOLO11 seed.

Called by run_improved.py through EVAL_RUNLINE:
    uv run python train_eval.py --model network_<gene_id> --variant_dir <models dir>

The variant file is a copy of sota/FRED_LLM_GE/seeds/seed_yolo11.py with
mutated classes; it must define DroneDetector(num_classes) whose
forward(event) returns raw [B, 5 + num_classes, H/s, W/s] logits and whose
predict(event) returns pixel-space boxes. "--model network" trains the seed.

Data: FRED event frames are streamed from Hugging Face with
data/fred_stream.py and labelled from each sequence's coordinates.txt.
Decoded, resized frames are cached under data_cache/ so every candidate in a
run reuses the same tensors.

Training uses a short proxy budget (few epochs on every Nth frame). Scores are
only for ranking candidates against each other during evolution; elites are
retrained at full budget later. Every candidate is validated on the same
frames, scored by the shared map_reporter.py, which writes
results/<gene_id>_results.csv for run_improved.py.
"""

import os
import sys
import json
import time
import zipfile
import hashlib
import argparse
import subprocess
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
FRED_DIR = SCRIPT_DIR.parents[2]
SEED_FILE = FRED_DIR / "seeds" / "seed_yolo11.py"
for path in (SCRIPT_DIR.parents[1], FRED_DIR / "data", FRED_DIR / "phase0_data"):
    sys.path.append(str(path))
import map_reporter
from result_contract import DetectionResult, write_results
from fred_stream import REVISION, STEP_US, stream_event_frames
from coco_adapter import parse_fred_annotation

# Provisional proxy budget; must match the values the seed baseline uses
# (to be agreed with Member 3).
PROXY_EPOCHS = 3
FRAME_STEP = 10

# Sequences from FRED's Hugging Face train/ folder. Validation sequences are
# held out of training so every candidate is scored on the same unseen frames.
TRAIN_SEQUENCES = "36"
VAL_SEQUENCES = "1"

# Detections below this confidence are dropped before NMS and scoring.
EVAL_CONF = 0.05

# map_reporter only accepts validation reports declared as coming from FRED's
# challenging-train split. Confirm the HF train/ folder matches it.
BENCHMARK_SPLIT = "challenging"
OFFICIAL_SPLIT = "challenging_train"

# Worst-case values for FITNESS_WEIGHTS = (1.0, 1.0, -1.0).
FAILED_RESULT = DetectionResult(0.0, 0.0, 999999999)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_failure_results(gene_id, start_time, message):
    """Record worst-case fitness for generated models that cannot be evaluated."""
    train_time = time.time() - start_time
    write_results(SCRIPT_DIR, gene_id, FAILED_RESULT)
    print(f"ERROR evaluating generated model: {message}")
    print(f"Wrote worst-case fitness after {train_time:.1f}s")
    print('='*120);print('job done');print('='*120)


def load_variant(model_name, variant_dir):
    """Import the seed or an LLM-generated network_<id>.py by file path."""
    path = SEED_FILE if model_name == "network" else Path(variant_dir) / f"{model_name}.py"
    spec = importlib.util.spec_from_file_location(f"fred_yolo11_{model_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


# ---------------------------------------------------------------- data

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
    temp_path = cache_path.with_suffix(".tmp")
    torch.save(data, temp_path)
    os.replace(temp_path, cache_path)
    return data


def load_split(split, sequences, frame_step, img_width):
    parts = [load_sequence(split, s, frame_step, img_width) for s in sequences]
    return {
        "images": torch.cat([p["images"] for p in parts]),
        "boxes": [b for p in parts for b in p["boxes"]],
        "orig_boxes": [b for p in parts for b in p["orig_boxes"]],
        "sample_ids": [i for p in parts for i in p["sample_ids"]],
        "orig_size": parts[0]["orig_size"],
    }


# ---------------------------------------------------------------- loss

def yolo_loss(logits, boxes, stride):
    """Single-scale YOLO loss matching DroneDetector.predict's decoding.

    Each ground-truth box is assigned to the grid cell containing its centre.
    That cell learns (sigmoid(tx), sigmoid(ty)) = centre offset in the cell
    and (tw, th) = log(size / stride); every cell learns objectness.
    """
    batch, channels, grid_h, grid_w = logits.shape
    raw = logits.permute(0, 2, 3, 1)  # [B, gh, gw, 5 + C]
    positive = torch.zeros(batch, grid_h, grid_w, dtype=torch.bool, device=logits.device)
    target = torch.zeros(batch, grid_h, grid_w, 4, device=logits.device)
    for b, image_boxes in enumerate(boxes):
        for x1, y1, x2, y2 in image_boxes.tolist():
            cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
            gx, gy = min(int(cx), grid_w - 1), min(int(cy), grid_h - 1)
            positive[b, gy, gx] = True
            target[b, gy, gx] = torch.tensor([
                cx - gx, cy - gy,
                np.log(max(x2 - x1, 1.0) / stride), np.log(max(y2 - y1, 1.0) / stride),
            ]).clamp(-4.0, 4.0)

    objectness = raw[..., 4]
    # Drones fill a handful of cells, so average positives and negatives
    # separately to stop the empty background from dominating.
    loss = F.binary_cross_entropy_with_logits(objectness[~positive],
                                              torch.zeros_like(objectness[~positive]))
    if positive.any():
        pos = raw[positive]
        loss = loss + F.binary_cross_entropy_with_logits(pos[:, 4], torch.ones_like(pos[:, 4]))
        loss = loss + 5.0 * (F.mse_loss(pos[:, 0:2].sigmoid(), target[positive][:, 0:2])
                             + F.mse_loss(pos[:, 2:4], target[positive][:, 2:4]))
        loss = loss + F.binary_cross_entropy_with_logits(pos[:, 5:], torch.ones_like(pos[:, 5:]))
    return loss


# ---------------------------------------------------------------- train / eval

def model_stride(model, images):
    """Stride of the model's output grid; a mutated backbone may change it."""
    model.eval()
    with torch.no_grad():
        grid = model(images[:1]).shape[2:]
    stride_y, stride_x = images.shape[2] / grid[0], images.shape[3] / grid[1]
    if abs(stride_x - stride_y) > 1e-6:
        raise ValueError(f"non-uniform output stride: {stride_x} x {stride_y}")
    return stride_x


def train(model, data, epochs, batch_size, lr, device, stride):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    generator = torch.Generator().manual_seed(0)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(data["images"]), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            images = data["images"][batch].to(device).float() / 255
            loss = yolo_loss(model(images), [data["boxes"][i] for i in batch], stride)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at epoch {epoch + 1}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}/{epochs}: loss {total / len(order):.4f}", flush=True)


def predict_frames(model, data, batch_size, device):
    """Predict every val frame and pair it with ground truth in original pixels."""
    model.eval()
    model.confidence_threshold = EVAL_CONF
    width, height = data["orig_size"]
    scale = torch.tensor([width / data["images"].shape[3], height / data["images"].shape[2]] * 2)
    frames = []
    with torch.no_grad():
        for start in range(0, len(data["images"]), batch_size):
            images = data["images"][start:start + batch_size].to(device).float() / 255
            for offset, result in enumerate(model.predict(images)):
                i = start + offset
                boxes = (result["boxes"].cpu() * scale).tolist()
                scores = result["scores"].cpu().tolist()
                kept = [([max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)], s)
                        for (x1, y1, x2, y2), s in zip(boxes, scores)]
                kept = [(b, min(1.0, s)) for b, s in kept if b[2] > b[0] and b[3] > b[1]]
                target = data["orig_boxes"][i].tolist()
                frames.append({
                    "sample_id": data["sample_ids"][i],
                    "width": width,
                    "height": height,
                    "target": {"boxes": target, "labels": [0] * len(target)},
                    "prediction": {"boxes": [b for b, _ in kept],
                                   "labels": [0] * len(kept),
                                   "scores": [s for _, s in kept]},
                })
    return frames


def report_metadata(variant_path, model_path, config):
    return {
        "dataset_revision": REVISION,
        "manifest_identity": hashlib.sha256(json.dumps(
            {k: config[k] for k in ("train_sequences", "val_sequences", "frame_step")},
            sort_keys=True).encode()).hexdigest(),
        "split_identity": f"train={config['train_sequences']};val={config['val_sequences']}",
        "annotation_policy": "fred_coordinates_txt_clipped_to_image",
        "model_identity": sha256_file(variant_path),
        "checkpoint_identity": sha256_file(model_path),
        "code_revision": code_revision(),
        "postprocessing": f"DroneDetector.predict conf={EVAL_CONF} nms_iou=0.5",
        "training_config_identity": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "modality": "event",
        "benchmark_split": BENCHMARK_SPLIT,
        "project_split": "validation",
        "official_split": OFFICIAL_SPLIT,
    }


def main(gene_id, model_name, variant_dir, train_sequences, val_sequences,
         epochs, frame_step, img_width, batch_size, lr, device):
    start_time = time.time()
    torch.manual_seed(0)
    config = {"train_sequences": train_sequences, "val_sequences": val_sequences,
              "epochs": epochs, "frame_step": frame_step, "img_width": img_width,
              "batch_size": batch_size, "lr": lr, "seed": 0}

    # This is LLM Guided Code
    # Import the variant module dynamically and build the detector
    try:
        module, variant_path = load_variant(model_name, variant_dir)
        model = module.DroneDetector(num_classes=1).to(device)
    except Exception as e:
        # If the LLM-generated architecture is broken, write error results and exit
        write_failure_results(gene_id, start_time, repr(e))
        return

    param_count = sum(p.numel() for p in model.parameters())

    train_data = load_split("train", train_sequences, frame_step, img_width)
    val_data = load_split("train", val_sequences, frame_step, img_width)
    print(f"train frames: {len(train_data['images'])}, val frames: {len(val_data['images'])}, "
          f"params: {param_count}, device: {device}", flush=True)

    try:
        stride = model_stride(model, train_data["images"].to(device).float() / 255)
        model.output_stride = stride  # keep predict's decoding consistent with training
        train(model, train_data, epochs, batch_size, lr, device, stride)

        # Save the trained model under trained_models/<gene_id>.pt
        model_dir = SCRIPT_DIR / "trained_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = model_dir / f"{gene_id}.pt"
        torch.save(model.state_dict(), model_path)

        # Predict on the val frames and score them with map_reporter
        frames = predict_frames(model, val_data, batch_size, device)
        payload = {
            "metadata": report_metadata(variant_path, model_path, config),
            "frames": frames,
            "params": param_count,
            "expected_sample_ids": val_data["sample_ids"],
        }
        result, report_path = map_reporter.report(payload, sota_root=SCRIPT_DIR,
                                                  gene_id=gene_id, mode="validation")
    except Exception as e:
        write_failure_results(gene_id, start_time, repr(e))
        return
    train_time = time.time() - start_time

    print(f"mAP50: {result.map50}, mAP50:95: {result.map50_95}, "
          f"Params: {result.params}, Time: {train_time:.1f}s")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print('='*120);print('job done');print('='*120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an evolved FRED YOLO11 variant")
    parser.add_argument("--model", type=str, default="network",
                        help='Variant module name like "network_XXXX"; "network" trains the seed')
    parser.add_argument("--variant_dir", type=str, default=str(SCRIPT_DIR / "models"),
                        help="Directory where models are written by LLM-GE")
    parser.add_argument("--train_sequences", type=str,
                        default=os.getenv("FRED_TRAIN_SEQUENCES", TRAIN_SEQUENCES),
                        help="Comma-separated FRED train/ sequence IDs to train on")
    parser.add_argument("--val_sequences", type=str,
                        default=os.getenv("FRED_VAL_SEQUENCES", VAL_SEQUENCES),
                        help="Comma-separated FRED train/ sequence IDs to validate on")
    parser.add_argument("--epochs", type=int,
                        default=int(os.getenv("FRED_EPOCHS", PROXY_EPOCHS)),
                        help="Proxy training epochs")
    parser.add_argument("--frame_step", type=int,
                        default=int(os.getenv("FRED_FRAME_STEP", FRAME_STEP)),
                        help="Use every Nth event frame of each sequence")
    parser.add_argument("--img_width", type=int, default=int(os.getenv("FRED_IMG_WIDTH", "640")),
                        help="Resize frames to this width (height keeps the aspect ratio)")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("FRED_BATCH", "16")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("FRED_LR", "1e-3")))
    parser.add_argument("--device", type=str,
                        default=os.getenv("FRED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    train_sequences = [s.strip() for s in args.train_sequences.split(",") if s.strip()]
    val_sequences = [s.strip() for s in args.val_sequences.split(",") if s.strip()]
    if set(train_sequences) & set(val_sequences):
        parser.error("train and val sequences must not overlap")
    if args.epochs <= 0 or args.frame_step <= 0 or args.img_width < 32:
        parser.error("epochs and frame_step must be positive, img_width at least 32")

    # this will get the gene id value: "network_XXXX" -> "XXXX"
    try:
        gene_id = args.model.split('network_')[1]
    except IndexError:
        gene_id = 'seed'

    main(
        gene_id,
        args.model,
        args.variant_dir,
        train_sequences,
        val_sequences,
        epochs=args.epochs,
        frame_step=args.frame_step,
        img_width=args.img_width,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
