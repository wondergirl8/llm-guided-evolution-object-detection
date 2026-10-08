"""
Train and evaluate one LLM-generated YOLO11 variant on FRED.

Called by run_improved.py through EVAL_RUNLINE:
    uv run python train_eval.py --model network_<gene_id> --variant_dir <models dir>

The variant module must define build_config(base_config) -> dict, which
returns an edited copy of the YOLO11 model YAML mapping.

Training uses a short proxy budget (few epochs on a fraction of the training
images). Scores are only for ranking candidates against each other during
evolution; elites are retrained at full budget later. Validation always uses
the full val split so every candidate is scored on the same frames.

After training, predictions on the val split are scored by the shared
map_reporter.py, which writes results/<gene_id>_results.csv for
run_improved.py and results/<gene_id>_evaluation.json for people.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import importlib
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = SCRIPT_DIR.parents[1]
sys.path.append(str(PHASE1_DIR))
import map_reporter
from result_contract import DetectionResult, write_results

# Provisional proxy budget; must match the values the seed baseline uses
# (to be agreed with Member 3). The formal budget is still an open project
# decision (DG-P1-08 / DG-P1-09 in PHASE1_DETECTION_PLAN.md).
PROXY_EPOCHS = 2
PROXY_FRACTION = 0.25

# Prediction settings passed to map_reporter. A low confidence threshold keeps
# low-score boxes so mAP is computed over the full precision/recall curve.
PREDICT_CONF = 0.001
PREDICT_IOU = 0.7
PREDICT_MAX_DET = 100

# Worst-case values for FITNESS_WEIGHTS = (1.0, 1.0, -1.0).
FAILED_RESULT = DetectionResult(0.0, 0.0, 999999999)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


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


def build_candidate(module, weights, config_dir):
    """Apply the variant's build_config to the checkpoint's YOLO11 YAML."""
    from ultralytics import YOLO

    base_config = YOLO(weights).model.yaml
    config = module.build_config(deepcopy(base_config))
    # Single drone class with the standard detection head is protected.
    config["nc"] = 1
    if config["head"][-1][2] != "Detect":
        raise ValueError("candidate removed the YOLO11 Detect head")
    # Ultralytics infers the model scale from the YAML file name, so keep the
    # checkpoint's name (e.g. yolo11m.yaml) for the scale to match the weights.
    config_path = Path(config_dir) / f"{Path(weights).stem}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return YOLO(str(config_path)).load(weights)


def val_images(data):
    """Return the val image paths and their YOLO label directory."""
    config = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    image_dir = Path(config["path"]) / config["val"]
    # Ultralytics convention: labels mirror images/ under labels/.
    label_dir = Path(str(image_dir).replace(f"{os.sep}images", f"{os.sep}labels", 1))
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no val images in {image_dir}")
    return images, label_dir


def read_targets(label_path, width, height):
    """Convert one YOLO label file (normalized xywh) to pixel xyxy boxes."""
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, xc, yc, w, h = map(float, line.split())
            x1, x2 = max(0.0, (xc - w/2) * width), min(width, (xc + w/2) * width)
            y1, y2 = max(0.0, (yc - h/2) * height), min(height, (yc + h/2) * height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return {"boxes": boxes, "labels": [0] * len(boxes)}


def predict_frames(model, images, label_dir, imgsz, batch, device):
    """Run the trained model on every val image and pair it with ground truth."""
    frames = []
    results = model.predict(source=[str(p) for p in images], conf=PREDICT_CONF,
                            iou=PREDICT_IOU, max_det=PREDICT_MAX_DET, imgsz=imgsz,
                            batch=batch, device=device, stream=True, verbose=False)
    for image, result in zip(images, results):
        height, width = result.orig_shape
        boxes, scores = [], []
        for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x1, y1 = max(0.0, box[0]), max(0.0, box[1])
            x2, y2 = min(float(width), box[2]), min(float(height), box[3])
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(min(1.0, max(0.0, score)))
        frames.append({
            "sample_id": image.stem,
            "width": int(width),
            "height": int(height),
            "target": read_targets(label_dir / f"{image.stem}.txt", width, height),
            "prediction": {"boxes": boxes, "labels": [0] * len(boxes), "scores": scores},
        })
    return frames


def report_metadata(data, variant_path, model_path, train_config):
    """Provenance map_reporter requires; read from the export's source.json."""
    source_path = Path(data).parent / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return {
        "dataset_revision": source["dataset_revision"],
        "manifest_identity": source["manifest_sha256"],
        "split_identity": source["project_split_approval_reference"],
        # "none" means the export kept the original FRED boxes unchanged.
        "annotation_policy": source.get("annotation_policy") or "none",
        "model_identity": sha256_file(variant_path),
        "checkpoint_identity": sha256_file(model_path),
        "code_revision": code_revision(),
        "postprocessing": (f"ultralytics predict conf={PREDICT_CONF} "
                           f"iou={PREDICT_IOU} max_det={PREDICT_MAX_DET}"),
        "training_config_identity": hashlib.sha256(
            json.dumps(train_config, sort_keys=True).encode()).hexdigest(),
        "modality": "event",
        "benchmark_split": "challenging",
        "project_split": "validation",
        "official_split": "challenging_train",
    }


def main(gene_id, module_name, variant_dir, data, weights, epochs, fraction,
         batch, imgsz, device):
    start_time = time.time()

    # This is LLM Guided Code
    # Import the variant module dynamically
    try:
        sys.path.append(str(variant_dir))
        module = importlib.import_module(module_name)
    except Exception as e:
        write_failure_results(gene_id, start_time, repr(e))
        return

    try:
        with tempfile.TemporaryDirectory() as config_dir:
            model = build_candidate(module, weights, config_dir)
    except Exception as e:
        # If the LLM-generated architecture is broken, write error results and exit
        write_failure_results(gene_id, start_time, repr(e))
        return

    param_count = sum(p.numel() for p in model.model.parameters())
    train_config = {"data": str(data), "weights": str(weights), "epochs": epochs,
                    "fraction": fraction, "batch": batch, "imgsz": imgsz, "seed": 0}

    try:
        # fraction subsets only the training images; val stays complete.
        model.train(data=str(data), epochs=epochs, fraction=fraction,
                    batch=batch, imgsz=imgsz, device=device, seed=0,
                    deterministic=True, project=str(SCRIPT_DIR / "runs"),
                    name=gene_id, exist_ok=True)

        # Save the best checkpoint under trained_models/<gene_id>.pt
        model_dir = SCRIPT_DIR / "trained_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = model_dir / f"{gene_id}.pt"
        shutil.copy(model.trainer.best, model_path)

        # Predict on the full val split and score it with map_reporter
        from ultralytics import YOLO
        trained = YOLO(str(model_path))
        images, label_dir = val_images(data)
        frames = predict_frames(trained, images, label_dir, imgsz, batch, device)
        payload = {
            "metadata": report_metadata(data, module.__file__, model_path, train_config),
            "frames": frames,
            "params": param_count,
            "expected_sample_ids": [image.stem for image in images],
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
    parser = argparse.ArgumentParser(description="Train and evaluate an evolved YOLO11 variant")
    parser.add_argument("--model", type=str, default="network",
                        help='Variant module name like "network_XXXX"')
    parser.add_argument("--variant_dir", type=str, default=str(SCRIPT_DIR / "models"),
                        help="Directory where models are written by LLM-GE")
    parser.add_argument("--data", type=str,
                        default=os.getenv("FRED_YOLO_DATA", "data/fred_yolo11_bringup/data.yaml"),
                        help="YOLO data YAML for the exported FRED subset")
    parser.add_argument("--weights", type=str,
                        default=os.getenv("FRED_YOLO_WEIGHTS", "yolo11m.pt"),
                        help="Pretrained YOLO11 checkpoint")
    parser.add_argument("--epochs", type=int,
                        default=int(os.getenv("FRED_YOLO_EPOCHS", PROXY_EPOCHS)),
                        help="Proxy training epochs")
    parser.add_argument("--fraction", type=float,
                        default=float(os.getenv("FRED_YOLO_FRACTION", PROXY_FRACTION)),
                        help="Share of training images used by the proxy budget")
    parser.add_argument("--batch", type=int, default=int(os.getenv("FRED_YOLO_BATCH", "8")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("FRED_YOLO_IMGSZ", "640")))
    parser.add_argument("--device", type=str, default=os.getenv("FRED_YOLO_DEVICE", "0"))
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")

    # this will get the gene id value: "network_XXXX" -> "XXXX"
    try:
        gene_id = args.model.split('network_')[1]
    except IndexError:
        gene_id = 'seed'

    main(
        gene_id,
        args.model,
        args.variant_dir,
        args.data,
        args.weights,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
