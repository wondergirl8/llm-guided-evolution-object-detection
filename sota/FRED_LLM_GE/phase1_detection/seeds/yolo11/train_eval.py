"""Train a protected Event-only YOLO11 candidate and use the shared COCO evaluator.

A bounded bring-up checks integration only; it does not establish a formal
baseline or authorize using smoke scores for research selection.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import yaml

try:
    from .validator import load_model_module
    from ... import map_reporter
    from ...adapters.yolo11 import yolo11_frame
    from ...result_contract import validate_gene_id
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from validator import load_model_module
    import map_reporter
    from adapters.yolo11 import yolo11_frame
    from result_contract import validate_gene_id

SCRIPT_DIR = Path(__file__).resolve().parent
PROXY_EPOCHS = 2
PROXY_FRACTION = 0.25
PREDICT_CONF = 0.001
PREDICT_IOU = 0.7
PREDICT_MAX_DET = 100
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_failure_results(gene_id, start_time, message, category="training_or_evaluation"):
    """Persist a failed attempt without publishing valid fitness or completion."""
    validate_gene_id(gene_id)
    directory = SCRIPT_DIR / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{gene_id}_failure.json"
    record = {"status": "failed", "gene_id": gene_id, "category": category,
              "message": message, "elapsed_seconds": time.time() - start_time,
              "code_revision": code_revision()}
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)
    print(f"FRED_RUN_FAILED [{category}]: {message}", flush=True)
    print(f"Failure record: {output}", flush=True)
    return output


def build_candidate(module, weights, config_dir):
    from ultralytics import YOLO

    if Path(weights).name != "yolo11m.pt":
        raise ValueError("bring-up requires the verified yolo11m.pt checkpoint")
    base = YOLO(weights)
    base_config = base.model.yaml
    if base_config.get("scale") not in (None, "m"):
        raise ValueError("checkpoint is not YOLO11m")
    config = module.build_config(deepcopy(base_config))
    if config.get("nc") != 1 or config["head"][-1][2] != "Detect":
        raise ValueError("candidate changed protected detection semantics")
    config["scale"] = "m"
    config_path = Path(config_dir) / "yolo11m.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model = YOLO(str(config_path)).load(weights)
    base_params = sum(p.numel() for p in base.model.parameters())
    params = sum(p.numel() for p in model.model.parameters())
    if model.model.yaml.get("scale") != "m" or params < 0.8 * base_params:
        raise ValueError("candidate was not constructed at the verified YOLO11m scale")
    return model


def validate_forward_shape(model, imgsz, device):
    """Exercise the real detection forward path before submitting training work."""
    import numpy as np
    results = model.predict(source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                            imgsz=imgsz, device=device, verbose=False)
    if len(results) != 1 or tuple(results[0].orig_shape) != (imgsz, imgsz):
        raise ValueError("candidate failed the detection forward shape check")


def _data_config(data):
    data = Path(data).resolve()
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("names") != {0: "drone"}:
        raise ValueError("FRED data must declare exactly one drone class")
    root = Path(config["path"])
    root = root if root.is_absolute() else data.parent / root
    return config, root.resolve()


def val_images(data):
    config, root = _data_config(data)
    image_dir = root / config["val"]
    if image_dir.parent.name != "images":
        raise ValueError("export must use images/<split> and labels/<split>")
    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no val images in {image_dir}")
    return images, label_dir


def read_targets(label_path, width, height):
    label_path = Path(label_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"missing labels; empty-frame labels must exist: {label_path}")
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"malformed/non-finite YOLO label: {label_path}")
        cls, xc, yc, w, h = values
        if cls != 0 or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"invalid drone label: {label_path}")
        x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
        # Tolerance accounts only for eight-decimal export serialization.
        if min(x1, y1) < -1e-7 or max(x2, y2) > 1+1e-7:
            raise ValueError(f"out-of-bounds exported label: {label_path}")
        boxes.append([max(0, x1)*width, max(0, y1)*height,
                      min(1, x2)*width, min(1, y2)*height])
    return {"boxes": boxes, "labels": [0] * len(boxes)}


def validate_data(data):
    from PIL import Image
    config, root = _data_config(data)
    source = json.loads((Path(data).parent / "source.json").read_text(encoding="utf-8"))
    for key in ("dataset_revision", "manifest_sha256", "project_split_approval_reference"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"export is missing provenance: {key}")
    if source["dataset_revision"] != "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd":
        raise ValueError("export differs from the pinned FRED release")
    identities = []
    for split in ("train", "val"):
        image_dir = root / config[split]
        if image_dir.parent.name != "images":
            raise ValueError("export must use images/<split>")
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"no {split} images")
        stems = {p.stem for p in images}
        if len(stems) != len(images):
            raise ValueError("duplicate frame identity")
        identities.append(stems)
        label_dir = image_dir.parent.parent / "labels" / image_dir.name
        for image in images:
            with Image.open(image) as frame:
                width, height = frame.size
                frame.verify()
            read_targets(label_dir / f"{image.stem}.txt", width, height)
    if identities[0] & identities[1]:
        raise ValueError("train/validation frame identities overlap")
    return source


def predict_frames(model, images, label_dir, imgsz, batch, device):
    results = iter(model.predict(source=[str(p) for p in images], conf=PREDICT_CONF,
                                iou=PREDICT_IOU, max_det=PREDICT_MAX_DET, imgsz=imgsz,
                                batch=batch, device=device, stream=True, verbose=False))
    frames = []
    for image in images:
        result = next(results, None)
        if result is None:
            raise ValueError("model returned fewer predictions than validation images")
        if Path(result.path).resolve() != image.resolve():
            raise ValueError("prediction order/identity differs from validation input")
        height, width = result.orig_shape
        frames.append(yolo11_frame(
            sample_id=image.stem,
            target=read_targets(label_dir / f"{image.stem}.txt", width, height),
            result=result,
        ))
    if next(results, None) is not None:
        raise ValueError("model returned extra validation predictions")
    return frames


def report_metadata(data, variant_path, model_path, train_config):
    source = json.loads((Path(data).parent / "source.json").read_text(encoding="utf-8"))
    code_files = [Path(__file__), Path(__file__).with_name("network.py"),
                  Path(__file__).with_name("validator.py"),
                  Path(map_reporter.__file__), Path(map_reporter.__file__).with_name("fitness.py"),
                  Path(map_reporter.__file__).with_name("result_contract.py"),
                  Path(map_reporter.__file__).parent / "adapters" / "yolo11.py"]
    return {
        "dataset_revision": source["dataset_revision"],
        "manifest_identity": source["manifest_sha256"],
        "split_identity": source["project_split_approval_reference"],
        "annotation_policy": source.get("annotation_policy") or "none",
        "model_identity": sha256_file(variant_path),
        "checkpoint_identity": sha256_file(model_path),
        "code_revision": code_revision(),
        "code_files_sha256": {str(p.name): sha256_file(p) for p in code_files},
        "environment_versions": {name: importlib.metadata.version(name)
                                 for name in ("PyYAML", "pycocotools", "Pillow")},
        "source_record_sha256": sha256_file(Path(data).parent / "source.json"),
        "postprocessing": f"ultralytics predict conf={PREDICT_CONF} iou={PREDICT_IOU} max_det={PREDICT_MAX_DET}",
        "training_config_identity": hashlib.sha256(json.dumps(train_config, sort_keys=True).encode()).hexdigest(),
        "modality": "event", "benchmark_split": "challenging",
        "project_split": "validation", "official_split": "challenging_train",
        "experiment_purpose": source.get("purpose", "unspecified"),
        "candidate_gene_id": train_config["gene_id"],
    }


def run(model_name, variant_dir, data, weights, epochs, batch, imgsz, device,
        preflight_only=False, run_id=None, fraction=1.0):
    if min(epochs, batch, imgsz) <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid training budget")
    module = load_model_module(model_name, Path(variant_dir))
    gene_id = "seed" if model_name == "network" else model_name.removeprefix("network_")
    output_id = validate_gene_id(run_id or gene_id)
    for path in (SCRIPT_DIR / "runs" / output_id,
                 SCRIPT_DIR / "results" / f"{output_id}_results.csv",
                 SCRIPT_DIR / "results" / f"{output_id}_evaluation.json",
                 SCRIPT_DIR / "trained_models" / f"{output_id}.pt"):
        if path.exists():
            raise FileExistsError(f"refusing to reuse run artifacts: {path}")
    validate_data(Path(data))
    train_config = {"data": str(Path(data).resolve()), "weights": weights,
                    "epochs": epochs, "fraction": fraction, "batch": batch,
                    "imgsz": imgsz, "seed": 0, "device": str(device),
                    "deterministic": True, "gene_id": gene_id}
    with tempfile.TemporaryDirectory(prefix="fred-yolo11-config-") as config_dir:
        model = build_candidate(module, weights, config_dir)
        params = sum(p.numel() for p in model.model.parameters())
        validate_forward_shape(model, imgsz, device)
        if preflight_only:
            print(f"preflight passed: {model_name}; scale m; {params} parameters", flush=True)
            return None
        model.train(data=str(data), epochs=epochs, fraction=fraction, batch=batch,
                    imgsz=imgsz, device=device, seed=0, deterministic=True,
                    project=str(SCRIPT_DIR / "runs"), name=output_id, exist_ok=False, save=True)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError("training produced no best checkpoint")
        model_dir = SCRIPT_DIR / "trained_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{output_id}.pt"
        shutil.copyfile(best, model_path)
        from ultralytics import YOLO
        trained = YOLO(str(model_path))
        images, label_dir = val_images(data)
        payload = {"metadata": report_metadata(data, module.__file__, model_path, train_config),
                   "frames": predict_frames(trained, images, label_dir, imgsz, batch, device),
                   "params": params, "expected_sample_ids": [p.stem for p in images]}
        result, report_path = map_reporter.report(payload, sota_root=SCRIPT_DIR,
                                                  gene_id=output_id, mode="validation")
    print(f"mAP50: {result.map50}, mAP50:95: {result.map50_95}, Params: {result.params}")
    print(f"Saved model: {model_path}\nSaved report: {report_path}")
    print("job done", flush=True)
    return SCRIPT_DIR / "results" / f"{output_id}_results.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="network")
    parser.add_argument("--variant_dir", type=Path, default=SCRIPT_DIR / "models")
    parser.add_argument("--data", type=Path, default=Path(os.getenv("FRED_YOLO_DATA", "data/fred_yolo11_bringup/data.yaml")))
    parser.add_argument("--weights", default=os.getenv("FRED_YOLO_WEIGHTS", "yolo11m.pt"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("FRED_YOLO_EPOCHS", PROXY_EPOCHS)))
    parser.add_argument("--fraction", type=float, default=float(os.getenv("FRED_YOLO_FRACTION", PROXY_FRACTION)))
    parser.add_argument("--batch", type=int, default=int(os.getenv("FRED_YOLO_BATCH", "2")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("FRED_YOLO_IMGSZ", "640")))
    parser.add_argument("--device", default=os.getenv("FRED_YOLO_DEVICE", "0"))
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    output_id = args.run_id or ("seed" if args.model == "network" else args.model.removeprefix("network_"))
    try:
        validate_gene_id(output_id)
    except ValueError as error:
        parser.error(str(error))
    try:
        started = time.time()
        run(args.model, args.variant_dir, args.data, args.weights, args.epochs,
            args.batch, args.imgsz, args.device, args.preflight_only, args.run_id, args.fraction)
    except Exception as error:
        category = "existing_run" if isinstance(error, FileExistsError) else "training_or_evaluation"
        write_failure_results(output_id, locals().get("started", time.time()), repr(error), category)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
