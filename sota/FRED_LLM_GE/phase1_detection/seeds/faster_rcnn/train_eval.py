"""Standalone seed/candidate training for the provisional Faster R-CNN pilot."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import importlib.metadata

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
FRED_DIR = SCRIPT_DIR.parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache-faster-rcnn" / "huggingface"))
for path in (ROOT, SCRIPT_DIR.parents[1], FRED_DIR / "data", FRED_DIR / "phase0_data"):
    sys.path.insert(0, str(path))
import numpy as np
import torch
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.validator import SEED, validate_source
from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.network import build_detector
from sota.FRED_LLM_GE.phase1_detection.adapters.faster_rcnn import faster_rcnn_frame
from result_contract import DetectionResult, write_results, validate_gene_id
import map_reporter
from event_data import load_split, TRAIN_SEQUENCES, VAL_SEQUENCES, FRAME_STEP
from fred_stream import REVISION


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def load_variant(name, directory):
    path = SEED if name == "network" else Path(directory) / f"{name}.py"
    genes = validate_source(path)
    spec = importlib.util.spec_from_file_location("fred_faster_rcnn_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model_config(), genes, path


def prepare_batch(model, data, indices, training):
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    images, samples = [], []
    for i in indices:
        image = data["images"][i]
        h, w = image.shape[-2:]
        ow, oh = data["orig_size"]
        sample = DetDataSample(metainfo={
            "img_shape": (h, w), "ori_shape": (oh, ow),
            "scale_factor": (w / ow, h / oh), "img_id": data["sample_ids"][i]})
        sample.gt_instances = InstanceData(
            bboxes=data["boxes"][i].clone(),
            labels=torch.zeros(len(data["boxes"][i]), dtype=torch.long))
        images.append(image)
        samples.append(sample)
    return model.data_preprocessor({"inputs": images, "data_samples": samples}, training)


def synthetic_data():
    return {"images": torch.randint(0, 256, (2, 3, 64, 96), dtype=torch.uint8),
            "boxes": [torch.tensor([[20., 20., 40., 40.]]), torch.empty((0, 4))],
            "orig_boxes": [torch.tensor([[40., 40., 80., 80.]]), torch.empty((0, 4))],
            "orig_size": (192, 128), "sample_ids": ["synthetic:0", "synthetic:1"]}


def train(model, data, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(data["images"])).tolist()
        for start in range(0, len(order), args.batch_size):
            batch = prepare_batch(model, data, order[start:start + args.batch_size], True)
            optimizer.zero_grad()
            loss, _ = model.parse_losses(model(**batch, mode="loss"))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise ValueError("Non-finite gradients")
            optimizer.step()
        print(f"epoch {epoch + 1}/{args.epochs}: final batch loss={loss.item():.5f}", flush=True)


def predict_frames(model, data, batch_size):
    model.eval()
    frames = []
    with torch.no_grad():
        for start in range(0, len(data["images"]), batch_size):
            indices = list(range(start, min(start + batch_size, len(data["images"]))))
            batch = prepare_batch(model, data, indices, False)
            predictions = model(**batch, mode="predict")  # MMDetection restores original pixels.
            if len(predictions) != len(indices):
                raise ValueError("Prediction count differs from input count")
            for i, prediction in zip(indices, predictions):
                target = {"boxes": data["orig_boxes"][i].tolist(),
                          "labels": [0] * len(data["orig_boxes"][i])}
                frames.append(faster_rcnn_frame(prediction, sample_id=data["sample_ids"][i], target=target))
    return frames


def preflight(model):
    """Reject broken model interfaces before streaming or full candidate training."""
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        data = synthetic_data()
        model.train()
        batch = prepare_batch(model, data, [0, 1], True)
        loss, _ = model.parse_losses(model(**batch, mode="loss"))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite preflight loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise ValueError("Non-finite preflight gradients")
        model.zero_grad(set_to_none=True)
        predict_frames(model, data, 2)


def run(args):
    gene_id = "smoke" if args.smoke else ("seed" if args.model == "network" else args.model.removeprefix("network_"))
    validate_gene_id(gene_id)
    run_dir = SCRIPT_DIR / "reports" / gene_id
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = vars(args).copy()
    start = time.time()
    stage = "candidate_validation"
    try:
        cfg, genes, path = load_variant(args.model, args.variant_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "candidate_source.py").write_text(path.read_text())
        source_paths = [Path(__file__), SCRIPT_DIR / "network.py", SCRIPT_DIR / "validator.py", SCRIPT_DIR.parents[1] / "event_data.py"]
        write_json(run_dir / "code_identity.json", {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest() for source in source_paths})
        stage = "model_initialization"
        model, transfer = build_detector(cfg, args.checkpoint)
        model.to(args.device)
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        write_json(run_dir / "initialization.json", transfer)
        write_json(run_dir / "resolved_config.json", {"model": cfg.to_dict(), "training": config, "genes": genes,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "mmcv", "mmengine", "mmdet", "datasets", "pycocotools")}})
        params = sum(parameter.numel() for parameter in model.parameters())
        stage = "model_preflight"
        preflight(model)
        stage = "data_loading"
        if args.smoke:
            train_data = val_data = synthetic_data()
            args.epochs = 1
        else:
            train_data = load_split("train", args.train_sequences.split(","), args.frame_step, args.img_width)
            val_data = load_split("train", args.val_sequences.split(","), args.frame_step, args.img_width)
        stage = "training"
        train(model, train_data, args)
        model_path = SCRIPT_DIR / "trained_models" / f"{gene_id}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"state_dict": model.state_dict(), "config": config, "genes": genes}, temporary)
        os.replace(temporary, model_path)
        stage = "evaluation"
        frames = predict_frames(model, val_data, args.batch_size)
        if args.smoke:
            write_json(run_dir / "smoke.json", {"status": "passed", "device": args.device, "frames": frames, "params": params})
            write_json(run_dir / "status.json", {"status": "success", "mode": "smoke", "device": args.device})
            return
        metadata = {
            "dataset_revision": REVISION,
            "manifest_identity": hashlib.sha256(json.dumps({key: config[key] for key in ("train_sequences", "val_sequences", "frame_step")}, sort_keys=True).encode()).hexdigest(),
            "split_identity": f"train={args.train_sequences};val={args.val_sequences}",
            "annotation_policy": "fred_coordinates_txt_clipped_to_image",
            "model_identity": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_identity": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "postprocessing": "MMDetection rescale=True score=0.05 rcnn_nms=0.5 max_per_img=100",
            "training_config_identity": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "modality": "event", "benchmark_split": "challenging",
            "project_split": "validation", "official_split": "challenging_train"}
        result, report = map_reporter.report({"metadata": metadata, "frames": frames,
            "params": params, "expected_sample_ids": val_data["sample_ids"]},
            sota_root=SCRIPT_DIR, gene_id=gene_id, mode="validation")
        write_json(run_dir / "status.json", {"status": "success", "exploratory": True,
            "seconds": time.time() - start, "fitness": vars(result), "report": str(report),
            "training_frames": len(train_data["images"]), "validation_frames": len(frames),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") else None})
        print(f"mAP50={result.map50}, mAP50:95={result.map50_95}, params={params}", flush=True)
    except Exception as error:
        write_json(run_dir / "status.json", {"status": "failed", "stage": stage,
            "error": repr(error), "traceback": traceback.format_exc(), "training": config})
        if not args.smoke:
            write_results(SCRIPT_DIR, gene_id, DetectionResult(0.0, 0.0, 999999999))
        raise
    finally:
        print("job done", flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="network")
    parser.add_argument("--variant_dir", default=str(SCRIPT_DIR / "models"))
    parser.add_argument("--checkpoint", default=str(SCRIPT_DIR / "checkpoints/coco.pth"))
    parser.add_argument("--train_sequences", default=TRAIN_SEQUENCES)
    parser.add_argument("--val_sequences", default=VAL_SEQUENCES)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--frame_step", type=int, default=FRAME_STEP)
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ("train_sequences", "val_sequences"):
        entries = [entry.strip() for entry in getattr(args, name).split(",")]
        if not entries or any(not entry.isdigit() for entry in entries) or len(set(entries)) != len(entries):
            parser.error(f"{name} requires distinct numeric sequence IDs")
        setattr(args, name, ",".join(entries))
    if set(args.train_sequences.split(",")) & set(args.val_sequences.split(",")):
        parser.error("Training and validation sequences overlap")
    if min(args.epochs, args.batch_size, args.frame_step) <= 0 or args.img_width < 32 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Invalid training budget or learning rate")
    run(args)


if __name__ == "__main__":
    cli()
