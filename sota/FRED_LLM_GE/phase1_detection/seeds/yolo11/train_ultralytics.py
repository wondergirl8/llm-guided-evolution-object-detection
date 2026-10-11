"""Train an actual YOLO11 candidate from scratch; reuse the team's FRED cache.

This is the option-2 engineering experiment, not a paper reproduction. The
standalone seed and candidates use the same data, budget, loss and evaluator.
"""

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import torch
from torch import nn
from torch.nn import functional as F

import train_eval as shared
import option_b

SCRIPT_DIR = Path(__file__).resolve().parent
SEED_FILE = shared.FRED_DIR / "seeds" / "seed_ultralytics_yolo11.py"
MARKER = "# --OPTION--"


def load_variant(model_name, variant_dir):
    path = SEED_FILE if model_name == "network" else Path(variant_dir) / f"{model_name}.py"
    parts = path.read_text().split(MARKER)
    reference = SEED_FILE.read_text().split(MARKER)
    if len(parts) != 3 or ast.dump(ast.parse(parts[0])) != ast.dump(ast.parse(reference[0])):
        raise ValueError("Candidate changed the protected YOLO11 wrapper/head/loss")
    for part, name in zip(parts[1:], ("YOLO11Backbone", "YOLO11Neck")):
        tree = ast.parse(part)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef) or tree.body[0].name != name:
            raise ValueError(f"Mutable section must define only {name}")
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
            raise ValueError("Imports belong in the protected preamble")
    if model_name != "network" and ast.dump(ast.parse(path.read_text())) == ast.dump(ast.parse(SEED_FILE.read_text())):
        raise ValueError("Generated candidate is unchanged from the seed")
    spec = importlib.util.spec_from_file_location(f"fred_ultralytics_{model_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def preflight(model, device):
    """Check resource, P3/P4/P5, loss, and gradients before downloading data."""
    params = sum(p.numel() for p in model.parameters())
    if params <= 0 or params > 2 * model.detector.reference_params:
        raise ValueError("Candidate exceeds twice the official seed parameter count")
    model.train()
    # Check gradients without altering BatchNorm statistics of random weights.
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
    images = torch.zeros(2, 3, 64, 64, device=device)
    with torch.no_grad():
        features = model.detector.model[:2](images)
        head = model.detector.model[-1]
        expected_channels = [block[0].conv.in_channels for block in head.cv2]
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise ValueError("Neck must return three P3/P4/P5 feature maps")
        for feature, channels, stride in zip(features, expected_channels, (8, 16, 32)):
            if tuple(feature.shape) != (2, channels, 64 // stride, 64 // stride):
                raise ValueError("Candidate broke the protected P3/P4/P5 interface")
    batch = {"img": images, "batch_idx": torch.tensor([0, 1], device=device),
             "cls": torch.zeros(2, 1, device=device),
             "bboxes": torch.tensor([[.5, .5, .25, .25]] * 2, device=device)}
    loss, _ = model.loss(batch)
    if not torch.isfinite(loss).all():
        raise ValueError("Candidate preflight loss is non-finite")
    loss.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("Candidate preflight gradients are absent or non-finite")
    model.zero_grad(set_to_none=True)
    model.train()
    return {"params": params, "loss": float(loss.detach().sum()), "status": "passed"}


def images_for_model(images, device):
    images = images.to(device).float() / 255
    height, width = images.shape[-2:]
    return F.pad(images, (0, -width % 32, 0, -height % 32), value=114 / 255)


def training_batch(data, indices, device):
    images = images_for_model(data["images"][indices], device)
    height, width = images.shape[-2:]
    boxes, batch_indices = [], []
    for index, data_index in enumerate(indices.tolist()):
        xyxy = data["boxes"][data_index].to(device)
        if (not torch.isfinite(xyxy).all() or
                (len(xyxy) and not ((xyxy[:, 2:] > xyxy[:, :2]).all() and (xyxy >= 0).all()))):
            raise ValueError("Cached training boxes must be finite positive-area xyxy")
        xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2,
                          xyxy[:, 2:] - xyxy[:, :2]), dim=1)
        boxes.append(xywh / xywh.new_tensor([width, height, width, height]))
        batch_indices.append(torch.full((len(xyxy),), index, device=device))
    boxes = torch.cat(boxes)
    return {"img": images, "bboxes": boxes,
            "cls": torch.zeros(len(boxes), 1, device=device),
            "batch_idx": torch.cat(batch_indices)}


def train(model, data, epochs, batch_size, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    generator = torch.Generator().manual_seed(0)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(data["images"]), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            loss, _ = model.loss(training_batch(data, indices, device))
            loss = loss.sum()
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch + 1}")
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"Non-finite gradients at epoch {epoch + 1}")
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1}/{epochs}: loss per image {total / len(order):.4f}", flush=True)


def predict_frames(model, data, batch_size, device):
    model.eval()
    model.confidence_threshold = shared.EVAL_CONF
    width, height = data["orig_size"]
    scale = torch.tensor([width / data["images"].shape[3], height / data["images"].shape[2]] * 2)
    frames = []
    for start in range(0, len(data["images"]), batch_size):
        images = images_for_model(data["images"][start:start + batch_size], device)
        for offset, prediction in enumerate(model.predict(images)):
            index = start + offset
            boxes = prediction["boxes"].cpu() * scale
            boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, width)
            boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, height)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            targets = data["orig_boxes"][index].tolist()
            frames.append({"sample_id": data["sample_ids"][index], "width": width, "height": height,
                           "target": {"boxes": targets, "labels": [0] * len(targets)},
                           "prediction": {"boxes": boxes[keep].tolist(),
                                          "scores": prediction["scores"].cpu()[keep].tolist(),
                                          "labels": prediction["labels"].cpu()[keep].tolist()}})
    return frames


def main(args):
    gene_id = args.model.removeprefix("network_") if args.model != "network" else "seed"
    shared.map_reporter.validate_gene_id(gene_id)
    result_csv = SCRIPT_DIR / "results" / f"{gene_id}_results.csv"
    if not args.preflight_only:
        result_csv.unlink(missing_ok=True)  # Never reuse fitness from a failed rerun.
    started = time.time()
    try:
        if importlib.metadata.version("ultralytics") != "8.3.203":
            raise ValueError("This experiment requires ultralytics==8.3.203")
        torch.manual_seed(0)
        module, variant_path = load_variant(args.model, args.variant_dir)
        model = module.DroneDetector(num_classes=1, scale=args.scale).to(args.device)
        check = preflight(model, args.device)
        print(f"Preflight: {json.dumps(check)}", flush=True)
        if args.preflight_only:
            return
        config = {key: getattr(args, key) for key in ("train_sequences", "val_sequences", "epochs",
                  "frame_step", "img_width", "batch_size", "lr", "scale")}
        config.update(seed=0, initialization="random-no-pretrained-weights",
                      ultralytics="8.3.203", loss="v8DetectionLoss", padding="bottom-right-stride32-114",
                      optimizer="AdamW", weight_decay=5e-4,
                      loss_weights={k: getattr(model.detector.args, k) for k in ("box", "cls", "dfl")})
        config["data_profile"] = getattr(args, "data_profile", "a")
        data_options = {}
        if config["data_profile"] == "b":
            manifest = option_b.load_manifest()
            if (args.train_sequences != manifest["train_sequences"] or
                    args.val_sequences != manifest["val_sequences"] or args.frame_step != 50):
                raise ValueError("Option B must use the frozen manifest and every 50th frame")
            config["option_b_manifest_sha256"] = manifest["sha256"]
            config["archive_sources"] = manifest["archive_sources"]
            data_options = {"sequence_sources": manifest["archive_sources"], "use_archive_index": True}
        train_data = shared.load_split("train", args.train_sequences, args.frame_step, args.img_width, **data_options)
        val_data = shared.load_split("train", args.val_sequences, args.frame_step, args.img_width, **data_options)
        print(f"train frames: {len(train_data['images'])}, val frames: {len(val_data['images'])}", flush=True)
        train(model, train_data, args.epochs, args.batch_size, args.lr, args.device)
        checkpoint = SCRIPT_DIR / "trained_models" / f"{gene_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "model_sha256": shared.sha256_file(variant_path)}, checkpoint)
        frames = predict_frames(model, val_data, args.batch_size, args.device)
        metadata = shared.report_metadata(variant_path, checkpoint, config)
        if config["data_profile"] == "b":
            metadata["manifest_identity"] = config["option_b_manifest_sha256"]
            metadata["split_identity"] = manifest["profile"] + ":" + manifest["sha256"]
        metadata.update(postprocessing=f"Ultralytics NMS conf={shared.EVAL_CONF} iou=0.5 max_det=100 timeout=disabled",
                        experiment="option2-ultralytics-scratch-engineering", training_config=config,
                        trainer_sha256=shared.sha256_file(Path(__file__)),
                        seed_sha256=shared.sha256_file(SEED_FILE), torch_version=torch.__version__)
        result, report = shared.map_reporter.report(
            {"metadata": metadata, "frames": frames, "params": check["params"],
             "expected_sample_ids": val_data["sample_ids"]},
            sota_root=SCRIPT_DIR, gene_id=gene_id,
        )
        print(f"mAP50={result.map50:.6f}, mAP50:95={result.map50_95:.6f}, params={result.params}")
        print(f"Saved report: {report}; elapsed {time.time() - started:.1f}s")
        (SCRIPT_DIR / "results" / f"{gene_id}_failure.json").unlink(missing_ok=True)
        print("job done", flush=True)
    except Exception:
        # Preserve a traceback so run_improved marks this individual invalid.
        failure = SCRIPT_DIR / "results" / f"{gene_id}_failure.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps({"model": args.model, "error": traceback.format_exc()}, indent=2))
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="network")
    parser.add_argument("--variant_dir", type=Path, default=SCRIPT_DIR / "models")
    parser.add_argument("--data_profile", choices=("a", "b"), default=os.getenv("FRED_DATA_PROFILE", "a"))
    parser.add_argument("--scale", choices=list("nsmlx"), default=os.getenv("FRED_YOLO_SCALE", "m"))
    parser.add_argument("--train_sequences", default=os.getenv("FRED_TRAIN_SEQUENCES"))
    parser.add_argument("--val_sequences", default=os.getenv("FRED_VAL_SEQUENCES"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("FRED_EPOCHS", "3")))
    parser.add_argument("--frame_step", type=int, default=int(os.environ["FRED_FRAME_STEP"]) if "FRED_FRAME_STEP" in os.environ else None)
    parser.add_argument("--img_width", type=int, default=int(os.getenv("FRED_IMG_WIDTH", "640")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("FRED_BATCH", "4")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("FRED_LR", "1e-3")))
    parser.add_argument("--device", default=os.getenv("FRED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest = option_b.load_manifest() if args.data_profile == "b" else None
    args.frame_step = args.frame_step if args.frame_step is not None else (50 if manifest else shared.FRAME_STEP)
    if manifest and args.frame_step != 50:
        parser.error("Option B requires --frame_step 50")
    for key in ("train_sequences", "val_sequences"):
        cohort = manifest[key] if manifest else (shared.TRAIN_SEQUENCES if key == "train_sequences" else shared.VAL_SEQUENCES).split(",")
        raw = getattr(args, key)
        values = [s.strip() for s in raw.split(",") if s.strip()] if raw is not None else list(cohort)
        if manifest:
            if values != cohort:
                parser.error(f"Option B {key} must match the frozen manifest; remove old sequence overrides")
        elif not values or not set(values) <= set(cohort) or len(values) != len(set(values)):
            parser.error(f"{key} must be a nonempty unique subset of the team's existing cohort: {','.join(cohort)}")
        setattr(args, key, values)
    if set(args.train_sequences) & set(args.val_sequences):
        parser.error("train and val sequences must not overlap")
    if min(args.epochs, args.frame_step, args.batch_size) <= 0 or args.img_width < 32 or args.lr <= 0:
        parser.error("epochs/frame_step/batch_size/lr must be positive; img_width >= 32")
    return args


if __name__ == "__main__":
    main(parse_args())
