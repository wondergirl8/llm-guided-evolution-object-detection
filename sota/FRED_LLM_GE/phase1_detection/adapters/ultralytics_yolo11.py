"""Family adapter for Ultralytics YOLO11 candidates (seeds/seed_yolo11n.yaml).

A candidate is an Ultralytics model YAML (LLM-GE names variants
network_<id>.py, so the file is read as YAML whatever its extension). It is
built with ultralytics.YOLO and trained from scratch with the Ultralytics
trainer, so the loss, target assignment and augmentation are Ultralytics'
own. Inference goes through yolo11.yolo11_prediction, so scoring matches the
other families.

The trainer reads images from disk, so the protocol's cached training frames
are exported once to an Ultralytics dataset (PNG images plus YOLO-format
labels) under the protocol cache directory and shared by every candidate.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
try:
    from ..yolo11_search_space import load_and_validate, validate_built_model
    from .base import FamilyAdapter
    from .yolo11 import yolo11_prediction
except ImportError:
    from yolo11_search_space import load_and_validate, validate_built_model
    from base import FamilyAdapter
    from yolo11 import yolo11_prediction

# The trainer always validates after the final epoch; candidates are scored
# separately on the protocol's val frames, so it only gets this many training
# frames to keep that pass short.
TRAINER_VAL_FRAMES = 64


def _write_split(data, indices, root):
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    height, width = data["images"].shape[2:]
    for i in indices:
        name = f"{i:06d}"
        # Cached frames are RGB; PNG keeps the event frames lossless.
        Image.fromarray(data["images"][i].permute(1, 2, 0).numpy()).save(
            image_dir / f"{name}.png", compress_level=1)
        lines = []
        for x1, y1, x2, y2 in data["boxes"][i].tolist():
            lines.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        (label_dir / f"{name}.txt").write_text("\n".join(lines))


def export_dataset(data, proxy, cache_dir):
    """Write the training frames as an Ultralytics dataset once; return its data YAML.

    Built in a private temp directory and renamed into place, so concurrent
    candidates never see a half-written dataset.
    """
    key = {k: proxy[k] for k in ("hf_split", "train_sequences", "frame_step", "img_width")}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    final = Path(cache_dir) / "ultralytics" / f"train_{digest}"
    if not (final / "data.yaml").is_file():
        final.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f"{final.name}.", dir=final.parent))
        try:
            _write_split(data, range(len(data["images"])), temp / "train")
            _write_split(data, range(min(TRAINER_VAL_FRAMES, len(data["images"]))), temp / "val")
            (temp / "export.json").write_text(json.dumps(key, sort_keys=True))
            # Paths are relative to the dataset root so the folder can be moved.
            (temp / "data.yaml").write_text(yaml.safe_dump(
                {"path": str(final), "train": "train/images", "val": "val/images",
                 "names": {0: "drone"}}, sort_keys=False))
            os.rename(temp, final)
        except OSError:
            # Another candidate finished the export first; use theirs.
            if not (final / "data.yaml").is_file():
                raise
        finally:
            shutil.rmtree(temp, ignore_errors=True)
    return final / "data.yaml"


def _ultralytics_device(device):
    device = str(device)
    if device == "cuda":
        return 0
    return device.split(":", 1)[1] if device.startswith("cuda:") else device


class UltralyticsYolo11Adapter(FamilyAdapter):
    name = "yolo11"

    def training_options(self):
        """Return the frozen optimizer, schedule and augmentation overrides."""
        training = self.settings["training"]
        return {**{key: value for key, value in training.items() if key != "augmentation"},
                **training["augmentation"]}

    def build(self, variant_path, num_classes, device):
        from ultralytics import YOLO

        config = load_and_validate(variant_path)
        if num_classes != config["nc"]:
            raise ValueError("candidate class count differs from the frozen YOLO11 seed")
        self.workdir = Path(tempfile.mkdtemp(prefix="yolo11_candidate_"))
        # Ultralytics takes the model scale from the file name, not the YAML.
        model_yaml = self.workdir / f"yolo11{config.get('scale', 'n')}.yaml"
        model_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
        # Built from YAML: random initialization, no pretrained weights.
        model = YOLO(str(model_yaml), task="detect")
        self.architecture_hash, self.architecture_params = validate_built_model(model)
        return model

    def train(self, model, data, proxy, device, args_path=None):
        # Disk datasets (export_split.py) come as {"data_yaml": ...}; in-memory
        # proxy frames are exported once on first use.
        data_yaml = data.get("data_yaml") or export_dataset(
            data, proxy, self.fred_dir / proxy["cache_dir"])
        trained = False
        try:
            model.train(
                data=str(data_yaml),
                epochs=proxy["epochs"],
                imgsz=proxy["img_width"],
                batch=proxy["batch_size"],
                seed=proxy["seed"],
                deterministic=True,
                pretrained=False,
                device=_ultralytics_device(device),
                workers=self.settings.get("workers", 8),
                project=str(self.workdir),
                name="train",
                exist_ok=True,
                val=False,
                plots=False,
                verbose=False,
                **self.training_options(),
            )
            trained = True
        finally:
            if args_path is not None:
                trainer_args = self.workdir / "train" / "args.yaml"
                if trainer_args.is_file():
                    args_path = Path(args_path)
                    args_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(trainer_args, args_path)
                elif trained:
                    raise FileNotFoundError(f"Ultralytics did not write {trainer_args}")

    def predict(self, model, images, proxy, device):
        # Cached frames are RGB; Ultralytics treats numpy input as BGR.
        batch = [np.ascontiguousarray(image.permute(1, 2, 0).numpy()[..., ::-1])
                 for image in images]
        results = model.predict(batch, imgsz=proxy["img_width"], conf=proxy["eval_conf"],
                                iou=proxy["nms_iou"], device=_ultralytics_device(device),
                                verbose=False)
        out = []
        for result in results:
            prediction = yolo11_prediction(result)
            out.append((torch.tensor(prediction["boxes"], dtype=torch.float32).reshape(-1, 4),
                        torch.tensor(prediction["scores"], dtype=torch.float32)))
        return out

    def postprocessing(self, proxy):
        return (f"ultralytics YOLO.predict imgsz={proxy['img_width']} conf={proxy['eval_conf']} "
                f"nms_iou={proxy['nms_iou']}")

    def param_count(self, model):
        return sum(p.numel() for p in model.model.parameters())

    def save(self, model, path):
        model.save(path)
