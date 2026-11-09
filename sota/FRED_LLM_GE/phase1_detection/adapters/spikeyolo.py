"""FRED adapter using the official Gen1 model, dataset, loss and SGD trainer."""
import json
import hashlib
import random
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch

from .base import FamilyAdapter
from ..spikeyolo_search_space import GENOME_VERSION, built_architecture, build, model_config
from ..spikeyolo_source import activate, verify_source


def training_class():
    activate()
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from spikingjelly.clock_driven.functional import reset_net

    class FredSpikeTrainer(DetectionTrainer):
        """Keep batches independent; report only the shared FRED evaluator."""
        def build_dataset(self, img_path, mode="train", batch=None):
            from ultralytics.data.dataset import YOLODataset
            from ultralytics.utils import colorstr
            cache_root = self.save_dir / "label_cache"
            cache_root.mkdir(parents=True, exist_ok=True)

            class PrivateCacheDataset(YOLODataset):
                def cache_labels(self, path):
                    # Read shared export caches if compatible, but never let
                    # the old fork rewrite YOLO11's warmed labels.cache.
                    key = hashlib.sha256("\n".join(self.im_files).encode()).hexdigest()
                    return super().cache_labels(cache_root / f"{key}.cache")

            dataset = PrivateCacheDataset(
                img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                augment=mode == "train", hyp=self.args, rect=self.args.rect,
                cache=False, single_cls=self.args.single_cls, stride=32,
                pad=0.0 if mode == "train" else 0.5, prefix=colorstr(f"{mode}: "),
                use_segments=False, use_keypoints=False, classes=self.args.classes,
                data=self.data, fraction=1.0)
            if mode == "train" and len(dataset) != self.fred_expected_train_frames:
                raise ValueError("official SpikeYOLO loader dropped training frames; inspect corrupt-image/label warnings")
            return dataset

        def preprocess_batch(self, batch):
            shape = tuple(batch["img"].shape[-2:])
            if shape != (self.args.imgsz, self.args.imgsz):
                raise ValueError(f"actual SpikeYOLO training shape {shape} differs from configured imgsz")
            self.fred_batch_shape = shape
            reset_net(self.model)
            return super().preprocess_batch(batch)

        def validate(self):
            # No architecture selection on upstream Gen1-specific metrics.
            # The final epoch is scored by candidate_runner/map_reporter.
            if not torch.isfinite(self.tloss).all():
                raise ValueError("nonfinite SpikeYOLO epoch training loss")
            return {}, -float(self.tloss.sum())

        def final_eval(self):
            # Avoid reloading upstream pickle checkpoints under Torch 2.6.
            # The adapter retains and scores final-epoch FP32 weights.
            return None

        def optimizer_step(self):
            if not torch.isfinite(self.loss).all():
                raise ValueError("nonfinite SpikeYOLO training loss")
            return super().optimizer_step()

    return FredSpikeTrainer


class SpikeYoloAdapter(FamilyAdapter):
    name = "spikeyolo"

    def __init__(self, settings, fred_dir):
        super().__init__(settings, fred_dir)
        if settings["initialization"] != "scratch" or settings["upstream"] != verify_source():
            raise ValueError("SpikeYOLO requires the pinned official scratch source")
        if settings.get("genome_version") != GENOME_VERSION:
            raise ValueError("SpikeYOLO requires genome spikeyolo-graph-v2")
        if settings["checkpoint_policy"] != "final_epoch_fp32":
            raise ValueError("unsupported SpikeYOLO checkpoint policy")

    def build(self, variant_path, num_classes, device):
        if num_classes != 1:
            raise ValueError("SpikeYOLO FRED head requires one drone class")
        # Direct candidate_runner calls get the same real graph preflight as
        # mutation/controller paths before any expensive training.
        self.architecture_hash, self.architecture_params = built_architecture(variant_path)
        self.config = model_config(variant_path)
        model = build(variant_path, device)
        if model.model[-1].__class__.__name__ != "SpikeDetect" or model.stride.tolist() != [8, 16, 32]:
            raise ValueError("SpikeYOLO detection head or strides changed")
        return model

    def train(self, model, data, proxy, device, args_path=None):
        activate()
        from spikingjelly.clock_driven.functional import reset_net
        from ultralytics.utils import DEFAULT_CFG_DICT

        random.seed(proxy["seed"])
        np.random.seed(proxy["seed"])
        torch.manual_seed(proxy["seed"])
        if not data.get("data_yaml"):
            raise ValueError("SpikeYOLO requires the verified challenging-search export")
        self.workdir = Path(args_path).parent if args_path else Path(tempfile.mkdtemp(prefix="spikeyolo_"))
        recipe = self.settings["training"]
        if recipe["optimizer"] != "SGD":
            raise ValueError("SpikeYOLO recipe must freeze optimizer=SGD")
        unknown = set(recipe) - set(DEFAULT_CFG_DICT)
        if unknown:
            raise ValueError(f"unsupported official SpikeYOLO training options: {unknown}")
        config_path = self.workdir / "snn_yolov8s.yaml"
        self.workdir.mkdir(parents=True, exist_ok=True)
        import yaml
        config_path.write_text(yaml.safe_dump(self.config, sort_keys=False))
        trainer = training_class()(overrides=dict(
            recipe, model=str(config_path), data=str(data["data_yaml"]),
            epochs=proxy["epochs"], imgsz=proxy["img_width"], batch=proxy["batch_size"],
            seed=proxy["seed"], deterministic=True, pretrained=False,
            device=str(device).replace("cuda:", "") if str(device) != "cuda" else "0",
            workers=self.settings["workers"], project=str(self.workdir),
            name="official_train", exist_ok=False, val=False, plots=False,
            patience=0, verbose=False))
        # setup_model recognizes a pre-built official module; no second random
        # initialization and no automatic checkpoint download/load.
        trainer.model = model
        trainer.fred_expected_train_frames = data["frames"]
        # Baselines never start external experiment-tracking integrations.
        from ultralytics.utils import callbacks
        trainer.callbacks = callbacks.get_default_callbacks()
        trainer.train()
        reset_net(model)
        if trainer.epoch + 1 != proxy["epochs"]:
            raise ValueError("SpikeYOLO did not complete its full frozen epoch budget")
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite SpikeYOLO final weights")
        if args_path:
            actual = Path(trainer.save_dir) / "args.yaml"
            if not actual.is_file():
                raise FileNotFoundError(actual)
            (self.workdir / "official_args.yaml").write_bytes(actual.read_bytes())
            runtime = dict(device=str(device), python=platform.python_version(),
                           torch=torch.__version__, cuda=torch.version.cuda,
                           actual_batch_shape=list(trainer.fred_batch_shape),
                           actual_train_frames=len(trainer.train_loader.dataset),
                           gpu=(torch.cuda.get_device_name(torch.device(device))
                                if torch.device(device).type == "cuda" else None))
            (self.workdir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
            (self.workdir / "model.json").write_text(json.dumps(self.config, indent=2) + "\n")
        self.trainer = trainer

    @torch.no_grad()
    def predict(self, model, images, proxy, device):
        activate()
        from spikingjelly.clock_driven.functional import reset_net
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.ops import non_max_suppression, scale_boxes

        model.eval()
        letterbox = LetterBox(proxy["img_width"], auto=False, stride=32)
        results = []
        # Bound inference memory and reset neuron state for every independent frame.
        for image in images:
            rgb = image.permute(1, 2, 0).numpy()
            pixels = letterbox(image=rgb)
            inputs = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
            if inputs.shape[-2:] != (proxy["img_width"], proxy["img_width"]):
                raise ValueError("actual SpikeYOLO inference resolution differs from protocol")
            inputs = inputs[None].to(device=device, dtype=torch.float32) / 255
            reset_net(model)
            prediction = model(inputs)
            detections = non_max_suppression(prediction, proxy["eval_conf"], proxy["nms_iou"],
                                            max_det=self.settings["max_detections"])[0]
            boxes = scale_boxes(inputs.shape[2:], detections[:, :4], rgb.shape[:2])
            results.append((boxes.cpu(), detections[:, 4].cpu()))
            reset_net(model)
        return results

    def postprocessing(self, proxy):
        return (f'official Gen1 SpikeDetect; conf={proxy["eval_conf"]}; '
                f'NMS IoU={proxy["nms_iou"]}; max={self.settings["max_detections"]}; '
                'RGB released frame, MS_GetT T=1, reset per sample, original-pixel xyxy')

    def save(self, model, path):
        from spikingjelly.clock_driven.functional import reset_net
        reset_net(model)
        torch.save(model.state_dict(), path)
