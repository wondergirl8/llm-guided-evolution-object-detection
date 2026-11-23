"""Randomly initialized MMDetection Faster R-CNN for the shared runner."""

from types import SimpleNamespace

import torch

from .base import FamilyAdapter
from ..seeds.faster_rcnn_scratch.network import build_config, build_detector
from ..seeds.faster_rcnn_scratch.validator import validate_source
from ..seeds.faster_rcnn.train_eval import prepare_batch, train, preflight


class ResizedImages:
    """Resize only the requested batch, retaining the export on disk."""

    def __init__(self, images, width):
        self.images = images
        height = round(images.shape[-2] * width / images.shape[-1] / 4) * 4
        self.shape = (len(images), 3, height, width)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        scalar = isinstance(index, int)
        indices = torch.tensor([index]) if scalar else index
        raw = self.images[indices]
        resized = torch.nn.functional.interpolate(
            raw.float(), size=self.shape[-2:], mode="bilinear", align_corners=False,
            antialias=True).round().clamp_(0, 255).to(torch.uint8)
        return resized[0] if scalar else resized


def resized_data(data, width):
    images = ResizedImages(data["images"], width)
    old_h, old_w = data["images"].shape[-2:]
    scale = torch.tensor([images.shape[-1] / old_w, images.shape[-2] / old_h] * 2)
    return dict(data, images=images, boxes=[b * scale for b in data["boxes"]])


def preserving_preflight(model):
    """Check loss/gradients/predictions without altering training initialization."""
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    training = model.training
    import random
    import numpy as np
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            preflight(model)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        model.load_state_dict(state, strict=True)
        model.zero_grad(set_to_none=True)
        model.train(training)


class FasterRCNNScratchAdapter(FamilyAdapter):
    name = "faster_rcnn_scratch"

    def build(self, variant_path, num_classes, device):
        if num_classes != 1:
            raise ValueError("FRED Faster R-CNN supports one foreground class")
        genes = validate_source(variant_path)
        model, self.initialization = build_detector(
            build_config(genes), seed=self.settings.get("search", {}).get("seed", 0))
        model.to(device)
        preserving_preflight(model)
        self.genes = genes
        return model

    def train(self, model, data, proxy, device, *, resume_state=None, on_epoch=None):
        train(model, resized_data(data, proxy["img_width"]), SimpleNamespace(
            epochs=proxy["epochs"], batch_size=proxy["batch_size"], lr=self.settings["lr"],
            momentum=self.settings["momentum"], weight_decay=self.settings["weight_decay"]),
            resume_state=resume_state, on_epoch=on_epoch)

    def predict(self, model, images, proxy, device):
        # The shared runner rescales these input-pixel boxes to original pixels.
        data = {"images": images, "boxes": [torch.empty(0, 4) for _ in images],
                "orig_size": (images.shape[-1], images.shape[-2]),
                "sample_ids": [str(i) for i in range(len(images))]}
        data = resized_data(data, proxy["img_width"])
        model.roi_head.test_cfg.score_thr = proxy["eval_conf"]
        model.roi_head.test_cfg.nms.iou_threshold = proxy["nms_iou"]
        model.roi_head.test_cfg.max_per_img = 100
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(model, data, list(range(len(images))), False)
            predictions = model(**batch, mode="predict")
        if len(predictions) != len(images):
            raise ValueError("Faster R-CNN prediction count differs from batch")
        return [(p.pred_instances.bboxes.detach().cpu(),
                 p.pred_instances.scores.detach().cpu()) for p in predictions]

    def save(self, model, path):
        torch.save({"state_dict": model.state_dict(), "initialization": self.initialization,
                    "genes": self.genes, "settings": self.settings}, path)

    def postprocessing(self, proxy):
        return (f"MMDetection rescale=True conf={proxy['eval_conf']} "
                f"nms_iou={proxy['nms_iou']} max_per_img=100")
