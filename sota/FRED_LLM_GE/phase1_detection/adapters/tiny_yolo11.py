"""Family adapter for the tutorial-sized YOLO11-inspired seed (seeds/seed_yolo11.py).

The seed and its LLM-generated variants define DroneDetector(num_classes),
whose forward(event) returns raw [B, 5 + num_classes, H/s, W/s] logits and
whose predict(event) returns pixel-space boxes. Training uses the single-scale
YOLO loss below, matching DroneDetector.predict's decoding. Moved unchanged
from seeds/yolo11/train_eval.py.
"""

import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .base import FamilyAdapter
except ImportError:
    from base import FamilyAdapter


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


def model_stride(model, images):
    """Stride of the model's output grid; a mutated backbone may change it."""
    model.eval()
    with torch.no_grad():
        grid = model(images[:1]).shape[2:]
    stride_y, stride_x = images.shape[2] / grid[0], images.shape[3] / grid[1]
    if abs(stride_x - stride_y) > 1e-6:
        raise ValueError(f"non-uniform output stride: {stride_x} x {stride_y}")
    return stride_x


class TinyYolo11Adapter(FamilyAdapter):
    name = "tiny_yolo11"

    def build(self, variant_path, num_classes, device):
        spec = importlib.util.spec_from_file_location(
            f"fred_yolo11_{variant_path.stem}", variant_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DroneDetector(num_classes=num_classes).to(device)

    def train(self, model, data, proxy, device):
        # One frame is enough to measure the output grid; moving the whole
        # training set to the GPU as float32 took ~60 GB at 1280 width.
        stride = model_stride(model, data["images"][:1].to(device).float() / 255)
        model.output_stride = stride  # keep predict's decoding consistent with training

        epochs, batch_size = proxy["epochs"], proxy["batch_size"]
        opt = torch.optim.AdamW(model.parameters(), lr=self.settings["lr"], weight_decay=5e-4)
        generator = torch.Generator().manual_seed(proxy["seed"])
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

    def predict(self, model, images, proxy, device):
        # DroneDetector._nms has its IoU threshold built in; refuse a protocol
        # that asks for a different one rather than silently ignoring it.
        if proxy["nms_iou"] != 0.5:
            raise ValueError("tiny_yolo11 DroneDetector.predict always uses nms_iou=0.5")
        model.eval()
        model.confidence_threshold = proxy["eval_conf"]
        with torch.no_grad():
            results = model.predict(images.to(device).float() / 255)
        return [(r["boxes"].cpu(), r["scores"].cpu()) for r in results]

    def postprocessing(self, proxy):
        return f"DroneDetector.predict conf={proxy['eval_conf']} nms_iou={proxy['nms_iou']}"
