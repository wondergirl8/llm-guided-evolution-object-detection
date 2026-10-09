"""Tutorial-sized YOLO11-inspired seed for the FRED RGB-event MVP.

This is deliberately a small, self-contained detector rather than a faithful
reproduction of every production YOLO11 block.  The purpose of this seed is to
give LLM-GE a valid genome that can be imported, trained, mutated, and scored.

Tensor convention used throughout the file:
    image tensor: [batch, channels, height, width]
    raw head output: [batch, 5 + num_classes, grid_height, grid_width]

The two pairs of option-marker lines delimit the only mutable regions.
They are at top-level indentation on purpose: the evolutionary code can
replace each complete class without accidentally breaking a method or loop.
"""

from typing import Dict, List

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Small protected building block: convolution, normalization, and SiLU.

    A 3x3 convolution learns local spatial patterns.  BatchNorm stabilizes the
    scale of activations, and SiLU (x * sigmoid(x)) gives a smooth nonlinearity
    that is commonly used in YOLO-family networks.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# --OPTION--
class SensorFusionBackbone(nn.Module):
    """Merge RGB and event data, then extract a high-resolution feature map.

    RGB and event frames each have three channels.  Concatenating them along
    the channel axis creates [B, 6, H, W].  The first convolution learns how
    color, brightness, motion, and event polarity should interact; this is
    early feature fusion rather than merely adding the two modalities.

    Two stride-2 convolutions reduce H,W to H/4,W/4.  Keeping this output much
    larger than a classification feature map matters for tiny drones: a small
    object can disappear if it is downsampled too aggressively.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 64):
        super().__init__()
        if in_channels != 6:
            raise ValueError("FRED fusion expects exactly 6 input channels")
        self.features = nn.Sequential(
            ConvBlock(6, 32, stride=2),
            ConvBlock(32, 48, stride=2),
            ConvBlock(48, out_channels, stride=1),
        )

    def forward(self, rgb: Tensor, event: Tensor) -> Tensor:
        if rgb.ndim != 4 or event.ndim != 4:
            raise ValueError("rgb and event must both have shape [B, 3, H, W]")
        if rgb.shape[1] != 3 or event.shape[1] != 3:
            raise ValueError("rgb and event must each contain exactly 3 channels")
        if rgb.shape[0] != event.shape[0] or rgb.shape[2:] != event.shape[2:]:
            raise ValueError("rgb and event must have matching batch and spatial dimensions")
        fused = torch.cat((rgb, event), dim=1)  # [B, 6, H, W]
        return self.features(fused)  # [B, 64, H/4, W/4]


# --OPTION--


# --OPTION--
class ContextNeck(nn.Module):
    """SPPF-style context aggregator that preserves the feature-map size.

    A 5x5 max-pool with stride 1 sees a local neighborhood without changing
    the [H/4, W/4] grid.  Applying it three times gives progressively larger
    effective receptive fields.  Concatenating the original map and all three
    pooled maps lets the head use both sharp local evidence and wider context.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.reduce = ConvBlock(channels, hidden, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.fuse = ConvBlock(hidden * 4, channels, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.fuse(torch.cat((x, p1, p2, p3), dim=1))


# --OPTION--
class DroneDetector(nn.Module):
    """Protected model wrapper shared by the FRED seed-model pool.

    ``forward`` intentionally returns raw logits so a training loop can apply
    a task-specific YOLO loss.  ``predict`` is the inference convenience API:
    it converts logits into pixel-coordinate boxes and performs lightweight
    class-aware NMS without requiring torchvision.
    """

    def __init__(self, num_classes: int = 1, confidence_threshold: float = 0.25):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.backbone = SensorFusionBackbone()
        self.neck = ContextNeck(channels=64)
        self.head = nn.Sequential(
            ConvBlock(64, 64, stride=1),
            nn.Conv2d(64, 5 + num_classes, kernel_size=1),
        )
        self.output_stride = 4

    def forward(self, rgb: Tensor, event: Tensor) -> Tensor:
        features = self.backbone(rgb, event)
        context = self.neck(features)
        return self.head(context)

    @staticmethod
    def _nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.50) -> Tensor:
        """Return indices kept by simple greedy IoU suppression."""
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break
            candidate = order[1:]
            chosen = boxes[current]
            others = boxes[candidate]
            x1 = torch.maximum(chosen[0], others[:, 0])
            y1 = torch.maximum(chosen[1], others[:, 1])
            x2 = torch.minimum(chosen[2], others[:, 2])
            y2 = torch.minimum(chosen[3], others[:, 3])
            intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_chosen = (chosen[2] - chosen[0]).clamp(min=0) * (chosen[3] - chosen[1]).clamp(min=0)
            area_others = (others[:, 2] - others[:, 0]).clamp(min=0) * (others[:, 3] - others[:, 1]).clamp(min=0)
            iou = intersection / (area_chosen + area_others - intersection + 1e-6)
            order = candidate[iou <= iou_threshold]
        return torch.stack(keep) if keep else boxes.new_empty((0,), dtype=torch.long)

    def predict(self, rgb: Tensor, event: Tensor) -> List[Dict[str, Tensor]]:
        """Decode one dense grid prediction per image into detection dictionaries.

        At each grid cell, the head emits (tx, ty, tw, th, objectness, class
        logits).  Sigmoid maps tx,ty to offsets inside the cell; exp maps width
        and height to positive pixel sizes.  The cell coordinates and the
        output stride then convert those values to image-space xyxy boxes.
        """
        logits = self.forward(rgb, event)
        batch, _, grid_h, grid_w = logits.shape
        raw = logits.view(batch, 5 + self.num_classes, grid_h, grid_w).permute(0, 2, 3, 1)
        device = logits.device
        yy, xx = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing="ij",
        )
        grid_x, grid_y = xx.float(), yy.float()
        results: List[Dict[str, Tensor]] = []
        for image_raw in raw:
            centers = torch.stack(
                ((image_raw[..., 0].sigmoid() + grid_x) * self.output_stride,
                 (image_raw[..., 1].sigmoid() + grid_y) * self.output_stride),
                dim=-1,
            )
            sizes = image_raw[..., 2:4].clamp(-4.0, 4.0).exp() * self.output_stride
            objectness = image_raw[..., 4].sigmoid()
            class_scores = image_raw[..., 5:].sigmoid()
            scores, labels = (objectness.unsqueeze(-1) * class_scores).max(dim=-1)
            boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
            mask = scores >= self.confidence_threshold
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
            if boxes.numel():
                keep = []
                for class_id in labels.unique():
                    class_indices = torch.where(labels == class_id)[0]
                    keep.append(class_indices[self._nms(boxes[class_indices], scores[class_indices])])
                keep = torch.cat(keep) if keep else labels.new_empty((0,))
                keep = keep[scores[keep].argsort(descending=True)]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results


if __name__ == "__main__":
    # A tiny smoke test for local development; the evolutionary runner imports
    # DroneDetector and does not execute this demonstration block.
    model = DroneDetector(num_classes=1)
    rgb = torch.randn(2, 3, 128, 128)
    event = torch.randn(2, 3, 128, 128)
    print(model(rgb, event).shape)
    print({key: value.shape for key, value in model.predict(rgb, event)[0].items()})
