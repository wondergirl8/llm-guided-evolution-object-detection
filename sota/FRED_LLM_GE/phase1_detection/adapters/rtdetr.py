"""Original RT-DETR (v1) Event seed adapter for the shared FRED candidate runner.

The seed owns its architecture, matcher and loss. This adapter resizes the
runner's uint8 frames to the protocol image width, zero-pads them to a multiple
of 32 (the encoder upsamples by exactly 2x), converts pixel xyxy boxes to the
padded frame's normalized cxcywh, trains a standalone baseline or candidate, and
maps predictions back to the given images' pixels before the shared confidence
and NMS settings.
"""

import ast
import importlib.util
import re
from pathlib import Path

import torch
from torch.nn import functional as F

try:
    from .base import FamilyAdapter
except ImportError:
    from base import FamilyAdapter


_COMPONENTS = ("PResNet", "HybridEncoder", "RTDETRTransformer")
_MARKER = "# --OPTION--"
_STRIDE = 32


def _source_parts(path):
    source = Path(path).read_text()
    if source.count(_MARKER) != 3:
        raise ValueError("RT-DETR candidates require exactly three option markers")
    parts = re.split(r"^# --OPTION--[ \t]*$", source, flags=re.MULTILINE)
    if len(parts) != 4 or not all(part.strip() for part in parts):
        raise ValueError("RT-DETR candidate has an invalid marker layout")
    ast.parse(source, filename=str(path))
    for component, part in zip(_COMPONENTS, parts[1:]):
        body = ast.parse(part, filename=str(path)).body
        if len(body) != 1 or not isinstance(body[0], ast.ClassDef) or body[0].name != component:
            raise ValueError(f"RT-DETR section must contain only class {component}")
    return parts


class _FrameGroups(torch.utils.data.Dataset):
    """Each item is one optimizer step's frames, decoded in a DataLoader worker."""

    def __init__(self, images, groups):
        self.images, self.groups = images, groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        return group, self.images[group]


def _input_geometry(height, width, img_width):
    """Resized (h, w) at the protocol width and the zero-padded model input size."""
    if min(height, width, img_width) <= 0:
        raise ValueError("frame and protocol sizes must be positive")
    resized_w = img_width
    resized_h = max(1, round(height * img_width / width))
    padded = tuple(-(-size // _STRIDE) * _STRIDE for size in (resized_h, resized_w))
    return (resized_h, resized_w), padded


def _prepare(raw, img_width, device):
    """uint8 [B,3,H,W] frames -> [0,1] model input, resized and right/bottom zero-padded."""
    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.uint8 or raw.ndim != 4 or raw.shape[1] != 3:
        raise ValueError("RT-DETR images must be uint8 [B,3,H,W] Event tensors")
    (resized_h, resized_w), (padded_h, padded_w) = _input_geometry(*raw.shape[-2:], img_width)
    event = raw.to(device=device, non_blocking=True).float().div_(255.0)
    if (resized_h, resized_w) != tuple(event.shape[-2:]):
        event = F.interpolate(event, size=(resized_h, resized_w), mode="bilinear",
                              align_corners=False, antialias=True).clamp_(0.0, 1.0)
    return F.pad(event, (0, padded_w - resized_w, 0, padded_h - resized_h))


def _targets_from_pixels(image_boxes, width, height, img_width, device):
    """Approved pixel xyxy boxes -> normalized cxcywh in the padded model input.

    Validates without repairing annotations: out-of-bounds boxes are rejected.
    """
    if width <= 0 or height <= 0:
        raise ValueError("RT-DETR target frame dimensions must be positive")
    boxes = torch.as_tensor(image_boxes, dtype=torch.float32, device=device)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not torch.isfinite(boxes).all():
        raise ValueError("RT-DETR target boxes must be finite [N,4] pixel xyxy")
    if boxes.numel():
        if ((boxes[:, :2] < 0).any() or (boxes[:, 2] > width).any()
                or (boxes[:, 3] > height).any()
                or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("RT-DETR target contains an invalid or out-of-bounds box")
    (resized_h, resized_w), (padded_h, padded_w) = _input_geometry(height, width, img_width)
    scale = boxes.new_tensor((resized_w / width / padded_w, resized_h / height / padded_h) * 2)
    normalized = boxes * scale
    center = (normalized[:, :2] + normalized[:, 2:]) / 2
    size = normalized[:, 2:] - normalized[:, :2]
    return {"boxes": torch.cat((center, size), dim=-1),
            "labels": torch.zeros(len(boxes), dtype=torch.long, device=device)}


def _nms(boxes, scores, threshold):
    """Greedy single-class NMS for the runner's provisional proxy protocol."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remainder = order[1:]
        low = torch.maximum(boxes[current, :2], boxes[remainder, :2])
        high = torch.minimum(boxes[current, 2:], boxes[remainder, 2:])
        intersection = (high - low).clamp_min(0).prod(-1)
        area_current = (boxes[current, 2:] - boxes[current, :2]).prod()
        area_other = (boxes[remainder, 2:] - boxes[remainder, :2]).prod(-1)
        iou = intersection / (area_current + area_other - intersection).clamp_min(1e-8)
        order = remainder[iou <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


class RTDETRAdapter(FamilyAdapter):
    name = "rtdetr"

    def build(self, variant_path, num_classes, device):
        if num_classes != 1:
            raise ValueError("the FRED RT-DETR seed supports only the drone class")
        seed_path = self.variant_path("network", None)
        seed_parts = _source_parts(seed_path)
        candidate_parts = _source_parts(variant_path)
        if candidate_parts[0] != seed_parts[0]:
            raise ValueError("RT-DETR candidate changed the protected preamble")
        spec = importlib.util.spec_from_file_location(f"fred_rtdetr_{Path(variant_path).stem}", variant_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import RT-DETR candidate: {variant_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.RTDETREventDetector(**self.settings.get("model", {})).to(device)
        criterion = module.RTDETRCriterion().to(device)
        # Reject broken candidate interfaces before streaming or training data.
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, 3, 320, 320), device=device))
        model.train()
        self.criterion = criterion
        return model

    def train(self, model, data, proxy, device):
        images, boxes = data.get("images"), data.get("boxes")
        if images is None or not hasattr(images, "shape") or len(images.shape) != 4 or images.shape[1] != 3:
            raise ValueError("RT-DETR training requires [N,3,H,W] Event images")
        if not isinstance(boxes, list) or len(boxes) != len(images):
            raise ValueError("RT-DETR training requires one box tensor per frame")
        if isinstance(images, torch.Tensor) and images.dtype != torch.uint8:
            raise ValueError("RT-DETR training images must be uint8")
        if len(images) == 0:
            raise ValueError("RT-DETR training needs nonempty frames")
        if not hasattr(self, "criterion"):
            raise RuntimeError("build the RT-DETR adapter before training")
        epochs, batch_size, img_width = proxy["epochs"], proxy["batch_size"], proxy["img_width"]
        microbatch = self.settings.get("microbatch_size", 1)
        if min(epochs, batch_size, microbatch) <= 0:
            raise ValueError("epochs, batch size, and microbatch size must be positive")
        width, height = images.shape[-1], images.shape[-2]
        # Validate all cached targets before the first optimizer step.
        for image_boxes in boxes:
            _targets_from_pixels(image_boxes, width, height, img_width, "cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.settings["lr"],
            weight_decay=self.settings.get("weight_decay", 1e-4),
        )
        generator = torch.Generator().manual_seed(proxy["seed"])
        workers = self.settings.get("workers", 0)
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(images), generator=generator)
            groups = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
            # Workers decode upcoming PNG frames while the GPU trains; the
            # frame order and optimizer steps are the same as without them.
            loader = torch.utils.data.DataLoader(
                _FrameGroups(images, groups), batch_size=None, num_workers=workers,
                pin_memory=torch.device(device).type == "cuda",
            )
            total = 0.0
            for group, group_images in loader:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for offset in range(0, len(group), microbatch):
                    selected = group[offset:offset + microbatch]
                    event = _prepare(group_images[offset:offset + microbatch], img_width, device)
                    targets = [_targets_from_pixels(boxes[int(i)], width, height, img_width, device)
                               for i in selected]
                    losses = self.criterion(model(event, targets), targets)
                    weight = len(selected) / len(group)
                    (losses["loss_total"] * weight).backward()
                    # Kept on the GPU so each microbatch does not wait for a sync.
                    batch_loss = batch_loss + losses["loss_total"].detach() * weight
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.settings.get("grad_clip", 0.1),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                total += batch_loss * len(group)
            print(f"epoch {epoch + 1}/{epochs}: RT-DETR loss {float(total) / len(images):.4f}", flush=True)

    def predict(self, model, images, proxy, device):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("RT-DETR inference requires [B,3,H,W] Event tensors")
        if images.dtype != torch.uint8:
            raise ValueError("RT-DETR inference images must be uint8")
        threshold, nms_iou = proxy["eval_conf"], proxy["nms_iou"]
        if not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1:
            raise ValueError("RT-DETR proxy confidence and NMS IoU must be in [0,1]")
        model.eval()
        model.confidence_threshold = threshold
        microbatch = self.settings.get("microbatch_size", 1)
        if microbatch <= 0:
            raise ValueError("RT-DETR microbatch size must be positive")
        height, width = images.shape[-2:]
        (resized_h, resized_w), _ = _input_geometry(height, width, proxy["img_width"])
        to_image = torch.tensor((width / resized_w, height / resized_h) * 2)
        bounds = torch.tensor((width, height, width, height), dtype=torch.float32)
        output = []
        with torch.no_grad():
            for start in range(0, len(images), microbatch):
                event = _prepare(images[start:start + microbatch], proxy["img_width"], device)
                for record in model.predict(event):
                    boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
                    scores = torch.tensor(record["scores"], dtype=torch.float32)
                    labels = record["labels"]
                    if (len(boxes) != len(scores) or len(boxes) != len(labels)
                            or any(label != 0 for label in labels)
                            or not torch.isfinite(boxes).all()
                            or not torch.isfinite(scores).all()
                            or (scores < 0).any() or (scores > 1).any()):
                        raise ValueError("RT-DETR prediction contains invalid boxes or scores")
                    # Boxes may reach into the zero padding; keep the part inside the frame.
                    boxes = torch.minimum((boxes * to_image).clamp_min(0), bounds)
                    keep = (boxes[:, 2:] > boxes[:, :2]).all(-1)
                    boxes, scores = boxes[keep], scores[keep]
                    selected = _nms(boxes, scores, nms_iou)
                    output.append((boxes[selected], scores[selected]))
        return output

    def postprocessing(self, proxy):
        return (f"RT-DETR resize_width={proxy['img_width']} pad_to={_STRIDE} "
                f"conf={proxy['eval_conf']} nms_iou={proxy['nms_iou']}")
