"""D-FINE-inspired Event seed adapter for the shared FRED candidate runner.

The seed owns its architecture and fixed loss. This adapter translates the
runner's uint8 frames and pixel xyxy boxes, trains a standalone baseline or
candidate, and applies the shared proxy confidence/NMS settings at inference.
"""

import ast
import importlib.util
import re
from pathlib import Path

import torch

try:
    from .base import FamilyAdapter
except ImportError:
    from base import FamilyAdapter


_COMPONENTS = ("EventBackbone", "MultiScaleEncoder", "DistributionRefinementDecoder")
_MARKER = "# --OPTION--"


def _source_parts(path):
    source = Path(path).read_text()
    if source.count(_MARKER) != 3:
        raise ValueError("D-FINE candidates require exactly three option markers")
    parts = re.split(r"^# --OPTION--[ \t]*$", source, flags=re.MULTILINE)
    if len(parts) != 4 or not all(part.strip() for part in parts):
        raise ValueError("D-FINE candidate has an invalid marker layout")
    ast.parse(source, filename=str(path))
    for component, part in zip(_COMPONENTS, parts[1:]):
        body = ast.parse(part, filename=str(path)).body
        if len(body) != 1 or not isinstance(body[0], ast.ClassDef) or body[0].name != component:
            raise ValueError(f"D-FINE section must contain only class {component}")
    return parts


def _targets_from_pixels(image_boxes, width, height, device):
    """Convert already approved pixel xyxy boxes; never repair annotations."""
    if width <= 0 or height <= 0:
        raise ValueError("D-FINE target frame dimensions must be positive")
    boxes = torch.as_tensor(image_boxes, dtype=torch.float32, device=device)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not torch.isfinite(boxes).all():
        raise ValueError("D-FINE target boxes must be finite [N,4] pixel xyxy")
    if boxes.numel():
        if ((boxes[:, :2] < 0).any() or (boxes[:, 2] > width).any()
                or (boxes[:, 3] > height).any()
                or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("D-FINE target contains an invalid or out-of-bounds box")
    scale = boxes.new_tensor((width, height, width, height))
    normalized = boxes / scale
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


class DFINEAdapter(FamilyAdapter):
    name = "dfine"

    def build(self, variant_path, num_classes, device):
        if num_classes != 1:
            raise ValueError("the FRED D-FINE seed supports only the drone class")
        seed_path = self.variant_path("network", None)
        seed_parts = _source_parts(seed_path)
        candidate_parts = _source_parts(variant_path)
        if candidate_parts[0] != seed_parts[0]:
            raise ValueError("D-FINE candidate changed the protected preamble")
        spec = importlib.util.spec_from_file_location(f"fred_dfine_{Path(variant_path).stem}", variant_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import D-FINE candidate: {variant_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.DFINEEventDetector(**self.settings.get("model", {})).to(device)
        criterion = module.DFINECriterion(reg_max=model.reg_max).to(device)
        # Reject broken candidate interfaces before streaming or training data.
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, 3, 65, 97), device=device))
        model.train()
        self.criterion = criterion
        return model

    def train(self, model, data, proxy, device):
        images, boxes = data.get("images"), data.get("boxes")
        if images is None or not hasattr(images, "shape") or len(images.shape) != 4 or images.shape[1] != 3:
            raise ValueError("D-FINE training requires [N,3,H,W] Event images")
        if not isinstance(boxes, list) or len(boxes) != len(images):
            raise ValueError("D-FINE training requires one box tensor per frame")
        if isinstance(images, torch.Tensor) and images.dtype != torch.uint8:
            raise ValueError("D-FINE training images must be uint8")
        if len(images) == 0 or min(images.shape[-2:]) < 32:
            raise ValueError("D-FINE training needs nonempty frames of at least 32x32")
        if not hasattr(self, "criterion"):
            raise RuntimeError("build the D-FINE adapter before training")
        epochs, batch_size = proxy["epochs"], proxy["batch_size"]
        microbatch = self.settings.get("microbatch_size", 1)
        if min(epochs, batch_size, microbatch) <= 0:
            raise ValueError("epochs, batch size, and microbatch size must be positive")
        width, height = images.shape[-1], images.shape[-2]
        # Validate all cached targets before the first optimizer step.
        for image_boxes in boxes:
            _targets_from_pixels(image_boxes, width, height, "cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.settings["lr"],
            weight_decay=self.settings.get("weight_decay", 1e-4),
        )
        generator = torch.Generator().manual_seed(proxy["seed"])
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(images), generator=generator)
            total = 0.0
            for start in range(0, len(order), batch_size):
                group = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for offset in range(0, len(group), microbatch):
                    selected = group[offset:offset + microbatch]
                    raw = images[selected]
                    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.uint8:
                        raise ValueError("D-FINE training images must decode to uint8 tensors")
                    event = raw.to(device=device, dtype=torch.float32).div_(255.0)
                    targets = [_targets_from_pixels(boxes[int(i)], width, height, device)
                               for i in selected]
                    losses = self.criterion(model(event), targets)
                    weight = len(selected) / len(group)
                    (losses["loss_total"] * weight).backward()
                    batch_loss += float(losses["loss_total"].detach()) * weight
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.settings.get("grad_clip", 0.1),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                total += batch_loss * len(group)
            print(f"epoch {epoch + 1}/{epochs}: D-FINE loss {total / len(images):.4f}", flush=True)

    def predict(self, model, images, proxy, device):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("D-FINE inference requires [B,3,H,W] Event tensors")
        if images.dtype != torch.uint8:
            raise ValueError("D-FINE inference images must be uint8")
        threshold, nms_iou = proxy["eval_conf"], proxy["nms_iou"]
        if not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1:
            raise ValueError("D-FINE proxy confidence and NMS IoU must be in [0,1]")
        model.eval()
        model.confidence_threshold = threshold
        output = []
        microbatch = self.settings.get("microbatch_size", 1)
        if microbatch <= 0:
            raise ValueError("D-FINE microbatch size must be positive")
        height, width = images.shape[-2:]
        with torch.no_grad():
            for start in range(0, len(images), microbatch):
                event = images[start:start + microbatch].to(device=device, dtype=torch.float32).div_(255.0)
                for record in model.predict(event):
                    boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
                    scores = torch.tensor(record["scores"], dtype=torch.float32)
                    labels = record["labels"]
                    if (len(boxes) != len(scores) or len(boxes) != len(labels)
                            or any(label != 0 for label in labels)
                            or not torch.isfinite(boxes).all()
                            or not torch.isfinite(scores).all()
                            or (scores < 0).any() or (scores > 1).any()
                            or (boxes[:, :2] < 0).any()
                            or (boxes[:, 2] > width).any() or (boxes[:, 3] > height).any()
                            or (boxes[:, 2:] <= boxes[:, :2]).any()):
                        raise ValueError("D-FINE prediction contains invalid boxes or scores")
                    selected = _nms(boxes, scores, nms_iou)
                    output.append((boxes[selected], scores[selected]))
        return output

    def postprocessing(self, proxy):
        return f"D-FINE conf={proxy['eval_conf']} nms_iou={proxy['nms_iou']}"
