"""D-FINE-inspired Event seed adapter for the shared FRED candidate runner.

The seed owns its architecture and fixed loss. This adapter translates the
runner's uint8 frames and pixel xyxy boxes, trains a standalone baseline or
candidate, and applies the shared proxy confidence/NMS settings at inference.
"""

import ast
import hashlib
import importlib.util
import re
import time
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


class _FrameGroups(torch.utils.data.Dataset):
    """Each item is one optimizer step's frames, decoded in a DataLoader worker."""

    def __init__(self, images, groups):
        self.images, self.groups = images, groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        return group, self.images[group]


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

    def preflight(self, variant_path, proxy):
        """Check training and inference interfaces before submitting a GPU job."""
        model = self.build(variant_path, num_classes=1, device="cpu")
        max_params = self.settings.get("search_resource_limits", {}).get("max_params")
        if max_params is not None and self.param_count(model) > max_params:
            raise ValueError("D-FINE candidate exceeds parameter limit")
        images = torch.randint(0, 256, (2, 3, 65, 97), dtype=torch.uint8,
                               generator=torch.Generator().manual_seed(0))
        targets = [_targets_from_pixels(boxes, 97, 65, "cpu") for boxes in (
            torch.empty(0, 4), torch.tensor([[20., 15., 30., 25.]]),
        )]
        loss = self.criterion(model(images.float().div(255)), targets)["loss_total"]
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("D-FINE preflight produced missing or non-finite gradients")
        if len(self.predict(model, images, proxy, "cpu")) != len(images):
            raise ValueError("D-FINE preflight prediction count does not match the input")

    def screen_resources(self, model, image_shape, batch_size, device,
                         selection_limits=None):
        """One full-resolution forward/backward before a structured candidate trains.

        Optional absolute limits belong to the experiment protocol. The probe
        does not update parameters or BatchNorm statistics.
        """
        if len(image_shape) != 2 or min(image_shape) < 32 or batch_size < 1:
            raise ValueError("invalid D-FINE resource-screen shape or batch")
        limits = dict(self.settings.get("search_resource_limits", {}))
        if selection_limits is not None:
            limits["max_peak_bytes"] = min(
                limits.get("max_peak_bytes", float("inf")),
                selection_limits["max_peak_memory_bytes"])
        if not isinstance(limits, dict) or set(limits) - {"max_params", "max_peak_bytes", "max_step_ms"}:
            raise ValueError("invalid D-FINE search resource limits")
        if any(type(value) not in (int, float) or value <= 0 for value in limits.values()):
            raise ValueError("D-FINE search resource limits must be positive numbers")
        height, width = image_shape
        params = self.param_count(model)
        if params > limits.get("max_params", float("inf")):
            raise ValueError(f"D-FINE candidate parameters {params} exceed resource limit")
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        event = torch.zeros((batch_size, 3, height, width), device=device)
        one_box = torch.tensor([[width * .4, height * .4, width * .6, height * .6]])
        targets = [_targets_from_pixels(one_box if index == 0 else torch.empty(0, 4),
                                        width, height, device)
                   for index in range(batch_size)]
        is_cuda = torch.device(device).type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            loss = self.criterion(model(event), targets)["loss_total"]
            loss.backward()
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            peak_bytes = torch.cuda.max_memory_allocated(device) if is_cuda else None
            if peak_bytes is not None and peak_bytes > limits.get("max_peak_bytes", float("inf")):
                raise ValueError(f"D-FINE candidate peak memory {peak_bytes} exceeds resource limit")
            if elapsed_ms > limits.get("max_step_ms", float("inf")):
                raise ValueError(f"D-FINE candidate probe time {elapsed_ms:.1f} ms exceeds resource limit")
            return {"full_resolution_probe_batch": batch_size,
                    "full_resolution_probe_shape": [height, width],
                    "full_resolution_probe_params": params,
                    "full_resolution_probe_step_ms": elapsed_ms,
                    "full_resolution_probe_peak_bytes": peak_bytes}
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

    def train(self, model, data, proxy, device, on_epoch_end=None):
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
        schedule = self.settings.get("lr_schedule", "constant")
        if schedule not in ("constant", "cosine"):
            raise ValueError("D-FINE lr_schedule must be constant or cosine")
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.settings["lr"] * 0.1)
            if schedule == "cosine" else None)
        generator = torch.Generator().manual_seed(proxy["seed"])
        workers = self.settings.get("workers", 0)
        order_digest = hashlib.sha256()
        self.training_epoch_order_sha256 = []
        self.training_epoch_losses = []
        self.training_epoch_seconds = []
        loss_terms = ("loss_cls", "loss_bbox", "loss_giou", "loss_fgl", "loss_go_lsd")
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(images), generator=generator)
            order_bytes = order.numpy().tobytes()
            order_digest.update(order_bytes)
            self.training_epoch_order_sha256.append(hashlib.sha256(order_bytes).hexdigest())
            groups = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
            # Workers decode upcoming PNG frames while the GPU trains; the
            # frame order and optimizer steps are the same as without them.
            loader = torch.utils.data.DataLoader(
                _FrameGroups(images, groups), batch_size=None, num_workers=workers,
                pin_memory=torch.device(device).type == "cuda",
            )
            total = 0.0
            term_totals = {name: 0.0 for name in loss_terms}
            for group, group_images in loader:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                batch_terms = {name: 0.0 for name in loss_terms}
                group_targets = sum(len(boxes[int(i)]) for i in group)
                for offset in range(0, len(group), microbatch):
                    selected = group[offset:offset + microbatch]
                    raw = group_images[offset:offset + microbatch]
                    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.uint8:
                        raise ValueError("D-FINE training images must decode to uint8 tensors")
                    event = raw.to(device=device, non_blocking=True).float().div_(255.0)
                    targets = [_targets_from_pixels(boxes[int(i)], width, height, device)
                               for i in selected]
                    losses = self.criterion(model(event), targets)
                    frame_weight = len(selected) / len(group)
                    target_weight = sum(len(target["boxes"]) for target in targets) / max(group_targets, 1)
                    # Classification and distillation are frame means; localization
                    # is normalized by the number of targets in the optimizer step.
                    weighted = {
                        name: losses[name] * (frame_weight if name in ("loss_cls", "loss_go_lsd")
                                               else target_weight)
                        for name in loss_terms
                    }
                    weighted_total = (weighted["loss_cls"] + 5 * weighted["loss_bbox"]
                                      + 2 * weighted["loss_giou"] + weighted["loss_fgl"]
                                      + 0.5 * weighted["loss_go_lsd"])
                    weighted_total.backward()
                    # Kept on the GPU so each microbatch does not wait for a sync.
                    batch_loss = batch_loss + weighted_total.detach()
                    for name in loss_terms:
                        batch_terms[name] = batch_terms[name] + weighted[name].detach()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.settings.get("grad_clip", 0.1),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                total += batch_loss * len(group)
                for name in loss_terms:
                    term_totals[name] = term_totals[name] + batch_terms[name] * len(group)
            train_loss = float(total) / len(images)
            self.training_epoch_losses.append({
                name: float(value) / len(images) for name, value in term_totals.items()
            })
            self.training_epoch_seconds.append(time.perf_counter() - epoch_start)
            learning_rate = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch + 1}/{epochs}: D-FINE loss {train_loss:.4f}", flush=True)
            if scheduler is not None:
                scheduler.step()
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, train_loss, learning_rate)
        self.training_order_sha256 = order_digest.hexdigest()

    def predict(self, model, images, proxy, device):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("D-FINE inference requires [B,3,H,W] Event tensors")
        if images.dtype != torch.uint8:
            raise ValueError("D-FINE inference images must be uint8")
        threshold, nms_iou = proxy["eval_conf"], proxy["nms_iou"]
        if not 0 <= threshold <= 1 or (nms_iou is not None and not 0 <= nms_iou <= 1):
            raise ValueError("D-FINE confidence must be in [0,1]; NMS IoU must be in [0,1] or None")
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
                    selected = (_nms(boxes, scores, nms_iou) if nms_iou is not None
                                else torch.arange(len(boxes)))
                    output.append((boxes[selected], scores[selected]))
        return output

    def postprocessing(self, proxy):
        nms = "none" if proxy["nms_iou"] is None else proxy["nms_iou"]
        return f"D-FINE conf={proxy['eval_conf']} nms_iou={nms}"
