"""Compact D-FINE-inspired detector for FRED's released Event-frame PNGs.

The three architecture sections are the only LLM-GE mutation units. Data access,
target association, loss, prediction formatting, and benchmark policy stay outside
those sections. This is a project-owned prototype, not the official D-FINE model.

Inputs are floating Event tensors in [0, 1] with shape [B, C, H, W]. Targets for
DFINECriterion are approved, model-input-frame boxes in normalized cxcywh format.
The seed does not fetch FRED data or interpret source annotations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F


def event_pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert a released RGB-encoded Event PNG to [3,H,W] float32 in [0,1].

    RGB here is the PNG storage mode, not a separate RGB-camera modality. The
    uint8/255 scale is a prototype contract and must be frozen with Phase 0 before
    formal training. Other image modes fail rather than being silently converted.
    """
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("expected a released Event PNG decoded in RGB mode")
    if image.width < 32 or image.height < 32:
        raise ValueError("Event frame must be at least 32 pixels on each side")
    pixels = np.array(image, copy=True)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Event PNG must have three uint8 channels")
    return torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255.0)


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    low, high = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((low + high) / 2, high - low), dim=-1)


def _inverse_sigmoid(x: Tensor) -> Tensor:
    x = x.clamp(1e-5, 1 - 1e-5)
    return torch.logit(x)


def _decode_distributions(reference: Tensor, corner_logits: Tensor, bin_values: Tensor) -> Tensor:
    """Fixed four-edge geometry shared by the seed and its contract checks."""
    probabilities = corner_logits.softmax(-1)
    offsets = (probabilities * bin_values).sum(-1)
    center = reference[..., :2]
    width, height = reference[..., 2], reference[..., 3]
    # Bounded predictions are model geometry, not repairs to FRED annotations.
    top = height * (0.5 + offsets[..., 0]).clamp_min(1e-4)
    bottom = height * (0.5 + offsets[..., 1]).clamp_min(1e-4)
    left = width * (0.5 + offsets[..., 2]).clamp_min(1e-4)
    right = width * (0.5 + offsets[..., 3]).clamp_min(1e-4)
    xyxy = torch.stack((
        (center[..., 0] - left).clamp(0, 1),
        (center[..., 1] - top).clamp(0, 1),
        (center[..., 0] + right).clamp(0, 1),
        (center[..., 1] + bottom).clamp(0, 1),
    ), dim=-1)
    return _xyxy_to_cxcywh(xyxy)


def _pairwise_iou_and_giou(boxes_a: Tensor, boxes_b: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU/GIoU for valid normalized xyxy boxes."""
    a, b = boxes_a[:, None, :], boxes_b[None, :, :]
    low = torch.maximum(a[..., :2], b[..., :2])
    high = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (high - low).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union.clamp_min(1e-8)
    enclosing_low = torch.minimum(a[..., :2], b[..., :2])
    enclosing_high = torch.maximum(a[..., 2:], b[..., 2:])
    enclosing = (enclosing_high - enclosing_low).prod(-1).clamp_min(1e-8)
    giou = iou - (enclosing - union) / enclosing
    return iou, giou


def _validate_targets(targets: Sequence[dict[str, Tensor]], batch_size: int, device: torch.device) -> None:
    if len(targets) != batch_size:
        raise ValueError("one target dictionary is required for each Event frame")
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or set(target) != {"boxes", "labels"}:
            raise ValueError(f"target {index} must contain only boxes and labels")
        boxes, labels = target["boxes"], target["labels"]
        if not isinstance(boxes, Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"target {index} boxes must have shape [N,4]")
        if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
            raise ValueError(f"target {index} labels must have shape [N]")
        if boxes.device != device or labels.device != device:
            raise ValueError(f"target {index} must be on the prediction device")
        if not boxes.is_floating_point() or not torch.isfinite(boxes).all():
            raise ValueError(f"target {index} boxes must be finite floating values")
        if labels.dtype not in (torch.int64, torch.int32) or torch.any(labels != 0):
            raise ValueError(f"target {index} must contain only integer drone labels (0)")
        if boxes.numel():
            xyxy = _cxcywh_to_xyxy(boxes)
            if (boxes[:, 2:] <= 0).any() or (xyxy < 0).any() or (xyxy > 1).any():
                raise ValueError(f"target {index} has an invalid or out-of-bounds box")


def _validate_prediction(prediction: dict, batch_size: int, num_queries: int, reg_max: int) -> None:
    expected = {
        "pred_logits": (batch_size, num_queries, 1),
        "pred_boxes": (batch_size, num_queries, 4),
        "pred_corners": (batch_size, num_queries, 4, reg_max + 1),
        "ref_points": (batch_size, num_queries, 4),
    }
    for key, shape in expected.items():
        value = prediction.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{key} must have shape {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains a non-finite value")
    for key in ("pred_boxes", "ref_points"):
        boxes = prediction[key]
        xyxy = _cxcywh_to_xyxy(boxes)
        if (boxes[..., 2:] <= 0).any() or (xyxy < 0).any() or (xyxy > 1).any():
            raise ValueError(f"{key} contains an invalid normalized box")
    fixed_bins = _make_bin_values(reg_max).to(
        device=prediction["pred_corners"].device, dtype=prediction["pred_corners"].dtype
    )
    expected_boxes = _decode_distributions(
        prediction["ref_points"], prediction["pred_corners"], fixed_bins
    )
    if not torch.allclose(prediction["pred_boxes"], expected_boxes, rtol=1e-4, atol=1e-5):
        raise ValueError("pred_boxes must use the fixed four-edge distribution decoder")


class ConvNormAct(nn.Sequential):
    """Shared protected convolution block, stable across architecture mutations."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(math.gcd(out_channels, 8), out_channels),
            nn.SiLU(),
        )


class DFINECriterion(nn.Module):
    """Fixed D-FINE-inspired one-to-one, FGL, and GO-LSD training objective.

    Loss weights are prototype values, not an approved FRED experiment protocol.
    The caller must use ``loss_total`` exactly once for backward(). Matching uses
    SciPy; the model's inference path does not need that dependency.
    """

    def __init__(self, reg_max: int = 16, temperature: float = 2.0):
        super().__init__()
        if reg_max < 4 or reg_max % 2 or temperature <= 0:
            raise ValueError("reg_max must be even and >=4; temperature must be positive")
        self.reg_max = reg_max
        self.temperature = temperature
        self.register_buffer("bin_values", _make_bin_values(reg_max))

    @torch.no_grad()
    def _match(self, output: dict, targets: Sequence[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError("DFINECriterion requires the declared SciPy matcher dependency") from exc

        assignments = []
        for batch_index, target in enumerate(targets):
            count = target["boxes"].shape[0]
            if count == 0:
                empty = torch.empty(0, dtype=torch.long, device=output["pred_boxes"].device)
                assignments.append((empty, empty))
                continue
            if count > output["pred_boxes"].shape[1]:
                raise ValueError("ground-truth box count exceeds the fixed query count")
            predicted = output["pred_boxes"][batch_index]
            reference = target["boxes"]
            _, giou = _pairwise_iou_and_giou(
                _cxcywh_to_xyxy(predicted), _cxcywh_to_xyxy(reference)
            )
            class_cost = -output["pred_logits"][batch_index].sigmoid()
            box_cost = torch.cdist(predicted, reference, p=1)
            cost = class_cost + 5 * box_cost - 2 * giou
            rows, columns = linear_sum_assignment(cost.cpu().numpy())
            assignments.append((
                torch.as_tensor(rows, dtype=torch.long, device=predicted.device),
                torch.as_tensor(columns, dtype=torch.long, device=predicted.device),
            ))
        return assignments

    def _fgl(self, corners: Tensor, references: Tensor, target_boxes: Tensor, iou: Tensor) -> Tensor:
        reference_xyxy = _cxcywh_to_xyxy(references)
        target_xyxy = _cxcywh_to_xyxy(target_boxes)
        width = references[:, 2].clamp_min(1e-5)
        height = references[:, 3].clamp_min(1e-5)
        offsets = torch.stack((
            (reference_xyxy[:, 1] - target_xyxy[:, 1]) / height,
            (target_xyxy[:, 3] - reference_xyxy[:, 3]) / height,
            (reference_xyxy[:, 0] - target_xyxy[:, 0]) / width,
            (target_xyxy[:, 2] - reference_xyxy[:, 2]) / width,
        ), dim=-1)
        # Distribution support is bounded; L1/GIoU still supervise larger errors.
        values = self.bin_values.to(dtype=corners.dtype)
        offsets = offsets.clamp(values[0].item(), values[-1].item())
        right = torch.searchsorted(values.contiguous(), offsets.contiguous()).clamp(1, self.reg_max)
        left = right - 1
        left_value, right_value = values[left], values[right]
        right_weight = (offsets - left_value) / (right_value - left_value).clamp_min(1e-8)
        log_prob = F.log_softmax(corners, dim=-1)
        left_log = log_prob.gather(-1, left.unsqueeze(-1)).squeeze(-1)
        right_log = log_prob.gather(-1, right.unsqueeze(-1)).squeeze(-1)
        edge_loss = -((1 - right_weight) * left_log + right_weight * right_log)
        # A positive floor keeps early FGL gradients when initial IoU is zero.
        return (edge_loss.mean(-1) * (0.25 + 0.75 * iou.detach())).sum()

    def _go_lsd(
        self,
        layers: list[dict],
        matches: list[list[tuple[Tensor, Tensor]]],
        targets: Sequence[dict[str, Tensor]],
    ) -> Tensor:
        teacher = layers[-1]["pred_corners"].detach()
        teacher_scores = layers[-1]["pred_logits"].detach().sigmoid().squeeze(-1)
        zero = teacher.sum() * 0.0
        if len(layers) == 1:
            return zero
        result = zero
        query_count = teacher.shape[1]
        for layer in layers[:-1]:
            student_log = F.log_softmax(layer["pred_corners"] / self.temperature, dim=-1)
            teacher_log = F.log_softmax(teacher / self.temperature, dim=-1)
            teacher_prob = teacher_log.exp()
            kl = (teacher_prob * (teacher_log - student_log)).sum(-1).mean(-1)
            kl = kl * (self.temperature ** 2)
            for batch_index, target in enumerate(targets):
                union_mask = torch.zeros(query_count, dtype=torch.bool, device=teacher.device)
                for layer_matches in matches:
                    union_mask[layer_matches[batch_index][0]] = True
                matched_count = union_mask.sum().to(dtype=kl.dtype)
                unmatched_count = (~union_mask).sum().to(dtype=kl.dtype)
                normalizer = (matched_count.sqrt() + unmatched_count.sqrt()).clamp_min(1)
                if target["boxes"].shape[0]:
                    iou, _ = _pairwise_iou_and_giou(
                        _cxcywh_to_xyxy(layers[-1]["pred_boxes"][batch_index].detach()),
                        _cxcywh_to_xyxy(target["boxes"]),
                    )
                    quality = iou.max(dim=1).values
                else:
                    quality = teacher_scores[batch_index].new_zeros(query_count)
                matched_weight = quality * matched_count.sqrt() / normalizer
                unmatched_weight = teacher_scores[batch_index] * unmatched_count.sqrt() / normalizer
                weights = torch.where(union_mask, matched_weight, unmatched_weight)
                result = result + (kl[batch_index] * weights).sum() / weights.sum().clamp_min(1e-6)
        return result / (len(layers) - 1) / len(targets)

    def forward(self, outputs: dict, targets: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not isinstance(outputs, dict) or "aux_outputs" not in outputs:
            raise ValueError("model output must include aux_outputs")
        batch_size, num_queries, _ = outputs["pred_logits"].shape
        _validate_targets(targets, batch_size, outputs["pred_logits"].device)
        layers = list(outputs["aux_outputs"]) + [outputs]
        for layer in layers:
            _validate_prediction(layer, batch_size, num_queries, self.reg_max)
        matches = [self._match(layer, targets) for layer in layers]
        zero = outputs["pred_logits"].sum() * 0.0
        loss_cls = loss_bbox = loss_giou = loss_fgl = zero
        total_targets = max(sum(target["boxes"].shape[0] for target in targets), 1)
        for layer, layer_matches in zip(layers, matches):
            labels = torch.zeros_like(layer["pred_logits"])
            for batch_index, (query_indices, target_indices) in enumerate(layer_matches):
                labels[batch_index, query_indices, 0] = 1
                if not query_indices.numel():
                    continue
                predicted = layer["pred_boxes"][batch_index, query_indices]
                actual = targets[batch_index]["boxes"][target_indices]
                iou_matrix, giou_matrix = _pairwise_iou_and_giou(
                    _cxcywh_to_xyxy(predicted), _cxcywh_to_xyxy(actual)
                )
                iou, giou = iou_matrix.diag(), giou_matrix.diag()
                loss_bbox = loss_bbox + F.l1_loss(predicted, actual, reduction="sum") / total_targets
                loss_giou = loss_giou + (1 - giou).sum() / total_targets
                loss_fgl = loss_fgl + self._fgl(
                    layer["pred_corners"][batch_index, query_indices],
                    layer["ref_points"][batch_index, query_indices], actual, iou,
                ) / total_targets
            loss_cls = loss_cls + F.binary_cross_entropy_with_logits(
                layer["pred_logits"], labels, reduction="mean"
            )
        layer_count = len(layers)
        losses = {
            "loss_cls": loss_cls / layer_count,
            "loss_bbox": loss_bbox / layer_count,
            "loss_giou": loss_giou / layer_count,
            "loss_fgl": loss_fgl / layer_count,
            "loss_go_lsd": self._go_lsd(layers, matches, targets),
        }
        losses["loss_total"] = (
            losses["loss_cls"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
            + losses["loss_fgl"] + 0.5 * losses["loss_go_lsd"]
        )
        if any(not torch.isfinite(value) for value in losses.values()):
            raise ValueError("D-FINE loss contains a non-finite value")
        return losses


def _make_bin_values(reg_max: int) -> Tensor:
    position = torch.linspace(-1.0, 1.0, reg_max + 1)
    return position.sign() * position.abs().square() * 2.0


class DFINEEventDetector(nn.Module):
    """Protected Event-only wrapper with fixed prediction semantics."""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        d_model: int = 64,
        num_queries: int = 64,
        num_layers: int = 3,
        reg_max: int = 16,
        confidence_threshold: float = 0.0,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("this prototype expects the verified three-channel Event PNG encoding")
        if min(width, d_model, num_queries, num_layers) <= 0 or d_model % 4:
            raise ValueError("model dimensions must be positive and d_model divisible by four")
        if reg_max < 4 or reg_max % 2:
            raise ValueError("reg_max must be an even integer >= 4")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0,1]")
        self.in_channels = in_channels
        self.num_queries = num_queries
        self.reg_max = reg_max
        self.confidence_threshold = confidence_threshold
        self.backbone = EventBackbone(in_channels, width)
        self.encoder = MultiScaleEncoder(self.backbone.out_channels, d_model)
        self.decoder = DistributionRefinementDecoder(d_model, num_queries, num_layers, reg_max)

    def forward(self, event: Tensor) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        if not isinstance(event, Tensor) or event.ndim != 4 or event.shape[1] != self.in_channels:
            raise ValueError("event must have shape [B,3,H,W]")
        if event.shape[0] < 1 or min(event.shape[2:]) < 32:
            raise ValueError("Event batch must be nonempty and at least 32x32")
        if not event.is_floating_point() or not torch.isfinite(event).all():
            raise ValueError("event must contain finite floating values")
        if (event < 0).any() or (event > 1).any():
            raise ValueError("event values must use the fixed [0,1] scale")
        outputs = self.decoder(self.encoder(self.backbone(event)))
        _validate_prediction(outputs, event.shape[0], self.num_queries, self.reg_max)
        if len(outputs["aux_outputs"]) != self.decoder.num_layers - 1:
            raise ValueError("decoder returned the wrong number of auxiliary layers")
        for layer in outputs["aux_outputs"]:
            _validate_prediction(layer, event.shape[0], self.num_queries, self.reg_max)
        return outputs

    @torch.no_grad()
    def predict(self, event: Tensor) -> list[dict[str, list]]:
        """Return NMS-free detections in model-input pixels; call eval() first.

        The evaluation adapter must invert any resize/pad transform before
        comparing these boxes with original-image FRED annotations.
        """
        if self.training:
            raise RuntimeError("call model.eval() before predict()")
        outputs = self.forward(event)
        height, width = event.shape[-2:]
        scores = outputs["pred_logits"].sigmoid().squeeze(-1)
        xyxy = _cxcywh_to_xyxy(outputs["pred_boxes"])
        scale = xyxy.new_tensor((width, height, width, height))
        results = []
        for image_boxes, image_scores in zip(xyxy, scores):
            selected = image_scores >= self.confidence_threshold
            selected_boxes = image_boxes[selected] * scale
            selected_scores = image_scores[selected]
            order = selected_scores.argsort(descending=True)
            selected_boxes = selected_boxes[order]
            selected_scores = selected_scores[order]
            results.append({
                "boxes": selected_boxes.cpu().tolist(),
                "scores": selected_scores.cpu().tolist(),
                "labels": [0] * selected_scores.numel(),
            })
        return results


# --OPTION--
class EventBackbone(nn.Module):
    """Three-scale CNN Event backbone with a small-target stride-4 map."""

    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, width, stride=2),
            ConvNormAct(width, width, stride=2),
        )
        self.stage8 = nn.Sequential(
            ConvNormAct(width, width * 2, stride=2),
            ConvNormAct(width * 2, width * 2),
        )
        self.stage16 = nn.Sequential(
            ConvNormAct(width * 2, width * 4, stride=2),
            ConvNormAct(width * 4, width * 4),
        )
        self.out_channels = (width, width * 2, width * 4)

    def forward(self, event: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p4 = self.stem(event)
        p8 = self.stage8(p4)
        p16 = self.stage16(p8)
        return p4, p8, p16


# --OPTION--
class MultiScaleEncoder(nn.Module):
    """Project and fuse three spatial scales without changing their sizes."""

    def __init__(self, backbone_channels: tuple[int, int, int], d_model: int):
        super().__init__()
        self.project = nn.ModuleList(
            nn.Conv2d(channels, d_model, kernel_size=1)
            for channels in backbone_channels
        )
        self.refine = nn.ModuleList(
            ConvNormAct(d_model, d_model) for _ in range(3)
        )

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if len(features) != 3:
            raise ValueError("encoder requires exactly three backbone feature maps")
        p4, p8, p16 = (project(feature) for project, feature in zip(self.project, features))
        p16 = self.refine[2](p16)
        p8 = self.refine[1](p8 + F.interpolate(p16, size=p8.shape[-2:], mode="nearest"))
        p4 = self.refine[0](p4 + F.interpolate(p8, size=p4.shape[-2:], mode="nearest"))
        return p4, p8, p16


# --OPTION--
class DistributionRefinementDecoder(nn.Module):
    """Encoder-selected queries with iterative four-edge distribution refinement."""

    def __init__(self, d_model: int, num_queries: int, num_layers: int, reg_max: int):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.reg_max = reg_max
        self.max_memory_tokens = max(256, 4 * num_queries)
        self.position = nn.Linear(2, d_model)
        self.level_embedding = nn.Parameter(torch.zeros(3, d_model))
        self.objectness = nn.Linear(d_model, 1)
        self.query_xy = nn.Linear(d_model, 2)
        self.query_wh = nn.Linear(d_model, 2)
        nn.init.zeros_(self.query_xy.weight)
        nn.init.zeros_(self.query_xy.bias)
        nn.init.constant_(self.query_wh.bias, -2.0)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
                dropout=0.0, batch_first=True, activation="gelu",
            )
            for _ in range(num_layers)
        )
        self.class_heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in range(num_layers))
        self.corner_heads = nn.ModuleList(
            nn.Linear(d_model, 4 * (reg_max + 1)) for _ in range(num_layers)
        )
        self.register_buffer("bin_values", _make_bin_values(reg_max))

    def _tokens(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        token_sets, coordinate_sets = [], []
        for level, feature in enumerate(features):
            batch, channels, height, width = feature.shape
            yy, xx = torch.meshgrid(
                (torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5) / height,
                (torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5) / width,
                indexing="ij",
            )
            coordinates = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
            coordinates = coordinates.expand(batch, -1, -1)
            tokens = feature.flatten(2).transpose(1, 2)
            tokens = tokens + self.position(coordinates) + self.level_embedding[level]
            token_sets.append(tokens)
            coordinate_sets.append(coordinates)
        return torch.cat(token_sets, dim=1), torch.cat(coordinate_sets, dim=1)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> dict:
        if len(features) != 3:
            raise ValueError("decoder requires exactly three encoded feature maps")
        tokens, coordinates = self._tokens(features)
        batch, token_count, channels = tokens.shape
        if token_count < self.num_queries:
            raise ValueError("input feature maps contain fewer positions than queries")
        objectness = self.objectness(tokens).squeeze(-1)
        memory_indices = objectness.topk(min(token_count, self.max_memory_tokens), dim=1).indices
        query_indices = memory_indices[:, :self.num_queries]
        memory = tokens.gather(1, memory_indices.unsqueeze(-1).expand(-1, -1, channels))
        query = tokens.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, channels))
        anchors = coordinates.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 2))
        center = (_inverse_sigmoid(anchors) + self.query_xy(query)).sigmoid()
        center = center.clamp(1e-4, 1 - 1e-4)
        available_size = 2 * torch.minimum(center, 1 - center)
        reference_size = self.query_wh(query).sigmoid() * available_size
        reference = torch.cat((center, reference_size), dim=-1)
        accumulated = query.new_zeros(batch, self.num_queries, 4, self.reg_max + 1)
        predictions = []
        for layer, class_head, corner_head in zip(
            self.layers, self.class_heads, self.corner_heads
        ):
            query = layer(query, memory)
            accumulated = accumulated + corner_head(query).reshape(
                batch, self.num_queries, 4, self.reg_max + 1
            )
            predictions.append({
                "pred_logits": class_head(query) + objectness.gather(1, query_indices).unsqueeze(-1),
                "pred_boxes": _decode_distributions(reference, accumulated, self.bin_values),
                "pred_corners": accumulated,
                "ref_points": reference,
            })
        return {**predictions[-1], "aux_outputs": predictions[:-1]}
