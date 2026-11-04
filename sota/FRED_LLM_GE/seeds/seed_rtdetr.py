"""Original RT-DETR (v1) detector for FRED's released Event-frame PNGs.

Ported from the official PyTorch implementation, lyuwenyu/RT-DETR
(rtdetr_pytorch, commit df2a96c8c3a098567a1149bf0248da6efe6d495d, Apache-2.0),
which the FRED paper cites as RT-DETR. The registry/config system is replaced by
explicit constructor arguments; the architecture, Hungarian matcher, criterion and
contrastive denoising follow the official code. Models are trained from scratch:
no pretrained weights are loaded and no normalization layer is frozen.

The three architecture sections (backbone, encoder, decoder) are the only LLM-GE
mutation units. Inputs are Event tensors in [0, 1] with shape [B, 3, H, W], H and
W multiples of 32. Targets for RTDETRCriterion are model-input-frame boxes in
normalized cxcywh format with drone label 0. The seed does not fetch FRED data.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init


# ---------------------------------------------------------------- helpers

def get_activation(act, inplace: bool = True) -> nn.Module:
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    modules = {"silu": nn.SiLU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU}
    if act.lower() not in modules:
        raise ValueError(f"unsupported activation: {act}")
    module = modules[act.lower()]()
    if hasattr(module, "inplace"):
        module.inplace = inplace
    return module


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all() or not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError("generalized_box_iou requires valid xyxy boxes")
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Multi-scale deformable attention via grid_sample (official pure-PyTorch path)."""
    bs, _, n_head, c = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_value_list.append(F.grid_sample(
            value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False))
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, len_q, n_levels * n_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
    return output.sum(-1).reshape(bs, n_head * c, len_q).permute(0, 2, 1)


# ---------------------------------------------------------------- official building blocks

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride,
                              padding=(kernel_size - 1) // 2 if padding is None else padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        stride1, stride2 = (stride, 1) if variant == "a" else (1, stride)
        width = ch_out
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ("conv", ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)),
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2c(self.branch2b(self.branch2a(x)))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act="relu", variant="b"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(block(
                ch_in, ch_out,
                stride=2 if i == 0 and stage_num != 2 else 1,
                shortcut=i != 0, variant=variant, act=act))
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class CSPRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=None, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        return output if self.norm is None else self.norm(output)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (reference_points.reshape(bs, len_q, 1, self.num_levels, 1, 2)
                                  + sampling_offsets / offset_normalizer)
        elif reference_points.shape[-1] == 4:
            sampling_locations = (reference_points[:, :, None, :, None, :2]
                                  + sampling_offsets / self.num_points
                                  * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError("last dim of reference_points must be 2 or 4")
        output = deformable_attention_core_func(value, value_spatial_shapes, sampling_locations,
                                                attention_weights)
        return self.output_proj(output)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0,
                 activation="relu", n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes,
                memory_level_start_index, attn_mask=None, memory_mask=None, query_pos_embed=None):
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed), reference_points,
                               memory, memory_spatial_shapes, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt.clamp(min=-65504, max=65504))


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self, tgt, ref_points_unact, memory, memory_spatial_shapes, memory_level_start_index,
                bbox_head, score_head, query_pos_head, attn_mask=None, memory_mask=None):
        output = tgt
        dec_out_bboxes, dec_out_logits = [], []
        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_points = ref_points_detach
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            output = layer(output, ref_points_input, memory, memory_spatial_shapes,
                           memory_level_start_index, attn_mask, memory_mask, query_pos_embed)
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


def get_contrastive_denoising_training_group(targets, num_classes, num_queries, class_embed,
                                             num_denoising=100, label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    """Official contrastive denoising queries (training only)."""
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None
    num_group = max(num_denoising // max_gt_num, 1)
    bs = len(num_gts)
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = (1 - negative_gt_mask).squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    num_denoising = int(max_gt_num * 2 * num_group)
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = inverse_sigmoid(box_xyxy_to_cxcywh(known_bbox))
    input_query_class = class_embed(input_query_class)
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # matching queries cannot see the reconstruction; reconstruction groups cannot see each other
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        start, end = max_gt_num * 2 * i, max_gt_num * 2 * (i + 1)
        if i == 0:
            attn_mask[start:end, end:num_denoising] = True
        if i == num_group - 1:
            attn_mask[start:end, :start] = True
        else:
            attn_mask[start:end, end:num_denoising] = True
            attn_mask[start:end, :start] = True
    dn_meta = {"dn_positive_idx": dn_positive_idx, "dn_num_group": num_group,
               "dn_num_split": [num_denoising, num_queries]}
    return input_query_class, input_query_bbox, attn_mask, dn_meta


# ---------------------------------------------------------------- fixed matcher and criterion

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
            xyxy = box_cxcywh_to_xyxy(boxes)
            if (boxes[:, 2:] <= 0).any() or (xyxy < 0).any() or (xyxy > 1).any():
                raise ValueError(f"target {index} has an invalid or out-of-bounds box")


def _validate_prediction(prediction: dict, batch_size: int, num_queries: int, name: str) -> None:
    for key, shape in (("pred_logits", (batch_size, num_queries, 1)),
                       ("pred_boxes", (batch_size, num_queries, 4))):
        value = prediction.get(key) if isinstance(prediction, dict) else None
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} {key} must have shape {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} {key} contains a non-finite value")
    boxes = prediction["pred_boxes"]
    if (boxes < 0).any() or (boxes > 1).any():
        raise ValueError(f"{name} pred_boxes must be sigmoid-normalized cxcywh in [0,1]")


class HungarianMatcher(nn.Module):
    """Official one-to-one matcher with the focal classification cost."""

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError("HungarianMatcher requires the declared SciPy dependency") from exc
        bs, num_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device
        sizes = [len(t["boxes"]) for t in targets]
        if sum(sizes) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return [(empty, empty) for _ in targets]
        out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        out_prob = out_prob[:, tgt_ids]
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class - neg_cost_class
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(bs, num_queries, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64, device=device),
                 torch.as_tensor(j, dtype=torch.int64, device=device)) for i, j in indices]


class RTDETRCriterion(nn.Module):
    """Official SetCriterion with losses ['vfl', 'boxes'] for one class.

    Weights follow the official config (vfl 1, bbox 5, giou 2; VFL alpha 0.75,
    gamma 2). loss_total sums every weighted term, as the official trainer does.
    """

    def __init__(self, num_classes=1, alpha=0.75, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha, self.gamma = alpha, gamma
        self.matcher = HungarianMatcher()
        self.weight_dict = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}

    @staticmethod
    def _src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_vfl(self, outputs, targets, indices, num_boxes):
        idx = self._src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()
        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        return {"loss_vfl": loss.mean(1).sum() * src_logits.shape[1] / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                       box_cxcywh_to_xyxy(target_boxes)))
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou.sum() / num_boxes}

    def _weighted(self, outputs, targets, indices, num_boxes, suffix=""):
        losses = {}
        for compute in (self.loss_vfl, self.loss_boxes):
            for key, value in compute(outputs, targets, indices, num_boxes).items():
                losses[key + suffix] = value * self.weight_dict[key]
        return losses

    @staticmethod
    def _cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        device = targets[0]["labels"].device
        indices = []
        for i, target in enumerate(targets):
            num_gt = len(target["labels"])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(dn_num_group)
                if len(dn_positive_idx[i]) != len(gt_idx):
                    raise ValueError("denoising query count does not match the targets")
                indices.append((dn_positive_idx[i], gt_idx))
            else:
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                indices.append((empty, empty))
        return indices

    def forward(self, outputs: dict, targets: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not isinstance(outputs, dict) or "aux_outputs" not in outputs:
            raise ValueError("training output must include aux_outputs")
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        _validate_targets(targets, batch_size, outputs["pred_logits"].device)
        main = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
        num_boxes = max(float(sum(len(t["labels"]) for t in targets)), 1.0)
        losses = self._weighted(main, targets, self.matcher(main, targets), num_boxes)
        for i, aux in enumerate(outputs["aux_outputs"]):
            losses.update(self._weighted(aux, targets, self.matcher(aux, targets), num_boxes, f"_aux_{i}"))
        if "dn_aux_outputs" in outputs:
            if "dn_meta" not in outputs:
                raise ValueError("dn_aux_outputs requires dn_meta")
            indices = self._cdn_matched_indices(outputs["dn_meta"], targets)
            dn_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            for i, aux in enumerate(outputs["dn_aux_outputs"]):
                losses.update(self._weighted(aux, targets, indices, dn_boxes, f"_dn_{i}"))
        losses["loss_total"] = sum(losses.values())
        if any(not torch.isfinite(value) for value in losses.values()):
            raise ValueError("RT-DETR loss contains a non-finite value")
        return losses


# ---------------------------------------------------------------- protected wrapper

# Official per-depth settings (configs/rtdetr/*.yml): encoder hidden_dim,
# dim_feedforward and CSP expansion, decoder width, and number of decoder layers.
DEPTH_PRESETS = {
    18: {"hidden_dim": 256, "dim_feedforward": 1024, "expansion": 0.5,
         "decoder_dim": 256, "num_decoder_layers": 3},
    34: {"hidden_dim": 256, "dim_feedforward": 1024, "expansion": 0.5,
         "decoder_dim": 256, "num_decoder_layers": 4},
    50: {"hidden_dim": 256, "dim_feedforward": 1024, "expansion": 1.0,
         "decoder_dim": 256, "num_decoder_layers": 6},
    101: {"hidden_dim": 384, "dim_feedforward": 2048, "expansion": 1.0,
          "decoder_dim": 256, "num_decoder_layers": 6},
}


class RTDETREventDetector(nn.Module):
    """Protected Event-only wrapper with fixed input, output and prediction semantics."""

    stride_multiple = 32

    def __init__(self, in_channels: int = 3, depth: int = 50, num_queries: int = 300,
                 num_denoising: int = 100, hidden_dim: int | None = None,
                 dim_feedforward: int | None = None, expansion: float | None = None,
                 decoder_dim: int | None = None, num_decoder_layers: int | None = None,
                 confidence_threshold: float = 0.0):
        super().__init__()
        if in_channels != 3:
            raise ValueError("this seed expects the verified three-channel Event PNG encoding")
        if depth not in DEPTH_PRESETS:
            raise ValueError(f"depth must be one of {sorted(DEPTH_PRESETS)}")
        preset = dict(DEPTH_PRESETS[depth])
        for key, value in (("hidden_dim", hidden_dim), ("dim_feedforward", dim_feedforward),
                           ("expansion", expansion), ("decoder_dim", decoder_dim),
                           ("num_decoder_layers", num_decoder_layers)):
            if value is not None:
                preset[key] = value
        if (min(num_queries, preset["hidden_dim"], preset["decoder_dim"], preset["num_decoder_layers"]) <= 0
                or num_denoising < 0):
            raise ValueError("model dimensions must be positive")
        if preset["hidden_dim"] % 8 or preset["decoder_dim"] % 8:
            raise ValueError("hidden_dim and decoder_dim must be divisible by the 8 attention heads")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0,1]")
        self.in_channels = in_channels
        self.depth = depth
        self.num_queries = num_queries
        self.num_decoder_layers = preset["num_decoder_layers"]
        self.confidence_threshold = confidence_threshold
        self.backbone = PResNet(depth)
        if list(self.backbone.out_strides) != [8, 16, 32] or len(self.backbone.out_channels) != 3:
            raise ValueError("PResNet must return three maps at strides 8, 16 and 32")
        self.encoder = HybridEncoder(list(self.backbone.out_channels), preset["hidden_dim"],
                                     preset["expansion"], preset["dim_feedforward"])
        if len(self.encoder.out_channels) != 3:
            raise ValueError("HybridEncoder must return three feature maps")
        self.decoder = RTDETRTransformer(1, preset["decoder_dim"], num_queries,
                                         list(self.encoder.out_channels),
                                         self.num_decoder_layers, num_denoising)

    def _check_features(self, features, channels, size, name):
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise ValueError(f"{name} must return three feature maps")
        height, width = size
        for feature, stride, expected in zip(features, (8, 16, 32), channels):
            shape = (height // stride, width // stride)
            if (not isinstance(feature, Tensor) or feature.ndim != 4
                    or tuple(feature.shape[-2:]) != shape or feature.shape[1] != expected):
                raise ValueError(f"{name} map at stride {stride} must be [B,{expected},{shape[0]},{shape[1]}]")

    def forward(self, event: Tensor, targets: Sequence[dict[str, Tensor]] | None = None) -> dict:
        if not isinstance(event, Tensor) or event.ndim != 4 or event.shape[1] != self.in_channels:
            raise ValueError("event must have shape [B,3,H,W]")
        height, width = event.shape[-2:]
        if event.shape[0] < 1 or height % self.stride_multiple or width % self.stride_multiple:
            raise ValueError("Event batch must be nonempty with H and W multiples of 32")
        if not event.is_floating_point() or not torch.isfinite(event).all():
            raise ValueError("event must contain finite floating values")
        if (event < 0).any() or (event > 1).any():
            raise ValueError("event values must use the fixed [0,1] scale")
        if self.training:
            if targets is None:
                raise ValueError("training forward requires targets for contrastive denoising")
            _validate_targets(targets, event.shape[0], event.device)
        features = self.backbone(event)
        self._check_features(features, self.backbone.out_channels, (height, width), "PResNet")
        features = self.encoder(features)
        self._check_features(features, self.encoder.out_channels, (height, width), "HybridEncoder")
        outputs = self.decoder(features, list(targets) if self.training else None)
        batch_size = event.shape[0]
        _validate_prediction(outputs, batch_size, self.num_queries, "decoder")
        if self.training:
            aux = outputs.get("aux_outputs")
            if not isinstance(aux, list) or len(aux) != self.num_decoder_layers:
                raise ValueError("decoder must return one aux_outputs entry per decoder layer "
                                 "(earlier layers plus the encoder top-k proposals)")
            for layer in aux:
                _validate_prediction(layer, batch_size, self.num_queries, "aux")
            if "dn_aux_outputs" in outputs:
                meta = outputs.get("dn_meta")
                if not isinstance(meta, dict):
                    raise ValueError("dn_aux_outputs requires dn_meta")
                dn_queries = meta["dn_num_split"][0]
                for layer in outputs["dn_aux_outputs"]:
                    _validate_prediction(layer, batch_size, dn_queries, "denoising")
        return outputs

    @torch.no_grad()
    def predict(self, event: Tensor) -> list[dict[str, list]]:
        """Return detections in model-input pixels, best first; call eval() first.

        The adapter must invert any resize before comparing these boxes with
        original-image FRED annotations.
        """
        if self.training:
            raise RuntimeError("call model.eval() before predict()")
        outputs = self.forward(event)
        height, width = event.shape[-2:]
        scores = outputs["pred_logits"].sigmoid().squeeze(-1)
        xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"]).clamp(0, 1)
        scale = xyxy.new_tensor((width, height, width, height))
        results = []
        for image_boxes, image_scores in zip(xyxy, scores):
            selected = image_scores >= self.confidence_threshold
            selected_boxes = image_boxes[selected] * scale
            selected_scores = image_scores[selected]
            order = selected_scores.argsort(descending=True)
            results.append({
                "boxes": selected_boxes[order].cpu().tolist(),
                "scores": selected_scores[order].cpu().tolist(),
                "labels": [0] * int(selected_scores.numel()),
            })
        return results


# --OPTION--
class PResNet(nn.Module):
    """ResNet-vd backbone returning stride-8, -16 and -32 maps (official PResNet)."""

    def __init__(self, depth: int, variant: str = "d", num_stages: int = 4,
                 return_idx=(1, 2, 3), act: str = "relu"):
        super().__init__()
        block_nums = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}[depth]
        ch_in = 64
        if variant in ["c", "d"]:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]
        self.conv1 = nn.Sequential(OrderedDict([
            (name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, name in conv_def
        ]))
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock
        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]
        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            self.res_layers.append(Blocks(block, ch_in, ch_out_list[i], block_nums[i], i + 2,
                                          act=act, variant=variant))
            ch_in = _out_channels[i]
        self.return_idx = list(return_idx)
        self.out_channels = [_out_channels[i] for i in self.return_idx]
        self.out_strides = [_out_strides[i] for i in self.return_idx]

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


# --OPTION--
class HybridEncoder(nn.Module):
    """Attention on the stride-32 map plus CCFF top-down/bottom-up fusion (official)."""

    def __init__(self, in_channels, hidden_dim: int = 256, expansion: float = 1.0,
                 dim_feedforward: int = 1024, feat_strides=(8, 16, 32), nhead: int = 8,
                 dropout: float = 0.0, enc_act: str = "gelu", use_encoder_idx=(2,),
                 num_encoder_layers: int = 1, pe_temperature: float = 10000.0,
                 depth_mult: float = 1.0, act: str = "silu"):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.out_channels = [hidden_dim for _ in range(len(self.in_channels))]
        self.out_strides = self.feat_strides
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(hidden_dim))
            for channels in self.in_channels)
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, activation=enc_act)
        self.encoder = nn.ModuleList(
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(self.use_encoder_idx)))
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult),
                                               act=act, expansion=expansion))
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult),
                                               act=act, expansion=expansion))

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0, device=None):
        grid_w = torch.arange(int(w), dtype=torch.float32, device=device)
        grid_h = torch.arange(int(h), dtype=torch.float32, device=device)
        # tokens are flattened row-major from [B, C, H, W], so the grid is (h, w)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        if embed_dim % 4:
            raise ValueError("embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        if len(feats) != len(self.in_channels):
            raise ValueError("encoder received the wrong number of feature maps")
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature, device=src_flatten.device
                ).to(src_flatten.dtype)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, proj_feats[idx - 1]], dim=1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            downsample_feat = self.downsample_convs[idx](outs[-1])
            outs.append(self.pan_blocks[idx](torch.concat([downsample_feat, inner_outs[idx + 1]], dim=1)))
        return outs


# --OPTION--
class RTDETRTransformer(nn.Module):
    """Query selection from encoder tokens plus a deformable-attention decoder (official)."""

    def __init__(self, num_classes: int = 1, hidden_dim: int = 256, num_queries: int = 300,
                 feat_channels=(256, 256, 256), num_decoder_layers: int = 6,
                 num_denoising: int = 100, feat_strides=(8, 16, 32), num_levels: int = 3,
                 num_decoder_points: int = 4, nhead: int = 8, dim_feedforward: int = 1024,
                 dropout: float = 0.0, activation: str = "relu", label_noise_ratio: float = 0.5,
                 box_noise_scale: float = 1.0, learnt_init_query: bool = False,
                 eval_idx: int = -1, eps: float = 1e-2):
        super().__init__()
        feat_channels, feat_strides = list(feat_channels), list(feat_strides)
        if len(feat_channels) > num_levels or len(feat_strides) != len(feat_channels):
            raise ValueError("feat_channels and feat_strides must describe at most num_levels maps")
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.input_proj = nn.ModuleList(
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(channels, hidden_dim, 1, bias=False)),
                ("norm", nn.BatchNorm2d(hidden_dim))]))
            for channels in feat_channels)
        in_channels = feat_channels[-1]
        for _ in range(num_levels - len(feat_channels)):
            self.input_proj.append(nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, hidden_dim, 3, 2, padding=1, bias=False)),
                ("norm", nn.BatchNorm2d(hidden_dim))])))
            in_channels = hidden_dim
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout,
                                                activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList(nn.Linear(hidden_dim, num_classes)
                                            for _ in range(num_decoder_layers))
        self.dec_bbox_head = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, num_layers=3)
                                           for _ in range(num_decoder_layers))
        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                proj_feats.append(self.input_proj[i](feats[-1] if i == len_srcs else proj_feats[-1]))
        feat_flatten, spatial_shapes, level_start_index = [], [], [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])
        level_start_index.pop()
        return torch.concat(feat_flatten, 1), spatial_shapes, level_start_index

    def _generate_anchors(self, spatial_shapes, grid_size=0.05, dtype=torch.float32, device="cpu"):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs = memory.shape[0]
        anchors, valid_mask = self._generate_anchors(spatial_shapes, dtype=memory.dtype, device=memory.device)
        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        if enc_outputs_class.shape[1] < self.num_queries:
            raise ValueError("input feature maps contain fewer positions than queries")
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])).detach()
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)
        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets=None):
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets, self.num_classes, self.num_queries, self.denoising_class_embed,
                    num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale)
        else:
            denoising_class = denoising_bbox_unact = attn_mask = dn_meta = None
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        out_bboxes, out_logits = self.decoder(
            target, init_ref_points_unact, memory, spatial_shapes, level_start_index,
            self.dec_bbox_head, self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        if self.training:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b}
                                  for a, b in zip(out_logits[:-1], out_bboxes[:-1])]
            out["aux_outputs"].append({"pred_logits": enc_topk_logits, "pred_boxes": enc_topk_bboxes})
            if dn_meta is not None:
                out["dn_aux_outputs"] = [{"pred_logits": a, "pred_boxes": b}
                                         for a, b in zip(dn_out_logits, dn_out_bboxes)]
                out["dn_meta"] = dn_meta
        return out
