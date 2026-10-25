"""YOLO target assignment, loss, and decoding for the EMS two-scale head."""
from __future__ import annotations

import math

import torch
from torch import nn


def xywh_to_xyxy(boxes):
    output = boxes.clone()
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return output


def bbox_ciou(first, second, eps=1e-7):
    first_xyxy, second_xyxy = xywh_to_xyxy(first), xywh_to_xyxy(second)
    intersection = ((torch.minimum(first_xyxy[:, 2:], second_xyxy[:, 2:]) -
                     torch.maximum(first_xyxy[:, :2], second_xyxy[:, :2])).clamp(0).prod(1))
    area1 = (first_xyxy[:, 2:] - first_xyxy[:, :2]).clamp(0).prod(1)
    area2 = (second_xyxy[:, 2:] - second_xyxy[:, :2]).clamp(0).prod(1)
    union = area1 + area2 - intersection + eps
    iou = intersection / union
    enclosing = (torch.maximum(first_xyxy[:, 2:], second_xyxy[:, 2:]) -
                 torch.minimum(first_xyxy[:, :2], second_xyxy[:, :2]))
    center_distance = ((first[:, :2] - second[:, :2]) ** 2).sum(1)
    c2 = (enclosing ** 2).sum(1) + eps
    v = (4 / math.pi ** 2) * (torch.atan(second[:, 2] / (second[:, 3] + eps)) -
                              torch.atan(first[:, 2] / (first[:, 3] + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_distance / c2 - alpha * v


class EMSYoloLoss:
    """Single-class form of upstream ComputeLoss with its anchor assignment."""

    def __init__(self, model, settings):
        self.model = model
        self.box_weight = settings["box"]
        self.object_weight = settings["obj"]
        self.anchor_threshold = settings["anchor_t"]
        self.object_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([settings["obj_pw"]], device=next(model.parameters()).device))
        self.balance = (4.0, 1.0)

    def __call__(self, predictions, targets):
        box_loss = torch.zeros((), device=targets.device)
        object_loss = torch.zeros((), device=targets.device)
        assignments = self._targets(predictions, targets)
        for level, prediction in enumerate(predictions):
            batch, anchor, row, column, target_box, matched_anchor = assignments[level]
            target_object = torch.zeros_like(prediction[..., 4])
            if len(batch):
                selected = prediction[batch, anchor, row, column]
                center = selected[:, :2].sigmoid() * 2 - 0.5
                size = (selected[:, 2:4].sigmoid() * 2) ** 2 * matched_anchor
                iou = bbox_ciou(torch.cat((center, size), 1), target_box)
                box_loss = box_loss + (1 - iou).mean()
                target_object[batch, anchor, row, column] = iou.detach().clamp(0)
            object_loss = (object_loss
                           + self.object_loss(prediction[..., 4], target_object) * self.balance[level])
        return (box_loss * self.box_weight + object_loss * self.object_weight) * predictions[0].shape[0]

    def _targets(self, predictions, targets):
        assignments = []
        offsets = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                               dtype=torch.float32, device=targets.device) * 0.5
        for level, prediction in enumerate(predictions):
            height, width = prediction.shape[2:4]
            anchors = self.model.detect.anchors[level].to(targets.device)
            anchors_grid = anchors / self.model.strides[level]
            if not len(targets):
                empty = torch.empty(0, dtype=torch.long, device=targets.device)
                assignments.append((empty, empty, empty, empty,
                                    torch.empty(0, 4, device=targets.device),
                                    torch.empty(0, 2, device=targets.device)))
                continue
            gain = torch.tensor([1, 1, width, height, width, height], device=targets.device)
            scaled = targets * gain
            candidates = []
            for anchor_index, anchor in enumerate(anchors_grid):
                ratio = scaled[:, 4:6] / anchor
                valid = torch.maximum(ratio, 1 / ratio).amax(1) < self.anchor_threshold
                for target in scaled[valid]:
                    center = target[2:4]
                    inverse = torch.tensor([width, height], device=targets.device) - center
                    mask = torch.stack(
                        (torch.ones((), dtype=torch.bool, device=targets.device),
                         (center[0] % 1 < 0.5) & (center[0] > 1),
                         (center[1] % 1 < 0.5) & (center[1] > 1),
                         (inverse[0] % 1 < 0.5) & (inverse[0] > 1),
                         (inverse[1] % 1 < 0.5) & (inverse[1] > 1)))
                    for offset in offsets[mask]:
                        cell = (center - offset).long()
                        candidates.append((target, anchor_index, cell, offset, anchor))
            if candidates:
                batch = torch.stack([item[0][0] for item in candidates]).long()
                anchor_index = torch.tensor([item[1] for item in candidates], device=targets.device)
                cells = torch.stack([item[2] for item in candidates])
                columns = cells[:, 0].clamp(0, width - 1)
                rows = cells[:, 1].clamp(0, height - 1)
                target_boxes = torch.stack([
                    torch.cat((item[0][2:4] - item[2], item[0][4:6])) for item in candidates])
                matched = torch.stack([item[4] for item in candidates])
            else:
                batch = anchor_index = rows = columns = torch.empty(
                    0, dtype=torch.long, device=targets.device)
                target_boxes = torch.empty(0, 4, device=targets.device)
                matched = torch.empty(0, 2, device=targets.device)
            assignments.append((batch, anchor_index, rows, columns, target_boxes, matched))
        return assignments


def decode_predictions(model, predictions):
    """Return pixel xyxy boxes, objectness scores, and class scores at model input size."""
    decoded = []
    for level, prediction in enumerate(predictions):
        batch, anchors, height, width, _ = prediction.shape
        y, x = torch.meshgrid(torch.arange(height, device=prediction.device),
                              torch.arange(width, device=prediction.device), indexing="ij")
        grid = torch.stack((x, y), -1).reshape(1, 1, height, width, 2)
        anchor = model.detect.anchors[level].reshape(1, anchors, 1, 1, 2)
        values = prediction.sigmoid()
        center = (values[..., :2] * 2 - 0.5 + grid) * model.strides[level]
        size = (values[..., 2:4] * 2) ** 2 * anchor
        boxes = xywh_to_xyxy(torch.cat((center, size), -1)).reshape(batch, -1, 4)
        scores = (values[..., 4] * values[..., 5]).reshape(batch, -1)
        decoded.append((boxes, scores))
    return torch.cat([item[0] for item in decoded], 1), torch.cat([item[1] for item in decoded], 1)
