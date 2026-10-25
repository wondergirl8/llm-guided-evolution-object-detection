"""Frozen SSD geometry and multibox loss for the experimental FRED seed."""
import math

import torch
from torch.nn import functional as F
from torchvision.ops import box_iou


def anchors_for(height, width, sizes, device):
    """Normalized cxcywh anchors in the head's level/y/x/anchor order."""
    levels = []
    for stride, size in zip((4, 8, 16), sizes):
        h, w = math.ceil(height / stride), math.ceil(width / stride)
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        centers = torch.stack(((x + 0.5) / w, (y + 0.5) / h), -1).reshape(-1, 1, 2)
        wh = torch.tensor([(size * math.sqrt(r) / width, size / math.sqrt(r) / height)
                           for r in (0.5, 1.0, 2.0, 4.0)], device=device)
        levels.append(torch.cat((centers.expand(-1, 4, -1), wh.expand(h * w, -1, -1)), -1).reshape(-1, 4))
    return torch.cat(levels)


def xyxy(anchors):
    return torch.cat((anchors[..., :2] - anchors[..., 2:] / 2,
                      anchors[..., :2] + anchors[..., 2:] / 2), -1)


def decode(locations, anchors):
    centers = anchors[:, :2] + locations[..., :2] * 0.1 * anchors[:, 2:]
    sizes = anchors[:, 2:] * (locations[..., 2:] * 0.2).clamp(-8, 8).exp()
    return torch.cat((centers - sizes / 2, centers + sizes / 2), -1).clamp(0, 1)


def multibox_loss(output, anchors, targets):
    """IoU>=0.5 positives + unique forced matches; 3:1 hard negatives.

    targets contain normalized xyxy boxes for the one foreground class. Empty
    images use three negatives so they still teach background suppression.
    """
    locations, logits = output['locations'], output['confidences']
    if locations.shape[1:] != (len(anchors), 4) or logits.shape[1:] != (len(anchors), 2):
        raise ValueError('SSD anchors and predictions disagree')
    losses = []
    for loc, cls, boxes in zip(locations, logits, targets):
        labels = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
        encoded = torch.zeros_like(loc)
        if len(boxes):
            if (not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any()
                    or (boxes < 0).any() or (boxes > 1).any()):
                raise ValueError('invalid normalized target boxes')
            overlaps = box_iou(xyxy(anchors), boxes)
            best_iou, matches = overlaps.max(1)
            # Force a different anchor for each target, even for tiny drones.
            available = torch.ones(len(anchors), dtype=torch.bool, device=anchors.device)
            for target in range(len(boxes)):
                if not available.any():
                    raise ValueError('more targets than anchors')
                index = overlaps[:, target].masked_fill(~available, -1).argmax()
                matches[index], best_iou[index], available[index] = target, 1, False
            labels[best_iou >= 0.5] = 1
            matched = boxes[matches]
            center = (matched[:, :2] + matched[:, 2:]) / 2
            size = matched[:, 2:] - matched[:, :2]
            encoded = torch.cat(((center - anchors[:, :2]) / anchors[:, 2:] / 0.1,
                                 (size / anchors[:, 2:]).log() / 0.2), -1)
        positive = labels == 1
        npos = int(positive.sum())
        classification = F.cross_entropy(cls, labels, reduction='none')
        nneg = min(3 * max(npos, 1), int((~positive).sum()))
        negative_loss = classification.masked_fill(positive, -torch.inf).topk(nneg).values.sum()
        regression = F.smooth_l1_loss(loc[positive], encoded[positive], reduction='sum')
        losses.append((classification[positive].sum() + negative_loss + regression) / max(npos, 1))
    return torch.stack(losses).mean()
