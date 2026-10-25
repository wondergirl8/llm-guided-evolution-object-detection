"""Scratch-trained EMS-YOLO adapter for released FRED event frames."""
import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

from .base import FamilyAdapter
from ..ems_yolo_search_space import build, built_architecture
from ..seeds.ems_yolo.detection import EMSYoloLoss, decode_predictions


class EMSYoloAdapter(FamilyAdapter):
    name = "ems_yolo"

    def __init__(self, settings, fred_dir):
        super().__init__(settings, fred_dir)
        recipe = settings["training"]
        if settings.get("initialization") != "scratch":
            raise ValueError("EMS-YOLO preliminary comparison requires scratch initialization")
        if recipe["optimizer"] != "SGD":
            raise ValueError("EMS-YOLO adapter currently implements the frozen SGD recipe")
        if (recipe["input_size"] % 32 or recipe["time_steps"] < 2
                or recipe["microbatch_size"] < 1 or recipe["epochs"] < 1):
            raise ValueError("invalid EMS-YOLO input, temporal, microbatch, or epoch settings")
        if recipe["augmentation"] != {"fliplr": 0.5}:
            raise ValueError("EMS-YOLO preliminary protocol freezes augmentation to fliplr=0.5")
        if recipe["ema"] is not False or recipe["mosaic"] is not False:
            raise ValueError("EMS-YOLO preliminary protocol does not implement EMA or mosaic")

    def build(self, variant_path, num_classes, device):
        built_architecture(variant_path)
        return build(variant_path, num_classes).to(device)

    def _letterbox(self, images, boxes, device):
        """Resize with preserved aspect ratio and convert targets to normalized xywh."""
        size = self.settings["training"]["input_size"]
        images = images.to(device=device, dtype=torch.float32) / 255
        _, _, height, width = images.shape
        scale = min(size / width, size / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        images = F.interpolate(images, (resized_height, resized_width), mode="bilinear",
                               align_corners=False)
        left, top = (size - resized_width) // 2, (size - resized_height) // 2
        images = F.pad(images, (left, size - resized_width - left,
                                top, size - resized_height - top), value=114 / 255)
        targets = []
        for batch_index, sample_boxes in enumerate(boxes):
            if not len(sample_boxes):
                continue
            value = sample_boxes.to(device=device, dtype=torch.float32).clone()
            value[:, [0, 2]] = value[:, [0, 2]] * scale + left
            value[:, [1, 3]] = value[:, [1, 3]] * scale + top
            xywh = torch.empty_like(value)
            xywh[:, :2] = (value[:, :2] + value[:, 2:]) / (2 * size)
            xywh[:, 2:] = (value[:, 2:] - value[:, :2]) / size
            targets.append(torch.cat((torch.full((len(value), 1), batch_index, device=device),
                                      torch.zeros((len(value), 1), device=device), xywh), 1))
        target = torch.cat(targets) if targets else torch.empty(0, 6, device=device)
        return images, target, (scale, left, top, width, height)

    def _temporal(self, images):
        return images[:, None].expand(-1, self.settings["training"]["time_steps"], -1, -1, -1)

    def _optimizer(self, model):
        recipe = self.settings["training"]
        batch_norm, weights, biases = [], [], []
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm3d) and module.weight is not None:
                batch_norm.append(module.weight)
            elif hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                weights.append(module.weight)
            if hasattr(module, "bias") and isinstance(module.bias, torch.nn.Parameter):
                biases.append(module.bias)
        optimizer = torch.optim.SGD(batch_norm, lr=recipe["lr0"],
                                    momentum=recipe["momentum"], nesterov=True)
        optimizer.add_param_group({"params": weights, "weight_decay": recipe["weight_decay"]})
        optimizer.add_param_group({"params": biases})
        return optimizer

    def train(self, model, data, proxy, device):
        recipe = self.settings["training"]
        device = torch.device(device)
        random.seed(proxy["seed"])
        np.random.seed(proxy["seed"])
        torch.manual_seed(proxy["seed"])
        generator = torch.Generator().manual_seed(proxy["seed"])
        optimizer = self._optimizer(model)
        loss_function = EMSYoloLoss(model, recipe["loss"])
        amp_enabled = recipe["amp"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        nominal = recipe["nominal_batch_size"]
        accumulate_groups = max(round(nominal / proxy["batch_size"]), 1)
        microbatch = recipe["microbatch_size"]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        groups_since_step = 0
        for epoch in range(proxy["epochs"]):
            progress = epoch / max(proxy["epochs"] - 1, 1)
            cosine = recipe["lrf"] + (1 - recipe["lrf"]) * (1 + math.cos(math.pi * progress)) / 2
            warmup = min(1.0, (epoch + 1) / max(recipe["warmup_epochs"], 1e-9))
            learning_rate = recipe["lr0"] * cosine * warmup
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
                if "momentum" in group:
                    group["momentum"] = (recipe["warmup_momentum"] +
                                         warmup * (recipe["momentum"] - recipe["warmup_momentum"]))
            order = torch.randperm(len(data["images"]), generator=generator)
            epoch_loss = 0.0
            for group in order.split(proxy["batch_size"]):
                group_loss = 0.0
                for indices in group.split(microbatch):
                    images = data["images"][indices]
                    sample_boxes = [data["boxes"][int(index)] for index in indices]
                    images, targets, _ = self._letterbox(images, sample_boxes, device)
                    flip = torch.rand(len(images), generator=generator) < recipe["augmentation"]["fliplr"]
                    if flip.any():
                        flip_device = flip.to(device)
                        images[flip_device] = images[flip_device].flip(-1)
                        for local_index in torch.where(flip_device)[0]:
                            selected = targets[:, 0] == local_index
                            targets[selected, 2] = 1 - targets[selected, 2]
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        loss = loss_function(model(self._temporal(images)), targets)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite EMS-YOLO training loss")
                    scaled = loss * len(indices) / (len(group) * accumulate_groups)
                    scaler.scale(scaled).backward()
                    group_loss += float(loss.detach()) * len(indices)
                epoch_loss += group_loss
                groups_since_step += 1
                if groups_since_step == accumulate_groups:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"],
                                                   error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    groups_since_step = 0
            if groups_since_step:
                scaler.unscale_(optimizer)
                # The final epoch fragment contains fewer groups than the nominal
                # accumulation window. Restore its mean-gradient scale before step.
                correction = accumulate_groups / groups_since_step
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"],
                                               error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                groups_since_step = 0
            print(f'EMS-YOLO epoch {epoch + 1}/{proxy["epochs"]}: '
                  f'loss={epoch_loss / len(order):.6f} lr={learning_rate:.6g}', flush=True)

    @torch.no_grad()
    def predict(self, model, images, proxy, device):
        device = torch.device(device)
        model.eval()
        results = []
        for chunk in images.split(self.settings["training"]["microbatch_size"]):
            inputs, _, geometry = self._letterbox(chunk, [torch.empty(0, 4)] * len(chunk), device)
            predictions = model(self._temporal(inputs))
            boxes, scores = decode_predictions(model, predictions)
            scale, left, top, width, height = geometry
            for sample_boxes, sample_scores in zip(boxes, scores):
                valid = sample_scores >= proxy["eval_conf"]
                sample_boxes, sample_scores = sample_boxes[valid], sample_scores[valid]
                sample_boxes[:, [0, 2]] = (sample_boxes[:, [0, 2]] - left) / scale
                sample_boxes[:, [1, 3]] = (sample_boxes[:, [1, 3]] - top) / scale
                sample_boxes[:, [0, 2]] = sample_boxes[:, [0, 2]].clamp(0, width)
                sample_boxes[:, [1, 3]] = sample_boxes[:, [1, 3]].clamp(0, height)
                valid = (sample_boxes[:, 2:] > sample_boxes[:, :2]).all(1)
                sample_boxes, sample_scores = sample_boxes[valid], sample_scores[valid]
                keep = nms(sample_boxes, sample_scores, proxy["nms_iou"])
                keep = keep[:self.settings["max_detections"]]
                results.append((sample_boxes[keep].cpu(), sample_scores[keep].cpu()))
        return results

    def postprocessing(self, proxy):
        return (f'EMS two-scale sigmoid score; conf={proxy["eval_conf"]}; '
                f'NMS IoU={proxy["nms_iou"]}; max={self.settings["max_detections"]}')
