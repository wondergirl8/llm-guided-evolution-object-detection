"""A PyTorch 2.x compatibility port of the public EMS-ResNet34 YOLO graph.

The block equations, surrogate gradient, threshold, decay, two-scale head and
seed topology follow BICLab/EMS-YOLO commit fbfc2ca.  The public implementation
hard-codes a global time window and performs one convolution per Python loop.
This port makes T an input dimension and folds T*B for the same per-timestep
convolution, which is necessary for repeatable use by candidate_runner.py.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

THRESHOLD = 0.5
SURROGATE_LENS = 0.5
MEMBRANE_DECAY = 0.25


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value > THRESHOLD).to(value.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        window = (value - THRESHOLD).abs() < SURROGATE_LENS
        return gradient * window.to(gradient.dtype) / (2 * SURROGATE_LENS)


surrogate_spike = _SurrogateSpike.apply


class MembraneUpdate(nn.Module):
    """Stateless EMS leaky integrate-and-fire update over one input window."""

    def __init__(self, analog=False):
        super().__init__()
        self.analog = analog

    def forward(self, inputs):
        membrane = torch.zeros_like(inputs[0])
        spike = torch.zeros_like(inputs[0])
        outputs = []
        for step in range(inputs.shape[0]):
            membrane = (inputs[step] if step == 0 else
                        membrane * MEMBRANE_DECAY * (1 - spike.detach()) + inputs[step])
            spike = F.silu(membrane) if self.analog else surrogate_spike(membrane)
            outputs.append(spike)
        return torch.stack(outputs)


class TemporalConv(nn.Conv2d):
    """Apply one shared 2-D convolution independently at every T,B position."""

    def forward(self, inputs):
        time, batch, channels, height, width = inputs.shape
        output = super().forward(inputs.reshape(time * batch, channels, height, width))
        return output.reshape(time, batch, *output.shape[1:])


class TemporalBatchNorm(nn.Module):
    """BatchNorm3d over B,C,T,H,W, matching upstream EMS batch_norm_2d."""

    def __init__(self, channels, scale=THRESHOLD):
        super().__init__()
        self.norm = nn.BatchNorm3d(channels)
        nn.init.constant_(self.norm.weight, scale)
        nn.init.zeros_(self.norm.bias)

    def forward(self, inputs):
        values = inputs.permute(1, 2, 0, 3, 4).contiguous()
        return self.norm(values).permute(2, 0, 1, 3, 4).contiguous()


class InputStem(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = TemporalConv(inputs, outputs, 7, 2, 3, bias=False)
        self.norm = TemporalBatchNorm(outputs)

    def forward(self, values):
        return self.norm(self.conv(values))


class AnalogConv(nn.Module):
    def __init__(self, inputs, outputs, kernel=3):
        super().__init__()
        self.conv = TemporalConv(inputs, outputs, kernel, 1, kernel // 2, bias=False)
        self.norm = TemporalBatchNorm(outputs)
        self.activation = MembraneUpdate(analog=True)

    def forward(self, values):
        return self.activation(self.norm(self.conv(values)))


class EMSBlock(nn.Module):
    """Upstream BasicBlock_2 with an explicit shape-safe projection."""

    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.spike1 = MembraneUpdate()
        self.conv1 = TemporalConv(inputs, outputs, 3, stride, 1, bias=False)
        self.norm1 = TemporalBatchNorm(outputs)
        self.spike2 = MembraneUpdate()
        self.conv2 = TemporalConv(outputs, outputs, 3, 1, 1, bias=False)
        self.norm2 = TemporalBatchNorm(outputs, scale=0.2 * THRESHOLD)
        self.project = None
        if stride != 1 or inputs != outputs:
            self.project = nn.Sequential(MembraneUpdate(),
                                         TemporalConv(inputs, outputs, 1, stride, bias=False),
                                         TemporalBatchNorm(outputs))

    def forward(self, values):
        residual = self.norm2(self.conv2(self.spike2(self.norm1(self.conv1(self.spike1(values))))))
        shortcut = values if self.project is None else self.project(values)
        return residual + shortcut


class EMSConcatDown(nn.Module):
    """Upstream Concat_res2 transition: learn new channels, retain old channels."""

    def __init__(self, inputs, outputs):
        super().__init__()
        if outputs <= inputs:
            raise ValueError("EMS concatenating transition requires outputs > inputs")
        self.spike1 = MembraneUpdate()
        self.conv1 = TemporalConv(inputs, outputs, 3, 2, 1, bias=False)
        self.norm1 = TemporalBatchNorm(outputs)
        self.spike2 = MembraneUpdate()
        self.conv2 = TemporalConv(outputs, outputs, 3, 1, 1, bias=False)
        self.norm2 = TemporalBatchNorm(outputs, scale=0.2 * THRESHOLD)
        self.shortcut = nn.Sequential(
            MembraneUpdate(),
            TemporalConv(inputs, outputs - inputs, 1, 1, bias=False),
            TemporalBatchNorm(outputs - inputs),
        )

    def forward(self, values):
        residual = self.norm2(self.conv2(self.spike2(self.norm1(self.conv1(self.spike1(values))))))
        shortcut = torch.cat((self.shortcut(values), values), dim=2)
        time, batch, channels, height, width = shortcut.shape
        shortcut = F.max_pool2d(shortcut.reshape(time * batch, channels, height, width), 2, 2)
        shortcut = shortcut.reshape(time, batch, channels, height // 2, width // 2)
        return residual + shortcut


class EMSStage(nn.Module):
    def __init__(self, inputs, outputs, depth, stride, concat_down=False):
        super().__init__()
        if depth < 1:
            raise ValueError("EMS stage depth must be positive")
        blocks = [EMSConcatDown(inputs, outputs) if concat_down else
                  EMSBlock(inputs, outputs, stride)]
        blocks.extend(EMSBlock(outputs, outputs) for _ in range(depth - 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, values):
        return self.blocks(values)


class TemporalUpsample(nn.Module):
    def forward(self, inputs):
        time, batch, channels, height, width = inputs.shape
        output = F.interpolate(inputs.reshape(time * batch, channels, height, width),
                               scale_factor=2, mode="nearest")
        return output.reshape(time, batch, channels, height * 2, width * 2)


class EMSDetect(nn.Module):
    def __init__(self, channels, anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.outputs_per_anchor = num_classes + 5
        anchors = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anchors.reshape(len(channels), -1, 2))
        self.convs = nn.ModuleList(TemporalConv(c, self.outputs_per_anchor * self.num_anchors, 1)
                                   for c in channels)

    @property
    def num_anchors(self):
        return self.anchors.shape[1]

    def forward(self, features):
        predictions = []
        for feature, conv in zip(features, self.convs):
            value = conv(feature)
            time, batch, _, height, width = value.shape
            value = value.reshape(time, batch, self.num_anchors,
                                  self.outputs_per_anchor, height, width)
            predictions.append(value.permute(0, 1, 2, 4, 5, 3).mean(0).contiguous())
        return predictions


class EMSYolo(nn.Module):
    """EMS-ResNet backbone and two-scale YOLO head for one FRED class."""

    strides = (16, 32)

    def __init__(self, num_classes=1, base_channels=64, stage_depths=(3, 4, 6, 3),
                 neck_channels=256, anchors=((10, 14, 23, 27, 37, 58),
                                             (81, 82, 135, 169, 216, 269))):
        super().__init__()
        if num_classes != 1:
            raise ValueError("the FRED EMS-YOLO family requires one foreground class")
        widths = tuple(base_channels * 2 ** index for index in range(4))
        self.stem = InputStem(3, widths[0])
        self.stages = nn.ModuleList([
            EMSStage(widths[0], widths[0], stage_depths[0], 2),
            EMSStage(widths[0], widths[1], stage_depths[1], 2, concat_down=True),
            EMSStage(widths[1], widths[2], stage_depths[2], 2, concat_down=True),
            EMSStage(widths[2], widths[3], stage_depths[3], 2, concat_down=True),
        ])
        self.large_reduce = EMSBlock(widths[3], neck_channels)
        self.large_feature = AnalogConv(neck_channels, neck_channels * 2)
        self.medium_reduce = EMSBlock(neck_channels, neck_channels // 2)
        self.upsample = TemporalUpsample()
        self.medium_feature = AnalogConv(neck_channels // 2 + widths[2], neck_channels)
        self.detect = EMSDetect((neck_channels, neck_channels * 2), anchors, num_classes)
        # Preserve the public YOLO builder's prior: roughly eight objects per
        # 640-pixel image and a strong single-class prior. There is no separate
        # class loss when nc=1, so this also keeps score=obj*class meaningful.
        with torch.no_grad():
            for convolution, stride in zip(self.detect.convs, self.strides):
                bias = convolution.bias.reshape(self.detect.num_anchors, -1)
                bias[:, 4] += torch.log(torch.tensor(8 / (640 / stride) ** 2))
                bias[:, 5:] += torch.log(torch.tensor(0.6 / (num_classes - 0.999999)))
                convolution.bias.copy_(bias.reshape(-1))

    def forward(self, inputs):
        if inputs.ndim != 5 or inputs.shape[2] != 3:
            raise ValueError("EMS-YOLO input must have shape [B,T,3,H,W]")
        values = inputs.permute(1, 0, 2, 3, 4).contiguous()
        values = self.stem(values)
        saved = []
        for stage in self.stages:
            values = stage(values)
            saved.append(values)
        large_base = self.large_reduce(saved[-1])
        large = self.large_feature(large_base)
        medium = self.medium_reduce(large_base)
        medium = self.upsample(medium)
        if medium.shape[-2:] != saved[2].shape[-2:]:
            raise ValueError("EMS neck/backbone spatial shapes do not align")
        medium = self.medium_feature(torch.cat((medium, saved[2]), dim=2))
        return self.detect((medium, large))
