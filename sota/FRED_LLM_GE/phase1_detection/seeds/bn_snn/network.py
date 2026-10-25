"""BN-SNN: bistable spiking ResNet backbone with an SSD detection head.

This is a standalone PyTorch implementation of the architectural ideas in
"BN-SNN: Spiking neural networks with bistable neurons for object detection":

* BIF neurons use subtractive reset and phase-gated firing.
* Residual shortcuts contain synchronous BIF neurons so both paths reach the
  addition at the same temporal phase.
* SSD prediction layers operate on several spatial scales and aggregate their
  predictions over the simulated time window.

The module intentionally avoids a framework-specific SNN dependency so it can
be used by the non-evolutionary baseline before LLM-GE is introduced.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class SurrogateSpike(torch.autograd.Function):
    """Heaviside spike in the forward pass with a triangular surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane_error: Tensor, slope: float) -> Tensor:
        ctx.save_for_backward(membrane_error)
        ctx.slope = slope
        return (membrane_error >= 0).to(membrane_error.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (membrane_error,) = ctx.saved_tensors
        # The smooth derivative is only used for optimization; hardware still
        # observes the binary event emitted by the forward pass.
        gradient = ctx.slope * (1.0 - ctx.slope * membrane_error.abs()).clamp_min(0.0)
        return grad_output * gradient, None


def surrogate_spike(membrane_error: Tensor, slope: float = 1.0) -> Tensor:
    return SurrogateSpike.apply(membrane_error, slope)


class BIFNeuron(nn.Module):
    """Bistable integrate-and-fire neuron with temporal gating.

    During the active half of a phase period, accumulated membrane charge can
    emit a spike. During the inactive half, charge is retained but no spike is
    emitted. This accumulation/firing alternation suppresses redundant spikes,
    balances the activation pattern, and reduces event-driven computation.
    """

    def __init__(self, channels: int, *, threshold: float = 1.0, phase_period: int = 2, phase_offset: int = 0):
        super().__init__()
        if phase_period < 2:
            raise ValueError("phase_period must be at least 2")
        self.threshold = float(threshold)
        self.phase_period = int(phase_period)
        self.phase_offset = int(phase_offset)
        self.channels = int(channels)
        self.mem: Tensor | None = None
        self.last_spike: Tensor | None = None

    def reset_state(self) -> None:
        """Clear short-term membrane memory between independent samples/batches."""

        self.mem = None
        self.last_spike = None

    def forward(self, current: Tensor, time_index: int) -> Tensor:
        if self.mem is None or self.mem.shape != current.shape:
            self.mem = torch.zeros_like(current)
        self.mem = self.mem + current

        phase = (time_index + self.phase_offset) % self.phase_period
        active_phase = phase < (self.phase_period // 2)
        candidate = surrogate_spike(self.mem - self.threshold)
        spike = candidate if active_phase else torch.zeros_like(candidate)

        # Subtracting the threshold preserves residual membrane charge instead
        # of discarding it, which is the BIF conversion's key anti-information-
        # loss mechanism compared with a hard reset-to-zero IF neuron.
        self.mem = self.mem - spike.detach() * self.threshold
        self.last_spike = spike
        return spike


class BIFPair(nn.Module):
    """Complementary BIF neurons used as one feature activation unit."""

    def __init__(self, channels: int, *, threshold: float = 1.0, phase_period: int = 2):
        super().__init__()
        self.first = BIFNeuron(channels, threshold=threshold, phase_period=phase_period, phase_offset=0)
        self.second = BIFNeuron(channels, threshold=threshold, phase_period=phase_period, phase_offset=phase_period // 2)

    def reset_state(self) -> None:
        self.first.reset_state()
        self.second.reset_state()

    def forward(self, current: Tensor, time_index: int) -> Tensor:
        # The complementary phases let one neuron accumulate while the other
        # releases stored charge, preserving information flow at every step.
        return 0.5 * (self.first(current, time_index) + self.second(current, time_index))


class SynchronousNeuron(BIFNeuron):
    """BIF neuron placed on a residual shortcut to correct phase skew."""

    def __init__(self, channels: int, *, threshold: float = 1.0):
        # A one-step active phase avoids adding a separate temporal delay while
        # still making the shortcut spike-based and phase-compatible.
        super().__init__(channels, threshold=threshold, phase_period=2, phase_offset=1)


class SpikingResidualBlock(nn.Module):
    """Two-convolution BIF residual block with an explicit synchronous path."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, phase_period: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.neuron1 = BIFPair(out_channels, phase_period=phase_period)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.neuron2 = BIFPair(out_channels, phase_period=phase_period)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.synchronous = SynchronousNeuron(out_channels)

    def reset_state(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_state"):
                module.reset_state()

    def forward(self, x: Tensor, time_index: int) -> Tensor:
        residual = self.shortcut(x)
        # This synchronous neuron is the important shortcut block that aligns
        # the residual and convolutional paths before their elementwise sum.
        residual = self.synchronous(residual, time_index)

        y = self.neuron1(self.bn1(self.conv1(x)), time_index)
        y = self.neuron2(self.bn2(self.conv2(y)), time_index)
        return y + residual


class SpikingResNet(nn.Module):
    """Compact ResNet backbone returning three multiscale spike feature maps."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32,
                 stage_depths=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
        )
        self.stem_neuron = BIFPair(base_channels)
        if len(stage_depths) != 4 or any(type(d) is not int or d < 1 for d in stage_depths):
            raise ValueError("stage_depths must contain four positive integers")
        previous = base_channels
        for index, depth in enumerate(stage_depths):
            channels = base_channels * 2 ** index
            blocks = [SpikingResidualBlock(previous, channels, stride=1 if index == 0 else 2)]
            blocks.extend(SpikingResidualBlock(channels, channels) for _ in range(depth - 1))
            setattr(self, f"stage{index + 1}", nn.Sequential(*blocks))
            previous = channels

    def reset_state(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_state"):
                module.reset_state()

    def forward(self, x: Tensor, time_index: int) -> list[Tensor]:
        x = self.stem_neuron(self.stem(x), time_index)
        features = []
        for index in range(1, 5):
            for block in getattr(self, f"stage{index}"):
                x = block(x, time_index)
            if index > 1:
                features.append(x)
        return features


class SSDDetectionHead(nn.Module):
    """Multiscale SSD localization and classification convolutions."""

    def __init__(self, feature_channels: Iterable[int], num_classes: int, anchors_per_location: int = 4):
        super().__init__()
        self.num_classes = int(num_classes)
        self.anchors_per_location = int(anchors_per_location)
        channels = list(feature_channels)
        self.loc_heads = nn.ModuleList(nn.Conv2d(c, anchors_per_location * 4, 3, padding=1) for c in channels)
        self.cls_heads = nn.ModuleList(nn.Conv2d(c, anchors_per_location * num_classes, 3, padding=1) for c in channels)

    @staticmethod
    def _flatten_prediction(prediction: Tensor, values_per_anchor: int) -> Tensor:
        # [B,A*V,H,W] -> [B,H*W*A,V], matching SSD's anchor-major target layout.
        batch, _, height, width = prediction.shape
        prediction = prediction.view(batch, -1, values_per_anchor, height, width)
        return prediction.permute(0, 3, 4, 1, 2).reshape(batch, -1, values_per_anchor)

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        locations = [self._flatten_prediction(head(feature), 4) for head, feature in zip(self.loc_heads, features)]
        confidences = [self._flatten_prediction(head(feature), self.num_classes) for head, feature in zip(self.cls_heads, features)]
        return torch.cat(locations, dim=1), torch.cat(confidences, dim=1)


class BNSNN(nn.Module):
    """Complete temporal BN-SNN detector.

    Input shape is ``[B,T,2,H,W]`` as emitted by ``dataset.py``. The SSD head
    returns time-aggregated predictions. ``firing_rate`` is a diagnostic mean
    binary activity, not measured energy or an evolution fitness objective.
    """

    def __init__(self, num_classes: int, *, in_channels: int = 2, base_channels: int = 32, anchors_per_location: int = 4,
                 stage_depths=(2, 2, 2, 2)):
        super().__init__()
        self.backbone = SpikingResNet(in_channels=in_channels, base_channels=base_channels,
                                      stage_depths=stage_depths)
        self.head = SSDDetectionHead([base_channels * 2, base_channels * 4, base_channels * 8], num_classes, anchors_per_location)
        self._spike_sum = 0.0
        self._spike_count = 0

    def reset_state(self) -> None:
        self.backbone.reset_state()
        self._spike_sum = 0.0
        self._spike_count = 0

    def _record_spikes(self) -> None:
        for module in self.modules():
            spike = getattr(module, "last_spike", None)
            if spike is not None:
                self._spike_sum += float(spike.detach().sum())
                self._spike_count += spike.numel()

    @property
    def firing_rate(self) -> float:
        return self._spike_sum / max(self._spike_count, 1)

    def forward(self, events: Tensor) -> dict[str, Tensor | float]:
        if (events.ndim != 5 or events.shape[1] < 2 or events.shape[2] != 2
                or min(events.shape[-2:]) < 32 or not events.is_floating_point()):
            raise ValueError("events must be floating [B,T>=2,2,H>=32,W>=32]")
        self.reset_state()
        location_steps: list[Tensor] = []
        confidence_steps: list[Tensor] = []
        for time_index in range(events.shape[1]):
            features = self.backbone(events[:, time_index], time_index)
            locations, confidences = self.head(features)
            location_steps.append(locations)
            confidence_steps.append(confidences)
            self._record_spikes()
        return {
            "locations": torch.stack(location_steps, dim=1).mean(dim=1),
            "confidences": torch.stack(confidence_steps, dim=1).mean(dim=1),
            "firing_rate": self.firing_rate,
        }


__all__ = ["BIFNeuron", "BIFPair", "SynchronousNeuron", "SpikingResidualBlock", "SpikingResNet", "SSDDetectionHead", "BNSNN"]
