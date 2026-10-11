"""Actual Ultralytics YOLO11, randomly initialized, with two mutable classes.

The protected wrapper retains the official Detect head and detection loss.
The initial backbone/neck execute the official YOLO11 graph unchanged. LLM-GE
can rewrite their PyTorch implementations while preserving P3/P4/P5 interfaces.
"""

from types import SimpleNamespace

import torch
from torch import nn
from ultralytics.nn.modules import C2PSA, C3k2, Conv, SPPF
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.nms import non_max_suppression


class EvolvableDetectionModel(DetectionModel):
    """Group the official graph into backbone, neck, and protected Detect head."""

    def __init__(self, num_classes=1, scale="m"):
        if num_classes != 1 or scale not in ("n", "s", "m", "l", "x"):
            raise ValueError("Expected drone class 0 and a YOLO11 scale n/s/m/l/x")
        config = yaml_model_load(f"yolo11{scale}.yaml")
        super().__init__(cfg=config, ch=3, nc=num_classes, verbose=False)
        layers = list(self.model.children())
        if len(layers) != 24:
            raise ValueError("Pinned YOLO11 graph must have 24 layers")
        self.reference_params = sum(p.numel() for p in self.parameters())
        self.model = nn.Sequential(
            YOLO11Backbone(layers[:11]), YOLO11Neck(layers[11:23]), layers[23]
        )
        self.args = SimpleNamespace(**DEFAULT_CFG_DICT)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        # DetectionModel initializes head strides before grouping the graph.
        if len(self.model) != 3:
            return super()._predict_once(x, profile, visualize, embed)
        return self.model[2](self.model[1](self.model[0](x)))


class DroneDetector(nn.Module):
    """Protected entry point. No checkpoint download or weight inheritance."""

    def __init__(self, num_classes=1, scale="m", confidence_threshold=0.05):
        super().__init__()
        self.detector = EvolvableDetectionModel(num_classes, scale)
        self.confidence_threshold = confidence_threshold

    def forward(self, event):
        return self.detector(event)

    def loss(self, batch):
        return self.detector.loss(batch)

    @torch.no_grad()
    def predict(self, event):
        if self.training:
            raise ValueError("Call eval() before predict()")
        raw = self(event)
        if not torch.isfinite(raw[0]).all():
            raise ValueError("Non-finite YOLO11 predictions")
        decoded = non_max_suppression(
            raw, conf_thres=self.confidence_threshold, iou_thres=0.5,
            nc=1, max_det=100, max_time_img=float("inf"),
        )
        return [{"boxes": d[:, :4], "scores": d[:, 4], "labels": d[:, 5].long()}
                for d in decoded]


# --OPTION--
class YOLO11Backbone(nn.Module):
    """Official YOLO11 layers 0..10; return P3/8, P4/16, P5/32.

    layers: Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv,
    C3k2, SPPF, C2PSA. Each is a real, randomly initialized Ultralytics
    module. LLM edits may add/replace blocks or alter their computation.
    Preserve the output channels of layers 4, 6 and 10 and their strides.
    A C3k2 output width can be read as layers[index].cv2.conv.out_channels.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, event):
        features = []
        x = event
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in (4, 6, 10):
                features.append(x)
        return features


# --OPTION--
class YOLO11Neck(nn.Module):
    """Official YOLO11 layers 11..22; return Detect's P3, P4, P5 inputs.

    Local indices: 0 Upsample, 1 Concat, 2 C3k2, 3 Upsample, 4 Concat,
    5 C3k2, 6 Conv, 7 Concat, 8 C3k2, 9 Conv, 10 Concat, 11 C3k2.
    LLM edits may create new PyTorch blocks and feature-processing paths.
    Preserve the output channels of C3k2 layers 5/8/11 and strides 8/16/32.
    Read output widths as layers[index].cv2.conv.out_channels; do not hardcode
    widths because the same class supports different official YOLO11 scales.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, features):
        p3, p4, p5 = features
        m = self.layers
        middle = m[2](m[1]([m[0](p5), p4]))
        small = m[5](m[4]([m[3](middle), p3]))
        medium = m[8](m[7]([m[6](small), middle]))
        large = m[11](m[10]([m[9](medium), p5]))
        return [small, medium, large]
