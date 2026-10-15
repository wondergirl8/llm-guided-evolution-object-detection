"""Interface each model family implements for candidate_runner.py.

The runner owns everything shared across families: the protocol, data,
checkpoint path, scoring and failure handling. A family adapter only builds,
trains and runs its own models, so each family keeps its framework's loss and
training loop while being scored the same way as every other family.
"""

from pathlib import Path

import torch


class FamilyAdapter:
    """Base class for a model family; subclasses implement build/train/predict."""

    name = ""

    def __init__(self, settings, fred_dir):
        """settings: this family's entry under `families` in protocol.yaml."""
        self.settings = settings
        self.fred_dir = Path(fred_dir)

    def variant_path(self, model_name, variant_dir):
        """File defining the candidate: the seed for "network", else network_<id>."""
        if model_name == "network":
            return self.fred_dir / self.settings["seed_file"]
        return Path(variant_dir) / f"{model_name}.py"

    def build(self, variant_path, num_classes, device):
        """Construct the candidate model from its file."""
        raise NotImplementedError

    def train(self, model, data, proxy, device):
        """Train in place on the proxy training split (dict from proxy_data.load_split)."""
        raise NotImplementedError

    def predict(self, model, images, proxy, device):
        """Detect on a uint8 [B, 3, H, W] batch.

        Returns one (boxes, scores) pair per image: xyxy boxes in the pixels of
        the given images and confidences in [0, 1], after the family's own NMS
        at proxy["eval_conf"] and proxy["nms_iou"].
        """
        raise NotImplementedError

    def postprocessing(self, proxy):
        """Short description of inference postprocessing for report metadata."""
        raise NotImplementedError

    def param_count(self, model):
        return sum(p.numel() for p in model.parameters())

    def save(self, model, path):
        torch.save(model.state_dict(), path)
