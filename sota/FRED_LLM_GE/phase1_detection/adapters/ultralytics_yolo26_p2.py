"""YOLO26n-P2 Event-frame adapter, reusing the YOLO11 Ultralytics data path."""

import hashlib
import json
import tempfile
from pathlib import Path

import yaml

try:
    from .ultralytics_yolo11 import UltralyticsYolo11Adapter
except ImportError:
    from ultralytics_yolo11 import UltralyticsYolo11Adapter


class UltralyticsYolo26P2Adapter(UltralyticsYolo11Adapter):
    name = "yolo26_p2"

    def __init__(self, settings, fred_dir):
        super().__init__(settings, fred_dir)
        self.inference_head = settings["inference_head"]
        if self.inference_head not in ("one_to_many", "one_to_one"):
            raise ValueError("YOLO26 inference_head must be one_to_many or one_to_one")

    def _validate_candidate(self, config):
        seed = yaml.safe_load(self.variant_path("network", None).read_text())
        if not isinstance(config, dict) or set(config) != set(seed):
            raise ValueError("YOLO26n-P2 candidate must keep the seed YAML keys")
        for key in ("nc", "scale", "end2end", "reg_max", "scales"):
            if type(config[key]) is not type(seed[key]) or config[key] != seed[key]:
                raise ValueError(f"YOLO26n-P2 {key} is fixed")
        for section in ("backbone", "head"):
            layers, reference = config[section], seed[section]
            if not isinstance(layers, list) or len(layers) != len(reference):
                raise ValueError(f"YOLO26n-P2 {section} must keep its layer count")
            for index, (layer, original) in enumerate(zip(layers, reference)):
                if not isinstance(layer, list) or len(layer) != 4:
                    raise ValueError(f"YOLO26n-P2 {section} layer {index} is malformed")
                source, repeats, module, args = layer
                old_source, old_repeats, old_module, old_args = original
                if source != old_source or module != old_module:
                    raise ValueError(f"YOLO26n-P2 {section} layer {index} routing and module are fixed")
                if module not in ("Conv", "C3k2"):
                    if layer != original:
                        raise ValueError(f"YOLO26n-P2 {section} layer {index} is fixed")
                    continue
                if not isinstance(args, list) or len(args) != len(old_args):
                    raise ValueError(f"YOLO26n-P2 {section} layer {index} arguments are malformed")
                channels = args[0]
                if (type(channels) is not int or channels % 64
                        or not max(64, old_args[0] // 2) <= channels <= old_args[0] * 2):
                    raise ValueError(f"YOLO26n-P2 {section} layer {index} channels are out of bounds")
                if module == "Conv":
                    if (type(repeats) is not int or repeats != old_repeats
                            or type(args[1]) is not int or args[1] not in (1, 3, 5)
                            or type(args[2]) is not int or args[2] != old_args[2]):
                        raise ValueError(f"YOLO26n-P2 {section} layer {index} stride/repeats are fixed")
                elif (type(repeats) is not int or not 1 <= repeats <= 4
                      or args[1:] != old_args[1:]):
                    raise ValueError(f"YOLO26n-P2 {section} layer {index} C3k2 arguments are fixed")

    @staticmethod
    def architecture_hash(variant_path):
        """Hash parsed YAML so formatting and comments do not create new designs."""
        config = yaml.safe_load(Path(variant_path).read_text())
        return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def preflight(self, variant_path, parent_path):
        """Validate one focused edit and construct the model on CPU before GPU work."""
        config = yaml.safe_load(Path(variant_path).read_text())
        self._validate_candidate(config)
        parent = yaml.safe_load(Path(parent_path).read_text())
        self._validate_candidate(parent)
        changed_layers = sum(
            layer != previous
            for section in ("backbone", "head")
            for layer, previous in zip(config[section], parent[section])
        )
        if changed_layers != 1:
            raise ValueError(
                f"YOLO26 proposal must change exactly one layer relative to its parent; "
                f"found {changed_layers}"
            )

        from ultralytics import YOLO

        with tempfile.TemporaryDirectory(prefix="yolo26_p2_preflight_") as directory:
            model_yaml = Path(directory) / "yolo26n-p2.yaml"
            model_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
            YOLO(str(model_yaml), task="detect", verbose=False)
        return self.architecture_hash(variant_path)

    def build(self, variant_path, num_classes, device):
        from ultralytics import YOLO

        config = yaml.safe_load(Path(variant_path).read_text())
        self._validate_candidate(config)
        config["nc"] = num_classes
        self.workdir = Path(tempfile.mkdtemp(prefix="yolo26_p2_candidate_"))
        model_yaml = self.workdir / "yolo26n-p2.yaml"
        model_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
        # The inherited trainer uses pretrained=False, matching YOLO11 candidate runs.
        return YOLO(str(model_yaml), task="detect")

    def training_overrides(self):
        # Only the standalone ablation runner supplies these controlled settings.
        return self.settings.get("training_overrides", {})

    def prediction_overrides(self):
        # Ultralytics 8.4.175 selects the NMS-free one-to-one head only with
        # nms=False. Pin both modes explicitly so scoring matches report metadata.
        return {"nms": self.inference_head == "one_to_many"}

    def postprocessing(self, proxy):
        if self.inference_head == "one_to_one":
            return (f"ultralytics YOLO26 head=one_to_one nms=False "
                    f"imgsz={proxy['img_width']} conf={proxy['eval_conf']}; no NMS")
        return (f"ultralytics YOLO26 head=one_to_many nms=True "
                f"imgsz={proxy['img_width']} conf={proxy['eval_conf']} "
                f"nms_iou={proxy['nms_iou']}")
