"""YOLO26n-P2 Event-frame adapter, reusing the YOLO11 Ultralytics data path."""

import tempfile
from pathlib import Path

import yaml

try:
    from .ultralytics_yolo11 import UltralyticsYolo11Adapter
except ImportError:
    from ultralytics_yolo11 import UltralyticsYolo11Adapter


class UltralyticsYolo26P2Adapter(UltralyticsYolo11Adapter):
    name = "yolo26_p2"

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

    def postprocessing(self, proxy):
        return (f"ultralytics YOLO26 end2end predict imgsz={proxy['img_width']} "
                f"conf={proxy['eval_conf']}; no NMS")
