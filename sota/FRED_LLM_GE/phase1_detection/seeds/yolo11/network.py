"""Project-owned YOLO11 architecture knobs for the first FRED bring-up.

Only the literal GENOME below is offered to LLM-GE. The checkpoint, data,
training protocol, detection head, and evaluator live outside this file.
"""

from copy import deepcopy


# These are YOLO11 C3k2/C2PSA repeat positions in the official model YAML.
_LAYERS = {"backbone_p3": ("backbone", 4, "C3k2"),
           "attention": ("backbone", 10, "C2PSA"),
           "neck_p3": ("head", 5, "C3k2")}
_ALLOWED_REPEATS = (2, 3, 4)


def build_config(base_config, genome=None):
    """Return a YOLO11 YAML mapping with only approved architecture edits."""
    changes = GENOME if genome is None else genome
    if set(changes) != set(_LAYERS):
        raise ValueError("YOLO11 genome must contain exactly the approved genes")
    config = deepcopy(base_config)
    if config.get("task", "detect") != "detect":
        raise ValueError("checkpoint is not a detection model")
    if len(config.get("backbone", ())) != 11 or len(config.get("head", ())) != 13:
        raise ValueError("checkpoint does not match the expected YOLO11 layout")
    if config.get("head", [])[-1][2] != "Detect":
        raise ValueError("checkpoint does not have a YOLO11 Detect head")
    for name, (section, index, module) in _LAYERS.items():
        repeat = changes[name]
        if type(repeat) is not int or repeat not in _ALLOWED_REPEATS:
            raise ValueError(f"invalid {name} repeat: {repeat!r}")
        layer = config[section][index]
        if layer[2] != module or layer[1] != 2:
            raise ValueError(f"unexpected YOLO11 layer at {section}[{index}]")
        layer[1] = repeat
    config["nc"] = 1
    return config


# --OPTION--
GENOME = {
    "backbone_p3": 2,
    "attention": 2,
    "neck_p3": 2,
}
