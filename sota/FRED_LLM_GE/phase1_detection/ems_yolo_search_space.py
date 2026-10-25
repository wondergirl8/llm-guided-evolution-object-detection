"""Validated Python-assignment genome for the experimental EMS-YOLO family."""
import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch

from .result_contract import DetectionResult, write_results
from .seeds.ems_yolo.network import EMSYolo

SEED = Path(__file__).resolve().parents[1] / "seeds/seed_ems_yolo.py"
NAMES = ("BASE_CHANNELS", "STAGE_DEPTHS", "NECK_CHANNELS")


def load_and_validate(path):
    source = Path(path).read_text()
    parts = source.split("# --OPTION--")
    if len(parts) != len(NAMES) + 1:
        raise ValueError("require exactly three mutable assignment blocks")
    for part in parts[1:]:
        if len(ast.parse(part).body) != 1:
            raise ValueError("each EMS mutable block must contain exactly one assignment")
    tree, reference = ast.parse(source), ast.parse(SEED.read_text())
    if (len(tree.body) != len(reference.body)
            or any(ast.dump(a) != ast.dump(b)
                   for a, b in zip(tree.body[:-len(NAMES)], reference.body[:-len(NAMES)]))):
        raise ValueError("protected EMS imports and build_model must equal the seed")
    values = {}
    for node, name in zip(tree.body[-len(NAMES):], NAMES):
        if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != name):
            raise ValueError(f"expected one literal assignment to {name}")
        values[name] = ast.literal_eval(node.value)
    base, depths, neck = (values[name] for name in NAMES)
    if type(base) is not int or base not in (48, 64, 80):
        raise ValueError("BASE_CHANNELS must be one of 48, 64, 80")
    if (type(depths) is not tuple or len(depths) != 4
            or any(type(value) is not int or not 1 <= value <= 8 for value in depths)
            or not 6 <= sum(depths) <= 24):
        raise ValueError("STAGE_DEPTHS must contain four values in 1..8 with total depth 6..24")
    if type(neck) is not int or neck not in (128, 192, 256, 320):
        raise ValueError("NECK_CHANNELS must be one of 128, 192, 256, 320")
    return values


def build(path, num_classes=1):
    cfg = load_and_validate(path)
    return EMSYolo(num_classes=num_classes, base_channels=cfg["BASE_CHANNELS"],
                   stage_depths=cfg["STAGE_DEPTHS"], neck_channels=cfg["NECK_CHANNELS"])


@lru_cache(maxsize=1)
def seed_parameter_count():
    with torch.random.fork_rng(devices=[]):
        return sum(parameter.numel() for parameter in build(SEED).parameters())


def built_architecture(path):
    cfg = load_and_validate(path)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = build(path).eval()
        outputs = model(torch.zeros(1, 4, 3, 64, 64))
        expected = ((1, 3, 4, 4, 6), (1, 3, 2, 2, 6))
        if tuple(tuple(value.shape) for value in outputs) != expected:
            raise ValueError(f"EMS detection stride/head contract violated: "
                             f"{[tuple(value.shape) for value in outputs]}")
        if not all(torch.isfinite(value).all() for value in outputs):
            raise ValueError("EMS preflight produced nonfinite predictions")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        seed_parameters = seed_parameter_count()
        if not 0.5 * seed_parameters <= parameters <= 2.0 * seed_parameters:
            raise ValueError(f"parameter count {parameters} outside 0.5x--2x seed range "
                             f"[{0.5 * seed_parameters:.0f}, {2 * seed_parameters:.0f}]")
        shapes = [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
    identity = json.dumps([cfg, shapes], sort_keys=True).encode()
    return hashlib.sha256(identity).hexdigest(), parameters


def differs_from_parents(child, *parents):
    graph, _ = built_architecture(child)
    parent_graphs = [built_architecture(parent)[0] for parent in parents]
    if graph in parent_graphs:
        return False
    child_cfg = load_and_validate(child)
    if len(parents) == 1:
        parent_cfg = load_and_validate(parents[0])
        assignments = sum(child_cfg[name] != parent_cfg[name] for name in NAMES)
        depth_changes = sum(a != b for a, b in zip(child_cfg["STAGE_DEPTHS"],
                                                   parent_cfg["STAGE_DEPTHS"]))
        if assignments != 1 or depth_changes > 1:
            raise ValueError("mutation must change one assignment or one stage-depth entry")
    return True


def retry_prompt(prompt, previous_error, attempt):
    if previous_error:
        return (f"{prompt}\nPrevious proposal rejected: {previous_error}\n"
                "Correct only the supplied assignment and return a different valid value.")
    return prompt


def evaluation_cache_key(architecture_hash, fingerprint):
    value = json.dumps([architecture_hash, fingerprint], sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def write_cached_fitness(sota_root, gene_id, fitness):
    write_results(sota_root, gene_id,
                  DetectionResult(fitness[0], fitness[1], int(fitness[2])))
