"""Restricted Python genome policy. Never execute LLM-authored code."""
import ast
import hashlib
import json
from pathlib import Path

import torch

from .result_contract import DetectionResult, write_results
from .seeds.bn_snn.network import BNSNN

SEED = Path(__file__).resolve().parents[1] / 'seeds/seed_bn_snn.py'


def load_and_validate(path):
    source = Path(path).read_text()
    parts = source.split('# --OPTION--')
    if len(parts) != 3:
        raise ValueError('require exactly two mutable assignment blocks')
    for part in parts[1:]:
        if len(ast.parse(part).body) != 1:
            raise ValueError('each mutable block must contain exactly one assignment')
    tree = ast.parse(source)
    reference = ast.parse(SEED.read_text())
    if len(tree.body) != len(reference.body) or any(
            ast.dump(a) != ast.dump(b) for a, b in zip(tree.body[:-2], reference.body[:-2])):
        raise ValueError('protected imports and build_model must equal the seed')
    values = {}
    for node, name in zip(tree.body[-2:], ('BASE_CHANNELS', 'STAGE_DEPTHS')):
        if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != name):
            raise ValueError(f'expected a literal assignment to {name}')
        values[name] = ast.literal_eval(node.value)
    width, depths = values['BASE_CHANNELS'], values['STAGE_DEPTHS']
    if type(width) is not int or width not in range(16, 49, 8):
        raise ValueError('BASE_CHANNELS must be one of 16, 24, 32, 40, 48')
    if (type(depths) is not tuple or len(depths) != 4
            or any(type(d) is not int or d not in (1, 2, 3) for d in depths)):
        raise ValueError('STAGE_DEPTHS must be a four-integer tuple, each in 1..3')
    return values


def build(path, num_classes=1):
    if num_classes != 1:
        raise ValueError('BN-SNN FRED adapter requires one foreground class')
    cfg = load_and_validate(path)
    return BNSNN(2, base_channels=cfg['BASE_CHANNELS'], stage_depths=cfg['STAGE_DEPTHS'])


def built_architecture(path):
    cfg = load_and_validate(path)
    # Preflight must not change initialization randomness in the calling process.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = build(path).eval()
        output = model(torch.zeros(1, 2, 2, 32, 48))
        # Three scales at strides 4, 8, 16, four anchors per spatial location.
        anchors = 4 * (8 * 12 + 4 * 6 + 2 * 3)
        if output['locations'].shape != (1, anchors, 4) or output['confidences'].shape != (1, anchors, 2):
            raise ValueError('SSD output shape contract violated')
        if not all(torch.isfinite(output[k]).all() for k in ('locations', 'confidences')):
            raise ValueError('nonfinite preflight output')
        graph = [(name, type(layer).__name__, repr(layer.extra_repr()))
                 for name, layer in model.named_modules()]
        params = sum(p.numel() for p in model.parameters())
    identity = json.dumps([cfg, graph], sort_keys=True).encode()
    return hashlib.sha256(identity).hexdigest(), params


def differs_from_parents(child, *parents):
    graph, _ = built_architecture(child)
    if len(parents) == 1:
        a, b = load_and_validate(child), load_and_validate(parents[0])
        changes = int(a['BASE_CHANNELS'] != b['BASE_CHANNELS']) + sum(
            x != y for x, y in zip(a['STAGE_DEPTHS'], b['STAGE_DEPTHS']))
        if changes > 1:
            raise ValueError('mutation must change only width or one stage depth')
    return all(graph != built_architecture(parent)[0] for parent in parents)


def retry_prompt(prompt, previous_error, attempt):
    if previous_error:
        prompt += f'\nPrevious proposal rejected: {previous_error}\nReturn a different valid assignment.'
    return prompt


def evaluation_cache_key(architecture_hash, fingerprint):
    return hashlib.sha256(json.dumps([architecture_hash, fingerprint], sort_keys=True).encode()).hexdigest()


def write_cached_fitness(sota_root, gene_id, fitness):
    write_results(sota_root, gene_id, DetectionResult(fitness[0], fitness[1], int(fitness[2])))
