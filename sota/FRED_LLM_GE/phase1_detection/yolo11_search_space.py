"""The controlled YOLO11n architecture search space and built-model identity."""

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

import yaml


SEED_YAML = Path(__file__).resolve().parents[1] / "seeds" / "seed_yolo11n.yaml"
SEED_PARAMS = 2_590_035
MIN_PARAMS = math.ceil(SEED_PARAMS * 0.5)
MAX_PARAMS = SEED_PARAMS * 2
MODULES = {"Conv", "GhostConv", "DWConv", "C3k2", "C2f", "C3", "C3Ghost",
           "RepC3", "A2C2f", "Bottleneck", "SPPF", "SPP", "C2PSA", "PSA"}
FROZEN_MODULES = {"Concat", "nn.Upsample", "Detect"}
CHANNEL_MODULES = MODULES


def load_and_validate(path, seed_path=SEED_YAML):
    """Return a candidate config only when protected fields and layer topology hold."""
    candidate = yaml.safe_load(Path(path).read_text())
    seed = yaml.safe_load(Path(seed_path).read_text())
    if not isinstance(candidate, dict) or set(candidate) != set(seed):
        raise ValueError("YOLO11 candidate must have exactly the seed's top-level keys")
    for key in ("nc", "scale", "scales"):
        if candidate[key] != seed[key]:
            raise ValueError(f"protected YOLO11 field changed: {key}")
    if candidate["nc"] != 1 or candidate["scale"] != "n":
        raise ValueError("YOLO11 search requires one class and scale n")

    index = 0
    for section in ("backbone", "head"):
        layers, original = candidate[section], seed[section]
        if not isinstance(layers, list) or len(layers) != len(original):
            raise ValueError(f"{section} must keep the seed's {len(original)} layers")
        for layer, base in zip(layers, original):
            if not isinstance(layer, list) or len(layer) != 4:
                raise ValueError(f"layer {index} must be [from, repeats, module, args]")
            source, repeats, module, args = layer
            if base[2] in FROZEN_MODULES and layer != base:
                raise ValueError(f"layer {index} is frozen ({base[2]})")
            if source != base[0]:
                raise ValueError(f"layer {index} changed protected from links")
            links = source if isinstance(source, list) else [source]
            if any(not isinstance(link, int) or isinstance(link, bool) or
                   (link != -1 and not 0 <= link < index) for link in links):
                raise ValueError(f"layer {index} has invalid from link")
            if module in FROZEN_MODULES:
                if layer != base:
                    raise ValueError(f"layer {index} is frozen ({module})")
            elif module not in MODULES:
                raise ValueError(f"layer {index} uses unapproved module {module!r}")
            elif not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 4:
                raise ValueError(f"layer {index} repeats must be 1..4")
            elif not isinstance(args, list) or not args:
                raise ValueError(f"layer {index} needs module arguments")
            else:
                if module in CHANNEL_MODULES:
                    channels = args[0]
                    if (not isinstance(channels, int) or isinstance(channels, bool) or
                            channels <= 0 or channels % 64 or channels > 2 * base[3][0]):
                        raise ValueError(f"layer {index} channels must be a multiple of 64 "
                                         "and at most twice the seed width")
                if module in {"Conv", "GhostConv", "DWConv"}:
                    expected_stride = base[3][2] if base[2] in {"Conv", "GhostConv", "DWConv"} else 1
                    if len(args) < 3 or args[2] != expected_stride:
                        raise ValueError(f"layer {index} changed the downsampling pattern")
            index += 1
    if candidate["head"][-1] != seed["head"][-1] or candidate["head"][-1][2] != "Detect":
        raise ValueError("Detect must remain the final layer")
    return candidate


def built_architecture(path):
    """Validate and build one YAML on CPU; return its graph hash and parameter count."""
    from ultralytics import YOLO

    config = load_and_validate(path)
    with tempfile.TemporaryDirectory(prefix="yolo11_preflight_") as directory:
        model_path = Path(directory) / "yolo11n.yaml"
        model_path.write_text(yaml.safe_dump(config, sort_keys=False))
        return validate_built_model(YOLO(str(model_path), task="detect"))


def differs_from_parents(child_path, *parent_paths):
    """Reject a candidate whose built graph is identical to an input parent."""
    child_hash, _ = built_architecture(child_path)
    for parent_path in parent_paths:
        parent_hash, _ = built_architecture(parent_path)
        if child_hash == parent_hash:
            return False
    return True


def retry_prompt(prompt, error, attempt):
    """Give the operator actionable feedback without changing the original template."""
    if attempt == 0:
        return prompt
    return (f'{prompt}\n\nPrevious proposal was rejected: {error}. '
            'Change exactly one mutable layer in this section. Preserve its from link, '
            'the layer count, and frozen layers. Return the complete YAML section in '
            'one fenced yaml block.')


def mutation_prompt(prompt, error, attempt):
    """End each YOLO11 mutation request with the required response contract."""
    prompt = retry_prompt(prompt, error, attempt)
    return (f'{prompt}\n\nNow return exactly one complete fenced yaml block. '
            'Change exactly one mutable layer from the supplied section; do not '
            'repeat it unchanged or list alternative versions. Include the whole '
            'replacement backbone or head section and no explanation.')


def candidate_chunks(response):
    """Return complete YAML blocks in output order, leaving validation to the caller."""
    return [block.strip() for block in re.findall(
        r'```(?:yaml|yml)[ \t]*\r?\n(.*?)```', response or '', re.IGNORECASE | re.DOTALL
    ) if block.strip()]


def repair_channel_values(chunk):
    """Round numeric YAML channel widths into the allowed range before validation."""
    try:
        section = yaml.safe_load(chunk)
    except yaml.YAMLError:
        return chunk
    if not isinstance(section, dict) or len(section) != 1:
        return chunk
    name = next(iter(section))
    if name not in ('backbone', 'head') or not isinstance(section[name], list):
        return chunk
    original = yaml.safe_load(SEED_YAML.read_text())[name]
    if len(section[name]) != len(original):
        return chunk
    changed = False
    for layer, base in zip(section[name], original):
        if (not isinstance(layer, list) or len(layer) != 4 or
                layer[2] not in CHANNEL_MODULES or base[2] in FROZEN_MODULES or
                not isinstance(layer[3], list) or not layer[3]):
            continue
        width = layer[3][0]
        if isinstance(width, (int, float)) and not isinstance(width, bool) and math.isfinite(width):
            repaired = max(64, min(2 * base[3][0], 64 * round(width / 64)))
            if repaired != width:
                layer[3][0] = repaired
                changed = True
    if changed:
        print('Repaired numeric YOLO11 channel widths to allowed multiples of 64', flush=True)
        return yaml.safe_dump(section, sort_keys=False)
    return chunk


def changed_layer_count(child_path, parent_path):
    """Count effective YAML layer edits for a mutation; crossover may combine parents."""
    child = load_and_validate(child_path)
    parent = load_and_validate(parent_path)
    return sum(a != b for section in ('backbone', 'head')
               for a, b in zip(child[section], parent[section]))


def evaluation_cache_key(architecture_hash, fingerprint):
    """Tie reuse to this FRED protocol, its training seed, and the built graph."""
    protocol_path = Path(__file__).resolve().parents[1] / "configs" / "phase1" / "protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    events = fingerprint.get('evolution_settings', {}).get('fred_events', '')
    identity = [fingerprint["sha256"], protocol["proxy"]["seed"], events,
                architecture_hash]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def write_cached_fitness(sota_root, gene_id, fitness):
    """Preserve the ordinary FRED result artifact for a reused evaluation."""
    try:
        from .result_contract import DetectionResult, write_results
    except ImportError:
        from result_contract import DetectionResult, write_results
    write_results(sota_root, gene_id,
                  DetectionResult(fitness[0], fitness[1], int(fitness[2])))


def validate_built_model(model):
    """Check the instantiated detector and hash its effective module graph."""
    network = model.model
    strides = [int(value) for value in network.stride.tolist()]
    if strides != [8, 16, 32]:
        raise ValueError(f"YOLO11 detection strides must be 8/16/32, got {strides}")
    params = sum(parameter.numel() for parameter in network.parameters())
    if not MIN_PARAMS <= params <= MAX_PARAMS:
        raise ValueError(f"YOLO11 parameter count {params} outside [{MIN_PARAMS}, {MAX_PARAMS}]")
    graph = {"modules": repr(network.model),
             "routing": [getattr(layer, "f", None) for layer in network.model],
             "strides": strides,
             "state_shapes": [(name, list(value.shape)) for name, value in network.state_dict().items()]}
    digest = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
    return digest, params
