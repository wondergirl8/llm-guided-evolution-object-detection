"""Versioned SpikeYOLO graph genome using only the pinned official Gen1 blocks."""
import copy
import hashlib
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from .spikeyolo_source import SOURCE, verify_source
from .result_contract import DetectionResult, write_results

SEED = Path(__file__).resolve().parents[1] / "seeds/seed_spikeyolo.yaml"
GENOME_VERSION = "spikeyolo-graph-v2"
SECTIONS = ("backbone", "head")
BLOCKS = {"MS_ConvBlock", "MS_AllConvBlock"}
CHANNEL_MODULES = {"MS_DownSampling", "MS_StandardConv", "SpikeSPPF"}
FROZEN_MODULES = {"MS_GetT", "Concat", "nn.Upsample", "SpikeDetect"}
# Raw repeats are interpreted by the official s depth multiplier, 0.33.
# 1 and 3 are aliases; 3 remains legal to preserve the exact stock YAML.
REPEATS = (1, 3, 6, 9, 12)
KERNELS = (1, 3, 5, 7)
EXPANSIONS = (1, 2, 3, 4)


class _GenomeLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently accepting a hidden last value."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("genome mapping keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_GenomeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _yaml(text):
    try:
        return yaml.load(text, Loader=_GenomeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid SpikeYOLO YAML: {exc}") from exc


def stock_config():
    """Exact upstream s configuration with the FRED one-class adaptation."""
    config = yaml.safe_load((SOURCE / "ultralytics/cfg/models/v8/snn_yolov8.yaml").read_text())
    return dict(config, nc=1, scale="s")


def _integer(value, choices):
    return type(value) is int and value in choices


def validate_config(config):
    """Validate all constructor arguments before the official parser can see them."""
    reference = dict(stock_config(), genome_version=GENOME_VERSION)
    if not isinstance(config, dict) or set(config) != set(reference):
        raise ValueError("SpikeYOLO genome requires exactly the stock keys plus genome_version")
    for key in ("nc", "scale", "scales", "genome_version"):
        if json.dumps(config[key], sort_keys=True) != json.dumps(reference[key], sort_keys=True):
            raise ValueError(f"protected SpikeYOLO field changed: {key}")

    global_index = 0
    for section in SECTIONS:
        layers, original = config[section], reference[section]
        if not isinstance(layers, list) or len(layers) != len(original):
            raise ValueError(f"{section} must preserve its {len(original)} layer positions")
        for layer, base in zip(layers, original):
            location = f"{section} layer {global_index}"
            global_index += 1
            if not isinstance(layer, list) or len(layer) != 4:
                raise ValueError(f"{location}: expected [from, repeats, module, args]")
            source, repeats, module, args = layer
            # JSON comparison distinguishes booleans/floats from integer routing.
            if json.dumps(source) != json.dumps(base[0]):
                raise ValueError(f"{location}: protected from links changed")
            if not isinstance(module, str) or not isinstance(args, list):
                raise ValueError(f"{location}: invalid module or arguments")
            if base[2] in FROZEN_MODULES:
                if json.dumps(layer) != json.dumps(base):
                    raise ValueError(f"{location}: frozen {base[2]} changed")
                continue
            if base[2] in BLOCKS:
                if module not in BLOCKS or not _integer(repeats, REPEATS):
                    raise ValueError(f"{location}: use an official MS block and repeats {REPEATS}")
                if (len(args) != 2 or not _integer(args[0], EXPANSIONS)
                        or not _integer(args[1], (3, 5, 7))):
                    raise ValueError(f"{location}: block args must be [expansion 1..4, kernel 3/5/7]")
                continue
            if module != base[2] or not _integer(repeats, (1,)):
                raise ValueError(f"{location}: preserve this projection/downsampling module and repeats=1")
            width = args[0] if args else None
            low, high = max(64, base[3][0] // 2), 2 * base[3][0]
            if module in ("SpikeSPPF", "MS_StandardConv"):
                high = min(high, reference["scales"]["s"][2])
            if type(width) is not int or width % 64 or not low <= width <= high:
                raise ValueError(f"{location}: output channels must be multiples of 64 in [{low}, {high}]")
            if module == "MS_DownSampling":
                if (len(args) != 5 or not _integer(args[1], (3, 5, 7))
                        or type(args[2]) is not int or args[2] != base[3][2]
                        or type(args[3]) is not int
                        or args[3] != base[3][3] + (args[1] - base[3][1]) // 2
                        or type(args[4]) is not bool or args[4] != base[3][4]):
                    raise ValueError(f"{location}: preserve stride/first_layer; adjust padding with kernel")
            elif module == "MS_StandardConv":
                if (len(args) != 3 or not _integer(args[1], KERNELS)
                        or type(args[2]) is not int or args[2] != base[3][2]):
                    raise ValueError(f"{location}: use [channels, kernel 1/3/5/7, unchanged stride]")
            elif module == "SpikeSPPF":
                if len(args) != 2 or not _integer(args[1], (3, 5, 7)):
                    raise ValueError(f"{location}: use [channels, pooling kernel 3/5/7]")
            else:
                raise ValueError(f"{location}: unsupported mutable module {module}")
    return config


def load_and_validate(path):
    parts = Path(path).read_text().split("# --OPTION--")
    if len(parts) != 3:
        raise ValueError("require v2 YAML with exactly two --OPTION-- sections: backbone and head")
    config = _yaml(parts[0])
    if not isinstance(config, dict) or set(config) != {"nc", "scale", "scales", "genome_version"}:
        raise ValueError("protected genome preamble changed")
    for section, part in zip(SECTIONS, parts[1:]):
        values = _yaml(part)
        if not isinstance(values, dict) or set(values) != {section}:
            raise ValueError(f"mutable section must contain only {section}")
        config[section] = values[section]
    return validate_config(config)


def model_config(path):
    verify_source()
    config = copy.deepcopy(load_and_validate(path))
    config.pop("genome_version")
    return config


def build(path, device="cpu"):
    from .spikeyolo_source import build_official
    return build_official(model_config(path), device)


@lru_cache(maxsize=128)
def _preflight(config_text, source_identity):
    # Keep the official fork out of controller/operator import state.
    result = subprocess.run(
        [sys.executable, "-m", "sota.FRED_LLM_GE.phase1_detection.spikeyolo_smoke",
         "--config-json", config_text], cwd=SEED.parents[3],
        text=True, capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise ValueError(f"official SpikeYOLO preflight failed: {result.stderr[-4000:]}")
    return json.loads(result.stdout.splitlines()[-1])


def built_architecture(path):
    config = model_config(path)
    # A changed source manifest must not reuse a prior in-process preflight.
    source_identity = hashlib.sha256((SOURCE / "UPSTREAM.json").read_bytes()).hexdigest()
    result = _preflight(json.dumps(config, sort_keys=True), source_identity)
    return result["architecture_hash"], result["params"]


def _effective_layers(path):
    config = load_and_validate(path)
    depth = config["scales"][config["scale"]][0]
    layers = copy.deepcopy(config["backbone"] + config["head"])
    for layer in layers:
        n = layer[1]
        layer[1] = max(round(n * depth), 1) if n > 1 else n
    return layers


def changed_layer_count(child, parent):
    return sum(a != b for a, b in zip(_effective_layers(child), _effective_layers(parent)))


def differs_from_parents(child, *parents):
    if len(parents) == 1:
        changed = changed_layer_count(child, parents[0])
        if changed == 0:
            return False
        if changed != 1:
            raise ValueError("mutation must change exactly one effective mutable layer")
    graph, _ = built_architecture(child)
    return all(graph != built_architecture(parent)[0] for parent in parents)


def candidate_chunks(response):
    return [block.strip() for block in re.findall(
        r"```(?:yaml|yml)[ \t]*\r?\n(.*?)```",
        response or "", re.IGNORECASE | re.DOTALL) if block.strip()]


def repair_channel_values(chunk):
    # Operator interface shared with YOLO11. Reject invalid widths explicitly;
    # silently rounding would alter the LLM's proposed architecture.
    return chunk


def retry_prompt(prompt, previous_error, attempt):
    feedback = f"\nPrevious proposal rejected: {previous_error}\n" if previous_error else "\n"
    return (prompt + feedback +
            "Return the complete corresponding backbone or head YAML section in one yaml fence. "
            "Keep layer count, routing, frozen layers and constructor contracts.")


def mutation_prompt(prompt, previous_error, attempt):
    return (retry_prompt(prompt, previous_error, attempt) +
            " Change exactly one effective mutable layer; keep all other layers unchanged.")


def evaluation_cache_key(architecture_hash, fingerprint):
    return hashlib.sha256(json.dumps(
        [GENOME_VERSION, architecture_hash, fingerprint], sort_keys=True).encode()).hexdigest()


def write_cached_fitness(sota_root, gene_id, fitness):
    write_results(sota_root, gene_id, DetectionResult(fitness[0], fitness[1], int(fitness[2])))
