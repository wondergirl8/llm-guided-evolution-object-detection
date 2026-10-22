"""RT-DETR (v1) search-space policy and built-model identity.

Same controller interface as yolo11_search_space.py (built_architecture,
differs_from_parents, retry_prompt, evaluation_cache_key, write_cached_fitness),
for the Python seed seeds/seed_rtdetr.py. A candidate must keep the seed's
protected preamble and its three mutable classes (PResNet, HybridEncoder,
RTDETRTransformer); inside them any change is allowed.

Identity is functional rather than textual: the candidate is built on CPU with a
fixed random seed and run on a fixed input, and the hash covers its parameter
shapes and rounded outputs. Renames, reformatting, docstring edits and
regrouping the same layers therefore hash like the parent and are rejected as
no-op mutations, while a parameter-free change to forward() still counts.
"""

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import torch
import yaml

try:
    from .adapters.rtdetr import _source_parts
    from .yolo11_search_space import evaluation_cache_key, write_cached_fitness
except ImportError:
    from adapters.rtdetr import _source_parts
    from yolo11_search_space import evaluation_cache_key, write_cached_fitness

__all__ = ["built_architecture", "differs_from_parents", "retry_prompt",
           "evaluation_cache_key", "write_cached_fitness"]

FRED_DIR = Path(__file__).resolve().parents[1]
SEED_PY = FRED_DIR / "seeds" / "seed_rtdetr.py"
PROTOCOL = FRED_DIR / "configs" / "phase1" / "protocol.yaml"
# Candidates may shrink to a quarter of the seed or grow to twice its size.
MIN_PARAMS_RATIO, MAX_PARAMS_RATIO = 0.25, 2.0
# Smallest multiple of 32 whose stride-8 map holds the protocol's 300 queries.
_PROBE_SIZE = (256, 320)


def _model_settings():
    return dict(yaml.safe_load(PROTOCOL.read_text())["families"]["rtdetr"].get("model", {}))


def _load_module(path):
    parts = _source_parts(path)
    if parts[0] != _source_parts(SEED_PY)[0]:
        raise ValueError("RT-DETR candidate changed the protected preamble")
    spec = importlib.util.spec_from_file_location(f"rtdetr_policy_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(path):
    """Build on CPU, run a fixed probe in eval and train mode, return (hash, params)."""
    module = _load_module(path)
    torch.manual_seed(0)
    model = module.RTDETREventDetector(**_model_settings())
    params = sum(parameter.numel() for parameter in model.parameters())
    generator = torch.Generator().manual_seed(0)
    event = torch.rand((1, 3, *_PROBE_SIZE), generator=generator)
    target = [{"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
               "labels": torch.zeros(1, dtype=torch.long)}]
    model.train()
    with torch.no_grad():
        model(event, target)  # Fails early on a broken training contract.
    model.eval()
    with torch.no_grad():
        outputs = model(event)
    rounded = [torch.round(outputs[key].double() * 1e4).to(torch.int64).flatten().tolist()
               for key in ("pred_logits", "pred_boxes")]
    graph = {"shapes": [list(value.shape) for value in model.state_dict().values()],
             "outputs": rounded}
    return hashlib.sha256(json.dumps(graph).encode()).hexdigest(), params


@lru_cache(maxsize=1)
def _seed_params(settings_key):
    return _identity(SEED_PY)[1]


def built_architecture(path):
    """Validate and build one candidate on CPU; return its graph hash and parameter count."""
    digest, params = _identity(path)
    seed_params = _seed_params(json.dumps(_model_settings(), sort_keys=True))
    low, high = int(seed_params * MIN_PARAMS_RATIO), int(seed_params * MAX_PARAMS_RATIO)
    if not low <= params <= high:
        raise ValueError(f"RT-DETR parameter count {params} outside [{low}, {high}]")
    return digest, params


def differs_from_parents(child_path, *parent_paths):
    """Reject a candidate that behaves exactly like one of its parents."""
    child_hash, _ = built_architecture(child_path)
    return all(built_architecture(parent)[0] != child_hash for parent in parent_paths)


def retry_prompt(prompt, error, attempt):
    """Feed the rejection reason back so the next proposal can fix it."""
    if attempt == 0:
        return prompt
    return (f"{prompt}\n\nYour previous proposal was rejected: {error}. Return the "
            "complete class in one python code block. Keep the fixed interface, and "
            "make a real architectural change: a rename, reformat, docstring edit, or "
            "regrouping of the same layers does not count.")
