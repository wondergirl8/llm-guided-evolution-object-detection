"""Bounded Faster R-CNN genome policy for the shared evolution controller."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from numbers import Real

from .seeds.faster_rcnn_scratch.validator import validate_source
from .result_contract import DetectionResult, write_results

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv-faster-rcnn/bin/python"


def _inspect(path):
    from .seeds.faster_rcnn_scratch.network import build_config, build_detector
    cfg = build_config(validate_source(path))
    model, _ = build_detector(cfg, seed=0)
    architecture = {"config": cfg.to_dict(), "modules": [
        (name, type(module).__module__, type(module).__qualname__)
        for name, module in model.named_modules()], "tensors": [
        (name, list(tensor.shape), str(tensor.dtype))
        for name, tensor in model.state_dict().items()]}
    digest = hashlib.sha256(json.dumps(architecture, sort_keys=True).encode()).hexdigest()
    return digest, sum(p.numel() for p in model.parameters())


@lru_cache(maxsize=512)
def _built(path, source_hash):
    if Path(sys.prefix).resolve() == PYTHON.parent.parent.resolve():
        return _inspect(path)
    completed = subprocess.run(
        [str(PYTHON), "-m", __name__, "--inspect", path], cwd=ROOT,
        text=True, capture_output=True, check=True)
    return tuple(json.loads(completed.stdout.splitlines()[-1]))


def built_architecture(path):
    path = Path(path).resolve()
    validate_source(path)
    return _built(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def differs_from_parents(path, *parents):
    try:
        digest, _ = built_architecture(path)
        changed = all(digest != built_architecture(parent)[0] for parent in parents)
        reason = "accepted" if changed else "unchanged_parent"
    except Exception as exc:
        _record(path, f"invalid_architecture: {exc}")
        raise
    _record(path, reason)
    return changed


def _record(path, reason):
    with Path(str(path) + ".proposals.jsonl").open("a") as handle:
        handle.write(json.dumps({"source": Path(path).read_text(), "reason": reason}) + "\n")


def record_rejection(path, reason):
    if Path(path).is_file():
        _record(path, reason)


def evaluation_cache_key(architecture_hash, fingerprint):
    return hashlib.sha256(json.dumps(
        {"architecture": architecture_hash, "protocol": fingerprint}, sort_keys=True).encode()).hexdigest()


def write_cached_fitness(sota_root, gene_id, fitness):
    map50, map50_95, params = fitness
    # The controller parses every CSV fitness column as a float.
    if (isinstance(params, bool) or not isinstance(params, Real)
            or not math.isfinite(params) or params < 0 or params != int(params)):
        raise ValueError("cached params must be a finite nonnegative integer count")
    write_results(sota_root, gene_id, DetectionResult(map50, map50_95, int(params)))


def candidate_chunks(response):
    return re.findall(r"```(?:python)?\s*\n(.*?)```", response or "", re.DOTALL)


def repair_channel_values(proposal):
    # Illegal literals are rejected; there is no silent architecture repair.
    return proposal


def changed_layer_count(path, parent):
    left, right = validate_source(path), validate_source(parent)
    from .seeds.faster_rcnn_scratch.validator import SCHEMA
    return sum(any(left[k] != right[k] for k in fields)
               for fields in SCHEMA["groups"].values())


def retry_prompt(prompt, previous_error, attempt):
    if not previous_error:
        return prompt
    return prompt + f"\nPrevious proposal rejected: {previous_error}\nReturn a legal changed gene class."


def seed_baseline(sota_root, fingerprint):
    """Reuse only a verified seed-0 stock run belonging to this new run."""
    from .faster_rcnn_baseline import summarize
    run_root = Path(sota_root).parent
    summary = summarize(run_root, expected_fingerprint=fingerprint)
    import shutil
    source = run_root / "baseline/seed_0"
    for directory in ("results", "trained_models", "training_args"):
        shutil.copytree(source / directory, Path(sota_root) / directory, dirs_exist_ok=True)
    run = summary["runs"][0]
    return run["mAP50"], run["mAP50_95"], run["params"]


def validate_run(sota_root, fingerprint):
    from .faster_rcnn_baseline import summarize
    run_root = Path(sota_root).parent
    smoke = json.loads((run_root / "smoke.json").read_text())
    if (smoke.get("status") != "passed" or smoke.get("device") != "cuda"
            or not smoke.get("all_backbone_stages_have_gradients")):
        raise ValueError("Evolution requires a successful GPU smoke test for this run")
    smoke_identity = smoke.get("identity", {})
    if (smoke_identity.get("sources") != fingerprint["source_hashes"]
            or any(fingerprint["detector_versions"].get(k) != v
                   for k, v in smoke_identity.get("versions", {}).items())):
        raise ValueError("GPU smoke test belongs to different code or dependencies")
    summarize(run_root, expected_fingerprint=fingerprint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", required=True)
    args = parser.parse_args()
    print(json.dumps(_inspect(args.inspect)))
