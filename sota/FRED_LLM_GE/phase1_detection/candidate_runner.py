"""Train and score one Phase 1 candidate under the shared proxy protocol.

    uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \\
        --family tiny_yolo11 --model network_<gene_id> --variant_dir <models dir>

"--model network" trains the family's seed. Every family gets the same frames,
budget and scoring from configs/phase1/protocol.yaml; only building, training
and inference come from the family adapter (adapters/). Results go to
<sota_root>/results/<gene_id>_results.csv for run_improved.py, the trained
weights to <sota_root>/trained_models/<gene_id>.pt.

Scores are only for ranking candidates against each other during evolution;
elites are retrained at full budget later.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml

try:
    from . import map_reporter
    from .adapters.dfine import DFINEAdapter
    from .adapters.tiny_yolo11 import TinyYolo11Adapter
    from .adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter
    from .export_split import dataset_parts, events_from_tag
    from .proxy_data import REVISION, load_exported_split, load_split
    from .result_contract import DetectionResult, write_results
except ImportError:
    import map_reporter
    from adapters.dfine import DFINEAdapter
    from adapters.tiny_yolo11 import TinyYolo11Adapter
    from adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter
    from export_split import dataset_parts, events_from_tag
    from proxy_data import REVISION, load_exported_split, load_split
    from result_contract import DetectionResult, write_results

FRED_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = FRED_DIR / "configs" / "phase1" / "protocol.yaml"
DEFAULT_SOTA_ROOT = FRED_DIR / "phase1_detection" / "seeds" / "yolo11"
ADAPTERS = {adapter.name: adapter for adapter in
            (TinyYolo11Adapter, UltralyticsYolo11Adapter, DFINEAdapter)}

# Worst-case values for FITNESS_WEIGHTS = (1.0, 1.0, -1.0).
FAILED_RESULT = DetectionResult(0.0, 0.0, 999999999)


def load_protocol(path):
    with open(path) as file:
        protocol = yaml.safe_load(file)
    proxy = protocol["proxy"]
    for key in ("train_sequences", "val_sequences"):
        proxy[key] = [str(s) for s in proxy[key]]
    if set(proxy["train_sequences"]) & set(proxy["val_sequences"]):
        raise ValueError("train and val sequences must not overlap")
    if proxy["epochs"] <= 0 or proxy["frame_step"] <= 0 or proxy["img_width"] < 32:
        raise ValueError("epochs and frame_step must be positive, img_width at least 32")
    return protocol


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=FRED_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_failure_results(sota_root, gene_id, start_time, message):
    """Record worst-case fitness for candidates that cannot be evaluated."""
    train_time = time.time() - start_time
    write_results(sota_root, gene_id, FAILED_RESULT)
    print(f"ERROR evaluating generated model: {message}")
    print(f"Wrote worst-case fitness after {train_time:.1f}s")
    print('='*120);print('job done');print('='*120)


def predict_frames(adapter, model, data, proxy, device):
    """Predict every val frame and pair it with ground truth in original pixels."""
    width, height = data["orig_size"]
    scale = torch.tensor([width / data["images"].shape[3], height / data["images"].shape[2]] * 2)
    frames = []
    batch_size = proxy["batch_size"]
    for start in range(0, len(data["images"]), batch_size):
        images = data["images"][start:start + batch_size]
        for offset, (boxes, scores) in enumerate(adapter.predict(model, images, proxy, device)):
            i = start + offset
            boxes = (boxes * scale).tolist()
            scores = scores.tolist()
            kept = [([max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)], s)
                    for (x1, y1, x2, y2), s in zip(boxes, scores)]
            kept = [(b, min(1.0, s)) for b, s in kept if b[2] > b[0] and b[3] > b[1]]
            target = data["orig_boxes"][i].tolist()
            frames.append({
                "sample_id": data["sample_ids"][i],
                "width": width,
                "height": height,
                "target": {"boxes": target, "labels": [0] * len(target)},
                "prediction": {"boxes": [b for b, _ in kept],
                               "labels": [0] * len(kept),
                               "scores": [s for _, s in kept]},
            })
    return frames


def report_metadata(adapter, proxy, config, variant_path, model_path):
    return {
        "dataset_revision": REVISION,
        "manifest_identity": hashlib.sha256(json.dumps(
            {k: proxy[k] for k in ("train_sequences", "val_sequences", "frame_step")},
            sort_keys=True).encode()).hexdigest(),
        "split_identity": (f"train={','.join(proxy['train_sequences'])};"
                           f"val={','.join(proxy['val_sequences'])}"),
        "annotation_policy": "fred_coordinates_txt_clipped_to_image",
        "model_identity": sha256_file(variant_path),
        "checkpoint_identity": sha256_file(model_path),
        "code_revision": code_revision(),
        "postprocessing": adapter.postprocessing(proxy),
        "training_config_identity": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "modality": proxy["modality"],
        "benchmark_split": proxy["benchmark_split"],
        "project_split": "validation",
        "official_split": proxy["official_split"],
    }


def proxy_data(proxy):
    """Train/val data for the 8-sequence proxy: cached frames held in memory."""
    cache_dir = FRED_DIR / proxy["cache_dir"]
    split, step, width = proxy["hf_split"], proxy["frame_step"], proxy["img_width"]
    return (load_split(cache_dir, split, proxy["train_sequences"], step, width),
            load_split(cache_dir, split, proxy["val_sequences"], step, width))


def challenging_search(protocol, family=None, roots=None):
    """Settings and data for the challenging search split (export_split.py --set search).

    Candidates train on challenging_train minus the held-out val_sequences
    and are scored on the val_sequences. D-FINE reads the exported images
    lazily; Ultralytics consumes the export's data.yaml directly. roots, if
    given, replaces the (train, val) export folders.
    """
    challenging = protocol["challenging"]
    data_root = FRED_DIR / challenging["data_root"]
    (train_root, train, train_step), (val_root, val, val_step) = dataset_parts(
        challenging, data_root, "search")
    if roots:
        # Frames rendered elsewhere in the same layout (render_runner.py).
        train_root, val_root = roots
    data_yaml = train_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"{data_yaml} missing: run export_split.py --set search "
                                "(all shards, then --finalize)")
    proxy = dict(protocol["proxy"], train_sequences=[s for s, _ in train],
                 val_sequences=[s for s, _ in val], frame_step=train_step,
                 val_frame_step=val_step, epochs=challenging["search"]["epochs"])
    if family == "dfine":
        train_data = load_exported_split(train_root, train)
    else:
        frames = len((train_root / "train.txt").read_text().split())
        train_data = {"data_yaml": data_yaml, "frames": frames}
    return proxy, train_data, lambda: load_exported_split(
        val_root, val)


def run(gene_id, model_name, variant_dir, family, protocol, sota_root, device, data_roots=None):
    start_time = time.time()
    adapter = ADAPTERS[family](protocol["families"][family], FRED_DIR)
    if adapter.settings.get("data") == "challenging_search":
        proxy, train_data, load_val = challenging_search(protocol, family, data_roots)
    else:
        proxy = protocol["proxy"]
        train_data = load_val = None
    torch.manual_seed(proxy["seed"])
    config = {"family": family, "proxy": proxy, "family_settings": adapter.settings}

    # This is LLM Guided Code
    # Build the candidate; a broken LLM-generated architecture gets worst-case fitness
    try:
        variant_path = adapter.variant_path(model_name, variant_dir)
        model = adapter.build(variant_path, num_classes=1, device=device)
    except Exception as e:
        write_failure_results(sota_root, gene_id, start_time, repr(e))
        return

    param_count = adapter.param_count(model)

    if load_val is None:
        train_data, val_data = proxy_data(proxy)
    else:
        val_data = load_val()
    train_frames = train_data.get("frames") or len(train_data["images"])
    print(f"family: {family}, train frames: {train_frames}, "
          f"val frames: {len(val_data['images'])}, params: {param_count}, device: {device}",
          flush=True)

    try:
        adapter.train(model, train_data, proxy, device)

        model_dir = Path(sota_root) / "trained_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{gene_id}.pt"
        adapter.save(model, model_path)

        frames = predict_frames(adapter, model, val_data, proxy, device)
        payload = {
            "metadata": report_metadata(adapter, proxy, config, variant_path, model_path),
            "frames": frames,
            "params": param_count,
            "expected_sample_ids": val_data["sample_ids"],
        }
        result, report_path = map_reporter.report(payload, sota_root=sota_root,
                                                  gene_id=gene_id, mode="validation")
    except Exception as e:
        write_failure_results(sota_root, gene_id, start_time, repr(e))
        return
    train_time = time.time() - start_time

    print(f"mAP50: {result.map50}, mAP50:95: {result.map50_95}, "
          f"Params: {result.params}, Time: {train_time:.1f}s")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print('='*120);print('job done');print('='*120)


def main():
    parser = argparse.ArgumentParser(description="Train and score one Phase 1 candidate")
    parser.add_argument("--family", choices=sorted(ADAPTERS), default="tiny_yolo11")
    parser.add_argument("--model", type=str, default="network",
                        help='Variant module name like "network_XXXX"; "network" trains the seed')
    parser.add_argument("--variant_dir", type=str, default=str(DEFAULT_SOTA_ROOT / "models"),
                        help="Directory where models are written by LLM-GE")
    parser.add_argument("--protocol", type=str, default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--events", type=str, default=None,
                        help='Event frames, e.g. "raw_w10000_polarity" (default: protocol\'s '
                             "challenging.events)")
    parser.add_argument("--sota_root", type=str, default=str(DEFAULT_SOTA_ROOT),
                        help="Where results/ and trained_models/ are written")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # this will get the gene id value: "network_XXXX" -> "XXXX"
    try:
        gene_id = args.model.split('network_')[1]
    except IndexError:
        gene_id = 'seed'

    # A GPU job whose GPU CUDA cannot see would otherwise fall back to the CPU
    # and run for hours; fail now so LLM-GE is not left waiting on it.
    if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        message = (f"GPU allocated (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}) "
                   "but CUDA is unavailable on this node; not training on CPU")
        write_failure_results(args.sota_root, gene_id, time.time(), message)
        raise SystemExit(1)

    protocol = load_protocol(args.protocol)
    if args.events:
        protocol["challenging"]["events"] = events_from_tag(args.events)
    run(gene_id, args.model, args.variant_dir, args.family, protocol, args.sota_root, args.device)


if __name__ == "__main__":
    main()
