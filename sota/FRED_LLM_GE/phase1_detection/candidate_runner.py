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
import subprocess
import time
from pathlib import Path

import torch
import yaml

try:
    from . import map_reporter
    from .adapters.tiny_yolo11 import TinyYolo11Adapter
    from .adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter
    from .proxy_data import REVISION, load_split
    from .result_contract import DetectionResult, write_results
except ImportError:
    import map_reporter
    from adapters.tiny_yolo11 import TinyYolo11Adapter
    from adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter
    from proxy_data import REVISION, load_split
    from result_contract import DetectionResult, write_results

FRED_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = FRED_DIR / "configs" / "phase1" / "protocol.yaml"
DEFAULT_SOTA_ROOT = FRED_DIR / "phase1_detection" / "seeds" / "yolo11"
ADAPTERS = {adapter.name: adapter for adapter in (TinyYolo11Adapter, UltralyticsYolo11Adapter)}

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


def run(gene_id, model_name, variant_dir, family, protocol, sota_root, device):
    start_time = time.time()
    proxy = protocol["proxy"]
    torch.manual_seed(proxy["seed"])
    adapter = ADAPTERS[family](protocol["families"][family], FRED_DIR)
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

    cache_dir = FRED_DIR / proxy["cache_dir"]
    split, step, width = proxy["hf_split"], proxy["frame_step"], proxy["img_width"]
    train_data = load_split(cache_dir, split, proxy["train_sequences"], step, width)
    val_data = load_split(cache_dir, split, proxy["val_sequences"], step, width)
    print(f"family: {family}, train frames: {len(train_data['images'])}, "
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

    run(gene_id, args.model, args.variant_dir, args.family, load_protocol(args.protocol),
        args.sota_root, args.device)


if __name__ == "__main__":
    main()
