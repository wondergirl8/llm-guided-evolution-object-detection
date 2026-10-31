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
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

if __package__:
    from . import map_reporter
    from .adapters.dfine import DFINEAdapter
    from .adapters.tiny_yolo11 import TinyYolo11Adapter
    from .adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter
    from .adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter
    from .export_split import dataset_parts, events_from_tag
    from .fitness import evaluate_frames
    from .proxy_data import REVISION, load_exported_split, load_split
    from .result_contract import DetectionResult, write_results
else:
    # SLURM invokes this file directly. Make the repository namespace package
    # importable so adapters retain their relative imports in that mode too.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from sota.FRED_LLM_GE.phase1_detection import map_reporter
    from sota.FRED_LLM_GE.phase1_detection.adapters.dfine import DFINEAdapter
    from sota.FRED_LLM_GE.phase1_detection.adapters.tiny_yolo11 import TinyYolo11Adapter
    from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter
    from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter
    from sota.FRED_LLM_GE.phase1_detection.export_split import dataset_parts, events_from_tag
    from sota.FRED_LLM_GE.phase1_detection.fitness import evaluate_frames
    from sota.FRED_LLM_GE.phase1_detection.proxy_data import REVISION, load_exported_split, load_split
    from sota.FRED_LLM_GE.phase1_detection.result_contract import DetectionResult, write_results

FRED_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = FRED_DIR / "configs" / "phase1" / "protocol.yaml"
FAMILY_SOTA_ROOTS = {
    "tiny_yolo11": FRED_DIR / "phase1_detection" / "seeds" / "yolo11",
    "yolo11": FRED_DIR / "phase1_detection" / "seeds" / "ultralytics_yolo11",
    "yolo26_p2": FRED_DIR / "phase1_detection" / "seeds" / "yolo26_p2",
    "dfine": FRED_DIR / "phase1_detection" / "seeds" / "dfine",
}
ADAPTERS = {adapter.name: adapter for adapter in
            (TinyYolo11Adapter, UltralyticsYolo11Adapter, UltralyticsYolo26P2Adapter,
             DFINEAdapter)}

# Excluded by the orchestrator's INVALID_FITNESS_MAX filter.
FAILED_RESULT = DetectionResult(-math.inf, -math.inf, math.inf)
INFRASTRUCTURE_ERROR = "ERROR infrastructure: "


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


def dfine_runtime_code_sha256():
    """Identify the shared D-FINE training, data, and evaluation code, including local edits."""
    digest = hashlib.sha256()
    for name in ("phase1_detection/candidate_runner.py", "phase1_detection/adapters/dfine.py",
                 "phase1_detection/fitness.py", "phase1_detection/map_reporter.py",
                 "phase1_detection/proxy_data.py", "phase1_detection/export_split.py"):
        digest.update(name.encode())
        digest.update((FRED_DIR / name).read_bytes())
    return digest.hexdigest()


def target_boxes_sha256(data):
    """Identify ordered frame IDs and approved original-pixel training/eval boxes."""
    if len(data["sample_ids"]) != len(data["orig_boxes"]):
        raise ValueError("sample IDs and target boxes must align")
    digest = hashlib.sha256()
    for sample_id, boxes in zip(data["sample_ids"], data["orig_boxes"]):
        value = torch.as_tensor(boxes, dtype=torch.float32).cpu().contiguous()
        digest.update(sample_id.encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initial_weights_sha256(model):
    """Hash the initial model state before the first optimizer step."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def runtime_environment(device):
    """Record the execution stack used for a D-FINE training run."""
    return {
        "hardware": (torch.cuda.get_device_name(device) if torch.device(device).type == "cuda"
                     else platform.processor() or platform.machine()),
        "device": str(device),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def code_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=FRED_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_failure_results(sota_root, gene_id, start_time, message):
    """Record worst-case fitness for candidates that cannot be evaluated."""
    train_time = time.time() - start_time
    write_results(sota_root, gene_id, FAILED_RESULT, reason=message)
    print(f"ERROR evaluating generated model: {message}")
    print(f"Wrote worst-case fitness after {train_time:.1f}s")
    print('='*120);print('job done');print('='*120)


def _record_run_failure(sota_root, gene_id, start_time, family, device, error):
    """Keep an unavailable CUDA node distinct from a bad candidate."""
    message = repr(error)
    if (family == "dfine" and torch.device(device).type == "cuda"
            and isinstance(error, RuntimeError) and "cuda" in str(error).lower()
            and "busy or unavailable" in str(error).lower()):
        print(f"{INFRASTRUCTURE_ERROR}{message}", flush=True)
        raise SystemExit(1) from error
    write_failure_results(sota_root, gene_id, start_time, message)


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


def inference_ms_per_frame(adapter, model, data, proxy, device):
    """Median detector-only latency for one preloaded frame."""
    image = data["images"][:1].to(device)
    for _ in range(3):
        adapter.predict(model, image, proxy, device)  # warm up kernels and allocations
    elapsed = []
    for _ in range(9):
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        adapter.predict(model, image, proxy, device)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
    return statistics.median(elapsed)


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


def challenging_search(protocol, family=None):
    """Settings and data for the challenging search split (export_split.py --set search).

    Candidates train on challenging_train minus the held-out val_sequences
    and are scored on the val_sequences. D-FINE reads the exported images
    lazily; Ultralytics consumes the export's data.yaml directly.
    """
    challenging = protocol["challenging"]
    data_root = FRED_DIR / challenging["data_root"]
    (train_root, train, train_step), (val_root, val, val_step) = dataset_parts(
        challenging, data_root, "search")
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


def run(gene_id, model_name, variant_dir, family, protocol, sota_root, device,
        diagnostics=False, training_arm=None, refinement_bins=None,
        dfine_search_space="legacy"):
    start_time = time.time()
    if family == "dfine" and torch.device(device).type == "cuda":
        try:
            torch.empty(1, device=device)
        except (RuntimeError, AssertionError) as exc:
            print(f"{INFRASTRUCTURE_ERROR}CUDA unavailable on allocated device: {exc}", flush=True)
            raise SystemExit(1) from exc
    try:
        adapter = ADAPTERS[family](protocol["families"][family], FRED_DIR)
        if adapter.settings.get("data") == "challenging_search":
            proxy, train_data, load_val = challenging_search(protocol, family)
        else:
            proxy = protocol["proxy"]
            train_data = load_val = None
        torch.manual_seed(proxy["seed"])
        config = {"family": family, "proxy": proxy, "family_settings": adapter.settings}

        # Build and prepare the candidate before starting expensive training.
        variant_path = adapter.variant_path(model_name, variant_dir)
        model = adapter.build(variant_path, num_classes=1, device=device)
        param_count = adapter.param_count(model)
        structured_genome = None
        selection_limits = None
        if family == "dfine":
            from src.fred_dfine_genome import genome_path, load_genome
            if dfine_search_space == "motifs_v1":
                structured_genome = load_genome(variant_path, FRED_DIR / "seeds/seed_dfine.py")
                if torch.device(device).type == "cuda" and any(
                    name in os.environ for name in (
                        'FRED_DFINE_MAX_INFERENCE_MS_PER_FRAME',
                        'FRED_DFINE_MAX_PEAK_MEMORY_BYTES')):
                    from src.fred_dfine_selection import resource_limits
                    selection_limits = resource_limits()
            elif genome_path(variant_path).is_file():
                raise ValueError("structured D-FINE candidate requires --dfine-search-space motifs_v1")
        if load_val is None:
            train_data, val_data = proxy_data(proxy)
        else:
            val_data = load_val()
        train_frames = train_data.get("frames") or len(train_data["images"])
        resource_probe = None
        if structured_genome is not None and torch.device(device).type == "cuda":
            resource_probe = adapter.screen_resources(
                model, train_data["images"].shape[-2:],
                adapter.settings.get("microbatch_size", 1), device,
                selection_limits=selection_limits)
        initial_hash = initial_weights_sha256(model) if family == "dfine" else None
        environment = runtime_environment(device) if family == "dfine" else None
        runtime_code_hash = dfine_runtime_code_sha256() if family == "dfine" else None
    except Exception as e:
        _record_run_failure(sota_root, gene_id, start_time, family, device, e)
        return
    print(f"family: {family}, train frames: {train_frames}, "
          f"val frames: {len(val_data['images'])}, params: {param_count}, device: {device}",
          flush=True)

    try:
        training_history = []
        final_frames = None
        training_peak_bytes = None
        if structured_genome is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        if family == "dfine" or training_arm is not None:
            def record_epoch(epoch, train_loss, learning_rate):
                nonlocal final_frames
                point = {"epoch": epoch, "train_loss": train_loss, "lr": learning_rate}
                if family == "dfine":
                    point["order_sha256"] = adapter.training_epoch_order_sha256[-1]
                    point.update(adapter.training_epoch_losses[-1])
                if training_arm is not None and (epoch == 1 or epoch % 5 == 0
                                                 or epoch == proxy["epochs"]):
                    frames = predict_frames(adapter, model, val_data, proxy, device)
                    score = evaluate_frames(frames, params=param_count,
                                            expected_sample_ids=val_data["sample_ids"])
                    point.update(mAP50=score.map50, mAP50_95=score.map50_95)
                    if epoch == proxy["epochs"]:
                        final_frames = frames
                training_history.append(point)

            adapter.train(model, train_data, proxy, device, on_epoch_end=record_epoch)
        elif family == "yolo11":
            args_path = Path(sota_root) / "training_args" / gene_id / "args.yaml"
            adapter.train(model, train_data, proxy, device, args_path=args_path)
            print(f"Saved Ultralytics training arguments: {args_path}", flush=True)
        else:
            adapter.train(model, train_data, proxy, device)
        if structured_genome is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            training_peak_bytes = int(torch.cuda.max_memory_allocated(device))

        model_dir = Path(sota_root) / "trained_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{gene_id}.pt"
        adapter.save(model, model_path)

        frames = final_frames if final_frames is not None else predict_frames(
            adapter, model, val_data, proxy, device)
        metadata = report_metadata(adapter, proxy, config, variant_path, model_path)
        if family == "dfine":
            if structured_genome is not None:
                from src.fred_dfine_selection import VERSION as selection_version
                metadata["genome_version"] = structured_genome["version"]
                metadata["genome_genes"] = structured_genome["genes"]
                metadata["genome_renderer_sha256"] = hashlib.sha256(
                    (FRED_DIR.parent.parent / "src/fred_dfine_genome.py").read_bytes()
                ).hexdigest()
                metadata["search_resource_probe"] = resource_probe
                if torch.device(device).type == "cuda":
                    metadata["resource_measurement_version"] = selection_version
                    if selection_limits is not None:
                        metadata["selection_resource_limits"] = selection_limits
                    metadata["training_peak_memory_bytes"] = training_peak_bytes
                    metadata["preloaded_inference_ms_per_frame"] = inference_ms_per_frame(
                        adapter, model, val_data, proxy, device)
                    metadata["inference_device"] = str(device)
            metadata["training_seed"] = proxy["seed"]
            metadata["initial_weights_sha256"] = initial_hash
            metadata["training_order_sha256"] = adapter.training_order_sha256
            metadata["training_seconds"] = sum(adapter.training_epoch_seconds)
            metadata["training_sample_ids_sha256"] = hashlib.sha256(
                json.dumps(train_data["sample_ids"]).encode()).hexdigest()
            metadata["runtime_environment"] = environment
            metadata["runtime_code_sha256"] = runtime_code_hash
            metadata["training_targets_sha256"] = target_boxes_sha256(train_data)
            metadata["validation_targets_sha256"] = target_boxes_sha256(val_data)
            metadata["validation_sample_ids_sha256"] = hashlib.sha256(
                json.dumps(val_data["sample_ids"]).encode()).hexdigest()
            recipe = dict(config, proxy={key: value for key, value in proxy.items()
                                         if key != "seed"})
            metadata["training_recipe_identity"] = hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()).hexdigest()
            if training_arm is not None:
                metadata["training_arm"] = training_arm
                metadata["training_recipe"] = config
            if refinement_bins is not None:
                metadata["refinement_bins"] = refinement_bins
                training_settings = dict(adapter.settings, model={
                    key: value for key, value in adapter.settings["model"].items()
                    if key != "reg_max"})
                shared_config = dict(
                    config, family_settings=training_settings,
                    proxy={key: value for key, value in proxy.items() if key != "seed"})
                metadata["model_ablation_protocol_identity"] = hashlib.sha256(
                    json.dumps(shared_config, sort_keys=True).encode()).hexdigest()
            if diagnostics and structured_genome is None:
                metadata["preloaded_inference_ms_per_frame"] = inference_ms_per_frame(
                    adapter, model, val_data, proxy, device)
                metadata["inference_device"] = str(device)
        payload = {
            "metadata": metadata,
            "frames": frames,
            "params": param_count,
            "expected_sample_ids": val_data["sample_ids"],
        }
        if family == "dfine" or training_arm is not None:
            payload["training_history"] = training_history
        report_options = ({"diagnostics": True}
                          if diagnostics or structured_genome is not None else {})
        result, report_path = map_reporter.report(payload, sota_root=sota_root,
                                                  gene_id=gene_id, mode="validation",
                                                  **report_options)
    except Exception as e:
        _record_run_failure(sota_root, gene_id, start_time, family, device, e)
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
    parser.add_argument("--variant_dir", type=str, default=None,
                        help="Directory where models are written by LLM-GE")
    parser.add_argument("--protocol", type=str, default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the training/evaluation random seed")
    parser.add_argument("--events", type=str, default=None,
                        help="Event frame tag, e.g. raw_w10000_polarity")
    parser.add_argument("--sota_root", type=str, default=None,
                        help="Where results/ and trained_models/ are written")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnostics", action="store_true",
                        help="include optional COCO diagnostic breakdowns in the report")
    parser.add_argument("--training-seed", type=int,
                        help="D-FINE confirmation seed; write diagnostics outside evolution results")
    parser.add_argument("--repeat-id", type=str,
                        help="D-FINE unchanged-seed repeat label, used with --training-seed")
    parser.add_argument("--dfine-training-arm", choices=("baseline", "longer_budget", "cosine_lr"),
                        help="D-FINE-only training comparison; requires --training-seed")
    parser.add_argument("--dfine-reg-max", type=int,
                        help="D-FINE-only refinement-bin ablation; requires --training-seed")
    parser.add_argument("--dfine-search-space", choices=("legacy", "motifs_v1"),
                        default="legacy", help="D-FINE candidate genome version")
    args = parser.parse_args()

    sota_root = args.sota_root or str(FAMILY_SOTA_ROOTS[args.family])
    variant_dir = args.variant_dir or str(Path(sota_root) / "models")

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
        if args.family == "dfine":
            print(f"{INFRASTRUCTURE_ERROR}{message}", flush=True)
        else:
            write_failure_results(sota_root, gene_id, time.time(), f"gpu_unavailable: {message}")
        raise SystemExit(1)

    protocol = load_protocol(args.protocol)
    if args.seed is not None:
        if args.seed < 0 or args.training_seed is not None:
            parser.error("--seed must be nonnegative and cannot be combined with --training-seed")
        protocol["proxy"]["seed"] = args.seed
    if args.events:
        protocol["challenging"]["events"] = events_from_tag(args.events)
    if args.family != "dfine" and args.dfine_search_space != "legacy":
        parser.error("--dfine-search-space requires --family dfine")
    if args.dfine_reg_max is not None:
        if (args.family != "dfine" or args.training_seed is None
                or args.dfine_training_arm is not None):
            parser.error("--dfine-reg-max requires --family dfine and --training-seed, "
                         "without --dfine-training-arm")
        if args.dfine_reg_max < 4 or args.dfine_reg_max % 2:
            parser.error("--dfine-reg-max must be an even integer of at least 4")
        protocol["families"]["dfine"]["model"]["reg_max"] = args.dfine_reg_max
    if args.dfine_training_arm is not None:
        if args.family != "dfine" or args.training_seed is None:
            parser.error("--dfine-training-arm requires --family dfine and --training-seed")
        if args.dfine_training_arm == "longer_budget":
            protocol["challenging"]["search"]["epochs"] *= 2
        protocol["families"]["dfine"]["lr_schedule"] = (
            "cosine" if args.dfine_training_arm == "cosine_lr" else "constant")
    if args.repeat_id is not None and (
            args.family != "dfine" or args.model != "network" or args.training_seed is None
            or args.dfine_training_arm is not None or args.dfine_reg_max is not None
            or not re.fullmatch(r"[A-Za-z0-9_-]+", args.repeat_id)):
        parser.error("--repeat-id requires an unchanged D-FINE seed and --training-seed; "
                     "use letters, digits, '_' or '-' in the label")
    if args.training_seed is not None:
        if args.family != "dfine" or args.training_seed < 0:
            parser.error("--training-seed requires --family dfine and a nonnegative seed")
        protocol["proxy"]["seed"] = args.training_seed
        if args.dfine_reg_max is not None:
            sota_root = str(Path(sota_root) / "model_ablations" /
                            f"reg_max_{args.dfine_reg_max}" / f"seed_{args.training_seed}")
        elif args.dfine_training_arm is None:
            sota_root = str(Path(sota_root) / "diagnostic_repeats" /
                            f"seed_{args.training_seed}")
            if args.repeat_id is not None:
                sota_root = str(Path(sota_root) / f"repeat_{args.repeat_id}")
        else:
            sota_root = str(Path(sota_root) / "training_ablations" /
                            args.dfine_training_arm / f"seed_{args.training_seed}")
        if args.repeat_id is not None and Path(sota_root).exists():
            parser.error(f"repeat output already exists: {sota_root}")
        if any((Path(sota_root) / folder / f"{gene_id}{suffix}").exists()
               for folder, suffix in (("results", "_evaluation.json"),
                                      ("results", "_results.csv"),
                                      ("trained_models", ".pt"))):
            parser.error(f"confirmation run already exists: {sota_root}/{gene_id}")

    run_options = {}
    if args.family == "dfine":
        run_options = {"diagnostics": args.diagnostics or args.training_seed is not None,
                       "training_arm": args.dfine_training_arm,
                       "refinement_bins": args.dfine_reg_max,
                       "dfine_search_space": args.dfine_search_space}
    elif args.diagnostics:
        run_options["diagnostics"] = True
    run(gene_id, args.model, variant_dir, args.family, protocol,
        sota_root, args.device, **run_options)


if __name__ == "__main__":
    main()
