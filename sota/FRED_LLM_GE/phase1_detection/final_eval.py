"""Train a YOLO11-family model on FRED challenging_train and score it once on challenging_test.

    uv run python sota/FRED_LLM_GE/phase1_detection/final_eval.py --name yolo11n_scratch
    uv run python sota/FRED_LLM_GE/phase1_detection/final_eval.py --name best_evolved \\
        --model network_<gene_id> --variant_dir <run's models dir>
    # optional: start from COCO-pretrained weights where layers match
    uv run python sota/FRED_LLM_GE/phase1_detection/final_eval.py --name yolo11n_coco \\
        --weights yolo11n.pt

Training uses the dataset written by export_split.py --set final (official
challenging_train, every <train_frame_step>th frame). The trained model is then run on every frame
of the official challenging_test split (downloaded a sequence at a time) and scored
by map_reporter in held-out mode; the report goes to
<data_root>/held_out_reports/<name>_evaluation.json. Use this only on models
whose selection is already frozen: test results must not feed back into search.
"""

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

try:
    from . import map_reporter
    from .adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter, _ultralytics_device
    from .candidate_runner import code_revision, load_protocol, sha256_file
    from .export_split import events_from_tag, events_suffix
    from .fred_splits import challenging_split, sequence_frames
    from .proxy_data import REVISION, STEP_US, load_annotations
    from .result_contract import validate_gene_id
except ImportError:
    import map_reporter
    from adapters.ultralytics_yolo11 import UltralyticsYolo11Adapter, _ultralytics_device
    from candidate_runner import code_revision, load_protocol, sha256_file
    from export_split import events_from_tag, events_suffix
    from fred_splits import challenging_split, sequence_frames
    from proxy_data import REVISION, STEP_US, load_annotations
    from result_contract import validate_gene_id

FRED_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = FRED_DIR / "configs" / "phase1" / "protocol.yaml"


def render_sequence(job):
    """Frames of one test sequence as lossless PNG bytes, so a worker process can read it."""
    folder, sequence, report, frame_step, download_dir, events = job
    rendered = []
    for image, timestamp in sequence_frames(folder, sequence, report, frame_step, download_dir,
                                            events=events):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "PNG", compress_level=1)
        rendered.append((buffer.getvalue(), timestamp))
    return rendered


def predict_test_split(adapter, model, test, settings, device, report_dir, workers=1):
    """Read every test frame, predict in batches, pair with ground truth.

    With workers > 1, sequences are downloaded and rendered by that many
    processes (raw-event rendering is single-core, ~15 min a sequence) while
    the GPU predicts; workers=1 reads them in this process.
    """
    frames, expected = [], []
    batch, pending = [], []

    def flush():
        for (sample_id, target, size), (boxes, scores) in zip(
                pending, adapter.predict(model, torch.stack(batch), settings, device)):
            width, height = size
            kept = [([max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)], min(1.0, s))
                    for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist())]
            kept = [(b, s) for b, s in kept if b[2] > b[0] and b[3] > b[1]]
            frames.append({
                "sample_id": sample_id, "width": width, "height": height,
                "target": {"boxes": target, "labels": [0] * len(target)},
                "prediction": {"boxes": [b for b, _ in kept], "labels": [0] * len(kept),
                               "scores": [s for _, s in kept]},
            })
        batch.clear()
        pending.clear()

    jobs = [(folder, sequence, report_dir / f"bad_timestamps_test_{sequence}.csv",
             settings["test_frame_step"], report_dir / "downloads", settings.get("events"))
            for sequence, folder in test]
    # spawn, not fork: the parent has already initialized CUDA.
    with (multiprocessing.get_context("spawn").Pool(workers) if workers > 1
          else contextlib.nullcontext()) as pool:
        sequences = pool.imap(render_sequence, jobs) if pool else map(render_sequence, jobs)
        for number, ((sequence, folder), rendered) in enumerate(zip(test, sequences), 1):
            annotations, count = None, 0
            for data, timestamp in rendered:
                image = Image.open(io.BytesIO(data)).convert("RGB")
                if annotations is None:
                    annotations = load_annotations(folder, sequence, *image.size)
                frame_index = round(timestamp * 1_000_000 / STEP_US)
                sample_id = f"{folder}/{sequence}:{frame_index}"
                expected.append(sample_id)
                batch.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
                pending.append((sample_id, annotations.get(frame_index, []), image.size))
                count += 1
                if len(batch) == settings["batch_size"]:
                    flush()
            if batch:
                flush()
            print(f"test {number}/{len(test)} {folder}/{sequence}: {count} frames", flush=True)
    return frames, expected


def main():
    parser = argparse.ArgumentParser(description="Final train + held-out test on FRED challenging")
    parser.add_argument("--name", required=True, help="Report name, e.g. yolo11n_scratch")
    parser.add_argument("--model", default="network",
                        help='"network" for the seed or network_<id> for an evolved candidate')
    parser.add_argument("--variant_dir", default=None,
                        help="Folder holding network_<id>.py (the run's models dir)")
    parser.add_argument("--weights", default=None,
                        help="Optional pretrained checkpoint (e.g. yolo11n.pt); default from scratch")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--events", default=None,
                        help='Event frames, e.g. "raw_w10000_polarity" (default: protocol\'s '
                             "challenging.events)")
    parser.add_argument("--epochs", type=int, default=None, help="Override protocol final.epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Training seed (default protocol proxy.seed); repeat with others to confirm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_workers", type=int, default=4,
                        help="Processes downloading and rendering test sequences (each holds one "
                             "sequence's events in memory)")
    # Training and testing can run as two jobs when together they exceed the
    # Slurm time limit (10 ms test frames are rendered from raw events, ~10 h).
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--train_only", action="store_true",
                       help="Train and save final_runs/<name>/model.pt; skip the test")
    stage.add_argument("--test_only", action="store_true",
                       help="Score final_runs/<name>/model.pt from an earlier --train_only run")
    args = parser.parse_args()
    validate_gene_id(args.name)

    start = time.time()
    protocol = load_protocol(args.protocol)
    challenging, proxy = protocol["challenging"], protocol["proxy"]
    final = challenging["final"]
    epochs = args.epochs or final["epochs"]
    seed = proxy["seed"] if args.seed is None else args.seed
    data_root = FRED_DIR / challenging["data_root"]
    events = events_from_tag(args.events) if args.events else challenging.get("events")
    data_yaml = (data_root / f"final_train_step{final['train_frame_step']}{events_suffix(events)}"
                 / "data.yaml")
    if not data_yaml.is_file():
        raise SystemExit(f"{data_yaml} missing: run export_split.py --set final "
                         "(all shards, then --finalize)")
    _, test = challenging_split(data_root)
    out = data_root / "final_runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    adapter = UltralyticsYolo11Adapter(protocol["families"]["yolo11"], FRED_DIR)
    variant_path = adapter.variant_path(args.model, args.variant_dir)
    checkpoint, train_info = out / "model.pt", out / "train_info.json"
    if args.test_only:
        # The model must be the one --train_only trained for this name and setting.
        if not checkpoint.is_file() or not train_info.is_file():
            raise SystemExit(f"{checkpoint} or {train_info} missing: run with --train_only first")
        trained = json.loads(train_info.read_text())
        current = {"variant_identity": sha256_file(variant_path), "events": events}
        if any(trained[k] != v for k, v in current.items()):
            raise SystemExit(f"{train_info} was trained with {trained}, not {current}")
        if trained["checkpoint_identity"] != sha256_file(checkpoint):
            raise SystemExit(f"{checkpoint} changed since --train_only saved it")
        epochs, seed, args.weights = trained["epochs"], trained["seed"], trained["weights"]
        from ultralytics import YOLO
        model = YOLO(str(checkpoint), task="detect")
    else:
        model = adapter.build(variant_path, num_classes=1, device=args.device)
    params = adapter.param_count(model)
    print(f"{args.name}: {variant_path.name}, {params} params, weights={args.weights or 'scratch'}, "
          f"{epochs} epochs, seed {seed}" + (" (trained earlier)" if args.test_only else ""), flush=True)

    if not args.test_only:
        model.train(
            data=str(data_yaml), epochs=epochs, imgsz=final["img_width"], batch=final["batch_size"],
            seed=seed, deterministic=True, pretrained=args.weights or False,
            device=_ultralytics_device(args.device), workers=adapter.settings.get("workers", 8),
            project=str(out), name="train", exist_ok=True, val=False, plots=False, verbose=False,
        )
        model.save(checkpoint)
    if args.train_only:
        train_info.write_text(json.dumps({
            "variant_identity": sha256_file(variant_path), "events": events, "epochs": epochs,
            "seed": seed, "weights": args.weights, "checkpoint_identity": sha256_file(checkpoint),
        }, indent=2) + "\n")
        print(f"Saved model: {checkpoint}; score it with --test_only (same --name, --model, --events)")
        return

    settings = {"events": events, "img_width": final["img_width"], "eval_conf": proxy["eval_conf"],
                "nms_iou": proxy["nms_iou"], "batch_size": final["batch_size"],
                "test_frame_step": final["test_frame_step"]}
    frames, expected = predict_test_split(adapter, model, test, settings, args.device, out,
                                          args.test_workers)

    config = {"family": "yolo11", "final": final, "proxy_eval": settings, "epochs": epochs,
              "seed": seed,
              "weights": args.weights, "variant": variant_path.name}
    split_files = data_root / "splits"
    metadata = {
        "dataset_revision": REVISION,
        "manifest_identity": hashlib.sha256(b"".join(
            (split_files / f"{s}_split.txt").read_bytes()
            for s in ("challenging_train", "challenging_test"))).hexdigest(),
        "split_identity": (f"train=challenging_train(step {final['train_frame_step']});"
                           f"test=challenging_test(step {final['test_frame_step']})"),
        "annotation_policy": "fred_coordinates_txt_clipped_to_image",
        "model_identity": sha256_file(variant_path),
        "checkpoint_identity": sha256_file(checkpoint),
        "code_revision": code_revision(),
        "postprocessing": adapter.postprocessing(settings),
        "training_config_identity": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
        "modality": proxy["modality"],
        "benchmark_split": "challenging",
        "project_split": "held_out_test",
        "official_split": "challenging_test",
    }
    payload = {"metadata": metadata, "frames": frames, "params": params,
               "expected_sample_ids": expected}
    result, report_path = map_reporter.report(payload, sota_root=data_root, gene_id=args.name,
                                              mode="held-out")
    # --test_only loads the checkpoint without build(), so there is no workdir.
    if hasattr(adapter, "workdir"):
        shutil.rmtree(adapter.workdir, ignore_errors=True)
    print(f"challenging_test mAP50: {result.map50}, mAP50:95: {result.map50_95}, "
          f"Params: {result.params}, test frames: {len(frames)}, Time: {time.time() - start:.1f}s")
    print(f"Saved model: {checkpoint}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
