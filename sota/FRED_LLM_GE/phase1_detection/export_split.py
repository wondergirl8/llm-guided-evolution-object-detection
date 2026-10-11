"""Export FRED's official challenging_train split as Ultralytics datasets.

    # one or more CPU jobs, each taking every Nth sequence of the set
    uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --shard 0 --num_shards 4
    # once every shard is done
    uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --finalize

Sets (configs/phase1/protocol.yaml, `challenging`):
    search  search_train_step<k>: challenging_train minus search.val_sequences,
            search_val_step<k>: search.val_sequences (scored by candidate_runner.py)
    final   final_train_step<k>: all of challenging_train (for final_eval.py)

Each sequence is downloaded from Hugging Face and its event frames written at
their native 1280x720 size to <part>/images/<seq>_<frame>.png, with YOLO-format
labels in labels/ and the exact pixel boxes in done/<seq>.json. A sequence is
marked done only after all its frames are written, so a killed job can simply
be rerun. --finalize checks every sequence is done and writes data.yaml.
challenging_test is never exported; final_eval.py reads it from the downloads.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

try:
    from .fred_splits import challenging_split, sequence_frames
    from .proxy_data import load_annotations, STEP_US
except ImportError:
    from fred_splits import challenging_split, sequence_frames
    from proxy_data import load_annotations, STEP_US

FRED_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = FRED_DIR / "configs" / "phase1" / "protocol.yaml"
SETS = ("search", "final")


def yolo_label(boxes, width, height):
    """YOLO label text (class cx cy w h, normalized) for pixel xyxy boxes."""
    return "\n".join(
        f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        for x1, y1, x2, y2 in boxes)


def dataset_parts(challenging, data_root, name):
    """[(part_root, [(sequence, hf_folder)], frame_step)] for an export set."""
    train, _ = challenging_split(data_root)
    if name == "final":
        step = challenging["final"]["train_frame_step"]
        return [(data_root / f"final_train_step{step}", train, step)]
    search = challenging["search"]
    val_ids = {str(s) for s in search["val_sequences"]}
    unknown = val_ids - {s for s, _ in train}
    if unknown:
        raise ValueError(f"search val_sequences not in challenging_train: {sorted(unknown)}")
    train_step, val_step = search["train_frame_step"], search["val_frame_step"]
    return [
        (data_root / f"search_train_step{train_step}",
         [(s, f) for s, f in train if s not in val_ids], train_step),
        (data_root / f"search_val_step{val_step}",
         [(s, f) for s, f in train if s in val_ids], val_step),
    ]


def export_sequence(sequence, folder, frame_step, root):
    done = root / "done" / f"{sequence}.json"
    if done.is_file():
        return json.loads(done.read_text())["frames"]
    for sub in ("images", "labels", "done", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    report = root / "reports" / f"bad_timestamps_{sequence}.csv"
    frames = sequence_frames(folder, sequence, report, frame_step, root / "downloads")
    records, annotations = [], None
    for image, timestamp in frames:
        if annotations is None:
            # Clip boxes to the frame, as proxy_data does for the search data.
            annotations = load_annotations(folder, sequence, *image.size)
        frame_index = round(timestamp * 1_000_000 / STEP_US)
        name = f"{sequence}_{frame_index:06d}"
        boxes = annotations.get(frame_index, [])
        image.convert("RGB").save(root / "images" / f"{name}.png", compress_level=1)
        (root / "labels" / f"{name}.txt").write_text(yolo_label(boxes, *image.size))
        records.append({"name": name, "frame_index": frame_index, "boxes": boxes})
    if not records:
        raise ValueError(f"no event frames read for {folder}/{sequence}")
    temp = done.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps({"sequence": sequence, "hf_folder": folder,
                                "frame_step": frame_step, "frames": len(records),
                                "size": list(image.size), "records": records}))
    os.replace(temp, done)
    return len(records)


def finalize(sequences, root, val_frames):
    missing = [s for s, _ in sequences if not (root / "done" / f"{s}.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} sequences not exported yet in {root.name}: {missing}")
    names = sorted(p.stem for p in (root / "images").glob("*.png"))
    expected = sum(json.loads((root / "done" / f"{s}.json").read_text())["frames"]
                   for s, _ in sequences)
    if len(names) != expected:
        raise SystemExit(f"found {len(names)} images but done markers record {expected}")
    images = [str(root / "images" / f"{n}.png") for n in names]
    (root / "train.txt").write_text("\n".join(images) + "\n")
    # Evenly spaced frames for Ultralytics' end-of-training check only.
    step = max(1, len(images) // val_frames)
    (root / "val.txt").write_text("\n".join(images[::step][:val_frames]) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(root), "train": "train.txt", "val": "val.txt",
         "names": {0: "drone"}}, sort_keys=False))
    print(f"{root.name}: {len(sequences)} sequences, {len(images)} frames -> {root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Export FRED challenging_train for Ultralytics")
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("need 0 <= shard < num_shards")

    challenging = yaml.safe_load(Path(args.protocol).read_text())["challenging"]
    data_root = FRED_DIR / challenging["data_root"]
    parts = dataset_parts(challenging, data_root, args.set)
    if args.finalize:
        for root, sequences, _ in parts:
            finalize(sequences, root, challenging["trainer_val_frames"])
        return
    work = [(root, s, f, step) for root, sequences, step in parts for s, f in sequences]
    mine = work[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {len(work)} sequences",
          flush=True)
    for root, sequence, folder, step in mine:
        count = export_sequence(sequence, folder, step, root)
        print(f"{root.name} {folder}/{sequence}: {count} frames", flush=True)


if __name__ == "__main__":
    main()
