"""Keep each challenging_train sequence's raw events locally, for rendering frames later.

    # 1. one or more CPU jobs, each taking every Nth sequence: download the events
    uv run python sota/FRED_LLM_GE/phase1_detection/cache_events.py --shard 0 --num_shards 4
    # 2. same sharding: keep only the events the search split's frames can use
    uv run python sota/FRED_LLM_GE/phase1_detection/cache_events.py --slice --shard 0 --num_shards 4
    # check both steps are complete
    uv run python sota/FRED_LLM_GE/phase1_detection/cache_events.py --check

Step 1 downloads each sequence zip, keeps only <id>/Event/events.hdf5 as
<data_root>/events/<id>.hdf5 (the rest is RGB and the released PNGs) and
deletes the zip. A dense sequence's file is over 1 GB and decodes to several
GB, too much to decode for every renderer LLM-GE tries, so step 2 keeps only
the events within MAX_WINDOW_US before some frame of the search export, as
<data_root>/events_search/<id>.npy (a structured array like load_events gives).
render_runner.py renders from those. Frame times and labels need no caching:
the released search export names every image <seq>_<frame_index>, and a frame
ends at frame_index * STEP_US.

A file appears only once fully written, so a killed job can simply be rerun.
"""

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import yaml

try:
    from .export_split import DEFAULT_PROTOCOL, FRED_DIR, STEP_US, dataset_parts
    from .fred_splits import challenging_split, downloaded_sequence
except ImportError:
    from export_split import DEFAULT_PROTOCOL, FRED_DIR, STEP_US, dataset_parts
    from fred_splits import challenging_split, downloaded_sequence
from fred_events import iter_event_chunks  # on sys.path via proxy_data

# Longest window a renderer may use (render_runner.py enforces it).
MAX_WINDOW_US = 50_000
RELEASED = {"source": "released", "window_us": STEP_US, "encoding": "palette"}


def events_dir(challenging):
    return FRED_DIR / challenging["data_root"] / "events"


def slices_dir(challenging):
    return FRED_DIR / challenging["data_root"] / "events_search"


def cache_sequence(sequence, folder, out_dir, download_dir):
    """Extract one sequence's events.hdf5 to out_dir/<id>.hdf5; returns its size in bytes."""
    target = out_dir / f"{sequence}.hdf5"
    if target.is_file():
        return target.stat().st_size
    member = f"{int(sequence)}/Event/events.hdf5"
    with downloaded_sequence(folder, sequence, download_dir) as archive:
        with zipfile.ZipFile(archive) as zf:
            expected = zf.getinfo(member).file_size
            temp = target.with_suffix(f".{os.getpid()}.tmp")
            with zf.open(member) as src, open(temp, "wb") as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
    written = temp.stat().st_size
    if written != expected:
        temp.unlink()
        raise ValueError(f"{folder}/{sequence}: extracted {written} bytes, expected {expected}")
    os.replace(temp, target)
    return expected


def used_by_frames(t, ends, max_window_us=MAX_WINDOW_US):
    """Mask of event times t with end - max_window_us < t < end for some frame end (ends sorted)."""
    following = np.searchsorted(ends, t, side="right")  # first frame ending after each event
    inside = following < len(ends)
    keep = np.zeros(len(t), bool)
    keep[inside] = ends[following[inside]] - t[inside] < max_window_us
    return keep


def frame_ends(challenging):
    """{sequence: sorted frame end times in us} of the released search export."""
    data_root = FRED_DIR / challenging["data_root"]
    ends = {}
    for root, sequences, _ in dataset_parts(dict(challenging, events=RELEASED), data_root, "search"):
        for sequence, _ in sequences:
            done = root / "done" / f"{sequence}.json"
            if not done.is_file():
                raise FileNotFoundError(f"{done} missing: run export_split.py --set search")
            records = json.loads(done.read_text())["records"]
            ends[sequence] = np.array(sorted(r["frame_index"] * STEP_US for r in records), np.int64)
    return ends


def slice_sequence(hdf5, ends, target):
    """Write the events of hdf5 that the frames ending at ends can use; returns (kept, total)."""
    if target.is_file():
        kept = len(np.load(target, mmap_mode="r"))
        return kept, None
    parts, total, last = [], 0, None
    for chunk in iter_event_chunks(hdf5):
        t = chunk["t"]
        if len(t) and ((last is not None and t[0] < last) or np.any(np.diff(t) < 0)):
            raise ValueError(f"{hdf5}: event timestamps are not sorted")
        last = t[-1] if len(t) else last
        total += len(chunk)
        parts.append(chunk[used_by_frames(t, ends)])
    events = np.concatenate(parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(temp, events)
    os.replace(temp, target)
    return len(events), total


def main():
    parser = argparse.ArgumentParser(description="Cache FRED raw events for challenging_train")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--slice", action="store_true",
                        help="keep only the events the search frames can use (after downloading)")
    parser.add_argument("--check", action="store_true", help="only report which sequences are missing")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("need 0 <= shard < num_shards")

    challenging = yaml.safe_load(Path(args.protocol).read_text())["challenging"]
    data_root = FRED_DIR / challenging["data_root"]
    out_dir, slice_dir = events_dir(challenging), slices_dir(challenging)
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_dir.mkdir(parents=True, exist_ok=True)
    train, _ = challenging_split(data_root)

    if args.check:
        failed = False
        for directory, suffix in ((out_dir, ".hdf5"), (slice_dir, ".npy")):
            missing = [s for s, _ in train if not (directory / f"{s}{suffix}").is_file()]
            total = sum(p.stat().st_size for p in directory.glob(f"*{suffix}"))
            print(f"{len(train) - len(missing)}/{len(train)} sequences in {directory} ({total / 1e9:.1f} GB)"
                  + (f"; missing: {missing}" if missing else ""))
            failed |= bool(missing)
        raise SystemExit(1 if failed else 0)

    mine = train[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {len(train)} sequences", flush=True)
    ends = frame_ends(challenging) if args.slice else None
    for sequence, folder in mine:
        if args.slice:
            kept, total = slice_sequence(out_dir / f"{sequence}.hdf5", ends[sequence],
                                         slice_dir / f"{sequence}.npy")
            shown = f"{kept:,} events" + (f" of {total:,} ({kept / total:.1%})" if total else " (already sliced)")
            print(f"{folder}/{sequence}: {shown}", flush=True)
        else:
            size = cache_sequence(sequence, folder, out_dir, data_root / "downloads")
            print(f"{folder}/{sequence}: {size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
