"""Stream FRED event frames and check remote timestamp metadata."""

import csv
import re
from pathlib import Path

from datasets import Image, load_dataset

REVISION = "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd"
STEP_US = 33_333


def stream_event_frames(split, sequence_id, bad_timestamps_path=None):
    sequence_id = int(sequence_id)
    archive = (
        f"hf://datasets/GabrieleMagrini/FRED@{REVISION}/"
        f"{split}/{sequence_id}.zip"
    )
    rows = load_dataset(
        "imagefolder", data_files={split: archive}, split=split,
        streaming=True, drop_labels=True, drop_metadata=True,
    )
    rows = rows.cast_column("image", Image(decode=False))
    candidates, rgb, bad = [], [], []
    for row in rows:
        path = row["image"]["path"] or ""
        if "/RGB/" in path:
            match = re.search(
                rf"/RGB/Video_{sequence_id}_(\d{{2}})_(\d{{2}})_(\d{{2}})\.(\d{{6}})\.jpg::",
                path,
            )
            if match and all(
                int(value) < limit
                for value, limit in zip(match.groups()[:3], (24, 60, 60))
            ):
                hours, minutes, seconds, micros = map(int, match.groups())
                rgb_time = (hours * 3600 + minutes * 60 + seconds) * 1_000_000 + micros
                rgb.append((rgb_time, path))
            else:
                bad.append((path, "invalid_rgb_timestamp", "", ""))
            continue
        if "/Event/Frames/" not in path:
            continue
        pattern = rf"/Event/Frames/Video_{sequence_id}_frame_(\d+)\.png::"
        match = re.search(pattern, path)
        counter = int(match[1]) if match else None
        if counter is None or counter < STEP_US or counter % STEP_US:
            bad.append((path, "unexpected_counter", "", counter or ""))
            continue
        candidates.append((counter, path))
    candidates.sort()
    frames, expected = [], STEP_US
    for counter, path in candidates:
        if counter > expected:
            bad.append(("", "missing_counter", expected, counter))
        if counter < expected:
            bad.append((path, "duplicate_counter", expected, counter))
            continue
        frames.append((counter, path))
        expected = counter + STEP_US
    rgb.sort()
    if len(rgb) != len(frames):
        bad.append(("", "rgb_event_count_mismatch", len(frames), len(rgb)))
    for (previous, _), (current, path) in zip(rgb, rgb[1:]):
        gap = current - previous
        if not STEP_US / 2 <= gap <= STEP_US * 1.5:
            bad.append((path, "rgb_interval_mismatch", STEP_US, gap))
    report = Path(
        bad_timestamps_path or Path(__file__).with_name("bad_timestamps.csv")
    )
    with report.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((
            "split", "sequence_id", "member", "issue",
            "expected", "observed",
        ))
        writer.writerows((split, sequence_id, *finding) for finding in bad)
    for counter, path in frames:
        image = Image().decode_example({"path": path, "bytes": None})
        yield image, counter / 1_000_000


if __name__ == "__main__":
    from argparse import ArgumentParser
    from itertools import islice

    parser = ArgumentParser(description="Inspect a few remote FRED event frames")
    parser.add_argument("split")
    parser.add_argument("sequence_id", type=int)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    for image, timestamp_s in islice(
        stream_event_frames(args.split, args.sequence_id), args.limit
    ):
        print(f"{timestamp_s:.6f}s {image.size}")
