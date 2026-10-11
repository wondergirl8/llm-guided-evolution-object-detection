"""Stream FRED event frames and check remote timestamp metadata."""

import csv
import re
from pathlib import Path
import zipfile

from datasets import Image, load_dataset
from huggingface_hub import HfFileSystem
from PIL import Image as PILImage

REVISION = "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd"
STEP_US = 33_333


def indexed_archive_paths(archive, source=None):
    """Read the remote ZIP directory, without reading every RGB/event image."""
    if source is not None:
        for member in source.namelist():
            if member.endswith((".png", ".jpg")):
                yield f"zip://{member}::{archive}"
        return
    with HfFileSystem().open(archive.removeprefix("hf://"), "rb", block_size=65536) as handle:
        with zipfile.ZipFile(handle) as source:
            yield from indexed_archive_paths(archive, source)


def stream_event_frames(split, sequence_id, bad_timestamps_path=None, frame_step=1,
                        *, use_archive_index=False):
    sequence_id = int(sequence_id)
    archive = (
        f"hf://datasets/GabrieleMagrini/FRED@{REVISION}/"
        f"{split}/{sequence_id}.zip"
    )
    if use_archive_index:
        with HfFileSystem().open(archive.removeprefix("hf://"), "rb", block_size=65536) as handle:
            with zipfile.ZipFile(handle) as source:
                def decode(path):
                    member = path.split("::", 1)[0].removeprefix("zip://")
                    with source.open(member) as file:
                        with PILImage.open(file) as image:
                            return image.copy()
                yield from checked_event_frames(split, sequence_id, bad_timestamps_path, frame_step,
                                                indexed_archive_paths(archive, source), decode)
        return
    rows = load_dataset(
        "imagefolder", data_files={split: archive}, split=split,
        streaming=True, drop_labels=True, drop_metadata=True,
    )
    rows = rows.cast_column("image", Image(decode=False))
    paths = (row["image"]["path"] or "" for row in rows)
    yield from checked_event_frames(split, sequence_id, bad_timestamps_path, frame_step, paths,
                                    lambda path: Image().decode_example({"path": path, "bytes": None}))



def zip_event_frames(zip_path, split, sequence_id, bad_timestamps_path=None, frame_step=1):
    """Reuse timestamp checks for the team's temporary whole-ZIP CPU export."""
    sequence_id = int(sequence_id)
    with zipfile.ZipFile(zip_path) as archive:
        def decode(path):
            member = path.split("::", 1)[0].removeprefix("zip://")
            with archive.open(member) as file:
                with PILImage.open(file) as image:
                    return image.copy()
        yield from checked_event_frames(split, sequence_id, bad_timestamps_path, frame_step,
                                        indexed_archive_paths(str(zip_path), archive), decode)

def checked_event_frames(split, sequence_id, bad_timestamps_path, frame_step, paths, decode):
    candidates, rgb, bad = [], [], []
    for path in paths:
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
        pattern = rf"/Event/Frames/Video_{sequence_id}_(?:frame_)?(\d+)\.png::"
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
    # frame_step > 1 decodes only every Nth valid frame (timestamp checks above
    # still cover the whole sequence).
    for counter, path in frames[::frame_step]:
        image = decode(path)
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
