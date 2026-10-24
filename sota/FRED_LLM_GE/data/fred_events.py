"""Render FRED event frames from the raw events in each sequence's events.hdf5.

FRED's released Event/Frames PNGs are a fixed 33.3 ms accumulation. Rendering
from the raw events lets the accumulation window and the pixel encoding vary
while keeping the released frame times, so labels and sample IDs are unchanged.

With window_us=33333 and encoding="palette" the output is identical to the
released PNGs; check a downloaded sequence with

    uv run python sota/FRED_LLM_GE/data/fred_events.py verify <id>.zip

events.hdf5 is compressed with Prophesee's ECF codec, which h5py cannot read.
The codec (data/ecf/) is compiled with g++ on first use and called via ctypes.
"""

import ctypes
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from fred_stream import STEP_US, _select_frames, _write_report

ECF_DIR = Path(__file__).with_name("ecf")
ECF_LIB = ECF_DIR / "build" / "libecf.so"
WIDTH, HEIGHT = 1280, 720
# Metavision's dark palette, as used by FRED's released frames.
BACKGROUND, NEGATIVE, POSITIVE = (30, 37, 52), (64, 126, 200), (255, 255, 255)
ENCODINGS = ("palette", "polarity")


def _ecf_library():
    """Load the ECF decoder, compiling it first if needed."""
    if not ECF_LIB.is_file():
        ECF_LIB.parent.mkdir(exist_ok=True)
        # Per-process temp name: parallel export shards may compile at once.
        temp = ECF_LIB.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(
            ["g++", "-O2", "-std=c++14", "-shared", "-fPIC", f"-I{ECF_DIR}",
             str(ECF_DIR / "ecf_codec.cpp"), str(ECF_DIR / "ecf_capi.cpp"), "-o", str(temp)],
            check=True)
        os.replace(temp, ECF_LIB)
    lib = ctypes.CDLL(str(ECF_LIB))
    lib.ecf_decompressed_size.restype = ctypes.c_size_t
    lib.ecf_decompressed_size.argtypes = [ctypes.c_char_p]
    lib.ecf_decode.restype = ctypes.c_size_t
    lib.ecf_decode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p]
    return lib


def load_events(hdf5_path):
    """All CD events of an events.hdf5 as a structured array (x, y, p, t), sorted by t."""
    import h5py

    lib = _ecf_library()
    with h5py.File(hdf5_path, "r") as file:
        dataset = file["CD/events"]
        dtype, total = dataset.dtype, dataset.shape[0]
        events = np.empty(total, dtype)
        for i in range(dataset.id.get_num_chunks()):
            start = dataset.id.get_chunk_info(i).chunk_offset[0]
            _, raw = dataset.id.read_direct_chunk((start,))
            count = min(dataset.chunks[0], total - start)
            decoded = np.empty(lib.ecf_decompressed_size(raw) // dtype.itemsize, dtype)
            size = lib.ecf_decode(raw, len(raw), decoded.ctypes.data)
            if size // dtype.itemsize < count:
                raise ValueError(f"{hdf5_path}: chunk {i} decoded {size} bytes, "
                                 f"expected {count} events")
            events[start:start + count] = decoded[:count]
    if np.any(np.diff(events["t"]) < 0):
        raise ValueError(f"{hdf5_path}: event timestamps are not sorted")
    return events


def render(events, end_us, window_us=STEP_US, encoding="palette"):
    """RGB frame of the events with end_us - window_us < t < end_us.

    palette:  FRED's released look; each pixel shows its latest event's polarity
              (white positive, blue negative) on a dark background.
    polarity: red = any positive event, green = any negative event, blue = 0,
              so a pixel with both polarities keeps both.
    Both ends of the window are open: that reproduces the released frames exactly.
    """
    t = events["t"]
    lo = np.searchsorted(t, end_us - window_us, side="right")
    hi = np.searchsorted(t, end_us, side="left")
    window = events[lo:hi]
    y, x, positive = window["y"], window["x"], window["p"] == 1
    if encoding == "palette":
        image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
        image[:] = BACKGROUND
        # Fancy assignment keeps the last write per pixel, i.e. the latest event.
        image[y, x] = np.where(positive[:, None], POSITIVE, NEGATIVE)
    elif encoding == "polarity":
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        image[y[positive], x[positive], 0] = 255
        image[y[~positive], x[~positive], 1] = 255
    else:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    return Image.fromarray(image)


def zip_rendered_frames(zip_path, split, sequence_id, bad_timestamps_path=None,
                        frame_step=1, window_us=STEP_US, encoding="palette"):
    """Like fred_stream.zip_event_frames, but frames are rendered from events.hdf5.

    Frame times are the released frames' (same selection and timestamp checks),
    so every frame keeps its annotation.
    """
    sequence_id = int(sequence_id)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        frames, bad = _select_frames(sequence_id, members)
        _write_report(bad_timestamps_path, split, sequence_id, bad)
        hdf5 = f"{sequence_id}/Event/events.hdf5"
        with tempfile.TemporaryDirectory(dir=Path(zip_path).parent) as temp:
            events = load_events(archive.extract(hdf5, temp))
    for counter, _ in frames[::frame_step]:
        yield render(events, counter, window_us, encoding), counter / 1_000_000


def verify(zip_path):
    """Count released frames of a sequence zip that the palette renderer does not reproduce."""
    zip_path = Path(zip_path)
    sequence_id = int(zip_path.stem)
    mismatched, total = [], 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        frames, _ = _select_frames(sequence_id, members)
        with tempfile.TemporaryDirectory() as temp:
            events = load_events(archive.extract(f"{sequence_id}/Event/events.hdf5", temp))
        for counter, name in frames:
            with archive.open(name) as file:
                released = np.asarray(Image.open(file).convert("RGB"))
            pixels = int((np.asarray(render(events, counter)) != released).any(-1).sum())
            total += 1
            if pixels:
                mismatched.append((counter, pixels))
    return total, mismatched


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Check rendered frames against FRED's released frames")
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("zips", nargs="+", help="downloaded <sequence_id>.zip files")
    args = parser.parse_args()
    failed = False
    for path in args.zips:
        total, mismatched = verify(path)
        failed |= bool(mismatched)
        print(f"{path}: {total - len(mismatched)}/{total} frames identical"
              + (f"; first mismatches (counter, pixels): {mismatched[:5]}" if mismatched else ""))
    raise SystemExit(1 if failed else 0)
