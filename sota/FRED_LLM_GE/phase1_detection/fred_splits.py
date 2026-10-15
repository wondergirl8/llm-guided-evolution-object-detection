"""Official FRED challenging-split membership and where each sequence is stored.

The sequence lists come from FRED's own repository at the revision pinned in
configs/phase0/default.yaml and are checked against the SHA-256 recorded there.
The Hugging Face train/ and test/ folders are not the challenging split (they
hold 184 and 47 sequences; challenging train/test are 172 and 59), so each
sequence's folder is looked up from the pinned dataset revision.
"""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import yaml

try:
    from .proxy_data import REVISION
except ImportError:
    from proxy_data import REVISION
from fred_stream import zip_event_frames  # on sys.path via proxy_data
from fred_events import zip_rendered_frames

FRED_DIR = Path(__file__).resolve().parents[1]
PHASE0_CONFIG = FRED_DIR / "configs" / "phase0" / "default.yaml"
SPLITS = ("challenging_train", "challenging_test")


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def parse_split_file(text):
    """FRED split files list one sequence per line as "<id>/"."""
    ids = [line.strip().rstrip("/") for line in text.splitlines() if line.strip()]
    return [str(int(i)) for i in ids]


def official_sequences(name, cache_dir):
    """Sequence IDs (strings) of an official challenging split, checksum-verified."""
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    source = yaml.safe_load(PHASE0_CONFIG.read_text())["fred_repository"]
    url, expected = source[f"{name}_url"], source[f"{name}_sha256"]
    path = Path(cache_dir) / "splits" / f"{name}_split.txt"
    if not path.is_file():
        with urllib.request.urlopen(url, timeout=60) as response:
            _atomic_write(path, response.read())
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"{name} split file checksum {actual} does not match {expected}")
    return parse_split_file(data.decode())


def hf_folders(cache_dir):
    """{sequence_id: "train" or "test"}: the Hugging Face folder holding each sequence."""
    path = Path(cache_dir) / "splits" / f"hf_folders_{REVISION}.json"
    if path.is_file():
        return json.loads(path.read_text())
    from huggingface_hub import HfFileSystem

    folders = {}
    for folder in ("train", "test"):
        for entry in HfFileSystem().ls(f"datasets/GabrieleMagrini/FRED@{REVISION}/{folder}",
                                       detail=False):
            if entry.endswith(".zip"):
                sequence = str(int(entry.rsplit("/", 1)[1][:-len(".zip")]))
                if sequence in folders:
                    raise ValueError(f"sequence {sequence} is in both Hugging Face folders")
                folders[sequence] = folder
    _atomic_write(path, json.dumps(folders, sort_keys=True).encode())
    return folders


def challenging_split(cache_dir):
    """(train, test) lists of (sequence_id, hf_folder); verifies they are disjoint and stored."""
    folders = hf_folders(cache_dir)
    train = official_sequences("challenging_train", cache_dir)
    test = official_sequences("challenging_test", cache_dir)
    if set(train) & set(test):
        raise ValueError("challenging train and test splits overlap")
    missing = [s for s in train + test if s not in folders]
    if missing:
        raise ValueError(f"sequences missing from the Hugging Face dataset: {missing}")
    return [(s, folders[s]) for s in train], [(s, folders[s]) for s in test]


def sequence_frames(folder, sequence, report, frame_step, download_dir, events=None):
    """Event frames of one sequence, from a temporary whole-zip download.

    Same frames and timestamp checks as fred_stream.stream_event_frames, but
    downloading the zip (~80 MB/s) is far faster than streaming its frames.
    The zip is deleted once the frames have been read. events is protocol.yaml's
    challenging.events: with source "raw", frames are rendered from
    events.hdf5 (data/fred_events.py) instead of read from the released PNGs.
    """
    from huggingface_hub import hf_hub_download

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    temp = tempfile.mkdtemp(prefix=f"{folder}_{sequence}.", dir=download_dir)
    try:
        archive = hf_hub_download("GabrieleMagrini/FRED", f"{folder}/{int(sequence)}.zip",
                                  repo_type="dataset", revision=REVISION, local_dir=temp)
        if events and events["source"] == "raw":
            yield from zip_rendered_frames(archive, folder, sequence, report, frame_step,
                                           events["window_us"], events["encoding"])
        else:
            yield from zip_event_frames(archive, folder, sequence, report, frame_step)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
