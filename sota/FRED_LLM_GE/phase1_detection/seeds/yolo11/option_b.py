"""Frozen sequence-level option-B engineering workload; no held-out test access."""

import hashlib
import json
import sys
from pathlib import Path

import yaml

FRED_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = FRED_ROOT / "configs/phase1/yolo11_option_b.json"
PROTOCOL_PATH = FRED_ROOT / "configs/phase1/protocol.yaml"


def load_manifest():
    payload = MANIFEST_PATH.read_bytes()
    manifest = json.loads(payload)
    pinned = yaml.safe_load((FRED_ROOT / "configs/phase0/default.yaml").read_text())
    source = pinned["fred_repository"]
    if (manifest["dataset_revision"] != pinned["source"]["revision"] or
            manifest["source_repository_revision"] != source["revision"]):
        raise ValueError("Option B differs from the pinned FRED revisions")
    official = {}
    for split in ("train", "test"):
        text = manifest[f"official_{split}_text"]
        if hashlib.sha256(text.encode()).hexdigest() != source[f"challenging_{split}_sha256"]:
            raise ValueError(f"Official challenging-{split} checksum mismatch")
        official[split] = [s.strip().rstrip("/") for s in text.splitlines() if s.strip()]
    train, val = manifest["train_sequences"], manifest["val_sequences"]
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    search = protocol["challenging"]["search"]
    expected_val = [str(s) for s in search["val_sequences"]]
    expected_train = [s for s in official["train"] if s not in expected_val]
    if train != expected_train or val != expected_val:
        raise ValueError("Option B must retain its frozen 152/20 sequence partition")
    if set(train + val) & set(official["test"]):
        raise ValueError("Option B includes official challenging-test sequences")
    if (manifest["frame_step"] != search["train_frame_step"] or
            manifest["val_frame_step"] != search["val_frame_step"]):
        raise ValueError("Option B must sample every 50th frame")
    sources = manifest["archive_sources"]
    if set(sources) != set(official["train"]) or not set(sources.values()) <= {"train", "test"}:
        raise ValueError("Option B archive mapping must cover challenging-train exactly")
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest


def load_exported_data(manifest, img_width):
    """Reuse the team CPU export; images remain on disk until a batch is needed."""
    repo_root = str(FRED_ROOT.parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from sota.FRED_LLM_GE.phase1_detection.proxy_data import load_exported_split

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    data_root = FRED_ROOT / protocol["challenging"]["data_root"]
    parts = []
    for kind, step in (("train", manifest["frame_step"]), ("val", manifest["val_frame_step"])):
        root = data_root / f"search_{kind}_step{step}"
        if not (root / "data.yaml").is_file():
            raise FileNotFoundError(f"{root}/data.yaml missing: run scripts/prepare_yolo11_option_b.sh first")
        sequences = [(s, manifest["archive_sources"][s]) for s in manifest[f"{kind}_sequences"]]
        data = load_exported_split(root, sequences, frame_step=step)
        if data["orig_size"][0] != img_width:
            raise ValueError(f"Option B native exported width is {data['orig_size'][0]}, requested {img_width}")
        parts.append(data)
    return tuple(parts)
