"""Frozen sequence-level option-B engineering workload; no held-out test access."""

import hashlib
import json
from pathlib import Path
import random

import yaml

FRED_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = FRED_ROOT / "configs/phase1/yolo11_option_b.json"


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
    expected_val = sorted(random.Random(0).sample(official["train"], 20), key=int)
    expected_train = [s for s in official["train"] if s not in expected_val]
    if train != expected_train or val != expected_val:
        raise ValueError("Option B must retain its frozen 152/20 sequence partition")
    if set(train + val) & set(official["test"]):
        raise ValueError("Option B includes official challenging-test sequences")
    if manifest["frame_step"] != 50:
        raise ValueError("Option B must sample every 50th frame")
    sources = manifest["archive_sources"]
    if set(sources) != set(official["train"]) or not set(sources.values()) <= {"train", "test"}:
        raise ValueError("Option B archive mapping must cover challenging-train exactly")
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest
