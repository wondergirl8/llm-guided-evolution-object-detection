"""Stable identity for the inputs to a FRED evolution run."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import yaml


def compute_fingerprint(root, family, run_tag, seed_path):
    root = Path(root)
    fred_root = root / "sota" / "FRED_LLM_GE"
    protocol_path = fred_root / "configs" / "phase1" / "protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    files = {"protocol": protocol_path, "seed": Path(seed_path)}
    if protocol["families"][family].get("data") == "challenging_search":
        data_root = fred_root / protocol["challenging"]["data_root"]
        search = protocol["challenging"]["search"]
        files.update({
            "challenging_train": data_root / "splits" / "challenging_train_split.txt",
            "search_train": data_root / f"search_train_step{search['train_frame_step']}" / "train.txt",
            "search_val": data_root / f"search_val_step{search['val_frame_step']}" / "train.txt",
        })
    file_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in files.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    identity = {"files": file_hashes, "family": family, "run_tag": run_tag,
                "git_commit": commit, "torch_version": importlib.metadata.version("torch")}
    if family == "yolo11":
        identity["ultralytics_version"] = importlib.metadata.version("ultralytics")
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def verify_fingerprint(record, path, expected):
    """Refuse legacy or incompatible checkpoints before their data is resumed."""
    if expected is not None and record.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"FRED checkpoint fingerprint missing or mismatched: {path}; "
                           "use the original protocol or a fresh run tag")
