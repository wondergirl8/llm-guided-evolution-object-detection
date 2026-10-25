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
    if family == "bn_snn":
        cache = fred_root / protocol["families"][family]["raw_cache"]
        files.update({"raw_train": cache / "train_manifest.json",
                      "raw_val": cache / "val_manifest.json"})
        for name in ("candidate_runner.py", "bn_snn_data.py", "bn_snn_search_space.py",
                     "adapters/bn_snn.py", "seeds/bn_snn/network.py", "seeds/bn_snn/detection.py",
                     "bn_snn_baseline.py", "map_reporter.py", "fitness.py", "result_contract.py",
                     "proxy_data.py", "fred_splits.py", "export_split.py"):
            files[name] = fred_root / "phase1_detection" / name
        for name in ("data/fred_events.py", "data/fred_stream.py", "configs/phase0/default.yaml"):
            files[name] = fred_root / name
    if family == "ems_yolo":
        for name in ("candidate_runner.py", "ems_yolo_search_space.py",
                     "adapters/ems_yolo.py", "seeds/ems_yolo/network.py",
                     "seeds/ems_yolo/detection.py", "ems_yolo_baseline.py",
                     "map_reporter.py", "fitness.py", "result_contract.py",
                     "proxy_data.py", "fred_splits.py", "export_split.py"):
            files[name] = fred_root / "phase1_detection" / name
        upstream = fred_root / "third_party/ems_yolo"
        for path in sorted(upstream.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[f"upstream/{path.relative_to(upstream)}"] = path
    file_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in files.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    identity = {"files": file_hashes, "family": family, "run_tag": run_tag,
                "git_commit": commit, "torch_version": importlib.metadata.version("torch")}
    if family == "yolo11":
        identity["ultralytics_version"] = importlib.metadata.version("ultralytics")
    if family == "bn_snn":
        identity["dependencies"] = {name: importlib.metadata.version(name)
                                    for name in ("torchvision", "numpy", "h5py", "pycocotools")}
    if family == "ems_yolo":
        identity["dependencies"] = {name: importlib.metadata.version(name)
                                    for name in ("torchvision", "numpy", "pycocotools")}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def verify_fingerprint(record, path, expected):
    """Refuse legacy or incompatible checkpoints before their data is resumed."""
    if expected is not None and record.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"FRED checkpoint fingerprint missing or mismatched: {path}; "
                           "use the original protocol or a fresh run tag")
