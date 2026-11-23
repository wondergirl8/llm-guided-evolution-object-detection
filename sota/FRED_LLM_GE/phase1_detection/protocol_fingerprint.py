"""Stable identity for the inputs to a FRED evolution run."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import yaml


def scratch_code_identity(root):
    """Freeze source before queued exports/baselines start, including dirty edits."""
    root = Path(root)
    paths = [root / "run_improved.py", root / "server.py"]
    for directory in ("src", "scripts/faster_rcnn_scratch", "templates/FRED_FasterRCNNScratch",
                      "sota/FRED_LLM_GE/phase1_detection", "sota/FRED_LLM_GE/configs",
                      "sota/FRED_LLM_GE/data", "sota/FRED_LLM_GE/phase0_data"):
        for parent, directories, files in os.walk(root / directory):
            directories[:] = [name for name in directories if name not in (
                "data_cache", "models", "results", "reports", "held_out_reports",
                "trained_models", "final_results", "__pycache__")]
            paths.extend(Path(parent) / name for name in files if Path(name).suffix in (
                ".py", ".yaml", ".sh", ".sbatch", ".lock", ".txt"))
    paths.append(root / "sota/FRED_LLM_GE/seeds/seed_faster_rcnn_scratch.py")
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


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
    if family == "faster_rcnn_scratch":
        if not run_tag.startswith("faster_rcnn_scratch_"):
            raise ValueError("Faster R-CNN official-split runs require a fresh FRED_RUN_TAG")
        snapshot = root / "fred_runs" / run_tag / "code_snapshot.json"
        code = scratch_code_identity(root)
        if snapshot.exists() and json.loads(snapshot.read_text()) != code:
            raise ValueError("Source changed since launch; use a new run tag or restore the frozen code")
        python = root / ".venv-faster-rcnn/bin/python"
        dependencies = ("torch", "torchvision", "mmdet", "mmcv", "mmengine", "numpy", "pycocotools")
        command = ("import importlib.metadata as m,json; print(json.dumps("
                   + "{n:m.version(n) for n in " + repr(dependencies) + "}))")
        identity["detector_versions"] = json.loads(subprocess.check_output(
            [str(python), "-c", command], text=True))
        identity["torch_version"] = identity["detector_versions"]["torch"]
        identity["source_hashes"] = code
        data_root = fred_root / protocol["challenging"]["data_root"]
        from .proxy_data import REVISION
        identity["dataset_revision"] = REVISION
        manifests = [data_root / "splits/challenging_test_split.txt",
                     data_root / "splits" / f"hf_folders_{REVISION}.json"]
        for part in ("search_train", "search_val"):
            step = protocol["challenging"]["search"][
                "train_frame_step" if part == "search_train" else "val_frame_step"]
            manifests += sorted((data_root / f"{part}_step{step}" / "done").glob("*.json"))
        identity["export_hashes"] = {str(p.relative_to(data_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in manifests}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def verify_fingerprint(record, path, expected):
    """Refuse legacy or incompatible checkpoints before their data is resumed."""
    if expected is not None and record.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"FRED checkpoint fingerprint missing or mismatched: {path}; "
                           "use the original protocol or a fresh run tag")
