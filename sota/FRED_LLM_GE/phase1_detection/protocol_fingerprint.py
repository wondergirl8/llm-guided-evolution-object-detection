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
    if family in ("yolo26_p2", "dfine"):
        # These families previously resumed tagged checkpoints without any
        # fingerprint. Include local edits, not only the Git commit, because
        # a PACE checkout may have uncommitted experiment code.
        adapter = ("ultralytics_yolo26_p2" if family == "yolo26_p2" else "dfine")
        proposal = ("fred_yolo26_proposals" if family == "yolo26_p2"
                    else "fred_dfine_proposals")
        files.update({
            "controller": root / "run_improved.py",
            "family_constants": root / "src/cfg/constants_fred_yolo11.py",
            "llm_utils": root / "src/llm_utils.py",
            "candidate_runner": fred_root / "phase1_detection/candidate_runner.py",
            "export_split": fred_root / "phase1_detection/export_split.py",
            "fitness": fred_root / "phase1_detection/fitness.py",
            "map_reporter": fred_root / "phase1_detection/map_reporter.py",
            "adapter": fred_root / f"phase1_detection/adapters/{adapter}.py",
            "mutation": root / "src/llm_mutation.py",
            "crossover": root / "src/llm_crossover.py",
            "proposal": root / f"src/{proposal}.py",
            "slurm_config": root / "slurm-config/slurm_config.yaml",
        })
        if family == "dfine":
            files.update({
                "dfine_genome": root / "src/fred_dfine_genome.py",
                "dfine_selection": root / "src/fred_dfine_selection.py",
            })
        prompt_group = "FRED_YOLO26_P2" if family == "yolo26_p2" else "FRED_DFINE"
        for path in sorted((root / "templates" / prompt_group).rglob("*.txt")):
            files[f"prompt:{path.relative_to(root)}"] = path
        for path in sorted((root / "templates/CrossOver").glob("*.txt")):
            files[f"prompt:{path.relative_to(root)}"] = path
    file_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in files.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    identity = {"files": file_hashes, "family": family, "run_tag": run_tag,
                "git_commit": commit, "torch_version": importlib.metadata.version("torch")}
    if family in ("yolo11", "yolo26_p2"):
        identity["ultralytics_version"] = importlib.metadata.version("ultralytics")
    if family == "dfine":
        identity["scipy_version"] = importlib.metadata.version("scipy")
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def verify_fingerprint(record, path, expected):
    """Refuse legacy or incompatible checkpoints before their data is resumed."""
    if expected is not None and record.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"FRED checkpoint fingerprint missing or mismatched: {path}; "
                           "use the original protocol or a fresh run tag")
