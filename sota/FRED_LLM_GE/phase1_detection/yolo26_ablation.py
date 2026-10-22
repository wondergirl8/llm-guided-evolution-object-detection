"""Run one controlled YOLO26 pilot ablation on the search-validation export.

Experiments are separate from LLM-GE and the historical YOLO26 result root.
Use --check to validate and build every architecture without FRED data or a GPU.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path

import torch
import yaml

try:
    from . import audit_yolo26_data_path, candidate_runner, map_reporter
    from .adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter
except ImportError:
    import audit_yolo26_data_path
    import candidate_runner
    import map_reporter
    from adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter


FRED_DIR = candidate_runner.FRED_DIR
DEFAULT_PLAN = FRED_DIR / "configs/phase1/yolo26_ablations.yaml"
DEFAULT_OUTPUT = FRED_DIR / "phase1_detection/seeds/yolo26_p2/ablations"
TRAIN_OPTIONS = {"optimizer", "lr0", "momentum", "warmup_bias_lr", "mosaic", "hsv_s"}


def architecture(plan, case):
    """Apply at most one channel edit to a verified source YAML."""
    source = FRED_DIR / plan["sources"][case["model"]]
    config = yaml.safe_load(source.read_text())
    edit = case.get("edit")
    if edit:
        if set(edit) != {"section", "index", "channels"}:
            raise ValueError("an architecture edit needs section, index and channels")
        section, index, channels = edit["section"], edit["index"], edit["channels"]
        if section not in ("backbone", "head") or type(index) is not int:
            raise ValueError("architecture edit needs a backbone/head layer index")
        if type(channels) is not int or not 0 <= index < len(config[section]):
            raise ValueError("architecture edit has invalid channels or layer index")
        old = config[section][index][3][0]
        if old == channels:
            raise ValueError("architecture edit must change the channel count")
        config[section][index][3][0] = channels
    return config, source


def load_plan(path, protocol):
    plan = yaml.safe_load(Path(path).read_text())
    cases = plan["experiments"]
    seeds = plan["repeat_seeds"]
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(
            type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("repeat_seeds must contain distinct nonnegative integers")
    settings = protocol["families"]["yolo26_p2"]
    adapter = UltralyticsYolo26P2Adapter(settings, FRED_DIR)
    for name, case in cases.items():
        if set(case) - {"model", "compare_to", "epochs", "train", "edit"}:
            raise ValueError(f"{name}: unknown experiment setting")
        config, _ = architecture(plan, case)
        adapter._validate_candidate(config)
        epochs = case.get("epochs", protocol["challenging"]["search"]["epochs"])
        if type(epochs) is not int or epochs <= 0:
            raise ValueError(f"{name}: epochs must be a positive integer")
        options = case.get("train", {})
        if not isinstance(options, dict) or set(options) - TRAIN_OPTIONS:
            raise ValueError(f"{name}: unsupported training setting")
        for key, value in options.items():
            if key == "optimizer" and value not in ("auto", "SGD", "AdamW"):
                raise ValueError(f"{name}: unsupported optimizer")
            if key == "lr0" and (type(value) not in (float, int) or not 0 < value <= 1):
                raise ValueError(f"{name}: lr0 must be in (0, 1]")
            if key == "momentum" and (type(value) not in (float, int) or not 0 < value < 1):
                raise ValueError(f"{name}: momentum must be in (0, 1)")
            if key == "warmup_bias_lr" and (
                    type(value) not in (float, int) or not 0 <= value <= 1):
                raise ValueError(f"{name}: warmup_bias_lr must be in [0, 1]")
            if key in ("mosaic", "hsv_s") and (
                    type(value) not in (float, int) or not 0 <= value <= 1):
                raise ValueError(f"{name}: {key} must be in [0, 1]")
        if "lr0" in options and options.get("optimizer", "auto") == "auto":
            raise ValueError(f"{name}: optimizer=auto ignores lr0; choose an explicit optimizer")
        reference = case.get("compare_to")
        if reference is None:
            continue  # independent reference recipes
        if reference not in cases or cases[reference]["model"] != case["model"]:
            raise ValueError(f"{name}: comparison must use the same source model")
        previous = cases[reference]
        previous_config, _ = architecture(plan, previous)
        layer_changes = sum(a != b for section in ("backbone", "head") for a, b in zip(
            config[section], previous_config[section]))
        epoch_changes = epochs != previous.get("epochs", protocol["challenging"]["search"]["epochs"])
        previous_options = previous.get("train", {})
        option_changes = sum(options.get(key) != previous_options.get(key) for key in
                             set(options) | set(previous_options))
        if layer_changes + epoch_changes + option_changes != 1:
            raise ValueError(f"{name}: comparison must change exactly one factor")
    return plan


def check_architectures(plan, protocol):
    """Build each distinct architecture once before expensive training."""
    from ultralytics import YOLO

    adapter = UltralyticsYolo26P2Adapter(protocol["families"]["yolo26_p2"], FRED_DIR)
    parameter_counts = {}
    for name, case in plan["experiments"].items():
        config, _ = architecture(plan, case)
        model_yaml = yaml.safe_dump(config, sort_keys=False)
        if model_yaml not in parameter_counts:
            with tempfile.TemporaryDirectory(prefix="yolo26_ablation_check_") as folder:
                path = Path(folder) / "yolo26n-p2.yaml"
                path.write_text(model_yaml)
                model = YOLO(str(path), task="detect", verbose=False)
                parameter_counts[model_yaml] = adapter.param_count(model)
        print(f"{name}: {parameter_counts[model_yaml]} parameters", flush=True)


def run_experiment(plan, plan_path, protocol, protocol_path, name, seed, output_root, device):
    cases = plan["experiments"]
    if name not in cases or seed not in plan["repeat_seeds"]:
        raise ValueError("choose a listed experiment and repeat seed")
    run_dir = Path(output_root) / name / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; choose a fresh output root")

    case = cases[name]
    config, source = architecture(plan, case)
    code_files = ("yolo26_ablation.py", "audit_yolo26_data_path.py",
                  "candidate_runner.py", "fitness.py",
                  "map_reporter.py", "adapters/ultralytics_yolo11.py",
                  "adapters/ultralytics_yolo26_p2.py")
    code_sha = hashlib.sha256(b"".join(
        (Path(__file__).parent / file).read_bytes() for file in code_files)).hexdigest()
    plan_sha = candidate_runner.sha256_file(plan_path)
    protocol_sha = candidate_runner.sha256_file(protocol_path)
    source_sha = candidate_runner.sha256_file(source)
    data_root = FRED_DIR / protocol["challenging"]["data_root"]
    data_path_audit = audit_yolo26_data_path.audit_search(data_root, protocol)
    if any(data_path_audit[split]["issue_count"] for split in ("train", "search_val")):
        raise ValueError("search export data-path audit found mismatches; "
                         "run audit_yolo26_data_path.py for details")
    proxy, train_data, load_val = candidate_runner.challenging_search(protocol, "yolo26_p2")
    export_sha = candidate_runner.sha256_file(train_data["data_yaml"])
    val_data = load_val()
    proxy.update(seed=seed, epochs=case.get("epochs", proxy["epochs"]))
    settings = dict(protocol["families"]["yolo26_p2"],
                    training_overrides=case.get("train", {}))
    adapter = UltralyticsYolo26P2Adapter(settings, FRED_DIR)
    torch.manual_seed(seed)

    run_dir.mkdir(parents=True)
    audit_file = run_dir / "data_path_audit.json"
    audit_file.write_text(json.dumps(data_path_audit, indent=2) + "\n")
    variant = run_dir / "model.yaml"
    variant.write_text(yaml.safe_dump(config, sort_keys=False))
    model = adapter.build(variant, num_classes=1, device=device)
    adapter.workdir = run_dir / "ultralytics"
    params = adapter.param_count(model)
    initial_weights = []
    model.add_callback("on_train_start", lambda trainer: initial_weights.append(
        candidate_runner.initial_weights_sha256(trainer.model)))
    start = time.perf_counter()
    adapter.train(model, train_data, proxy, device)
    training_seconds = time.perf_counter() - start
    if len(initial_weights) != 1:
        raise RuntimeError("could not capture the model state before training")

    trainer_dir = adapter.workdir / "train"
    args_file, curve_file = trainer_dir / "args.yaml", trainer_dir / "results.csv"
    if not args_file.is_file() or not curve_file.is_file():
        raise RuntimeError("Ultralytics did not save resolved arguments and training curves")
    with curve_file.open(newline="") as handle:
        completed_epochs = sum(1 for _ in csv.DictReader(handle))
    if completed_epochs != proxy["epochs"]:
        raise RuntimeError(f"training stopped after {completed_epochs} of {proxy['epochs']} epochs")
    resolved_optimizer = type(model.trainer.optimizer).__name__
    resolved_lr0 = model.trainer.args.lr0
    optimizer_group = model.trainer.optimizer.param_groups[0]
    resolved_momentum = (optimizer_group["betas"][0] if "betas" in optimizer_group
                         else optimizer_group["momentum"])
    resolved_warmup_bias_lr = model.trainer.args.warmup_bias_lr
    requested = case.get("train", {})
    if ("optimizer" in requested and resolved_optimizer != requested["optimizer"]
            or "lr0" in requested and resolved_lr0 != requested["lr0"]
            or "momentum" in requested and resolved_momentum != requested["momentum"]
            or "warmup_bias_lr" in requested
            and resolved_warmup_bias_lr != requested["warmup_bias_lr"]):
        raise RuntimeError(f"resolved optimizer settings differ from {name}: "
                           f"{resolved_optimizer}, lr0={resolved_lr0}, "
                           f"momentum={resolved_momentum}, "
                           f"warmup_bias_lr={resolved_warmup_bias_lr}")
    checkpoint = run_dir / "trained_models" / "ablation.pt"
    checkpoint.parent.mkdir()
    adapter.save(model, checkpoint)
    frames = candidate_runner.predict_frames(adapter, model, val_data, proxy, device)
    lineage = {"family": "yolo26_p2", "proxy": proxy, "family_settings": settings,
               "challenging": protocol["challenging"],
               "experiment": name, "compare_to": case.get("compare_to"),
               "source_model_sha256": source_sha}
    metadata = candidate_runner.report_metadata(adapter, proxy, lineage, variant, checkpoint)
    metadata.update(ablation_experiment=name, ablation_comparison=case.get("compare_to"),
                    ablation_seed=seed, source_model_sha256=source_sha,
                    ablation_code_sha256=code_sha, ablation_plan_sha256=plan_sha,
                    ablation_protocol_sha256=protocol_sha,
                    training_export_sha256=export_sha,
                    training_marker_sha256=data_path_audit["train"]["marker_sha256"],
                    training_label_sha256=data_path_audit["train"]["label_sha256"],
                    validation_marker_sha256=data_path_audit["search_val"]["marker_sha256"],
                    validation_label_sha256=data_path_audit["search_val"]["label_sha256"],
                    data_path_audit_sha256=candidate_runner.sha256_file(audit_file),
                    validation_sample_ids_sha256=hashlib.sha256(json.dumps(
                        val_data["sample_ids"], separators=(",", ":")).encode()).hexdigest(),
                    resolved_train_args_sha256=candidate_runner.sha256_file(args_file),
                    resolved_optimizer=resolved_optimizer, resolved_lr0=resolved_lr0,
                    resolved_momentum=resolved_momentum,
                    resolved_warmup_bias_lr=resolved_warmup_bias_lr,
                    initial_weights_sha256=initial_weights[0],
                    training_seconds=training_seconds,
                    completed_epochs=completed_epochs,
                    runtime_environment=candidate_runner.runtime_environment(device),
                    training_curve_sha256=candidate_runner.sha256_file(curve_file),
                    ultralytics_version=importlib.metadata.version("ultralytics"))
    result, _ = map_reporter.report(
        {"metadata": metadata, "frames": frames, "params": params,
         "expected_sample_ids": val_data["sample_ids"]},
        sota_root=run_dir, gene_id="ablation", diagnostics=True)
    print(f"{name}, seed {seed}: mAP50:95={result.map50_95:.6f}; {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--protocol", type=Path, default=candidate_runner.DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, help="restored search-export root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment", help="one name from the ablation plan")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check", action="store_true", help="validate and build every architecture on CPU")
    args = parser.parse_args()
    protocol = candidate_runner.load_protocol(args.protocol)
    plan = load_plan(args.plan, protocol)
    if args.check:
        check_architectures(plan, protocol)
        return
    if not args.experiment:
        parser.error("--experiment is required unless --check is used")
    if args.data_root:
        protocol["challenging"]["data_root"] = str(args.data_root.resolve())
    data_root = FRED_DIR / protocol["challenging"]["data_root"]
    search = protocol["challenging"]["search"]
    train_yaml = data_root / f"search_train_step{search['train_frame_step']}" / "data.yaml"
    val_root = data_root / f"search_val_step{search['val_frame_step']}"
    if not train_yaml.is_file() or not val_root.is_dir():
        parser.error(f"restore the original search export under {data_root} "
                     f"(requires {train_yaml} and {val_root})")
    run_experiment(plan, args.plan, protocol, args.protocol, args.experiment,
                   args.seed, args.output_dir, args.device)


if __name__ == "__main__":
    main()
