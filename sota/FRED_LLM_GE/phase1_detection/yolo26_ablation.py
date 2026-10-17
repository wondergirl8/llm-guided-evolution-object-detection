"""Run one controlled YOLO26 pilot ablation on the search-validation export.

Experiments are separate from LLM-GE and the historical YOLO26 result root.
Use --check to validate and build every architecture without FRED data or a GPU.
"""

import argparse
import hashlib
import importlib.metadata
import tempfile
from pathlib import Path

import torch
import yaml

try:
    from . import candidate_runner, map_reporter
    from .adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter
except ImportError:
    import candidate_runner
    import map_reporter
    from adapters.ultralytics_yolo26_p2 import UltralyticsYolo26P2Adapter


FRED_DIR = candidate_runner.FRED_DIR
DEFAULT_PLAN = FRED_DIR / "configs/phase1/yolo26_ablations.yaml"
DEFAULT_OUTPUT = FRED_DIR / "phase1_detection/seeds/yolo26_p2/ablations"
TRAIN_OPTIONS = {"optimizer", "lr0", "mosaic", "hsv_s"}


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
            if key in ("mosaic", "hsv_s") and (
                    type(value) not in (float, int) or not 0 <= value <= 1):
                raise ValueError(f"{name}: {key} must be in [0, 1]")
        reference = case.get("compare_to")
        if reference is None:
            continue  # seed and compact are separate reference architectures
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
    code_files = ("yolo26_ablation.py", "candidate_runner.py", "fitness.py",
                  "map_reporter.py", "adapters/ultralytics_yolo11.py",
                  "adapters/ultralytics_yolo26_p2.py")
    code_sha = hashlib.sha256(b"".join(
        (Path(__file__).parent / file).read_bytes() for file in code_files)).hexdigest()
    plan_sha = candidate_runner.sha256_file(plan_path)
    protocol_sha = candidate_runner.sha256_file(protocol_path)
    source_sha = candidate_runner.sha256_file(source)
    proxy, train_data, load_val = candidate_runner.challenging_search(protocol, "yolo26_p2")
    export_sha = candidate_runner.sha256_file(train_data["data_yaml"])
    val_data = load_val()
    proxy.update(seed=seed, epochs=case.get("epochs", proxy["epochs"]))
    settings = dict(protocol["families"]["yolo26_p2"],
                    training_overrides=case.get("train", {}))
    adapter = UltralyticsYolo26P2Adapter(settings, FRED_DIR)
    torch.manual_seed(seed)

    run_dir.mkdir(parents=True)
    variant = run_dir / "model.yaml"
    variant.write_text(yaml.safe_dump(config, sort_keys=False))
    model = adapter.build(variant, num_classes=1, device=device)
    adapter.workdir = run_dir / "ultralytics"
    params = adapter.param_count(model)
    adapter.train(model, train_data, proxy, device)

    trainer_dir = adapter.workdir / "train"
    args_file, curve_file = trainer_dir / "args.yaml", trainer_dir / "results.csv"
    if not args_file.is_file() or not curve_file.is_file():
        raise RuntimeError("Ultralytics did not save resolved arguments and training curves")
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
                    resolved_train_args_sha256=candidate_runner.sha256_file(args_file),
                    resolved_optimizer=type(model.trainer.optimizer).__name__,
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
    run_experiment(plan, args.plan, protocol, args.protocol, args.experiment,
                   args.seed, args.output_dir, args.device)


if __name__ == "__main__":
    main()
