"""Train and validate one generated YOLO11 model on a bounded FRED export."""

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import yaml

from validator import load_model_module


SEED_DIR = Path(__file__).resolve().parent
RESULT_HEADER = ("map50", "map50_95", "param_count")


def _write_results(path: Path, map50: float, map50_95: float, param_count: int):
    values = (float(map50), float(map50_95), int(param_count))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite YOLO11 result")
    if not (0 <= values[0] <= 1 and 0 <= values[1] <= 1 and values[2] > 0):
        raise ValueError("invalid YOLO11 result range")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                     dir=path.parent, delete=False) as temp:
        writer = csv.writer(temp)
        writer.writerow(RESULT_HEADER)
        writer.writerow(values)
        temp_path = Path(temp.name)
    temp_path.replace(path)


def _validate_data(data: Path):
    if not data.is_file():
        raise FileNotFoundError(f"YOLO data YAML is missing: {data}")
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if config.get("names") != {0: "drone"}:
        raise ValueError("FRED YOLO data must declare one drone class")
    data_root = Path(config["path"])
    if not (data_root / "source.json").is_file():
        raise ValueError("FRED export provenance is missing")
    for split in ("train", "val"):
        image_dir = data_root / config[split]
        if not image_dir.is_dir() or not any(image_dir.glob("*.png")):
            raise ValueError(f"no exported FRED images for {split}: {image_dir}")


def run(model_name: str, variant_dir: Path, data: Path, weights: str,
        epochs: int, batch: int, imgsz: int, device: str,
        preflight_only: bool = False) -> Path | None:
    if epochs <= 0 or batch <= 0 or imgsz <= 0:
        raise ValueError("epochs, batch and imgsz must be positive")
    _validate_data(data)
    module = load_model_module(model_name, variant_dir)

    from ultralytics import YOLO, __version__ as ultralytics_version

    base_model = YOLO(weights)
    base_config = base_model.model.yaml
    candidate_config = module.build_config(base_config)
    # Recheck model structure against the chosen checkpoint before GPU use.
    if candidate_config["nc"] != 1 or candidate_config["head"][-1][2] != "Detect":
        raise ValueError("candidate changed protected detection semantics")
    gene_id = "seed" if model_name == "network" else model_name.removeprefix("network_")
    run_root = SEED_DIR / "runs"
    run_root.mkdir(exist_ok=True)
    results_path = SEED_DIR / "results" / f"{gene_id}_results.csv"
    if not preflight_only and results_path.exists():
        raise FileExistsError(f"refusing to replace existing gene results: {results_path}")
    if not preflight_only:
        results_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{gene_id}-config-", dir=run_root) as temp_name:
        config_path = Path(temp_name) / "candidate.yaml"
        config_path.write_text(yaml.safe_dump(candidate_config, sort_keys=False), encoding="utf-8")
        model = YOLO(str(config_path)).load(weights)
        if preflight_only:
            print(f"preflight passed: {model_name}; ultralytics {ultralytics_version}")
            return None
        # A candidate's metrics always come from a newly trained checkpoint.
        model.train(data=str(data), epochs=epochs, batch=batch, imgsz=imgsz,
                    device=device, seed=0, deterministic=True, project=str(run_root),
                    name=gene_id, exist_ok=False, save=True)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError(f"training produced no best checkpoint: {best}")
        trained = YOLO(str(best))
        metrics = trained.val(data=str(data), split="val", imgsz=imgsz,
                              batch=batch, device=device)
        param_count = sum(parameter.numel() for parameter in trained.model.parameters())
        provenance = {
            "gene_id": gene_id,
            "model_source_sha256": hashlib.sha256(
                (SEED_DIR / "network.py" if model_name == "network" else
                 variant_dir / f"{model_name}.py").read_bytes()).hexdigest(),
            "data_yaml": str(data.resolve()),
            "data_source": json.loads((data.parent / "source.json").read_text(encoding="utf-8")),
            "weights": weights,
            "ultralytics_version": ultralytics_version,
            "epochs": epochs, "batch": batch, "imgsz": imgsz,
            "seed": 0, "checkpoint": str(best),
        }
        (results_path.with_suffix(".json")).write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        _write_results(results_path, metrics.box.map50, metrics.box.map, param_count)
    print("job done", flush=True)
    return results_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant_dir", type=Path, default=SEED_DIR / "models")
    parser.add_argument("--data", type=Path,
                        default=Path(os.getenv("FRED_YOLO_DATA", "data/fred_yolo11_smoke/data.yaml")))
    parser.add_argument("--weights", default=os.getenv("FRED_YOLO_WEIGHTS", "yolo11m.pt"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("FRED_YOLO_EPOCHS", "1")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("FRED_YOLO_BATCH", "2")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("FRED_YOLO_IMGSZ", "640")))
    parser.add_argument("--device", default=os.getenv("FRED_YOLO_DEVICE", "0"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    run(args.model, args.variant_dir, args.data, args.weights,
        args.epochs, args.batch, args.imgsz, args.device, args.preflight_only)


if __name__ == "__main__":
    main()
