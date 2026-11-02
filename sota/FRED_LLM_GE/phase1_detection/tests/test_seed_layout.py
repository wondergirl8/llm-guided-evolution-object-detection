"""Keep standalone runners and evolution pointed at the same model-local seeds."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
FRED_DIR = ROOT / "sota/FRED_LLM_GE"


@pytest.mark.parametrize("detector,family,folder,filename", [
    ("yolo11", "yolo11", "ultralytics_yolo11", "network.yaml"),
    ("yolo11", "tiny_yolo11", "yolo11", "network.py"),
    ("yolo11", "dfine", "dfine", "network.py"),
    ("faster_rcnn", "yolo11", "faster_rcnn", "seed.py"),
    ("faster_rcnn_scratch", "yolo11", "faster_rcnn_scratch", "seed.py"),
])
def test_configured_seed_is_owned_by_model(detector, family, folder, filename):
    env = dict(os.environ, LLMGE_DETECTOR=detector, FRED_SEED_FAMILY=family,
               FRED_RUN_TAG="", FRED_EVENTS="")
    completed = subprocess.run(
        [sys.executable, "-c",
         "import json; from src.cfg import constants as c; "
         "print(json.dumps([c.SEED_NETWORK, c.SOTA_ROOT, c.TRAIN_FILE]))"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    seed, output, runner = map(Path, json.loads(completed.stdout))
    model_dir = FRED_DIR / "phase1_detection/seeds" / folder
    assert seed == model_dir / filename
    assert seed.is_file() and seed.stat().st_size > 0
    assert output == model_dir
    assert runner.is_file() and runner.stat().st_size > 0

    if detector == "yolo11":
        protocol = yaml.safe_load((FRED_DIR / "configs/phase1/protocol.yaml").read_text())
        assert FRED_DIR / protocol["families"][family]["seed_file"] == seed
    else:
        # Validators must protect exactly the source used by the orchestrator.
        import importlib
        validator = importlib.import_module(
            f"sota.FRED_LLM_GE.phase1_detection.seeds.{folder}.validator"
        )
        assert validator.SEED == seed
        validator.validate_source(seed)
