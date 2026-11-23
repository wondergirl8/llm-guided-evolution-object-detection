"""Official challenging-split Faster R-CNN scratch experiment."""
import os
import runpy
from pathlib import Path

# The shared FRED configuration owns run paths and controller safeguards.
_shared = runpy.run_path(str(Path(__file__).with_name("constants_fred_yolo11.py")),
                         init_globals={"_FAMILY_OVERRIDE": "faster_rcnn_scratch"})
globals().update({key: value for key, value in _shared.items() if not key.startswith("_")})

PORT = int(os.getenv("LLMGE_PORT", "8295"))
HOSTNAME_DIR = os.path.join(RUN_ROOT or ROOT_DIR, "hostname.log")
GLOBAL_DATA_PATH = os.path.join(RUN_ROOT or ROOT_DIR, "global_data")
SLURM_CONFIG_DIR = os.path.join(ROOT_DIR, "slurm-config", "faster_rcnn_scratch")
EVAL_RUNLINE = (os.path.join(ROOT_DIR, ".venv-faster-rcnn/bin/python")
                + " {} --family faster_rcnn_scratch --sota_root " + SOTA_ROOT
                + " --model {} --variant_dir {VARIANT_DIR}")
ISLAND_CONTROLLER_RUN_NAME = "fred_faster_rcnn_scratch_20gen"
start_population_size = population_size = 4
offspring_size = num_elites = 2
SLURM_MIXT_APPLY_QUALITY_CONTROL = False
DNA_TXT = "FRED MMDetection Faster R-CNN R50-FPN scratch prototype-v1"
