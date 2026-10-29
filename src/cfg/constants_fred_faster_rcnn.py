"""Bounded Event-only Faster R-CNN development pilot."""
from .constants_fred_yolo11 import *

SOTA_ROOT = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "phase1_detection", "seeds", "faster_rcnn")
SEED_NETWORK = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "seeds", "seed_faster_rcnn.py")
VARIANT_DIR = os.path.join(SOTA_ROOT, "models")
TRAIN_FILE = os.path.join(SOTA_ROOT, "train_eval.py")
DEFAULT_PROMPT_GROUP = "FRED_FasterRCNN/Normal"
PROMPTS = f"templates/{DEFAULT_PROMPT_GROUP}/*.txt"
CONSTANT_RULES_PATH = "templates/FRED_FasterRCNN/ConstantRules.txt"
OUTPUT_DIR = os.path.join(ROOT_DIR, "fred_faster_rcnn_output")
GLOBAL_DATA_PATH = os.path.join(ROOT_DIR, "fred_faster_rcnn_global_data")
HOSTNAME_DIR = os.path.join(ROOT_DIR, "fred_faster_rcnn_hostname.log")
SLURM_CONFIG_DIR = os.path.join(ROOT_DIR, "slurm-config", "faster_rcnn")
PORT = int(os.getenv("LLMGE_PORT", "8294"))
EVAL_RUNLINE = os.path.join(ROOT_DIR, ".venv-faster-rcnn", "bin", "python") + " {} --model {} --variant_dir {VARIANT_DIR}"
SLURM_MIXT_INPUT_X = SEED_NETWORK
SLURM_MIXT_INPUT_Y = os.path.join(VARIANT_DIR, "network_x.py")
SLURM_MIXT_OUTPUT = os.path.join(VARIANT_DIR, "network_z.py")
ISLAND_CONTROLLER_RUN_NAME = "fred_faster_rcnn_pilot"
ISLAND_CONTROLLER_NUM_ISLANDS = 1
ISLAND_CONTROLLER_PROMPT_GROUPS = DEFAULT_PROMPT_GROUP
num_generations = 3  # range(1, 3): two breeding generations.
DNA_TXT = "FRED MMDetection Faster R-CNN R50-FPN prototype-v1"
