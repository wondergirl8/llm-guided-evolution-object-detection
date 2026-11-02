"""Isolated twenty-generation Faster R-CNN scratch experiment."""
from .constants_fred_yolo11 import *
SOTA_ROOT=os.path.join(ROOT_DIR,"sota","FRED_LLM_GE","phase1_detection","seeds","faster_rcnn_scratch")
SEED_NETWORK=os.path.join(ROOT_DIR,"sota","FRED_LLM_GE","seeds","seed_faster_rcnn_scratch.py")
VARIANT_DIR=os.path.join(SOTA_ROOT,"models"); TRAIN_FILE=os.path.join(SOTA_ROOT,"train_eval.py")
DEFAULT_PROMPT_GROUP="FRED_FasterRCNNScratch/Normal"; PROMPTS=f"templates/{DEFAULT_PROMPT_GROUP}/*.txt"; CONSTANT_RULES_PATH="templates/FRED_FasterRCNNScratch/ConstantRules.txt"
OUTPUT_DIR=os.path.join(ROOT_DIR,"fred_faster_rcnn_scratch_output"); GLOBAL_DATA_PATH=os.path.join(ROOT_DIR,"fred_faster_rcnn_scratch_global_data"); HOSTNAME_DIR=os.path.join(ROOT_DIR,"fred_faster_rcnn_scratch_hostname.log"); SLURM_CONFIG_DIR=os.path.join(ROOT_DIR,"slurm-config","faster_rcnn_scratch")
PORT=int(os.getenv("LLMGE_PORT","8295")); EVAL_RUNLINE=os.path.join(ROOT_DIR,".venv-faster-rcnn","bin","python")+" {} --model {} --variant_dir {VARIANT_DIR}"
SLURM_MIXT_INPUT_X=SEED_NETWORK; SLURM_MIXT_INPUT_Y=os.path.join(VARIANT_DIR,"network_x.py"); SLURM_MIXT_OUTPUT=os.path.join(VARIANT_DIR,"network_z.py")
ISLAND_CONTROLLER_RUN_NAME="fred_faster_rcnn_scratch_20gen"; ISLAND_CONTROLLER_NUM_ISLANDS=1; ISLAND_CONTROLLER_PROMPT_GROUPS=DEFAULT_PROMPT_GROUP
num_generations=21; start_population_size=4; population_size=4; num_elites=2; crossover_probability=.5; mutation_probability=.8
DNA_TXT="FRED MMDetection Faster R-CNN R50-FPN scratch prototype-v1"
