"""Option-2 LLM-GE configuration: actual Ultralytics YOLO11 from scratch.

Paths are derived from this checkout so the same configuration can be used on
a laptop or on PACE ICE. The official YOLO11 backbone/neck are editable Python
classes; the head/loss and training protocol are protected. No COCO weights.
"""

import math
import os
import platform


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOTA_ROOT = os.path.join(
    ROOT_DIR, "sota", "FRED_LLM_GE", "phase1_detection", "seeds", "yolo11"
)
DATA_PATH = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "phase0_data")
SEED_NETWORK = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE", "seeds", "seed_ultralytics_yolo11.py")
MODEL = "network"
VARIANT_DIR = os.path.join(SOTA_ROOT, "models")
TRAIN_FILE = os.path.join(SOTA_ROOT, "train_ultralytics.py")

# run_improved.py resolves the prompt glob against ROOT_DIR and loads the
# domain rules separately. Keep rules outside the selectable prompt glob.
DEFAULT_PROMPT_GROUP = "FRED/Ultralytics"
PROMPT_GROUP_TEMPLATE = "templates/{prompt_group}/**/*.txt"
PROMPTS = f"templates/{DEFAULT_PROMPT_GROUP}/*.txt"
CONSTANT_RULES_PATH = "templates/FRED/UltralyticsRules.txt"

OUTPUT_DIR = os.getenv("LLMGE_OUTPUT_DIR", os.path.join(ROOT_DIR, "fred_ultralytics_yolo11_output"))
SLURM_OUTPUT_PATH = os.path.join(ROOT_DIR, "run_job_outputs") + os.sep
SLURM_CONFIG_DIR = os.path.join(ROOT_DIR, "slurm-config")
ISLAND_TEMP_SCRIPT = os.path.join("src", "island_temp_script_{ISLAND_NUM}.sh")
GLOBAL_DATA_PATH = os.path.join(ROOT_DIR, "global_data")
HOSTNAME_DIR = os.path.join(ROOT_DIR, "hostname.log")
ENVIRONMENT_DIR = os.path.join(ROOT_DIR, ".venv")

CLUSTER = os.getenv("LLMGE_CLUSTER", "pace-ice")
# Not 8137: other LLM-GE servers on shared GPU nodes use that default.
PORT = int(os.getenv("LLMGE_PORT", "8293"))
# Same shared Llama 3.3 70B the MuJoCo runs serve through server.sh.
MODEL_PATH = os.getenv(
    "LLMGE_MODEL_PATH",
    "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/",
)
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1648"))
LLM_MODEL = os.getenv("LLMGE_LLM_MODEL", "mixtral")
LLM_QWEN = "qwen25"
LLM_MIXTRAL = "mixtral"
LLM_LLAMA3 = "llama3"
LLM_GEMMA2 = "gemma2"
LLM_GEMMA3 = "gemma3"
LLM_DEEPSEEK = "deepseek"
LLM_GEMINI = "gemini"
# run_improved.py defaults to LLM_MIXTRAL when that identifier is present.
ISLAND_LLMS = list(dict.fromkeys((LLM_MIXTRAL, LLM_MODEL)))
MAX_ISLANDS = len(ISLAND_LLMS)
# Generate code through the uvicorn LLM server started by server.sh.
LOCAL_LLM = os.getenv("LOCAL_LLM", "true").lower() in ("true", "1", "yes")
INFERENCE_SUBMISSION = os.getenv("INFERENCE_SUBMISSION", "false").lower() in (
    "true", "1", "yes"
)

LOCAL = os.getenv("LOCAL", "false").lower() in ("true", "1", "yes")
RUN_COMMAND = "bash" if LOCAL else "sbatch"
DELAYED_CHECK = not LOCAL
MACOS = platform.system() == "Darwin"
RUNLINE_AMP = ""
EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(
    os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(40 * 60))
)

# The generated candidate is named network_<gene_id>.py. The trainer must
# accept --model and --variant_dir and write results/<gene_id>_results.csv.
RUNLINE_TMP = "{}_{}"
EVAL_RUNLINE = "uv run python {} --model {} --variant_dir {VARIANT_DIR}"

SLURM_MIXT_INPUT_X = SEED_NETWORK
SLURM_MIXT_INPUT_Y = os.path.join(VARIANT_DIR, "network_x.py")
SLURM_MIXT_OUTPUT = os.path.join(VARIANT_DIR, "network_z.py")
SLURM_MIXT_TOP_P = 0.15
SLURM_MIXT_TEMPERATURE = 0.1
SLURM_MIXT_APPLY_QUALITY_CONTROL = True
SLURM_MIXT_BIT = 8
ISLAND_CONTROLLER_RUN_NAME = "fred_yolo11_islands"
ISLAND_CONTROLLER_NUM_ISLANDS = 1
ISLAND_CONTROLLER_LLMS = LLM_MODEL
ISLAND_CONTROLLER_PROMPT_GROUPS = DEFAULT_PROMPT_GROUP

QC_CHECK_BOOL = False
HUGGING_FACE_BOOL = False
PROB_QC = 0.0
PROB_EOT = 0.0
NUM_EOT_ELITES = 1
GENERATION = 0

# CSV order expected by run_improved.py: mAP50, mAP50:95, parameter count.
FITNESS_WEIGHTS = (1.0, 1.0, -1.0)
INVALID_FITNESS_MAX = tuple(math.copysign(math.inf, -w) for w in FITNESS_WEIGHTS)
PLACEHOLDER_FITNESS = tuple(int(-w * 9_999_999_999) for w in FITNESS_WEIGHTS)

# range(1, num_generations): default 3 runs two generations after initialization.
# Use fresh checkpoints for this scratch experiment; never resume the small seed.
num_generations = int(os.getenv("LLMGE_GENERATIONS", "3"))
start_population_size = int(os.getenv("LLMGE_POPULATION", "4"))
population_size = start_population_size
if num_generations < 2 or population_size < 4:
    raise ValueError("Scratch evolution needs LLMGE_GENERATIONS >= 2 and LLMGE_POPULATION >= 4")
crossover_probability = 0.5
mutation_probability = 0.8
num_elites = 2
# Reserve one elite slot for the highest-mAP50 model; SPEA2 alone kept the
# smallest model and dropped the most accurate ones.
KEEP_BEST_ELITE = True
hof_size = 4
max_gen_attempts = 5
migration_gen = 0
DNA_TXT = "FRED actual Ultralytics YOLO11 / random weights / architecture evolution"
FORBIDDEN_PATTERNS = []
