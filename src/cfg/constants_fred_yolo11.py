"""LLM-GE configuration for the FRED event-only drone detector seeds.

Paths are derived from this checkout so the same configuration can be used on
a laptop or on PACE ICE. SEED_FAMILY picks the seed: "yolo11" evolves the
Ultralytics YOLO11n model YAML (sota/FRED_LLM_GE/seeds/seed_yolo11n.yaml);
"tiny_yolo11" evolves Ruhi's tutorial-sized DroneDetector
(sota/FRED_LLM_GE/seeds/seed_yolo11.py); "dfine" selects the Event-only
D-FINE-inspired seed; "rtdetr" selects the original RT-DETR (v1) port
(sota/FRED_LLM_GE/seeds/seed_rtdetr.py). Every family is trained and
scored by phase1_detection/candidate_runner.py under
configs/phase1/protocol.yaml, which writes SOTA_ROOT/results/<gene_id>_results.csv.
"""

import math
import os
import platform


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRED_DIR = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE")
SEED_FAMILY = os.getenv("FRED_SEED_FAMILY", "yolo11")
_FAMILIES = {
    # SOTA_ROOT holds models/, results/ and trained_models/ for the family.
    "yolo11": {
        "sota_root": os.path.join(FRED_DIR, "phase1_detection", "seeds", "ultralytics_yolo11"),
        "seed": os.path.join(FRED_DIR, "seeds", "seed_yolo11n.yaml"),
        "prompt_group": "FRED_YOLO11/Normal",
        "rules": "templates/FRED_YOLO11/ConstantRules.txt",
        # 80 epochs of YOLO11n at 1280 can outlast the 40 min default; the
        # evaluation job itself is limited to 4 h.
        "eval_timeout": 3 * 60 * 60 + 30 * 60,
    },
    "tiny_yolo11": {
        "sota_root": os.path.join(FRED_DIR, "phase1_detection", "seeds", "yolo11"),
        "seed": os.path.join(FRED_DIR, "seeds", "seed_yolo11.py"),
        "prompt_group": "FRED/Normal",
        "rules": "templates/FRED/ConstantRules.txt",
        "eval_timeout": 40 * 60,
    },
    "dfine": {
        "sota_root": os.path.join(FRED_DIR, "phase1_detection", "seeds", "dfine"),
        "seed": os.path.join(FRED_DIR, "seeds", "seed_dfine.py"),
        "prompt_group": "FRED_DFINE/Normal",
        "rules": "templates/FRED_DFINE/ConstantRules.txt",
        "eval_timeout": 3 * 60 * 60 + 30 * 60,
    },
    "rtdetr": {
        "sota_root": os.path.join(FRED_DIR, "phase1_detection", "seeds", "rtdetr"),
        "seed": os.path.join(FRED_DIR, "seeds", "seed_rtdetr.py"),
        "prompt_group": "FRED_RTDETR/Normal",
        "rules": "templates/FRED_RTDETR/ConstantRules.txt",
        "eval_timeout": 3 * 60 * 60 + 30 * 60,
    },
}
_FAMILY = _FAMILIES[SEED_FAMILY]
SOTA_ROOT = _FAMILY["sota_root"]
DATA_PATH = os.path.join(FRED_DIR, "phase0_data")
SEED_NETWORK = _FAMILY["seed"]
MODEL = "network"
VARIANT_DIR = os.path.join(SOTA_ROOT, "models")
TRAIN_FILE = os.path.join(FRED_DIR, "phase1_detection", "candidate_runner.py")

# run_improved.py resolves the prompt glob against ROOT_DIR and loads the
# domain rules separately. Keep rules outside the selectable prompt glob.
DEFAULT_PROMPT_GROUP = _FAMILY["prompt_group"]
PROMPT_GROUP_TEMPLATE = "templates/{prompt_group}/**/*.txt"
PROMPTS = f"templates/{DEFAULT_PROMPT_GROUP}/*.txt"
CONSTANT_RULES_PATH = _FAMILY["rules"]

OUTPUT_DIR = os.path.join(ROOT_DIR, f"fred_{SEED_FAMILY}_output" if SEED_FAMILY in ("dfine", "rtdetr")
                          else "fred_yolo11_output")
SLURM_OUTPUT_PATH = os.path.join(ROOT_DIR, "run_job_outputs") + os.sep
# A run can submit its jobs with its own copy of slurm_config.yaml (e.g. to
# request faster GPUs) without changing the shared file.
SLURM_CONFIG_DIR = os.getenv("LLMGE_SLURM_CONFIG_DIR", os.path.join(ROOT_DIR, "slurm-config"))
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
# A mutation re-emits a whole class; RT-DETR's decoder class is ~2,900 tokens.
LLM_MAX_NEW_TOKENS = int(os.getenv(
    "LLM_MAX_NEW_TOKENS",
    {"dfine": "4096", "rtdetr": "6144"}.get(SEED_FAMILY, "1648")))
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
    os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(_FAMILY["eval_timeout"]))
)

# The generated candidate is named network_<gene_id>.py. The runner trains
# and scores it and writes SOTA_ROOT/results/<gene_id>_results.csv.
RUNLINE_TMP = "{}_{}"
EVAL_RUNLINE = (f"uv run python {{}} --family {SEED_FAMILY} --sota_root {SOTA_ROOT} "
                "--model {} --variant_dir {VARIANT_DIR}")

SLURM_MIXT_INPUT_X = SEED_NETWORK
SLURM_MIXT_INPUT_Y = os.path.join(VARIANT_DIR, "network_x.py")
SLURM_MIXT_OUTPUT = os.path.join(VARIANT_DIR, "network_z.py")
SLURM_MIXT_TOP_P = 0.15
SLURM_MIXT_TEMPERATURE = 0.1
SLURM_MIXT_APPLY_QUALITY_CONTROL = True
SLURM_MIXT_BIT = 8
ISLAND_CONTROLLER_RUN_NAME = (f"fred_{SEED_FAMILY}_islands" if SEED_FAMILY in ("dfine", "rtdetr")
                              else "fred_yolo11_islands")
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

# run_improved.py loops range(1, num_generations), so 21 runs generations
# 1-20 after the seed population. Resubmitting resumes from the latest
# checkpoint in fred_yolo11_checkpoints.
num_generations = 21
start_population_size = 4
population_size = 4
crossover_probability = 0.5
mutation_probability = 0.8
num_elites = 2
# Reserve one elite slot for the highest-mAP50 model; SPEA2 alone kept the
# smallest model and dropped the most accurate ones.
KEEP_BEST_ELITE = True
hof_size = 4
max_gen_attempts = 5
migration_gen = 0
DNA_TXT = "FRED YOLO11"
FORBIDDEN_PATTERNS = []
