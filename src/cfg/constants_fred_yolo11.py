"""LLM-GE configuration for the FRED event-only drone detector seeds.

Paths are derived from this checkout so the same configuration can be used on
a laptop or on PACE ICE. SEED_FAMILY picks the seed: "yolo11" evolves the
Ultralytics YOLO11n model YAML (sota/FRED_LLM_GE/seeds/seed_yolo11n.yaml);
"tiny_yolo11" evolves Ruhi's tutorial-sized DroneDetector
(sota/FRED_LLM_GE/seeds/seed_yolo11.py); "dfine" selects the Event-only
D-FINE-inspired seed; "renderer" evolves how raw events are drawn into frames
(sota/FRED_LLM_GE/seeds/seed_renderer.py), scored by
phase1_detection/render_runner.py with a fixed YOLO11n. Every family is trained and
scored by phase1_detection/candidate_runner.py under
configs/phase1/protocol.yaml, which writes SOTA_ROOT/results/<gene_id>_results.csv.
"""

import math
import os
import platform
import re


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRED_DIR = os.path.join(ROOT_DIR, "sota", "FRED_LLM_GE")
SEED_FAMILY = os.getenv("FRED_SEED_FAMILY", "yolo11")
RUN_TAG = os.getenv("FRED_RUN_TAG", "")
if RUN_TAG and not re.fullmatch(
    rf"{re.escape(SEED_FAMILY)}_[0-9]{{8}}_eot[01]_[a-z0-9][a-z0-9_-]*", RUN_TAG
):
    raise ValueError(
        "FRED_RUN_TAG must be <family>_YYYYMMDD_eot0-or-eot1_<experiment>, "
        "for example yolo11_20261017_eot0_trial1"
    )
RUN_ROOT = os.path.join(ROOT_DIR, "fred_runs", RUN_TAG) if RUN_TAG else None
RUN_CHECKPOINT_DIR = os.path.join(RUN_ROOT, "checkpoints") if RUN_ROOT else None
_FAMILIES = {
    # SOTA_ROOT holds models/, results/ and trained_models/ for the family.
    "yolo11": {
        "sota_root": os.path.join(FRED_DIR, "phase1_detection", "seeds", "ultralytics_yolo11"),
        "seed": os.path.join(FRED_DIR, "seeds", "seed_yolo11n.yaml"),
        "prompt_group": "FRED_YOLO11/Normal",
        "rules": "templates/FRED_YOLO11/ConstantRules.txt",
        "eval_timeout": 8 * 60 * 60,
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
    "renderer": {
        "sota_root": os.path.join(FRED_DIR, "phase1_detection", "seeds", "renderer"),
        "seed": os.path.join(FRED_DIR, "seeds", "seed_renderer.py"),
        "prompt_group": "FRED_RENDER/Normal",
        "rules": "templates/FRED_RENDER/ConstantRules.txt",
        "eval_timeout": 3 * 60 * 60 + 30 * 60,
        "train_file": os.path.join(FRED_DIR, "phase1_detection", "render_runner.py"),
    },
}
# Renderer family only: every renderer draws frames from exactly this many
# microseconds of events (e.g. 2000), so LLM-GE evolves only the encoding.
# Unset lets each renderer choose its own WINDOW_US.
RENDER_WINDOW_US = int(os.getenv("FRED_RENDER_WINDOW_US") or 0) if SEED_FAMILY == "renderer" else 0
if RENDER_WINDOW_US:
    _FAMILIES["renderer"].update(prompt_group="FRED_RENDER_FIXED/Normal",
                                 rules="templates/FRED_RENDER_FIXED/ConstantRules.txt")
_FAMILY = _FAMILIES[SEED_FAMILY]
_RUN_NAME = {"dfine": "fred_dfine", "renderer": "fred_renderer"}.get(SEED_FAMILY, "fred_yolo11")
CANDIDATE_POLICY_MODULE = (
    "sota.FRED_LLM_GE.phase1_detection.yolo11_search_space"
    if SEED_FAMILY == "yolo11" else None
)
CROSSOVER_TEMPLATE_DIR = (
    "templates/FRED_YOLO11/CrossOver" if SEED_FAMILY == "yolo11"
    else "templates/CrossOver"
)
# Event frames candidates train on: unset for FRED's released frames, or a tag
# such as raw_w10000_polarity (phase1_detection/event_sweep.py). A tagged run
# keeps its genes, results, outputs and checkpoints apart from the released run.
FRED_EVENTS = os.getenv("FRED_EVENTS", "")
_RUN_SUFFIX = (f"_{FRED_EVENTS}" if FRED_EVENTS else
               f"_w{RENDER_WINDOW_US}" if RENDER_WINDOW_US else "")
SOTA_ROOT = (os.path.join(RUN_ROOT, "candidates") if RUN_ROOT else
             _FAMILY["sota_root"] + _RUN_SUFFIX)
DATA_PATH = os.path.join(FRED_DIR, "phase0_data")
SEED_NETWORK = _FAMILY["seed"]
if RENDER_WINDOW_US:
    # The LLM starts from the seed renderer with the fixed window written in.
    SEED_NETWORK = os.path.join(SOTA_ROOT, f"seed_renderer_w{RENDER_WINDOW_US}.py")
    if not os.path.isfile(SEED_NETWORK):
        with open(_FAMILY["seed"]) as _file:
            _seed, _count = re.subn(r"^WINDOW_US = \d+$", f"WINDOW_US = {RENDER_WINDOW_US}",
                                    _file.read(), flags=re.M)
        if _count != 1:
            raise ValueError(f"{_FAMILY['seed']}: expected one WINDOW_US line")
        _seed = _seed.replace("# FRED's released frames: 33.3 ms of events,",
                              f"# FRED's released palette, from {RENDER_WINDOW_US / 1000:g} ms of events:")
        os.makedirs(SOTA_ROOT, exist_ok=True)
        with open(SEED_NETWORK, "w") as _file:
            _file.write(_seed)
MODEL = "network"
VARIANT_DIR = os.path.join(SOTA_ROOT, "models")
TRAIN_FILE = _FAMILY.get("train_file", os.path.join(FRED_DIR, "phase1_detection", "candidate_runner.py"))

# run_improved.py resolves the prompt glob against ROOT_DIR and loads the
# domain rules separately. Keep rules outside the selectable prompt glob.
DEFAULT_PROMPT_GROUP = _FAMILY["prompt_group"]
PROMPT_GROUP_TEMPLATE = "templates/{prompt_group}/**/*.txt"
PROMPTS = f"templates/{DEFAULT_PROMPT_GROUP}/*.txt"
CONSTANT_RULES_PATH = _FAMILY["rules"]

OUTPUT_DIR = (os.path.join(RUN_ROOT, "operator_jobs") if RUN_ROOT else
              os.path.join(ROOT_DIR, f"fred_renderer_output{_RUN_SUFFIX}") if SEED_FAMILY == "renderer" else
              os.path.join(ROOT_DIR, "fred_dfine_output" if SEED_FAMILY == "dfine"
                           else "fred_yolo11_output"))
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
LLM_MAX_NEW_TOKENS = int(os.getenv(
    "LLM_MAX_NEW_TOKENS", "4096" if SEED_FAMILY == "dfine" else "1648"))
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
# This clock starts at sbatch submission, including queue time. The evaluation
# job's own four-hour Slurm limit still caps its runtime after it starts.
EVAL_NO_PROGRESS_TIMEOUT_SECONDS = int(
    os.getenv("LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS", str(_FAMILY["eval_timeout"]))
)

# The generated candidate is named network_<gene_id>.py. The runner trains
# and scores it and writes SOTA_ROOT/results/<gene_id>_results.csv.
RUNLINE_TMP = "{}_{}"
# A renderer gene draws its own frames, so it takes no family or --events.
EVAL_RUNLINE = ((f"uv run python {{}} --sota_root {SOTA_ROOT} --model {{}} --variant_dir {{VARIANT_DIR}}"
                 + (f" --window_us {RENDER_WINDOW_US}" if RENDER_WINDOW_US else ""))
                if SEED_FAMILY == "renderer" else
                (f"uv run python {{}} --family {SEED_FAMILY} --sota_root {SOTA_ROOT} "
                 "--model {} --variant_dir {VARIANT_DIR}"
                 + (f" --events {FRED_EVENTS}" if FRED_EVENTS else "")))

SLURM_MIXT_INPUT_X = SEED_NETWORK
SLURM_MIXT_INPUT_Y = os.path.join(VARIANT_DIR, "network_x.py")
SLURM_MIXT_OUTPUT = os.path.join(VARIANT_DIR, "network_z.py")
SLURM_MIXT_TOP_P = 0.15
SLURM_MIXT_TEMPERATURE = 0.1
SLURM_MIXT_APPLY_QUALITY_CONTROL = True
SLURM_MIXT_BIT = 8
ISLAND_CONTROLLER_RUN_NAME = f"{_RUN_NAME}_islands{_RUN_SUFFIX}"
ISLAND_CONTROLLER_NUM_ISLANDS = 1
ISLAND_CONTROLLER_LLMS = LLM_MODEL
ISLAND_CONTROLLER_PROMPT_GROUPS = DEFAULT_PROMPT_GROUP

QC_CHECK_BOOL = False
HUGGING_FACE_BOOL = False
PROB_QC = 0.0
PROB_EOT = 0.0
if RUN_TAG and f"_eot{int(PROB_EOT > 0)}_" not in RUN_TAG:
    raise ValueError("FRED_RUN_TAG EoT flag does not match PROB_EOT")
NUM_EOT_ELITES = 1
GENERATION = 0

# CSV order expected by run_improved.py: mAP50, mAP50:95, parameter count
# (for the renderer family: WINDOW_US, the microseconds of events per frame).
FITNESS_WEIGHTS = (1.0, 1.0, -1.0)
INVALID_FITNESS_MAX = tuple(math.copysign(math.inf, -w) for w in FITNESS_WEIGHTS)
PLACEHOLDER_FITNESS = tuple(int(-w * 9_999_999_999) for w in FITNESS_WEIGHTS)

# run_improved.py checkpoints generation 0 (seed and initial candidates), then
# runs generations 1 through num_generations - 1. Resubmitting resumes from the latest
# checkpoint in the configured run directory.
num_generations = int(os.getenv('FRED_NUM_GENERATIONS', '21'))
if num_generations < 1:
    raise ValueError('FRED_NUM_GENERATIONS must include generation 0')
EVAL_GPU_CONSTRAINT = os.getenv('FRED_EVAL_GPU_CONSTRAINT', 'H200')
# Respect SLURM's per-job CUDA_VISIBLE_DEVICES instead of the shared template's
# hard-coded 0. PACE may remap an allocated physical GPU to logical device 0.
PRESERVE_SLURM_CUDA_VISIBLE_DEVICES = True
# Retry a GPU allocation that CUDA cannot use; this is an infrastructure
# failure, not evidence that the candidate architecture is invalid.
EVAL_GPU_RETRY_LIMIT = 2
FRED_MAX_REFILL_ATTEMPTS = 24
start_population_size = 8
# Six selected parents enter the operators; two unmodified elites bring the
# next generation to at most eight distinct candidates.
population_size = 8
offspring_size = 6
crossover_probability = 0.5
mutation_probability = 0.8
num_elites = 2
# SPEA2 selects both elites using all three objectives; no individual gets a
# reserved slot based on mAP50, mAP50:95, or parameter count alone.
KEEP_BEST_ELITE = False
if offspring_size + num_elites != population_size:
    raise ValueError("FRED offspring_size + num_elites must equal population_size")
hof_size = 4
max_gen_attempts = 5
migration_gen = 0
DNA_TXT = "FRED YOLO11"
# Renderers may only use numpy, already imported in the protected preamble.
FORBIDDEN_PATTERNS = ([("import ", "renderer may not import modules"),
                       ("open(", "renderer may not do file I/O"),
                       ("exec(", "renderer may not run code"), ("eval(", "renderer may not run code")]
                      if SEED_FAMILY == "renderer" else [])
