# Option 2: actual YOLO11 architecture evolution from scratch

This branch starts from `MosesTheRedSea-main` at `0c4e892b0`. The user requested
the team's option 2: actual Ultralytics YOLO11 as generation-zero architecture,
LLM-written backbone/neck mutations and crossovers, every individual initialized
randomly. No COCO checkpoint or parent checkpoint is loaded. This is a bounded
engineering pilot, not a finalized research protocol or FRED paper reproduction.
It does not change the older primary-study pretraining recommendations.

## What changed

- `seeds/seed_ultralytics_yolo11.py` constructs pinned Ultralytics 8.3.203 YOLO11.
  Default scale is **m**, matching the project's recommended starting scale;
  **n** is available for cheap plumbing checks. The unchanged seed executes the
  official graph exactly, including its Detect head and P3/P4/P5 outputs.
- Two mutable Python classes contain the actual backbone and neck. They may
  add/replace PyTorch blocks, not merely choose among preset parameter values.
  The protected wrapper, Detect head, detection loss and NMS remain fixed.
- `seeds/yolo11/train_ultralytics.py` uses official `v8DetectionLoss` with the
  team's AdamW loop, cached FRED event frames, annotation handling and COCO
  evaluator. It pads cached tensors on the bottom/right to multiples of 32;
  labels and predictions retain their original coordinate meaning. It does
  not change raw data or create another FRED loader.
- Configuration selects the new seed, trainer and `FRED/Ultralytics` prompts.
  The existing LLM-GE mutation/crossover/selection/checkpoint code is reused.
  New run output directories prevent resuming the small-seed population.
- Invalid shapes, excessive parameters (>2x seed), changed protected code,
  unchanged seed candidates, non-finite loss/gradients/predictions fail. A
  failed rerun removes old fitness and records its traceback rather than
  reporting a successful zero-accuracy candidate.

Ruhi's `seed_yolo11.py`, `train_eval.py` and original prompts remain available.
There is no new universal runner, new model family, or broad repository rewrite.

## Controlled engineering workload

The pilot uses YOLO11m, train sequence **5**, validation sequence **1**, every
200th event frame, width 320, stride-32 padding, one epoch, batch size 2,
AdamW learning rate 0.001 / weight decay 0.0005, random seed 0, and official
loss weights. Seed and candidates use identical settings. NMS uses confidence
0.05, IoU 0.5, maximum 100 detections and no time-based batch truncation.

The controller has population 4 and runs two generations after initialization
(`LLMGE_GENERATIONS=3` because the existing loop starts at 1). Every candidate
starts fresh; inheritance applies to architecture, not trained weights.

Outside the launcher, the trainer defaults to the team's existing full proxy
cohort, width 640, batch 4, three epochs. CLI subsets must stay within the team's
existing training/validation memberships. The donor streaming loader and its
annotation clipping policy remain provisional engineering inputs; this change
does not establish that they constitute a frozen formal benchmark. The tiny
pilot cannot establish accuracy improvement or superiority to the FRED paper.

## Run on ICE

Use a **fresh checkout of this branch** so old small-seed results/checkpoints do
not mix with this attempt. Run from the repository root, using the team's normal
ICE modules and environment. The code is not automatically pushed by Codex.

```bash
module load uv
uv sync

# Reuse the team's reachable LLM server at the configured port (default 8293).
# Set this to its actual node hostname, not the login-node hostname.
read -r -p "Active team LLM server hostname: " yolo11_server_host
printf '%s\n' "$yolo11_server_host" > hostname.log

bash scripts/run_yolo11_scratch_pilot.sh
```

The launcher checks the installed version, configuration and live server before
submission. It prints a **baseline job ID**, **evolution job ID**, artifact
directory, `squeue` command and `sacct` command. Evolution is dependent on the
seed baseline succeeding. Controller jobs submit candidates through the team's
existing Slurm configuration. They do not launch an additional LLM server.

The baseline's checkpoint, report and CSV are copied into the isolated pilot
directory. Inspect the generation-two checkpoint, global lineage files,
candidate reports and failure records. A generation checkpoint alone is not
proof that mutations improved accuracy or that crossover occurred; inspect the
ancestry and generated source files. Existing Pareto plotting scripts can use
the same three-objective CSV/checkpoint format.

## Local checks

Use the branch's Python environment with pinned Ultralytics installed:

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  sota/FRED_LLM_GE/phase1_detection/tests \
  sota/FRED_LLM_GE/phase0_data/test_fred_stream.py -q

OMP_NUM_THREADS=2 .venv/bin/python \
  sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/train_ultralytics.py \
  --preflight-only --scale m
```

The integration test uses **synthetic frames and deterministic mocked LLM
responses**. It exercises real mutation/crossover file assembly, candidate
imports, gradients, training, NMS, COCO scoring, CSV ingestion, selection and
checkpoint reload into the next generation. It is not live LLM or FRED evidence.
Seed predictions are also compared exactly against the official n/m graphs,
and the original small detector still trains/predicts.

Official source inspected for this implementation:
https://github.com/ultralytics/ultralytics/tree/v8.3.203
and https://docs.ultralytics.com/reference/nn/tasks/.
