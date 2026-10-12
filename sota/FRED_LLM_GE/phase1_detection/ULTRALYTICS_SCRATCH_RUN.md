# Option 2: actual YOLO11 architecture evolution from scratch

This branch starts from `MosesTheRedSea-main` at `0c4e892b0`. The user requested
the team's option 2: actual Ultralytics YOLO11 as generation-zero architecture,
LLM-written backbone/neck mutations and crossovers, every individual initialized
randomly. No COCO checkpoint or parent checkpoint is loaded. This is an
engineering experiment, not a finalized research protocol or FRED paper reproduction.
It does not change the older primary-study pretraining recommendations.

## What changed

- `seeds/seed_ultralytics_yolo11.py` constructs pinned Ultralytics 8.3.203 YOLO11.
  The standalone seed defaults to **m**; the option-B launcher selects **n**
  to match the team's current search setup. The unchanged seed executes the
  official graph exactly, including its Detect head and P3/P4/P5 outputs.
- Two mutable Python classes contain the actual backbone and neck. They may
  add/replace PyTorch blocks, not merely choose among preset parameter values.
  The protected wrapper, Detect head, detection loss and NMS remain fixed.
- `seeds/yolo11/train_ultralytics.py` uses official `v8DetectionLoss` with the
  existing AdamW loop, FRED event frames, annotation handling and COCO
  evaluator. Option B reuses the team's native PNG export and lazy disk loader. It pads cached tensors on the bottom/right to multiples of 32;
  labels and predictions retain their original coordinate meaning. It does
  not change raw data or the protected evaluator.
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

### Option B: team protocol and long runs

This adopts the provisional search setup from `fred-run-setup` at
`664b67fe7f5e62329db7710b7634b3c6e960906c`: YOLO11n, random weights, width 1280,
batch 16, 20 epochs; 152 training sequences sampled every 50th valid event frame
and 20 validation sequences sampled every 10th. The exact team validation IDs
are frozen in `configs/phase1/yolo11_option_b.json` (profile v2); results from the
older v1 random partition cannot be resumed into this workload. All 59 official
challenging-test sequences remain excluded from search. Archive folder names
are storage locations; 33 challenging-training sequences live in HF `test/`.

The team code supplies `export_split.py`, `fred_splits.py`, `proxy_data.py` and
`protocol.yaml`. The existing timestamp checker is reused for downloaded ZIPs.
CPU jobs download one archive per sequence, export sampled native event PNGs,
YOLO labels and exact pixel-box records, then remove the temporary archive.
Markers include the pinned dataset revision and are checked on reruns. A final
CPU job checks completeness. Images stay on disk until a training or validation
batch needs them, including shuffled training batches. No full-cohort tensor
allocation is required.

**Training distinction:** we preserve our protected AdamW loop (lr 0.001,
weight decay 0.0005), v8DetectionLoss, stride-32 padding and NMS. We reuse their
data and search budget; this is not identical to their full Ultralytics `.train`
recipe with its optimizer/augmentation defaults. This keeps arbitrary Python
backbone/neck edits executable instead of reconstructing a YAML model and
losing those edits. Baseline and evolved candidates use the same loop.

On ICE, from this branch after `uv sync`, authenticate Hugging Face if needed
and prepare the data:

```bash
uv run python -c "from huggingface_hub import login; login()"
bash scripts/prepare_yolo11_option_b.sh
```

The script prints an eight-shard CPU array (at most four active shards), dependent
finalization job, log folder and status command. Preparation is idempotent;
rerun failed shards or rerun the launcher. Export completion and frame totals
must come from ICE logs, not a timing estimate. The CPU jobs have four-hour
limits and request 16 GB RAM each; no GPU is requested for data preparation.
The initial 8 GB request caused an observed ICE OOM kill in export shard 1;
completed sequence markers allow retrying only the failed shard.

After preparation succeeds, check that the LLM server is still running and
ready at the configured port (8293). Start a fresh server with the existing
`SUBMIT_ISLAND_CONTROLLER=0 sbatch --parsable server.sh` if needed. Then:

```bash
bash scripts/run_yolo11_option_b.sh
```

The launcher validates complete export markers and server readiness, then
submits a four-hour GPU seed baseline. Evolution runs only after baseline
success. The CPU controller has a 7h45m allocation and submits GPU candidate
jobs through the existing four-hour/32GB scheduler configuration. Population
is 4 and the default target is **1,000 generations** (`LLMGE_GENERATIONS=1001`);
this is a stopping target across resumed allocations, not a throughput promise.
Queue delays, candidate cost and invalid candidates determine actual progress.
The current server also has an eight-hour limit and may end before the controller.
The controller stops when the server becomes unavailable rather than repeatedly
creating failed jobs. Interrupted work resumes from the last *completed*
population checkpoint; an unfinished generation may be repeated.

The launcher prints its exact run folder and resume command. After the
controller ends, with a ready server:

```bash
bash scripts/run_yolo11_option_b.sh --resume /absolute/path/to/the/run-folder
```

Resume restores the saved settings, skips the completed baseline, and verifies
seed/trainer/protocol/manifest identities. A live controller for the same run
blocks duplicate submission; a file lock also protects execution. Checkpoints
are written atomically, including generation 0 after initialization. Resume
pairs the population with the same generation's global state. If initialization
was interrupted before generation 0 completed, it restarts initialization while
reusing the completed baseline. Keep the same
checkout, generated model files, CSVs, checkpoints and global lineage directory.
Already submitted child jobs may outlive the controller; inspect the queue
before starting another allocation.

Each run saves its manifest, protocol, settings and baseline evidence. Candidate
reports include source revisions, manifest identity and trainer/seed hashes.
The metadata, clipping and split remain an engineering workload pending the
formal grouping/protocol gates. Full ICE throughput and accuracy are unverified.

### Optional small setup pilot

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
  --preflight-only --scale n
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
