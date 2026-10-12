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

If the previous server has ended, use this single command from the checkout:

```bash
bash scripts/resume_yolo11_option_b.sh
```

It reads `latest-option-b.txt`, rejects an active controller/resume/server job,
starts a fresh server and queues a small CPU job dependent on the server
starting. That job waits up to 40 minutes for model loading and checks the
server's Slurm job identity before submitting the controller. It reuses the
saved baseline and exported data. It prints server/resume IDs and the resume
log; the controller ID appears in that log and `controller-job.txt` after
startup. An optional run-folder argument selects an older run. This resumes
one allocation; it does not renew eight-hour server jobs indefinitely.

For automatic renewal toward the saved generation target, use:

```bash
bash scripts/resume_yolo11_option_b.sh --auto
```

This also attaches renewal to a saved run whose server/controller or readiness
job is already active. It schedules a small dependent CPU continuation job;
after an expected controller timeout or server expiration, that job resumes
the same run with a fresh server. Any outstanding candidate jobs are tracked
and allowed to finish before renewal, using Slurm dependencies rather than
holding a CPU allocation while they wait. The saved baseline and exported
dataset are reused on every allocation. Queue waits and model loading still
create gaps between allocations.

Renewal stops when the saved target checkpoint exists, on a real code failure,
manual cancellation, or after three successive allocations without checkpoint
progress. An unfinished generation can be repeated after a timeout; only
completed population checkpoints and saved initialization parents are resumed.
Current server/controller/continuation IDs remain in their run-folder marker
files, and continuation decisions are logged in `chain-<job>.log`.

To stop future renewals while allowing current jobs to finish:

```bash
touch "$(cat fred_ultralytics_yolo11_output/latest-option-b.txt)/STOP-AUTO"
```

Once current jobs end, remove that marker and run `--auto` again to re-enable
renewal. A genuine failure disables `AUTO-ENABLED`; inspect the controller and
chain logs before restarting it. The continuation cancels its own old server
after target completion or a real failure so the GPU allocation does not idle.

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

Initialization retains valid parents across at most three replenishment
batches (`LLMGE_INITIALIZATION_BATCHES`). Each batch saves its valid parents
in checkpoint 0, so partial success is available on resume. Exhausting the
budget stops with a diagnosis instead of generating an unbounded stream of
failed jobs. The active generation counter is restored after replenishment.

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

## Malformed LLM responses

The first live scratch evolution attempt produced prompt prose in candidate
files: an empty extracted response fell back to the entire mutation prompt.
Mutation now raises on empty generation rather than writing a candidate, and
the scratch response parser accepts only one complete Python class with the
expected name, constructor and forward method. Fenced and plain responses are
supported; empty/truncated blocks, prose, wrong classes and added imports are
rejected. The configured `max_gen_attempts` bounds retries with deterministic
format and model-error feedback. Unchanged mutation chunks and empty crossovers fail instead
of appearing to produce new valid architectures. Original provider output and
rejection reasons remain in the LLM job logs.

The protected training/evaluation protocol is unchanged, so the saved seed
baseline and CPU export can be reused. Stop the old controller before updating
code, then resume the existing run directory. Existing failure files remain
for diagnosis. Parser checks prevent malformed Python from reaching GPU
training. Before writing a mutation or crossover, generation now runs the
assembled model in a separate CPU process with a 60-second limit. It checks
construction, protected code, parameter count, square and rectangular
P3/P4/P5 interfaces, finite loss and gradients. Actual errors go back to the
LLM with its rejected proposal. GPU training retains its existing preflight;
passing a small CPU check cannot guarantee a successful full training run.

The instructions include the constructor signatures from pinned Ultralytics
8.3.203 and require preserving stage strides and connections. Completion
budget defaults to 4096 tokens, with concise class-only output. The Llama
instruction server uses its tokenizer's chat template and batches requests
only when their generation settings match. Each candidate has a
`models/network_<gene>.generation.json` audit containing response hashes,
rejections and accepted CPU results; full responses remain in job logs.
Each resumed launch saves generation-code hashes and limits in a separate
`generation-session-*.json`, leaving the training identity unchanged.

The latest inspected `fred-run-setup` branch at `5072e8ef` evolves constrained
YAML graphs and uses Ultralytics' full training recipe. Its latest runtime
fix warms YOLO label caches; our protected custom loop does not use those
caches. This branch adopts stronger structural guidance without merging a
different model representation or changing the saved baseline protocol.
No claim of a successful live generation is made by local tests.

## Local checks

The Pareto plotting script includes Michelle's numbered point badges and
separate metrics table from `fred-run-setup` commit `c875ec920`. Closely spaced
points have stacked badges. Plot this run's own global snapshots and preserved
baseline (the team's screenshot uses a different detector/training run):

```bash
root=$(cat fred_ultralytics_yolo11_output/latest-option-b.txt)
uv run python analysis/scripts/fred_pareto_graphs.py \
  --checkpoints "$root/global" --seed_results "$root/baseline/seed_results.csv" \
  --models_dir sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/models \
  --output "$root/pareto"
```

The output includes per-generation/final PNGs, a GIF when multiple generations
exist, and candidate/Pareto-model CSVs. Model paths use this branch's actual
Ultralytics seed. With no valid candidate scores, a saved baseline is plotted
alone; without either, the command reports that evaluations are not ready.

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
