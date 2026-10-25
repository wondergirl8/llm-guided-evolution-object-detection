# BN-SNN: experimental raw-event family

This extension implements the family setup described in [PHASE1_RUN_CHECKLIST.md](PHASE1_RUN_CHECKLIST.md). Use the family key `bn_snn`. It preserves the existing bistable-neuron prototype and trains it directly from scratch with surrogate gradients. No pretrained checkpoint is loaded.

**Scope:** this is a project-owned experimental model inspired by [Yasir and Kim (2025)](https://doi.org/10.1371/journal.pone.0327513), not a verified reproduction. In that paper, BN means **bistable neurons**. Our backbone also contains batch normalization. The paper's ANN-to-SNN conversion study and this directly trained prototype are different methods.

The input is `[batch, time, polarity, height, width]`. Each sample contains raw asynchronous events grouped into four equal time bins over a 33,333-microsecond window. No rendered PNG pixels enter the model. Computation still uses dense PyTorch convolutions at discrete time steps: this is **not an event-by-event sparse execution engine**, and events within one bin lose their relative order. Search frame subsampling also means we do not train on every event in a complete flight. Membrane state persists between bins of a window and resets between windows.

The formal Phase 1 plan puts alternate raw representations outside its initial frame-based scope (Appendix A.6). This user-requested extension is kept as a separate experimental family. It does not establish completion of the project's formal data/seed gates. Establish its own baseline before evolution; do not interpret a difference from YOLO as an architecture-only comparison when representation and optimizer also differ.

## What changed and why

| File | Responsibility and reason |
| --- | --- |
| `seeds/seed_bn_snn.py` | Small Python genome with two mutable assignment blocks. Imports and builder are protected. Candidates retain the controller's `network_<id>.py` naming convention. |
| `phase1_detection/seeds/bn_snn/network.py` | Existing BIF dynamics, surrogate derivative, batch normalization and temporal averaging retained. Stage execution now supports configurable depths, with default depths still `(2,2,2,2)`. Channels and projections remain derived consistently. Input checks reject unsupported tensor shapes. |
| `phase1_detection/bn_snn_search_space.py` | Parses Python syntax without executing candidate code. Rejects changed protected code, extra statements, calls, unsupported widths/depths and marker corruption. Builds and checks SSD output shapes, computes graph identity, rejects no-op edits and supplies the existing cache interface. |
| `templates/FRED_BNSNN/` | Mutation/crossover instructions for Python assignments, not YOLO YAML. Retry feedback uses the existing shared operators; training and neuron definitions cannot be changed by a response. |
| `.gitignore` | Excludes the derived raw-event cache from source control. |
| `phase1_detection/bn_snn_data.py` | Prepares a separate raw-event cache; preserves annotation times, boxes and sample IDs from the existing search export. Uses the existing timestamp parser and ECF decoder. Stores only the raw events inside annotated search windows, then loads those events through bounded memory maps and bins counts on demand. |
| `phase1_detection/seeds/bn_snn/detection.py` | Frozen SSD anchors, box encoding/decoding, IoU assignment and multibox loss. The original backbone did not supply a trainable detection objective or decoded boxes. |
| `phase1_detection/adapters/bn_snn.py` | Builds, trains and predicts through `FamilyAdapter`. Microbatches limit temporal activation memory; loss weighting preserves the effective optimizer batch. Prediction returns pixel `xyxy` boxes and drone confidence. |
| `phase1_detection/candidate_runner.py` | Registers BN-SNN in both import modes, dispatches raw-event loading, supports spatial dimensions at the end of either 4-D images or 5-D event tensors, and records BN-SNN recipe, raw-manifest identity and duration. Existing shared scoring and failure sentinels remain in use. |
| `phase1_detection/bn_snn_baseline.py` | Standalone seeds 0/1/2 through the same candidate runner, without an LLM. Refuses overwrites and invalid/stale results; summarizes metric means and sample standard deviations. The YOLO-specific baseline runner stays unchanged. |
| `phase1_detection/protocol_fingerprint.py` | BN-SNN fingerprints include raw manifests, protected implementation/evaluation sources and relevant dependency versions. Cache reuse cannot silently cross these protocol identities. |
| `configs/phase1/protocol.yaml` | Records the provisional BN-SNN temporal representation, fixed optimizer and SSD settings. Existing families retain their configured numerical settings. |
| `src/cfg/constants_fred_yolo11.py` | Adds the family seed, prompts and policy. Requires a tagged output directory and rejects frame-encoding overrides for BN-SNN. This is the shared FRED config despite its historical name. |

No changes are needed in `run_improved.py`, `llm_mutation.py`, `llm_crossover.py`, the server, or shell/SLURM scripts. The existing optional-policy interface provides preflight, parent no-op rejection, duplicate evaluation reservation/cache reuse, and generation-0 diversity. Later generations retain the controller's existing selection behavior; they are not newly forced to refill to eight.

## What can evolve

The first block is `BASE_CHANNELS`, one of `16, 24, 32, 40, 48`. The second is `STAGE_DEPTHS`, a tuple of four values in `1..3`. The stock seed uses width 32 and depths `(2,2,2,2)` and has **2,894,152 trainable parameters** with one foreground class.

Width scales all stages together; depths change the number of residual blocks in individual stages. There are 5 × 3⁴ = **405 permitted configurations**. These bounds are a conservative initial experiment choice, not values claimed to be optimal by the paper. They provide more than eight distinct initial architectures while keeping stage connectivity fixed. Mutation changes width or one depth entry; crossover may combine depth entries. The policy rejects a child matching either parent.

Protected: two input polarity channels; four-stage topology; downsampling and residual projections; stride-4/8/16 detection features; four anchors per location; two internal classifier outputs (background 0, drone 1); surrogate gradients; thresholds/phases; state reset; temporal averaging; anchors; loss; optimizer; event representation; split/label access; decoder; evaluator. The adapter translates the single foreground class into the evaluator's class 0.

A graph hash describes the instantiated architecture, excluding random weights and comments. The evaluation-cache key additionally includes the run fingerprint, including protocol seed. The same architecture trained under a different recipe/seed is not interchangeable. The map cache is process-local and capped at two sequences: it keeps file mappings open, not trained results or whole sequences copied into RAM.

## Initial training recipe

| Setting | Initial value and reason |
| --- | --- |
| Data partition and annotation times | Same challenging-train search split as existing export. No challenging-test samples are prepared for search. HF folder names are not official split labels. |
| Resolution | Native 1280×720; no new spatial interpolation or target-coordinate conversion. |
| Event window | 33,333 µs, matching the existing released-frame window. Both endpoints are open. |
| Temporal bins | Four fixed equal-duration bins, covering two complete periods of the current period-two BIF gates. Provisional cost/temporal-detail choice. |
| Encoding | Separate negative/positive event **counts**, not `/255` images or binary occupancy. Silent bins remain zero. |
| Epochs, effective batch, seed | BN-SNN-specific `families.bn_snn.training.epochs` (currently 8), shared `proxy.batch_size` (16), `proxy.seed` (0). Each BN-SNN baseline/candidate uses this recipe; other families retain `challenging.search.epochs` (currently 20). |
| Optimizer | Explicit AdamW, lr 0.001, weight decay 0.0001, gradient norm clip 1.0. A starting recipe for this directly trained prototype, not a tuned or paper-reproduced recipe. |
| Microbatch | 1, with gradient accumulation to the effective batch. BatchNorm observes microbatches/time steps, not the accumulated batch of 16. |
| AMP / augmentation | Disabled / none. No Ultralytics AMP probe, pretrained weights, or image-color augmentation. |
| SSD priors | Pixel sizes 16/32/64 at strides 4/8/16; aspect ratios 0.5/1/2/4 per location. Initial small-object hypotheses; held fixed across seed/candidates. |
| SSD loss | Positives at IoU ≥0.5 plus unique forced matches for targets; cross-entropy + Smooth L1, normalized by positive count; 3:1 hard negative mining. Empty images contribute three background anchors. |
| Postprocessing | Shared confidence/NMS thresholds; top 1,000 before NMS, at most 300 output detections. Shared COCO evaluator retains its existing maxDet=100 metric definition. |
| Evolution objectives | Existing mAP50, mAP50:95 and parameter count. Firing activity is not an energy measurement and is not an added fitness objective. |

Changing epochs, temporal bins, optimizer, anchors or source code requires a new baseline and fresh evolutionary tag. Adding this protocol section also changes the existing whole-protocol fingerprint: old checkpoints cannot be resumed under this edited protocol. Preserve the original checkout/protocol when resuming an old experiment.

## 1. Local checks before using GPUs

From the repository root, using the existing environment:

```bash
module load uv
readlink src/cfg/constants.py
LLMGE_AUTO_START_SERVER=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m pytest -q \
  sota/FRED_LLM_GE/phase1_detection/tests/test_bn_snn.py \
  tests/operators/test_bn_snn_operators.py
uv run python -m sota.FRED_LLM_GE.phase1_detection.candidate_runner --help
```

`readlink` should identify `constants_fred_yolo11.py`; the help output must list `bn_snn`. Tests use synthetic events and mocked operator responses, not a server or live FRED downloads. Existing environment dependencies are PyTorch/torchvision, NumPy, PyYAML, h5py, Hugging Face packages and pycocotools; raw preparation also needs `g++` for the repository's ECF decoder.

## 2. Prepare raw events once

First complete the existing checklist's search export and finalize step. BN-SNN reuses its `done/*.json` labels, sequence membership and subsampled annotation times. It does **not** use the exported images as input. This choice avoids creating a second, inconsistent label parser.

Use CPU allocations with sufficient memory to decode one raw sequence. Keep Hugging Face's disposable cache on node-local storage rather than home. For example, submit all eight shards with:

```bash
mkdir -p run_job_outputs/raw_events
sbatch --array=0-7 -c 4 --mem=64G -t 4:00:00 \
  --output=run_job_outputs/raw_events/raw-%A_%a.out \
  --wrap 'module load uv
export HF_HUB_CACHE=/tmp/$USER/huggingface-$SLURM_JOB_ID/hub
export HF_XET_CACHE=/tmp/$USER/huggingface-$SLURM_JOB_ID/xet
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/tmp/$USER/uv-cache-$SLURM_JOB_ID
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$UV_CACHE_DIR"
uv run python -m sota.FRED_LLM_GE.phase1_detection.bn_snn_data \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 8'
```

Run shards **0 through 7**, each once, using the same `--num-shards 8`. Raw downloads need HF access. Preparation writes only the new `families.bn_snn.raw_cache`, never raw source archives or the existing image export.

Cache format `annotated_windows_v1` retains each event tuple `(x,y,p,t)` within every configured 33,333-microsecond sample window and records exact per-sample offsets. It omits events that no search sample can read. Temporal binning still happens lazily during training, so this is raw-event input rather than a cache of rendered PNGs or fixed voxel grids.

Repositories that started with the older full-sequence `.npy` cache can convert and resume without redownloading completed sequences. Run the conversion as one job first so each old file is removed before the next is converted:

```bash
sbatch -c 4 --mem=64G -t 4:00:00 \
  --output=run_job_outputs/raw_events/compact-resume-%j.out \
  --wrap 'module load uv
export HF_HUB_CACHE=/tmp/$USER/huggingface-$SLURM_JOB_ID/hub
export HF_XET_CACHE=/tmp/$USER/huggingface-$SLURM_JOB_ID/xet
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/tmp/$USER/uv-cache-$SLURM_JOB_ID
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$UV_CACHE_DIR"
uv run python -m sota.FRED_LLM_GE.phase1_detection.bn_snn_data \
  --shard 0 --num-shards 1'
```

For every completed sequence this writes `SEQUENCE.windows.npy`, atomically updates its JSON index, and only then deletes the old `SEQUENCE.npy`. It then downloads and prepares any missing sequences. The operation is resumable after interruption.

After all shards succeed:

```bash
uv run python -m sota.FRED_LLM_GE.phase1_detection.bn_snn_data --finalize
```

Expect the same search membership/window counts as the existing export (currently train 152 sequences / 10,053 windows and val 20 sequences / 6,733 windows). Finalize checks cached file hashes, the cache format and the configured window duration; preparation reports timestamp findings in `raw_cache/bad_timestamps_<sequence>.csv`. Missing or stale inputs cause an error. Rerunning a completed shard verifies and reuses its cache. Selected event tuples are uncompressed, while events outside the sampled search windows are discarded because training cannot read them. Check per-sequence decoding memory and final cache size before starting GPU work.

Training reads the prepared cache and labels without downloading data. Memory maps are read-only. Treat finalized event files as immutable; after any intentional cache changes, rerun finalize and start a fresh experiment.

## 3. Establish the standalone baseline

Do this inside a GPU allocation, **without starting an LLM server**. The baseline command refuses an existing seed output directory; review failed attempts and select a fresh output root rather than overwriting them.

```bash
uv run python -m sota.FRED_LLM_GE.phase1_detection.bn_snn_baseline run-seed --seed 0
```

Inspect `baseline_validation/phase1_detection/bn_snn_stock/seed_0/` under `sota/FRED_LLM_GE` for:

- `results/seed_results.csv`: finite mAP50, mAP50:95, positive parameter count;
- `results/seed_evaluation.json`: validation-only provenance and temporal settings;
- `training_args/seed/args.yaml`: exact recipe, initialization and data manifests;
- `training_args/seed/run.json`: metrics and total wall time (setup, training, save and evaluation);
- `trained_models/seed.pt` and `baseline.json`: weights and verified baseline identity.

Only after seed 0 completes and GPU memory/runtime are acceptable:

```bash
uv run python -m sota.FRED_LLM_GE.phase1_detection.bn_snn_baseline run-seed --seed 1
uv run python -m sota.FRED_LLM_GE.phase1_detection.bn_snn_baseline run-seed --seed 2
uv run python -m sota.FRED_LLM_GE.phase1_detection.bn_snn_baseline summarize
```

These may be separate GPU allocations. `summary.json` records means and sample standard deviations. Small CPU contract tests establish software compatibility, not baseline accuracy. A successful full FRED baseline is still required before evolving candidates. Confirm the candidate's maximum permitted size fits the evaluation GPU/time allocation as well as the stock model.

## 4. Select a fresh evolutionary run

After the baseline is validated, use the existing server/run workflow. These example commands submit jobs only when **you execute them**:

```bash
export FRED_SEED_FAMILY=bn_snn
export FRED_RUN_TAG=bn_snn_20261024_eot0_trial1
export FRED_NUM_GENERATIONS=12
unset FRED_EVENTS
export LLMGE_PORT=8487
export SUBMIT_ISLAND_CONTROLLER=0
mkdir -p run_job_outputs/{server,islands,evolution,evaluation}
uv run python -c 'from src.cfg import constants as c; print(c.SEED_FAMILY, c.SEED_NETWORK, c.CANDIDATE_POLICY_MODULE, c.RUN_CHECKPOINT_DIR)'
sbatch --parsable server.sh
```

Choose your own unused tag and port. Keep only the intended server/controller pair for this checkout because `hostname.log` is shared. Wait for the server log to report **Uvicorn running**, then replace the example job ID:

```bash
export LLMGE_SERVER_JOB_ID=YOUR_SERVER_JOB_ID
sbatch --parsable run.sh
```

Tagged state goes under `fred_runs/$FRED_RUN_TAG/`; BN-SNN requires the tag so the shell script's historical YOLO checkpoint fallback is not used. The existing controller includes the stock seed in generation 0 and attempts eight distinct successful graphs. With `FRED_NUM_GENERATIONS=12`, checkpoints are 0 through 11. Server/evaluation time limits still apply; an initial integration does not guarantee completion in one allocation.

## Validation and remaining evidence

Local verification on 2026-10-24: **55 tests passed** in the focused BN-SNN, operator, candidate-runner, D-FINE, fingerprint, result-contract and YOLO training-protocol suite. `git diff --check` passed. D-FINE tests emitted existing worker-count and multiprocessing warnings (27 warnings total); no tests failed. Family activation printed the BN-SNN seed, policy and tagged checkpoint directory correctly.

Automated tests cover fixed-window timing/polarity, state reset, surrogate gradients, safe genome parsing, graph hash/no-op semantics, real shared mutation/crossover control flow with mocked responses, a CPU optimization step, shared evaluator artifacts, failure sentinels, raw preparation reuse/corruption, and existing detector regressions. No server, SLURM job or full FRED training run was started during implementation.

Still required on PACE: actual ECF raw download/preparation, full-resolution GPU memory/time measurement, finite learning curves and detection quality over the three baseline seeds, and a short real-LLM evolutionary run. This integration does not claim paper-equivalent bistable dynamics, neuromorphic energy savings, or an optimal recipe.
