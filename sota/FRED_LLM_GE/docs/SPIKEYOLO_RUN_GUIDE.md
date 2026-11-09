# Official SpikeYOLO Gen1 integration for FRED

This implementation adds the family key `spikeyolo`. It uses official
[BICLab Gen1 source at dc51f35](https://github.com/BICLab/SpikeYOLO/tree/dc51f35d0b6e970cd157080c98157a41af4392f5/SpikeYOLO_for_Gen1),
not a project-owned approximation. The stock architecture is the `s` scale used
by upstream `train.py`, initialized from scratch as requested. The only graph
adaptation is the one-class drone head. Its parameter count is 23,122,451.

The upstream model YAML uses `MS_GetT [3,1]`, so a released FRED RGB Event frame
gets a time dimension of length one. This is not the separately advertised T=2
Gen1 checkpoint and does not consume raw asynchronous FRED events. Original
Gen1 scores or energy claims are not FRED baseline results.

Read [PHASE1_RUN_CHECKLIST.md](PHASE1_RUN_CHECKLIST.md) for the shared verification
protocol. This guide specializes its family setup and baseline steps. Full
baseline runs and any cluster submissions require manual execution after review.

## Files and execution path

```text
sota/FRED_LLM_GE/
  third_party/spikeyolo/
    UPSTREAM.json             # pinned revision and per-file hashes
    compat.patch              # device handling and configurable resolution
    ultralytics/              # official fork, including model YAML and loss
    spikingjelly-0.0.0.0.12/   # official bundled SpikingJelly package
    INTEGRATION.md             # provenance, licenses and deviations
  seeds/seed_spikeyolo.yaml    # v2 backbone/head YAML genome, two option blocks
  seeds/seed_spikeyolo.py      # compatibility builder for the stock YAML
  phase1_detection/
    spikeyolo_source.py        # source integrity and import isolation
    spikeyolo_search_space.py  # strict graph genome -> official model YAML
    adapters/spikeyolo.py      # official trainer + shared FRED interface
    spikeyolo_smoke.py         # small CPU forward/backward check
    spikeyolo_training_smoke.py
    spikeyolo_baseline.py      # independent run-seed/verify-data/summarize
    tests/test_spikeyolo.py
  scripts/spikeyolo_stock.sbatch
templates/FRED_SPIKEYOLO/{ConstantRules.txt,Normal/,CrossOver/}
```

The shared `candidate_runner.py` imports/registers the adapter in both execution
modes. `src/cfg/constants_fred_yolo11.py` selects its seed, templates, mutation
policy and isolated output paths. `configs/phase1/protocol.yaml` owns the recipe.
RT-DETR/Faster R-CNN remain unregistered placeholders; their files are not used.

## Review the frozen recipe

Baseline and candidates use the same `challenging_search` train/validation
exports, 20 epochs, batch 16 and image size 1280 from the shared protocol.
SpikeYOLO uses the official Gen1 SGD defaults (lr0=0.02, lrf=0.004, warmup=10,
momentum=0.937, nbs=64, AMP, no image augmentation). This is a FRED adaptation
of the official implementation, not reproduction of the upstream 100-epoch
Gen1 experiment. Optimizer settings differ from YOLO11 and must be disclosed
in cross-family comparisons. Neither memory use nor runtime at 1280 is
established by the tiny CPU smoke test; the SLURM resource requests below are
the checklist starting allocation, not a measured runtime guarantee.

The official model, parser, spiking blocks and loss are retained. Four small source edits permit device-neutral construction, CPU smoke training,
and honoring configured resolution instead of upstream LetterBox's forced 320x320.
The adapter checks every actual training and inference tensor against the protocol.
Label caches are written inside each run so the old fork cannot rewrite YOLO11's
shared export caches. Runtime settings are private and external tracking callbacks
are disabled.
The adapter resets neuron state for independent samples, prevents upstream
Gen1 validation from selecting weights, and scores final-epoch FP32 weights
with the existing COCO FRED evaluator. Official final/best pickle checkpoints
may also be written by the trainer, but `trained_models/seed.pt` is the reported
final-epoch state dict. Actual trainer options are preserved separately.

## 1. Install and check locally

Run from the repository root. Do not replace the project's PyPI Ultralytics
with the old fork: source activation is specific to a fresh SpikeYOLO process.

```bash
module load uv
uv sync --extra spikeyolo
export LLMGE_AUTO_START_SERVER=0
uv run --extra spikeyolo python -m pytest -q \
  sota/FRED_LLM_GE/phase1_detection/tests/test_spikeyolo.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_challenging_split.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_protocol_fingerprint.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_result_contract.py \
  sota/FRED_LLM_GE/phase1_detection/tests/test_candidate_runner.py \
  tests/operators/test_fred_population_integrity.py \
  tests/operators/test_fred_seed_generation.py
OMP_NUM_THREADS=1 uv run --extra spikeyolo python -m \
  sota.FRED_LLM_GE.phase1_detection.spikeyolo_smoke --backward
OMP_NUM_THREADS=1 uv run --extra spikeyolo python -m \
  sota.FRED_LLM_GE.phase1_detection.spikeyolo_training_smoke
uv run --extra spikeyolo python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py --help
```

The first smoke uses 64x64 inputs and two loss/backward steps. The second uses
four synthetic images for two CPU batches through the official dataset/trainer;
it writes only to a temporary directory and removes it afterward. Neither
downloads FRED, submits jobs, nor runs LLM-GE.

Review and commit code before baseline training; the fingerprint includes the
Git commit, protocol, seed, runner, adapter, source snapshot and package versions. The schema is recorded as
`families.spikeyolo.genome_version` and in evaluation metadata.
Changing them invalidates the baseline. Record the commit and local status:

```bash
git rev-parse HEAD
git status --short
```

## 2. Verify the shared released Event-frame export

Unset frame-encoding overrides and follow checklist section 2 if export is
incomplete. Reuse its eight-shard export; do not change the dataset split.

```bash
unset FRED_EVENTS
mkdir -p run_job_outputs/export run_job_outputs/baseline
uv run --extra spikeyolo python sota/FRED_LLM_GE/phase1_detection/export_split.py \
  --set search --finalize | tee run_job_outputs/export/finalize-search.log
rg 'search_train_step50: 152 sequences, 10053 frames' run_job_outputs/export/finalize-search.log
rg 'search_val_step10: 20 sequences, 6733 frames' run_job_outputs/export/finalize-search.log
uv run --extra spikeyolo python -m \
  sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline verify-data
```

Review the per-sequence timestamp reports as specified in the checklist.
`verify-data` requires the current sequence/frame counts and completed export
markers. Both partitions must come from `challenging_train`; `challenging_test`
is excluded from architecture selection.

## 3. Run the three independent stock baselines manually

No LLM server or evolution controller is needed. Each run uses `--model network`
and a fresh `seed_N` output directory. Existing directories are refused.

For the full three-seed array after reviewing code and allocation:

```bash
BASELINE_JOB_ID=$(sbatch --parsable --export=ALL sota/FRED_LLM_GE/scripts/spikeyolo_stock.sbatch)
BASELINE_JOB_ID=${BASELINE_JOB_ID%%;*}
squeue -j "$BASELINE_JOB_ID"
sacct -j "$BASELINE_JOB_ID" --format=JobID,State,ExitCode,Elapsed,MaxRSS -P
rg -n 'Traceback|ERROR|Exception' run_job_outputs/baseline/spikeyolo-"$BASELINE_JOB_ID"_*.out
```

Require tasks 0, 1 and 2 to finish `COMPLETED`/`0:0`. An empty queue is not
success. Since runtime/memory are unmeasured at FRED resolution, you can first
use `sbatch --array=0 ...`, inspect seed 0, then `sbatch --array=1-2 ...` with
the same script and frozen protocol. Never resubmit seed 0 into its existing
directory. Inspect failed evidence and deliberately archive it before retrying.

Equivalent foreground commands on an already allocated GPU:

```bash
uv run --extra spikeyolo python -m sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline run-seed --seed 0
uv run --extra spikeyolo python -m sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline run-seed --seed 1
uv run --extra spikeyolo python -m sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline run-seed --seed 2
```

Each seed must contain:

```text
baseline_validation/phase1_detection/spikeyolo_stock/seed_N/
  baseline_start.json         # fingerprint saved before training (new runs)
  baseline.json
  trained_models/seed.pt
  results/seed_results.csv
  results/seed_evaluation.json
  training_args/seed/
    args.yaml                 # shared protocol and scratch initialization
    official_args.yaml        # actual official trainer settings, pretrained=false
    model.json                # exact resolved stock graph
    runtime.json              # actual batch size in pixels and device/software
    run.json                  # metrics, parameters, wall time
    official_train/           # original trainer artifacts and args
```

Summarize after all three runs succeed:

```bash
uv run --extra spikeyolo python -m sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline summarize
uv run --extra spikeyolo python -m json.tool \
  sota/FRED_LLM_GE/baseline_validation/phase1_detection/spikeyolo_stock/summary.json
```

The summary reports per-seed metrics, parameters, elapsed wall time, means and
sample standard deviations. Validation rejects failed/nonfinite fitness,
incorrect split/coverage, changed stock graph, wrong random seed, altered
checkpoints, trainer options, stale code/data/dependencies or missing artifacts.
Do not describe the family as validated before this succeeds and results have
been reviewed under the checklist baseline gate.

### Recovering completed training after a validation-only failure

The original validator compared full-precision JSON metrics with the shared CSV
writer's six-decimal values using exact equality. For example, JSON
`0.42908857953944` becomes CSV `0.429089`. The corrected validator applies that
same serialization before comparison, while retaining exact parameter and
6,733-frame coverage checks. Do not edit the CSV or report to make them agree.

New runs save `baseline_start.json` before launching training. If training and
evaluation completed but the final validation step failed, retain all artifacts
and logs, fix the validator, then validate and create the missing completion
record without a GPU:

```bash
uv run --extra spikeyolo python -m sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline \
  recover-seed --seed 0
```

Recovery refuses an existing `baseline.json`. It checks export completeness,
stock architecture, trainer options, checkpoint hash, metrics, and provenance.
Only the baseline validator's file hash and Git commit may differ from the
observation; every other fingerprinted execution input must match. The original
training revision must agree with the observation. Both identities and the
recovery audit remain recorded. Normal, unrecovered runs retain strict complete
fingerprint equality during summarization.

**Existing seed 0 (job `6776381_0`):** the legacy runner did not persist its
pre-training fingerprint. After the user confirmed no model/training/evaluation,
protocol, or dependency changes since submission, a dated
`seed_0/baseline_recovery_observation.json` was captured **after training and before
the validator fix**, including all eight saved-artifact hashes. This is explicitly
an attested post-training observation, not a historical pre-training fingerprint.
Its recovery command is:

```bash
uv run --extra spikeyolo python -m sota.FRED_LLM_GE.phase1_detection.spikeyolo_baseline \
  recover-seed --seed 0 --observation baseline_recovery_observation.json
```

Run that command only while `baseline.json` is missing; once recovered, proceed
with seeds 1 and 2. Retain the original failed SLURM status and traceback as
evidence of the validation bug; recovery does not change scheduler history.
It does not certify other failed jobs or incomplete training/evaluation.

```bash
sbatch --array=1-2 --export=ALL sota/FRED_LLM_GE/scripts/spikeyolo_stock.sbatch
```

Review and finish commits before submitting those two runs, then keep their code,
environment, and protocol frozen until summarization. Recovered seed 0 can coexist
with those runs across a validator-only commit: the summary verifies execution
input hashes and includes its recovery audit. Do not launch evolution until all
three saved runs pass `summarize` and the results have been reviewed.

## 4. Prepared evolution hook (after the baseline gate)

The family now uses **spikeyolo-graph-v2** in `seeds/seed_spikeyolo.yaml`.
Like YOLO11, it has two `# --OPTION--` blocks containing complete `backbone:`
and `head:` YAML sections. Generated candidates retain the controller's `.py`
filename convention but contain YAML. `head:` includes the neck and its final
frozen `SpikeDetect` row. The Python stock builder remains as a compatibility
entry point; it contains no evolvable architecture.

The stock YAML resolves to the same pinned official s graph, including original
raw repeats `(3,6,9,1)` in the four backbone residual stages. No upstream model,
block implementation, neuron, loss or parser changes are needed for this expansion.

| Exposed decision | Legal edits and reason |
| --- | --- |
| Output widths | Ten downsampling/projection/SPPF rows; multiples of 64 between half and twice each stock raw width. SPPF/projector maxima also respect upstream scale clipping, avoiding redundant widths. The official parser derives connected input widths and detection inputs. |
| Residual block choice | Nine backbone/neck positions can select official `MS_ConvBlock` or `MS_AllConvBlock`; both preserve channels, time axis, spatial size and residual interfaces. ANN substitutions are excluded. |
| Residual depth | Raw repeats `1,3,6,9,12`, giving effective depths `1,1,2,3,4` at s scale. Raw 3 remains valid for exact stock fidelity; effective no-ops are rejected. Projection/downsampling rows stay at one repeat because upstream repeating those constructors need not preserve channels or strides. |
| Expansion | Each residual block exposes its official MLP ratio `1,2,3,4`; this changes internal capacity without changing its external channel interface. |
| Receptive field | Residual separable kernels and SPPF pooling kernels allow `3,5,7`; standard projection kernels allow `1,3,5,7`. Downsamplers allow `3,5,7` with padding coupled to kernel to retain the original output scale. |

Nine backbone rows (global indices 1..9) and ten neck rows (10,12,14,16,18,19,
20,22,23,25) are mutable. Every layer position, `from` link, `Concat` axis,
upsample, downsampling stride, first-layer flag, `MS_GetT`, detection row,
class count, scale constants and neuron implementation remain protected.
Training, temporal representation, scratch initialization and evaluation stay frozen.

Mutation changes exactly one **effective layer**, and can change several coupled
constructor fields within that row. Crossover can combine multiple rows inside
one corresponding section. Its prompts allow one extra bounded edit if simple
inheritance would copy a parent. It therefore is not restricted to strict parental
inheritance. Parents from the old depth-tuple genome cannot be crossed with v2.

Validation occurs before training: strict safe YAML with duplicate-key rejection,
version/preamble and topology checks, exact constructor signatures, per-field
ranges, effective mutation size, official model construction, P3/P4/P5 feature
shapes and strides, finite predictions and neuron-state reset checks. CPU preflight
uses both `[1,3,64,64]` and `[2,3,64,96]` inputs. The adapter also calls preflight,
so directly invoking `candidate_runner.py` cannot bypass these graph checks.
The backward smoke checks representative choices in every new decision category.

Architecture hashes include the instantiated operation graph, routing, strides
and state tensor shapes. This distinguishes weight-free pooling-kernel changes
and merges raw-repeat aliases. Cached evaluations remain tied to the protocol,
training seed, v2 schema and pinned code/source identity.

These bounds implement the requested wider prototype search; they do not establish
that every legal combination fits an H200 at FRED resolution. Parameter counts are
reported, and no unmeasured global parameter multiplier is borrowed from YOLO11.
The formal resource envelope still needs hardware evidence under DG-P1-05.

**Use a fresh run tag and freeze the new code version.** The old one-tuple `.py`
genome is intentionally rejected instead of silently migrated. Protocol, seed,
policy and graph identities change, so old cached results/checkpoints and baseline
summaries cannot be resumed as v2 results. Stock numerical architecture and training
recipe are unchanged; the usual three-seed baseline gate still applies.

After baseline and research review, family resolution can be inspected without
launching anything:

```bash
export FRED_SEED_FAMILY=spikeyolo
export FRED_RUN_TAG=spikeyolo_20261101_eot0_review1
unset FRED_EVENTS
# Keep optional dependencies installed in the operator/controller environment.
export UV_NO_SYNC=1
uv run --extra spikeyolo python -c 'from src.cfg import constants as c; print(c.SEED_FAMILY, c.SEED_NETWORK, c.CANDIDATE_POLICY_MODULE)'
```

Use a new dated tag when a new experiment is actually approved. No changes to
`run.sh`, `server.sh`, the LLM server, split exporter or evaluator are needed.
