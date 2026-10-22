# Phase 1 detection runner

## YOLO26n-P2 staged prototype

Project/task decision (2026-10-12): the user approved Steps 1 through 4 of a
separate, runnable YOLO26n-P2 prototype before the core Phase 1 freeze. This
is a narrow exception to the timing restriction in `PHASE1_DETECTION_PLAN.md`
Appendix A and `DG-P1-10`. It authorizes the fixed seed, standalone runner
integration, local verification, and gated LLM-GE/final-evaluation wiring.
It does not authorize formal FRED comparisons or an actual LLM-GE evolution
run. The reason is to verify the model and integration path incrementally.
The pre-freeze data policy and target-GPU resource use remain unverified, so
prototype outputs must not be reported as formal benchmark results. The family
has separate result and checkpoint paths; rollback is removal of its seed,
adapter, registration, prompts, and this note. Step 1 validation compares the
YAML with the pinned Ultralytics `8.4.175` P2 configuration and checks a
one-class build with P2/P3/P4/P5 detection strides.

After preparing the shared challenging-search export with the commands below,
run the fixed seed without LLM-GE:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family yolo26_p2 --model network
```

The runner selects `phase1_detection/seeds/yolo26_p2` as the default result and
checkpoint directory for this family. It shares the Event-frame export,
training budget, and evaluator with the YOLO11 adapter. The family entry in
`configs/phase1/protocol.yaml` explicitly selects `inference_head: one_to_many`:
Ultralytics applies NMS using `proxy.nms_iou`. Set `one_to_one` only for a
separately versioned NMS-free comparison. The previous adapter reported "no
NMS" but actually used the default one-to-many/NMS path, so historical report
metadata describing NMS-free inference is incorrect; its stored scores remain
the scores of the one-to-many path.

To re-score saved checkpoints on the **same search-validation export**, restore
that export at the protocol's `challenging.data_root`, then run from the
repository root (replace the checkpoint path with the handoff or cluster path):

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/rescore_yolo26.py \
  --checkpoint-dir sota/FRED_LLM_GE/phase1_detection/seeds/yolo26_p2/trained_models \
  --output-dir sota/FRED_LLM_GE/phase1_detection/seeds/yolo26_p2/head_rescore \
  --diagnostics
```

The command reads each checkpoint's sibling `results/<gene_id>_evaluation.json`
and checks the checkpoint, architecture, dataset revision, split, and frame
count before inference. It runs both heads at confidence 0.001 and 0.05; the
one-to-many head also uses NMS IoU 0.5 and 0.7. `--gene-id seed` limits a run,
and `--checkpoint-dir`, `--reference-results-dir`, and `--data-root` can point
to a restored handoff. Output goes only to the separate `--output-dir`, with
one validation report per setting; rerunning skips matching completed reports.
`head_rescore_summary.csv` collects the comparisons after all settings finish,
including AP75, small-object AP/recall, the AP50-to-AP75 gap, missed targets,
and localization counts when diagnostics are enabled. Its deltas compare each
setting against the re-scored one-to-many `eval_conf`/`nms_iou` baseline when
that setting is included in the sweep. `head_rescore_sequences.csv` lists the
per-sequence diagnostic results. A baseline rescore that differs from the
historical AP is a reproducibility warning; do not interpret policy deltas until
the export and runtime difference is resolved.
The NMS-free head is a separate inference choice and may score lower.

Add `--diagnostics` to the re-scoring command or to `candidate_runner.py` to
include AP75, COCO small-object AP and recall, per-sequence AP, IoU-0.50 false
negative counts, and up to ten example misses and boxes that match at IoU 0.50
but fail at 0.75. These are stored in the JSON report's `diagnostics` field;
the three-column fitness CSV and candidate ranking stay the same. Re-running
the re-scorer with the flag adds diagnostics to matching comparison reports
without changing the historical source reports. The original validation export
is required to compute them.

The three full-size families read the same challenging-search export, but
their default writable paths are separate:

| Family | Candidates, results, and trained checkpoints | Evolution state |
| --- | --- | --- |
| `yolo11` | `phase1_detection/seeds/ultralytics_yolo11/` | `fred_yolo11_output/`, `fred_yolo11_checkpoints/` |
| `yolo26_p2` | `phase1_detection/seeds/yolo26_p2_coverage_v1/` for the new LLM search | `fred_yolo26_p2_map5095_coverage_v1_output/`, `fred_yolo26_p2_map5095_coverage_v1_checkpoints/` |
| `dfine` | `phase1_detection/seeds/dfine/` | `fred_dfine_output/`, `fred_dfine_checkpoints/` |

Standalone YOLO26 seed runs still default to `phase1_detection/seeds/yolo26_p2/`.
The older `tiny_yolo11` tutorial family also has its own paths. An explicit
`--sota_root` overrides a family's result root and makes the default
`--variant_dir` follow that root.

Step 3 verification (2026-10-12): a one-epoch CPU smoke run on synthetic
64-pixel Event-like frames completed training, prediction, checkpoint save and
reload. A separate end-to-end runner smoke produced a validation CSV and JSON
through the shared evaluator. A 1280-pixel, batch-1 CPU prediction also passed.
These checks do not establish FRED accuracy.
No prepared FRED export or target GPU was available locally, so memory use at
the protocol's 1280-pixel, batch-16 setting remains unverified.

Step 4 wiring (2026-10-12): `FRED_SEED_FAMILY=yolo26_p2` selects the P2 seed,
its own bounded backbone/head mutation prompts, and separate evolution output,
island, result, and checkpoint paths. Candidate validation keeps all layer
routing, P2/P3/P4/P5 detection, the end-to-end head, and scale fixed. It permits
only bounded Conv channel/kernel and C3k2 channel/repeat changes. After a
validated standalone FRED baseline, the post-core `DG-P1-10` decision, and a
target-GPU resource check, the evolution entry point is:

```bash
FRED_SEED_FAMILY=yolo26_p2 bash run.sh
```

The YOLO26 pilot search now ranks candidates by search-validation mAP50:95,
using fewer parameters only to break an exact accuracy tie. It still records
mAP50 and keeps the multi-metric Pareto archive for tradeoff review. This
selection policy is versioned as `map50_95_primary_params_tiebreak_v1` in both
population and global checkpoints. The default YOLO26 output/checkpoint paths
have a matching `map5095_coverage_v1` suffix; older checkpoints cannot be
resumed under the new selection and coverage policies. This is a pilot search
setting, not the formal `DG-P1-06` fitness-contract freeze.

The first four candidates in a fresh YOLO26 LLM search now target narrower
P2, P3, P4, and P5 output layers (head indices 19, 22, 25, and 28), one per
candidate. Each proposal must reduce channels and the constructed model's
parameter count. Its proposal audit records the layer, value, architecture
hash, and before/after parameter counts. Before any evaluation job, the
controller writes `initial_yolo26_coverage.json` and refuses an incomplete or
duplicate panel. Subsequent mutation and crossover still use the broader
bounded edit space. This deliberate panel responds to the archived run, in
which all four evaluated variants widened only layer 0, layer 19, or both;
it does not assert that narrower layers will improve AP.

## YOLO26 controlled ablations

The separate [ablation plan](../configs/phase1/yolo26_ablations.yaml) compares
training duration, optimizer, learning rate, mosaic, and Event-image saturation.
The archived `seed` run used `optimizer=auto`, which selected AdamW at `lr0=0.002`,
momentum 0.9, and bias warmup LR 0.0 for one class. `seed_fixed_adamw` pins
those resolved values as the training comparison anchor, so the longer run
cannot silently switch optimizer. The optimizer comparison uses the same
learning rate on both sides. The plan also halves P4 or P5 output width
individually while leaving the seed's P2/P3 paths intact. The proposed
compact-model tests would
restore only the P2 or P3 detection-output layer relative to that compact model,
but that candidate source is absent from this workspace. They are omitted
from the runnable plan until its exact source is restored. Every listed
comparison changes one factor. Screen recipes on seed 0, then run the anchor
and promising recipes on paired seeds 1 and 2 before drawing a training-variance
conclusion.

First validate every proposed architecture locally without FRED data:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/yolo26_ablation.py --check
```

After restoring the same challenging-search export used by the historical run,
run one experiment/seed at a time, for example:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/yolo26_ablation.py \
  --experiment seed_fixed_adamw --seed 0 \
  --data-root sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging
```

Each run has its own directory under `seeds/yolo26_p2/ablations/`, with a model
YAML, checkpoint, diagnostic evaluation JSON, Ultralytics `args.yaml` (trainer
arguments), and `results.csv` (training curves). The report records the actual
optimizer, learning rate, momentum, bias warmup LR, initial model hash,
completed epochs, training time, validation sample identity, and file hashes.
Existing run directories
are refused to prevent accidental overwrites. This is a pilot search-validation
study, separate from the LLM-GE candidate rankings and the held-out test.

After the anchor and a selected arm have all three paired seeds, compare them:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/compare_yolo26_ablations.py \
  --experiment seed_no_mosaic \
  --output sota/FRED_LLM_GE/phase1_detection/seeds/yolo26_p2/ablations/no_mosaic_comparison.json
```

Omit `--experiment` only after every training arm has all three seeds. The
comparison rejects missing repeats, changed data/model/evaluator/runtime,
incomplete training, and different initial weights for paired seeds. Its gain
screen is exploratory; training results do not change the LLM-GE ranking or
authorize held-out evaluation.

## YOLO26 search-export data-path audit

The four archived YOLO26 evaluation logs each warn about slow sampled image
access. Train-image reads were reported at 4.4–5.8 MB/s and the same three
training frames (`119_001151`, `119_001751`, `119_002301`) appear in the
duplicate-label messages in every run. These messages come from the shared
`labels.cache`, so they are not four independent scans or twelve distinct
data errors. Ultralytics reported 10,053
training images, 1,826 backgrounds, and zero corrupt images in its cache scan.
Its speed warning samples only five files, so the logs do not establish a
training bottleneck or an accuracy effect. The original export is absent from
the handoff, so the duplicate source and actual I/O rate cannot be verified
locally.

On the machine holding the **original** search export, run the read-only audit
before the ablations, and save a second report after them:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/audit_yolo26_data_path.py \
  --data-root PATH/TO/challenging \
  --output PATH/OUTSIDE/EXPORT/yolo26_data_path_before.json
```

The audit compares every marker's pixel boxes to its YOLO label file, counts
duplicate rows separately when pixel boxes are already equal and when distinct
pixel boxes round to the same six-decimal label, checks image presence,
`train.txt` and `data.yaml` paths, and verifies the `trainer_val` copies. It
also records a deterministic bounded stat/read sample for each split; those
timings are host- and cache-dependent and are not detector latency. A
marker/label/path mismatch exits nonzero; duplicate rows are reported for
review without modifying labels. The ablation runner performs this check before
every training run, saves `data_path_audit.json` in that run directory, and
records marker and label hashes so paired comparisons reject changed exports.
Compare the before/after marker and label hashes to detect export changes
during trials. The markers hold post-clipping
boxes, so original FRED sequence ZIPs would still be needed to establish
whether a marker duplicate originated in source annotations or clipping.

Once candidate selection is frozen, prepare the final export and run one
held-out evaluation with the same family selector. The default remains YOLO11.
YOLO26-P2 final runs and reports are under the shared data root's `yolo26_p2`
directory.

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --finalize
uv run python sota/FRED_LLM_GE/phase1_detection/final_eval.py \
  --family yolo26_p2 --name yolo26n_p2_scratch
```

For an evolved candidate, add `--model network_<gene_id> --variant_dir
sota/FRED_LLM_GE/phase1_detection/seeds/yolo26_p2/models` to the final
evaluation command. Do not use challenging_test to choose candidates.

## D-FINE-inspired Event seed

The model is one file, [`../seeds/seed_dfine.py`](../seeds/seed_dfine.py). The
`dfine` family adapter converts the shared runner's `uint8` Event images and
pixel `xyxy` boxes to the seed's normalized training inputs. The runner uses
the same challenging-search export, sample IDs, evaluator, and result format as
the `yolo11` family. Its D-FINE model, matching, FGL, and GO-LSD losses live in
the seed file; generated candidates may replace only its three architecture
classes.

From the repository root, prepare the search export once:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set search --finalize
```

Run the unchanged seed without LLM-GE:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network
```

The result and checkpoint are written under the selected `sota_root`. For a
local interface check without downloading FRED, run:

```bash
python -m pytest -q sota/FRED_LLM_GE/phase1_detection/tests/test_dfine_adapter.py
```

Before ranking new D-FINE architectures, check whether the unchanged seed
repeats. First score one saved seed checkpoint twice on the same search
validation export:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/rescore_dfine.py \
  --checkpoint PATH/TO/trained_models/seed.pt \
  --reference-report PATH/TO/results/seed_evaluation.json \
  --device cuda --output PATH/TO/seed_rescore.json
```

The command checks checkpoint, model source, postprocessing, and validation
sample IDs against the original report. It reads the exported validation
frames locally and does not query Hugging Face. Its two passes record mAP and
a hash of all predictions. Then train the unchanged seed twice with the same seed,
protocol, export, and hardware:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 --repeat-id a --device cuda
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 --repeat-id b --device cuda
```

These runs have separate output folders under
`seeds/dfine/diagnostic_repeats/seed_0/` and refuse an existing repeat folder.
Each report records initial-weight, training-order, training-sample, model,
checkpoint, and recipe hashes; hardware and PyTorch/CUDA versions; and loss,
learning rate, and frame order for every epoch. Compare these fields and the
final metrics between the two reports. If the same checkpoint gives different
predictions, investigate evaluation. If initial weights or frame order differ,
investigate setup and seeding. If those match but early losses diverge,
investigate training runtime and operations before attributing a score change
to an architecture. The original results already show different first-epoch
losses across nominally identical D-FINE candidates; they lack these new
records, so they cannot identify the exact cause of the 19.14%–63.85% mAP50
spread. Use the search-validation export for these checks.

After both repeats, run the unchanged seed with two other training seeds
(`--training-seed 1` and `--training-seed 2`, without `--repeat-id`). Review all
three seeds with the two same-seed runs and the saved checkpoint re-score:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/compare_dfine_repeatability.py \
  --rescore PATH/TO/seed_rescore.json \
  --repeat PATH/TO/seed_0/repeat_a/results/seed_evaluation.json \
           PATH/TO/seed_0/repeat_b/results/seed_evaluation.json \
  --independent PATH/TO/seed_1/results/seed_evaluation.json \
                PATH/TO/seed_2/results/seed_evaluation.json \
  --output PATH/TO/seed_repeatability_review.json
```

The review identifies scoring, setup, or training-history divergence and
reports the mAP range across independent seeds. `complete: true` means the
same-seed run repeated and at least two additional seeds were compared; it
does not mean the seed meets an accuracy target. Old reports without initial
weights and epoch history are rejected rather than treated as repeatability
evidence.

Before changing D-FINE's detection or box logic, audit the search-validation
Event export with a saved seed checkpoint. Keep the original FRED sequence ZIPs
locally under `SOURCE_ZIPS/<hf_folder>/<sequence>.zip` so the audit can compare
exported frame indices and clipped boxes with `coordinates.txt` without a
download:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/audit_dfine_event_export.py \
  --checkpoint PATH/TO/trained_models/seed.pt \
  --reference-report PATH/TO/results/seed_evaluation.json \
  --source-zips PATH/TO/SOURCE_ZIPS \
  --device cuda --output-dir PATH/TO/seed_event_audit
```

The read-only audit checks every validation PNG's mode and size, marker/sample
identity, pixel boxes, YOLO label round-trip, target hash, and prediction box
coordinates. With source ZIPs it also checks source frame selection, one PNG's
pixels per sequence, and original annotations against the export's current
clipping behavior. `audit.json` lists any mismatch and counts clipped, dropped,
or malformed annotations. The output folder contains green target and orange
prediction overlays for small drones, empty frames, sequences with many misses,
misses, and localization errors. Inspect these images manually. Missing source
ZIPs or an older report without a validation-target hash leave `audit_complete`
false. Clipped or dropped source boxes require a Phase 0 annotation-policy
decision; this tool reports them and does not change labels or the dataset.
The current D-FINE search export uses native-size images; the runner's inverse
resize mapping is also checked with a nontrivial synthetic case in the audit
test.

To diagnose the D-FINE seed and leading candidates, train each on the same
search-validation export with at least two independent `--training-seed` values
(three are preferable if compute allows). For example, run the seed and a
candidate once each with seed 0, then repeat both commands with seed 1:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network_GENE_ID --training-seed 0
```

Replace `GENE_ID` with the candidate's ID and restore its `network_GENE_ID.py`
in `seeds/dfine/models/` first. The optional `--variant_dir` selects another
source directory. Each confirmation writes a checkpoint and diagnostic JSON
under `seeds/dfine/diagnostic_repeats/seed_N/`, away from evolution results;
an existing result for the same model and seed is refused. The JSON records
the training seed, model and training-recipe identities, and validation sample
IDs. The diagnostics include AP75, small-object AP/recall, per-sequence AP and
misses, and counts plus examples of boxes matched at IoU 0.50 but missed at
0.75. Compare the resulting JSON reports with
`compare_dfine_diagnostics.py <report-1.json> <report-2.json> ...`; it checks
that every model has at least two seeds and that the reports share one
evaluation and training protocol, then prints per-run metrics, seed spread,
and per-sequence results. Use only search validation for this analysis.
Without the original export and model files, these measurements cannot be
reconstructed from the screenshots.

For Step 4, re-score the saved seed and selected D-FINE candidate checkpoints
with `--diagnostics`. This adds AP75, small-object AP/recall, IoU-0.50 misses,
IoU-0.50-to-0.75 localization misses, and per-sequence metrics to each output.
The optional overlays draw ground-truth boxes in green and predictions in
orange on up to three Event frames from each error category:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/rescore_dfine.py \
  --checkpoint PATH/TO/trained_models/seed.pt \
  --reference-report PATH/TO/reports/seed_evaluation.json \
  --device cuda --diagnostics --examples-dir PATH/TO/review/seed_frames \
  --output PATH/TO/review/seed_diagnostics.json
uv run python sota/FRED_LLM_GE/phase1_detection/rescore_dfine.py \
  --checkpoint PATH/TO/trained_models/CANDIDATE_ID.pt \
  --reference-report PATH/TO/reports/CANDIDATE_ID_evaluation.json \
  --model network_CANDIDATE_ID --variant-dir PATH/TO/models \
  --device cuda --diagnostics --examples-dir PATH/TO/review/CANDIDATE_ID_frames \
  --output PATH/TO/review/CANDIDATE_ID_diagnostics.json
uv run python sota/FRED_LLM_GE/phase1_detection/compare_dfine_diagnostics.py \
  --checkpoint-review PATH/TO/review/seed_diagnostics.json \
  PATH/TO/review/CANDIDATE_ID_diagnostics.json
```

Use the same frozen validation export and protocol for every checkpoint. The
comparison refuses mismatched sample IDs, evaluation settings, training
recipes, and sequence ground truth. It prints each run, per-sequence results,
source identity, and candidate-minus-seed differences. Higher AP/recall and
fewer misses are better; a negative miss-count difference indicates
improvement. Its single-checkpoint mode is exploratory until Step 3 establishes
training repeatability; use the default comparison
mode after independent-seed confirmations. The overlays show all postprocessed
predictions and ground-truth boxes for the selected frame. The supplied result
bundle has checkpoints and reports but no Event frame export, so image review
must run where that export is available. Do not use challenging_test here.

For Step 5, compare postprocessing without retraining. The sweep loads one
saved checkpoint, evaluates confidence cutoffs 0.01, 0.05, and 0.10 with the
current IoU-0.5 NMS and with NMS disabled, then rechecks the original baseline.
It records mAP50, mAP50:95, AP75, small-object AP/recall, prediction count,
and changes from the baseline for each setting:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/compare_dfine_postprocessing.py \
  --checkpoint PATH/TO/trained_models/seed.pt \
  --reference-report PATH/TO/reports/seed_evaluation.json \
  --device cuda --output PATH/TO/review/seed_postprocessing.json
uv run python sota/FRED_LLM_GE/phase1_detection/compare_dfine_postprocessing.py \
  --checkpoint PATH/TO/trained_models/CANDIDATE_ID.pt \
  --reference-report PATH/TO/reports/CANDIDATE_ID_evaluation.json \
  --model network_CANDIDATE_ID --variant-dir PATH/TO/models \
  --device cuda --output PATH/TO/review/CANDIDATE_ID_postprocessing.json
```

Run the seed and selected candidate checkpoints on the same search-validation
export. Confirm both baseline checks pass before interpreting policy differences.
Choose one policy only after reviewing the same metric tradeoffs across those
fixed checkpoints; then freeze that choice in the protocol for subsequent
comparisons. The current protocol stays at confidence 0.05 and IoU-0.5 NMS
until those measurements exist. The held-out test is not used for this choice.

For a D-FINE training comparison with the architecture fixed, run the same
`--model` and `--training-seed` under three separate arms:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 --dfine-training-arm baseline
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 --dfine-training-arm longer_budget
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 --dfine-training-arm cosine_lr
```

Repeat the three commands with another seed and, if available, the same leading
candidate model. `baseline` keeps the search protocol's 20 epochs and constant
AdamW learning rate; `longer_budget` doubles only the epochs to 40; `cosine_lr`
keeps 20 epochs and changes only the learning-rate schedule (cosine with a
minimum of 10% of its starting value). These are provisional experiments,
not a change to the evolution training recipe. Each run writes under
`seeds/dfine/training_ablations/<arm>/seed_N/`. Its evaluation JSON contains
total training loss, the five raw loss terms, and learning rate for every epoch;
search-validation mAP50 and mAP50:95 at epoch 1, every fifth epoch, and the
final epoch; final AP75 diagnostics; and training time in seconds. The total
loss is `loss_cls + 5*loss_bbox + 2*loss_giou + loss_fgl + 0.5*loss_go_lsd`;
the individual terms are reported before those weights. Training time sums
epoch durations and excludes the interim validation checks.

After running all three arms with seeds 0 and 1, compare their reports:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/compare_dfine_training_arms.py \
  sota/FRED_LLM_GE/phase1_detection/seeds/dfine/training_ablations/{baseline,longer_budget,cosine_lr}/seed_{0,1}/results/seed_evaluation.json
```

The comparator requires complete paired seeds, the same model, data,
postprocessing, runtime environment, initialization and frame order, and
training settings that differ only in epochs or schedule. It reports each run,
per-arm averages, and paired differences from the baseline. Inspect mAP50:95,
AP75, the loss curves, seed spread, and training time before choosing a recipe;
an apparent gain from one seed is insufficient. The original search export and
model files are required for real measurements. Do not use the held-out test
for this choice.

Step 7 starts by measuring whether the seed's selected query locations reach
the missed small drones. Run this on a saved checkpoint with the original
search-validation Event export:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/analyze_dfine_queries.py \
  --checkpoint PATH/TO/trained_models/seed.pt \
  --reference-report PATH/TO/reports/seed_evaluation.json \
  --device cuda --output PATH/TO/review/seed_query_coverage.json
```

The report counts small objects (original area under 32x32 pixels) and misses
(no postprocessed prediction at IoU 0.5). `available` means *any* encoder
token's spatial center is inside the GT box; `selected` means one of the
decoder's top-ranked initial query centers is inside it. These are location-coverage
diagnostics, not AP scores. If many missed small objects are available but not
selected, test this single query-selection variant: half the fixed query slots
are reserved for stride-4 tokens, while the total query and memory limits stay
unchanged. The seed and its default training protocol are untouched.

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/make_dfine_p4_query_variant.py \
  --output sota/FRED_LLM_GE/phase1_detection/variants/dfine/network_p4_quota.py
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 \
  --sota_root sota/FRED_LLM_GE/phase1_detection/seeds/dfine/query_ablations
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network_p4_quota \
  --variant_dir sota/FRED_LLM_GE/phase1_detection/variants/dfine \
  --training-seed 0 \
  --sota_root sota/FRED_LLM_GE/phase1_detection/seeds/dfine/query_ablations
```

Repeat both training commands with seed 1 and compare their diagnostic reports
using `compare_dfine_diagnostics.py`. Check mAP50:95, AP75, small-object AP and
recall, and the new checkpoint's query coverage. If selected coverage is already
high, do not infer that query selection is the bottleneck; inspect the
classification and localization losses and the box diagnostics first. If even
all stride-4 token centers miss many GT boxes, this query variant cannot fix
that spatial-coverage limit. Test one architecture idea at a time.

The same step also has one opt-in localization experiment for the D-FINE seed.
Run it if higher-IoU box accuracy is the main weakness. It compares the
protocol's 16 refinement bins with 32, keeping the same model source, search
split, training budget, loss weights, and seed:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 --dfine-reg-max 16
uv run python sota/FRED_LLM_GE/phase1_detection/candidate_runner.py \
  --family dfine --model network --training-seed 0 --dfine-reg-max 32
```

Repeat with seed 1 before selecting an arm. Reports and checkpoints go under
`seeds/dfine/model_ablations/reg_max_<bins>/seed_N/`, away from evolution
results; an existing run is refused. Each diagnostic JSON records mAP50,
mAP50:95, AP75, parameter count, and median adapter inference time for one
preloaded validation frame (one warmup, three timed calls, same device). Match
the reports' `model_identity` and `model_ablation_protocol_identity` for the
same source, training, and evaluation settings; compare the same seeds and
device. Favor mAP50:95 and
AP75 for localization, with parameter count and inference time as resource
checks. This experiment does not change the default D-FINE search recipe or
select a model automatically. The original search export is required for
actual scores; freeze the selected model before a single held-out evaluation.

Step 8 confirms a real architecture change against the seed with paired
training seeds and the same frozen search-validation data, recipe, and
postprocessing. Use new confirmation runs: the earlier result bundle lacks
the per-run timing and exact target/code hashes required by this check. For
the stride-4 query variant, run both models with seeds 0,
1, and preferably 2, then review their diagnostic reports:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/compare_dfine_diagnostics.py \
  --confirm-gains \
  --output sota/FRED_LLM_GE/phase1_detection/seeds/dfine/query_ablations/gain_review.json \
  sota/FRED_LLM_GE/phase1_detection/seeds/dfine/query_ablations/diagnostic_repeats/seed_{0,1,2}/results/{seed,p4_quota}_evaluation.json
```

Use the same command with the actual candidate ID for another mutation. The
confirmation check requires at least two paired seeds and identical code,
training-frame identities, training order within each pair, evaluation setup,
and hardware. Three seeds are preferable. It reports mAP50:95, AP75, small AP,
and small recall for each pair. A metric passes its conservative variation
screen only when every paired gain is positive and its mean gain exceeds the
larger observed seed-to-seed range of the seed and candidate. This is a simple
screen, not a significance test. The review also records parameter-count and
preloaded inference-time changes; timing uses the same first validation frame,
one warmup, and three measured passes. Review accuracy and resource cost before
calling an architecture better or promoting it. No promotion happens
automatically, and the held-out test is not used for this choice.

After freezing the selected source and refinement-bin count from search
validation, the existing final export can train D-FINE on challenging_train
and score challenging_test once:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final
uv run python sota/FRED_LLM_GE/phase1_detection/export_split.py --set final --finalize
uv run python sota/FRED_LLM_GE/phase1_detection/final_eval.py \
  --family dfine --name frozen_dfine --dfine-reg-max 32
```

The `32` above is illustrative; use the value selected from the ablation.
For an evolved source, also pass its `--model network_<gene_id>` and
`--variant_dir`. D-FINE final outputs are isolated under the final export's
`dfine/` directory, and the evaluator refuses to repeat an existing named run.

`FRED_SEED_FAMILY=dfine bash run.sh` selects the D-FINE LLM-GE configuration.
Before submitting a generated D-FINE candidate, the controller checks its
training loss, backward pass, and predictions on small synthetic Event frames.
Evaluation jobs use Slurm's assigned GPU and check that CUDA works before
loading data. A clear GPU or scheduler failure retries the same gene once;
the controller records both attempts separately from candidate-code failures.
Failed candidate scores and reasons are recorded, while those genes are excluded
from parent selection and the Pareto archive. Formal evolution requires a
validated non-evolutionary baseline and the post-core `DG-P1-10` decision.
The current shared data path clips out-of-bounds
annotations and is provisional relative to the unresolved Phase 0 policy; do
not treat proxy scores as an approved FRED benchmark result.
`final_eval.py` supports YOLO11, YOLO26-P2, and D-FINE.
