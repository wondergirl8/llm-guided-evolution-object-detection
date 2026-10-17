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
  --output-dir sota/FRED_LLM_GE/phase1_detection/seeds/yolo26_p2/head_rescore
```

The command reads each checkpoint's sibling `results/<gene_id>_evaluation.json`
and checks the checkpoint, architecture, dataset revision, split, and frame
count before inference. It runs both heads at confidence 0.001 and 0.05; the
one-to-many head also uses NMS IoU 0.5 and 0.7. `--gene-id seed` limits a run,
and `--checkpoint-dir`, `--reference-results-dir`, and `--data-root` can point
to a restored handoff. Output goes only to the separate `--output-dir`, with
one validation report per setting; rerunning skips matching completed reports.
`head_rescore_summary.csv` collects the comparisons after all settings finish.
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
| `yolo26_p2` | `phase1_detection/seeds/yolo26_p2/` | `fred_yolo26_p2_map5095_v1_output/`, `fred_yolo26_p2_map5095_v1_checkpoints/` |
| `dfine` | `phase1_detection/seeds/dfine/` | `fred_dfine_output/`, `fred_dfine_checkpoints/` |

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
have a matching `map5095_v1` suffix; older checkpoints cannot be resumed under
the new selection policy. This is a pilot search setting, not the formal
`DG-P1-06` fitness-contract freeze.

## YOLO26 controlled ablations

The separate [ablation plan](../configs/phase1/yolo26_ablations.yaml) compares
the seed with a longer budget, optimizer, learning rate, and Event-image
augmentation changes. It also halves P4 or P5 output width individually while
leaving the seed's P2/P3 paths intact. The proposed compact-model tests would
restore only the P2 or P3 detection-output layer relative to that compact model,
but that candidate source is absent from this workspace. They are omitted
from the runnable plan until its exact source is restored. Every listed
comparison changes one factor, and each experiment should be repeated with
seeds 0, 1, and 2 before drawing a training-variance conclusion.

First validate every proposed architecture locally without FRED data:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/yolo26_ablation.py --check
```

After restoring the same challenging-search export used by the historical run,
run one experiment/seed at a time, for example:

```bash
uv run python sota/FRED_LLM_GE/phase1_detection/yolo26_ablation.py \
  --experiment seed_p4_half --seed 0 \
  --data-root sota/FRED_LLM_GE/phase1_detection/seeds/yolo11/data_cache/challenging
```

Each run has its own directory under `seeds/yolo26_p2/ablations/`, with a model
YAML, checkpoint, diagnostic evaluation JSON, Ultralytics `args.yaml` (resolved
trainer arguments), and `results.csv` (training curves). The report records the
actual optimizer class and file hashes. Existing run directories
are refused to prevent accidental overwrites. This is a pilot search-validation
study, separate from the LLM-GE candidate rankings and the held-out test.

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
training loss and learning rate for every epoch, search-validation mAP50 and
mAP50:95 at epoch 1, every fifth epoch, and the final epoch, plus the usual
final diagnostics. Compare arms for the **same model and seed** before judging
the effect; inspect the curve for continued improvement or instability before
choosing a training recipe. The original search export and model files are
required for real measurements. Do not use the held-out test for this choice.

Step 6 has one opt-in localization experiment for the D-FINE seed. Run it only
if the Step 4 diagnostics show that higher-IoU box accuracy is the main
weakness. It compares the protocol's 16 refinement bins with 32, keeping the
same model source, search split, training budget, loss weights, and seed:

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
