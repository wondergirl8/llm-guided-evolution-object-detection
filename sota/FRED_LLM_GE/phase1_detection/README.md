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
training budget, and evaluator with the YOLO11 adapter. Its end-to-end head
filters by confidence without NMS.

The three full-size families read the same challenging-search export, but
their default writable paths are separate:

| Family | Candidates, results, and trained checkpoints | Evolution state |
| --- | --- | --- |
| `yolo11` | `phase1_detection/seeds/ultralytics_yolo11/` | `fred_yolo11_output/`, `fred_yolo11_checkpoints/` |
| `yolo26_p2` | `phase1_detection/seeds/yolo26_p2/` | `fred_yolo26_p2_output/`, `fred_yolo26_p2_checkpoints/` |
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

`FRED_SEED_FAMILY=dfine bash run.sh` selects the D-FINE LLM-GE configuration,
but formal evolution requires a validated non-evolutionary baseline and the
post-core `DG-P1-10` decision. The current shared data path clips out-of-bounds
annotations and is provisional relative to the unresolved Phase 0 policy; do
not treat proxy scores as an approved FRED benchmark result.
`final_eval.py` supports the YOLO11 and YOLO26-P2 Ultralytics families.
