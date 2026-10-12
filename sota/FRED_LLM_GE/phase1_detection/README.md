# Phase 1 detection runner

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
  --family dfine --model network \
  --sota_root sota/FRED_LLM_GE/phase1_detection/seeds/dfine \
  --variant_dir sota/FRED_LLM_GE/phase1_detection/seeds/dfine/models
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
not treat proxy scores as an approved FRED benchmark result. The existing
`final_eval.py` currently handles Ultralytics YOLO11 only.
