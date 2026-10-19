# Analysis

Centralized location for analysis scripts and dashboard-related work for llm-guided-evolution.

## Purpose
This folder is for post-run analysis only. Scripts here should read experiment outputs, logs, and result files without modifying raw artifacts.

## Structure
- `scripts/`: runnable analysis scripts
- `utils/`: shared helper functions
- `dashboard/`: future dashboard-related code
- outputs should be written to `results/analysis/`

## Conventions
- do not modify raw run outputs
- scripts should run end-to-end
- prefer reusable helper functions over duplicated logic
- use descriptive snake_case filenames


## FRED 2D Pareto views

`scripts/fred_pareto_graphs_2d.py` adds two views alongside the existing
3D script (`scripts/fred_pareto_graphs.py`), which remains unchanged:

- **mAP50 vs. parameters**
- **mAP50:95 vs. parameters**

Parameters are on the horizontal axis in millions (lower is better); accuracy
is on the vertical axis in percent (higher is better). Gray points show valid
completed models, red points show the original three-objective Pareto members,
and the red step line shows the best accuracy achievable under each parameter
budget in that particular view. A three-objective member need not lie on the
step line. The baseline seed is marked with a star and participates in both
front calculations. Failed or malformed fitness is excluded. This changes
visualization only, not the controller's fitness or selection policy.

Run from the experiment checkout after checkpoints and baseline results exist.
Load only trusted run pickles. For the current YOLO26n-P2 namespace:

```bash
uv run python analysis/scripts/fred_pareto_graphs_2d.py \
  --family YOLO26n-P2 \
  --checkpoints fred_yolo26_p2_map5095_v1_checkpoints \
  --seed_results sota/FRED_LLM_GE/phase1_detection/seeds/yolo26_p2/results/seed_results.csv \
  --models_dir sota/FRED_LLM_GE/phase1_detection/seeds/yolo26_p2/models \
  --seed_file sota/FRED_LLM_GE/seeds/seed_yolo26n_p2.yaml \
  --output analysis/results/analysis/yolo26_p2_pareto_2d
```

For D-FINE:

```bash
uv run python analysis/scripts/fred_pareto_graphs_2d.py \
  --family D-FINE \
  --checkpoints fred_dfine_checkpoints \
  --seed_results sota/FRED_LLM_GE/phase1_detection/seeds/dfine/results/seed_results.csv \
  --models_dir sota/FRED_LLM_GE/phase1_detection/seeds/dfine/models \
  --seed_file sota/FRED_LLM_GE/seeds/seed_dfine.py \
  --output analysis/results/analysis/dfine_pareto_2d
```

The 2D script defaults to its own `analysis/results/analysis/fred_pareto_2d`
output folder. Use a separate output folder from the 3D script so their CSVs
and best-score history stay separate.

Each 2D run writes `pareto2d_mAP50_final.png` and
`pareto2d_mAP50_95_final.png`, per-generation PNGs, one GIF for each metric
(unless `--no_gif` is supplied), the best-score history, and CSVs with the
joint-front and per-view flags. Inputs are read only. Use one consistent shared
evaluation protocol within each figure; trainer epoch metrics are not substitutes
for the checkpoint fitness. Keep different family runs in separate output folders.

The script reads saved `global_gen_*.pkl` files, including their historical
evaluations. Legacy records without a status are accepted when their fitness is
valid; explicit failed or unfinished statuses are excluded. Plotting does not
independently verify evaluation reports or checkpoint provenance. A saved
generation containing only the baseline is labeled **baseline only**, with no
frontier line. If no saved global generation exists, the script stops with a
clear error rather than inventing an evolutionary front.
