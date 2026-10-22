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

## FRED detector Pareto plots

Set `FRED_RUN_TAG` to the tag used by the evolution run, then run
`uv run python analysis/scripts/fred_pareto_graphs.py`. For a legacy run
without a tag, pass `--checkpoints` and `--models_dir` explicitly.

The stock seed is evaluated once through the same candidate runner as the
initial LLM-created candidates and saved in `global_gen_0.pkl` as gene ID
`seed`. Both 2D accuracy-versus-parameter fronts include it. The independent
three-seed baseline summary estimates training noise; it is not another
evolution candidate and is not inserted into these fronts. Failed and
incomplete evaluations are omitted; `candidates.csv` records every valid
evaluated candidate through the last checkpoint. The two displayed 2D
fronts can differ from the three-objective front in `pareto_models.csv`.
