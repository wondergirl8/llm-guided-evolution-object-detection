# Faster R-CNN Scratch Analysis

## V2: Resolution-Aware 20-Epoch Experiments

See [the V2 workflow guide](v2/README.md) for mutable widths 640/960/1280,
exact batch-level training resume, automatic eight-hour Slurm continuation,
and the required budget-review stop after the initial eight candidates.
The V2 run is separate from the historical official2 analysis below.

Plots and reports for the official challenging-split evolution are collected
here, using the existing `analysis/scripts/fred_pareto_graphs.py` generator.
This is an analysis-output directory; it does not change the phase-1 training
protocol or move the authoritative experiment checkpoints.

## Official2 Run

Run: `faster_rcnn_scratch_20261022_eot0_official2`.

- [Final parameter Pareto plots](plots/faster_rcnn_scratch_20261022_eot0_official2/pareto_params_final.png):
  mAP50 versus parameter count and mAP50:95 versus parameter count, with separate
  two-objective fronts in the two panels.
- [Parameter-front animation](plots/faster_rcnn_scratch_20261022_eot0_official2/pareto_params_fronts.gif).
- [All-generation accuracy plot](plots/faster_rcnn_scratch_20261022_eot0_official2/pareto2d_all_generations.png).
- [Best validation scores by generation](plots/faster_rcnn_scratch_20261022_eot0_official2/best_by_generation.png).
- [Analysis report](reports/faster_rcnn_scratch_20261022_eot0_official2/summary.md).
- [All candidate scores](reports/faster_rcnn_scratch_20261022_eot0_official2/candidates.csv).
- [Three-objective Pareto candidates](reports/faster_rcnn_scratch_20261022_eot0_official2/pareto_models.csv).
- [Frozen finalist manifest](reports/faster_rcnn_scratch_20261022_eot0_official2/finalists.json).

All scores here are three-epoch **search-validation** scores, not official test
scores. The generator retains candidate IDs, including identical-architecture
aliases. Final training deduplicates architectures according to the frozen
finalist manifest.

`pareto2d_models.csv` and `.md` refer to mAP50 versus mAP50:95, not accuracy
versus parameters. `pareto_models.csv` uses all three objectives. See the
analysis report for the distinction.

## Reproduce

From the repository root:

```bash
.venv/bin/python analysis/scripts/fred_pareto_graphs.py \
  --family faster_rcnn_scratch \
  --checkpoints fred_runs/faster_rcnn_scratch_20261022_eot0_official2/checkpoints \
  --models_dir fred_runs/faster_rcnn_scratch_20261022_eot0_official2/candidates/models \
  --seed_file sota/FRED_LLM_GE/seeds/seed_faster_rcnn_scratch.py \
  --title 'FRED Faster R-CNN Scratch - Official Challenging Search Validation' \
  --output sota/FRED_LLM_GE/phase2/faster_rcnn_scratch/plots/faster_rcnn_scratch_20261022_eot0_official2
```

The generator emits CSV/Markdown tables beside its images. For this collected
output, those tables are relocated to the matching `reports/<run_tag>/` folder.
The frozen finalist manifest is a copy of the authoritative run manifest.
