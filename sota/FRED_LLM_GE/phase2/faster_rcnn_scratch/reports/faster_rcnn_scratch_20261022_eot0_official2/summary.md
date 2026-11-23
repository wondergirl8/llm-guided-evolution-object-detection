# Official Challenging Search: Faster R-CNN Scratch

## Provenance

- Run: `faster_rcnn_scratch_20261022_eot0_official2`.
- Source: `fred_runs/faster_rcnn_scratch_20261022_eot0_official2/checkpoints/global_gen_0.pkl`
  through `global_gen_20.pkl`, including archived candidate history.
- Generator: `analysis/scripts/fred_pareto_graphs.py`, unmodified.
- Plot directory: `../../plots/faster_rcnn_scratch_20261022_eot0_official2/`.
- Scores: search validation only, three epochs per independently initialized
  uncached architecture, seed zero.
- Search data: 152 official-training sequences sampled every 50th frame;
  validation: fixed 20 official-training sequences sampled every 10th frame.
- No official challenging-test results are included.

## Frozen Finalists

| Representative candidate | Validation mAP50 | Validation mAP50:95 | Parameters | Role |
|---|---:|---:|---:|---|
| `xXx4LVf0TsysH0uUoegVF3Yu1cg` | 77.89% | 33.22% | 32,365,515 | Three-objective front |
| `xXxDm4Rc7PnF4MlHAosTSnD0oVz` | 77.25% | 34.85% | 37,997,515 | Three-objective front |
| `xXxa3DQeZXW6ldg74KcrAfT1Ued` | 77.77% | 34.45% | 41,197,515 | Three-objective front |
| `seed` | 68.36% | 26.92% | 41,348,181 | Stock control |

The copied `finalists.json` retains architecture hashes, aliases, source hashes,
validation fitness, ancestry, and protocol identity. Multiple gene IDs for the
same architecture share one final-training run.

## Reading The Plots

`pareto_params_final.png` contains the two requested diagrams:

1. Validation mAP50 versus parameter count: the 32,365,515-parameter finalist
   dominates the other architectures in this projection.
2. Validation mAP50:95 versus parameter count: the 32,365,515-parameter and
   37,997,515-parameter finalists form the nondominated tradeoff.

The 41,197,515-parameter finalist remains nondominated when all three objectives
are used jointly, despite not lying on either parameter/accuracy 2D front.
The plot generator retains equivalent candidate-ID ties, so coincident points
and repeated front rows do not imply distinct architectures.

The generator also produces `pareto2d_all_generations.png` and
`pareto2d_gen_<n>.png`, which plot mAP50 against mAP50:95, not parameter count.
Corresponding `pareto2d_models.csv`/`.md` omit parameter count from the dominance
rule. Use `pareto_models.csv` for the three-objective candidate front and
`finalists.json` for the architecture-deduplicated frozen final-training set.

## Training Versus Evaluation

The displayed AP values belong to the short-budget architecture search. Final
training starts from fresh weights for 20 epochs on all 172 official-training
sequences sampled every fifth frame. Only after a finalist finishes that
training does its final checkpoint undergo official-test evaluation across all
59 official-test sequences. Search completion is not final-test completion.
