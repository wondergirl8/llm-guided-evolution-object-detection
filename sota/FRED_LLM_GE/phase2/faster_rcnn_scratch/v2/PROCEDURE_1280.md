# Fixed-1280 Faster R-CNN experiment

Authorized November 15, 2026, by the request to run evolution only at width 1280
for 20 generations and assess Pareto architectures across seeds using the supplied
Experimentation Guidelines. The original pasted procedure is preserved verbatim
as `experiment_procedure.txt` and hashed in each new run's authorization record.

The previous resolution pilot remains a separate experiment and supplies no fitness
or checkpoints to this protocol. It used frame step 5, whereas this procedure
explicitly uses frame step 50 during search.

## Frozen recipe

| Stage | Official membership | Sampling | Epochs | Seeds |
|---|---|---|---:|---|
| Search | 152 training sequences | Every 50th frame | 20 | 0 |
| Search validation | Fixed 20 held-out official-training sequences | Every 10th frame | — | — |
| Final retraining | All 172 official-training sequences | Every 5th frame | 20 | 1–30 |
| Final testing | All 59 official-test sequences | Every frame | — | Matching final checkpoints |

Every model uses width 1280 with preserved aspect ratio and original-coordinate
evaluation. Resolution cannot be mutated; the original eight architecture fields
remain mutable. ResNet-50 shape, random initialization, unfrozen stages, norm_eval
with trainable affine parameters, batch 2 and SGD recipe are unchanged.

Generation 0 has stock plus seven globally distinct LLM variants. Generations
1–20 retain two elites and train six globally novel successful offspring, giving
128 unique successful search configurations. Failed, pending, repeated and no-op
proposals cannot fill a new slot. After 24 attempts per slot the run blocks for
review instead of changing search method or population size.

The three Pareto objectives maximize validation mAP50 and mAP50:95 and minimize
parameters. Source, protocol and model identities are frozen from the entire
successful archive at generation 20. Add a stock control if absent from the front.
Each frozen architecture/control is freshly trained and tested 30 times, with no
interim test evaluation or test-based reselection. This is one evolutionary search;
30 retraining seeds do not measure repeatability between independent searches and
do not eliminate training-seed noise during single-seed search.

## Statistical decisions

The supplied procedure fixes 30 final repeats, replacing the pilot's unresolved
seed-count decision. The already agreed primary metric remains mAP50:95; practical
target 2 percentage points; nominal alpha .05; planned power .80. Thirty repeats
do not guarantee that power for unknown finalist variances.

Use two-sided Welch comparisons of evolved minus stock means, with Holm correction
across all frozen evolved finalists. Report all seed scores, mean, sample SD,
descriptive mean/difference 95% intervals, t, degrees of freedom, raw and adjusted
p-values, and whether the estimated gain reaches 2 points. Positive significant
differences and 2-point practical gains are distinct statements. No repeated
significance testing is performed to decide when to stop collecting seeds.

The procedure's one-sided example tests baseline superiority, so it is not copied
as the hypothesis direction for higher-is-better AP. Its statement that a
nonsignificant result likely occurred by chance is not a valid statistical
conclusion. Nonsignificance is not evidence of equivalence. Welch assumes
independent observations; using the same seed labels across architectures may
induce dependence, which is recorded as an analysis limitation. Variation across
training seeds on a fixed test set does not estimate dataset sampling uncertainty
or a distribution for the paper's single reported score. Historical test exposure
is explicitly recorded in the protocol.

## Scheduling and cost

Launch with `python -m sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.cli launch
--tag faster_rcnn_scratch_YYYYMMDD_eot0_v2_1280procedure --procedure-1280 --submit`.
The same CLI provides status/resume/stop. Two training workers maximum; every
allocation is at most eight hours, with a CPU afterany successor pre-armed before
worker submissions. Resume uses exact batch checkpoints; scheduler query errors
are recorded and retried rather than interpreted as model failures. Explicit
partitions avoid the previous BadConstraints failure.

Final cost is `30 × (number of evolved finalists + 1 stock control)` independent
20-epoch full-training evaluations. The pilot measured roughly 35 GPU-hours per
1280 search model on 152 sequences at step 5. Final training on 172 sequences plus
full testing will cost more; this is a rough planning reference, not measured
throughput for the step-50 search or final testing. No accuracy or wall-time claim
is inferred from that estimate. LLM proposals run in separate GPU allocations only
during proposal batches.

Outputs: `checkpoints/global_gen_*.pkl`, `reports/pareto_gen_*.json`, existing Pareto
plots, `finalists.json`, per-task checkpoint identities/predictions/reports,
`reports/final_seed_samples.csv`, `reports/final_seed_statistics.json`, and
`reports/final_comparison.md`. Old pilot artifacts are never modified.

## Shared final-evaluation entry point

Faster R-CNN final tasks can be launched using the shared script:

```bash
.venv-faster-rcnn/bin/python sota/FRED_LLM_GE/phase1_detection/final_eval.py \
  --family faster_rcnn_scratch --run fred_runs/<run-tag> --task final_000_s01
```

This requires an approved generation-20 frozen finalist and its scheduled final
task. It rejects search models, unapproved testing, changed protocol/source, and
seeds outside the frozen recipe. It dispatches to the experiment's original
MMDetection worker, preserving checkpoint identities, batch-level resume, and
per-sequence prediction recovery. YOLO11 and RT-DETR retain their existing CLI.

Already-frozen runs use an explicitly recorded controller routing adapter rather
than editing their source snapshots or checkpoint fingerprints. The adapter and
shared `final_eval.py` are copied under `run/control/` and hash-verified using
`final_eval_entrypoint.json`. Search submissions use the original frozen code;
final submissions enter `control/final_eval.py`. CPU successors retain this routing
across eight-hour allocations. An adapter restart uses
`run/code/.venv/bin/python run/control/final_eval_supervisor.py --run <run>` in a
CPU allocation after reconciling the previous supervisor; never run a second
supervisor concurrently. This routing does not authorize interim official tests.
