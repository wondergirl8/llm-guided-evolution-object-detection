# Final-only retraining after a stopped search

The user selected `g05_slot4_p01` from the generation-8 archive and explicitly
chose 50 final-training epochs. The source search remains stopped. This is a new
experiment, not a change to its frozen 20-epoch search protocol.

| Setting | Final-only recipe |
|---|---|
| Architecture | `g05_slot4_p01`, 28,957,013 parameters |
| Resolution | Width 1280, aspect ratio preserved |
| Initialization | Fresh random weights independently for each seed |
| Training | All 172 official challenging-training sequences, every fifth frame |
| Images per epoch | 113,340 verified exported images |
| Epochs | 50 complete epochs per seed; no early stopping or interim test scoring |
| Seeds | 1–30 |
| Testing | All 59 official challenging-test sequences, every released frame |
| Optimizer / normalization | Same fixed SGD and BatchNorm policy as search |
| Concurrency | At most two one-GPU training/evaluation workers |
| Allocation limit | Eight hours; checkpoint before shutdown and resume exact batch |

`final_only.configure` verifies that the architecture was successfully scored by
the selected cutoff, matches the source genes, and retains official membership.
The new protocol and authorization freeze the architecture, epoch budget, seeds,
cutoff and source checkpoint. No search model weights are copied or loaded.
The worker checks the resolved architecture and parameter count before training.

The shared `phase1_detection/final_eval.py` entry point supports an explicit
`final_only_after_search_cutoff` authorization. This permits the approved
generation-8 selection without pretending generation 20 completed. Other evolution
protocols retain their generation-20 gate. Final checkpoints, optimizer state,
sample order and RNG states are separately keyed by the new protocol and seed;
existing 20-epoch checkpoints cannot supply final fitness for this run.

Example launch (a fresh tag is required):

```bash
.venv/bin/python -m sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.cli launch \
  --tag faster_rcnn_scratch_YYYYMMDD_eot0_v2_final_g05s4_50ep \
  --final-from /absolute/path/to/stopped-search \
  --final-model g05_slot4_p01 --final-epochs 50 --selection-generation 8 --submit
```

Use the same CLI's `status`, `resume` or `stop` with `--run /absolute/path/to/final-run`.
Automatic CPU successors are pre-armed before worker submission. Reviewed unhealthy
GPU nodes are excluded explicitly in Slurm arguments. No LLM server is used.

Each seed's task directory contains provenance, batch-resume checkpoints, a final
checkpoint, sequence prediction chunks and the official-test report. After all
30 seeds complete, `reports/final_seed_samples.csv` and
`reports/final_seed_statistics.json` summarize the distribution. This initial
final-only run contains one selected architecture; it makes no significance claim
against stock or the paper. A later stock comparison requires a matching final
recipe and its own seed sample.
