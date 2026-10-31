# D-FINE motifs v1 selection and feedback

`FRED_DFINE_SEARCH_SPACE=motifs_v1` uses selection version
`map5095_resource_gated_v1`. This is a new search experiment policy; legacy
D-FINE runs retain their previous selection path. It does not change the
standalone seed baseline or the held-out test evaluation.

## Before launching the search

Run the unchanged D-FINE seed with `--dfine-search-space motifs_v1 --device cuda`
on the intended PACE-ICE GPU and frozen search-validation split. The runner
records detector latency and training peak memory in its validation report.
Review repeated seed measurements and choose two **absolute**, fixed ceilings:

```text
FRED_DFINE_MAX_INFERENCE_MS_PER_FRAME=<positive milliseconds per frame>
FRED_DFINE_MAX_PEAK_MEMORY_BYTES=<positive integer bytes>
```

The controller requires both settings before a motifs v1 search begins.
It does not invent limits from a single noisy run. Record the chosen ceilings
and their seed measurements with the run configuration. The thresholds enter
the protocol fingerprint, so changing either requires a fresh checkpoint/run
tag. Keep the same GPU type and software environment for comparisons.

## Measurements and eligibility

Every motifs v1 CUDA candidate records:

- preloaded single-frame detector inference latency: median of nine timed
  predictions after three warmups, with CUDA synchronization;
- allocated GPU memory peak observed during the full training call;
- the earlier full-resolution forward/backward probe peak, batch and shape;
- validation mAP50, mAP50:95, AP75, small-object AP/recall, IoU .50 misses,
  and boxes matched at .50 but missed at .75.

Only the first two resource measurements are eligibility limits. The
controller matches the report's identity, accuracy values and recorded
ceilings to its result CSV and run configuration, requires the measurements and
diagnostics, and excludes candidates above either ceiling. The full-resolution
probe uses the memory ceiling as an early screen when the search worker has
the ceilings. A missing or malformed report is a visible invalid outcome.
The reported peak is PyTorch allocated device memory, not total GPU-reserved
memory or a hardware-wide utilization reading.

## Selection and LLM feedback

Among eligible candidates, parent and elite selection prioritize validation
mAP50:95. Exact ties use mAP50, then fewer parameters. The best mAP50:95
candidate therefore keeps an elite slot. The existing Pareto archive remains
separate and retains accuracy/parameter trade-offs from eligible candidates.
AP for small objects remains diagnostic evidence; it is not a second objective
in this version.

The mutation prompt includes the fixed goal and ceilings, the selected
parent's validation accuracy, diagnostics and resource use, its current
three-gene genome, and the last attempted mechanism from its proposal audit.
Generation-zero proposals from the stock seed have no measured parent fitness
until the seed evaluation finishes, so those prompts state the goal and caps
without fabricating scores. Each full prompt, LLM reply, accepted genome, and
rejected attempt remains in the existing `*.proposal_attempts.json` audit.
No held-out test result enters search feedback.

## Verification status

Offline tests cover report validation, resource rejection, AP-first selection,
prompt feedback, and checkpoint versioning. Live PACE-ICE latency variability
and resource ceilings still need measurement before the next search run.
