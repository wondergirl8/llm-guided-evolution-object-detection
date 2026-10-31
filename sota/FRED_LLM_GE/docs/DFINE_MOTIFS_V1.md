# D-FINE bounded motif search prototype (v1)

This is an opt-in architecture search path for the project-owned D-FINE-inspired
seed. It implements the structured-search portion of the D-FINE improvement
roadmap. It does not change the stock seed, criterion, evaluator, or fitness
selection. It is a **prototype genome**, not the frozen formal search space:
the Phase 1 resource envelope, training repeatability, and `DG-P1-07`/`DG-P1-10`
decisions still govern formal use.

## Enable it for a fresh experiment

Set `FRED_SEED_FAMILY=dfine` and `FRED_DFINE_SEARCH_SPACE=motifs_v1` before
starting the controller, and use a fresh `FRED_RUN_TAG`. The controller records
the version in gene metadata, puts untagged outputs in separate `dfine_motifs_v1`
paths, and refuses to resume a checkpoint containing genes from another search
space. Existing D-FINE runs continue to use the legacy class-proposal path.

The worker receives `--dfine_search_space motifs_v1` explicitly in its command.
The evaluator receives `--dfine-search-space motifs_v1` for both seed and
candidates and refuses a missing or mismatched candidate sidecar.
The new prompt group is `templates/FRED_DFINE/MotifsV1`. Mutation asks the LLM
for exactly one JSON `value` from the listed choices. The builder in
`src/fred_dfine_genome.py` renders Python deterministically from the unchanged
seed and the complete three-gene JSON sidecar. Direct LLM-written Python is
not accepted in this mode. Crossover recombines a whole differing gene and
rejects a combination that is not novel. The prompt, response, rejection
reason, selected gene, genome, and architecture hash remain in the proposal
audit. The candidate report records the genome and its resource probe.

## Prototype genes

| Gene | Allowed values | Mechanism |
| --- | --- | --- |
| `backbone` | `none`; P4 depthwise 3×3 or 5×5 refinement with residual scale 0.25, 0.5, or 1.0 | Test whether local detail at stride 4 helps tiny drones without changing map size or channel count. |
| `encoder` | `none`; top-down weight 0.5 or 1.0 paired with bottom-up adaptive-pooling weight 0.25, 0.5, or 1.0 | Test whether bidirectional cross-scale exchange helps P4 features carry useful context. |
| `decoder` | `none`; reserve at least 25%, 50%, or 75% of query slots for top-ranked P4 positions | Test query access to high-resolution features while retaining the same query and 256-token memory limits. |

The 7 × 7 × 4 combinations yield 196 distinct Python ASTs, including the seed.
One gene changes per mutation. Generation zero requires one audited candidate
for each non-`none` mechanism, plus the unchanged seed. The remaining initial
slots draw from the same bounded catalogue. Candidate source and genome
sidecar must agree byte for byte with the deterministic renderer; legacy
candidates cannot be crossed into this version.

## Validation and resource envelope

Before GPU submission, the existing CPU preflight checks construction,
odd-sized feature shapes, loss/backward, and prediction. For this search space,
the worker also performs a full-resolution forward/backward on the allocated
GPU at the configured D-FINE microbatch size before training. The report
records probe shape, batch size, parameter count, elapsed time, and peak
allocated GPU bytes. The probe does not update weights or BatchNorm buffers.
It is a feasibility screen; its single cold step is not a stable inference
latency benchmark and it does not include optimizer-state memory. The
[selection policy](DFINE_SELECTION_V1.md) measures full-training peak memory
and warmed, preloaded inference separately.

Optional absolute caps can be set under `families.dfine.search_resource_limits`
in the protocol: `max_params`, `max_peak_bytes`, and `max_step_ms`. A violation
fails the candidate. No formal numerical cap is supplied here because the
approved PACE-ICE envelope must be derived from measured baseline resources.
Set and freeze those values, then use a new experiment tag before a formal run.

## Limits and next decision

The catalogue is deliberately small so the first structured study can identify
which of P4 detail, scale fusion, or P4 query access is worth expanding. It
does not yet implement other query-diversity or localization mechanisms, and
its `map5095_resource_gated_v1` selection policy keeps the three-metric fitness
record and Pareto archive but selects parents and elites by mAP50:95 after
resource gating. Compare a repeatable seed and
paired candidates before interpreting any higher GE score as an architecture
gain; keep challenging-test results outside all prompts and selection.
