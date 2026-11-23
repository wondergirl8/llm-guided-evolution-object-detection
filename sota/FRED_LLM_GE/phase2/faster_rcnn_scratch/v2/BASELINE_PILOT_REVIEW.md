# Baseline pilot and seed-budget review

Decision recorded November 10, 2026: finish the stock baseline pilot before
generating architectures or starting evolution. Train widths 640, 960, and 1280
with seeds 0, 1, and 2, preserving the frozen 20-epoch training protocol.
Prioritize seed 0 at each width before additional seeds. At most two workers run.

The separately versioned `pilot_supervisor.py` recovery controller uses original
frozen worker code and existing checkpoint identities. Its exact source hash is
recorded in `run/control/pilot_recovery.json`. It handles scheduler communication
errors without treating them as model failures or cancelling workers. The original
run manifest, protocol, and frozen code must remain unchanged.

The pilot stops at `awaiting_seed_budget_review`, before LLM proposals or evolution.
The controller writes `reports/baseline_seed_budget.json` with per-width seed
scores, means, sample SDs, and illustrative seed counts for gains of 1, 2, 3, and
5 percentage points. Planning uses 80% power, a two-sided 5% significance threshold,
equal group sizes, and equal population variance; it is not an exact Welch power
calculation. Estimates concern one independent comparison on fixed validation data.
Zero estimated variance is not interpreted as certainty or assigned a sample size.

After reviewing the pilot, explicitly record the primary metric, minimum relevant
gain, power, significance level, planned comparisons/multiplicity handling, seeds
per candidate, and fresh confirmation seeds. Three-seed variance estimates are
uncertain; more pilot repeats may be necessary. No final seed budget or variance
penalty has yet been adopted. Candidate scheduling and fitness aggregation must be
updated and tested before evolution is authorized. Official test data do not enter
this decision.

To stop the recovery controller, create `run/STOP` (the existing V2 stop command
does this too). Checkpointed jobs and pre-armed continuation are cancelled. Do not
delete the stop sentinel or manually re-enable evolution without review.
