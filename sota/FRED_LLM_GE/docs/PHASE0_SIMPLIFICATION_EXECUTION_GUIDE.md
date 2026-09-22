# Phase 0 Simplification Execution Guide

**Status:** Draft implementation guide  
**Scope:** Simplification of Phase 0 remote access, cache lifecycle, and runtime loading  
**Execution model:** Nine gated tasks completed one at a time

## 1. Purpose

This guide defines how to simplify the current Phase 0 data implementation without
weakening source integrity, reproducibility, cache safety, split protection, or the
canonical FRED sample contract.

The intended target is a preparation-first runtime model:

```text
pinned remote sequence archive
        ↓
atomic sequence preparation and validation
        ↓
immutable prepared sequence + metadata + completion marker
        ↓
read-only training/evaluation access
        ↓
bounded cleanup or eviction outside active reads
```

This is an execution guide, not a new source of project policy. It does not override:

- `docs/MASTER_PROJECT_PLAN.md`;
- `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md`;
- `docs/PHASE0_DATA_PLAN.md`;
- `docs/AGENTS.md`;
- an explicitly approved Phase 0 decision-gate record.

If this guide conflicts with an authoritative requirement, stop and resolve the
conflict before editing code.

## 2. Non-goals

This work must not:

- change the pinned FRED dataset or repository revisions;
- change official or project split membership;
- reinterpret annotations, timestamps, coordinates, or frame pairing;
- expose held-out annotations to candidate or model-development code;
- introduce an implicit full-dataset mirror;
- make cache paths part of canonical sample identity;
- combine this simplification with Phase 1 model implementation;
- claim that the Phase 0 freeze gate is complete;
- delete the current runtime path before its replacement is verified.

## 3. Invariants that every task must preserve

Every task in this guide must preserve the following observable behavior:

1. Remote reads use the configured full dataset revision and stable logical reference.
2. Missing or unverifiable remote content fails visibly.
3. Partial downloads and partial extractions are never accepted as completed entries.
4. Unsafe ZIP paths, duplicate members, and symbolic links remain rejected.
5. Runtime extraction remains limited to required released RGB frames, released event
   frames, canonical coordinates, and explicitly versioned project metadata.
6. Cache storage remains bounded, disposable, and rebuildable.
7. Warm runtime access does not perform a remote request or reopen the source ZIP for
   every sample.
8. Stable sample IDs, timestamps, annotations, splits, and provenance remain unchanged.
9. Evaluation-input mode does not query or return annotations.
10. Failures are reported as failures and do not publish apparently valid partial
    manifests or reports.

## 4. Rules for executing the tasks

- Execute only one numbered task per implementation batch.
- Begin every batch by reviewing the current working tree and the files named by that
  task. Preserve unrelated user changes.
- Record the applicable requirements and the exact files expected to change before
  implementation.
- Run the task's focused checks first, followed by the complete Phase 0 regression
  suite when the focused checks pass.
- Do not proceed to the next task while a required check is failing or skipped.
- Do not weaken or delete a safety mechanism until its replacement is implemented and
  proven in the same batch or an earlier completed batch.
- Do not perform a full remote download, full-dataset audit, destructive cache cleanup,
  or network-dependent benchmark without explicit authorization.
- Keep the public `FREDDataset` and `HFFredSource` behavior compatible until Task 9.
- Record decisions and evidence in durable files when required by a Phase 0 gate; do
  not rely only on terminal output.

Each task must end with a short report containing:

- files changed;
- behavior intentionally preserved;
- tests and checks run;
- evidence produced;
- unresolved limitations;
- whether the task's exit gate passed.

---

# Task 1 — Establish a recoverable known-good baseline

**Task ID:** `P0-S01`  
**Purpose:** Ensure that simplification work can be compared with and recovered to the
current working implementation.

## Preconditions

- Read the governing documents listed in Section 1.
- Inspect the complete Git working-tree status.
- Identify which changes predate this simplification effort.

## Actions

1. Record the current Phase 0 implementation files, configuration, dependency lock,
   and test files.
2. Run and record the current complete Phase 0 test count and results.
3. Run lint, Python compilation, YAML parsing, dependency-lock validation, and diff
   whitespace checks.
4. Record the current cache extraction version and public exports.
5. Record hashes or content identities for any existing representative manifest and
   validation artifacts that will be used for equivalence testing.
6. Create a recoverable Git checkpoint only when authorized. If a commit is not
   authorized, create a non-destructive baseline record without resetting, stashing,
   or rewriting existing user work.

## Required evidence

- Clean results for all available baseline checks, or an explicit list of pre-existing
  failures.
- A record of unrelated working-tree changes that must not be modified.
- A known representative development sequence and local artifact set for later
  comparisons.

## Exit gate

Do not begin Task 2 unless the baseline is reproducible and there is a safe recovery
path for every file that later tasks may change.

---

# Task 2 — Decide the runtime and cache operating model from evidence

**Task ID:** `P0-S02`  
**Purpose:** Determine whether cache mutation can be prohibited during active training
and evaluation runs.

## Preconditions

- Task 1 exit gate passed.
- No cache-policy code changes are made in this task.

## Actions

1. Measure or derive the archive and extracted sizes of the sequences required by a
   representative Phase 1 workload.
2. Record available cache capacity and usable disk capacity in the intended execution
   environment.
3. Establish the expected DataLoader worker count, process-start method, prefetch
   behavior, sequence ordering, and whether persistent workers will be used.
4. Determine whether the required workload can remain prepared and pinned for an
   entire run without exceeding the bounded cache.
5. Evaluate two permitted designs:

   - **Model A — run-scoped read-only cache:** prepare the complete workload before a
     run, prohibit cache mutation during the run, and clean up only between runs.
   - **Model B — coordinated active window:** one coordinator prepares and evicts a
     bounded sequence window while workers receive read-only access to explicitly
     pinned entries.

6. Select one model through `DG-P0-06` evidence. Do not silently select Model A merely
   because it is simpler.

## Required evidence

- Workload size and cache-capacity comparison.
- Worker and prefetch assumptions.
- Expected cold-start and repeated-run access pattern.
- A short decision record selecting Model A or Model B with rationale.

## Exit gate

Do not begin cache lifecycle changes until the selected model is supported by measured
or inventory-derived evidence.

---

# Task 3 — Freeze the observable safety contract in tests

**Task ID:** `P0-S03`  
**Purpose:** Make unsafe simplification detectable before implementation behavior is
changed.

## Preconditions

- Tasks 1 and 2 passed.
- The existing runtime implementation remains active.

## Actions

Add or strengthen contract-level tests for:

1. exact pinned dataset and object identity;
2. size/hash verification where metadata is available;
3. interrupted download and extraction cleanup;
4. atomic publication of complete cache entries;
5. ZIP traversal, duplicate member, and symlink rejection;
6. stale or incompatible cache-version rejection;
7. warm access with no remote call and no source-ZIP reopening per sample;
8. cache capacity enforcement;
9. concurrent or multi-worker access under the model selected in Task 2;
10. deterministic samples, manifests, validation findings, and provenance;
11. RGB-only, event-only, and paired modality behavior;
12. evaluation-input label isolation;
13. missing prepared content producing an actionable failure rather than a fallback;
14. old and new process boundaries reopening read-only resources safely.

Tests must assert observable outcomes rather than private implementation structure
unless the structure itself is a security boundary.

## Required verification

- New focused safety-contract tests pass against the current implementation.
- The complete Phase 0 suite still passes.
- No production behavior changes in this task.

## Exit gate

Do not begin removal work until the tests detect the safety failures that the removed
mechanisms previously prevented.

---

# Task 4 — Remove redundant internals without changing behavior

**Task ID:** `P0-S04`  
**Purpose:** Reduce accidental complexity before changing cache policy.

## Preconditions

- Task 3 contract tests pass.
- A before-change representative manifest/content identity is available.

## Candidate actions

Only remove candidates confirmed unused or redundant in the current repository:

1. remove unused private helpers such as an unreferenced metadata-path method;
2. remove redundant archive fields from `SequenceMaterialization` when the inventory
   already provides the stable archive logical reference;
3. remove duplicate in-memory materialization state if persisted metadata already
   provides the same information and performance remains acceptable;
4. replace explicit manual context-manager `__enter__`/`__exit__` handling with normal,
   exception-safe scoped helpers;
5. consolidate duplicated sequence validation and materialization checks;
6. keep one clearly documented runtime sequence-open interface for callers while
   preserving compatibility until Task 9.

## Prohibited actions in this task

- Do not remove capacity enforcement.
- Do not remove writer locks or reader protection.
- Do not change eviction timing.
- Do not change cache identity or extraction contents.
- Do not change manifests, sample records, or label-access behavior.

## Required verification

- Focused cache/source/dataset/CLI tests pass.
- Complete Phase 0 suite passes.
- Representative manifest records and content identity remain unchanged.
- Warm access still performs no network request or ZIP reopening per sample.

## Exit gate

The task passes only if it is a behavior-preserving simplification.

---

# Task 5 — Introduce explicit preparation, opening, and cleanup phases

**Task ID:** `P0-S05`  
**Purpose:** Separate mutable preparation work from read-only runtime access.

## Preconditions

- Tasks 1–4 passed.
- Task 2 selected and documented the cache operating model.

## Target responsibilities

### Prepare sequence

`prepare_sequence(sequence_id)` must:

- resolve the pinned remote object;
- download only when a compatible local object is unavailable;
- validate object identity and archive safety;
- extract required members to a temporary location;
- validate the prepared sequence;
- write versioned immutable sequence metadata;
- write a completion marker only after all prior work succeeds;
- atomically promote the completed sequence;
- remain idempotent when called repeatedly.

### Open prepared sequence

`open_prepared_sequence(sequence_id)` must:

- perform no network access;
- perform no extraction;
- perform no eviction;
- reject missing, partial, wrong-revision, or wrong-version entries;
- return a read-only logical sequence view.

### Cleanup cache

`cleanup_cache(...)` must:

- execute only under the policy selected in Task 2;
- never delete an entry active under that policy;
- remove whole entries only;
- remain bounded and deterministic enough to diagnose;
- never treat cleanup as source-data deletion.

## Prepared-entry minimum contents

```text
<entry>/
├── sequence/                  # released runtime files
├── sequence_metadata.json    # pinned identity and validated layout
└── _SUCCESS                  # written last
```

The exact names may differ, but identity metadata and the completion marker must be
versioned and unambiguous.

## Migration requirement

Use a new extraction/cache format version. Existing entries must either be migrated
through a validated atomic process or ignored and rebuilt on demand. They must not be
silently accepted under the new format.

## Exit gate

The preparation and opening paths must pass Task 3 tests before any existing runtime
path is deleted or disabled.

---

# Task 6 — Make dataset sample loading strictly read-only

**Task ID:** `P0-S06`  
**Purpose:** Ensure that hot sample delivery contains no remote or cache-management
side effects.

## Preconditions

- Task 5 preparation and opening paths pass their focused tests.

## Actions

1. Change the runtime dataset to consume only `open_prepared_sequence` or its approved
   equivalent.
2. Ensure `FREDDataset.__getitem__` performs only:

   - manifest lookup;
   - prepared sequence lookup;
   - requested modality decoding;
   - permitted annotation lookup;
   - construction of the returned sample.

3. Prohibit `__getitem__` from downloading, extracting, evicting, updating LRU
   metadata, or modifying prepared entries.
4. Fail with an actionable preparation-required error if a sequence is unavailable.
5. Preserve one read-only SQLite connection per process/worker or an equivalently
   verified manifest access strategy.
6. Preserve the evaluation-input rule that annotation storage is not queried.

## Required verification

- Instrumented tests prove zero network, extraction, cache deletion, and metadata
  writes during repeated `__getitem__` calls.
- Multi-worker reads comply with the operating model selected in Task 2.
- Modality-specific decoding remains selective.
- Complete regression suite passes.

## Exit gate

Do not remove the existing read-protection mechanism until all runtime consumers use
the read-only path.

---

# Task 7 — Move eviction outside active reads

**Task ID:** `P0-S07`  
**Purpose:** Remove per-sample lease complexity only after the new lifecycle makes it
unnecessary.

## Preconditions

- All runtime consumers are read-only under Task 6.
- The Task 2 operating model is implemented.

## Model A actions

If run-scoped read-only caching was selected:

1. prepare and verify the complete declared workload before a run;
2. acquire one run-level cache state or pin record;
3. prohibit preparation and eviction while the run is active;
4. allow cleanup only after the run closes;
5. remove per-sample lease creation, stale-reader PID files, and related retry logic.

## Model B actions

If a coordinated active window was selected:

1. centralize preparation and eviction in one coordinator;
2. provide workers immutable handles only to pinned window entries;
3. require explicit worker release/epoch transition before eviction;
4. retain only the minimum coordinator-level protection needed to prevent active-entry
   deletion;
5. remove independent worker eviction and per-sample capacity decisions.

## Required verification

- Active files cannot disappear during concurrent reads.
- Cache usage remains within its configured bound.
- Interrupted runs leave recoverable state.
- Stale run state is detected and recoverable without deleting source truth.
- Warm repeated sample access creates no per-sample lease file or metadata write.

## Exit gate

Reader leases may be removed only after tests prove that eviction cannot run against
active readers under the selected replacement model.

---

# Task 8 — Prove old/new semantic and performance equivalence

**Task ID:** `P0-S08`  
**Purpose:** Demonstrate that simplification changed lifecycle mechanics, not FRED
semantics.

## Preconditions

- Tasks 5–7 pass all focused and regression tests.
- The old implementation remains recoverable for comparison.

## Actions

1. Run the old and simplified implementations against the same pinned representative
   development sequence set.
2. Compare exactly:

   - sequence membership;
   - sample count and ordering;
   - sample IDs;
   - frame indexes and timestamps;
   - RGB and event logical/archive references;
   - image dimensions;
   - annotations and track metadata;
   - official and project splits;
   - provenance;
   - validation findings;
   - canonical manifest record hashes.

3. Run cold-cache and warm-cache measurements on the intended environment.
4. Record network calls, ZIP opens, cache writes, startup time, repeated sample rate,
   disk usage, and worker behavior.
5. Confirm that simplification did not create a full-dataset mirror requirement.
6. Use only permitted development data for human visual checks. Do not expose held-out
   labels through comparison tooling.

## Acceptance rule

Semantic outputs must be identical unless an independently authorized Phase 0 decision
requires a change. Performance need not be identical, but the simplified path must
satisfy the `DG-P0-06` acceptance criterion and must not introduce an avoidable hot-path
network, ZIP, or cache-mutation bottleneck.

## Exit gate

Do not remove the legacy implementation if semantic equivalence fails, the cache bound
is violated, or the simplified runtime performs unacceptable recurring work.

---

# Task 9 — Remove the legacy path and finalize the simplified contract

**Task ID:** `P0-S09`  
**Purpose:** End with one maintainable implementation rather than two competing cache
systems.

## Preconditions

- Tasks 1–8 passed.
- Equivalence and performance evidence is durable and reviewable.
- A recovery checkpoint exists.

## Actions

1. Remove the superseded lease-based or dual-path implementation.
2. Remove obsolete private helpers, compatibility branches, metadata fields, tests,
   and configuration only when their replacements are active and verified.
3. Preserve tests for the safety guarantees, even when the old implementation details
   disappear.
4. Expose one documented public runtime path and one preparation path.
5. Update operational documentation, cache format/version documentation, and the
   Phase 0 verification register.
6. Confirm that no repository code imports or calls removed interfaces.
7. Confirm that stale legacy cache entries are safely ignored or explicitly cleaned by
   a non-destructive user-invoked procedure.

## Required final verification

- Complete Phase 0 test suite.
- Lint and Python compilation.
- Configuration and dependency-lock validation.
- Diff whitespace check.
- Representative cold and warm smoke tests.
- Multi-worker read test under the selected operating model.
- Semantic artifact comparison from Task 8.
- Review that no unrelated files or user changes were modified.

## Exit gate

The simplification is complete only when one implementation remains, all required
checks pass, documentation matches actual behavior, and no known safety regression is
hidden.

---

# 5. Stop and rollback conditions

Stop the active task and preserve the last known-good state if any of the following
occurs:

- a sample, split, annotation, timestamp, or manifest identity changes unexpectedly;
- warm loading performs remote work per sample;
- a partial cache entry becomes visible as complete;
- cache usage exceeds the configured bound without an explicit failure;
- active files can be evicted during a read;
- evaluation-input code queries or returns annotations;
- the simplified path requires an unapproved full-dataset materialization;
- a required test becomes weaker merely to make the refactor pass;
- the implementation conflicts with an unresolved Phase 0 decision gate.

Rollback must restore code and artifacts from the Task 1 checkpoint. Do not use
destructive Git operations against a dirty working tree or remove user data to perform
a rollback.

# 6. Completion checklist

```text
[ ] P0-S01 known-good baseline recorded
[ ] P0-S02 cache/runtime model selected from evidence
[ ] P0-S03 observable safety contract covered by tests
[ ] P0-S04 redundant internals removed without behavior change
[ ] P0-S05 explicit prepare/open/cleanup lifecycle implemented
[ ] P0-S06 dataset hot path proven read-only
[ ] P0-S07 eviction moved outside active reads
[ ] P0-S08 semantic and performance equivalence demonstrated
[ ] P0-S09 legacy path removed and documentation finalized
```

Completing this checklist demonstrates completion of the simplification effort only.
It does not by itself satisfy every Phase 0 dataset audit, project split, visual
validation, held-out integrity, or final freeze requirement.
