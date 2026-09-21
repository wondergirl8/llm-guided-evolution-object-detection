# Phase 0 — FRED Data Infrastructure and Input Pipeline Plan
## FRED + LLM-GE Research Project

**Status:** Version 1.0 — Implementation-Ready Phase Specification  
**Document path:** `docs/PHASE0_DATA_PLAN.md`  
**Phase:** Phase 0 — Data Infrastructure and Input Pipeline  
**Primary purpose:** Acquire, reproduce, validate, standardize, and efficiently expose FRED data for all later phases without making event-data representation or preprocessing itself an optimization target.

---

# 1. Role, Authority, and How to Use This Plan

## 1.1 Phase authority

This document is the **phase-specific technical authority for Phase 0**.

It specializes, but does not replace or silently override:

- `docs/MASTER_PROJECT_PLAN.md`;
- `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md`;
- root-level `AGENTS.md`.

Authority is divided by domain:

- the **Master Plan** owns project-wide research architecture, benchmark/evaluation policy, phase relationships, LLM-GE policy, reproducibility policy, and cross-phase contracts;
- the **Engineering Implementation Standard** owns software-engineering behavior, testing, state safety, failure handling, dependency/API verification, and coding-agent completion requirements;
- this **Phase 0 Plan** owns Phase 0 data-infrastructure decisions;
- a **current task** owns only the immediate implementation scope;
- the **repository** is evidence of the current implementation, not a policy source.

This plan MUST NOT silently override a fixed project-wide decision or an applicable Engineering Implementation Standard `MUST` requirement.

If this plan, another governing document, and the repository materially disagree, the coding agent MUST treat that as plan/implementation drift and surface it rather than resolving the difference by guesswork.

## 1.2 Decision-status language

This plan uses four decision statuses.

### FIXED

A Phase 0 project decision that is binding for Version 1.0.

### REQUIRED VERIFICATION

A planning-time assumption or externally derived fact that MUST be checked against the pinned repository, actual dataset, files, or runtime before implementation relies on it.

### OPEN DECISION

A decision intentionally deferred until specified implementation evidence exists. An open decision is governed by a named decision gate in Section 3. The coding agent MUST NOT silently resolve a research-sensitive open decision.

### FUTURE RESEARCH

A possible later extension outside the Version 1.0 implementation scope.

No other decision-status labels are normative in this plan.

## 1.3 Decision status versus normative severity

Decision status and normative severity are separate concepts.

- `FIXED` means the underlying project decision is settled.
- `MUST` / `MUST NOT` define mandatory implementation behavior.
- `SHOULD` / `SHOULD NOT` define the expected default when applicable.
- `MAY` defines an allowed option.

A `FIXED` objective can therefore contain several possible `SHOULD` implementation techniques without making every technique mandatory.

## 1.4 Agent navigation matrix

The coding agent should read only the sections relevant to the current work package, in addition to `AGENTS.md`, relevant Master Plan sections, and relevant Engineering Standard rules.

| Current work | Primary Phase 0 sections |
|---|---|
| Source acquisition/storage | §§3–6 |
| Dataset inventory | §§4–6, 10 |
| Challenging split | §§3, 7, 10 |
| Train/validation split | §§3, 7, 10 |
| Annotation parser | §§4, 8, 10 |
| RGB/event/timestamp pairing | §§4, 8, 10 |
| Canonical sample schema | §§8–9 |
| Manifest | §§3, 9 |
| Cache/runtime loading | §§3, 6, 9, 11 |
| Validation/data quality | §10 |
| Generic adapter contract | §12 |
| Phase 0 artifacts/freeze | §§13–15 |

For a non-trivial task, the coding agent SHOULD create the short pre-task contract required by `AGENTS.md` and identify the applicable work-package ID from Section 14.

---

# 2. Phase Goal, Scope, and Non-Goals

## 2.1 Goal — FIXED

Phase 0 SHALL build a **reliable, deterministic, validated, model-neutral, and low-overhead FRED data layer** that can support:

- YOLO11;
- RT-DETR;
- Faster R-CNN;
- future custom seed models;
- standalone non-evolutionary baselines;
- LLM-GE-generated/evolved candidates;
- later tracking;
- later trajectory forecasting.

The system SHALL preserve official FRED data semantics while exposing stable, versioned interfaces for later phases.

Phase 0 is infrastructure for trustworthy experiments. It is **not** itself the primary research optimization target.

## 2.2 Core principle — FIXED

Version 1.0 follows:

> **Reproduce and validate before innovating.**

Phase 0 SHALL reproduce and use the released/established FRED representation and handling as faithfully as practical before any separate research effort investigates alternative event representations.

Version 1.0 SHALL NOT invent a new:

- event accumulation method;
- event voxel representation;
- time surface;
- temporal window;
- synchronization algorithm;
- camera-alignment strategy;
- annotation coordinate system;
- benchmark train/test split.

Engineering additions such as strict validation, manifests, caching, explicit schemas, deterministic indexing, diagnostics, and adapter interfaces are allowed because they preserve semantics rather than redefine the benchmark.

## 2.3 In scope

Phase 0 includes:

- official dataset acquisition and source verification;
- immutable source-data organization;
- sequence/file inventory;
- challenging-split reproduction and verification;
- creation of a leakage-safe project train/validation boundary from challenging-train data;
- RGB/event pairing and timestamp/index verification;
- strict annotation parsing;
- coordinate and identity validation;
- canonical sample and manifest design;
- RGB, event, and paired RGB/event data views;
- generic model-adapter contracts;
- split/evaluation data-access protection;
- cache/version management;
- deterministic and idempotent rebuild behavior;
- data-quality and leakage checks;
- visual sanity checks on non-held-out development data;
- trusted automated integrity checks on held-out test data;
- throughput/runtime benchmarking;
- Phase 0 artifacts, reports, tests, and completion gate.

## 2.4 Out of scope

Version 1.0 does **not** include:

- new raw-event representations;
- evolution of data preprocessing;
- training or evolving YOLO11, RT-DETR, or Faster R-CNN;
- selecting exact detector variants/frameworks;
- detector-specific production adapter implementation;
- detector architecture mutation spaces;
- LLM-GE population/generation parameters;
- model-fitness implementation;
- final model/checkpoint selection;
- tracking algorithms;
- forecasting models;
- cross-family detector crossover.

Detector-family-specific adapters and minimal model-forward integration become requirements of the **Phase 1 baseline integration gate**, once exact implementations are selected.

## 2.5 Scope guardrail for coding agents

Phase 0 implementation MUST NOT expand into Phase 1 merely because a future consumer is known.

In particular, a coding agent implementing this plan MUST NOT:

- select detector variants;
- implement model training/evolution;
- modify LLM-GE search logic;
- define final model fitness;
- build detector-specific pipelines before Phase 1 decisions require them;
- replace the released event representation.

---

# 3. Decision Register and Decision Gates

This section prevents coding convenience from silently resolving decisions that require evidence.

A gate may be owned by either:

- **PROJECT/RESEARCH** — the coding agent may gather evidence and recommend options, but MUST NOT finalize the decision without explicit project approval;
- **IMPLEMENTATION** — the coding agent MAY select the option after satisfying the gate evidence and MUST record the decision and rationale.

## 3.1 Fixed decision register

The following decisions are fixed for Phase 0 Version 1.0.

| ID | Fixed decision |
|---|---|
| `P0-D01` | Phase 0 is reproduction/validation infrastructure, not event-representation research. |
| `P0-D02` | Use official released FRED synchronized data products. |
| `P0-D03` | Use released event frames for routine Version 1.0 experiments. |
| `P0-D04` | Preserve raw event HDF5 files for verification/future research but do not regenerate event frames routinely. |
| `P0-D05` | Use `coordinates.txt` as the initial canonical annotation source, subject to verification. |
| `P0-D06` | Preserve timestamp, bounding box, track ID, original class/type, sequence ID, and frame identity. |
| `P0-D07` | Use one model-neutral canonical FRED layer. |
| `P0-D08` | Use thin downstream/model adapters rather than parallel dataset implementations. |
| `P0-D09` | Use the official FRED challenging split as the default project evaluation framework. |
| `P0-D10` | Create project train/validation membership only from challenging-train data. |
| `P0-D11` | Keep the official challenging test outside search/tuning feedback. |
| `P0-D12` | Split development data at a leakage-safe sequence/group level, never by random frame assignment. |
| `P0-D13` | Protect split, canonical labels, held-out labels, and benchmark definitions from LLM-GE candidate modification. |
| `P0-D14` | Perform expensive deterministic discovery/parsing/validation once and reuse validated artifacts. |
| `P0-D15` | Treat source FRED data as immutable. |
| `P0-D16` | Surface malformed/questionable data; never silently discard or repair it. |
| `P0-D17` | Preserve metadata needed by tracking and forecasting. |
| `P0-D18` | Phase 0 defines a generic adapter contract; model-family-specific adapter completion belongs to Phase 1. |

## 3.2 `DG-P0-01` — Source acquisition and storage

**Owner:** IMPLEMENTATION  
**Trigger:** Project server/storage environment is available for inspection.

**Evidence required:**

- available capacity;
- permissions;
- expected dataset/archive size and behavior;
- official download mechanisms available;
- local/high-throughput storage options;
- source/derived/cache separation feasibility.

**Decision produced:**

- physical raw-data root;
- derived/manifests/cache roots;
- official acquisition mechanism used on the server.

**Constraint:** paths remain configurable; reusable code MUST NOT embed user-specific absolute paths.

**Blocks:** acquisition automation and source freeze.

## 3.3 `DG-P0-02` — Leakage-safe development split

**Owner:** PROJECT/RESEARCH  
**Trigger:** source inventory and leakage/grouping audit are complete.

**Evidence required:**

- challenging-train sequence count and lengths;
- drone identity/class distribution;
- condition distribution where available;
- recording-session/repeated-scene relationships where discoverable;
- evidence for or against grouping stronger than sequence ID.

**Decision produced:**

- grouping unit;
- train/validation ratio;
- deterministic split seed or explicit membership procedure;
- frozen project split version.

**Blocks:** final project train/validation manifest.

## 3.4 `DG-P0-03` — Manifest physical backend

**Owner:** IMPLEMENTATION  
**Trigger:** canonical sample contract and expected access patterns are defined.

**Evidence required:**

- dataset scale;
- startup/loading speed requirements;
- sequential/random access needs;
- schema-validation requirements;
- concurrency behavior;
- portability and inspection needs;
- implementation complexity.

**Decision produced:** physical manifest/index format and schema storage mechanism.

Permitted candidates include Parquet, JSONL, SQLite, or another validated indexed representation.

**Hard requirements:** the chosen format MUST be versionable, deterministic, safely rebuildable, portable enough for the intended environment, and MUST avoid full annotation reparsing on every experiment.

## 3.5 `DG-P0-04` — Questionable/corrupted-data handling

**Owner:** PROJECT/RESEARCH  
**Trigger:** the initial full dataset audit has produced a known-data-issue registry.

**Evidence required:**

- issue types and frequencies;
- affected sequences/samples;
- upstream documentation/issues where relevant;
- correctness impact;
- whether issues can be resolved without changing benchmark semantics.

**Decision produced:**

- blocking versus warning categories;
- approved inclusion/exclusion rules;
- whether any source sequence/sample is excluded from formal experiments.

**Blocks:** final canonical manifest and Phase 0 freeze when unresolved issues could affect validity.

No coding agent may invent exclusion thresholds before this gate.

## 3.6 `DG-P0-05` — Visual-validation coverage

**Owner:** IMPLEMENTATION  
**Trigger:** inventory, split identity, and known issue classes are available.

**Evidence required:** dataset size, sequence/condition coverage, boundary cases, flagged sequences.

**Decision produced:** deterministic visual-validation sample set and seeded supplemental sample procedure.

The chosen set MUST include:

- project-train examples;
- project-validation examples;
- representative boundary cases;
- flagged/error-prone development sequences.

Official held-out test labels are not included in ordinary visual overlays; see §7.6 and §10.7.

## 3.7 `DG-P0-06` — Cache/runtime strategy and performance acceptance

**Owner:** IMPLEMENTATION  
**Trigger:** canonical manifest/loader exists on the intended server.

**Evidence required:**

- loader throughput;
- storage throughput;
- CPU utilization;
- memory behavior;
- worker/prefetch experiments;
- repeated-run startup cost;
- representative batch-delivery timing.

**Decision produced:**

- cache backend, if one is needed;
- worker/prefetch/persistent-worker settings for the environment;
- cache location;
- Phase 0 throughput acceptance statement supported by measurements.

Phase 0 does not need to guess a universal fixed throughput percentage before hardware is measured. Phase 1 baseline integration SHOULD additionally confirm that the chosen loader does not materially starve the selected detector during actual training/inference.

---

# 4. Upstream FRED Evidence and Verification Register

Planning-time upstream observations are useful evidence but are not immutable facts. They MUST be rechecked against the pinned repository/data revision before implementation depends on them.

## 4.1 Planning-time references

- Official FRED repository: `https://github.com/miccunifi/FRED`
- Official FRED dataset release used during planning: `https://huggingface.co/datasets/GabrieleMagrini/FRED`

Planning-time observations include:

- released sequences contain RGB, event data, and annotations;
- event data includes extracted event frames and raw `.hdf5` streams;
- the example loader uses `RGB/*.jpg` and `Event/Frames/*.png`;
- the example loader reads `coordinates.txt`;
- annotation documentation describes `time: x1, y1, x2, y2, id, class`;
- `coordinates.txt` and `coordinates_rgb.txt` are released;
- a challenging split exists under `dataset_splits/challenging/`;
- the planning-time challenging-split helper script constructs a derived view using supplied membership lists;
- the planning-time reference parser can skip malformed lines after parsing errors;
- upstream issues/questions have existed around coordinates, alignment/synchronization, annotations, and HDF5 handling.

These observations do not prove a problem exists in the exact pinned project data. They require project-side verification.

## 4.2 Verification register

| ID | Planning-time observation | Verify against | Required before |
|---|---|---|---|
| `P0-V01` | Official dataset/repository identities are correct | pinned repository + actual downloaded release | source freeze |
| `P0-V02` | Released event frames are available for routine use | actual dataset | canonical representation implementation |
| `P0-V03` | RGB/Event frame naming and one-to-one pairing convention | pinned repository + multiple actual sequences | pairing implementation |
| `P0-V04` | `coordinates.txt` format and semantic meaning | pinned docs/repository + actual files | strict parser |
| `P0-V05` | `coordinates.txt` uses the intended shared coordinate space | actual dimensions + overlays on development data | canonical box contract |
| `P0-V06` | Challenging split membership and helper behavior | pinned split lists/script | split freeze |
| `P0-V07` | Timestamp/index derivation and annotation-to-frame mapping | actual sequences/annotations | canonical sample freeze |
| `P0-V08` | Known upstream issue reports affect or do not affect pinned data | pinned repo/issues + project audit | Phase 0 freeze |
| `P0-V09` | Raw HDF5 inventory is present where expected | actual dataset | source inventory |
| `P0-V10` | All sequences follow the same or explicitly classified pairing conventions | full inventory/validation | manifest freeze |

The verification result MUST be recorded as evidence, not only printed to console.

---

# 5. High-Level Phase 0 Architecture

## 5.1 Fixed architecture

```text
                     OFFICIAL FRED SOURCE
                             │
                             ▼
                 immutable source-data layer
                             │
                             ▼
              official challenging split mapping
                             │
                             ▼
              project train/validation membership
                             │
                             ▼
         parsing + pairing + validation + indexing
                             │
                             ▼
              versioned canonical FRED manifest
                             │
                             ▼
               canonical FRED sample interface
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
        RGB view         Event view       RGB/Event view
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                     generic adapter contract
                             │
                             ▼
                 downstream phase/model adapters
```

The canonical data layer is the Phase 0 source of truth.

Models/adapters translate from this canonical representation. They do not redefine the dataset, split, annotations, or canonical identity.

## 5.2 Runtime principle — FIXED

Phase 0 is designed as:

> **Expensive/strict once; cheap repeatedly.**

One-time or infrequent work includes:

```text
download / source verification
        ↓
inventory
        ↓
split verification
        ↓
annotation parsing
        ↓
modality/timestamp pairing
        ↓
validation
        ↓
manifest/index build
        ↓
artifact freeze
```

Repeated model/LLM-GE runs should primarily do:

```text
load versioned manifest/index
        ↓
read required frame(s)
        ↓
thin adapter transformation
        ↓
model training/inference/evaluation
```

The following MUST NOT normally be repeated for every candidate:

- dataset download/extraction after a valid source exists;
- whole-dataset directory discovery;
- challenging-split construction;
- complete annotation reparsing when a validated reusable representation exists;
- full integrity scans;
- event-frame regeneration;
- synchronization reconstruction;
- train/validation membership generation.

---

# 6. Source Acquisition, Storage, and Immutability

## 6.1 Official source — FIXED

The project SHALL use the official FRED release or an explicitly verified official mirror.

The FRED GitHub repository is a benchmark/code reference and does not replace the full released dataset.

`DG-P0-01` selects the server-specific acquisition mechanism and physical storage locations.

## 6.2 Raw/source data immutability — FIXED

Original FRED source data is read-only project truth.

Normal project code MUST NOT:

- rewrite RGB images;
- rewrite released event frames;
- rewrite raw HDF5 streams;
- edit source annotations;
- rename source sequences in place;
- alter official split-list contents;
- silently delete suspect files/samples.

Derived products MUST live separately.

A conceptual layout is:

```text
data/
├── raw/          # immutable official source
├── derived/      # project-created derived artifacts
├── manifests/    # versioned indexes/split manifests
└── cache/        # disposable/rebuildable runtime cache
```

This layout is **non-normative**. Exact roots are configuration-driven through `DG-P0-01`.

## 6.3 Source inventory — FIXED

Before downstream use, Phase 0 MUST produce an auditable sequence inventory.

For every sequence, record at least:

- sequence ID;
- source location/archive identity;
- official split/source membership where available;
- RGB directory/presence status;
- event-frame directory/presence status;
- raw HDF5 presence/status;
- annotation-file presence;
- RGB frame count;
- event-frame count;
- annotation entry count;
- first/last frame identity;
- first/last annotation timestamp where parseable;
- basic file/readability status;
- Phase 0 validation status.

The inventory MUST be a machine-readable durable artifact, not only terminal output.

## 6.4 Configurable paths — FIXED

Reusable code MUST NOT contain hardcoded user/server paths such as `/home/<user>/...` or `/scratch/<username>/...`.

Environment-specific roots MUST come from configuration or documented environment variables.

---

# 7. Split Integrity, Leakage Prevention, and Evaluation Isolation

## 7.1 Official challenging split — FIXED

The official FRED challenging split is the default project evaluation framework.

Phase 0 MUST reproduce/verify the official challenging membership from the pinned FRED repository.

The implementation MAY use symlinks, manifests, or another derived view if:

- membership is exact;
- source data is not modified;
- unnecessary data duplication is avoided;
- split identity is versioned and auditable.

Official challenging train/test membership is protected project state.

LLM-GE and model adapters MUST NOT modify:

- train/test membership;
- sequence assignments;
- official split manifests;
- held-out evaluation policy.

## 7.2 Project train/validation boundary — FIXED

The official challenging test set MUST NOT be used for:

- LLM-GE fitness;
- architecture selection;
- prompt tuning;
- Evolution of Thought feedback;
- hyperparameter tuning;
- early stopping;
- model-family selection;
- manual search decisions based on test performance.

Project training and validation membership SHALL be derived only from challenging-train data.

## 7.3 Leakage-safe split unit — FIXED

Train/validation splitting MUST occur at a sequence-level or stronger leakage-safe grouping.

Individual frames from one sequence MUST NOT be randomly divided across train and validation.

Before the split is frozen, Phase 0 MUST investigate whether stronger grouping is necessary, including where metadata permits:

- same recording session;
- repeated scene;
- one continuous capture split into multiple sequence IDs.

`DG-P0-02` resolves the final grouping unit, ratio, and seed/membership method.

## 7.4 Leakage invariants — FIXED

At the sequence/group level:

```text
project_train ∩ project_validation = ∅
project_train ∩ official_challenging_test = ∅
project_validation ∩ official_challenging_test = ∅
```

At the canonical-sample level:

```text
train_sample_ids ∩ validation_sample_ids = ∅
train_sample_ids ∩ test_sample_ids = ∅
validation_sample_ids ∩ test_sample_ids = ∅
```

Violations are blocking failures.

Checks MUST detect, where applicable:

- sequence overlap;
- exact file overlap;
- symlink/path alias overlap;
- duplicate sample IDs;
- accidental duplicate membership.

Content hashing or another reliable mechanism SHOULD be used for high-risk overlap checks where practical.

## 7.5 Candidate-facing data protection — FIXED

Phase 0 defines data-access boundaries; the full model evaluator remains owned by the Master/Phase 1 architecture.

During validation/test inference, candidate/model code MUST NOT receive hidden target information such as:

- bounding boxes;
- target labels not legitimately part of the input;
- annotation paths/objects;
- held-out ground-truth objects;
- future frames unless the task explicitly permits them;
- unnecessary split labels;
- metadata that directly reveals target ground truth.

Trusted training code MAY access training inputs and training targets.

A protected evaluator/harness owns validation/test labels and scoring outside candidate code.

Candidate/evolved code MUST NOT receive write authority over:

- raw FRED source;
- canonical/split manifests;
- validation/test membership;
- canonical annotation values;
- held-out annotations;
- evaluator configuration/metric code.

The exact OS/container permission mechanism is environment-specific, but the access boundary is fixed.

## 7.6 Official test-ground-truth integrity policy — FIXED

To resolve the distinction between **dataset validation** and **model evaluation**, official test ground truth follows this policy:

1. A trusted Phase 0 integrity process MAY programmatically read official test annotations only for dataset-integrity checks required to establish that files parse, pair, and satisfy structural invariants.
2. Test ground truth MUST NOT be exposed to candidate/model code, prompts, fitness, model-selection logic, or ordinary development tooling.
3. Routine human visual overlays with test bounding boxes are NOT part of Phase 0 validation. Visual annotation/alignment checks use project-train and project-validation data.
4. If automated integrity checks identify a test-specific anomaly that cannot be resolved without human inspection, that inspection requires an explicit documented project decision and must be recorded in the known-data-issue registry.
5. Once Phase 0 is frozen, ordinary development/search workflows MUST NOT access official test ground truth. Final/frozen evaluation access is governed by the Master/Phase 1 evaluation protocol.

This policy allows dataset-integrity verification without turning test labels into development feedback.

## 7.7 Metadata leakage review — REQUIRED VERIFICATION

Phase 0 MUST inspect whether candidate-facing:

- filenames;
- sequence numbers;
- directory names;
- timestamps;
- metadata fields

could reveal unintended target information.

Identifiers are not automatically forbidden. The review must distinguish legitimate task context from evaluator-only information.

## 7.8 No data-dependent benchmark manipulation — FIXED

The project MUST NOT change evaluation membership or preprocessing after viewing test performance merely to improve scores.

If a legitimate benchmark/data defect is discovered after formal test evaluation starts:

1. stop the affected comparison;
2. document the issue;
3. determine whether the protocol is invalidated;
4. version any approved correction;
5. rerun all affected comparisons consistently.

Model architecture, checkpoint, prompts, and final model evaluation freeze policy are owned by the Master/Phase 1 plan, not by Phase 0.

---

# 8. Canonical FRED Representation and Sample Contract

## 8.1 Released event representation — FIXED

Version 1.0 uses the **already extracted event frames released by FRED**.

Routine training/evaluation MUST NOT regenerate the standard event-frame representation from raw HDF5 streams.

Raw HDF5 files SHALL be:

- preserved;
- inventoried;
- available for verification/future research;
- excluded from ordinary Version 1.0 preprocessing unless a verification task requires them.

## 8.2 Annotation source — FIXED, subject to verification

The initial canonical annotation source is:

```text
coordinates.txt
```

subject to `P0-V04` and `P0-V05`.

`coordinates_rgb.txt` remains available for verification/future RGB-specific analysis, but it is not a parallel source of canonical truth in Version 1.0.

## 8.3 Annotation preservation — FIXED

The canonical layer MUST preserve at least:

- `timestamp`;
- `x1`, `y1`, `x2`, `y2`;
- `track_id`;
- original `class_name`;
- source sequence ID;
- source frame index.

Detection-specific simplification, such as collapsing classes, belongs to Phase 1 adapters and MUST NOT destroy metadata required for tracking/forecasting.

## 8.4 Canonical bounding boxes — FIXED

Canonical boxes use:

```text
[x1, y1, x2, y2]
```

in the verified FRED common coordinate space.

Model-specific target conversion belongs downstream.

No clipping, coordinate repair, or silent correction is permitted before an explicit approved data-quality policy exists.

## 8.5 Coordinate verification — REQUIRED VERIFICATION

Phase 0 MUST verify:

- RGB dimensions;
- event-frame dimensions;
- padding behavior;
- coordinate bounds;
- relationship between `coordinates.txt` and actual images;
- out-of-bounds annotations;
- cross-sequence convention consistency;
- whether known coordinate issues affect pinned data.

Questionable annotations are recorded and classified rather than silently changed.

## 8.6 Pairing and timestamp policy — FIXED + REQUIRED VERIFICATION

Phase 0 follows released FRED pairing semantics rather than inventing a synchronization algorithm.

Before the pairing contract is frozen, verify on actual data:

- RGB/event frame counts;
- ordering and filename conventions;
- missing/duplicate/gapped frames;
- expected index offset;
- timestamp derivation;
- annotation-to-frame association;
- first/last-frame behavior;
- whether conventions vary across sequences.

No assumed timestamp formula may be copied blindly from planning notes.

## 8.7 Canonical sample contract — FIXED

The semantic contract is named:

```text
P0-CONTRACT-FRED-SAMPLE-v1
```

Conceptually:

```text
FREDSample
├── sample_id
├── sequence_id
├── frame_index
├── timestamp
├── rgb
│   ├── path/reference
│   └── verified metadata
├── event
│   ├── frame_path/reference
│   └── verified metadata
├── annotations
│   ├── boxes_xyxy
│   ├── track_ids
│   └── original_classes
├── split
│   ├── official_split
│   └── project_split
└── provenance
    ├── fred_dataset_version
    ├── fred_repository_revision
    ├── phase0_schema_version
    └── manifest_version
```

Exact Python types/classes are implementation details.

Changes to required semantic fields or meanings require a schema/interface version change.

## 8.8 Stable sample identity — FIXED

Every sample MUST have a deterministic stable project identifier derived from stable source identity, for example:

```text
<sequence_id>:<frame_index>
```

or an equivalent documented scheme.

Sample identity MUST NOT depend on:

- runtime enumeration order;
- mutable absolute paths;
- regenerated random UUIDs.

## 8.9 Data views — FIXED

The same canonical sample identity SHALL support:

```text
RGB view       → RGB frame + target metadata
Event view     → released event frame + target metadata
RGB/Event view → synchronized pair + shared target metadata
```

These are views of one canonical dataset, not independently indexed datasets.

---

# 9. Manifest, Cache, Configuration, and Data Access

## 9.1 Canonical manifest — FIXED

Phase 0 MUST produce a reusable sample-level canonical manifest/index.

At minimum it encodes or references:

- sample ID;
- sequence ID;
- frame index;
- RGB reference;
- event-frame reference;
- annotation association;
- timestamp;
- official split;
- project split;
- validation status;
- schema/version identifiers.

The physical backend is selected through `DG-P0-03`.

## 9.2 Manifest versioning — FIXED

A manifest MUST identify:

- manifest/schema version;
- source dataset identity/version;
- FRED repository revision;
- official challenging split file revision/hash;
- project split version;
- annotation policy/version;
- creation configuration;
- creation timestamp;
- validation status.

Formal experiments reference the manifest version rather than implicitly relying on "whatever is currently on disk."

## 9.3 Safe rebuilds — FIXED

Manifest generation MUST be deterministic and idempotent.

A rebuild MUST NOT:

- append duplicates;
- mix old/new records;
- silently alter split membership;
- replace a valid final artifact with a partial one.

Important writes SHOULD use a temporary artifact, validation, then atomic promotion/rename where practical.

## 9.4 Cache policy — FIXED

Caches exist only to reduce repeated deterministic work.

Caches are **not source truth**.

A cache MUST be disposable and rebuildable from:

- immutable source data;
- frozen split membership;
- Phase 0 configuration;
- versioned schema/annotation policy.

Appropriate cached material MAY include:

- parsed annotations;
- validated image metadata;
- deterministic indexes.

Cache backend/runtime settings are selected through `DG-P0-06`.

## 9.5 Cache invalidation — FIXED

Cached artifacts MUST contain enough identity to reject incompatible reuse.

Incompatibility includes, where applicable:

- different FRED source version;
- changed annotation policy;
- changed split version;
- changed manifest/schema version;
- changed canonical coordinate interpretation.

Stale-cache detection is correctness behavior, not only performance optimization.

## 9.6 Image transformations and augmentation boundary — FIXED

Canonical Phase 0 preserves released RGB/event frames.

Model-specific:

- resizing;
- normalization;
- padding;
- augmentation;
- target conversion

belong to downstream training/model adapters.

Phase 0 SHOULD NOT generate permanent model-specific resized dataset copies unless profiling demonstrates a strong need and the derived-cache design remains explicitly versioned/rebuildable.

Augmentations MUST NOT mutate canonical labels or source data.

## 9.7 Phase 0 configuration — FIXED

Phase 0 MUST provide a configuration source for the implemented environment.

It SHALL define, as applicable:

- raw dataset root;
- derived-data root;
- manifest location;
- cache root;
- challenging-split source;
- project split manifest;
- annotation source;
- schema version;
- supported modality selection;
- legitimate validation strictness/configuration;
- runtime settings chosen through `DG-P0-06`.

Exact configuration technology/file layout is an implementation choice.

Research-critical defaults MUST be explicit.

## 9.8 Single sources of truth — FIXED

The project MUST NOT independently redefine:

- split membership;
- class/annotation interpretation;
- coordinate conventions;
- manifest schema;
- sample-ID scheme.

Each concept has one authoritative project artifact/configuration.

## 9.9 Data access API semantics — FIXED

Scientifically meaningful state MUST be explicit.

A conceptual, non-normative API is:

```text
dataset = FREDData(
    split="train",
    modality="rgb_event",
    manifest_version="..."
)
```

Exact Python APIs are implementation-specific, but the interface must explicitly represent:

- project split;
- modality;
- manifest/schema version;
- training/evaluation access mode;
- downstream adapter selection where relevant.

Behavior MUST NOT be inferred from incidental path structure or tensor count.

---

# 10. Validation, Data Quality, and Testing Specification

## 10.1 Validation layers

Phase 0 validation proceeds through:

```text
source integrity
      ↓
sequence structure
      ↓
modality pairing
      ↓
annotation parsing
      ↓
coordinate/timestamp consistency
      ↓
split/leakage integrity
      ↓
canonical sample construction
      ↓
generic adapter boundary
      ↓
runtime throughput
```

## 10.2 Source integrity

Validate, as applicable:

- required source directories/sequences;
- readable/non-zero files;
- expected modality directories;
- annotation-file presence;
- HDF5 inventory presence where expected;
- no destructive modification.

Hashes/checksums SHOULD be recorded for frozen critical artifacts where practical.

## 10.3 Strict annotation parsing

The formal annotation parser MUST NOT silently ignore malformed lines.

Each input line must become one of:

- valid parsed annotation;
- explicit approved special case;
- validation finding requiring review.

Diagnostics for malformed data SHOULD retain:

- sequence identity;
- file path/reference;
- line number;
- offending content or safe representation;
- failure category.

## 10.4 RGB/event pairing checks

For every usable sequence, validate:

- RGB count;
- event-frame count;
- ordering;
- verified one-to-one pairing convention;
- missing frames;
- duplicate frames;
- unexpected gaps;
- readable image content.

A count mismatch is not automatically repairable.

## 10.5 Annotation/coordinate checks

Validate:

- annotation file readability;
- parseability/classification of every formal line;
- finite box values;
- `x2 > x1`;
- `y2 > y1`;
- parseable track IDs/classes/timestamps;
- verified annotation-to-frame association;
- coordinate compatibility with actual image dimensions.

## 10.6 Sequence/identity checks

Validate:

- stable sequence IDs;
- deterministic frame order;
- timestamp monotonicity where expected;
- preserved track IDs;
- no accidental global identity remapping;
- temporal gaps recorded rather than hidden.

## 10.7 Visual validation

Visual validation is required on development data.

The deterministic/seeded sample set chosen through `DG-P0-05` SHALL include:

- project-train examples;
- project-validation examples;
- boundary cases;
- flagged/error-prone development sequences.

Overlays SHOULD show, where useful:

- RGB frame;
- event frame;
- boxes;
- track IDs.

Visual inspection is intended to detect:

- coordinate offsets;
- wrong modality pairing;
- frame-index shifts;
- padding misunderstandings;
- box corruption.

Official held-out test labels are **not** included in routine human visual overlays. Test integrity is checked programmatically under §7.6.

## 10.8 Questionable/corrupted-data workflow

Questionable data MUST NOT be silently repaired or removed.

The workflow is:

```text
detect finding
    ↓
record evidence
    ↓
classify provisional severity
    ↓
check upstream evidence
    ↓
record in known-data-issue registry
    ↓
DG-P0-04 if policy decision is required
    ↓
versioned inclusion/exclusion handling
```

Any exclusion affecting formal experiments MUST be documented, reproducible, and consistently applied across compared models.

Two operational modes apply:

- **audit mode:** evidence collection may continue while unresolved findings are recorded;
- **freeze/formal mode:** unresolved findings capable of affecting correctness block the Phase 0 freeze.

## 10.9 Blocking versus warning findings

Blocking findings include, at minimum:

- train/validation/test overlap;
- unresolved RGB/event mismatch affecting sample correspondence;
- unresolved coordinate interpretation;
- unreadable required data;
- inconsistent frozen manifest membership;
- held-out labels exposed to candidate/model development code;
- invalid canonical sample construction.

A finding may be downgraded to warning only when there is documented evidence that it does not compromise correctness or scientific validity.

## 10.10 Known-upstream-issue review

Before Phase 0 freezes, review pinned FRED repository changes/issues relevant to:

- coordinates;
- synchronization;
- annotations;
- HDF5/event handling;
- split definitions.

For each relevant issue record:

- whether pinned project data is affected;
- evidence;
- project handling;
- whether a sequence/sample is blocked or excluded.

## 10.11 Determinism

Given the same:

- source dataset;
- pinned repository revision;
- Phase 0 configuration;
- split version/seed;

Phase 0 MUST reproduce the same:

- inventory;
- project split;
- sample IDs;
- canonical manifest records;
- annotation interpretation;
- deterministic validation results.

Unavoidable nondeterminism must be documented.

## 10.12 Smoke workflow

Phase 0 MUST provide a low-cost smoke workflow that:

1. loads a tiny deterministic non-held-out subset;
2. reads RGB/event pairs;
3. loads annotations;
4. builds canonical samples;
5. passes samples through the generic adapter/test boundary;
6. verifies target/data structure;
7. completes without improper held-out-label access.

A toy/debug subset MAY be derived only from project training data for CI/smoke work.

Any toy/debug subset MUST have explicit membership and MUST NOT be used for reportable model conclusions.

---

# 11. Runtime Performance Requirements

## 11.1 Performance objective — FIXED

Phase 0 is optimized for **amortized reuse**, not novel preprocessing.

Repeated experiments SHOULD consume already prepared/indexed data.

Appropriate techniques MAY include:

- precomputed manifests;
- cached parsed metadata;
- deterministic indexes;
- dataloader workers;
- prefetching;
- pinned host memory where appropriate;
- persistent workers where appropriate;
- high-throughput local storage where available;
- avoiding repeated re-encoding/decompression/directory scans.

All optimizations MUST preserve official data semantics.

## 11.2 Performance acceptance

Before Phase 0 freezes, run a representative throughput benchmark on the intended server.

The Phase 0 acceptance statement is:

> The validated reusable data path must not contain an avoidable recurring I/O or deterministic preprocessing bottleneck relative to the measured capabilities of the intended environment.

The benchmark MUST record enough context to reproduce the conclusion, including as applicable:

- server/storage identity;
- manifest/cache version;
- batch/sample configuration;
- worker/prefetch settings;
- warm-up handling;
- measured sample/batch delivery rate;
- startup/repeated-run cost;
- CPU/memory behavior.

`DG-P0-06` selects environment-specific settings and documents whether the pipeline satisfies the criterion.

Because exact detector implementations belong to Phase 1, Phase 1 baseline integration SHOULD confirm end-to-end that data delivery does not materially starve the selected detector during representative execution.

---

# 12. Generic Adapter and Downstream Consumer Contract

## 12.1 Phase 0 adapter boundary — FIXED

The boundary is:

```text
canonical Phase 0 sample
          ↓
generic adapter contract
          ↓
phase/model-specific adapter
          ↓
model/task-required input
```

Phase 0 defines what downstream adapters may rely on.

A downstream adapter MAY:

- resize;
- normalize;
- convert target format;
- collate;
- create model-specific metadata;
- apply configured model/training augmentation in the appropriate phase.

A downstream adapter MUST NOT:

- alter split membership;
- silently change canonical annotations;
- modify sample identity;
- regenerate the Version 1.0 event representation;
- hide invalid canonical data;
- expose held-out ground truth to candidate inference.

## 12.2 Adapter reproducibility

Any adapter behavior that affects research results MUST be configuration-driven and versioned in the consuming phase.

Examples include:

- input size;
- interpolation;
- normalization;
- augmentation;
- label conversion;
- padding;
- batching/collation.

## 12.3 Baseline/LLM-GE parity — FIXED

Standalone baselines and LLM-GE candidates must consume the same canonical Phase 0 source of truth.

```text
              canonical Phase 0
                     │
             downstream adapter
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
standalone baseline       LLM-GE candidate
```

LLM-GE MUST NOT receive an alternate dataset/preprocessing path that changes the scientific comparison.

## 12.4 Phase 0 versus Phase 1 responsibility

Phase 0 completion requires:

- `P0-CONTRACT-FRED-SAMPLE-v1`;
- generic adapter contract documentation;
- generic adapter/interface tests;
- smoke-test harness;
- sufficient RGB/event/multimodal access to allow downstream adapters.

Phase 0 completion does **not** require final YOLO11, RT-DETR, or Faster R-CNN production adapters.

Phase 1 baseline integration SHALL implement and validate model-family-specific adapters after exact seed implementations are fixed.

This removes the circular dependency between Phase 0 completion and Phase 1 model choices.

---

# 13. Phase 0 Artifacts and Cross-Phase Contracts

## 13.1 Required durable artifacts

Phase 0 produces or defines:

1. **Source inventory** — complete auditable source/sequence inventory.
2. **Official split manifest** — verified challenging-split membership.
3. **Project split manifest** — frozen development train/validation membership and reference to held-out test membership.
4. **Canonical dataset manifest/index** — sample-level model-neutral index.
5. **Known-data-issue registry** — all relevant findings and approved handling.
6. **Validation report** — machine-readable results plus human-readable summary.
7. **Visual validation outputs** — development-data overlays selected under `DG-P0-05`.
8. **Schema/interface documentation** — canonical sample and generic adapter contracts.
9. **Performance report** — representative server/storage throughput measurements.
10. **Verification register results** — `P0-V01`–`P0-V10` evidence/status.

Large datasets/heavy caches generally remain outside Git. Git SHOULD contain code, configuration, small manifests where practical, schema definitions, validation summaries, and durable references to external artifact locations.

## 13.2 Data-use auditability

A formal downstream run must be able to identify by reference:

- manifest version;
- project split;
- sample IDs/membership version;
- modality;
- downstream adapter/config;
- Phase 0 schema version;
- source FRED identity/revision;
- approved exclusions, if any.

This information need not be duplicated into every run directory if a stable referenced artifact provides it.

## 13.3 Output contract to Phase 1

Phase 1 receives:

- frozen Phase 0 manifest/index version;
- train/validation/held-out-test identities;
- `P0-CONTRACT-FRED-SAMPLE-v1`;
- RGB/event/paired access;
- validated annotation representation;
- generic adapter contract;
- validation report;
- known-data-issue registry;
- reproducibility/provenance metadata;
- runtime-loading guidance.

Phase 1 SHALL NOT need to reconstruct FRED synchronization, annotation semantics, or benchmark membership itself.

## 13.4 Metadata preserved for later phases

Phase 0 must retain enough information for:

### Phase 2 — Tracking

- sequence ID;
- chronological frame order;
- timestamp;
- track ID;
- bounding box;
- modality references.

### Phase 3 — Forecasting

- stable sequence history;
- timestamps;
- identities;
- trajectories derivable from annotations/tracks;
- consistent coordinates.

Detection-specific simplification MUST NOT destroy these fields in Phase 0.

## 13.5 Phase 0 README

`phase0_data/README.md` is operational documentation and SHOULD eventually contain:

- what Phase 0 is;
- environment/data configuration;
- canonical implemented commands;
- validation/rebuild instructions;
- artifact locations;
- link to this plan;
- current implementation status.

It SHOULD NOT duplicate this plan's normative policy.

---

# 14. Canonical Implementation Roadmap

This is the **single authoritative Phase 0 execution roadmap**. Other scope/summary sections do not define a competing implementation order.

Work-package IDs are used in coding tasks and pre-task contracts.

## Stage A — Evidence, environment, and acquisition

### `P0-A1` Pin and verify upstream references

**Work:**

- pin FRED repository revision;
- identify exact dataset release/source;
- begin `P0-V01`–`P0-V10` verification record.

**Exit:** upstream revisions are recorded and planning-time assumptions are explicitly marked verified/pending.

### `P0-A2` Resolve source storage/acquisition

**Gate:** `DG-P0-01`

**Work:**

- configure raw/derived/manifest/cache roots;
- implement/execute approved acquisition workflow;
- enforce source immutability expectations.

**Exit:** verified source exists and can be inventoried without modifying it.

## Stage B — Inventory and initial audit

### `P0-B1` Build source inventory

**Work:**

- discover sequences/files;
- collect counts/presence/readability metadata;
- record HDF5 and annotation status.

**Tests:** deterministic rerun; duplicate sequence detection; no source modification.

**Exit:** versioned source inventory artifact exists.

### `P0-B2` Review upstream/data-quality risk

**Work:**

- compare inventory/findings with relevant pinned upstream issues;
- seed known-data-issue registry.

**Exit:** affected/unaffected/unknown status recorded for relevant issue classes.

## Stage C — Official challenging split

### `P0-C1` Reproduce and verify challenging membership

**Work:**

- parse pinned official split definitions;
- create derived split mapping/manifest;
- verify exact membership and no source mutation.

**Tests:** membership equality, overlap checks, deterministic rebuild.

**Exit:** versioned official split manifest exists.

## Stage D — Project development split

### `P0-D1` Audit leakage grouping

**Work:**

- inspect sequence/session/scene relationships;
- quantify class/identity/condition distribution where available.

**Exit:** evidence package for `DG-P0-02`.

### `P0-D2` Freeze project train/validation membership

**Gate:** `DG-P0-02`

**Work:**

- implement approved grouping/ratio/seed or explicit membership;
- build versioned project split manifest;
- run sequence/sample overlap checks.

**Exit:** leakage-safe project split is frozen.

## Stage E — Parsing, pairing, and canonical representation

### `P0-E1` Implement strict annotation parser

**Governing verification:** `P0-V04`, `P0-V05`

**Work:**

- parse `coordinates.txt`;
- preserve required fields;
- emit explicit findings for malformed content.

**Tests:** valid lines, malformed lines, numeric/identity/class/timestamp edge cases, deterministic output.

**Exit:** parser passes strict unit tests and does not silently discard formal lines.

### `P0-E2` Verify/implement RGB-event-timestamp pairing

**Governing verification:** `P0-V03`, `P0-V07`, `P0-V10`

**Work:**

- establish verified ordering/index/timestamp association;
- classify sequence-specific deviations.

**Tests:** full-sequence count/order checks, boundary frames, missing/duplicate/gap cases.

**Exit:** pairing contract is evidence-backed.

### `P0-E3` Implement canonical sample contract

**Work:**

- implement `P0-CONTRACT-FRED-SAMPLE-v1`;
- implement stable sample IDs;
- expose RGB/event/paired views.

**Tests:** schema, identity stability, annotation preservation, deterministic construction.

**Exit:** canonical samples can be constructed independently of any detector family.

## Stage F — Canonical manifest and data-quality freeze inputs

### `P0-F1` Select manifest backend

**Gate:** `DG-P0-03`

**Exit:** backend choice/rationale recorded.

### `P0-F2` Build versioned canonical manifest

**Work:**

- generate canonical records;
- include provenance/split/validation fields;
- use safe deterministic writes.

**Tests:** idempotent rebuild, no duplicates, stable IDs, version compatibility.

**Exit:** validated manifest artifact exists.

### `P0-F3` Complete data-quality audit

**Work:**

- run source/pairing/annotation/identity/coordinate/leakage checks;
- update known-data-issue registry.

**Gate if needed:** `DG-P0-04`

**Exit:** every correctness-relevant finding is resolved, explicitly approved, or blocking.

## Stage G — Validation evidence

### `P0-G1` Resolve and run visual validation

**Gate:** `DG-P0-05`

**Work:**

- render development-data overlays;
- review representative, seeded, boundary, and flagged cases.

**Exit:** visual evidence supports coordinate/pairing interpretation.

### `P0-G2` Run trusted held-out integrity checks

**Work:**

- programmatically parse/validate test files/annotations;
- enforce §7.6 access restrictions;
- do not expose test labels to ordinary visual/model-development paths.

**Exit:** structural integrity result is recorded without search/model feedback.

### `P0-G3` Build full validation report

**Work:**

- aggregate machine-readable validation results;
- produce human-readable summary;
- include verification-register and known-issue status.

**Exit:** validation report is reviewable and reproducible.

## Stage H — Runtime loader, cache, and generic adapter

### `P0-H1` Implement model-neutral loader/API

**Work:**

- load by explicit split/modality/manifest version;
- avoid full rescans/reparsing during repeated use.

**Tests:** RGB/event/paired views, deterministic access, invalid configuration behavior.

**Exit:** canonical data can be loaded efficiently without model-specific assumptions.

### `P0-H2` Implement generic adapter contract and smoke harness

**Work:**

- define adapter boundary;
- provide generic/reference conversion/test fixture if useful;
- implement Phase 0 smoke workflow.

**Tests:** schema/interface validation; no split/label mutation; no held-out leakage.

**Exit:** downstream model adapters can be implemented in Phase 1 without redesigning Phase 0.

### `P0-H3` Select/cache runtime strategy and benchmark

**Gate:** `DG-P0-06`

**Work:**

- add only justified deterministic caches;
- benchmark loader/runtime path on intended environment;
- record settings/results.

**Exit:** reusable Phase 0 path satisfies the measured performance acceptance criterion.

## Stage I — Phase 0 freeze

### `P0-I1` Freeze artifacts/interfaces

**Work:**

- version manifest/schema/splits/configuration;
- finalize known-data-issue registry;
- finalize validation/performance reports;
- update Phase 0 README;
- document Phase 1 handoff.

### `P0-I2` Execute Phase 0 completion gate

Every mandatory gate/check in Section 15 must pass.

**Exit:** Phase 0 is frozen for Phase 1 baseline integration.

---

# 15. Phase 0 Completion Gate and Definition of Success

## 15.1 Core completion gate

Phase 0 Version 1.0 is complete only when all applicable mandatory items below are satisfied.

```text
[ ] Exact FRED repository/data identities are pinned and recorded.

[ ] Official FRED data has been acquired from a verified source.

[ ] Source data is protected from normal project writes.

[ ] A deterministic source inventory exists.

[ ] Official challenging split membership is reproduced and verified.

[ ] DG-P0-02 has produced a leakage-safe frozen project train/validation split.

[ ] Official challenging test remains outside model/LLM-GE search feedback.

[ ] Test-ground-truth access follows the protected integrity policy in §7.6.

[ ] RGB/event pairing and frame ordering are verified.

[ ] Timestamp/index and annotation-to-frame association are verified.

[ ] `coordinates.txt` semantics and coordinate interpretation are verified.

[ ] Bounding-box/annotation issues are resolved or explicitly recorded under approved policy.

[ ] Track IDs and original class/type metadata are preserved.

[ ] Stable deterministic sample IDs exist.

[ ] P0-CONTRACT-FRED-SAMPLE-v1 is implemented and documented.

[ ] RGB-only, event-only, and paired RGB/event canonical views are supported.

[ ] A versioned canonical manifest/index exists.

[ ] Manifest generation is deterministic, idempotent, and safe against partial overwrite.

[ ] Cache artifacts, if used, are rebuildable and reject incompatible versions.

[ ] Duplicate/leakage checks pass.

[ ] Malformed/questionable data is surfaced; no silent sample filtering occurs.

[ ] Known upstream FRED issues relevant to pinned data have been reviewed.

[ ] Development-data visual sanity checks support pairing/coordinate correctness.

[ ] Trusted automated held-out integrity checks pass or documented issues are resolved.

[ ] A low-cost Phase 0 smoke test passes.

[ ] Generic adapter contract and smoke harness exist.

[ ] Loader/runtime path has been benchmarked on the intended environment.

[ ] DG-P0-06 documents that no avoidable recurring Phase 0 preprocessing/I/O bottleneck remains.

[ ] A machine-readable and human-readable validation report exists.

[ ] The known-data-issue registry is complete for all identified formal-data findings.

[ ] Phase 0 artifacts/interfaces are versioned and the Phase 1 handoff is documented.
```

Any unresolved mandatory item blocks Phase 0 freeze.

Model-family-specific YOLO11, RT-DETR, and Faster R-CNN adapter-forward tests do **not** block Phase 0; they block the corresponding Phase 1 baseline integration until the exact model implementations are fixed.

## 15.2 Definition of success

Phase 0 succeeds when the project can state with evidence:

> The official FRED challenging benchmark has been acquired and reproduced without modifying source truth; project train/validation and held-out test boundaries are leakage-safe and versioned; synchronized RGB/event inputs and preserved annotations can be loaded deterministically through one validated canonical interface; malformed or questionable data is surfaced rather than hidden; later phases receive stable data and adapter contracts without reconstructing benchmark semantics; baseline and LLM-GE candidates are forced to use the same protected source of truth; held-out labels cannot enter evolutionary/model-selection feedback; and repeated model experiments reuse prevalidated/indexed data with low recurring overhead.

At that point, Phase 1 may implement its selected detector-family adapters, validate each non-evolutionary seed baseline, and only then proceed toward LLM-GE detection experiments under the Master Plan.

---

# Appendix A — Future Research (Non-Implementation Scope)

The following are explicitly outside Phase 0 Version 1.0:

- alternative event accumulation windows;
- voxel grids;
- time surfaces;
- learned event encoders;
- raw asynchronous event models;
- alternative synchronization strategies;
- event-native architectures;
- experimental storage/decoding approaches that alter data semantics;
- alternative multimodal representations.

Any such work is a separate controlled study and must compare against the frozen Version 1.0 representation rather than silently replacing it.

---

# Appendix B — Conceptual Canonical Workflows

The repository should converge toward a small number of canonical workflows rather than accumulating ad hoc scripts.

Conceptually:

```text
phase0 acquire
phase0 inventory
phase0 build-splits
phase0 build-manifest
phase0 validate
phase0 visualize
phase0 benchmark
phase0 smoke
```

These names are **non-normative examples**. Exact CLI/tooling is an implementation choice.

The important requirement is one canonical execution path per workflow rather than proliferating `*_new`, `*_fixed`, or `*_final` scripts.
