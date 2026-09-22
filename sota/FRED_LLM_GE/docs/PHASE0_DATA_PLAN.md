# Phase 0 — FRED Data Infrastructure and Input Pipeline Plan
## FRED + LLM-GE Research Project

**Status:** Version 1.1 — Implementation-Ready Remote-Access Phase Specification  
**Document path:** `docs/PHASE0_DATA_PLAN.md`  
**Phase:** Phase 0 — Data Infrastructure and Input Pipeline  
**Primary purpose:** Access, reproduce, validate, standardize, and efficiently expose FRED data for all later phases through the official remote source without making event-data representation or preprocessing itself an optimization target.

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

A Phase 0 project decision that is binding for Version 1.1.

### REQUIRED VERIFICATION

A planning-time assumption or externally derived fact that MUST be checked against the pinned repository, actual dataset, files, or runtime before implementation relies on it.

### OPEN DECISION

A decision intentionally deferred until specified implementation evidence exists. An open decision is governed by a named decision gate in Section 3. The coding agent MUST NOT silently resolve a research-sensitive open decision.

### FUTURE RESEARCH

A possible later extension outside the Version 1.1 implementation scope.

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
| Remote source access/cache | §§3–6 |
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

Version 1.1 follows:

> **Reproduce and validate before innovating.**

Phase 0 SHALL reproduce and use the released/established FRED representation and handling as faithfully as practical before any separate research effort investigates alternative event representations.

Version 1.1 SHALL NOT invent a new:

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

- official remote dataset/API access and source verification;
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

Version 1.1 does **not** include:

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

The following decisions are fixed for Phase 0 Version 1.1.

| ID | Fixed decision |
|---|---|
| `P0-D01` | Phase 0 is reproduction/validation infrastructure, not event-representation research. |
| `P0-D02` | Use official released FRED synchronized data products through the verified official remote-access mechanism/API. |
| `P0-D03` | Use released event frames for routine Version 1.1 experiments. |
| `P0-D04` | Keep raw event HDF5 content addressable in the pinned official remote source for verification/future research; do not routinely materialize or retain it locally, and do not regenerate event frames routinely. |
| `P0-D05` | Use `coordinates.txt` as the initial canonical annotation source, subject to verification. |
| `P0-D06` | Preserve timestamp, bounding box, track ID, original class/type, sequence ID, and frame identity. |
| `P0-D07` | Use one model-neutral canonical FRED layer. |
| `P0-D08` | Use thin downstream/model adapters rather than parallel dataset implementations. |
| `P0-D09` | Use the official FRED challenging split as the default project evaluation framework. |
| `P0-D10` | Create project train/validation membership only from challenging-train data. |
| `P0-D11` | Keep the official challenging test outside search/tuning feedback. |
| `P0-D12` | Split development data at a leakage-safe sequence/group level, never by random frame assignment. |
| `P0-D13` | Protect split, canonical labels, held-out labels, and benchmark definitions from LLM-GE candidate modification. |
| `P0-D14` | Perform expensive deterministic remote discovery/parsing/validation once where possible and reuse validated metadata/manifests/cache artifacts. |
| `P0-D15` | Treat the pinned official remote FRED release as immutable source truth; local materializations are disposable cache, never source truth. |
| `P0-D16` | Surface malformed/questionable data; never silently discard or repair it. |
| `P0-D17` | Preserve metadata needed by tracking and forecasting. |
| `P0-D18` | Phase 0 defines a generic adapter contract; model-family-specific adapter completion belongs to Phase 1. |
| `P0-D19` | Phase 0 MUST NOT require or create a complete local mirror of the FRED dataset; only metadata, manifests, validation artifacts, and workload-required content may be materialized locally. |

## 3.2 `DG-P0-01` — Remote source access and bounded cache strategy

**Owner:** IMPLEMENTATION  
**Trigger:** The intended execution environment can access the official FRED remote release/API.

**Evidence required:**

- exact official dataset identity and pin-able revision/version;
- verified API/client or official remote-access mechanism and its supported access granularity;
- authentication requirements, if any, without storing secrets in project configuration;
- remote file/listing/metadata behavior needed to build deterministic logical references;
- network availability, throughput, retry/error behavior, and expected provider limits;
- smallest practical materialization unit supported by the source (for example file, archive, sequence, or shard);
- available local cache capacity and permissions;
- source-reference/manifest/cache separation feasibility.

**Decision produced:**

- pinned remote dataset identity/revision;
- approved remote access mechanism/client;
- logical source-reference scheme used by manifests;
- local metadata/manifest roots;
- bounded local cache root, size policy, and eviction/reuse behavior;
- retry/failure policy for remote reads;
- any optional prefetch/materialization policy needed for efficient training.

**Constraints:**

- reusable code MUST NOT embed user-specific absolute paths;
- Phase 0 MUST NOT require or create a complete local mirror of FRED;
- locally materialized dataset content MUST be treated as disposable/rebuildable cache;
- canonical identity MUST come from the pinned remote source and stable logical references, not from a cache pathname;
- implementation MUST fail visibly when required remote content cannot be obtained or verified; it MUST NOT silently substitute another source/version.

**Blocks:** remote-source adapter implementation, source inventory, and source freeze.

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
**Trigger:** canonical manifest/loader and verified remote-source adapter exist on the intended environment.

**Evidence required:**

- remote fetch throughput and latency for representative access patterns;
- cold-cache and warm-cache loader throughput;
- cache hit/miss behavior and local cache growth;
- CPU utilization and memory behavior;
- worker/prefetch experiments;
- repeated-run startup cost;
- representative batch-delivery timing;
- provider/API limits or reliability constraints that materially affect execution.

**Decision produced:**

- bounded cache backend and capacity;
- cache key/versioning and eviction/reuse policy;
- prefetch/materialization unit and policy, if needed;
- worker/prefetch/persistent-worker settings for the environment;
- Phase 0 throughput acceptance statement supported by measurements.

Phase 0 does not need to guess a universal fixed throughput percentage before the environment is measured. It MUST avoid a design that performs a remote network request for every training sample when a bounded reusable cache/prefetch strategy can amortize those reads. Phase 1 baseline integration SHOULD additionally confirm that the chosen remote+cache loader does not materially starve the selected detector during actual training/inference.

---

# 4. Upstream FRED Evidence and Verification Register

Planning-time upstream observations are useful evidence but are not immutable facts. They MUST be rechecked against the pinned repository/data revision before implementation depends on them.

## 4.1 Planning-time references

- Official FRED repository: `https://github.com/miccunifi/FRED`
- Official FRED dataset release used during planning: `https://huggingface.co/datasets/GabrieleMagrini/FRED`
- Phase 0 Version 1.1 access policy: use the verified official remote API/client/source interface; do not require a full local dataset mirror.

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
| `P0-V01` | Official dataset/repository identities and remote revision are correct | pinned repository + pinned official remote dataset source/API | source freeze |
| `P0-V02` | Released event frames are remotely addressable/materializable for routine use | pinned remote source + on-demand content verification | canonical representation implementation |
| `P0-V03` | RGB/Event frame naming and one-to-one pairing convention | pinned repository + multiple sequences materialized/read through the remote source adapter | pairing implementation |
| `P0-V04` | `coordinates.txt` format and semantic meaning | pinned docs/repository + remotely read/materialized annotation files | strict parser |
| `P0-V05` | `coordinates.txt` uses the intended shared coordinate space | remotely obtained actual dimensions + overlays on development data | canonical box contract |
| `P0-V06` | Challenging split membership and helper behavior | pinned split lists/script | split freeze |
| `P0-V07` | Timestamp/index derivation and annotation-to-frame mapping | remotely read/materialized sequences/annotations | canonical sample freeze |
| `P0-V08` | Known upstream issue reports affect or do not affect pinned data | pinned repo/issues + project audit | Phase 0 freeze |
| `P0-V09` | Raw HDF5 objects are addressable in the pinned remote source where expected | remote listing/metadata plus targeted on-demand verification | source inventory |
| `P0-V10` | All sequences follow the same or explicitly classified pairing conventions | complete remote inventory plus evidence-backed validation/materialization as required | manifest freeze |

The verification result MUST be recorded as evidence, not only printed to console.

---

# 5. High-Level Phase 0 Architecture

## 5.1 Fixed architecture

```text
                 PINNED OFFICIAL FRED REMOTE SOURCE
                   (verified API/client/revision)
                              │
                              ▼
                    remote source adapter
                 logical refs + metadata access
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       metadata / source inventory    on-demand content fetch
                │                           │
                │                    bounded local cache
                │                    (disposable/rebuildable)
                └─────────────┬─────────────┘
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

The **pinned official remote FRED release** is the immutable source truth. The canonical Phase 0 manifest/interface is the project-facing source of truth for sample identity, split membership, annotation interpretation, and logical data references.

Local materialized images, event frames, archives, HDF5 files, or sequence bundles are cache artifacts only. Their presence or absolute location MUST NOT define scientific identity.

Models/adapters translate from the canonical representation. They do not redefine the dataset, split, annotations, canonical identity, remote revision, or cache policy.

## 5.2 Runtime principle — FIXED

Phase 0 is designed as:

> **Verify/index remotely once where possible; materialize only what is needed; reuse locally when useful.**

One-time or infrequent work includes:

```text
pin + verify official remote source/API
        ↓
remote inventory / metadata discovery
        ↓
split verification
        ↓
annotation parsing / metadata indexing
        ↓
targeted content verification
        ↓
modality/timestamp pairing validation
        ↓
manifest/index build
        ↓
artifact freeze
```

Repeated model/LLM-GE runs should primarily do:

```text
load versioned manifest/index
        ↓
resolve stable remote logical reference
        ↓
cache hit? ── yes ──► read cached content
        │
        no
        ↓
fetch smallest practical required unit
        ↓
verify + place in bounded cache
        ↓
thin adapter transformation
        ↓
model training/inference/evaluation
```

The following MUST NOT normally be repeated for every candidate:

- full-dataset download or extraction;
- whole-source remote discovery/listing when a validated reusable inventory exists;
- challenging-split construction;
- complete annotation reparsing when a validated reusable representation exists;
- full integrity scans;
- event-frame regeneration;
- synchronization reconstruction;
- train/validation membership generation;
- re-fetching content that is already valid in the compatible local cache.

The system MAY prefetch or materialize a workload-specific subset, sequence, archive, or shard when that is the smallest practical provider-supported unit and improves throughput. Such materialization remains cache, not a second source of truth.

# 6. Remote Source Access, Minimal Local State, and Immutability

## 6.1 Official remote source — FIXED

The project SHALL use the official FRED release through a verified official remote access mechanism/API/client, pinned to an exact dataset identity and revision/version wherever the provider supports revision pinning.

The FRED GitHub repository remains the benchmark/code reference. The official dataset host/release remains the data source.

Phase 0 MUST NOT require a complete local download or permanent local mirror of the approximately 205 GB FRED release.

`DG-P0-01` selects the verified remote access mechanism, logical reference scheme, and bounded cache behavior for the intended environment.

## 6.2 Remote source immutability — FIXED

The pinned official remote FRED release is read-only project truth.

Normal project code MUST NOT attempt to mutate or replace official source objects. In particular, project logic MUST NOT:

- rewrite source RGB images;
- rewrite released event frames;
- rewrite raw HDF5 content;
- edit source annotations;
- alter official sequence identities;
- alter official split-list contents;
- silently substitute another remote source/revision when an object is unavailable;
- silently delete/ignore suspect source objects from the canonical view.

Any project-created data product MUST be separate and reproducible from pinned remote source identity plus versioned project configuration/policy.

## 6.3 Minimal local state — FIXED

Local persistent state SHOULD contain only what is needed for reproducibility, validation, indexing, and efficient repeated access.

A conceptual layout is:

```text
data/
├── metadata/      # small remote inventories/source descriptors/verification records
├── derived/       # project-created small derived artifacts where needed
├── manifests/     # versioned indexes/split manifests/logical references
├── cache/         # bounded disposable materialized FRED content
└── validation/    # reports/overlays/diagnostics
```

There is intentionally no required `data/raw/` full-dataset mirror.

The exact layout is **non-normative**. Roots are configuration-driven through `DG-P0-01`.

Local cache contents MAY include provider-supported files, sequence archives, shards, extracted frames, or other materialized units needed by the current workload. Cache contents MUST remain disposable and MUST NOT be relied upon as the only copy of research-relevant source data.

## 6.4 Remote source inventory — FIXED

Before downstream use, Phase 0 MUST produce an auditable sequence/source inventory from the pinned remote source and supporting repository metadata.

For every sequence, record at least, when the remote source exposes or permits verification of the field:

- sequence ID;
- stable remote source/archive/object identity;
- pinned dataset revision/version;
- official split/source membership where available;
- RGB object/directory presence status;
- event-frame object/directory presence status;
- raw HDF5 object presence/status;
- annotation-object presence;
- RGB frame count or evidence-backed count;
- event-frame count or evidence-backed count;
- annotation entry count;
- first/last frame identity;
- first/last annotation timestamp where parseable;
- source readability/materialization status where tested;
- Phase 0 validation status.

The inventory MUST be a machine-readable durable artifact, not only terminal output.

If a field cannot be established from remote metadata alone, Phase 0 MAY materialize the smallest practical source unit required to verify it. Verification MUST NOT trigger an unconditional full-dataset mirror.

## 6.5 Remote-access and cache correctness — FIXED

Remote data access MUST be deterministic with respect to the pinned source revision and logical object reference.

The implementation MUST:

- identify the remote dataset/revision in every formal manifest;
- detect missing/unavailable remote objects and fail visibly;
- use bounded retry behavior for transient failures rather than infinite retry loops;
- verify materialized content using provider metadata, size/hash information, archive identity, or another reliable mechanism where available;
- key/invalidate cached content using enough source identity to prevent cross-version reuse;
- prevent partial downloads/materializations from being treated as valid cache entries;
- keep authentication secrets outside committed project files.

The implementation MUST NOT silently fall back to an unpinned mirror, different dataset revision, stale incompatible cache, or fabricated placeholder data.

## 6.6 Configurable remote/cache settings — FIXED

Reusable code MUST NOT contain hardcoded user/server paths such as `/home/<user>/...` or `/scratch/<username>/...`.

Environment-specific remote-access options, cache roots, and other local roots MUST come from configuration or documented environment variables.

Credentials/tokens MUST be supplied through an appropriate secret mechanism/environment and MUST NOT be committed to the repository.

# 7. Split Integrity, Leakage Prevention, and Evaluation Isolation

## 7.1 Official challenging split — FIXED

The official FRED challenging split is the default project evaluation framework.

Phase 0 MUST reproduce/verify the official challenging membership from the pinned FRED repository.

The implementation MAY use manifests, logical remote references, cached derived views, or another non-destructive representation if:

- membership is exact;
- official remote source data is not modified;
- unnecessary local materialization/data duplication is avoided;
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

- official remote FRED source or its pinned identity;
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

Version 1.1 uses the **already extracted event frames released by FRED**.

Routine training/evaluation MUST NOT regenerate the standard event-frame representation from raw HDF5 streams.

Raw HDF5 content SHALL:

- remain addressable through the pinned official remote source;
- be represented in the source inventory by stable logical references/availability metadata;
- be materialized locally only when a specific verification/future-research task requires it;
- remain excluded from ordinary Version 1.1 preprocessing and routine detector training;
- never be retained locally merely to create a complete source mirror.

A cached HDF5 object is disposable materialization, not source truth.

## 8.2 Annotation source — FIXED, subject to verification

The initial canonical annotation source is:

```text
coordinates.txt
```

subject to `P0-V04` and `P0-V05`.

`coordinates_rgb.txt` remains available for verification/future RGB-specific analysis, but it is not a parallel source of canonical truth in Version 1.1.

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
│   ├── remote_logical_reference
│   ├── optional_cache_reference
│   └── verified metadata
├── event
│   ├── remote_logical_reference
│   ├── optional_cache_reference
│   └── verified metadata
├── annotations
│   ├── boxes_xyxy
│   ├── track_ids
│   └── original_classes
├── split
│   ├── official_split
│   └── project_split
└── provenance
    ├── fred_dataset_identity
    ├── fred_dataset_revision
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
- mutable absolute/cache paths;
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
- stable RGB remote logical reference;
- stable event-frame remote logical reference;
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
- source dataset identity and pinned remote revision/version;
- FRED repository revision;
- official challenging split file revision/hash;
- project split version;
- annotation policy/version;
- creation configuration;
- creation timestamp;
- validation status.

Formal experiments reference the manifest version and pinned remote source identity rather than implicitly relying on "whatever is currently cached/on disk."

## 9.3 Safe rebuilds — FIXED

Manifest generation MUST be deterministic and idempotent.

A rebuild MUST NOT:

- append duplicates;
- mix old/new records;
- silently alter split membership;
- replace a valid final artifact with a partial one.

Important writes SHOULD use a temporary artifact, validation, then atomic promotion/rename where practical.

## 9.4 Cache policy — FIXED

Caches exist only to reduce repeated remote transfer and repeated deterministic work.

Caches are **not source truth** and MUST NOT be required to contain the full FRED release.

A cache MUST be bounded, disposable, and rebuildable from:

- the pinned official remote dataset identity/revision;
- stable logical source references;
- frozen split membership;
- Phase 0 configuration;
- versioned schema/annotation policy.

Appropriate cached material MAY include:

- workload-required RGB/event files;
- provider-supported sequence archives or shards;
- selectively materialized raw HDF5 objects for verification;
- parsed annotations;
- validated image metadata;
- deterministic indexes.

The cache SHOULD reuse compatible material across repeated baseline/LLM-GE runs and SHOULD avoid per-sample network traffic during hot training loops when practical.

Cache backend/runtime settings are selected through `DG-P0-06`.

## 9.5 Cache invalidation — FIXED

Cached artifacts MUST contain enough identity to reject incompatible reuse.

Incompatibility includes, where applicable:

- different FRED dataset identity/revision or remote object identity;
- changed annotation policy;
- changed split version;
- changed manifest/schema version;
- changed canonical coordinate interpretation;
- partial/corrupt materialization or failed source verification.

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

- official remote dataset/provider identity;
- pinned dataset revision/version;
- approved remote access mechanism/backend;
- remote logical-reference rules;
- non-secret remote-access options;
- local metadata/derived-data root;
- manifest location;
- bounded cache root and capacity/eviction policy;
- challenging-split source;
- project split manifest;
- annotation source;
- schema version;
- supported modality selection;
- legitimate validation strictness/configuration;
- runtime/prefetch/materialization settings chosen through `DG-P0-06`.

Exact configuration technology/file layout is an implementation choice.

Authentication credentials/tokens MUST NOT be committed to configuration files. They must be supplied through an appropriate environment/secret mechanism.

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
    manifest_version="...",
    source_revision="..."
)
```

Exact Python APIs are implementation-specific, but the interface must explicitly represent:

- project split;
- modality;
- manifest/schema version;
- pinned remote source identity/revision;
- training/evaluation access mode;
- downstream adapter selection where relevant.

The loader resolves canonical logical references through the approved remote-source adapter and bounded local cache. Callers MUST NOT need to know whether a sample was served from a cache hit or fetched/materialized remotely.

Behavior MUST NOT be inferred from incidental cache path structure or tensor count.

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

- required remote sequence/object identities;
- remote listing/metadata consistency;
- expected modality objects/directories/archives;
- annotation-object presence;
- HDF5 object inventory presence where expected;
- readable/non-zero content on targeted materialization checks;
- no destructive source modification;
- no silent source-version substitution.

Hashes/checksums or provider object identities SHOULD be recorded for frozen critical metadata/artifacts and materialized verification samples where practical.

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

For every usable sequence, validate using remote metadata and/or on-demand materialization as required:

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

Phase 0 is optimized for **amortized remote access and reuse**, not novel preprocessing and not a permanent full-dataset mirror.

Repeated experiments SHOULD consume already prepared/indexed metadata and reuse compatible cached data. The loader SHOULD fetch/materialize data at a granularity that avoids pathological per-frame network latency during training while still respecting the no-full-mirror requirement.

Appropriate techniques MAY include:

- precomputed manifests;
- cached parsed metadata;
- deterministic indexes;
- bounded sequence/archive/shard caching;
- workload-aware prefetching;
- dataloader workers;
- pinned host memory where appropriate;
- persistent workers where appropriate;
- avoiding repeated remote listings, transfers, re-encoding, decompression, and reparsing;
- cache warming for the specific planned workload when justified.

All optimizations MUST preserve official data semantics and pinned source identity.

## 11.2 Performance acceptance

Before Phase 0 freezes, run representative throughput benchmarks on the intended environment for at least:

- cold-cache access representative of first use;
- warm-cache access representative of repeated model/evolution runs.

The Phase 0 acceptance statement is:

> The validated reusable remote+cache data path must not contain an avoidable recurring network, I/O, or deterministic preprocessing bottleneck relative to the measured capabilities of the intended environment, while not requiring a complete local FRED mirror.

The benchmark MUST record enough context to reproduce the conclusion, including as applicable:

- server/environment identity;
- remote source/provider and pinned revision;
- cache location/capacity and current warm/cold state;
- manifest/cache version;
- batch/sample configuration;
- materialization/prefetch unit;
- worker/prefetch settings;
- warm-up handling;
- measured remote transfer and sample/batch delivery rate;
- cache hit/miss behavior;
- startup/repeated-run cost;
- CPU/memory behavior.

`DG-P0-06` selects environment-specific settings and documents whether the pipeline satisfies the criterion.

Because exact detector implementations belong to Phase 1, Phase 1 baseline integration SHOULD confirm end-to-end that the remote+cache data path does not materially starve the selected detector during representative execution.

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
- regenerate the Version 1.1 event representation;
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

1. **Remote source inventory** — complete auditable pinned-source/sequence/object inventory and access metadata.
2. **Official split manifest** — verified challenging-split membership.
3. **Project split manifest** — frozen development train/validation membership and reference to held-out test membership.
4. **Canonical dataset manifest/index** — sample-level model-neutral index.
5. **Known-data-issue registry** — all relevant findings and approved handling.
6. **Validation report** — machine-readable results plus human-readable summary.
7. **Visual validation outputs** — development-data overlays selected under `DG-P0-05`.
8. **Schema/interface documentation** — canonical sample and generic adapter contracts.
9. **Performance report** — representative remote-access, cold/warm-cache, and loader throughput measurements.
10. **Verification register results** — `P0-V01`–`P0-V10` evidence/status.

Dataset content and heavy caches remain outside Git. Git SHOULD contain code, non-secret configuration, small manifests where practical, schema definitions, validation summaries, and durable references to the pinned official remote source. A complete local FRED mirror is not a Phase 0 artifact.

## 13.2 Data-use auditability

A formal downstream run must be able to identify by reference:

- manifest version;
- project split;
- sample IDs/membership version;
- modality;
- downstream adapter/config;
- Phase 0 schema version;
- pinned remote FRED identity/revision;
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
- remote-access/cache runtime guidance.

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

## Stage A — Evidence, remote access, and source abstraction

### `P0-A1` Pin and verify upstream references

**Work:**

- pin FRED repository revision;
- identify and pin the exact official remote dataset release/source revision;
- begin `P0-V01`–`P0-V10` verification record.

**Exit:** upstream revisions are recorded and planning-time assumptions are explicitly marked verified/pending.

### `P0-A2` Resolve remote access and bounded cache strategy

**Gate:** `DG-P0-01`

**Work:**

- verify the official API/client/remote-access mechanism in the intended environment;
- record authentication/network/provider constraints;
- define logical source references;
- configure metadata/derived/manifest/cache roots;
- define bounded cache capacity and invalidation/eviction behavior;
- explicitly verify that no full local dataset mirror is required.

**Exit:** the pinned official remote source can be queried and a representative object can be materialized/read without creating a full dataset copy.

### `P0-A3` Implement remote source adapter and cache boundary

**Work:**

- implement source listing/metadata access required by Phase 0;
- implement deterministic resolution from logical reference to remote object;
- implement safe on-demand materialization through the bounded cache;
- implement partial-download protection, source-version checks, and visible failure behavior.

**Tests:** deterministic logical-reference resolution; cache hit/miss behavior; stale-version rejection; interrupted materialization handling; no source mutation.

**Exit:** downstream Phase 0 code can request remote FRED objects without depending directly on provider-specific paths or assuming a complete local dataset.

## Stage B — Remote inventory and initial audit

### `P0-B1` Build remote source inventory

**Work:**

- discover sequences/objects through the remote source adapter and pinned repository metadata;
- collect counts/presence/source-identity metadata without unconditional full materialization;
- record HDF5/event-frame/annotation availability;
- selectively materialize only the smallest practical units needed to verify fields unavailable from metadata.

**Tests:** deterministic rerun; duplicate sequence/object detection; pinned-revision identity; no full-mirror side effect; no source modification.

**Exit:** versioned remote source inventory artifact exists.

### `P0-B2` Review upstream/data-quality risk

**Work:**

- compare remote inventory/findings with relevant pinned upstream issues;
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

**Tests:** full-sequence logical count/order checks plus on-demand materialized boundary/content checks, missing/duplicate/gap cases.

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

- run remote-source/pairing/annotation/identity/coordinate/leakage checks;
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

- load by explicit split/modality/manifest version/pinned source revision;
- resolve remote logical references through the source adapter/cache;
- avoid full remote rescans/reparsing during repeated use;
- keep cache behavior transparent to model-facing callers.

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

- add only justified bounded deterministic caches/prefetching;
- benchmark cold-cache and warm-cache loader/runtime paths on intended environment;
- verify the hot path does not depend on one remote request per training sample when avoidable;
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

Phase 0 Version 1.1 is complete only when all applicable mandatory items below are satisfied.

```text
[ ] Exact FRED repository/data identities are pinned and recorded.

[ ] The official remote FRED dataset identity/revision and approved API/access mechanism are verified.

[ ] Phase 0 does not require or create a complete local FRED mirror.

[ ] Official remote source truth is protected from project writes/substitution.

[ ] A deterministic remote source inventory exists.

[ ] A bounded disposable cache/materialization strategy is implemented and version-aware.

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

[ ] Cache/materialized artifacts are bounded, rebuildable from the pinned remote source, protected against partial writes, and reject incompatible source/schema versions.

[ ] Duplicate/leakage checks pass.

[ ] Malformed/questionable data is surfaced; no silent sample filtering occurs.

[ ] Known upstream FRED issues relevant to pinned data have been reviewed.

[ ] Development-data visual sanity checks support pairing/coordinate correctness.

[ ] Trusted automated held-out integrity checks pass or documented issues are resolved.

[ ] A low-cost Phase 0 smoke test passes.

[ ] Generic adapter contract and smoke harness exist.

[ ] Cold-cache and warm-cache remote+loader paths have been benchmarked on the intended environment.

[ ] DG-P0-06 documents that no avoidable recurring remote-transfer, preprocessing, or I/O bottleneck remains and that the design does not depend on a full local dataset mirror.

[ ] A machine-readable and human-readable validation report exists.

[ ] The known-data-issue registry is complete for all identified formal-data findings.

[ ] Phase 0 artifacts/interfaces are versioned and the Phase 1 handoff is documented.
```

Any unresolved mandatory item blocks Phase 0 freeze.

Model-family-specific YOLO11, RT-DETR, and Faster R-CNN adapter-forward tests do **not** block Phase 0; they block the corresponding Phase 1 baseline integration until the exact model implementations are fixed.

## 15.2 Definition of success

Phase 0 succeeds when the project can state with evidence:

> The official FRED challenging benchmark is accessed from a pinned verified remote source without requiring a complete local dataset mirror; project train/validation and held-out test boundaries are leakage-safe and versioned; synchronized RGB/event inputs and preserved annotations can be resolved deterministically through one validated canonical interface; only workload-required content is materialized through a bounded disposable cache; malformed or questionable data is surfaced rather than hidden; later phases receive stable data and adapter contracts without reconstructing benchmark semantics; baseline and LLM-GE candidates are forced to use the same protected source of truth; held-out labels cannot enter evolutionary/model-selection feedback; and repeated model experiments reuse prevalidated metadata/manifests and compatible cached content with low recurring overhead.

At that point, Phase 1 may implement its selected detector-family adapters, validate each non-evolutionary seed baseline, and only then proceed toward LLM-GE detection experiments under the Master Plan.

---

# Appendix A — Future Research (Non-Implementation Scope)

The following are explicitly outside Phase 0 Version 1.1:

- alternative event accumulation windows;
- voxel grids;
- time surfaces;
- learned event encoders;
- raw asynchronous event models;
- alternative synchronization strategies;
- event-native architectures;
- experimental storage/decoding approaches that alter data semantics;
- alternative multimodal representations.

Any such work is a separate controlled study and must compare against the frozen Version 1.1 representation rather than silently replacing it.

---

# Appendix B — Conceptual Canonical Workflows

The repository should converge toward a small number of canonical workflows rather than accumulating ad hoc scripts.

Conceptually:

```text
phase0 verify-source
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
