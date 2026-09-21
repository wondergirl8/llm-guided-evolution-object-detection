# Phase 1 — Drone Detection Planning Document
## FRED + LLM-GE Research Project

**Status:** Version 1.0 — Implementation-Ready Phase Specification  
**Document path:** `docs/PHASE1_DETECTION_PLAN.md`  
**Phase:** Phase 1 — Drone Detection  
**Primary purpose:** Reproduce trustworthy FRED detector baselines, define controlled detector-family search spaces, adapt LLM-GE to object-detection architecture evolution, run family-specific Event-based evolutionary studies, preserve elite detectors, and freeze a versioned detection interface for Phase 2.

> **Implementation note:** This document defines the Phase 1 technical contract and one canonical execution path. It does not authorize a coding agent to implement the entire phase in one task. Non-trivial implementation work MUST target one named Phase 1 work package or a small explicitly coupled set of work packages.

---

# 1. Role, Authority, and How to Use This Plan

## 1.1 Phase authority

This document is the **phase-specific technical authority for Phase 1 — Drone Detection**.

It specializes, but does not replace or silently override:

- root-level `AGENTS.md`;
- `docs/MASTER_PROJECT_PLAN.md`;
- `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md`;
- `docs/PHASE0_DATA_PLAN.md`.

Authority is divided by domain:

- the **Master Plan** owns project-wide research architecture, LLM-GE policy, benchmark/evaluation policy, reproducibility policy, cross-phase interfaces, and project-wide experimental principles;
- the **Engineering Implementation Standard** owns software-engineering behavior, testing, state safety, external dependency verification, failure handling, and coding-agent completion requirements;
- the **Phase 0 Plan** owns the canonical FRED data representation, manifests, split integrity, leakage controls, and generic data/adapter contract;
- this **Phase 1 Plan** owns detector-family integration, detector baseline validation, architecture-evolution boundaries, candidate validation, detection fitness/search behavior, elite handling, and Phase 1 outputs;
- a **current implementation task** owns only its immediate scope;
- the **repository** is evidence of current implementation state, not a source of policy.

This plan MUST NOT silently override fixed decisions in the Master Plan, Phase 0 Plan, or an applicable Engineering Standard `MUST`.

If governing documents and repository behavior materially disagree, the coding agent MUST surface plan/implementation drift rather than resolve it by guesswork.

## 1.2 Decision-status language

This document uses four decision statuses.

### FIXED

A Phase 1 project decision accepted for Version 1.0.

### REQUIRED VERIFICATION

A planning-time assumption, upstream behavior, or externally derived fact that MUST be checked against the pinned implementation before the project relies on it.

### OPEN DECISION

A decision intentionally deferred until a named Phase 1 decision gate has the required evidence.

### FUTURE / EXPANSION RESEARCH

A potentially valuable later extension outside the first controlled Phase 1 implementation and experiment.

No coding agent may silently resolve an `OPEN DECISION` because an implementation shortcut happens to be convenient.

## 1.3 Decision status versus normative severity

Decision status and normative severity are separate.

- `FIXED` means the project decision is settled.
- `MUST` / `MUST NOT` define mandatory implementation behavior.
- `SHOULD` / `SHOULD NOT` define the expected default when applicable.
- `MAY` defines an allowed option.

A recommended implementation may still be pending a gate. A **recommended seed default is not implementation authorization** until its corresponding seed-selection gate has a recorded resolution.

## 1.4 Progressive-disclosure rule

The coding agent SHOULD NOT reload or reason over every Phase 1 section for every task.

For a non-trivial implementation task, use:

```text
AGENTS.md
    ↓
relevant Master Plan / Engineering Standard rules
    ↓
relevant Phase 0 interface sections
    ↓
relevant Phase 1 sections from the navigation matrix
    ↓
current Phase 1 work-package task brief
    ↓
relevant repository code
```

`references.md` and the future/expansion appendices are **not ordinary implementation context**. They are read only for an explicitly approved research/expansion task.

## 1.5 Navigation matrix

| Current work | Primary Phase 1 sections |
|---|---|
| Phase 0 handoff | §§2, 5, 7 |
| Seed-family selection | §§3–5, 6 |
| Model-family adapters | §§5–8 |
| Bring-up/resource profiling | §§6–9 |
| Formal baseline reproduction | §§8–10 |
| Training/evaluation protocol | §§4, 9–10 |
| YOLO11 genome | §11 |
| RT-DETR genome | §11 |
| Faster R-CNN genome | §11 |
| LLM-GE integration | §§4, 12–14 |
| Candidate validity/failure | §13 |
| Pilot evolution | §§14–15 |
| Primary Event evolution | §15 |
| Elite/final evaluation | §16 |
| Phase 2 handoff | §17 |
| Implementation roadmap | §18 |
| Completion gate | §19 |

For non-trivial work, the coding agent SHOULD use the short pre-task contract required by `AGENTS.md` and identify the relevant work-package ID from Section 18.

---

# 2. Phase Goal, Research Objective, Scope, and Non-Goals

## 2.1 Phase goal — FIXED

Phase 1 SHALL build a trustworthy and reproducible drone-detection research system that:

- consumes only the frozen Phase 0 FRED interface;
- validates at least three established detector families:
  - YOLO11;
  - RT-DETR;
  - Faster R-CNN;
- reproduces RGB and Event baselines for each selected seed family;
- adapts LLM-GE to architecture evolution without allowing the evolutionary process to redefine the benchmark;
- runs separate family-specific populations;
- initially evolves detector architecture under a frozen training/evaluation protocol;
- preserves multiple elite detectors instead of one premature winner;
- exports versioned detections and detector metadata for Phase 2 tracking.

## 2.2 Primary research question — FIXED

The first controlled LLM-GE study asks:

> **Can LLM-GE improve validated detector architectures on the FRED challenging setting when data representation, modality, training protocol, and evaluation protocol are held fixed?**

The first controlled study therefore targets:

```text
architecture evolution
while holding
data + modality + training + evaluation fixed
```

rather than jointly evolving every part of the detector/training system.

## 2.3 Secondary objectives

Phase 1 SHOULD additionally gather evidence about:

- whether different detector families benefit from different architectural changes;
- whether tiny-drone localization can improve without unacceptable compute growth;
- whether high-performing detection elites differ from elites useful to tracking;
- whether modern post-FRED ideas later outperform/complement LLM-GE-discovered changes;
- whether custom seeds later provide useful architectural diversity.

These objectives MUST NOT broaden the first controlled implementation/search before the core Phase 1 path is stable.

## 2.4 In scope for the core Phase 1 implementation

- Phase 0 handoff verification;
- exact seed-family implementation selection;
- RGB and Event detector adapters;
- model/evaluator bring-up and resource profiling;
- frozen baseline/evolution protocol;
- six non-evolutionary baseline tracks;
- baseline reproduction classification;
- detection metric implementation/verification;
- detector-family semantic genome design;
- candidate schema/build validation;
- model-build/tensor-shape validation;
- LLM-GE mutation/crossover/EoT integration;
- family-specific populations;
- candidate training/evaluation;
- failure classification;
- lineage/artifact tracking;
- resource measurement;
- architecture-only pilots;
- primary Event-based evolutionary studies;
- elite archives;
- full-budget elite confirmation;
- frozen held-out evaluation;
- Phase 2 detector prediction interface.

## 2.5 Out of scope for the core implementation

The core Phase 1 Version 1.0 study SHALL NOT include:

- changing the Phase 0 event representation;
- raw-event voxel/point-cloud processing;
- changing challenging split membership;
- changing project train/validation membership;
- test-set tuning;
- unrestricted edits to external detector repositories;
- cross-family crossover among YOLO11, RT-DETR, and Faster R-CNN;
- jointly evolving optimizer, loss, augmentation, architecture, and data representation from the first generation;
- tracking logic;
- trajectory forecasting;
- mandatory multimodal RGB/Event fusion;
- modern/custom-seed expansion before the core Phase 1 freeze.

Future/expansion studies are collected in Appendix A and do not block core Phase 1 completion.

---

# 3. Fixed Phase 1 Decisions and Recommended Seed Defaults

## 3.1 Fixed decision register

| ID | Decision |
|---|---|
| `P1-D01` | Phase 1 uses YOLO11, RT-DETR, and Faster R-CNN as mandatory established seed families. |
| `P1-D02` | Phase 1 consumes the frozen Phase 0 canonical dataset and MUST NOT reimplement FRED preprocessing. |
| `P1-D03` | Each selected seed must work non-evolutionarily before LLM-GE may modify it. |
| `P1-D04` | RGB and Event are separate baseline input tracks, producing six mandatory seed/modality baselines. |
| `P1-D05` | The first primary LLM-GE studies use Event input for YOLO11, RT-DETR, and Faster R-CNN separately. |
| `P1-D06` | Initial populations remain family-specific; no cross-family crossover. |
| `P1-D07` | The first LLM-GE study primarily evolves architecture while training/evaluation protocol is protected. |
| `P1-D08` | LLM-GE does not edit whole external detector repositories; it edits project-owned controlled candidate representations. |
| `P1-D09` | Initial candidate genomes contain semantically meaningful detector subsystems rather than arbitrary source-code chunks. |
| `P1-D10` | The official challenging test set is excluded from evolutionary fitness/search feedback. |
| `P1-D11` | Planned primary search fitness is project-validation `mAP50:95`, subject to metric-verification gate. |
| `P1-D12` | Resource statistics are recorded for every valid candidate and may act as eligibility guardrails. |
| `P1-D13` | Multiple elites are preserved per family; Phase 1 does not reduce the project to one detector prematurely. |
| `P1-D14` | Modern detector literature informs later research/search-space design but does not automatically replace the FRED-comparable seeds. |
| `P1-D15` | Primary Phase 1 detection is a single-class `drone` task. Original FRED type labels remain preserved in Phase 0. |
| `P1-D16` | Primary seed comparison uses COCO-only detector pretraining where an official compatible checkpoint exists, unless evidence-backed reproduction requires another public initialization. |
| `P1-D17` | Candidate initialization follows one project-owned loading contract: shape-compatible unchanged parameters may load from the canonical allowed checkpoint; new/incompatible parameters use deterministic framework-standard initialization and are reported explicitly. |
| `P1-D18` | Search-time candidates are normally evaluated with one controlled training seed; promoted baselines/elites are confirmed across multiple independent seeds before reportable conclusions. |
| `P1-D19` | Published FRED detector scores are external reference values, not tuning targets. |
| `P1-D20` | FRED-derived architecture statistics such as anchor priors may use project-training data only; official test ground truth MUST NOT influence design. |
| `P1-D21` | Modern/custom models remain separate post-core expansion studies unless explicitly promoted through the expansion gate. |

## 3.2 Recommended seed defaults pending gates

| Family | Recommended default | Rationale |
|---|---|---|
| YOLO11 | **YOLO11m**, official Ultralytics implementation | Middle-capacity detector intended to balance difficult detection capacity against repeated evolutionary cost. |
| RT-DETR | **original RT-DETR-R50**, official `lyuwenyu/RT-DETR` PyTorch implementation | Canonical original RT-DETR family anchor with a clear backbone/encoder/decoder mutation boundary. |
| Faster R-CNN | **Faster R-CNN R50-FPN in MMDetection** | Standard two-stage R50-FPN design with declarative FPN/RPN/anchor/RoI configuration suitable for semantic genomes. |

These recommendations are not claims about the exact unpublished FRED configurations.

A coding agent MUST NOT:

- treat a recommended default as finalized before its decision gate;
- choose a different seed merely because it scores better on held-out FRED test data;
- begin formal baseline/evolution implementation against an unresolved seed selection.

If exact FRED details remain unavailable after evidence review, the selected configuration SHALL be labeled `BEST-SUPPORTED-REPRODUCTION` or `PROJECT-BASELINE` rather than guessed to be an exact published reproduction.

---

# 4. Decision Gates and Verification Register

## 4.1 Gate ownership

Each gate is owned by either:

- **PROJECT/RESEARCH** — the coding agent may collect evidence and recommend options but MUST NOT finalize the decision without explicit project approval;
- **IMPLEMENTATION** — the coding agent MAY resolve the gate after collecting required evidence and MUST record the decision/rationale.

## 4.2 `DG-P1-01` — Confirm YOLO11 seed

**Owner:** PROJECT/RESEARCH  
**Recommended default:** YOLO11m from official Ultralytics.  
**Trigger:** FRED evidence, official YOLO11 configuration/checkpoints, and intended hardware have been reviewed.

**Evidence:**

- recoverable FRED scale/config evidence;
- official scale/config details;
- COCO-pretrained checkpoint availability;
- resource profile;
- repeated-candidate feasibility;
- reproduction rationale independent of held-out-score tuning.

**Produces:**

- selected YOLO11 seed;
- pinned Ultralytics revision/config;
- checkpoint origin;
- reproduction classification expectation.

## 4.3 `DG-P1-02` — Confirm RT-DETR seed

**Owner:** PROJECT/RESEARCH  
**Recommended default:** original RT-DETR-R50 from official `lyuwenyu/RT-DETR`.  
**Documented compute fallback:** RT-DETR-R50-m only if measured constraints make R50 impractical.

**Evidence:**

- recoverable FRED variant evidence;
- official R50/R50-m configuration;
- COCO-pretrained state;
- input/multiscale behavior;
- compute/memory;
- repeated-candidate feasibility.

**Produces:** selected/pinned repository/config/checkpoint and initialization policy.

Modern descendants such as RT-DETRv2/v4, D-FINE, and DEIM are not substitutes for the initial original RT-DETR anchor.

## 4.4 `DG-P1-03` — Confirm Faster R-CNN implementation

**Owner:** PROJECT/RESEARCH  
**Recommended default:** MMDetection Faster R-CNN R50-FPN.  
**Fallback:** Torchvision R50-FPN if MMDetection dependency/integration burden materially harms reproducibility/maintainability.

**Evidence:**

- recoverable FRED framework/backbone evidence;
- reproduction fidelity;
- Phase 0 integration;
- semantic architecture configurability;
- dependency/license burden;
- compute footprint;
- coding-agent maintainability.

**Produces:** selected framework/backbone/config/checkpoint/revision.

## 4.5 `DG-P1-04` — Freeze controlled training protocol

**Owner:** PROJECT/RESEARCH  
**Trigger:** selected family adapters can construct models and complete minimal trusted train/eval bring-up; initial resource/metric evidence exists.

**Evidence:**

- recoverable FRED training details;
- seed-framework official recommendations;
- comparable-budget analysis;
- measured compute;
- optimizer/LR behavior;
- augmentation behavior;
- stopping/checkpoint behavior;
- initial seed variability evidence where available.

**Produces:**

- full baseline/evolution training budget;
- optimizer/scheduler;
- augmentation;
- primary pretraining policy;
- candidate initialization/loading contract;
- search-time seed policy;
- multi-seed confirmation policy;
- checkpoint policy.

Architecture-only evolution may not change these settings without a separately authorized study.

## 4.6 `DG-P1-05` — Candidate resource envelope

**Owner:** PROJECT/RESEARCH  
**Trigger:** selected seeds have measured parameters/compute/memory/latency/training cost.

**Produces:**

- hard memory limit;
- size/compute eligibility limits;
- timeout policy;
- absolute and/or seed-relative limits.

No arbitrary numeric multiplier is fixed before seed/hardware evidence exists.

## 4.7 `DG-P1-06` — Exact validation fitness/evaluation contract

**Owner:** PROJECT/RESEARCH  
**Trigger:** evaluator/postprocessing implementation has been verified during seed bring-up.

**Evidence:** correct/reproducible `mAP50`, `mAP50:95`, postprocessing behavior, coordinate/label compatibility.

**Produces:** primary fitness, secondary metrics, tie-breaking policy.

**Planned default:** project-validation `mAP50:95`.

## 4.8 `DG-P1-07` — Final family genome ranges

**Owner:** PROJECT/RESEARCH  
**Trigger:** seed configs are pinned and prototype genome/build validation has established legal dependency constraints.

**Evidence:**

- legal dimensions/modules;
- interface compatibility;
- pretrained-weight implications;
- resource envelope;
- prototype candidate builds;
- invalid-combination evidence.

**Produces:** approved values/ranges, dependencies, invalid-combination rules, mutation-magnitude policy.

## 4.9 `DG-P1-08` — Evolution execution configuration

**Owner:** PROJECT/RESEARCH  
**Trigger:** family genome/builders and LLM-GE integration smoke tests are working.

**Evidence:**

- project-wide LLM-GE constraints;
- pilot compute budget;
- LLM provider/model availability;
- operator behavior;
- persistence/retry behavior;
- expected per-candidate cost.

**Produces, separately for pilot and formal runs where appropriate:**

- population size;
- generation count;
- mutation/crossover rates;
- EoT/feedback frequency;
- LLM provider/model and prompt-set version;
- retry/failure-budget policy;
- parallel-execution limits;
- run-state/checkpoint cadence.

This gate prevents an implementation agent from inventing evolutionary execution parameters.

## 4.10 `DG-P1-09` — Optional multi-fidelity screening

**Owner:** PROJECT/RESEARCH  
**Trigger:** representative candidates have matched short/full-budget evidence.

**Evidence:** rank correlation, false rejection/promotion behavior, compute saved, family-specific stability.

**Produces:** either:

- an approved screening/promotion contract; or
- an explicit decision not to use multi-fidelity screening.

Multi-fidelity is optional and is not on the default critical path.

## 4.11 `DG-P1-10` — Post-core modern/custom expansion

**Owner:** PROJECT/RESEARCH  
**Trigger:** the core Phase 1 study has frozen successfully.

**Produces:** explicitly authorized post-core experiments, if any.

Expansion does not alter the already frozen core Phase 1 protocol.

## 4.12 Verification register

Planning-time upstream/API observations MUST be rechecked against pinned implementations.

| ID | Required verification | Blocks |
|---|---|---|
| `P1-V01` | Frozen Phase 0 schema/manifest/split/adapter contract matches repository/runtime | all formal Phase 1 work |
| `P1-V02` | Selected Ultralytics revision/config/checkpoint API and YOLO architecture boundaries | YOLO adapter/genome |
| `P1-V03` | Selected RT-DETR revision/config/checkpoint API and architecture boundaries | RT-DETR adapter/genome |
| `P1-V04` | Selected Faster R-CNN framework/config/checkpoint/component boundaries | FRCNN adapter/genome |
| `P1-V05` | Primary single-class target mapping is identical across all family adapters | formal baselines |
| `P1-V06` | Evaluator/postprocessing metric semantics are correct on trusted development data | `DG-P1-06`, formal baselines |
| `P1-V07` | Partial checkpoint loading and deterministic new-parameter initialization work as specified for each family | formal candidate evolution |
| `P1-V08` | Seed resource profiles are measured on intended environment | `DG-P1-04`, `DG-P1-05` |
| `P1-V09` | Pinned LLM-GE branch control flow, candidate representation, persistence, fitness ingestion, and failure behavior are traced from actual code | LLM-GE adaptation |
| `P1-V10` | Candidate/output artifact paths and stale-result handling in LLM-GE are understood and testable | formal evolution |
| `P1-V11` | Prompt/version integration points in the selected LLM-GE implementation are verified | pilot evolution |
| `P1-V12` | Phase 2-required detection fields can be exported without detector-internal coupling | Phase 1 freeze |

Verification evidence SHALL be preserved as project artifacts/records rather than existing only in agent reasoning.

---

# 5. Phase 0 Handoff and Protected Data Contract

## 5.1 Required Phase 0 inputs

Phase 1 SHALL consume by versioned reference:

- frozen Phase 0 manifest/index;
- project train membership;
- project validation membership;
- official challenging-test identity;
- RGB view;
- Event view;
- paired RGB/Event view where later explicitly required;
- canonical boxes;
- preserved classes/track IDs/timestamps;
- Phase 0 validation report;
- known-data-issue registry;
- generic adapter contract;
- runtime-loading guidance.

Phase 1 SHALL NOT reconstruct:

- FRED synchronization;
- challenge-split membership;
- coordinate interpretation;
- event-frame generation;
- sample identity;
- annotation parsing.

## 5.2 Handoff validation

Before formal model work:

- verify Phase 0 schema version;
- verify manifest version;
- verify split version;
- verify annotation policy;
- verify known-data-issue handling;
- verify generic adapter expectations against actual code.

A mismatch blocks formal Phase 1 work.

## 5.3 Protected data/evaluation invariants

Phase 1 model/candidate code MUST NOT:

- modify Phase 0 source/manifests;
- alter train/validation/test membership;
- receive official-test labels during search;
- redefine canonical coordinates/annotations;
- redefine the official evaluator to improve a candidate score.

Published FRED held-out scores are post-hoc external references, not optimization targets.

---

# 6. Seed Selection and Family Bring-Up Contract

## 6.1 Seed authorization

No formal family adapter/baseline implementation is pinned until its seed-selection gate is resolved.

After a gate resolution, record:

- exact repository/framework revision;
- exact model/config;
- checkpoint source;
- license/dependency implications;
- reproduction category expectation;
- approved initialization source.

## 6.2 Mandatory baseline matrix — FIXED

| Family | RGB | Event |
|---|---|---|
| YOLO11 | `YOLO11-RGB` | `YOLO11-Event` |
| RT-DETR | `RTDETR-RGB` | `RTDETR-Event` |
| Faster R-CNN | `FRCNN-RGB` | `FRCNN-Event` |

These are six baseline systems, not six datasets.

## 6.3 Primary detection-class policy — FIXED

Primary Phase 1 detection uses exactly one semantic class:

```text
drone
```

The Phase 1 adapter SHALL map valid FRED drone instances consistently to this class across all three families and both RGB/Event views.

Phase 0 remains the source of preserved original drone-type labels.

## 6.4 Bring-up versus formal baseline

A **bring-up run** is not a reportable baseline.

Bring-up exists only to verify:

- model construction;
- adapter correctness;
- minimal forward/backward path;
- loss finiteness;
- evaluator/postprocessing behavior;
- resource feasibility;
- checkpoint loading;
- artifact/log output.

It MAY use a tiny deterministic project-training/development subset and short execution.

Formal baseline execution is prohibited until `DG-P1-04`, `DG-P1-05`, and `DG-P1-06` are resolved.

This distinction prevents implementation smoke testing from silently becoming the experimental protocol.

---

# 7. Model-Family Adapter Contract

## 7.1 Adapter responsibility

Each selected family receives a thin project-owned adapter:

```text
frozen Phase 0 sample
        ↓
family adapter
        ↓
official/framework-required input + target
        ↓
selected seed model
```

Adapters MAY perform only the transformations authorized by the frozen Phase 0 / Phase 1 protocol, such as:

- resizing;
- normalization;
- family-specific target encoding;
- collate/batch formatting;
- approved training augmentation after `DG-P1-04`.

Adapters MUST NOT:

- alter split membership;
- mutate Phase 0 canonical labels;
- hide invalid samples;
- inject test information;
- independently redefine class mapping;
- silently change evaluation coordinate semantics.

## 7.2 Adapter verification

For each family and modality, verify:

- input shape/dtype/range;
- box coordinate conversion;
- single-class mapping;
- empty-target behavior if applicable;
- batch/collate semantics;
- device transfer;
- deterministic target conversion where expected;
- no source/manifests mutation;
- minimal model forward path;
- minimal supervised loss path.

## 7.3 External repository boundary

Where practical, selected detector repositories/packages are treated as pinned external dependencies.

Project-specific integration SHOULD live in project-owned:

- configuration;
- adapters;
- candidate builders;
- wrappers/hooks that use supported extension points.

The coding agent MUST NOT patch unrelated external-repository internals merely because doing so is expedient.

An unavoidable upstream patch requires an explicit task scope, rationale, regression tests, and recorded divergence from the pinned upstream version.

---

# 8. Baseline Reproduction Contract

## 8.1 Baseline-before-evolution invariant

No detector family may enter formal LLM-GE evolution until its selected seed:

- constructs successfully;
- consumes Phase 0 inputs;
- trains under the frozen protocol;
- evaluates under the verified evaluator;
- produces reproducible metrics/artifacts;
- satisfies baseline-validation evidence.

## 8.2 Reproduction categories

Every baseline result SHALL be labeled:

### `PUBLISHED-REPRODUCTION`

Sufficient evidence exists that model/config/training/evaluation materially match the published FRED setup.

### `BEST-SUPPORTED-REPRODUCTION`

Some original details remain unavailable; the strongest available evidence is used and deviations are documented.

### `PROJECT-BASELINE`

The detector family is used in a deliberately selected project configuration that is not represented as the exact published setup.

## 8.3 Published-score non-target rule

If the original recipe cannot be recovered, the project MUST NOT tune:

- detector scale/backbone;
- epochs;
- learning rate;
- augmentation;
- preprocessing;
- anchor priors;
- postprocessing

merely until an official held-out score resembles a published FRED number.

Procedure:

```text
recover recipe if authoritative evidence exists
        ↓
otherwise freeze evidence-based project recipe
        ↓
verify protocol on project train/validation
        ↓
run formal baselines
        ↓
freeze model-selection decisions
        ↓
held-out evaluation only when allowed
        ↓
compare published values post hoc
```

## 8.4 Baseline evidence package

Each formal baseline preserves:

- family/config/revision;
- checkpoint origin;
- modality;
- Phase 0 versions;
- adapter version;
- training config;
- optimizer/scheduler;
- input resolution;
- augmentation;
- random seed;
- initialization/loading report;
- resource use;
- logs/checkpoint;
- validation metrics;
- held-out metrics only when permitted;
- development-data qualitative predictions;
- reproduction category/deviations;
- failures/limitations.

---

# 9. Frozen Controlled Experiment Protocol

## 9.1 Baseline reproduction versus evolution protocol

Two concepts are distinguished:

### Baseline Reproduction Contract

Attempts to reproduce the best-supported published family setup and records unavoidable deviations.

### Controlled Evolution Contract

Defines one frozen, fair architecture-evolution protocol used within each formal family study.

If reproduction evidence requires family-specific details, those differences are documented. The coding agent MUST NOT silently copy a family-specific reproduction quirk into another family or into the evolution protocol.

## 9.2 Variables protected during architecture evolution

After `DG-P1-04`–`DG-P1-06`, protect:

- Phase 0 data;
- modality for the current study;
- train/validation membership;
- metric implementation;
- training duration/budget;
- optimizer/settings;
- LR scheduler;
- augmentation;
- model-specific preprocessing;
- random-seed policy;
- checkpoint policy;
- evaluator;
- resource caps;
- confidence/NMS evaluation policy where applicable;
- experiment logging/provenance.

## 9.3 Evolvable variable

The principal evolvable variable is:

```text
detector architecture
```

represented by a bounded family-specific semantic genome.

## 9.4 Primary pretraining policy

For the controlled primary study, selected seeds SHOULD begin from the official compatible COCO-pretrained detector checkpoint for their selected implementation.

The primary comparison SHALL NOT silently give one family substantially stronger external supervision such as Objects365/VFM/proprietary initialization.

Any evidence-backed reproduction exception is documented through the seed/training gates.

## 9.5 Search-time versus report-time randomness

Default policy:

```text
search-time candidate
    → one controlled training seed

promoted baseline / elite
    → multiple independent full-budget seeds

reportable conclusion
    → aggregate performance + variability
```

The exact number of confirmation seeds is resolved through `DG-P1-04`; at least three is preferred when feasible.

An apparent gain smaller than normal training variability must not be represented as strong evidence of architectural improvement.

## 9.6 Candidate initialization/loading contract

The reusable Phase 1 contract is:

```text
unchanged + shape-compatible parameter/module
        ↓
load canonical allowed checkpoint value

new or shape-incompatible parameter/module
        ↓
deterministic framework-standard initialization
```

For every candidate record:

- checkpoint identity;
- loaded keys/modules;
- missing keys;
- unexpected keys;
- shape mismatches;
- newly initialized parameters;
- initialization method/seed.

Candidate code MUST NOT invent silent partial-loading behavior.

`P1-V07` MUST pass before formal evolution.

---

# 10. Detection Evaluation and Resource Contract

## 10.1 Required metrics

Verify at least:

- `mAP50`;
- `mAP50:95`.

Planned primary fitness:

```text
project-validation mAP50:95
```

subject to `DG-P1-06`.

Secondary metrics MAY include:

- precision;
- recall;
- false positives/negatives;
- parameters;
- FLOPs/equivalent compute;
- peak memory;
- inference latency/throughput;
- training time.

Secondary metrics do not automatically become fitness terms.

## 10.2 Test isolation

Official challenging-test performance MUST NOT be used for:

- fitness;
- mutation feedback;
- crossover selection;
- EoT feedback;
- parent/population selection;
- prompt revision;
- hyperparameter search;
- decisions to continue architecture search.

## 10.3 Resource measurements

Before `DG-P1-05`, measure selected seeds on intended hardware.

Every valid formal candidate SHALL record the resource fields required by the frozen resource contract.

An OOM/timeout/resource-cap violation is a visible candidate outcome, not a reason to silently alter the global protocol.

---

# 11. Family Genome Contracts

Exact numeric ranges belong in versioned machine-readable schemas after `DG-P1-07`; the Phase Plan owns conceptual semantic boundaries.

## 11.1 YOLO11 concept genome

Planning-time architecture evidence MUST be reverified against the pinned seed (`P1-V02`).

Initial semantic genes:

- `Y1` — bounded backbone stage depth;
- `Y2` — bounded backbone/channel capacity;
- `Y3` — approved C3k2 internal configuration;
- `Y4` — bounded high-level attention/C2PSA capacity;
- `Y5` — bounded neck multiscale feature-processing capacity.

Initially protect:

- input modality;
- P3/P4/P5 output-scale contract;
- Detect output semantics;
- target interpretation;
- loss;
- training resolution;
- augmentation;
- optimizer/scheduler;
- NMS/evaluation semantics.

The initial study MUST NOT silently replace YOLO11 with another YOLO generation.

## 11.2 RT-DETR concept genome

Planning-time architecture evidence MUST be reverified against the pinned seed (`P1-V03`).

Initial semantic genes:

- `R1` — HybridEncoder hidden capacity;
- `R2` — HybridEncoder depth;
- `R3` — compatible attention-head/FFN configuration;
- `R4` — cross-scale encoder/fusion capacity;
- `R5` — decoder depth;
- `R6` — object-query count.

Initially protect:

- selected backbone family;
- pretraining policy;
- feature-stride contract;
- Hungarian matcher;
- criterion/loss definitions and weights;
- postprocessor;
- optimizer/schedule;
- multiscale-training policy;
- training budget.

## 11.3 Faster R-CNN concept genome

Exact implementation depends on `DG-P1-03` and `P1-V04`.

Initial semantic genes:

- `F1` — FPN feature capacity;
- `F2` — RPN head architecture;
- `F3` — anchor-size policy;
- `F4` — anchor-aspect-ratio policy;
- `F5` — RoI feature-extraction configuration;
- `F6` — box-head capacity.

Any FRED-derived anchor candidate sets MUST use project-training annotations only.

Initially protect:

- selected backbone family;
- initialization policy;
- Phase 0 targets;
- loss definitions;
- optimizer/schedule;
- training budget;
- evaluator;
- final score/NMS evaluation policy.

## 11.4 Prototype schema before final ranges

To avoid a circular dependency:

1. implement a **prototype genome schema** with conservative legal examples;
2. implement constraint/build validation against the pinned seed;
3. collect build/resource evidence;
4. resolve `DG-P1-07`;
5. freeze the formal numeric ranges/dependencies in versioned schemas.

Prototype values are implementation-validation scaffolding and MUST NOT become the formal research search space by accident.

## 11.5 Mutation magnitude and crossover

Pilot mutations SHOULD emphasize one semantic gene at a time.

Magnitude labels:

- **SMALL** — one bounded change inside one gene;
- **MEDIUM** — a substantial change to one subsystem or two tightly coupled values;
- **LARGE** — multiple interacting genes/topology changes.

Pilot runs emphasize SMALL, use some MEDIUM, and minimize LARGE mutations.

Initial crossover occurs only between corresponding genes of the same family and compatible genome version.

Incompatible genome versions MUST NOT cross without an explicit migration/compatibility rule.

---

# 12. LLM-GE Integration Contract

## 12.1 Evidence before modification

Before adapting LLM-GE, the project MUST complete `P1-V09`–`P1-V11`.

The coding agent SHALL inspect the pinned `MosesTheRedSea-main` implementation and produce an integration map describing:

- population lifecycle;
- individual/candidate representation;
- mutation call path;
- crossover call path;
- EoT/feedback call path;
- candidate IDs/ancestry;
- fitness ingestion;
- invalid-fitness behavior;
- state persistence/resume;
- local/cluster execution path;
- output/artifact locations;
- stale/missing-result behavior;
- prompt/template loading/version points.

No unfamiliar LLM-GE interface may be invented from planning prose.

## 12.2 Preserve framework behavior where compatible

The project intends to adapt the selected branch rather than rewrite working evolutionary machinery unnecessarily.

Where compatible, preserve:

- population handling;
- mutation;
- crossover;
- EoT/feedback flow;
- candidate identity/ancestry;
- state persistence;
- invalid-fitness handling;
- supported execution behavior.

## 12.3 Replace task-specific assumptions through adapters

Phase 1 replaces example-task assumptions with project-owned:

- detector-family candidate builders;
- Phase 0/FRED access;
- object-detection training jobs;
- validation evaluator bridge;
- detection-fitness reader;
- detector artifact handling;
- failure classification;
- resource reporting.

The adapter boundary SHOULD allow the reusable LLM-GE core to remain ignorant of detector-internal implementation details.

## 12.4 Controlled candidate representation

LLM-GE SHALL modify a project-owned semantic candidate representation.

It SHALL NOT receive unrestricted authority to rewrite:

- Ultralytics internals;
- RT-DETR repository internals;
- MMDetection/Torchvision/Detectron2 internals;
- Phase 0;
- evaluator/metric code;
- LLM-GE orchestration itself.

## 12.5 Prompt isolation/versioning

A mutation/crossover/EoT prompt MAY contain:

- family/genome version;
- current gene values;
- allowed ranges/modules;
- target gene/subsystem;
- shape/interface constraints;
- candidate resource information;
- validation fitness;
- training-stability feedback;
- ancestry summary;
- approved EoT feedback.

It SHOULD NOT expose:

- held-out test metrics/labels;
- credentials/unrelated files;
- arbitrary repository internals;
- evaluator internals in a form encouraging benchmark manipulation.

Every Phase 1 prompt/operator version MUST be recorded as experiment provenance.

---

# 13. Candidate Validity and Failure Contract

## 13.1 Validity funnel

Every generated candidate passes increasingly expensive checks:

```text
candidate generated
      ↓
genome/schema validation
      ↓
protected-field validation
      ↓
build/config validation
      ↓
model construction
      ↓
shape/interface validation
      ↓
checkpoint-load/initialization validation
      ↓
parameter/compute/resource estimate
      ↓
tiny forward pass
      ↓
tiny backward pass
      ↓
smoke training
      ↓
resource eligibility
      ↓
formal candidate training
      ↓
validation evaluation
```

## 13.2 Pre-training validation

Before expensive training:

- genome parses;
- only allowed genes/modules/values are used;
- required genes exist;
- dependency rules pass;
- dimensions/channels/heads are legal;
- model constructs;
- outputs satisfy adapter/evaluator expectations;
- checkpoint loading follows the initialization contract;
- batch/input shapes work;
- gradients flow;
- loss is finite on a trusted tiny batch;
- obvious resource caps are satisfied.

## 13.3 Invalid candidate behavior

Invalid candidates SHALL:

- receive explicit failure classification;
- preserve candidate ID/ancestry;
- retain relevant error evidence;
- avoid unnecessary full training;
- not be represented as successful mutations;
- not corrupt population state.

## 13.4 Failure taxonomy

### Candidate representation failure

- invalid genome;
- forbidden field;
- unsupported combination.

### Model construction failure

- incompatible channels;
- invalid attention-head configuration;
- missing module;
- invalid feature wiring;
- initialization/load-contract violation.

### Training failure

- NaN/Inf;
- OOM;
- divergence;
- timeout;
- checkpoint failure.

### Evaluation failure

- malformed predictions;
- coordinate/format violation;
- evaluator exception;
- metric artifact missing.

### Evolution infrastructure failure

- missing/stale candidate result;
- lineage corruption;
- failed operator job;
- state persistence/recovery failure.

Scientific failures remain visible and auditable.

---

# 14. Pilot Evolution Contract

## 14.1 Pilot purpose

Pilot evolution validates the evolutionary system, not final detector quality.

Pilot work MUST NOT begin until:

- formal seed selection is resolved;
- adapters/bring-up work;
- controlled training/evaluation/resource protocol is frozen;
- prototype/final genome validation is available;
- LLM-GE integration map and smoke tests pass;
- `DG-P1-08` defines pilot execution parameters.

## 14.2 Pilot success criteria

All three family pilots must demonstrate:

- seed import;
- candidate generation;
- bounded mutations;
- candidate diversity;
- genome validation;
- cheap invalid-model rejection;
- valid candidate training;
- deterministic metric ingestion;
- ancestry preservation;
- population-state persistence/recovery;
- controlled failures;
- no benchmark/data mutation;
- no test feedback;
- resource measurement;
- reproducible execution.

## 14.3 Optional multi-fidelity branch

Multi-fidelity is not assumed to be valid simply because it is cheaper.

After pilots, the project MAY collect matched short/full-budget results.

`DG-P1-09` either approves a fixed screening/promotion contract or records that multi-fidelity is not used.

If not approved, formal primary evolution uses the frozen full candidate-evaluation protocol.

---

# 15. Formal Baselines and Primary Event Evolution

## 15.1 Formal six-baseline execution

Formal baselines occur **after** `DG-P1-04`, `DG-P1-05`, and `DG-P1-06`.

Run:

```text
YOLO11-RGB
YOLO11-Event
RTDETR-RGB
RTDETR-Event
FRCNN-RGB
FRCNN-Event
```

For each:

- preserve the evidence package in §8.4;
- classify reproduction status;
- verify no held-out-score tuning;
- verify single-class mapping;
- record approved pretraining/initialization;
- complete the approved baseline confirmation-seed policy.

A family cannot enter formal evolution until its required non-evolutionary baseline evidence is satisfactory.

## 15.2 Primary evolutionary modality — FIXED

The first formal evolutionary studies are:

```text
Study E-Y: YOLO11-Event
Study E-R: RT-DETR-Event
Study E-F: Faster-RCNN-Event
```

All use:

- frozen Phase 0 Event view;
- identical project validation membership;
- the controlled Phase 1 training/evaluation protocol;
- family-specific versioned genome;
- protected evaluator;
- no official-test feedback;
- formal execution parameters from `DG-P1-08`;
- multi-fidelity only if approved by `DG-P1-09`.

## 15.3 Within-family fairness

Every candidate within a formal family study uses the same protected training/evaluation variables except explicitly evolvable architecture fields.

## 15.4 Cross-family interpretation

Cross-family analysis SHOULD distinguish:

- absolute validation quality;
- improvement over family seed;
- parameters/compute;
- memory;
- latency;
- robustness;
- later downstream utility.

The project MUST NOT assume identical training difficulty across families.

---

# 16. Elite Archive, Confirmation, and Held-Out Evaluation

## 16.1 No single premature winner

Each family maintains an elite archive.

Possible retention reasons include:

- highest primary fitness;
- highest `mAP50`;
- high recall;
- strong accuracy/efficiency tradeoff;
- low latency;
- structurally distinct near-top candidate;
- later downstream-useful candidate when Phase 2 evidence becomes available.

Phase 1 MAY preserve non-dominated candidates across accuracy, compute, memory, and latency.

## 16.2 Elite metadata

Each retained elite records:

- candidate ID;
- family;
- genome version;
- parentage;
- prompt/operator-history references;
- architecture specification;
- training config;
- Phase 0 versions;
- validation metrics;
- resource profile;
- checkpoint reference;
- code/repository state;
- retention reason.

## 16.3 Full-budget multi-seed confirmation

Before a candidate is used for a reportable architectural claim:

- mandatory seed baselines;
- promoted top elites;
- candidates claimed to improve materially over a seed

SHOULD be retrained/evaluated across the approved independent full-budget seeds under the same frozen protocol.

Report aggregate performance and variability.

## 16.4 Test freeze

Before official challenging-test evaluation, freeze:

- architecture;
- genome version;
- checkpoint/retraining policy;
- Phase 0 versions;
- adapter;
- evaluation configuration;
- candidate-selection rule;
- relevant prompt/evolution decisions.

Held-out results MUST NOT trigger another search round under the same final-test protocol.

## 16.5 Core Phase 1 freeze occurs before expansion research

Once core held-out evaluation, reporting, and the Phase 2 handoff contract are complete, the project freezes **core Phase 1 Version 1.0**.

Only after this freeze may `DG-P1-10` authorize expansion studies.

---

# 17. Phase 1 Artifacts and Phase 2 Contract

## 17.1 Required durable artifacts

Core Phase 1 produces:

1. seed-selection decision records;
2. adapter implementations/tests;
3. family bring-up/profile evidence;
4. frozen controlled training/evaluation/resource protocol;
5. six formal baseline-validation packages;
6. versioned genome schemas/builders/validators;
7. LLM-GE integration map and adapters;
8. prompt/operator versions;
9. candidate lineage/state/failure artifacts;
10. pilot reports;
11. primary evolutionary-study reports;
12. family elite archives;
13. final elite checkpoints/confirmation summaries;
14. held-out evaluation report;
15. detector prediction contract;
16. Phase 2 handoff metadata.

## 17.2 Detection prediction contract

The semantic output contract is named:

```text
P1-CONTRACT-DETECTIONS-v1
```

It contains at least:

```text
sequence_id
frame_index / timestamp
detector_id
model_family
model_version
modality
boxes_xyxy
confidence
predicted class/label mapping
coordinate-space version
Phase 0 manifest version
Phase 1 adapter/evaluator version
code revision
```

Exact serialization is an implementation choice.

Semantic changes to required fields/meaning require a contract-version change.

## 17.3 Phase 2 independence

Phase 2 SHALL consume the frozen Phase 1 detector-output interface rather than arbitrary detector-internal files.

Phase 2 must be able to compare, where protocol permits:

- oracle/ground-truth detections;
- validated seed detector outputs;
- selected evolved elite outputs.

---

# 18. Canonical Phase 1 Implementation Roadmap

This is the **single authoritative core Phase 1 execution roadmap**.

A coding agent SHOULD be tasked with one work package at a time. Each non-trivial work package receives a short task brief specifying prerequisites, allowed files/components, prohibited scope, deliverables, tests, and exit criteria.

## Stage A — Phase 0 handoff

### `P1-A1` Verify Phase 0 contract

**Prerequisites:** frozen Phase 0.

**Work:**

- record schema/manifest/split versions;
- verify RGB/Event access;
- verify generic adapter contract;
- reconcile `P1-V01`.

**Must not:** reconstruct/modify Phase 0 semantics.

**Exit:** trusted FRED samples can be consumed without Phase 0 reimplementation.

## Stage B — Seed evidence and gate resolution

### `P1-B1` Resolve YOLO seed

**Gate:** `DG-P1-01`

**Exit:** selected/pinned YOLO implementation, checkpoint, and reproduction classification basis.

### `P1-B2` Resolve RT-DETR seed

**Gate:** `DG-P1-02`

**Exit:** selected/pinned RT-DETR implementation/checkpoint.

### `P1-B3` Resolve Faster R-CNN seed

**Gate:** `DG-P1-03`

**Exit:** selected/pinned Faster R-CNN framework/config/checkpoint.

No formal adapter/baseline implementation is pinned to a recommendation before the corresponding gate resolution.

## Stage C — Family adapters and minimal bring-up

### `P1-C1` Implement/verify YOLO adapter

**Read:** §§5–7 plus pinned upstream API.

**Deliverables:** adapter, config, adapter tests, minimal model construction/forward/loss path.

**Verification:** `P1-V02`, `P1-V05`.

### `P1-C2` Implement/verify RT-DETR adapter

**Verification:** `P1-V03`, `P1-V05`.

### `P1-C3` Implement/verify Faster R-CNN adapter

**Verification:** `P1-V04`, `P1-V05`.

### `P1-C4` Verify evaluator/postprocessing on trusted development data

**Work:**

- verify family prediction conversion;
- verify `mAP50` / `mAP50:95`;
- compare deterministic repeated evaluator results;
- test malformed prediction handling.

**Verification:** `P1-V06`.

**Exit:** family adapters/evaluator can support controlled bring-up, but no formal baseline conclusion has yet been produced.

## Stage D — Bring-up training and resource profiling

### `P1-D1` YOLO bring-up/profile

Short trusted training/evaluation sufficient to verify optimizer/checkpoint/resource behavior, not reportable performance.

### `P1-D2` RT-DETR bring-up/profile

### `P1-D3` Faster R-CNN bring-up/profile

### `P1-D4` Verify checkpoint initialization/loading contract

**Verification:** `P1-V07`.

### `P1-D5` Consolidate resource/protocol evidence

**Verification:** `P1-V08`.

**Exit:** evidence exists to resolve the training/resource/metric gates.

## Stage E — Freeze controlled experimental protocol

### `P1-E1` Freeze training/pretraining/randomness

**Gate:** `DG-P1-04`

### `P1-E2` Freeze resource envelope

**Gate:** `DG-P1-05`

### `P1-E3` Freeze metric/fitness contract

**Gate:** `DG-P1-06`

**Exit:** formal baselines and future architecture evolution have a stable controlled protocol.

## Stage F — Run six formal non-evolutionary baselines

### `P1-F1` YOLO formal RGB/Event baselines

### `P1-F2` RT-DETR formal RGB/Event baselines

### `P1-F3` Faster R-CNN formal RGB/Event baselines

### `P1-F4` Baseline reproduction review

**Work:**

- assemble evidence packages;
- classify reproduction status;
- verify held-out-score non-target rule;
- execute approved multi-seed confirmation for reportable baseline evidence.

**Exit:** all mandatory seed systems are scientifically/operationally validated before evolution.

## Stage G — Prototype and freeze family genomes

### `P1-G1` YOLO prototype genome/builder/validator

Concept genes `Y1–Y5`.

### `P1-G2` RT-DETR prototype genome/builder/validator

Concept genes `R1–R6`.

### `P1-G3` Faster R-CNN prototype genome/builder/validator

Concept genes `F1–F6`; project-training-only anchor analysis where used.

### `P1-G4` Resolve formal genome ranges

**Gate:** `DG-P1-07`

### `P1-G5` Freeze versioned family genome schemas

**Exit:** every family has a bounded, versioned, tested formal candidate representation.

## Stage H — Inspect and adapt LLM-GE

### `P1-H1` Inspect pinned LLM-GE implementation

**Work:** produce integration map required by §12.1.

**Verification:** `P1-V09`, `P1-V10`, `P1-V11`.

**Must not:** modify LLM-GE before relevant paths/interfaces are understood.

### `P1-H2` Implement candidate-generation bridge

Connect mutation/crossover operators to family genomes.

### `P1-H3` Implement execution/fitness bridge

Connect candidates to detection train/eval and validation-fitness artifacts.

### `P1-H4` Implement failure/lineage/state integration

Preserve candidate identity, ancestry, failures, persistence/recovery.

### `P1-H5` Freeze evolution execution configuration

**Gate:** `DG-P1-08`

**Exit:** LLM-GE detector integration is ready for controlled pilots.

## Stage I — Family pilots

### `P1-I1` YOLO pilot

### `P1-I2` RT-DETR pilot

### `P1-I3` Faster R-CNN pilot

**Exit:** all pilot success criteria in §14 pass.

## Stage J — Optional multi-fidelity decision branch

This stage is optional and does not block primary evolution unless the project intentionally investigates screening.

### `P1-J1` Collect matched short/full-budget evidence

### `P1-J2` Resolve multi-fidelity policy

**Gate:** `DG-P1-09`

If this branch is skipped, record "not used" and proceed with the frozen full candidate-evaluation protocol.

## Stage K — Primary Event evolution

### `P1-K1` YOLO11-Event formal study

### `P1-K2` RT-DETR-Event formal study

### `P1-K3` Faster-RCNN-Event formal study

**Exit:** complete family archives, lineage, failures, prompts/operators, resource evidence, and validation outcomes exist.

## Stage L — Elite archive and reportable confirmation

### `P1-L1` Build family elite archives

### `P1-L2` Multi-seed full-budget confirmation

### `P1-L3` Freeze candidate selection for held-out evaluation

**Exit:** reportable candidates selected without test feedback.

## Stage M — Official held-out evaluation

### `P1-M1` Evaluate frozen baseline/final elite set

### `P1-M2` Produce core Phase 1 comparison report

**Exit:** held-out results exist without contaminating search.

## Stage N — Freeze core Phase 1 handoff

### `P1-N1` Freeze detector prediction contract/archive

### `P1-N2` Produce Phase 2 handoff artifacts

### `P1-N3` Execute core Phase 1 completion gate

**Exit:** Phase 2 may begin formal tracking integration.

**Core Phase 1 Version 1.0 is now frozen.**

Post-core expansion research, if any, is authorized separately through `DG-P1-10` and Appendix A.

---

# 19. Core Phase 1 Completion Gate and Definition of Success

## 19.1 Completion gate

Core Phase 1 Version 1.0 is complete only when all applicable mandatory items pass.

```text
[ ] Frozen Phase 0 manifest/schema/split versions are verified and recorded.

[ ] DG-P1-01 resolved and exact YOLO11 seed/revision/checkpoint recorded.

[ ] DG-P1-02 resolved and exact RT-DETR seed/revision/checkpoint recorded.

[ ] DG-P1-03 resolved and exact Faster R-CNN framework/config/checkpoint recorded.

[ ] Recommended seed defaults were not treated as implementation authorization before their gates.

[ ] All three family adapters consume Phase 0 without redefining it.

[ ] Single-class `drone` mapping is verified consistently across adapters/modalities.

[ ] Evaluator/postprocessing and mAP metrics are verified on development data.

[ ] Family bring-up/profile runs complete without being mislabeled as formal baselines.

[ ] Candidate checkpoint-loading/initialization contract is implemented and tested.

[ ] DG-P1-04 frozen training/pretraining/randomness protocol exists.

[ ] DG-P1-05 frozen candidate resource envelope exists.

[ ] DG-P1-06 frozen validation fitness/evaluation contract exists.

[ ] YOLO11-RGB and YOLO11-Event formal baselines are validated.

[ ] RTDETR-RGB and RTDETR-Event formal baselines are validated.

[ ] FRCNN-RGB and FRCNN-Event formal baselines are validated.

[ ] Reproduction category/deviations are documented for all six baselines.

[ ] Published held-out FRED scores were not used as tuning targets.

[ ] Approved multi-seed baseline confirmation is complete.

[ ] Prototype YOLO, RT-DETR, and Faster R-CNN genome builders/validators exist.

[ ] DG-P1-07 formal genome ranges/dependencies are approved.

[ ] Versioned formal genome schemas exist for all three families.

[ ] Candidate code cannot mutate protected dataset/evaluator/training fields.

[ ] Candidate validity funnel rejects invalid models before unnecessary full training.

[ ] Pinned LLM-GE implementation was inspected and an integration map exists.

[ ] LLM-GE adaptation preserves candidate identity, lineage, failures, and state recovery.

[ ] Prompts/operators are versioned.

[ ] DG-P1-08 formal pilot/search execution parameters are recorded.

[ ] YOLO pilot passes.

[ ] RT-DETR pilot passes.

[ ] Faster R-CNN pilot passes.

[ ] Multi-fidelity is either explicitly not used or evidence-backed through DG-P1-09.

[ ] Primary YOLO11-Event evolutionary study completes.

[ ] Primary RT-DETR-Event evolutionary study completes.

[ ] Primary Faster-RCNN-Event evolutionary study completes.

[ ] Family elite archives exist with complete provenance.

[ ] Reportable baselines/elites receive approved multi-seed confirmation and variability reporting.

[ ] Candidate selection is frozen before official held-out evaluation.

[ ] Official held-out evaluation uses frozen candidates/protocol and does not feed back into search.

[ ] Core Phase 1 comparison report exists.

[ ] P1-CONTRACT-DETECTIONS-v1 exists and is versioned.

[ ] Phase 2 handoff artifacts are documented and reproducible.

[ ] Core Phase 1 is frozen before any optional modern/custom expansion begins.
```

Any unresolved mandatory item blocks the core Phase 1 freeze.

## 19.2 Definition of success

Phase 1 succeeds when the project can state with evidence:

> The three required detector families have trustworthy RGB and Event baselines on the frozen FRED data interface; seed choices are evidence-backed rather than guessed or tuned to published held-out scores; the primary task uses one consistent `drone` class and a frozen controlled pretraining/training/evaluation protocol; candidate initialization and randomness follow reproducible contracts; LLM-GE evolves only bounded family-specific architecture representations while data, benchmark membership, evaluator, and protected training variables remain fixed; invalid candidates are rejected cheaply and recorded transparently; three independent Event-based evolutionary studies produce reproducible family elite archives; reportable claims are confirmed across approved independent training seeds without held-out-test feedback; held-out evaluation occurs only after candidate selection is frozen; and `P1-CONTRACT-DETECTIONS-v1` allows Phase 2 to consume detector outputs without depending on detector internals.

---

# Appendix A — Post-Core Expansion Research

This appendix is **not part of the core Phase 1 implementation path**.

A coding agent MUST NOT implement these items unless `DG-P1-10` explicitly authorizes an expansion task after core Phase 1 Version 1.0 is frozen.

Possible future studies include:

## A.1 Training-method research

- localization losses;
- matching;
- optimizer/scheduler alternatives;
- augmentation;
- distillation;
- training-only tiny-object modules;
- D-FINE;
- DEIM;
- SET;
- NWD;
- DNTR;
- YOLO26/P2 small-target ideas.

## A.2 Architecture search expansion

### YOLO

- P2/4 detection scale;
- alternative neck fusion;
- approved modern YOLO components.

### RT-DETR

- high-resolution features;
- alternate localization heads;
- query mechanisms;
- stronger encoder/decoder modules;
- RT-DETRv2/v4-derived studies.

### Faster R-CNN

- denoising/cross-layer FPN;
- Transformer-style RoI head;
- alternative proposal modules.

Every widened search space requires a new versioned genome/experiment.

## A.3 RGB evolution

Equivalent RGB architecture evolution may be studied after the core Event experiments.

## A.4 Multimodal RGB/Event seeds

Possible later directions include:

- ER-DETR-like fusion;
- SPFD-inspired shared/private fusion;
- simple two-stream custom baselines;
- another justified fusion architecture.

Every new seed must pass non-evolutionary baseline validation before evolution.

## A.5 Custom/modern seeds

Examples considered in the planning material include:

- YOLO26 / YOLO26-P2;
- RT-DETRv2/v4;
- D-FINE / DEIM;
- RF-DETR;
- UAV-DETR;
- DNTR / SET / NWD-derived studies;
- other justified custom seeds.

These do not retroactively redefine the original three primary seeds.

## A.6 Event-native raw-event models

Models requiring raw events, voxelization, point clouds, or alternate temporal accumulation remain outside Phase 1 Version 1.0 and require an approved Phase 0 research extension first.

---

# Appendix B — Research Reference Policy

`references.md` is the curated Phase 1 research index.

Its references may inform:

- later comparisons;
- hypotheses;
- post-core search-space design;
- custom-seed research.

They do **not** become implementation requirements simply because they appear in the file.

For core Phase 1 work, `references.md` SHOULD NOT be loaded into the coding agent's context unless the current task specifically requires evidence from it.

Promotion from reference → experiment/search-space component requires an explicit decision/hypothesis and, for post-core expansion, `DG-P1-10`.

---

# Appendix C — Work-Package Task-Brief Template

The Phase Plan is the authority. A task brief is only a scoped execution view for one work package.

Use this template for non-trivial coding-agent tasks:

```text
Work package:
<e.g. P1-C1>

Objective:
<one concrete implementation outcome>

Prerequisites:
<resolved gates, existing artifacts, required revisions>

Governing sections:
<Phase 1 + Phase 0 + Engineering Standard references>

Required verification:
<P1-Vxx items>

Allowed implementation scope:
<directories/modules/files or ownership boundary>

Must preserve:
<existing behavior/interfaces/protected assets>

Must not implement:
<explicit adjacent/out-of-scope features>

External assumptions to verify:
<APIs/configs/checkpoints/framework behavior>

Deliverables:
<code/config/tests/artifacts>

Verification:
<unit/integration/smoke/regression commands and expected evidence>

Exit criteria:
<observable completion condition>
```

The brief MUST NOT silently add policy or resolve an open research decision.

---

# Appendix D — Research Progression Summary

The intended progression is:

```text
frozen Phase 0
      ↓
evidence-backed seed selection
      ↓
adapters + bring-up/profile
      ↓
frozen training/evaluation/resource protocol
      ↓
six formal baselines
      ↓
bounded versioned genomes
      ↓
verified LLM-GE integration
      ↓
family pilots
      ↓
primary Event architecture evolution
      ↓
multi-seed elite confirmation
      ↓
frozen held-out evaluation
      ↓
core Phase 1 / Phase 2 handoff freeze
      ↓
optional post-core expansion research
```

The guiding principle is:

> **Widen the research only after the narrower implementation and experiment are already trustworthy, reproducible, and frozen.**
