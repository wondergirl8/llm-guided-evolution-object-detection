# FRED + LLM-GE Research Project
## Master Project Planning Document

**Status:** Version 0.2 — Implementation-Governance Revision  
**Purpose:** Define the project-wide research architecture, repository ecosystem, evaluation policy, LLM-GE operating model, reproducibility requirements, phase interfaces, and overall repository organization for applying Large Language Model–Guided Evolution (LLM-GE) to the FRED benchmark. Project-wide software-engineering requirements are governed separately by `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md`.

---

# 1. Purpose of the Master Plan

This document coordinates the complete FRED + LLM-GE research project.

It is intentionally **not** a substitute for the four phase-specific planning documents. The Master Plan defines project-wide research/system rules and interfaces, while the phase plans define implementation- and experiment-level details for their respective problems. Software-engineering behavior is governed by `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md`.

The project is organized into:

1. **Phase 0 — Data Infrastructure and Input Pipeline**
2. **Phase 1 — Drone Detection**
3. **Phase 2 — Drone Tracking**
4. **Phase 3 — Drone Trajectory Forecasting**

The corresponding detailed plans will live under `docs/` and should be treated as children of this Master Plan.

The Master Plan is responsible for:

- the overall research goal;
- the relationship between the four phases;
- shared repository and dependency management;
- the project-wide LLM-GE workflow;
- baseline validation before evolution;
- prompt organization and versioning;
- project directory structure;
- reproducibility requirements;
- experiment logging;
- common evaluation rules;
- phase gates;
- cross-phase artifact interfaces;
- failure handling;
- elite archive propagation;
- project-wide compute and resource planning;
- overall research questions;
- publication and final evaluation strategy.

The four phase plans are responsible for the detailed implementation choices specific to their problem.

## 1.1 Implementation governance documents

Two additional project-wide documents govern implementation without duplicating this Master Plan:

- root-level `AGENTS.md` is the coding agent's execution/navigation contract;
- `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md` defines mandatory software-engineering and implementation behavior.

These documents are complementary rather than subordinate copies of the Master Plan.

### Authority by domain

- **Master Plan:** project-wide research architecture, benchmark/evaluation policy, phase relationships, LLM-GE operating model, reproducibility policy, and cross-phase contracts.
- **Engineering Implementation Standard:** software-engineering behavior, implementation safety, testing, change discipline, failure handling, and coding-agent verification.
- **Phase plan:** phase-specific technical requirements and decisions that specialize, but do not silently contradict, project-wide policy.
- **Current task:** immediate implementation scope.
- **Repository state:** factual evidence of the current implementation, not a policy source.

A more specific document may specialize a requirement within its own domain, but it must not silently override a fixed project-wide decision or an applicable Engineering Standard `MUST` requirement.

If authoritative requirements genuinely conflict, or if the repository materially contradicts a governing plan, the conflict should be surfaced rather than resolved by guesswork.

## 1.2 Decision-status interpretation

When implementing from this document:

- decisions explicitly identified as fixed/current are binding for this version;
- unresolved/open questions must remain unresolved until deliberately decided;
- examples are illustrative unless explicitly adopted;
- proposed, initial, possible, or future structures are not automatically implementation requirements;
- words such as `may`, `could`, `possible`, and `eventually` do not authorize speculative implementation;
- implementation convenience must not silently resolve a research-policy question.

This distinction is especially important because this Master Plan intentionally records both current decisions and future/unresolved design questions.

---

# 2. Overall Research Goal

The broad goal of the project is to investigate whether **LLM-Guided Evolution can produce useful model or system improvements for the perception pipeline defined by FRED**.

The project addresses three main perception problems:

1. drone detection;
2. drone tracking;
3. drone trajectory forecasting.

These problems are not treated as one large optimization problem. They form a dependency chain:

```text
FRED data
   ↓
Phase 0 — Data Infrastructure
   ↓
Phase 1 — Detection
   ↓
Phase 2 — Tracking
   ↓
Phase 3 — Trajectory Forecasting
   ↓
End-to-End Research Evaluation
```

The output quality of each phase can affect every downstream phase. Consequently, the project must evaluate both:

- **local phase performance**, and
- **downstream pipeline utility**.

A model that is locally best is not automatically assumed to be the best upstream source for the next phase.

---

# 3. Core Research Principles

The following principles apply across the entire project unless deliberately revised and documented.

## 3.1 Sequential experimental dependency

The main experimental pipeline remains:

```text
Phase 0 → Phase 1 → Phase 2 → Phase 3
```

Planning and literature work may overlap, but definitive downstream experiments should depend on stable, versioned upstream outputs.

## 3.2 Controlled benchmark definition

Benchmark-defining components should remain outside the unrestricted LLM-GE search space. This includes, unless a specific controlled study states otherwise:

- official dataset splits;
- labels and annotation semantics;
- evaluation definitions;
- data identities;
- sequence ordering;
- benchmark test protocols.

## 3.3 Fixed initial event representation

The first version of the project should reproduce and use the established FRED event representation rather than simultaneously turning event representation itself into an optimization target.

Alternative raw-event representations may become a later research extension.

## 3.4 Baseline before evolution

No seed family should enter LLM-GE experiments until a non-evolutionary baseline has been successfully integrated, trained/evaluated, and documented.

This is a project-wide rule.

Without this separation, an evolved model failure could be incorrectly attributed to LLM-GE when the actual cause is a broken FRED integration, incorrect training procedure, invalid data adapter, or evaluation error.

## 3.5 Elite archives rather than one winner

Each evolutionary phase should preserve multiple useful solutions when appropriate.

Possible reasons to retain multiple elites include:

- best raw task performance;
- strongest challenging-condition robustness;
- best computational efficiency;
- best accuracy/efficiency tradeoff;
- best downstream utility.

The exact archive policy belongs in the relevant phase plan.

## 3.6 Oracle versus realistic evaluation

Where downstream phases depend on upstream outputs, evaluation should distinguish the model itself from inherited upstream errors.

Examples include:

- tracking with ground-truth detections versus tracking with Phase 1 detections;
- forecasting from ground-truth trajectories versus forecasting from detector/tracker-generated trajectories.

---

# 4. Project-Wide Evaluation Policy

## 4.1 Default testing framework: FRED challenging split

The **FRED challenging split is the default general testing/evaluation split for this research project**.

The reason is that the challenging split is intended to expose model behavior across difficult operating conditions and therefore provides the primary robustness-oriented basis for comparing:

- original seed models;
- evolved models;
- competing architectures;
- elite candidates;
- downstream system variants.

Unless a phase-specific protocol explicitly requires otherwise, project-wide model comparison should prioritize results on the challenging split.

Other splits may still be used for purposes such as:

- reproducing an official baseline;
- debugging the pipeline;
- sanity checking;
- train/validation organization;
- comparison with prior published results.

However, these uses do not replace the challenging split as the default general evaluation framework for the project.

## 4.2 No test-set tuning

The planning documents must ensure that the final held-out evaluation procedure is not repeatedly used for architecture selection or prompt tuning.

The phase plans must define the exact train/validation/evaluation boundaries required to maintain a scientifically valid comparison.

## 4.3 Common reporting requirements

Every reportable experiment should identify at minimum:

- phase;
- task;
- seed family or model lineage;
- dataset split;
- data modality/representation;
- exact model/configuration identifier;
- prompt version where applicable;
- evolutionary generation and parentage where applicable;
- training configuration;
- random seed;
- software revision;
- evaluation metrics;
- resource/efficiency measurements where relevant;
- failure status if the run did not complete.

---

# 5. External Repository Ecosystem

The project currently depends on the following external repositories.

## 5.1 LLM-GE framework

**Repository:** `jasonzutty/llm-guided-evolution-fork`  
**Branch:** `MosesTheRedSea-main`

Project role:

- evolutionary orchestration;
- LLM-guided mutation;
- LLM-guided crossover where appropriate;
- Evolution of Thought / feedback mechanisms;
- population management;
- ancestry/lineage handling;
- fitness-driven selection;
- generation-level control.

The framework should be adapted rather than rewritten from scratch.

The project should preserve the reusable evolutionary core while replacing original problem-specific assumptions with adapters for FRED and the relevant model/system family.

## 5.2 FRED benchmark repository

**Repository:** `miccunifi/FRED`

Project role:

- benchmark reference;
- dataset organization;
- annotations;
- data loading reference;
- split definitions;
- RGB/event handling reference;
- detection/tracking/forecasting benchmark context.

FRED should be treated as a benchmark authority rather than as an unrestricted component of the evolutionary search space.

## 5.3 YOLO11 implementation

**Repository:** `ultralytics/ultralytics`

Project role:

- Phase 1 YOLO11 seed family implementation.

The repository is selected, but the exact YOLO11 variant/configuration is intentionally not fixed in the Master Plan.

## 5.4 RT-DETR implementation

**Repository:** `lyuwenyu/RT-DETR`

Project role:

- Phase 1 RT-DETR seed family implementation.

The exact model/configuration remains a Phase 1 planning decision.

## 5.5 Faster R-CNN implementation

One of the following implementation ecosystems will be selected later:

- `torchvision.models.detection` / `pytorch/vision`;
- Detectron2;
- MMDetection.

This choice remains intentionally open because it affects:

- model configurability;
- mutation boundaries;
- multimodal adaptation effort;
- training/evaluation integration;
- compatibility with the LLM-GE representation;
- reproducibility and maintenance burden.

The final choice should be documented before Phase 1 implementation is frozen.

---

# 6. External Repository Management Policy

External repositories should not be casually copied and modified without traceability.

Before formal experiments begin, the project should record for every external dependency:

- repository URL;
- selected branch;
- selected tag if applicable;
- exact Git commit;
- license notes;
- local modifications or patches;
- Python/package version constraints;
- hardware/runtime requirements.

A machine-readable repository/dependency manifest should eventually be created, for example:

```text
configs/repositories.yaml
```

The project may later decide whether external code is handled through:

- pinned package installations;
- Git submodules;
- managed external clones;
- patch files;
- a controlled vendor directory.

That implementation choice should be made only after the project repository itself exists.

---

# 7. Project-Wide LLM-GE Architecture

The intended architecture is:

```text
                  OUR FRED + LLM-GE PROJECT
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
   LLM-GE evolutionary core             FRED benchmark
          │                                   │
          └─────────────────┬─────────────────┘
                            │
                  phase/model adapters
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       YOLO11            RT-DETR        Faster R-CNN
```

The same conceptual structure should later generalize beyond detection:

```text
LLM-GE Core
   │
   ├── Phase 1 adapter → detector representation + detector evaluator
   ├── Phase 2 adapter → tracker representation + tracker evaluator
   └── Phase 3 adapter → forecaster representation + forecaster evaluator
```

The LLM-GE core should remain as reusable as practical.

Phase-specific behavior should be supplied through explicit adapters rather than by repeatedly rewriting the evolutionary engine.

---

# 8. General LLM-GE Experimental Procedure

The exact operators and search spaces are phase-specific, but the general project-wide procedure is expected to follow this pattern.

## Step 1 — Establish trusted data and evaluation

Before model evolution:

```text
FRED data
   ↓
validated Phase 0 pipeline
   ↓
phase-specific adapter
   ↓
trusted evaluation path
```

## Step 2 — Reproduce the seed without LLM-GE

For every seed family:

```text
official/reference implementation
   ↓
FRED integration
   ↓
training / evaluation
   ↓
challenging-split result
   ↓
saved baseline evidence
```

The seed baseline must be reproducible before evolution begins.

## Step 3 — Define the evolvable representation

The relevant phase plan must explicitly define:

- which components the LLM may change;
- which components are protected;
- how an individual is represented;
- how the representation maps back to executable code/configuration;
- how invalid candidates are detected;
- whether crossover is meaningful for the representation.

The LLM should not receive unrestricted authority over the complete repository.

## Step 4 — Generate candidate variation

Depending on the phase and experiment, candidate generation may use:

- LLM-guided mutation;
- LLM-guided crossover;
- Evolution of Thought feedback;
- selected evolutionary heuristics.

Prompt templates must be stored and versioned.

## Step 5 — Validate generated candidates

Before expensive training, candidates should pass appropriate checks such as:

- syntax/import validation;
- configuration validation;
- model construction;
- shape/interface validation;
- protected-component checks;
- minimal forward-pass test;
- parameter/resource sanity checks.

## Step 6 — Train and evaluate

A valid candidate is trained/evaluated using a controlled procedure.

The evaluation adapter returns a fitness record to the evolutionary framework.

## Step 7 — Assign fitness

Fitness may include one or more objectives such as:

- task performance;
- robustness;
- computational cost;
- parameter count;
- latency;
- memory;
- downstream suitability.

The exact objectives and selection strategy belong in the relevant phase plan.

## Step 8 — Selection and next generation

The evolutionary framework uses candidate fitness and lineage information to determine the next population.

## Step 9 — Preserve evidence

Every candidate that reaches evaluation should be traceable to:

- parent(s);
- mutation/crossover prompt;
- prompt version;
- generated change;
- source revision;
- training configuration;
- random seed;
- fitness;
- failure information.

## Step 10 — Produce an elite archive

The phase ends with a versioned archive of selected elite models/systems rather than an automatically chosen single winner.

---

# 9. Seed Baseline Validation Framework

A dedicated baseline-validation area will store experiments performed **without LLM-GE**.

This separates two questions:

1. **Does the seed model/system work correctly on our FRED pipeline?**
2. **Did LLM-GE improve, damage, or otherwise alter that working system?**

A phase should not start expensive evolutionary experiments when Question 1 is unresolved.

## 9.1 Baseline evidence to preserve

For each seed baseline, preserve as applicable:

- exact external repository commit;
- model configuration;
- checkpoint initialization;
- data configuration;
- modality/input configuration;
- training command/config;
- logs;
- final checkpoint;
- evaluation output;
- challenging-split metrics;
- plots/visualizations;
- failure notes;
- short human analysis;
- known deviations from the original implementation.

## 9.2 Baseline gate

A seed is considered ready for evolution only after:

- it can be built reliably;
- it can consume the intended FRED inputs;
- its training/evaluation path completes;
- its metrics are generated correctly;
- its result artifacts are stored;
- the result is reproducible to an acceptable tolerance.

Exact tolerances belong in the phase plans.

---

# 10. Prompt Management

All prompts used by LLM-GE must be treated as versioned experimental artifacts.

The root-level `prompts/` directory should contain project-wide prompt templates and phase-specific prompt families.

Expected categories may include:

```text
prompts/
├── README.md
├── system/
├── mutation/
├── crossover/
├── eot/
├── validation/
└── phase_specific/
    ├── phase0/
    ├── phase1/
    ├── phase2/
    └── phase3/
```

Each prompt used in a reportable run should have a stable identifier or file version.

Prompt changes should be logged because changing the prompt changes the effective search operator.

A result should therefore be reproducible not only from model code and random seeds, but also from the exact prompt configuration used to generate the candidate.

---

# 11. Proposed Project Repository Structure

The initial project repository should be organized approximately as follows.

```text
fred-llm-ge/
│
├── .gitignore
├── README.md
├── AGENTS.md
│
├── docs/
│   ├── MASTER_PROJECT_PLAN.md
│   ├── ENGINEERING_IMPLEMENTATION_STANDARD.md
│   ├── PHASE0_DATA_PLAN.md
│   ├── PHASE1_DETECTION_PLAN.md
│   ├── PHASE2_TRACKING_PLAN.md
│   └── PHASE3_FORECASTING_PLAN.md
│
├── prompts/
│   ├── README.md
│   ├── system/
│   ├── mutation/
│   ├── crossover/
│   ├── eot/
│   ├── validation/
│   └── phase_specific/
│       ├── phase0/
│       ├── phase1/
│       ├── phase2/
│       └── phase3/
│
├── src/
│   ├── common/
│   ├── llm_ge/
│   ├── data/
│   ├── evaluation/
│   ├── adapters/
│   ├── logging/
│   └── utils/
│
├── configs/
│   ├── repositories.yaml
│   ├── environments/
│   ├── experiments/
│   └── evaluation/
│
├── baseline_validation/
│   ├── README.md
│   ├── phase1_detection/
│   │   ├── yolo11/
│   │   ├── rtdetr/
│   │   └── faster_rcnn/
│   ├── phase2_tracking/
│   └── phase3_forecasting/
│
├── experiments/
│   ├── README.md
│   ├── phase1_detection/
│   ├── phase2_tracking/
│   └── phase3_forecasting/
│
├── artifacts/
│   ├── README.md
│   ├── checkpoints/
│   ├── elite_archives/
│   ├── frozen_interfaces/
│   └── reports/
│
├── logs/
│   ├── baseline/
│   ├── evolution/
│   └── failures/
│
├── scripts/
│   ├── setup/
│   ├── validation/
│   ├── training/
│   ├── evaluation/
│   └── evolution/
│
├── tests/
│   ├── data/
│   ├── models/
│   ├── adapters/
│   ├── evaluation/
│   └── evolution/
│
├── phase0_data/
│   ├── README.md
│   ├── configs/
│   ├── scripts/
│   ├── tests/
│   └── outputs/
│
├── phase1_detection/
│   ├── README.md
│   ├── configs/
│   ├── adapters/
│   ├── seeds/
│   ├── experiments/
│   ├── tests/
│   └── outputs/
│
├── phase2_tracking/
│   ├── README.md
│   ├── configs/
│   ├── adapters/
│   ├── seeds/
│   ├── experiments/
│   ├── tests/
│   └── outputs/
│
└── phase3_forecasting/
    ├── README.md
    ├── configs/
    ├── adapters/
    ├── seeds/
    ├── experiments/
    ├── tests/
    └── outputs/
```

This is the **initial organizational proposal**, not a requirement that every directory must exist on day one.

The repository can be simplified once implementation begins if some directories prove redundant.

---

# 12. Directory Responsibilities

## 12.1 `docs/`

Contains the project planning documents plus the shared engineering implementation standard:

```text
MASTER_PROJECT_PLAN.md
ENGINEERING_IMPLEMENTATION_STANDARD.md
PHASE0_DATA_PLAN.md
PHASE1_DETECTION_PLAN.md
PHASE2_TRACKING_PLAN.md
PHASE3_FORECASTING_PLAN.md
```

The Master Plan defines project-wide research/system policy and shared interfaces.

The Engineering Implementation Standard defines how project code must be engineered safely.

The phase plans contain the technical detail for each phase.

The root-level `AGENTS.md` is deliberately kept outside `docs/` because it is the coding agent's first operational entry point into this documentation system.

## 12.2 `src/`

Contains reusable project code that is not naturally owned by only one phase.

Examples:

- LLM-GE integration;
- shared interfaces;
- experiment logging;
- shared evaluation helpers;
- common data structures;
- adapter base classes;
- artifact/version utilities.

Phase-specific code can live inside its phase directory until it becomes clearly reusable.

## 12.3 `prompts/`

Stores all LLM prompt templates used in evolution and related validation.

Prompts are experimental inputs and should be versioned like code.

## 12.4 `baseline_validation/`

Stores the evidence from non-LLM-GE seed-model/system testing.

This directory exists specifically to answer:

> Was the original seed already working correctly before evolution?

It must remain clearly separated from evolved-model experiments.

## 12.5 `experiments/`

Stores experiment definitions, manifests, summaries, and references for formal evolutionary runs.

Large binary artifacts should not necessarily be committed directly to Git.

## 12.6 `artifacts/`

Stores or indexes durable outputs such as:

- selected checkpoints;
- elite archives;
- frozen phase interfaces;
- final reports.

Actual large files may later be redirected to external storage while this directory retains manifests and references.

## 12.7 `logs/`

Separates:

- baseline logs;
- evolution logs;
- failure logs.

## 12.8 Four phase directories

The root includes one implementation area per phase:

```text
phase0_data/
phase1_detection/
phase2_tracking/
phase3_forecasting/
```

Each phase has its own `README.md`.

The README should summarize:

- phase purpose;
- inputs;
- outputs;
- dependency on prior phases;
- link to its detailed planning document;
- important commands once implementation begins;
- current implementation status.

The README is operational documentation.

The phase planning document in `docs/` is the research/engineering specification.

---

# 13. Phase 0 — Master-Level Definition

## Goal

Build a reliable, deterministic, validated FRED data layer that can serve every downstream phase.

Phase 0 should preserve more information than any one downstream task requires.

The data layer should support appropriate representations of:

- RGB;
- synchronized event data;
- annotations;
- timestamps;
- sequence identity;
- drone identity where available;
- split membership.

## Master-level requirement

The project should have **one trusted data infrastructure** with separate downstream adapters rather than three unrelated loaders for detection, tracking, and forecasting.

## Phase 0 gate

Before Phase 1 formal experiments begin, the project must be able to demonstrate:

- deterministic loading;
- correct split membership;
- sample correspondence;
- timestamp consistency;
- annotation validity;
- coordinate validity;
- sequence integrity;
- identity preservation where applicable;
- visual/numerical sanity checks.

Exact acceptance tests belong in the Phase 0 Plan.

---

# 14. Phase 1 — Master-Level Definition

## Goal

Apply LLM-GE to FRED drone detection.

Current seed families:

1. YOLO11;
2. RT-DETR;
3. Faster R-CNN.

These families represent intentionally different detector philosophies.

Exact variants remain unresolved.

## Initial evolution strategy

The three model families should initially be treated as **separate evolutionary families**.

Conceptually:

```text
YOLO11 seed
   ↓
YOLO evolution
   ↓
YOLO elite archive

RT-DETR seed
   ↓
RT-DETR evolution
   ↓
RT-DETR elite archive

Faster R-CNN seed
   ↓
Faster R-CNN evolution
   ↓
Faster R-CNN elite archive
```

Direct cross-family code-level crossover should not be assumed valid.

If cross-family evolution is later attempted, it should use a deliberately designed shared representation rather than arbitrary code crossover.

## Phase 1 gate

Phase 1 should produce:

- validated non-evolutionary seed baselines;
- a stable LLM-GE detector procedure;
- reproducible challenging-split evaluation;
- versioned detector elites;
- documented detector outputs/interfaces for Phase 2.

Exact metrics, search spaces, and stopping criteria belong in the Phase 1 Plan.

---

# 15. Phase 2 — Master-Level Definition

## Goal

Associate detections over time and maintain stable tracks.

Phase 2 consumes versioned Phase 1 outputs.

## Required evaluation distinction

Phase 2 should distinguish:

### Oracle/upstream-controlled evaluation

Use ground-truth or reference detections to isolate tracker quality.

### Realistic pipeline evaluation

Use selected Phase 1 detector outputs to measure actual detector + tracker performance.

## Phase 2 gate

Phase 2 should produce:

- validated tracking baseline(s);
- stable tracker evaluation;
- an elite tracker/system archive;
- frozen track-output interfaces;
- selected trajectory sources for Phase 3.

Tracking seeds remain unresolved and belong in the Phase 2 Plan.

---

# 16. Phase 3 — Master-Level Definition

## Goal

Forecast future drone trajectories using historical track information.

## Required evaluation distinction

Phase 3 should distinguish:

### Oracle forecasting

Use ground-truth trajectory history to measure the forecasting model itself.

### Realistic/end-to-end forecasting

Use Phase 1 + Phase 2 generated trajectories.

## Phase 3 completion

The final phase should provide:

- forecasting baselines;
- evolved forecasting candidates;
- oracle results;
- realistic pipeline results;
- final elite/finalist systems;
- end-to-end research conclusions.

Exact forecasting seeds, horizons, representations, losses, and metrics belong in the Phase 3 Plan.

---

# 17. Phase Interface and Artifact Policy

Each phase must define an explicit output interface.

A downstream phase should not silently depend on whatever files happen to be produced by an upstream implementation.

A frozen interface should identify:

- artifact schema;
- coordinate conventions;
- timestamps;
- identity conventions;
- confidence representation;
- ordering;
- version;
- producing model;
- producing code revision.

Where practical, frozen interfaces should be stored or described under:

```text
artifacts/frozen_interfaces/
```

This makes it possible to reproduce downstream work even after upstream code continues to evolve.

---

# 18. Experiment Identity and Lineage

Every formal experiment should receive a unique run identifier.

A useful conceptual structure is:

```text
<phase>-<family>-<experiment>-<run>
```

For evolved candidates, the metadata should additionally record:

- candidate/gene ID;
- generation;
- parent ID(s);
- evolutionary operation;
- prompt ID;
- prompt version;
- mutation target;
- candidate source/config hash.

The exact naming scheme can be finalized once the repository exists.

---

# 19. Reproducibility Requirements

Every result intended for research comparison should be reproducible from recorded metadata.

At minimum, preserve:

## Code state

- project Git commit;
- external repository commits;
- local patch state.

## Data state

- FRED version/source;
- split identifier;
- preprocessing configuration;
- input modality;
- data cache/version where relevant.

## Model state

- seed family;
- exact seed configuration;
- initial checkpoint;
- evolved architecture/config;
- candidate lineage.

## Training state

- optimizer;
- schedule;
- batch size;
- epochs/steps;
- augmentation;
- random seed;
- hardware;
- precision mode.

## LLM-GE state

- LLM provider/model;
- generation parameters;
- mutation/crossover policy;
- prompt file and version;
- population configuration;
- generation number;
- parent selection;
- EoT/feedback state.

## Evaluation state

- split;
- metric implementation;
- inference configuration;
- thresholding/post-processing;
- efficiency measurement method.

---

# 20. Experiment Logging

The project should eventually use a structured experiment tracking system.

The specific platform is not yet fixed.

Whether the project uses:

- local structured logs;
- MLflow;
- Weights & Biases;
- TensorBoard plus manifests;
- another system,

the experiment schema should remain project-controlled.

Every formal run should expose the information required by Section 19.

---

# 21. Failure Handling and Triage

Failures must be classified rather than recorded simply as "run failed."

Suggested categories:

## Data failure

Examples:

- missing file;
- broken synchronization;
- invalid annotation;
- corrupted sample.

## Integration failure

Examples:

- seed implementation cannot consume project adapter;
- incorrect tensor shape;
- incompatible preprocessing.

## Candidate validity failure

Examples:

- generated syntax invalid;
- model cannot instantiate;
- protected component modified;
- invalid architecture dimensions.

## Training failure

Examples:

- out-of-memory;
- numerical instability;
- divergence;
- unexpected termination.

## Evaluation failure

Examples:

- missing predictions;
- invalid coordinate format;
- metric code error.

## Evolution infrastructure failure

Examples:

- LLM request failure;
- invalid prompt response;
- population bookkeeping error;
- missing parent artifact.

The project should preserve the failure reason and candidate lineage.

An invalid evolved candidate should normally be handled as an experimental outcome rather than crashing the entire evolutionary study.

A broken non-evolutionary baseline, however, should block evolution for that seed until the integration problem is resolved.

---

# 22. Project-Wide Compute Policy

Exact compute resources are not yet known and should not be invented in the Master Plan.

The final compute plan should distinguish:

- baseline reproduction budget;
- candidate validation budget;
- evolutionary search budget;
- full-training budget;
- final elite retraining budget;
- final evaluation budget.

Where possible, low-cost filters should occur before expensive training.

Conceptually:

```text
generated candidate
   ↓
static/config checks
   ↓
model build
   ↓
minimal forward test
   ↓
cheap training/screening if used
   ↓
full candidate evaluation
```

The phase plans should decide whether multi-fidelity evaluation is scientifically appropriate.

---

# 23. Overall Research Questions

The following questions are provisional and should be refined as the phase plans become concrete.

## RQ1

Can LLM-Guided Evolution improve FRED task performance over validated non-evolutionary seed systems?

## RQ2

What kinds of architectural or system changes does LLM-GE repeatedly discover when optimizing for challenging FRED conditions?

## RQ3

Are the models with the strongest local phase metrics also the most useful models for downstream phases?

## RQ4

Can a shared LLM-GE orchestration framework be reused effectively across detection, tracking, and trajectory forecasting when the individual representation and evaluator are changed?

## RQ5

What tradeoffs emerge between task accuracy, robustness, computational efficiency, and downstream utility?

## RQ6

How much does structured feedback such as Evolution of Thought contribute relative to simpler LLM-guided variation?

The final research questions should be frozen only after the detailed phase plans establish realistic experimental scopes.

---

# 24. Ablation Strategy at the Master Level

The Master Plan should require ablations that distinguish improvements caused by LLM-GE from improvements caused by unrelated changes.

Potential project-wide ablation dimensions include:

- original seed versus evolved candidate;
- LLM mutation with versus without EoT;
- mutation versus mutation + crossover where valid;
- local task fitness versus multi-objective fitness;
- different elite-selection strategies;
- upstream oracle versus realistic upstream outputs;
- evolved model local performance versus downstream performance.

Exact ablation matrices belong in the phase plans.

---

# 25. Publication and Final Evaluation Strategy

The final research story should not rely only on the single best evolved metric.

A strong final analysis should include:

- validated seed baselines;
- evolved results;
- challenging-split results;
- robustness analysis;
- efficiency tradeoffs;
- evolutionary trajectories/lineage;
- ablations;
- failure analysis;
- local versus downstream utility;
- oracle versus realistic pipeline results;
- reproducibility details.

Where scientifically appropriate, the project should report distributions across repeated runs rather than only one best run.

The exact statistical methodology remains unresolved and should be planned before final experiments.

---

# 26. Planning and Implementation Governance

The project should use a small set of complementary governance documents rather than duplicating policy across many files:

```text
AGENTS.md
    │
    ├── docs/MASTER_PROJECT_PLAN.md
    │       ├── docs/PHASE0_DATA_PLAN.md
    │       ├── docs/PHASE1_DETECTION_PLAN.md
    │       ├── docs/PHASE2_TRACKING_PLAN.md
    │       └── docs/PHASE3_FORECASTING_PLAN.md
    │
    └── docs/ENGINEERING_IMPLEMENTATION_STANDARD.md
```

This diagram is a navigation model, not a simple precedence chain. Authority is determined by domain as defined in Section 1.1.

The Master Plan owns:

- project-wide decisions;
- repository ecosystem;
- shared LLM-GE rules;
- common evaluation policy;
- reproducibility;
- common directory structure;
- phase interfaces;
- artifact/version policy;
- experiment identity;
- failure taxonomy.

The phase plans own:

- exact seed variants;
- phase-specific preprocessing details;
- exact model representation;
- exact evolvable/protected components;
- training procedures;
- phase-specific metrics;
- phase-specific fitness functions;
- prompt contents;
- phase-specific budgets;
- phase-specific phase-gate thresholds.

The Engineering Implementation Standard owns:

- software-engineering invariants;
- modularity, coupling, idempotency, and state-safety rules;
- interface/configuration discipline;
- dependency/API verification;
- implementation failure behavior;
- testing/regression expectations;
- coding-agent completion gates.

`AGENTS.md` owns:

- coding-agent document navigation;
- task-local reading strategy;
- conflict handling;
- decision-status interpretation;
- pre-task and completion workflow.

Duplication should be minimized. Short high-salience guardrail summaries are acceptable when they explicitly point back to the authoritative document.

---

# 27. Current Decisions Treated as Fixed for Version 0.2

The following are current project assumptions.

## Project structure

```text
Phase 0 → Phase 1 → Phase 2 → Phase 3
```

## Planning and governance structure

One Master Plan plus four detailed phase plans, supported by:

- root-level `AGENTS.md` as the coding-agent execution/navigation contract;
- `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md` as the project-wide software-engineering standard.

## FRED event handling

Initially reproduce/use the established FRED event representation.

## General evaluation priority

Use the **challenging split as the default general testing/evaluation framework**.

## Phase 1 seed families

- YOLO11;
- RT-DETR;
- Faster R-CNN.

## Project-wide repositories

- `jasonzutty/llm-guided-evolution-fork`, branch `MosesTheRedSea-main`;
- `miccunifi/FRED`;
- `ultralytics/ultralytics`;
- `lyuwenyu/RT-DETR`;
- one Faster R-CNN framework to be selected.

## Baseline requirement

Every seed must be validated without LLM-GE before formal evolution.

## Elite preservation

Do not automatically collapse every phase to one winner.

## Downstream evaluation

Separate local model quality from inherited upstream effects.

---

# 28. Important Unresolved Questions

These should not be prematurely fixed in the Master Plan.

## Project-wide

- exact LLM(s) used during evolution;
- API/provider strategy;
- evolutionary algorithm details;
- population size;
- generation count;
- selection strategy;
- mutation/crossover schedule;
- EoT schedule;
- prompt-generation policy;
- compute budget;
- experiment tracking platform;
- statistical methodology;
- artifact storage backend;
- external-repository integration mechanism.

## Phase 0

- exact local FRED files and storage layout;
- all synchronization edge cases;
- cache strategy;
- corrupted/questionable sequence policy.

## Phase 1

- exact YOLO11 variant;
- exact RT-DETR variant;
- Faster R-CNN framework;
- Faster R-CNN backbone/configuration;
- pretrained checkpoint policy;
- exact evolvable architecture boundaries;
- multimodal strategy;
- exact fitness formulation;
- exact efficiency objectives.

## Phase 2

- tracking seed systems;
- ByteTrack's precise role;
- tracker architecture versus parameter evolution;
- appearance-feature policy;
- event/RGB feature use;
- exact downstream detector-selection strategy.

## Phase 3

- forecasting seed systems;
- observation window;
- prediction horizon;
- coordinate representation;
- visual/event context policy;
- exact forecasting fitness/loss.

---

# 29. Recommended Planning Sequence From Here

With the Master Plan established, the next planning work should proceed in this order:

```text
1. Review and accept Master Plan Version 0.2 together with the implementation-governance documents.
2. Create the detailed Phase 0 Plan.
3. Create the detailed Phase 1 Plan.
4. Create the detailed Phase 2 Plan.
5. Create the detailed Phase 3 Plan.
6. Reconcile all four phase plans with the Master Plan.
7. Check phase interfaces and artifact contracts.
8. Check for duplicated or contradictory assumptions.
9. Freeze Master Plan Version 1.0 before large-scale implementation.
```

Some literature/repository investigation for later phases may occur in parallel, but phase dependencies should remain explicit.

---

# 30. Definition of Master-Plan Completion

The Master Plan is ready to freeze as Version 1.0 when:

- the project repository exists;
- the actual repository layout has been reconciled with this proposal;
- external repository integration is decided;
- project-wide experiment metadata is defined;
- reproducibility policy is accepted;
- challenging-split evaluation policy is operationally defined;
- baseline validation policy is implemented;
- `AGENTS.md` and `ENGINEERING_IMPLEMENTATION_STANDARD.md` are accepted and consistent with this Master Plan;
- the four phase plans exist;
- all cross-phase interfaces are specified;
- unresolved project-wide questions needed for implementation have been resolved;
- remaining unresolved questions are genuinely phase-specific or intentionally deferred research decisions.

---

# 31. Current Project View

```text
                         FRED DATA
                            │
                            ▼
              ┌─────────────────────────┐
              │ Phase 0: Data Pipeline  │
              └────────────┬────────────┘
                           │
                           ▼
          ┌─────────────────────────────────┐
          │ Phase 1: LLM-GE Drone Detection│
          └──────────────┬──────────────────┘
                         │
                 elite detector archive
                         │
                         ▼
          ┌─────────────────────────────────┐
          │ Phase 2: LLM-GE Drone Tracking │
          └──────────────┬──────────────────┘
                         │
                  elite tracker archive
                         │
                         ▼
       ┌────────────────────────────────────────┐
       │ Phase 3: LLM-GE Trajectory Forecasting│
       └──────────────────┬─────────────────────┘
                          │
                          ▼
                 FINAL SYSTEM ANALYSIS
```

The entire project is coordinated by:

```text
Master Plan
   +
Phase 0 Plan
   +
Phase 1 Plan
   +
Phase 2 Plan
   +
Phase 3 Plan
```

with shared infrastructure and governance for:

```text
coding-agent navigation
engineering implementation standards
repository pinning
data/evaluation rules
LLM-GE orchestration
prompt management
baseline validation
experiment logging
artifact versioning
failure handling
reproducibility
```

---

# 32. Immediate Next Step

The next detailed planning document should be:

```text
docs/PHASE0_DATA_PLAN.md
```

Before writing it, the Master Plan should be reviewed for structural changes.

Once the actual project Git repository is created and its final name/location are known, this document should be updated with:

- the real repository name;
- actual top-level structure;
- external-repository integration method;
- environment/bootstrap instructions;
- storage locations;
- compute environment information;
- collaboration/branching conventions if needed.

That update should produce the first repository-aware revision of the Master Plan.
