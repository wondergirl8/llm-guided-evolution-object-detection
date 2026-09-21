# Engineering Implementation Standard
## FRED + LLM-GE Research Project

**Status:** Version 1.0 — Implementation-Ready Engineering Standard  
**Scope:** All implementation work across Phase 0, Phase 1, Phase 2, Phase 3, baseline validation, shared infrastructure, and LLM-GE integration  
**Normative language:** `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, and `MAY`

---

# 1. Purpose

This document defines **how the project must be implemented safely and maintainably**.

It complements `MASTER_PROJECT_PLAN.md`.

The Master Plan remains authoritative for:

- research goals and architecture;
- FRED benchmark policy;
- project phases and phase gates;
- evaluation policy;
- baseline-before-evolution policy;
- LLM-GE experimental procedure;
- prompt management;
- experiment identity and lineage;
- reproducibility requirements;
- failure taxonomy;
- artifact and phase-interface policy.

This document does not redefine those research policies. It defines the software-engineering constraints used to implement them.

Phase plans may add stricter requirements but SHOULD NOT duplicate this standard.

---

# 2. Rule Severity, Applicability, and Exceptions

## 2.1 Normative severity

### MUST / MUST NOT

Mandatory when the rule is applicable. A coding agent may not silently waive or reinterpret a `MUST` requirement.

### SHOULD / SHOULD NOT

Expected default when applicable. A deviation requires a concrete technical reason and must not undermine a `MUST` requirement or research validity.

### MAY

Optional technique or implementation choice.

## 2.2 Applicability

The eight engineering invariants in Section 3 apply to **all implementation work**.

Detailed rules apply only when the current task touches the subsystem, boundary, artifact, state, dependency, test surface, or concern governed by that rule.

A rule MUST NOT be used as justification to expand a task into unrelated work.

Examples:

- a documentation-only change does not require new ML smoke tests;
- a parser bug fix does not require redesigning experiment logging;
- a Phase 1 adapter change does not require modifying Phase 3;
- an existing test framework need not be rewritten merely because testing rules apply.

When a rule is applicable, its normative severity remains binding.

## 2.3 Exceptions to `MUST` requirements

A coding agent MUST NOT self-authorize an exception to a `MUST` or `MUST NOT` rule.

An intentional exception requires an explicit project/task decision that records, as applicable:

- the exact rule being excepted;
- the narrow scope of the exception;
- why compliance is impossible or counterproductive;
- risks introduced;
- containment/rollback strategy;
- validation required to show the exception did not compromise correctness or research validity.

An exception to one rule does not waive unrelated rules.

## 2.4 No policy-by-convenience

Implementation convenience MUST NOT silently decide an unresolved research, benchmark, evaluation, or scientific-policy question.

If the governing planning documents deliberately leave a decision unresolved, the implementation MUST preserve that unresolved status unless the current task explicitly resolves it.

---

# 3. Non-Negotiable Engineering Invariants

The project is governed by the following eight invariants.

### INV-01 — Preserve unrelated working behavior

Changes MUST preserve behavior outside the requested scope unless an explicit requirement changes that behavior.

### INV-02 — Isolate components

Major subsystems MUST be independently testable and SHOULD be independently replaceable and removable.

### INV-03 — Keep baselines independent from LLM-GE

A validated non-evolutionary seed path MUST remain runnable without the LLM-GE subsystem.

### INV-04 — Make reruns safe

Operations intended to be rerun MUST NOT corrupt, duplicate, or unpredictably alter project state.

### INV-05 — Protect source truth

Raw/source FRED data, benchmark identities, protected evaluation definitions, and other Master-Plan-protected components MUST NOT be silently modified.

### INV-06 — Verify instead of guessing

External APIs, repository behavior, file formats, tensor shapes, configuration fields, and undocumented assumptions MUST be verified before they are relied upon.

### INV-07 — Never hide scientific failure

Failures, invalid outputs, numerical instability, malformed candidates, and evaluation errors MUST NOT be silently converted into successful results.

### INV-08 — Preserve traceability

Research-relevant outputs MUST retain the metadata and lineage required by the Master Plan.

---

# 4. Architecture and Component Boundaries

## ARCH-01 — Modular ownership [MUST]

Each major responsibility MUST have a clear owner.

Examples include:

- data loading and preprocessing;
- model adapters;
- baseline training;
- evaluation;
- LLM-GE orchestration;
- prompt handling;
- experiment tracking;
- artifact management.

A component MUST NOT accumulate unrelated responsibilities merely for implementation convenience.

## ARCH-02 — Loose coupling [MUST]

Components MUST communicate through documented interfaces rather than relying on another component's internal implementation.

Replacing one detector, tracker, forecaster, storage implementation, or evolutionary adapter SHOULD require minimal changes outside that component's boundary.

## ARCH-03 — Fault isolation [MUST]

A failure in one experiment, candidate, optional feature, or phase-specific implementation MUST NOT unnecessarily corrupt or invalidate unrelated components or previously valid artifacts.

Where practical, experimental failures SHOULD terminate only the affected run/candidate.

## ARCH-04 — Replaceability and removability [SHOULD]

New subsystems SHOULD be designed so that removing them restores the prior system behavior without requiring unrelated repairs.

Feature-specific logic SHOULD remain localized.

## ARCH-05 — Baseline independence [MUST]

Seed-model training/evaluation MUST remain independently runnable without invoking LLM-GE.

The evolutionary system MUST consume the baseline/model system through explicit interfaces or adapters.

## ARCH-06 — Separate reusable infrastructure from phase-specific logic [SHOULD]

Reusable project infrastructure SHOULD live in shared project areas.

Phase-specific behavior SHOULD remain in its phase until reuse is demonstrated.

Do not prematurely move code into shared infrastructure.

## ARCH-07 — Avoid speculative abstractions [MUST]

The implementation MUST NOT create generalized frameworks, factories, plugin systems, or abstraction layers solely for hypothetical future use.

Generalization SHOULD be driven by real repeated requirements.

---

# 5. State, Idempotency, and Reversibility

## STATE-01 — Idempotent rerunnable operations [MUST]

Setup, preprocessing, validation, artifact generation, and similar rerunnable operations MUST be safe to execute repeatedly wherever practical.

A rerun MUST NOT accidentally:

- duplicate data;
- append duplicate metadata;
- corrupt existing outputs;
- silently mix incompatible versions;
- produce an ambiguous state.

## STATE-02 — Reversible feature changes [SHOULD]

A newly added feature SHOULD be removable without changing unrelated system behavior.

The implementation SHOULD minimize irreversible changes to shared state.

## STATE-03 — Atomic important writes [SHOULD]

Important artifacts SHOULD be written atomically where practical.

For example:

```text
write temporary artifact
        ↓
validate successful completion
        ↓
rename/move to final location
```

Partially written files MUST NOT be treated as valid completed artifacts.

## STATE-04 — Checkpoint expensive stages [SHOULD]

Long-running or expensive stages SHOULD persist valid intermediate artifacts when doing so does not violate the experimental design.

A downstream failure SHOULD NOT require recomputing trusted upstream work without reason.

## STATE-05 — Controlled side effects [MUST]

Functions and commands MUST make meaningful side effects explicit.

A function SHOULD NOT unexpectedly:

- rewrite configuration;
- delete artifacts;
- mutate source data;
- download dependencies;
- start training;
- modify unrelated persistent state.

Computation and side effects SHOULD be separated where practical.

---

# 6. Data and Artifact Safety

## DATA-01 — Raw/source data immutability [MUST]

Original FRED source data MUST be treated as read-only during normal project operation.

Transformations MUST write to separate processed/intermediate/cache outputs.

## DATA-02 — No silent benchmark mutation [MUST]

The implementation MUST NOT silently modify official or project-frozen:

- dataset splits;
- annotation semantics;
- sequence identities;
- benchmark test protocols;
- protected evaluation definitions

MUST NOT be changed by implementation code unless a specifically authorized experiment requires it.

## DATA-03 — Explicit artifact lineage [MUST]

Research-relevant artifacts MUST be attributable to the producing run/configuration according to the Master Plan's experiment and reproducibility requirements.

## DATA-04 — Validate cached/derived artifacts [SHOULD]

Code SHOULD detect when a cached or derived artifact is incompatible with the current configuration, source version, schema, or preprocessing version.

Stale artifacts MUST NOT be silently accepted when doing so could invalidate results.

---

# 7. Interfaces and Contracts

## API-01 — Explicit interfaces [MUST]

Cross-component interfaces MUST define the information required to use them correctly.

As relevant, document or validate:

- input type;
- output type;
- tensor/array shape;
- dtype;
- coordinate convention;
- timestamp convention;
- class/identity semantics;
- side effects;
- failure behavior.

## API-02 — Boundary validation [MUST]

The implementation MUST validate correctness-critical data when it crosses subsystem boundaries.

Examples include:

- tensor rank/shape;
- dtype;
- class-ID range;
- coordinate bounds;
- sample/annotation correspondence;
- timestamps;
- expected files;
- configuration schema;
- prediction structure.

## API-03 — No magic inference of important state [SHOULD]

Scientifically or operationally meaningful behavior SHOULD be represented explicitly rather than inferred from incidental properties.

Prefer:

```text
input_modality = event
```

over implicit rules such as "two input tensors means event mode."

## API-04 — Frozen downstream interfaces [MUST]

Cross-phase outputs used as stable downstream inputs MUST follow the Master Plan's frozen-interface policy.

Downstream code MUST NOT depend on arbitrary implementation-specific temporary files.

## API-05 — Backward compatibility [SHOULD]

Existing public project interfaces SHOULD remain backward compatible unless the governing task explicitly changes the contract.

Contract changes MUST be documented and tested.

---

# 8. Configuration and Sources of Truth

## CFG-01 — Configuration-driven experiments [MUST]

Configuration choices for formal or reproducible experiments MUST be represented in configuration rather than embedded as ad hoc source-code changes.

Temporary debugging values MAY be supplied directly where appropriate, provided they cannot be mistaken for a reproducible experiment definition.

Examples include:

- model variant;
- dataset paths;
- modality;
- training hyperparameters;
- evaluation settings;
- random seed;
- evolutionary settings where applicable.

## CFG-02 — Single source of truth [MUST]

A project-wide value or definition MUST have one authoritative source when inconsistency could affect behavior, reproducibility, or research validity.

The implementation MUST NOT independently redefine values such as:

- class mappings;
- dataset split definitions;
- coordinate conventions;
- metric configuration;
- model identifiers;
- experiment metadata fields.

## CFG-03 — No hidden environment assumptions [MUST]

Environment-dependent behavior MUST be explicit.

Hardcoded user-specific absolute paths, machine-specific locations, or undocumented environment assumptions MUST NOT be introduced into reusable project code.

## CFG-04 — Sensible defaults only [MUST]

Defaults MUST be intentional, documented, and scientifically/operationally safe.

Missing required configuration MUST NOT silently trigger a guessed fallback.

---

# 9. External Dependencies and Repository Integration

## DEP-01 — Never invent external APIs [MUST]

The agent MUST NOT assume an unfamiliar external:

- method;
- function;
- class;
- command;
- CLI flag;
- configuration key;
- model capability;
- checkpoint field;
- repository path

exists because it appears plausible.

Verify it against the actual source/package/repository documentation available to the project.

## DEP-02 — Inspect before modifying [MUST]

Before changing unfamiliar external integration code, the agent MUST inspect the relevant execution path and understand:

```text
input
  ↓
adapter / loader
  ↓
external implementation
  ↓
returned output
  ↓
project consumer
```

## DEP-03 — Dependency isolation [MUST]

Dependency versions and external repository states MUST follow the Master Plan's repository/dependency policy.

Agents MUST NOT casually upgrade, replace, or globally modify dependencies to solve a local problem.

## DEP-04 — Adapt rather than fork blindly [SHOULD]

External repositories SHOULD be integrated through controlled adapters, patches, or documented mechanisms rather than copied and modified without traceability.

---

# 10. Change Management

## CHANGE-01 — Minimal-change principle [MUST]

The agent MUST implement the smallest change that correctly satisfies the task.

The change MUST NOT mix unrelated refactoring, renaming, formatting churn, directory restructuring, or dependency upgrades into feature work.

## CHANGE-02 — Preserve existing behavior [MUST]

Before changing a working path, the agent MUST identify the behavior that must remain unchanged.

After implementation, the agent MUST verify relevant regressions.

## CHANGE-03 — No opportunistic rewrites [MUST]

A local task MUST NOT be used as justification to rewrite a working subsystem unless the rewrite is required for correctness or explicitly requested.

## CHANGE-04 — Clear names [SHOULD]

Names SHOULD communicate domain intent.

Prefer specific names such as:

```text
load_fred_sequence
convert_events_to_representation
evaluate_detection_map
generate_mutation_prompt
```

over vague names such as:

```text
process
helper
data2
final_result
```

## CHANGE-05 — Canonical execution paths [SHOULD]

Each major workflow SHOULD have a clear canonical invocation.

Avoid parallel ad hoc scripts such as:

```text
train.py
train_new.py
train_fixed.py
train_final2.py
```

Experiment variation SHOULD normally come from configuration, not copied scripts.

---

# 11. Failure Handling and Scientific Integrity

## FAIL-01 — Fail fast on invalid assumptions [MUST]

When required conditions are violated, code MUST produce a clear failure rather than continue with invalid state.

## FAIL-02 — Do not swallow exceptions [MUST]

The implementation MUST NOT broadly suppress exceptions unless the exception is intentionally transformed into a documented failure state.

If an exception is caught, relevant context MUST be preserved.

## FAIL-03 — Do not mask scientific failures [MUST]

The implementation MUST NOT alter invalid experimental outputs merely to keep a pipeline running.

Examples of prohibited behavior include:

- replacing NaN loss with zero without an approved policy;
- inventing missing predictions;
- silently dropping invalid annotations from formal evaluation;
- treating a failed candidate as successfully evaluated;
- silently changing thresholds/configuration until a run completes.

## FAIL-04 — Classify failures [SHOULD]

Where relevant, failures SHOULD use the categories defined by the Master Plan:

- data;
- integration;
- candidate validity;
- training;
- evaluation;
- evolution infrastructure.

## FAIL-05 — Preserve failure evidence [MUST]

Formal experimental failures MUST preserve enough information to diagnose the failure and retain candidate/run lineage where applicable.

---

# 12. Reproducibility and Experiment Traceability

## REPRO-01 — Follow Master Plan metadata requirements [MUST]

Formal experiments MUST preserve the code, data, model, training, LLM-GE, and evaluation state required by the Master Plan.

## REPRO-02 — Controlled randomness [MUST]

Random seeds MUST be recorded for research-comparison runs.

Where practical, relevant libraries SHOULD be seeded consistently.

If full determinism is not achievable, the limitation SHOULD be documented rather than hidden.

## REPRO-03 — Record software state [MUST]

Reportable results MUST remain attributable to the project revision and relevant external repository/package revisions.

## REPRO-04 — Preserve prompt identity [MUST]

Any LLM-GE result that depends on a prompt MUST retain the prompt identity/version required by the Master Plan.

## REPRO-05 — Human-readable experiment organization [SHOULD]

Experiment outputs SHOULD use a consistent, navigable structure and stable run identities.

Do not rely on anonymous output directories whose provenance cannot be reconstructed.

---

# 13. Testing and Verification

## TEST-01 — Unit testing [SHOULD]

Deterministic isolated logic SHOULD have focused unit tests.

Examples include:

- parsing;
- coordinate conversions;
- metadata handling;
- configuration validation;
- lineage utilities;
- pure preprocessing functions.

## TEST-02 — Integration testing [MUST]

Cross-component interfaces that are important to the project MUST have integration validation.

Examples include:

- FRED loader → model adapter;
- model adapter → evaluator;
- candidate representation → executable model/configuration;
- upstream frozen artifact → downstream phase adapter.

## TEST-03 — Smoke testing [MUST]

Major executable workflows MUST support a low-cost smoke test where practical.

A smoke test is intended to answer:

> Can the pipeline complete correctly on a tiny workload?

It is not intended to establish model quality.

## TEST-04 — Regression testing [MUST]

When adding or changing behavior, the implementation MUST verify relevant previously working paths.

Examples:

```text
RGB baseline worked before event support
        ↓
verify RGB baseline still works

seed execution worked before LLM-GE adapter
        ↓
verify seed execution still works
```

## TEST-05 — Validate before expensive execution [MUST]

Where applicable, candidate/model workflows MUST perform the relevant low-cost validity checks before expensive training or evaluation:

```text
configuration/static validation
        ↓
imports/model construction
        ↓
interface/shape checks
        ↓
minimal forward pass
        ↓
training/evaluation
```

This complements the candidate-validation procedure in the Master Plan.

## TEST-06 — Test the public contract [SHOULD]

Tests SHOULD emphasize observable behavior and interfaces rather than overfitting to private implementation details.

---

# 14. Observability and Diagnostics

## OBS-01 — Actionable logging [MUST]

Failures and formal runs MUST expose enough context to identify:

- run/experiment identity;
- phase;
- relevant model/candidate;
- configuration;
- processing stage;
- failure reason.

## OBS-02 — Structured records for formal experiments [SHOULD]

Research runs SHOULD use structured metadata/logging rather than relying only on scattered console output.

## OBS-03 — No debug-print architecture [SHOULD]

Temporary debugging output SHOULD NOT become the primary observability mechanism of the project.

---

# 15. Documentation

## DOC-01 — Documentation is part of interface changes [MUST]

When a task changes any of the following, the relevant documentation MUST be updated:

- a public command;
- configuration schema;
- artifact format;
- cross-component interface;
- phase input/output;
- setup procedure

the relevant documentation MUST be updated.

## DOC-02 — Avoid duplication [MUST]

Project-wide policy MUST NOT be copied into multiple documents as competing normative definitions.

Reference the authoritative document instead; short guardrail summaries are allowed when they clearly point back to the authoritative rule.

Examples:

- research/evaluation policy → Master Plan;
- engineering behavior → this standard;
- phase-specific implementation → phase plan;
- operational commands/status → README.

## DOC-03 — Document why when necessary [SHOULD]

Non-obvious architectural constraints SHOULD include a brief rationale, especially where future agents might otherwise "simplify" the design in a way that breaks research validity.

---

# 16. Coding-Agent Operating Protocol

## 16.1 Task-local requirement selection

For non-trivial work, the agent SHOULD identify the applicable requirements before implementation.

A compact task contract should record:

- requested change;
- governing Master Plan / phase-plan requirements;
- applicable Engineering Standard rule IDs or rule groups;
- affected components;
- behavior/interfaces that must be preserved;
- external assumptions requiring verification;
- relevant verification steps.

This task-local contract is a working aid, not a new source of project policy.

## 16.2 Execution loop

Every coding task SHOULD follow this loop:

```text
1. Understand the requested change
        ↓
2. Read governing project/phase requirements
        ↓
3. Inspect the existing implementation
        ↓
4. Trace inputs, outputs, dependencies, and side effects
        ↓
5. Verify unfamiliar external APIs/assumptions
        ↓
6. Identify the smallest safe change
        ↓
7. Implement
        ↓
8. Run focused validation/tests
        ↓
9. Run relevant regression checks
        ↓
10. Inspect failures rather than bypassing them
        ↓
11. Update documentation/configuration if required
        ↓
12. Report what changed and any remaining limitations
```

## AGENT-01 — Evidence before implementation [MUST]

The agent MUST NOT implement unfamiliar behavior based only on what "normally" happens in similar libraries or projects.

The agent MUST inspect project or external evidence first.

## AGENT-02 — Explicit assumptions [MUST]

Material assumptions MUST be made explicit when they cannot be directly verified.

Do not silently convert an unresolved research decision into an implementation default.

## AGENT-03 — No false completion [MUST]

The agent MUST NOT claim a task is complete when:

- relevant tests were not run despite being available;
- known failures remain hidden;
- the implementation relies on an unverified invented interface;
- only the new path works while required existing paths are broken.

## AGENT-04 — Surface contradictions [MUST]

If authoritative documents, task instructions, and repository behavior materially conflict, the agent MUST surface the conflict rather than silently resolving it through guesswork.

---

# 17. Relationship to the Master Plan

`MASTER_PROJECT_PLAN.md` and this standard are authoritative in different domains:

- the Master Plan owns project-wide research/system policy;
- this standard owns software-engineering and implementation behavior;
- phase plans specialize phase-specific technical decisions;
- task instructions define the immediate scope of work;
- repository state is implementation evidence rather than policy.

A more specific document may specialize a requirement within its domain but MUST NOT silently contradict a project-wide fixed decision or applicable `MUST` rule.

The following topics remain governed primarily by `MASTER_PROJECT_PLAN.md` and SHOULD NOT be redefined here:

- baseline-before-evolution gate;
- challenging-split evaluation policy;
- official/protected benchmark components;
- LLM-GE evolutionary procedure;
- prompt versioning policy;
- elite archives;
- experiment identity and candidate lineage;
- cross-phase frozen artifacts/interfaces;
- detailed reproducibility fields;
- failure taxonomy;
- compute policy;
- research questions and ablations;
- publication/final evaluation strategy.

This engineering standard implements those requirements safely; it does not replace them.

---

# 18. Implementation Completion Gate

Before declaring an implementation task complete, verify the applicable items below.

```text
[ ] The requested behavior is implemented.

[ ] The implementation follows the relevant Master Plan and phase-plan requirements.

[ ] Existing unrelated behavior was preserved.

[ ] The change is localized and no unnecessary refactor/rewrite was introduced.

[ ] External APIs, repository behavior, and important assumptions were verified.

[ ] No raw/source FRED data or protected benchmark definition was silently modified.

[ ] Cross-component inputs/outputs and important shapes/schemas are validated.

[ ] Rerunning the operation does not corrupt or ambiguously duplicate state.

[ ] Important writes cannot be mistaken for complete artifacts when only partially written.

[ ] Baseline functionality remains independent from LLM-GE where applicable.

[ ] Relevant unit tests pass.

[ ] Relevant integration tests pass.

[ ] Relevant regression checks pass.

[ ] A minimal smoke test passes where applicable.

[ ] Failures are surfaced and no exception/scientific failure is silently hidden.

[ ] Formal experiment outputs preserve required identity, configuration, and lineage metadata.

[ ] Configuration remains the source of experiment choices rather than ad hoc source edits.

[ ] Documentation was updated if commands, configuration, interfaces, artifacts, or setup changed.

[ ] No known correctness issue is being hidden behind a successful exit status.
```

A task does not require every checkbox in every case. It requires every checkbox that is relevant to the scope of the task.

---

# 19. Guiding Engineering Principle

> Every subsystem should be understandable, independently testable, safely rerunnable, and replaceable with minimal impact on unrelated subsystems.

For this project, robustness means more than avoiding crashes. It means ensuring that software failures, experimental failures, and future changes do not silently compromise the validity, reproducibility, or maintainability of the research.
