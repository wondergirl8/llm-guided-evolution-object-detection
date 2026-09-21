# AGENTS.md
## FRED + LLM-GE Research Project — Coding Agent Execution Contract

**Status:** Version 1.0 — Implementation-Ready Governance  
**Applies to:** All coding agents, implementation tasks, refactors, experiments, scripts, tests, and project automation

---

## 1. Purpose

This file is the coding agent's **entry point** for implementation work.

It does not duplicate the project specifications. It tells the agent:

- where authoritative decisions live;
- what must be read for the current task;
- which safeguards always apply;
- how to handle conflicts and unresolved decisions;
- when implementation work is complete.

The project should be changed conservatively: make the smallest correct change and preserve unrelated working behavior.

---

## 2. Authority by Domain

The project does **not** use one simple "lowest document wins" hierarchy. Each source has authority over a different kind of decision.

### `docs/MASTER_PROJECT_PLAN.md`

Authoritative for project-wide research and system policy, including:

- research architecture and goals;
- FRED benchmark policy;
- project phases, dependencies, and phase gates;
- baseline-before-evolution policy;
- project-wide LLM-GE behavior;
- evaluation policy;
- repository ecosystem;
- experiment/reproducibility policy;
- cross-phase artifacts and interfaces.

### `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md`

Authoritative for software-engineering and coding-agent behavior, including:

- modularity and component boundaries;
- idempotency and state safety;
- interface and configuration discipline;
- dependency/API verification;
- change control;
- testing and regression protection;
- failure handling;
- implementation completion requirements.

### Relevant phase plan

Authoritative for phase-specific technical decisions that do not conflict with project-wide policy, including:

- exact phase inputs/outputs;
- seed/model choices;
- phase-specific preprocessing;
- model representation;
- evolvable/protected components;
- training/evaluation details;
- phase-specific metrics, fitness, and gates.

### Current task / issue / implementation request

Authoritative for the **scope of work requested now**.

A task may specialize an existing requirement, but it does not silently override a project-wide `MUST` rule or a fixed research decision.

### Existing repository state

The repository is **evidence of the current implementation**, not a policy document.

It tells the agent what actually exists, what currently works, and where plans may have diverged from implementation.

---

## 3. Conflict and Exception Rules

When sources disagree:

1. Determine whether they govern the same domain.
2. Prefer the more specific valid requirement **within that domain**.
3. A phase plan or task MUST NOT silently override a project-wide fixed decision or Engineering Standard `MUST` requirement.
4. If the repository contradicts a governing plan, treat that as plan/implementation drift and surface it rather than guessing.
5. If two authoritative requirements genuinely conflict, do not invent a resolution. Report the conflict and preserve existing safe behavior unless the task explicitly resolves it.

A coding agent MUST NOT self-authorize an exception to a `MUST` rule. Intentional exceptions must follow the exception process in `ENGINEERING_IMPLEMENTATION_STANDARD.md`.

---

## 4. How to Interpret Planning Language

When reading the Master Plan or a phase plan:

- **fixed/current decisions** and explicit `MUST` requirements are binding;
- **unresolved/open questions** MUST remain unresolved until deliberately decided;
- **examples** are illustrative unless explicitly adopted;
- **proposed/initial/possible** structures are not automatically implementation requirements;
- words such as `may`, `possible`, `eventually`, and `could` do not authorize speculative implementation;
- implementation convenience MUST NOT silently decide a research question.

If decision status is unclear and materially affects correctness or scientific validity, surface the uncertainty instead of guessing.

---

## 5. Read Only What the Task Needs

Do not reread every project document in full for every change.

For each task, read:

```text
AGENTS.md
    ↓
relevant Master Plan section(s)
    ↓
relevant Engineering Standard rule group(s)
    ↓
relevant phase-plan section(s)
    ↓
relevant repository code/tests/configuration
```

The eight Engineering Standard invariants apply to **all** implementation work.

Detailed engineering rules apply only when the task touches the concern governed by that rule. Do not expand a small task into unrelated work merely because the standard is comprehensive.

---

## 6. Pre-Task Contract for Non-Trivial Work

Before a non-trivial implementation, establish a short task contract containing:

```text
Task:
What is being changed?

Governing requirements:
Which Master Plan / Engineering Standard / phase-plan rules apply?

Affected components:
Which files/modules/interfaces are expected to change?

Must preserve:
Which existing behaviors/interfaces/artifacts must remain unchanged?

External assumptions to verify:
Which repository/API/file-format/model assumptions require evidence?

Verification:
Which unit, integration, regression, smoke, or interface checks are relevant?
```

This task contract is a working aid, not a new permanent governance document.

For a trivial, low-risk edit, this may be implicit.

---

## 7. Non-Negotiable Engineering Invariants

Every implementation task MUST preserve these invariants:

1. **Preserve unrelated behavior.** Do not change working behavior outside the requested scope without an explicit requirement.
2. **Isolate components.** Major subsystems must remain independently testable and should remain replaceable/removable.
3. **Keep baselines independent.** Validated non-evolutionary seed execution must not depend on LLM-GE.
4. **Make reruns safe.** Rerunnable operations must not corrupt, duplicate, or ambiguously alter project state.
5. **Protect source truth.** Raw FRED data and protected benchmark definitions must not be silently modified.
6. **Verify instead of guessing.** External APIs, formats, shapes, paths, and repository behavior must be verified.
7. **Never hide scientific failure.** Invalid runs/results must not be converted into apparent success.
8. **Preserve traceability.** Research outputs must retain the identity, configuration, code/data state, prompt, and lineage required by the Master Plan.

The complete normative definitions live in `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md`.

---

## 8. Implementation Discipline

For every task:

- inspect the existing code path before modifying an unfamiliar subsystem;
- make the smallest correct change;
- avoid unrelated refactors, renames, formatting churn, dependency upgrades, or directory restructuring;
- avoid speculative abstractions for hypothetical future requirements;
- use existing sources of truth instead of creating duplicates;
- keep feature-specific behavior localized;
- preserve backward compatibility unless the governing requirement changes the contract;
- verify unfamiliar external behavior against source/package/repository documentation available to the project;
- do not hide exceptions, malformed data, invalid candidates, numerical failures, or evaluation errors.

---

## 9. Baseline / LLM-GE Separation

Required conceptual structure:

```text
validated FRED + model system
        │
        ├── standalone baseline training/evaluation
        │
        └── LLM-GE evolutionary layer
                └── candidate generation/evaluation
```

Disabling or removing the LLM-GE layer MUST NOT make a validated seed baseline unusable.

LLM-GE should interact with seed/model systems through explicit adapters or controlled interfaces.

---

## 10. Verification Before Completion

Run only the checks relevant to the change, but run all relevant checks.

Depending on scope, this may include:

- unit tests;
- configuration/schema validation;
- interface/shape validation;
- model construction or minimal forward pass;
- integration tests;
- regression checks for previously working behavior;
- low-cost smoke tests.

A new path working is insufficient if required existing behavior regressed.

Use the completion gate in `docs/ENGINEERING_IMPLEMENTATION_STANDARD.md` before declaring non-trivial work complete.

---

## 11. Completion Reporting

When completing a task, report concisely:

- what changed;
- what was intentionally left unchanged;
- what verification was performed;
- any unresolved limitation, failed check, or plan/repository mismatch.

Do not claim successful completion when a known correctness issue is hidden or a required validation was skipped without explanation.

---

## 12. Guiding Rule

> Build each subsystem so that it can be understood, tested, rerun, replaced, and removed with minimal impact on unrelated subsystems.

This execution contract applies across Phase 0, Phase 1, Phase 2, Phase 3, baseline validation, shared infrastructure, and LLM-GE integration.
