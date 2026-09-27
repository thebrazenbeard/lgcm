# LGCM Qualification-First Upgrade Design

Date: 2026-09-27
Status: APPROVED
Base: `main@0043c60419de9ef0b52363b62aed8c0a52ef63f5`

## Purpose

LGCM remains a bounded research system for task-free, action-conditioned continual dynamics learning under recurring latent regimes. The upgrade strengthens correctness, trust boundaries, reproducibility, evaluation depth, and decision-usefulness before expanding algorithmic complexity.

## Design principles

1. Preserve task-free learning: evaluator regime/phase metadata must never be passed to the learner.
2. Preserve prequential evaluation: prediction is fixed before the current outcome updates learner state.
3. Prefer evidence over architectural expansion: new algorithmic mechanisms must justify themselves against baselines.
4. Keep learner quality separate from planner quality.
5. Make persistence self-contained and path-confined.
6. Make repository claims match implemented, tested source.
7. Treat multi-seed dispersion, resource use, and failure behavior as first-class evaluation outputs.

## Workstreams

### A. Correctness and trust

- Defensive-copy public array inputs so constructors cannot mutate caller-owned NumPy writeability state.
- Constrain every snapshot payload reference to the snapshot root.
- Require state-referenced payloads to be declared in the manifest.
- Validate restored covariance symmetry, finite state, dimensions, non-negative counters, expert identity consistency, and bootstrap dimensions.
- Add an actual update-in-progress guard around state mutation so snapshots cannot be taken from a partially committed update.
- Complete `LGCMConfig` validation for all numeric configuration fields.
- Make Ruff and mypy clean without weakening runtime behavior.

### B. Qualification laboratory

- Distinguish evaluator-visible segment labels from hidden dynamics-regime identity while never exposing either to the learner.
- Add a deterministic multi-seed experiment runner covering abrupt drift, gradual drift, recurrence, novel regimes, and expert-capacity pressure.
- Compare LGCM against persistence, observation-only, global ridge, and exponentially weighted RLS baselines.
- Emit deterministic JSON receipts with configuration, seeds, per-run metrics, aggregate mean/std/min/max, expert decisions, runtime, and bounded resource measurements.
- Measure overall and per-segment MSE, adaptation AUC, samples-to-criterion, forgetting/reacquisition deltas, switch/spawn counts, recurrence reuse, false spawns, and expert-count dispersion.
- Predeclare primary qualification metrics in documentation before interpreting results.

### C. Bounded planning

- Add a finite-horizon seeded random-shooting model-predictive planner.
- Accept an explicit caller objective and action bounds.
- Roll out only through model predictions; planning must not update the learner.
- Return the chosen first action plus reproducibility/score metadata.
- Evaluate planner performance separately from continual-learning qualification.

### D. Repository engineering

- Add CI for tests, strict warnings, Ruff, mypy, package build, and installed-wheel smoke tests.
- Run supported Python versions and at least Linux plus Windows where persistence behavior is relevant.
- Add coverage enforcement after focused trust-boundary tests raise the current 88% baseline.
- Enrich package metadata and add a source-available license expression / private publication protection compatible with current packaging standards.
- Add SECURITY.md and reproducibility documentation.
- Repair README links and remove or qualify claims until their implementation exists.
- Add technical contribution guidance without inventing legal CLA terms.

## Claim ceiling

Repository source, tests, and benchmark receipts may support claims about the exact tested implementation and experimental conditions only. They do not establish AGI, consciousness, unrestricted generality, deployment readiness, or future Vera integration.

## Hostile review outcome

The strongest objection is that infrastructure could outrun evidence for the core expert-routing idea. Accepted. Therefore the implementation order is:

1. correctness/trust,
2. reproducible comparative benchmark,
3. evidence review,
4. bounded planner,
5. only evidence-justified algorithmic expansion.

No additional detector, replay architecture, learned representation, or advanced planner is added solely for sophistication.
