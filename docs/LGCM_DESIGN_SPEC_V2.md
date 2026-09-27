# LGCM Design Specification V2

Status: implemented qualification-first architecture.

## Research question

LGCM asks whether an online learner can build an action-conditioned predictive model, detect when its current dynamics model is no longer adequate, preserve useful prior competence, recover a prior latent context when it returns, and support bounded model-based planning without receiving task identifiers.

LGCM-0 is intentionally narrow. It operates on bounded numerical state/action vectors and does not claim general perception, semantic understanding, unrestricted agency, AGI, or consciousness.

## Core architecture

### Shared dynamics model

A cumulative RLS model learns structure common across all committed non-terminal observations. It supplies a shared prior for newly spawned experts when configured.

### Context experts

Each context expert contains:

- an action-conditioned RLS residual model;
- an exponentially weighted residual-scale tracker;
- evidence count and feature-support estimates.

Dormant experts are retained rather than silently overwritten.

### Context gate

At each observed transition, the gate scores all experts using standardized predictive error plus a feature-support penalty. When mismatch evidence is active, the gate can:

- retain the current expert;
- switch to a better retained expert;
- accumulate unexplained mismatch;
- spawn a new expert after bounded patience;
- surface explicit capacity exhaustion.

No regime label is supplied to this decision.

### Change detection

A Page-Hinkley detector accumulates standardized active-expert error after minimum evidence. The detector is reset after a context transition.

### Bootstrap buffer

While mismatch is unresolved, the current transition is retained in a bounded bootstrap buffer. A new expert can consume this evidence at creation. The previously active expert is frozen during unexplained mismatch so retained competence is not trained toward the novel regime.

### Persistence

Snapshots persist:

- configuration;
- encoder state;
- shared and expert RLS weights/covariances;
- residual-scale state;
- context-gate state;
- change-detector state;
- bootstrap evidence;
- active/next expert identifiers;
- generation and sequence state.

Snapshot payloads are digest-bound by a manifest and path-confined to the snapshot directory. Loading validates dimensions, state counters, covariance symmetry, and manifest membership. Snapshotting is blocked while an update transaction is in progress.

### Evaluation

Evaluation is prequential: prediction happens before the current transition updates the learner.

Evaluator metadata distinguishes:

- `A_INITIAL`;
- `A_TO_B_DRIFT`;
- `B`;
- `A_RETURN`;
- `C`.

That metadata never enters learner prediction or update calls.

The qualification runner compares LGCM with persistence, observation-only RLS, cumulative global RLS, and exponentially weighted RLS across multiple seeds and abrupt/gradual streams.

### Planning

The planner is bounded random-shooting model-predictive control.

Given an observation, explicit objective, horizon, candidate count, seed, and scalar action bounds, it samples candidate action sequences, rolls them forward only through `model.predict`, sums objective cost over the horizon, and returns the first action from the lowest-cost sequence.

Planning never calls `model.update`. Planner quality is reported separately from learner qualification.

## Public invariants

1. `Experience` and `Prediction` own read-only copies of vector inputs and do not mutate caller arrays.
2. Experience sequence numbers are strictly increasing for each learner instance.
3. Prediction does not mutate learner state.
4. Evaluation predicts before updating.
5. Evaluator regime/segment metadata is not learner input.
6. Expert capacity exhaustion is explicit and terminal for further learning.
7. Snapshot payloads cannot escape the snapshot root.
8. State-referenced snapshot payloads must be manifest-bound.
9. Snapshotting cannot observe a partially committed update.
10. Planning is read-only with respect to learner state.

## Qualification boundary

The current evidence surface is a controlled numerical simulator. A passing test suite or qualification receipt demonstrates behavior only inside that tested boundary.

Algorithmic expansion is evidence-gated. Replay, learned representations, additional drift detectors, expert eviction/merging, or more advanced planners should be added only when a predeclared benchmark demonstrates a need and compares the change against the current baselines.

## Related documents

- `EVALUATION_PROTOCOL.md`
- `REPRODUCIBILITY.md`
- `specs/2026-09-27-qualification-first-upgrade-design.md`
- `plans/2026-09-27-qualification-foundation.md`
