# LGCM

**LGCM** is an experimental learned general cognitive model for **persistent continual learning**.

The project asks a narrow, falsifiable question before making larger intelligence claims:

> Can a learner build an action-conditioned predictive model from an ongoing stream, adapt when the generating process changes, preserve useful prior competence, recover prior internal models when old conditions return, and use the learned model for decision making without being told which task or regime it is in?

LGCM-0 is deliberately bounded. It operates on controlled numerical observation/action vectors and studies continual dynamics learning, context recall, prequential prediction, persistence, and bounded model-based planning. It does **not** claim AGI, general perception, semantics, consciousness, unrestricted agency, or solved representation learning.

## LGCM-0 direction

The approved V2 architecture uses:

- a slow shared dynamics model;
- persistent context-specific residual experts;
- label-free context inference from predictive evidence;
- explicit mismatch detection with switch/spawn consequences;
- bounded expert capacity with no silent eviction;
- deterministic fixed feature encoders in V0, with representation learning reserved for later qualification;
- a baseline ladder that includes persistence, observation-only prediction, global cumulative ridge, and exponentially weighted RLS;
- prequential evaluation where prediction is fixed before the current outcome can update learner state;
- restart-continuity tests and evaluator/learner separation;
- a bounded planner whose results are reported separately from learner quality.

The original one-model cumulative ridge idea remains in the project as a **baseline**, not as the LGCM candidate.

## Status

**Pre-implementation / initial build.**

The architecture has completed hostile design review and the implementation plan is being executed test-first.

See:

- `docs/LGCM_DESIGN_SPEC_V2.md`
- `docs/plans/2026-09-25-lgcm-0.md`

## Project boundary

LGCM is a standalone research implementation. A future Vera Mono integration is not implied by repository presence and would require separate qualification.

## License

This repository is source-visible but not open source. Noncommercial evaluation, research, security review, interoperability assessment, and contribution preparation are permitted under `LICENSE`. Commercial use requires a separate written license; see `COMMERCIAL_LICENSE.md`.
