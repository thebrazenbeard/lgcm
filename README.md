> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# LGCM

**LGCM** is an experimental learned general cognitive model for **persistent continual learning**.

The project asks a narrow, falsifiable question before making larger intelligence claims:

> Can a learner build an action-conditioned predictive model from an ongoing stream, adapt when the generating process changes, preserve useful prior competence, recover prior internal models when old conditions return, and use the learned model for decision making without being told which task or regime it is in?

LGCM-0 is deliberately bounded. It operates on controlled numerical observation/action vectors and studies continual dynamics learning, context recall, prequential prediction, persistence, and bounded model-based planning. It does **not** claim AGI, general perception, semantics, consciousness, unrestricted agency, or solved representation learning.

## LGCM-0 implementation

The current research implementation uses:

- a slow shared dynamics model;
- persistent context-specific residual experts;
- label-free context inference from predictive evidence, with a configurable
  absolute standardized-error ceiling for recalling dormant experts;
- explicit mismatch detection with switch/spawn consequences;
- bounded expert capacity with no silent eviction;
- deterministic fixed feature encoders in V0, with representation learning reserved for later qualification;
- a baseline ladder that includes persistence, observation-only prediction, global cumulative ridge, and exponentially weighted RLS;
- prequential evaluation where prediction is fixed before the current outcome can update learner state;
- restart-continuity tests and evaluator/learner separation.

Evaluator-only phase labels distinguish first exposure to regime `A` from
`A_RETURN`; these labels are never passed to the learner. A bounded
model-based planner and learned representations remain future work, not
implemented capabilities.

The original one-model cumulative ridge idea remains in the project as a **baseline**, not as the LGCM candidate.

## Status

**Initial implementation / active qualification.**

The repository contains the LGCM-0 Python package and focused tests for
baselines, change detection, context gating and recall, fixed encoders,
prequential evaluation, bounded model capacity, persistence, restart
equivalence, residual experts, and RLS primitives. Source and tests do not by
themselves establish broader continual-learning performance or future Vera
integration.

`update()` and `save_snapshot()` serialize access to a model instance so a
snapshot records a complete committed update, not mixed expert/shared weights
from two generations. This is an **in-process** guarantee; snapshots of the
same destination from separate processes or model instances require external
coordination. Snapshots do not provide transactional rollback if an update
itself fails after modifying model state.

## Reproduce bounded evidence

Use Python 3.12+ with NumPy 2+ and pytest 8+ in an isolated environment.
Inspect the source and dependencies before installing or executing code.

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python experiments/heldout_probe.py
```

The diagnostic probe compares LGCM against persistence, observation-only,
cumulative ridge, and EW-RLS baselines on four simulator seeds with abrupt and
gradual A/B/A-return/C regimes. It scores each prediction **before** the
current outcome updates the model. It is not a generalization or continual-
learning qualification gate: novel regime C can favor simpler baselines, and
the default context-recall threshold is not statistically calibrated.

The former README linked two `docs/` design files not present in the checked
repository tree. For currently inspectable behavior, use `src/lgcm/`,
`tests/`, and `experiments/` rather than treating those missing paths as
available design authority.

The repository disables VS Code automatic workspace tasks and includes
source-only startup guards (`tests/test_workspace_startup_safety.py`). Do not
reenable folder-open shell execution or run non-code assets as programs.

## Project boundary

LGCM is a standalone research implementation. A future Vera Mono integration is not implied by repository presence and would require separate qualification.

## License

This repository is source-visible but not open source. Noncommercial evaluation, research, security review, interoperability assessment, and contribution preparation are permitted under `LICENSE`. Commercial use requires a separate written license; see `COMMERCIAL_LICENSE.md`.
