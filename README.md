> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# LGCM

**LGCM** is an experimental Python library for **persistent continual learning of numerical dynamics**. It is a research component for developers evaluating whether an action-conditioned predictor can adapt to changing regimes while retaining and recalling previously learned contexts.

The project's testable question is:

> Can a learner predict outcomes from an ongoing stream of observations and actions, adapt when the generating process changes, preserve useful prior competence, and recover an earlier model when old conditions return—without being told which regime it is in?

LGCM-0 is deliberately bounded. It operates on controlled numerical observation/action vectors. It does **not** claim AGI, general perception, semantics, consciousness, unrestricted agency, or solved representation learning.

## What is reusable today

The `lgcm` Python package exposes a `ContextualWorldModel` configured by `LGCMConfig`, frozen `Experience` records with read-only array fields, predictions with evidence/adequacy metadata, update receipts, and snapshot save/load functions. The companion baselines and evaluation helpers let researchers compare the candidate model with simpler methods.

| Component | Implemented purpose | Boundary |
| --- | --- | --- |
| `ContextualWorldModel.predict(observation, action)` | Predict a next numerical observation from the current model state | A prediction, not a command or a warranted action |
| `ContextualWorldModel.update(experience)` | Update shared and context-specific regressors from a numbered transition | Numerical online updates only; no pretrained language-model weight training |
| `save_snapshot(model, path)` / `load_snapshot(path)` | Persist and restore the learned model state | In-process snapshot capture; coordinate separate processes externally |
| Baselines and `evaluate_prequential` | Evaluate predictions before exposing the corresponding outcomes | Diagnostic evidence, not proof of general intelligence or deployment readiness |

### Minimal Python example

Use Python 3.12+ and NumPy 2+. From a checkout, `PYTHONPATH=src` makes the package importable without installing it. The example is intentionally synthetic; replace its bounded numerical vectors and recorded transitions with your own validated data stream.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from lgcm import (
    ContextualWorldModel,
    Experience,
    LGCMConfig,
    load_snapshot,
    save_snapshot,
)

model = ContextualWorldModel(LGCMConfig(observation_dim=2, action_dim=1))

observation = [0.0, 0.0]
action = [0.2]
prediction = model.predict(observation, action)  # Before seeing the outcome

receipt = model.update(
    Experience(
        sequence=0,
        observation=observation,
        action=action,
        next_observation=[0.1, 0.0],
        source="example_stream",
    )
)

# Save to a fresh temporary path, avoiding pre-existing checkout artifacts.
with TemporaryDirectory(prefix="lgcm-example-") as temp_dir:
    snapshot_path = Path(temp_dir) / "snapshot"
    save_snapshot(model, snapshot_path)
    restored = load_snapshot(snapshot_path)
    assert restored.last_sequence == receipt.sequence
```

`prediction.mean` is the predicted next-observation vector; `prediction.adequacy` reflects bounded evidence status. A new model initially reports insufficient evidence. Supply strictly increasing nonnegative sequence numbers and dimensions matching the configuration; observation, action, and next-observation values must satisfy the configured bounds (default `[-1, 1]`). `source` identifies the data origin for the record; it is **not** an authentication or authorization claim.

## LGCM-0 implementation

The current research implementation uses:

- a slow shared dynamics model;
- persistent context-specific residual experts;
- label-free context inference from predictive evidence, with a configurable absolute standardized-error ceiling for recalling dormant experts;
- explicit mismatch detection with switch/spawn consequences;
- bounded expert capacity with no silent eviction;
- deterministic fixed feature encoders in V0, with representation learning reserved for later qualification;
- a baseline ladder that includes persistence, observation-only prediction, global cumulative ridge, and exponentially weighted RLS;
- prequential evaluation where prediction is fixed before the current outcome can update learner state;
- restart-continuity tests and evaluator/learner separation.

Evaluator-only phase labels distinguish first exposure to regime `A` from `A_RETURN`; these labels are never passed to the learner. A bounded model-based planner and learned representations remain future work, not implemented capabilities. The original one-model cumulative ridge idea remains in the project as a **baseline**, not as the LGCM candidate.

## Status and operational limits

**Initial implementation / active qualification.**

The repository contains focused tests for baselines, change detection, context gating and recall, fixed encoders, prequential evaluation, bounded model capacity, persistence, restart equivalence, residual experts, and RLS primitives. Source and tests do not by themselves establish broader continual-learning performance or safe integration into another product.

`update()`, `predict()`, `save_snapshot()`, and the public `active_expert_id`, `expert_ids`, `expert_weights()`, and `shared_weights` reads serialize on the model-instance lock. A read requested while an update is in progress waits for its committed boundary instead of returning mixed state. This is an **in-process** guarantee; snapshots of the same destination from separate processes or model instances require external coordination. Snapshots do not provide transactional rollback if an update itself fails after modifying model state.

LGCM does **not** provide host authorization, immutable user or provider identity bindings, consent management, permission inheritance, network execution, or an agent control plane. Calling its update or prediction APIs cannot grant an application permission to act. Any embedding host must independently authenticate callers, bind provider identity, authorize effects, isolate model state, and validate data and outputs. In particular, this code does not train or modify the weights of a hosted or pretrained language model.

## Reproduce bounded evidence

Use Python 3.12+ with NumPy 2+ and pytest 8+ in an isolated environment. Inspect the source and dependencies before installing or executing code.

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python experiments/heldout_probe.py
```

The diagnostic probe compares LGCM against persistence, observation-only, cumulative ridge, and EW-RLS baselines on four simulator seeds with abrupt and gradual A/B/A-return/C regimes. It scores each prediction **before** the current outcome updates the model. It is not a generalization or continual-learning qualification gate: novel regime C can favor simpler baselines, and the default context-recall threshold is not statistically calibrated.

The former README linked two `docs/` design files not present in the checked repository tree. For currently inspectable behavior, use `src/lgcm/`, `tests/`, and `experiments/` rather than treating those missing paths as available design authority.

The repository disables VS Code automatic workspace tasks and includes source-only startup guards (`tests/test_workspace_startup_safety.py`). Do not reenable folder-open shell execution or run non-code assets as programs.

## Project boundary

LGCM is a standalone research library. A host application's use of LGCM requires separate interface, safety, authorization, and performance qualification; presence in a larger ecosystem does not establish that integration.

## License

This repository is source-visible but not open source. Noncommercial evaluation, research, security review, interoperability assessment, and contribution preparation are permitted under `LICENSE`. Commercial use requires a separate written license; see `COMMERCIAL_LICENSE.md`.
