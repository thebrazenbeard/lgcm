> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

# LGCM

**LGCM** is an experimental task-free continual dynamics learner for bounded numerical environments.

Its core question is deliberately falsifiable:

> Can an online learner build an action-conditioned predictive model, adapt when latent dynamics change, preserve useful prior competence, recall a prior internal context when it returns, and use the learned model for bounded planning without being told which task or regime it is in?

LGCM-0 does **not** claim AGI, general perception, semantic understanding, consciousness, unrestricted agency, or deployment readiness.

## What is implemented

Current source includes:

- a slow shared action-conditioned RLS dynamics model;
- persistent context-specific residual experts;
- label-free expert scoring and switching;
- Page-Hinkley mismatch detection;
- bounded expert spawning with explicit capacity exhaustion and no silent eviction;
- deterministic identity and random-feature encoders;
- persistence, digest verification, snapshot path confinement, and restart equivalence;
- prequential evaluation with evaluator/learner separation;
- persistence, observation-only, global-ridge, and EW-RLS baselines;
- abrupt, gradual, recurrent, and novel-regime simulator segments;
- deterministic multi-seed qualification receipts with seed-level and aggregate metrics;
- bounded seeded random-shooting model-predictive planning that never updates the learner.

The original single cumulative ridge model is retained as a baseline, not treated as the LGCM candidate.

## Quick start

Requires Python 3.12 or newer.

```text
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```

On POSIX systems use `.venv/bin/python`.

A minimal learner:

```python
import numpy as np
import lgcm

model = lgcm.ContextualWorldModel(
    lgcm.LGCMConfig(observation_dim=2, action_dim=1)
)

prediction = model.predict(
    np.array([0.1, -0.1]),
    np.array([0.2]),
)
print(prediction.mean)
```

## Qualification

The qualification runner evaluates the candidate and all four baselines over the same seeded streams:

```python
import lgcm

result = lgcm.run_qualification(
    seeds=(0, 1, 2, 3, 4),
    lengths={"A": 80, "B": 80, "A_RETURN": 80, "C": 80},
    gradual_modes=(False, True),
)

print(result.receipt.canonical_json())
```

The canonical receipt excludes wall-clock timing. Resource timing is available separately through `result.resources`.

Do not infer a broad continual-learning claim from one seed or one aggregate metric. See [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).

## Bounded planning

```python
import numpy as np
import lgcm

result = lgcm.plan_action(
    model,
    np.array([0.1, -0.1]),
    lambda state: float(np.sum(state * state)),
    horizon=4,
    candidates=256,
    seed=7,
    lower_bound=-1.0,
    upper_bound=1.0,
)

print(result.action, result.score)
```

Planning calls prediction only. It does not learn from imagined transitions.

## Verification

Repository gates are:

```text
python -m pytest -q --cov=lgcm --cov-report=term-missing
ruff check src tests
python -m mypy src/lgcm
python -m build --sdist --wheel
python -m pip check
```

CI exercises Python 3.12, 3.13, and 3.14 on Ubuntu and Windows. Python 3.15 prerelease compatibility is non-blocking until 3.15 final.

## Documentation

- [Design specification V2](docs/LGCM_DESIGN_SPEC_V2.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Approved qualification-first design](docs/specs/2026-09-27-qualification-first-upgrade-design.md)
- [Implementation plan](docs/plans/2026-09-27-qualification-foundation.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

## Project boundary

LGCM is a standalone research implementation. Repository presence does not imply Vera Mono integration, installation, runtime selection, behavioral qualification, or deployment.

## License

This repository is source-visible but not open source. Noncommercial evaluation, research, security review, interoperability assessment, and contribution preparation are permitted under [LICENSE](LICENSE). Commercial use requires a separate written license; see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
