"""Reproducible LGCM-0 prequential baseline probe on unseen simulator seeds.

Run with ``PYTHONPATH=src python experiments/heldout_probe.py``.
This is a diagnostic measurement, not a performance qualification gate.
"""

from __future__ import annotations

import json
from statistics import mean

import numpy as np

import lgcm


SEEDS = (11, 23, 41, 59)
LENGTHS = {"A": 100, "B": 100, "A_RETURN": 100, "C": 100}
PHASES = ("A_initial", "B", "A_return", "C")


def _models():
    return {
        "persistence": lgcm.PersistencePredictor(2, 1),
        "observation_only": lgcm.ObservationOnlyWorldModel(2, 1),
        "cumulative_ridge": lgcm.GlobalRidgeWorldModel(2, 1),
        "ewrls_098": lgcm.EWRLSWorldModel(2, 1, forgetting_factor=0.98),
        "lgcm_default": lgcm.ContextualWorldModel(lgcm.LGCMConfig(2, 1)),
    }


def measure(model, stream, *, gradual):
    errors = []
    terminal = None
    for event in stream:
        experience = event.experience
        pred = model.predict(experience.observation, experience.action)
        errors.append(float(np.mean((pred.mean - experience.next_observation) ** 2)))
        try:
            model.update(experience)
        except RuntimeError as exc:
            terminal = {"sequence": experience.sequence, "error": str(exc)}
            break

    by_phase = {}
    first = 0
    sizes = [LENGTHS[key] for key in ("A", "B", "A_RETURN", "C")]
    labels = list(PHASES)
    if gradual:
        sizes.insert(1, 8)
        labels.insert(1, "A_to_B_drift")
    for label, size in zip(labels, sizes, strict=True):
        phase = errors[first : first + size]
        by_phase[label] = round(mean(phase), 8) if len(phase) == size else None
        first += size
    return {
        "steps_scored": len(errors),
        "terminal": terminal,
        "phase_mse": by_phase,
        "overall_mse": round(mean(errors), 8) if errors else None,
        "expert_count": len(model.expert_ids) if hasattr(model, "expert_ids") else None,
    }


def main():
    out = {"seeds": SEEDS, "lengths": LENGTHS, "runs": {}}
    for gradual in (False, True):
        mode = "gradual" if gradual else "abrupt"
        out["runs"][mode] = {}
        for seed in SEEDS:
            events = lgcm.make_continual_regime_stream(
                seed=seed, lengths=LENGTHS, gradual=gradual, drift_steps=8
            )
            out["runs"][mode][str(seed)] = {
                name: measure(model, events, gradual=gradual)
                for name, model in _models().items()
            }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
