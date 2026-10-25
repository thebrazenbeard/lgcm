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
PHASE_LABELS = {
    "A": "A_initial",
    "A_TO_B_DRIFT": "A_to_B_drift",
    "B": "B",
    "A_RETURN": "A_return",
    "C": "C",
}


def _models():
    return {
        "persistence": lgcm.PersistencePredictor(2, 1),
        "observation_only": lgcm.ObservationOnlyWorldModel(2, 1),
        "cumulative_ridge": lgcm.GlobalRidgeWorldModel(2, 1),
        "ewrls_098": lgcm.EWRLSWorldModel(2, 1, forgetting_factor=0.98),
        "lgcm_default": lgcm.ContextualWorldModel(lgcm.LGCMConfig(2, 1)),
    }


def measure(model, stream, *, gradual):
    # The existing gradual argument is retained for callers; event labels,
    # not fixed durations or a mode flag, define the reporting intervals.
    events = tuple(stream)
    expected = {label: 0 for label in PHASE_LABELS.values()}
    phase_errors = {label: [] for label in PHASE_LABELS.values()}
    for event in events:
        if event.regime not in PHASE_LABELS:
            raise ValueError(f"unknown evaluation regime: {event.regime}")
        expected[PHASE_LABELS[event.regime]] += 1

    errors = []
    terminal = None
    for event in events:
        experience = event.experience
        pred = model.predict(experience.observation, experience.action)
        error = float(np.mean((pred.mean - experience.next_observation) ** 2))
        errors.append(error)
        phase_errors[PHASE_LABELS[event.regime]].append(error)
        try:
            model.update(experience)
        except RuntimeError as exc:
            terminal = {"sequence": experience.sequence, "error": str(exc)}
            break

    by_phase = {
        label: (
            round(mean(phase_errors[label]), 8)
            if len(phase_errors[label]) == count
            else None
        )
        for label, count in expected.items()
        if count
    }
    return {
        "steps_scored": len(errors),
        "terminal": terminal,
        "phase_mse": by_phase,
        "overall_mse": (
            round(mean(errors), 8)
            if terminal is None and len(errors) == len(events) and errors
            else None
        ),
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
