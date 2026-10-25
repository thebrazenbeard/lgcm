"""Regression controls for diagnostic phase and truncation reporting."""

from pathlib import Path
import runpy
from statistics import mean

import numpy as np

import lgcm


measure = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "experiments" / "heldout_probe.py")
)["measure"]


LABELS = {
    "A": "A_initial",
    "A_TO_B_DRIFT": "A_to_B_drift",
    "B": "B",
    "A_RETURN": "A_return",
    "C": "C",
}


def _stream(*, gradual, drift_steps=5):
    return lgcm.make_continual_regime_stream(
        seed=37,
        lengths={"A": 2, "B": 3, "A_RETURN": 4, "C": 5},
        gradual=gradual,
        drift_steps=drift_steps,
    )


def test_phase_windows_follow_event_labels_not_default_lengths():
    stream = _stream(gradual=True, drift_steps=5)
    expected = {}
    for regime, label in LABELS.items():
        events = [event.experience for event in stream if event.regime == regime]
        expected[label] = round(
            mean(float(np.mean((e.next_observation - e.observation) ** 2)) for e in events),
            8,
        )

    result = measure(lgcm.PersistencePredictor(2, 1), stream, gradual=True)
    assert result["terminal"] is None
    assert result["steps_scored"] == len(stream) == 19
    assert result["phase_mse"] == expected
    assert result["overall_mse"] is not None


class _StopsAfterTwo:
    def predict(self, observation, action):
        return lgcm.Prediction(
            mean=np.zeros(2),
            feature_support=1.0,
            residual_scale=np.ones(2),
            mismatch_score=0.0,
            adequacy=lgcm.AdequacyState.INSUFFICIENT_EVIDENCE,
            active_expert=None,
        )

    def update(self, experience):
        if experience.sequence == 2:
            raise RuntimeError("synthetic capacity terminal")


def test_premature_terminal_cannot_return_comparable_aggregate():
    result = measure(_StopsAfterTwo(), _stream(gradual=False), gradual=False)
    assert result["steps_scored"] == 3
    assert result["terminal"] == {
        "sequence": 2,
        "error": "synthetic capacity terminal",
    }
    assert result["overall_mse"] is None
    assert result["phase_mse"]["A_initial"] is not None
    assert result["phase_mse"]["B"] is None
    assert result["phase_mse"]["A_return"] is None
    assert result["phase_mse"]["C"] is None
