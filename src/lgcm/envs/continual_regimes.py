from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..evaluation import EvaluatorEvent
from ..types import Experience


def _residual(regime: str, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    if regime == "A":
        return np.array(
            [0.16 * state[0] + 0.08 * action[0], -0.10 * state[1]],
            dtype=np.float64,
        )
    if regime == "B":
        return np.array(
            [-0.18 * state[0] - 0.10 * action[0], 0.17 * state[1]],
            dtype=np.float64,
        )
    if regime == "C":
        return np.array(
            [
                0.22 * state[0] * state[1] + 0.09 * action[0],
                -0.20 * state[0] * action[0] + 0.08 * state[1] ** 2,
            ],
            dtype=np.float64,
        )
    raise ValueError(f"unknown regime: {regime}")


def _step(
    state: np.ndarray,
    action: np.ndarray,
    *,
    base_a: np.ndarray,
    base_b: np.ndarray,
    base_c: np.ndarray,
    regime: str,
    blend: float | None = None,
) -> np.ndarray:
    base = base_a @ state + base_b @ action + 0.12 * np.sin(base_c @ state)
    if blend is None:
        extra = _residual(regime, state, action)
    else:
        extra = (
            (1.0 - blend) * _residual("A", state, action)
            + blend * _residual("B", state, action)
        )
    return np.tanh(base + extra).astype(np.float64)


def make_continual_regime_stream(
    *,
    seed: int,
    lengths: Mapping[str, int],
    gradual: bool,
    drift_steps: int = 8,
) -> tuple[EvaluatorEvent, ...]:
    required = ("A", "B", "A_RETURN", "C")
    if any(name not in lengths or int(lengths[name]) < 1 for name in required):
        raise ValueError("lengths must contain positive A, B, A_RETURN, and C")
    if gradual and drift_steps < 1:
        raise ValueError("drift_steps must be positive")

    rng = np.random.default_rng(seed)
    base_a = np.array([[0.52, 0.11], [-0.09, 0.48]], dtype=np.float64)
    base_a += rng.normal(0.0, 0.025, size=(2, 2))
    base_b = np.array([[0.42], [-0.31]], dtype=np.float64)
    base_b += rng.normal(0.0, 0.02, size=(2, 1))
    base_c = np.array([[0.65, -0.20], [0.20, 0.58]], dtype=np.float64)
    state = rng.uniform(-0.3, 0.3, size=2).astype(np.float64)

    events: list[EvaluatorEvent] = []
    sequence = 0

    def emit(regime_label: str, dynamics_regime: str, count: int, *, blend=None):
        nonlocal state, sequence
        for i in range(count):
            action = rng.uniform(-1.0, 1.0, size=1).astype(np.float64)
            local_blend = blend(i) if callable(blend) else blend
            next_state = _step(
                state,
                action,
                base_a=base_a,
                base_b=base_b,
                base_c=base_c,
                regime=dynamics_regime,
                blend=local_blend,
            )
            experience = Experience(
                sequence=sequence,
                observation=state.copy(),
                action=action.copy(),
                next_observation=next_state.copy(),
                source="simulator",
            )
            events.append(EvaluatorEvent(experience=experience, regime=regime_label))
            state = next_state
            sequence += 1

    emit("A", "A", int(lengths["A"]))
    if gradual:
        denom = max(1, drift_steps - 1)
        emit(
            "A_TO_B_DRIFT",
            "A",
            drift_steps,
            blend=lambda i: i / denom,
        )
    emit("B", "B", int(lengths["B"]))
    emit("A", "A", int(lengths["A_RETURN"]))
    emit("C", "C", int(lengths["C"]))
    return tuple(events)
