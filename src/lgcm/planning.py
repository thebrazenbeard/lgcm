from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
Objective = Callable[[FloatVector], float]


@dataclass(frozen=True)
class PlanResult:
    action: FloatVector
    score: float
    candidate_index: int
    seed: int
    horizon: int
    candidates: int

    def __post_init__(self) -> None:
        action = np.array(self.action, dtype=np.float64, copy=True, order="C")
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite one-dimensional vector")
        action.setflags(write=False)
        object.__setattr__(self, "action", action)
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")


def _action_dim(model) -> int:
    value = getattr(model, "action_dim", None)
    if value is None:
        config = getattr(model, "config", None)
        value = getattr(config, "action_dim", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("model must expose a positive integer action dimension")
    return value


def plan_action(
    model,
    observation,
    objective: Objective,
    *,
    horizon: int,
    candidates: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> PlanResult:
    """Choose the first action from the lowest cumulative-cost sampled rollout.

    Planning is read-only with respect to the learner: this function calls
    model.predict only and never calls model.update.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1:
        raise ValueError("candidates must be a positive integer")
    if (
        not np.isfinite(lower_bound)
        or not np.isfinite(upper_bound)
        or not lower_bound < upper_bound
    ):
        raise ValueError("action bounds must be finite and lower_bound < upper_bound")

    state0 = np.array(observation, dtype=np.float64, copy=True, order="C")
    if state0.ndim != 1 or not np.all(np.isfinite(state0)):
        raise ValueError("observation must be a finite one-dimensional vector")

    action_dim = _action_dim(model)
    rng = np.random.default_rng(seed)
    sequences = rng.uniform(
        lower_bound,
        upper_bound,
        size=(candidates, horizon, action_dim),
    ).astype(np.float64)

    best_index = -1
    best_score = float("inf")
    for candidate_index, sequence in enumerate(sequences):
        state = state0.copy()
        score = 0.0
        for action in sequence:
            prediction = model.predict(state, action)
            next_state = np.asarray(prediction.mean, dtype=np.float64)
            if (
                next_state.ndim != 1
                or next_state.shape != state.shape
                or not np.all(np.isfinite(next_state))
            ):
                raise ValueError("model prediction must preserve finite observation shape")
            step_score = float(objective(next_state))
            if not np.isfinite(step_score):
                raise ValueError("objective must return a finite scalar")
            score += step_score
            state = next_state

        if score < best_score:
            best_score = score
            best_index = candidate_index

    return PlanResult(
        action=sequences[best_index, 0],
        score=best_score,
        candidate_index=best_index,
        seed=int(seed),
        horizon=horizon,
        candidates=candidates,
    )
