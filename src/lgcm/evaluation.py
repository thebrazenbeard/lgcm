from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Experience


@dataclass(frozen=True)
class EvaluatorEvent:
    experience: Experience
    regime: str


@dataclass(frozen=True)
class EvaluationStep:
    sequence: int
    regime: str
    mse: float
    active_expert: int | None
    adequacy: str


@dataclass(frozen=True)
class EvaluationTrace:
    steps: tuple[EvaluationStep, ...]


def evaluate_prequential(model, events) -> EvaluationTrace:
    steps: list[EvaluationStep] = []
    for event in events:
        experience = event.experience
        prediction = model.predict(experience.observation, experience.action)
        error = experience.next_observation - prediction.mean
        mse = float(np.mean(error * error))
        adequacy = (
            prediction.adequacy.value
            if hasattr(prediction.adequacy, "value")
            else str(prediction.adequacy)
        )
        steps.append(
            EvaluationStep(
                sequence=experience.sequence,
                regime=event.regime,
                mse=mse,
                active_expert=prediction.active_expert,
                adequacy=adequacy,
            )
        )
        model.update(experience)
    return EvaluationTrace(tuple(steps))
