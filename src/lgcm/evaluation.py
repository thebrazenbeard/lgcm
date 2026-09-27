from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Experience


@dataclass(frozen=True)
class EvaluatorEvent:
    experience: Experience
    regime: str
    segment: str | None = None


@dataclass(frozen=True)
class EvaluationStep:
    sequence: int
    regime: str
    mse: float
    active_expert: int | None
    adequacy: str
    segment: str | None = None
    post_update_active_expert: int | None = None
    decision: str | None = None
    expert_count: int | None = None
    spawned_expert: int | None = None


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
        receipt = model.update(experience)
        raw_decision = getattr(receipt, "decision", None)
        decision = getattr(raw_decision, "value", None)
        if decision is None and raw_decision is not None:
            decision = str(raw_decision)
        steps.append(
            EvaluationStep(
                sequence=experience.sequence,
                regime=event.regime,
                segment=event.segment or event.regime,
                mse=mse,
                active_expert=prediction.active_expert,
                post_update_active_expert=getattr(
                    receipt,
                    "active_expert",
                    prediction.active_expert,
                ),
                adequacy=adequacy,
                decision=decision,
                expert_count=getattr(receipt, "expert_count", None),
                spawned_expert=getattr(receipt, "spawned_expert", None),
            )
        )
    return EvaluationTrace(tuple(steps))
