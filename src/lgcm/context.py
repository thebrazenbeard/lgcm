from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .residuals import ResidualScaleTracker
from .rls import RLSRegressor


@dataclass
class ContextExpert:
    expert_id: int
    regressor: RLSRegressor
    residual_scale: ResidualScaleTracker

    @property
    def evidence_count(self) -> int:
        return self.regressor.evidence_count


@dataclass(frozen=True)
class ExpertScore:
    expert_id: int
    standardized_error: float
    feature_support: float
    total_score: float
    adequate: bool


class ContextDecisionKind(str, Enum):
    KEEP = "keep"
    SWITCH = "switch"
    MISMATCH = "mismatch"
    SPAWN = "spawn"
    CAPACITY_EXHAUSTED = "capacity_exhausted"


@dataclass(frozen=True)
class ContextDecision:
    kind: ContextDecisionKind
    expert_id: int | None = None


class ContextGate:
    def __init__(
        self,
        *,
        max_experts: int,
        switch_margin: float,
        support_penalty: float,
        spawn_patience: int,
    ) -> None:
        if max_experts < 1:
            raise ValueError("max_experts must be positive")
        if not np.isfinite(switch_margin) or switch_margin < 0:
            raise ValueError("switch_margin must be finite and non-negative")
        if not np.isfinite(support_penalty) or support_penalty < 0:
            raise ValueError("support_penalty must be finite and non-negative")
        if spawn_patience < 1:
            raise ValueError("spawn_patience must be positive")
        self.max_experts = int(max_experts)
        self.switch_margin = float(switch_margin)
        self.support_penalty = float(support_penalty)
        self.spawn_patience = int(spawn_patience)
        self._unexplained_streak = 0

    @property
    def unexplained_streak(self) -> int:
        return self._unexplained_streak

    def reset_after_transition(self) -> None:
        self._unexplained_streak = 0

    def score_experts(
        self,
        experts: list[ContextExpert] | tuple[ContextExpert, ...],
        phi,
        context_target,
        *,
        active_expert_id: int,
    ) -> tuple[ExpertScore, ...]:
        features = np.ascontiguousarray(phi, dtype=np.float64)
        target = np.ascontiguousarray(context_target, dtype=np.float64)
        scores: list[ExpertScore] = []
        for expert in experts:
            prediction = expert.regressor.predict(features)
            residual = target - prediction
            support = expert.regressor.feature_support(features)
            adequate = expert.residual_scale.has_evidence
            if adequate:
                standardized = expert.residual_scale.standardized_magnitude(
                    residual
                )
            else:
                standardized = float("inf")
            total = standardized + self.support_penalty * (1.0 - support)
            scores.append(
                ExpertScore(
                    expert_id=expert.expert_id,
                    standardized_error=float(standardized),
                    feature_support=float(support),
                    total_score=float(total),
                    adequate=adequate,
                )
            )
        if not any(score.expert_id == active_expert_id for score in scores):
            raise ValueError("active expert is not present in expert bank")
        return tuple(scores)

    def decide(
        self,
        scores: tuple[ExpertScore, ...] | list[ExpertScore],
        *,
        active_expert_id: int,
        mismatch_triggered: bool,
        expert_count: int,
    ) -> ContextDecision:
        by_id = {score.expert_id: score for score in scores}
        if active_expert_id not in by_id:
            raise ValueError("active expert score missing")
        if not mismatch_triggered:
            self._unexplained_streak = 0
            return ContextDecision(ContextDecisionKind.KEEP, active_expert_id)

        active = by_id[active_expert_id]
        candidates = [
            score
            for score in scores
            if score.expert_id != active_expert_id and score.adequate
        ]
        if candidates:
            best = min(candidates, key=lambda item: item.total_score)
            if (
                not active.adequate
                or best.total_score + self.switch_margin < active.total_score
            ):
                self.reset_after_transition()
                return ContextDecision(ContextDecisionKind.SWITCH, best.expert_id)

        self._unexplained_streak += 1
        if self._unexplained_streak < self.spawn_patience:
            return ContextDecision(ContextDecisionKind.MISMATCH, active_expert_id)

        self.reset_after_transition()
        if expert_count >= self.max_experts:
            return ContextDecision(ContextDecisionKind.CAPACITY_EXHAUSTED, None)
        return ContextDecision(ContextDecisionKind.SPAWN, None)
