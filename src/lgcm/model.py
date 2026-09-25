from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .change_detection import PageHinkleyDetector
from .context import (
    ContextDecisionKind,
    ContextExpert,
    ContextGate,
)
from .encoders import IdentityFeatureEncoder
from .residuals import ResidualScaleTracker
from .rls import RLSRegressor
from .types import AdequacyState, Experience, Prediction


@dataclass(frozen=True)
class LGCMConfig:
    observation_dim: int
    action_dim: int
    regularization: float = 1.0
    shared_forgetting_factor: float = 1.0
    expert_forgetting_factor: float = 0.98
    residual_decay: float = 0.95
    residual_floor: float = 1e-3
    min_expert_evidence: int = 4
    max_experts: int = 8
    switch_margin: float = 0.25
    support_penalty: float = 0.25
    spawn_patience: int = 2
    mismatch_delta: float = 0.05
    mismatch_threshold: float = 5.0
    mismatch_min_evidence: int = 8
    bootstrap_size: int = 8
    new_expert_init: str = "shared_prior"
    lower_bound: float = -1.0
    upper_bound: float = 1.0

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if self.max_experts < 1:
            raise ValueError("max_experts must be positive")
        if self.bootstrap_size < 1:
            raise ValueError("bootstrap_size must be positive")
        if self.new_expert_init not in {"neutral", "shared_prior"}:
            raise ValueError("new_expert_init must be neutral or shared_prior")
        if not self.lower_bound < self.upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")


@dataclass(frozen=True)
class BootstrapRecord:
    encoded_features: NDArray[np.float64]
    target_delta: NDArray[np.float64]

    def __post_init__(self) -> None:
        features = np.ascontiguousarray(self.encoded_features, dtype=np.float64)
        target = np.ascontiguousarray(self.target_delta, dtype=np.float64)
        if features.ndim != 1 or target.ndim != 1:
            raise ValueError("bootstrap arrays must be one-dimensional")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
            raise ValueError("bootstrap arrays must be finite")
        features.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "encoded_features", features)
        object.__setattr__(self, "target_delta", target)


@dataclass(frozen=True)
class UpdateReceipt:
    generation: int
    sequence: int
    decision: ContextDecisionKind
    active_expert: int
    spawned_expert: int | None
    expert_count: int
    mismatch_score: float


class ContextualWorldModel:
    def __init__(self, config: LGCMConfig, *, encoder=None) -> None:
        self.config = config
        self.encoder = encoder or IdentityFeatureEncoder(
            config.observation_dim,
            config.action_dim,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
        )
        self._shared = RLSRegressor(
            self.encoder.feature_dim,
            config.observation_dim,
            regularization=config.regularization,
            forgetting_factor=config.shared_forgetting_factor,
        )
        self._experts: dict[int, ContextExpert] = {
            0: self._new_expert(0, initial_weights=None)
        }
        self._active_expert_id = 0
        self._next_expert_id = 1
        self._gate = ContextGate(
            max_experts=config.max_experts,
            switch_margin=config.switch_margin,
            support_penalty=config.support_penalty,
            spawn_patience=config.spawn_patience,
        )
        self._mismatch = PageHinkleyDetector(
            delta=config.mismatch_delta,
            threshold=config.mismatch_threshold,
            min_evidence=config.mismatch_min_evidence,
        )
        self._bootstrap: deque[BootstrapRecord] = deque(
            maxlen=config.bootstrap_size
        )
        self.generation = 0
        self.last_sequence = -1
        self._capacity_exhausted = False
        self._last_decision = ContextDecisionKind.KEEP

    def _new_expert(self, expert_id: int, *, initial_weights) -> ContextExpert:
        regressor = RLSRegressor(
            self.encoder.feature_dim,
            self.config.observation_dim,
            regularization=self.config.regularization,
            forgetting_factor=self.config.expert_forgetting_factor,
            initial_weights=initial_weights,
        )
        residual_scale = ResidualScaleTracker(
            self.config.observation_dim,
            decay=self.config.residual_decay,
            floor=self.config.residual_floor,
            min_evidence=self.config.min_expert_evidence,
        )
        return ContextExpert(expert_id, regressor, residual_scale)

    @property
    def active_expert_id(self) -> int:
        return self._active_expert_id

    @property
    def expert_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._experts))

    def expert_weights(self, expert_id: int) -> NDArray[np.float64]:
        return self._experts[expert_id].regressor.weights

    @property
    def shared_weights(self) -> NDArray[np.float64]:
        return self._shared.weights

    def _active_expert(self) -> ContextExpert:
        return self._experts[self._active_expert_id]

    def _validate_next_observation(self, value: NDArray[np.float64]) -> None:
        if value.shape != (self.config.observation_dim,):
            raise ValueError("experience next_observation dimension mismatch")
        if (
            np.any(value < self.config.lower_bound)
            or np.any(value > self.config.upper_bound)
        ):
            raise ValueError("next_observation must remain inside configured bounds")

    def predict(self, observation, action) -> Prediction:
        phi = self.encoder.encode(observation, action)
        expert = self._active_expert()
        delta = expert.regressor.predict(phi)
        mean = np.ascontiguousarray(observation, dtype=np.float64) + delta

        if self._capacity_exhausted:
            adequacy = AdequacyState.CAPACITY_EXHAUSTED
        elif not expert.residual_scale.has_evidence:
            adequacy = AdequacyState.INSUFFICIENT_EVIDENCE
        elif (
            self._mismatch.state.triggered
            or self._last_decision is ContextDecisionKind.MISMATCH
        ):
            adequacy = AdequacyState.SUSPECT_MODEL_MISMATCH
        else:
            adequacy = AdequacyState.ADEQUATE_WITHIN_EVIDENCE

        return Prediction(
            mean=mean,
            feature_support=expert.regressor.feature_support(phi),
            residual_scale=expert.residual_scale.scale,
            mismatch_score=self._mismatch.state.score,
            adequacy=adequacy,
            active_expert=self._active_expert_id,
        )

    def _spawn_expert(self) -> int:
        expert_id = self._next_expert_id
        initial_weights = (
            self._shared.weights
            if self.config.new_expert_init == "shared_prior"
            else None
        )
        expert = self._new_expert(expert_id, initial_weights=initial_weights)
        for record in self._bootstrap:
            prediction = expert.regressor.predict(record.encoded_features)
            residual = record.target_delta - prediction
            expert.regressor.update(
                record.encoded_features,
                record.target_delta,
            )
            expert.residual_scale.update(residual)
        self._experts[expert_id] = expert
        self._next_expert_id += 1
        self._active_expert_id = expert_id
        return expert_id

    def update(self, experience: Experience) -> UpdateReceipt:
        if self._capacity_exhausted:
            raise RuntimeError("LGCM context capacity exhausted")
        if experience.sequence <= self.last_sequence:
            raise ValueError("sequence must be strictly increasing")
        if experience.observation.shape != (self.config.observation_dim,):
            raise ValueError("experience observation dimension mismatch")
        if experience.action.shape != (self.config.action_dim,):
            raise ValueError("experience action dimension mismatch")
        self._validate_next_observation(experience.next_observation)

        phi = self.encoder.encode(experience.observation, experience.action)
        target_delta = np.ascontiguousarray(
            experience.next_observation - experience.observation,
            dtype=np.float64,
        )
        scores = self._gate.score_experts(
            list(self._experts.values()),
            phi,
            target_delta,
            active_expert_id=self._active_expert_id,
        )
        active_score = next(
            score
            for score in scores
            if score.expert_id == self._active_expert_id
        )

        mismatch_triggered = False
        if self._active_expert().residual_scale.has_evidence:
            mismatch_state = self._mismatch.update(
                active_score.standardized_error
            )
            mismatch_triggered = mismatch_state.triggered

        if mismatch_triggered:
            self._bootstrap.append(BootstrapRecord(phi, target_delta))
        else:
            self._bootstrap.clear()

        decision = self._gate.decide(
            scores,
            active_expert_id=self._active_expert_id,
            mismatch_triggered=mismatch_triggered,
            expert_count=len(self._experts),
        )
        spawned: int | None = None

        if decision.kind is ContextDecisionKind.SWITCH:
            assert decision.expert_id is not None
            self._active_expert_id = decision.expert_id
            self._mismatch.reset()
            self._bootstrap.clear()
        elif decision.kind is ContextDecisionKind.SPAWN:
            spawned = self._spawn_expert()
            self._mismatch.reset()
            self._bootstrap.clear()
        elif decision.kind is ContextDecisionKind.CAPACITY_EXHAUSTED:
            self._capacity_exhausted = True
            self._last_decision = decision.kind
            self.last_sequence = experience.sequence
            self.generation += 1
            return UpdateReceipt(
                generation=self.generation,
                sequence=experience.sequence,
                decision=decision.kind,
                active_expert=self._active_expert_id,
                spawned_expert=None,
                expert_count=len(self._experts),
                mismatch_score=self._mismatch.state.score,
            )

        # The shared model learns every committed non-terminal event.
        self._shared.update(phi, target_delta)

        # A spawn has already consumed the current event through bootstrap replay.
        # During an unexplained mismatch we freeze the old expert to protect
        # retained competence while bounded evidence accumulates.
        if decision.kind not in {
            ContextDecisionKind.SPAWN,
            ContextDecisionKind.MISMATCH,
        }:
            expert = self._active_expert()
            prediction = expert.regressor.predict(phi)
            residual = target_delta - prediction
            expert.regressor.update(phi, target_delta)
            expert.residual_scale.update(residual)

        self._last_decision = decision.kind
        self.last_sequence = experience.sequence
        self.generation += 1
        return UpdateReceipt(
            generation=self.generation,
            sequence=experience.sequence,
            decision=decision.kind,
            active_expert=self._active_expert_id,
            spawned_expert=spawned,
            expert_count=len(self._experts),
            mismatch_score=self._mismatch.state.score,
        )
