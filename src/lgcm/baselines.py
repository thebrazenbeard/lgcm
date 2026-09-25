from __future__ import annotations

import numpy as np

from .encoders import IdentityFeatureEncoder
from .residuals import ResidualScaleTracker
from .rls import RLSRegressor
from .types import AdequacyState, Experience, Prediction


class PersistencePredictor:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        residual_decay: float = 0.95,
        min_evidence: int = 4,
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self._validator = IdentityFeatureEncoder(observation_dim, action_dim)
        self._residuals = ResidualScaleTracker(
            observation_dim,
            decay=residual_decay,
            min_evidence=min_evidence,
        )
        self.generation = 0
        self.last_sequence = -1

    def predict(self, observation, action) -> Prediction:
        self._validator.encode(observation, action)
        mean = np.ascontiguousarray(observation, dtype=np.float64)
        adequacy = (
            AdequacyState.ADEQUATE_WITHIN_EVIDENCE
            if self._residuals.has_evidence
            else AdequacyState.INSUFFICIENT_EVIDENCE
        )
        return Prediction(
            mean=mean,
            feature_support=1.0,
            residual_scale=self._residuals.scale,
            mismatch_score=0.0,
            adequacy=adequacy,
            active_expert=None,
        )

    def update(self, experience: Experience) -> None:
        if experience.sequence <= self.last_sequence:
            raise ValueError("sequence must be strictly increasing")
        prediction = self.predict(experience.observation, experience.action)
        residual = experience.next_observation - prediction.mean
        self._residuals.update(residual)
        self.last_sequence = experience.sequence
        self.generation += 1


class _RLSWorldModel:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        regularization: float,
        forgetting_factor: float,
        ignore_action: bool = False,
        residual_decay: float = 0.95,
        min_evidence: int = 4,
        encoder=None,
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.ignore_action = ignore_action
        self.encoder = encoder or IdentityFeatureEncoder(observation_dim, action_dim)
        self.regressor = RLSRegressor(
            self.encoder.feature_dim,
            observation_dim,
            regularization=regularization,
            forgetting_factor=forgetting_factor,
        )
        self._residuals = ResidualScaleTracker(
            observation_dim,
            decay=residual_decay,
            min_evidence=min_evidence,
        )
        self.generation = 0
        self.last_sequence = -1

    def _features(self, observation, action):
        if self.ignore_action:
            action = np.zeros(self.action_dim, dtype=np.float64)
        return self.encoder.encode(observation, action)

    def predict(self, observation, action) -> Prediction:
        # Validate the provided action even when the baseline ignores its value.
        self.encoder.encode(observation, action)
        phi = self._features(observation, action)
        delta = self.regressor.predict(phi)
        mean = np.ascontiguousarray(observation, dtype=np.float64) + delta
        adequacy = (
            AdequacyState.ADEQUATE_WITHIN_EVIDENCE
            if self._residuals.has_evidence
            else AdequacyState.INSUFFICIENT_EVIDENCE
        )
        return Prediction(
            mean=mean,
            feature_support=self.regressor.feature_support(phi),
            residual_scale=self._residuals.scale,
            mismatch_score=0.0,
            adequacy=adequacy,
            active_expert=None,
        )

    def update(self, experience: Experience) -> None:
        if experience.sequence <= self.last_sequence:
            raise ValueError("sequence must be strictly increasing")
        if experience.observation.shape[0] != self.observation_dim:
            raise ValueError("experience observation dimension mismatch")
        if experience.next_observation.shape[0] != self.observation_dim:
            raise ValueError("experience next_observation dimension mismatch")
        if experience.action.shape[0] != self.action_dim:
            raise ValueError("experience action dimension mismatch")
        prediction = self.predict(experience.observation, experience.action)
        phi = self._features(experience.observation, experience.action)
        target = experience.next_observation - experience.observation
        residual = experience.next_observation - prediction.mean
        self.regressor.update(phi, target)
        self._residuals.update(residual)
        self.last_sequence = experience.sequence
        self.generation += 1


class ObservationOnlyWorldModel(_RLSWorldModel):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        regularization: float = 1.0,
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            regularization=regularization,
            forgetting_factor=1.0,
            ignore_action=True,
        )


class GlobalRidgeWorldModel(_RLSWorldModel):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        regularization: float = 1.0,
        encoder=None,
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            regularization=regularization,
            forgetting_factor=1.0,
            encoder=encoder,
        )


class EWRLSWorldModel(_RLSWorldModel):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        regularization: float = 1.0,
        forgetting_factor: float = 0.99,
        encoder=None,
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            regularization=regularization,
            forgetting_factor=forgetting_factor,
            encoder=encoder,
        )
