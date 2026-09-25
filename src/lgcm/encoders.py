from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import FloatVector


def _input_vector(value: NDArray[np.float64] | list[float] | tuple[float, ...], *, dim: int, name: str) -> FloatVector:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != dim:
        raise ValueError(f"{name} must have shape ({dim},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class IdentityFeatureEncoder:
    observation_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")

    @property
    def feature_dim(self) -> int:
        return 1 + self.observation_dim + self.action_dim

    def encode(self, observation: FloatVector, action: FloatVector) -> FloatVector:
        obs = _input_vector(observation, dim=self.observation_dim, name="observation")
        act = _input_vector(action, dim=self.action_dim, name="action")
        features = np.concatenate((np.ones(1, dtype=np.float64), obs, act))
        features.setflags(write=False)
        return features


class RandomFeatureEncoder:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        width: int,
        seed: int,
        max_feature_dim: int = 128,
    ) -> None:
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.width = width
        self.seed = seed
        self.max_feature_dim = max_feature_dim
        raw_dim = observation_dim + action_dim
        total = 1 + raw_dim + width
        if total > max_feature_dim:
            raise ValueError(
                f"feature dimension {total} exceeds configured maximum {max_feature_dim}"
            )
        rng = np.random.default_rng(seed)
        projection = rng.normal(0.0, 1.0 / np.sqrt(raw_dim), size=(width, raw_dim))
        self._projection = np.ascontiguousarray(projection, dtype=np.float64)
        self._projection.setflags(write=False)

    @property
    def feature_dim(self) -> int:
        return 1 + self.observation_dim + self.action_dim + self.width

    @property
    def projection(self) -> NDArray[np.float64]:
        return self._projection

    def encode(self, observation: FloatVector, action: FloatVector) -> FloatVector:
        obs = _input_vector(observation, dim=self.observation_dim, name="observation")
        act = _input_vector(action, dim=self.action_dim, name="action")
        raw = np.concatenate((obs, act))
        nonlinear = np.tanh(self._projection @ raw)
        features = np.concatenate((np.ones(1, dtype=np.float64), raw, nonlinear))
        features = np.ascontiguousarray(features, dtype=np.float64)
        features.setflags(write=False)
        return features
