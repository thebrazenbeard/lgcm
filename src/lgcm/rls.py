from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _vector(value, *, dim: int, name: str) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != dim:
        raise ValueError(f"{name} must have shape ({dim},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class RLSUpdateReceipt:
    error: FloatArray
    gain_norm: float
    denominator: float
    condition_number: float
    evidence_count: int


class RLSRegressor:
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        *,
        regularization: float = 1.0,
        forgetting_factor: float = 1.0,
        initial_weights=None,
    ) -> None:
        if feature_dim <= 0 or output_dim <= 0:
            raise ValueError("feature_dim and output_dim must be positive")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if (
            not np.isfinite(forgetting_factor)
            or forgetting_factor <= 0
            or forgetting_factor > 1
        ):
            raise ValueError("forgetting_factor must satisfy 0 < gamma <= 1")
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.regularization = float(regularization)
        self.forgetting_factor = float(forgetting_factor)
        if initial_weights is None:
            weights = np.zeros((feature_dim, output_dim), dtype=np.float64)
        else:
            weights = np.ascontiguousarray(initial_weights, dtype=np.float64)
            if weights.shape != (feature_dim, output_dim):
                raise ValueError(
                    f"initial_weights must have shape ({feature_dim}, {output_dim})"
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError("initial_weights must contain only finite values")
        self._weights = weights.copy()
        self._covariance = (
            np.eye(feature_dim, dtype=np.float64) / self.regularization
        )
        self.evidence_count = 0

    @property
    def weights(self) -> FloatArray:
        value = self._weights.copy()
        value.setflags(write=False)
        return value

    @property
    def covariance(self) -> FloatArray:
        value = self._covariance.copy()
        value.setflags(write=False)
        return value

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self._covariance))

    def predict(self, phi) -> FloatArray:
        features = _vector(phi, dim=self.feature_dim, name="phi")
        result = np.ascontiguousarray(features @ self._weights, dtype=np.float64)
        result.setflags(write=False)
        return result

    def feature_support(self, phi) -> float:
        features = _vector(phi, dim=self.feature_dim, name="phi")
        leverage = float(features @ self._covariance @ features)
        if not np.isfinite(leverage):
            raise FloatingPointError("feature leverage became non-finite")
        leverage = max(0.0, leverage)
        return float(1.0 / (1.0 + leverage))

    def update(self, phi, target) -> RLSUpdateReceipt:
        features = _vector(phi, dim=self.feature_dim, name="phi")
        expected = _vector(target, dim=self.output_dim, name="target")

        p_phi = self._covariance @ features
        denominator = float(
            self.forgetting_factor + features @ p_phi
        )
        if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
            raise FloatingPointError("RLS update denominator is invalid")

        prediction = features @ self._weights
        error = expected - prediction
        gain = p_phi / denominator
        new_weights = self._weights + np.outer(gain, error)
        row = features @ self._covariance
        new_covariance = (
            self._covariance - np.outer(gain, row)
        ) / self.forgetting_factor
        new_covariance = 0.5 * (new_covariance + new_covariance.T)

        if not np.all(np.isfinite(new_weights)):
            raise FloatingPointError("RLS weights became non-finite")
        if not np.all(np.isfinite(new_covariance)):
            raise FloatingPointError("RLS covariance became non-finite")
        condition = float(np.linalg.cond(new_covariance))
        if not np.isfinite(condition):
            raise FloatingPointError("RLS covariance condition became non-finite")

        self._weights = np.ascontiguousarray(new_weights, dtype=np.float64)
        self._covariance = np.ascontiguousarray(new_covariance, dtype=np.float64)
        self.evidence_count += 1

        receipt_error = np.ascontiguousarray(error, dtype=np.float64)
        receipt_error.setflags(write=False)
        return RLSUpdateReceipt(
            error=receipt_error,
            gain_norm=float(np.linalg.norm(gain)),
            denominator=denominator,
            condition_number=condition,
            evidence_count=self.evidence_count,
        )
