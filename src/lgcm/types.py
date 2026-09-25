from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]


class AdequacyState(str, Enum):
    ADEQUATE_WITHIN_EVIDENCE = "adequate_within_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SUSPECT_MODEL_MISMATCH = "suspect_model_mismatch"
    CAPACITY_EXHAUSTED = "capacity_exhausted"


def _readonly_vector(value: NDArray[np.float64] | list[float] | tuple[float, ...], *, name: str) -> FloatVector:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Experience:
    sequence: int
    observation: FloatVector
    action: FloatVector
    next_observation: FloatVector
    source: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not self.source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "observation", _readonly_vector(self.observation, name="observation"))
        object.__setattr__(self, "action", _readonly_vector(self.action, name="action"))
        object.__setattr__(
            self,
            "next_observation",
            _readonly_vector(self.next_observation, name="next_observation"),
        )


@dataclass(frozen=True)
class Prediction:
    mean: FloatVector
    feature_support: float
    residual_scale: FloatVector
    mismatch_score: float
    adequacy: AdequacyState
    active_expert: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _readonly_vector(self.mean, name="mean"))
        object.__setattr__(
            self,
            "residual_scale",
            _readonly_vector(self.residual_scale, name="residual_scale"),
        )
        if not np.isfinite(self.feature_support):
            raise ValueError("feature_support must be finite")
        if not np.isfinite(self.mismatch_score):
            raise ValueError("mismatch_score must be finite")
