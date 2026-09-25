from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ResidualScaleTracker:
    def __init__(
        self,
        output_dim: int,
        *,
        decay: float = 0.95,
        floor: float = 1e-6,
        min_evidence: int = 4,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("floor must be finite and positive")
        if min_evidence < 1:
            raise ValueError("min_evidence must be positive")
        self.output_dim = output_dim
        self.decay = float(decay)
        self.floor = float(floor)
        self.min_evidence = int(min_evidence)
        self._mean_square = np.zeros(output_dim, dtype=np.float64)
        self.evidence_count = 0

    @property
    def scale(self) -> NDArray[np.float64]:
        value = np.sqrt(np.maximum(self._mean_square, self.floor**2))
        value = np.ascontiguousarray(value, dtype=np.float64)
        value.setflags(write=False)
        return value

    @property
    def has_evidence(self) -> bool:
        return self.evidence_count >= self.min_evidence

    def standardized_magnitude(self, residual) -> float:
        value = np.ascontiguousarray(residual, dtype=np.float64)
        if value.ndim != 1 or value.shape[0] != self.output_dim:
            raise ValueError(f"residual must have shape ({self.output_dim},)")
        if not np.all(np.isfinite(value)):
            raise ValueError("residual must contain only finite values")
        z = value / self.scale
        return float(np.sqrt(np.mean(z * z)))

    def update(self, residual) -> None:
        value = np.ascontiguousarray(residual, dtype=np.float64)
        if value.ndim != 1 or value.shape[0] != self.output_dim:
            raise ValueError(f"residual must have shape ({self.output_dim},)")
        if not np.all(np.isfinite(value)):
            raise ValueError("residual must contain only finite values")
        square = value * value
        if self.evidence_count == 0:
            new_mean_square = square
        else:
            new_mean_square = (
                self.decay * self._mean_square
                + (1.0 - self.decay) * square
            )
        if not np.all(np.isfinite(new_mean_square)):
            raise FloatingPointError("residual scale became non-finite")
        self._mean_square = np.ascontiguousarray(new_mean_square, dtype=np.float64)
        self.evidence_count += 1
