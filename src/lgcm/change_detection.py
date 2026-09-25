from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PageHinkleyState:
    count: int
    mean: float
    cumulative: float
    minimum_cumulative: float
    score: float
    triggered: bool


class PageHinkleyDetector:
    def __init__(
        self,
        *,
        delta: float = 0.05,
        threshold: float = 5.0,
        min_evidence: int = 10,
    ) -> None:
        if not np.isfinite(delta) or delta < 0:
            raise ValueError("delta must be finite and non-negative")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if min_evidence < 1:
            raise ValueError("min_evidence must be positive")
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_evidence = int(min_evidence)
        self.reset()

    @property
    def state(self) -> PageHinkleyState:
        score = max(0.0, self._cumulative - self._minimum_cumulative)
        return PageHinkleyState(
            count=self._count,
            mean=self._mean,
            cumulative=self._cumulative,
            minimum_cumulative=self._minimum_cumulative,
            score=score,
            triggered=(
                self._count >= self.min_evidence and score > self.threshold
            ),
        )

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._cumulative = 0.0
        self._minimum_cumulative = 0.0

    def update(self, value: float) -> PageHinkleyState:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Page-Hinkley value must be finite")
        self._count += 1
        self._mean += (value - self._mean) / self._count
        self._cumulative += value - self._mean - self.delta
        self._minimum_cumulative = min(
            self._minimum_cumulative,
            self._cumulative,
        )
        return self.state
