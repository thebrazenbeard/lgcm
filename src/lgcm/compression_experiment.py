from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PredictiveCompressionStep:
    sequence: int
    regime: str
    full_mse: float
    compressed_mse: float
    source_state_bytes: int
    latent_state_bytes: int
    backing_state_bytes: int
    rehydration_count: int = 0
    rehydration_bytes: int = 0
    old_context_recall_correct: bool | None = None

    def validate(self) -> None:
        if (
            type(self.sequence) is not int
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative exact int")
        if type(self.regime) is not str or not self.regime:
            raise ValueError("regime must be a non-empty exact string")
        if not math.isfinite(self.full_mse) or not math.isfinite(
            self.compressed_mse
        ):
            raise ValueError("prediction errors must be finite")
        if self.full_mse < 0 or self.compressed_mse < 0:
            raise ValueError("prediction errors must be non-negative")
        for value, label in (
            (self.source_state_bytes, "source_state_bytes"),
            (self.latent_state_bytes, "latent_state_bytes"),
            (self.backing_state_bytes, "backing_state_bytes"),
            (self.rehydration_count, "rehydration_count"),
            (self.rehydration_bytes, "rehydration_bytes"),
        ):
            if (
                type(value) is not int
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative exact int")
        if self.source_state_bytes < 1:
            raise ValueError("source_state_bytes must be positive")
        if self.latent_state_bytes < 1:
            raise ValueError("latent_state_bytes must be positive")
        if self.latent_state_bytes > self.source_state_bytes:
            raise ValueError(
                "latent_state_bytes must not exceed source_state_bytes"
            )
        if self.backing_state_bytes < self.source_state_bytes:
            raise ValueError(
                "backing_state_bytes must cover at least source state bytes"
            )
        if self.rehydration_count == 0 and self.rehydration_bytes != 0:
            raise ValueError(
                "rehydration_bytes requires a positive rehydration_count"
            )
        if self.rehydration_count > 0 and self.rehydration_bytes < 1:
            raise ValueError(
                "positive rehydration_count requires rehydration_bytes"
            )
        if (
            self.old_context_recall_correct is not None
            and type(self.old_context_recall_correct) is not bool
        ):
            raise ValueError(
                "old_context_recall_correct must be bool or None"
            )


@dataclass(frozen=True, slots=True)
class PredictiveCompressionReport:
    step_count: int
    regime_count: int
    mean_full_mse: float
    mean_compressed_mse: float
    mean_mse_delta: float
    source_state_bytes_total: int
    latent_state_bytes_total: int
    backing_state_bytes_total: int
    rehydration_count: int
    rehydration_bytes_total: int
    compression_ratio: float
    old_context_recall_trials: int
    old_context_recall_accuracy: float | None
    claim_ceiling: str = (
        "EXPERIMENT_PROTOCOL_ONLY_LGCM0_ENCODERS_AND_QUALIFICATION_UNCHANGED"
    )


def evaluate_predictive_compression(
    steps: Iterable[PredictiveCompressionStep],
) -> PredictiveCompressionReport:
    ordered = tuple(steps)
    if not ordered:
        raise ValueError("predictive compression trace must not be empty")
    for item in ordered:
        if type(item) is not PredictiveCompressionStep:
            raise TypeError(
                "steps must contain exact PredictiveCompressionStep values"
            )
        item.validate()

    sequences = [item.sequence for item in ordered]
    if any(right <= left for left, right in zip(sequences, sequences[1:])):
        raise ValueError("sequence must be strictly increasing")

    step_count = len(ordered)
    full_mean = sum(item.full_mse for item in ordered) / step_count
    compressed_mean = (
        sum(item.compressed_mse for item in ordered) / step_count
    )
    source_total = sum(item.source_state_bytes for item in ordered)
    latent_total = sum(item.latent_state_bytes for item in ordered)
    recall = [
        item.old_context_recall_correct
        for item in ordered
        if item.old_context_recall_correct is not None
    ]

    return PredictiveCompressionReport(
        step_count=step_count,
        regime_count=len({item.regime for item in ordered}),
        mean_full_mse=full_mean,
        mean_compressed_mse=compressed_mean,
        mean_mse_delta=compressed_mean - full_mean,
        source_state_bytes_total=source_total,
        latent_state_bytes_total=latent_total,
        backing_state_bytes_total=sum(
            item.backing_state_bytes for item in ordered
        ),
        rehydration_count=sum(
            item.rehydration_count for item in ordered
        ),
        rehydration_bytes_total=sum(
            item.rehydration_bytes for item in ordered
        ),
        compression_ratio=source_total / latent_total,
        old_context_recall_trials=len(recall),
        old_context_recall_accuracy=(
            None
            if not recall
            else sum(bool(value) for value in recall) / len(recall)
        ),
    )
