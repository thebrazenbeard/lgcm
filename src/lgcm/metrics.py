from __future__ import annotations

from statistics import fmean

from .evaluation import EvaluationTrace


def interval_mse(trace: EvaluationTrace, *, regime: str | None = None) -> float:
    values = [
        step.mse for step in trace.steps if regime is None or step.regime == regime
    ]
    if not values:
        raise ValueError("no evaluation steps match requested interval")
    return float(fmean(values))


def samples_to_criterion(
    errors,
    *,
    threshold: float,
    consecutive: int = 1,
) -> int | None:
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    run = 0
    for index, value in enumerate(errors, start=1):
        if value <= threshold:
            run += 1
            if run >= consecutive:
                return index
        else:
            run = 0
    return None


def forgetting_delta(*, pre_error: float, return_error: float) -> float:
    return float(round(return_error - pre_error, 12))


def forward_transfer_ratio(
    *,
    fresh_samples: int | float,
    experienced_samples: int | float,
) -> float:
    if fresh_samples <= 0 or experienced_samples <= 0:
        raise ValueError("sample counts must be positive")
    return float(fresh_samples / experienced_samples)


def adaptation_error_auc(errors) -> float:
    values = [float(value) for value in errors]
    if not values:
        raise ValueError("errors must not be empty")
    return float(sum(values))
