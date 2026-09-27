from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from statistics import fmean
from time import perf_counter

import numpy as np

from .baselines import (
    EWRLSWorldModel,
    GlobalRidgeWorldModel,
    ObservationOnlyWorldModel,
    PersistencePredictor,
)
from .envs import make_continual_regime_stream
from .evaluation import EvaluationStep, evaluate_prequential
from .metrics import adaptation_error_auc, forgetting_delta, samples_to_criterion
from .model import ContextualWorldModel, LGCMConfig

MODEL_NAMES = (
    "lgcm",
    "persistence",
    "observation_only",
    "global_ridge",
    "ewrls",
)


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class QualificationRun:
    model: str
    seed: int
    gradual: bool
    overall_mse: float
    segment_mse: tuple[tuple[str, float], ...]
    criterion_samples: tuple[tuple[str, int | None], ...]
    adaptation_auc: tuple[tuple[str, float], ...]
    forgetting_delta: float | None
    expert_count: int | None
    switches: int | None
    spawns: int | None
    capacity_events: int | None
    return_spawns: int | None
    reused_initial_expert: bool | None


@dataclass(frozen=True)
class QualificationAggregate:
    model: str
    gradual: bool
    overall_mse: SummaryStats
    expert_count: SummaryStats | None


@dataclass(frozen=True)
class QualificationReceipt:
    schema_version: int
    seeds: tuple[int, ...]
    lengths: tuple[tuple[str, int], ...]
    gradual_modes: tuple[bool, ...]
    drift_steps: int
    criterion_threshold: float
    criterion_consecutive: int
    runs: tuple[QualificationRun, ...]
    aggregates: tuple[QualificationAggregate, ...]

    def canonical_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class QualificationResource:
    model: str
    seed: int
    gradual: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class QualificationResult:
    receipt: QualificationReceipt
    resources: tuple[QualificationResource, ...]


def _make_model(name: str):
    if name == "lgcm":
        return ContextualWorldModel(LGCMConfig(observation_dim=2, action_dim=1))
    if name == "persistence":
        return PersistencePredictor(2, 1)
    if name == "observation_only":
        return ObservationOnlyWorldModel(2, 1)
    if name == "global_ridge":
        return GlobalRidgeWorldModel(2, 1)
    if name == "ewrls":
        return EWRLSWorldModel(2, 1)
    raise ValueError(f"unknown qualification model: {name}")


def _segment_steps(steps: tuple[EvaluationStep, ...]) -> tuple[tuple[str, tuple[EvaluationStep, ...]], ...]:
    order: list[str] = []
    grouped: dict[str, list[EvaluationStep]] = {}
    for step in steps:
        segment = step.segment or step.regime
        if segment not in grouped:
            order.append(segment)
            grouped[segment] = []
        grouped[segment].append(step)
    return tuple((name, tuple(grouped[name])) for name in order)


def _tail_mean(values: list[float], *, width: int = 5) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return float(fmean(values[-min(width, len(values)) :]))


def _route_metrics(
    model,
    grouped: tuple[tuple[str, tuple[EvaluationStep, ...]], ...],
) -> tuple[int | None, int | None, int | None, int | None, int | None, bool | None]:
    if not isinstance(model, ContextualWorldModel):
        return (None, None, None, None, None, None)

    all_steps = [step for _, steps in grouped for step in steps]
    switches = sum(step.decision == "switch" for step in all_steps)
    spawns = sum(step.decision == "spawn" for step in all_steps)
    capacity_events = sum(step.decision == "capacity_exhausted" for step in all_steps)

    by_segment = dict(grouped)
    initial = by_segment.get("A_INITIAL", ())
    returned = by_segment.get("A_RETURN", ())
    established = [
        step.post_update_active_expert
        for step in initial[len(initial) // 2 :]
        if step.post_update_active_expert is not None
    ]
    initial_expert = Counter(established).most_common(1)[0][0] if established else None
    reused = (
        initial_expert is not None
        and any(
            step.post_update_active_expert == initial_expert
            for step in returned
        )
    )
    return_spawns = sum(step.decision == "spawn" for step in returned)
    return (
        len(model.expert_ids),
        switches,
        spawns,
        capacity_events,
        return_spawns,
        reused,
    )


def _summarize(
    *,
    model_name: str,
    seed: int,
    gradual: bool,
    model,
    trace,
    criterion_threshold: float,
    criterion_consecutive: int,
) -> QualificationRun:
    grouped = _segment_steps(trace.steps)
    segment_mse: list[tuple[str, float]] = []
    criteria: list[tuple[str, int | None]] = []
    aucs: list[tuple[str, float]] = []

    for name, steps in grouped:
        errors = [step.mse for step in steps]
        segment_mse.append((name, float(fmean(errors))))
        criteria.append(
            (
                name,
                samples_to_criterion(
                    errors,
                    threshold=criterion_threshold,
                    consecutive=criterion_consecutive,
                ),
            )
        )
        aucs.append((name, adaptation_error_auc(errors)))

    by_segment = {name: steps for name, steps in grouped}
    forgetting: float | None = None
    initial = by_segment.get("A_INITIAL")
    returned = by_segment.get("A_RETURN")
    if initial and returned:
        forgetting = forgetting_delta(
            pre_error=_tail_mean([step.mse for step in initial]),
            return_error=_tail_mean([step.mse for step in returned]),
        )

    (
        expert_count,
        switches,
        spawns,
        capacity_events,
        return_spawns,
        reused_initial_expert,
    ) = _route_metrics(model, grouped)

    return QualificationRun(
        model=model_name,
        seed=seed,
        gradual=gradual,
        overall_mse=float(fmean(step.mse for step in trace.steps)),
        segment_mse=tuple(segment_mse),
        criterion_samples=tuple(criteria),
        adaptation_auc=tuple(aucs),
        forgetting_delta=forgetting,
        expert_count=expert_count,
        switches=switches,
        spawns=spawns,
        capacity_events=capacity_events,
        return_spawns=return_spawns,
        reused_initial_expert=reused_initial_expert,
    )


def _stats(values: list[float]) -> SummaryStats:
    array = np.asarray(values, dtype=np.float64)
    return SummaryStats(
        mean=float(np.mean(array)),
        std=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
    )


def _aggregate(runs: tuple[QualificationRun, ...]) -> tuple[QualificationAggregate, ...]:
    groups: dict[tuple[str, bool], list[QualificationRun]] = {}
    for run in runs:
        groups.setdefault((run.model, run.gradual), []).append(run)

    aggregates: list[QualificationAggregate] = []
    for model in MODEL_NAMES:
        for gradual in (False, True):
            group = groups.get((model, gradual))
            if not group:
                continue
            expert_values = [
                float(run.expert_count)
                for run in group
                if run.expert_count is not None
            ]
            aggregates.append(
                QualificationAggregate(
                    model=model,
                    gradual=gradual,
                    overall_mse=_stats([run.overall_mse for run in group]),
                    expert_count=_stats(expert_values) if expert_values else None,
                )
            )
    return tuple(aggregates)


def run_qualification(
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    lengths: Mapping[str, int] | None = None,
    gradual_modes: tuple[bool, ...] = (False, True),
    drift_steps: int = 8,
    criterion_threshold: float = 0.01,
    criterion_consecutive: int = 5,
) -> QualificationResult:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if not gradual_modes:
        raise ValueError("gradual_modes must not be empty")
    if not np.isfinite(criterion_threshold) or criterion_threshold < 0:
        raise ValueError("criterion_threshold must be finite and non-negative")
    if criterion_consecutive < 1:
        raise ValueError("criterion_consecutive must be positive")
    if drift_steps < 1:
        raise ValueError("drift_steps must be positive")

    stream_lengths = dict(
        lengths
        or {
            "A": 80,
            "B": 80,
            "A_RETURN": 80,
            "C": 80,
        }
    )
    normalized_lengths = tuple(
        (name, int(stream_lengths[name]))
        for name in ("A", "B", "A_RETURN", "C")
    )

    runs: list[QualificationRun] = []
    resources: list[QualificationResource] = []
    for gradual in gradual_modes:
        for seed in seeds:
            events = make_continual_regime_stream(
                seed=int(seed),
                lengths=stream_lengths,
                gradual=gradual,
                drift_steps=drift_steps,
            )
            for model_name in MODEL_NAMES:
                model = _make_model(model_name)
                started = perf_counter()
                trace = evaluate_prequential(model, events)
                elapsed = perf_counter() - started
                runs.append(
                    _summarize(
                        model_name=model_name,
                        seed=int(seed),
                        gradual=gradual,
                        model=model,
                        trace=trace,
                        criterion_threshold=criterion_threshold,
                        criterion_consecutive=criterion_consecutive,
                    )
                )
                resources.append(
                    QualificationResource(
                        model=model_name,
                        seed=int(seed),
                        gradual=gradual,
                        elapsed_seconds=float(elapsed),
                    )
                )

    receipt_runs = tuple(runs)
    receipt = QualificationReceipt(
        schema_version=1,
        seeds=tuple(int(seed) for seed in seeds),
        lengths=normalized_lengths,
        gradual_modes=tuple(bool(value) for value in gradual_modes),
        drift_steps=int(drift_steps),
        criterion_threshold=float(criterion_threshold),
        criterion_consecutive=int(criterion_consecutive),
        runs=receipt_runs,
        aggregates=_aggregate(receipt_runs),
    )
    return QualificationResult(receipt=receipt, resources=tuple(resources))
