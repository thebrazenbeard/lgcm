from __future__ import annotations

import math

import pytest

from lgcm.compression_experiment import (
    PredictiveCompressionStep,
    evaluate_predictive_compression,
)


def _step(
    sequence: int,
    regime: str,
    *,
    full_mse: float,
    compressed_mse: float,
    source_state_bytes: int = 100,
    latent_state_bytes: int = 20,
    backing_state_bytes: int = 100,
    rehydration_count: int = 0,
    rehydration_bytes: int = 0,
    old_context_recall_correct: bool | None = None,
) -> PredictiveCompressionStep:
    return PredictiveCompressionStep(
        sequence=sequence,
        regime=regime,
        full_mse=full_mse,
        compressed_mse=compressed_mse,
        source_state_bytes=source_state_bytes,
        latent_state_bytes=latent_state_bytes,
        backing_state_bytes=backing_state_bytes,
        rehydration_count=rehydration_count,
        rehydration_bytes=rehydration_bytes,
        old_context_recall_correct=old_context_recall_correct,
    )


def test_trace_reports_predictive_cost_separately_from_compression_ratio():
    report = evaluate_predictive_compression(
        (
            _step(0, "A", full_mse=0.1, compressed_mse=0.2),
            _step(1, "A", full_mse=0.2, compressed_mse=0.4),
        )
    )

    assert report.step_count == 2
    assert report.mean_full_mse == pytest.approx(0.15)
    assert report.mean_compressed_mse == pytest.approx(0.30)
    assert report.mean_mse_delta == pytest.approx(0.15)
    assert report.source_state_bytes_total == 200
    assert report.latent_state_bytes_total == 40
    assert report.compression_ratio == pytest.approx(5.0)


def test_old_context_recall_is_not_hidden_by_average_prediction_error():
    report = evaluate_predictive_compression(
        (
            _step(
                0,
                "A",
                full_mse=0.1,
                compressed_mse=0.1,
                old_context_recall_correct=True,
            ),
            _step(1, "B", full_mse=0.1, compressed_mse=0.1),
            _step(
                2,
                "A",
                full_mse=0.1,
                compressed_mse=0.1,
                old_context_recall_correct=False,
            ),
        )
    )

    assert report.old_context_recall_trials == 2
    assert report.old_context_recall_accuracy == 0.5
    assert report.mean_mse_delta == pytest.approx(0.0)


def test_selective_rehydration_cost_is_reported_not_folded_into_latent_bytes():
    report = evaluate_predictive_compression(
        (
            _step(
                0,
                "A",
                full_mse=0.1,
                compressed_mse=0.1,
                rehydration_count=1,
                rehydration_bytes=35,
            ),
        )
    )

    assert report.latent_state_bytes_total == 20
    assert report.rehydration_count == 1
    assert report.rehydration_bytes_total == 35
    assert report.backing_state_bytes_total == 100


def test_sequence_must_preserve_prequential_order():
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_predictive_compression(
            (
                _step(2, "A", full_mse=0.1, compressed_mse=0.1),
                _step(2, "B", full_mse=0.1, compressed_mse=0.1),
            )
        )


def test_step_rejects_nonfinite_error_and_latent_larger_than_source():
    with pytest.raises(ValueError, match="finite"):
        _step(0, "A", full_mse=math.inf, compressed_mse=0.1).validate()

    with pytest.raises(ValueError, match="latent_state_bytes"):
        _step(
            0,
            "A",
            full_mse=0.1,
            compressed_mse=0.1,
            source_state_bytes=10,
            latent_state_bytes=11,
        ).validate()


def test_report_keeps_claim_ceiling_explicit():
    report = evaluate_predictive_compression(
        (_step(0, "A", full_mse=0.1, compressed_mse=0.1),)
    )

    assert (
        report.claim_ceiling
        == "EXPERIMENT_PROTOCOL_ONLY_LGCM0_ENCODERS_AND_QUALIFICATION_UNCHANGED"
    )
