import json

import numpy as np
import pytest

import lgcm


def _run():
    return lgcm.run_qualification(
        seeds=(2, 5),
        lengths={"A": 16, "B": 16, "A_RETURN": 16, "C": 16},
        gradual_modes=(False, True),
        criterion_threshold=0.02,
        criterion_consecutive=2,
    )


def test_qualification_public_api_exists():
    assert hasattr(lgcm, "run_qualification")
    assert hasattr(lgcm, "QualificationReceipt")
    assert hasattr(lgcm, "QualificationResult")


def test_qualification_receipt_is_deterministic_for_same_inputs():
    left = _run()
    right = _run()

    assert left.receipt.canonical_json() == right.receipt.canonical_json()
    assert json.loads(left.receipt.canonical_json())["schema_version"] == 1
    assert all(item.elapsed_seconds >= 0.0 for item in left.resources)


def test_qualification_runs_all_models_in_abrupt_and_gradual_streams():
    result = _run()
    identities = {
        (run.model, run.gradual, run.seed)
        for run in result.receipt.runs
    }
    expected_models = {
        "lgcm",
        "persistence",
        "observation_only",
        "global_ridge",
        "ewrls",
    }

    for seed in (2, 5):
        for gradual in (False, True):
            assert {
                model
                for model, mode, run_seed in identities
                if mode is gradual and run_seed == seed
            } == expected_models


def test_qualification_distinguishes_initial_and_return_segments():
    result = _run()
    lgcm_runs = [run for run in result.receipt.runs if run.model == "lgcm"]
    assert lgcm_runs

    for run in lgcm_runs:
        segment_names = {name for name, _ in run.segment_mse}
        assert "A_INITIAL" in segment_names
        assert "A_RETURN" in segment_names
        assert "B" in segment_names
        assert "C" in segment_names
        assert run.expert_count is not None
        assert run.switches is not None
        assert run.spawns is not None
        assert run.return_spawns is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": ()},
        {"gradual_modes": ()},
        {"criterion_threshold": np.nan},
        {"criterion_consecutive": 0},
        {"drift_steps": 0},
    ],
)
def test_qualification_rejects_invalid_controls(overrides):
    kwargs = {
        "seeds": (0,),
        "lengths": {"A": 4, "B": 4, "A_RETURN": 4, "C": 4},
        "gradual_modes": (False,),
        "criterion_threshold": 0.02,
        "criterion_consecutive": 1,
        "drift_steps": 2,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        lgcm.run_qualification(**kwargs)


def test_qualification_aggregate_reports_dispersion():
    result = _run()
    assert len(result.receipt.aggregates) == 10
    for aggregate in result.receipt.aggregates:
        assert aggregate.overall_mse.minimum <= aggregate.overall_mse.mean
        assert aggregate.overall_mse.mean <= aggregate.overall_mse.maximum
        assert aggregate.overall_mse.std >= 0.0
