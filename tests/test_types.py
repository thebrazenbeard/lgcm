import numpy as np
import pytest

import lgcm


def test_public_contracts_are_exported():
    assert hasattr(lgcm, "Experience")
    assert hasattr(lgcm, "AdequacyState")
    assert hasattr(lgcm, "Prediction")


def test_experience_coerces_to_contiguous_read_only_float64():
    base = np.arange(8.0, dtype=np.float64)[::2]
    exp = lgcm.Experience(
        sequence=0,
        observation=base,
        action=[0.25],
        next_observation=[0.1, 0.2, 0.3, 0.4],
        source="simulator",
    )
    assert exp.observation.dtype == np.float64
    assert exp.observation.flags.c_contiguous
    assert not exp.observation.flags.writeable
    assert not exp.action.flags.writeable
    assert not exp.next_observation.flags.writeable


def test_experience_does_not_mutate_or_alias_caller_arrays():
    observation = np.array([0.1, 0.2], dtype=np.float64)
    action = np.array([0.3], dtype=np.float64)
    next_observation = np.array([0.4, 0.5], dtype=np.float64)

    exp = lgcm.Experience(
        sequence=0,
        observation=observation,
        action=action,
        next_observation=next_observation,
        source="simulator",
    )

    assert observation.flags.writeable
    assert action.flags.writeable
    assert next_observation.flags.writeable

    observation[0] = 0.9
    action[0] = 0.8
    next_observation[0] = 0.7
    np.testing.assert_array_equal(exp.observation, [0.1, 0.2])
    np.testing.assert_array_equal(exp.action, [0.3])
    np.testing.assert_array_equal(exp.next_observation, [0.4, 0.5])


def test_prediction_does_not_mutate_or_alias_caller_arrays():
    mean = np.array([0.1, 0.2], dtype=np.float64)
    residual_scale = np.array([0.3, 0.4], dtype=np.float64)

    prediction = lgcm.Prediction(
        mean=mean,
        feature_support=1.0,
        residual_scale=residual_scale,
        mismatch_score=0.0,
        adequacy=lgcm.AdequacyState.INSUFFICIENT_EVIDENCE,
        active_expert=None,
    )

    assert mean.flags.writeable
    assert residual_scale.flags.writeable

    mean[0] = 0.9
    residual_scale[0] = 0.8
    np.testing.assert_array_equal(prediction.mean, [0.1, 0.2])
    np.testing.assert_array_equal(prediction.residual_scale, [0.3, 0.4])


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation", [[0.0, 1.0]]),
        ("action", [np.nan]),
        ("next_observation", [np.inf]),
    ],
)
def test_experience_rejects_invalid_vectors(field, value):
    kwargs = dict(
        sequence=0,
        observation=[0.0],
        action=[0.0],
        next_observation=[0.0],
        source="simulator",
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        lgcm.Experience(**kwargs)