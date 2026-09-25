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
