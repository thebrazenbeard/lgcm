import numpy as np
import pytest

import lgcm


def test_public_encoders_are_exported():
    assert hasattr(lgcm, "IdentityFeatureEncoder")
    assert hasattr(lgcm, "RandomFeatureEncoder")


def test_identity_encoder_includes_bias_and_bounded_raw_inputs():
    encoder = lgcm.IdentityFeatureEncoder(observation_dim=2, action_dim=1)
    features = encoder.encode(np.array([0.5, -0.25]), np.array([1.0]))
    np.testing.assert_allclose(features, [1.0, 0.5, -0.25, 1.0])
    assert not features.flags.writeable


def test_encoder_rejects_coordinates_outside_v0_bounds():
    encoder = lgcm.IdentityFeatureEncoder(observation_dim=2, action_dim=1)
    with pytest.raises(ValueError, match="bounded"):
        encoder.encode(np.array([1.01, 0.0]), np.array([0.0]))


def test_random_encoder_is_seeded_and_not_single_seed_magic():
    a = lgcm.RandomFeatureEncoder(2, 1, width=8, seed=7)
    b = lgcm.RandomFeatureEncoder(2, 1, width=8, seed=7)
    c = lgcm.RandomFeatureEncoder(2, 1, width=8, seed=8)
    obs = np.array([0.2, -0.4])
    act = np.array([0.3])
    np.testing.assert_allclose(a.encode(obs, act), b.encode(obs, act))
    assert not np.allclose(a.encode(obs, act), c.encode(obs, act))


def test_random_encoder_enforces_total_feature_ceiling():
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        lgcm.RandomFeatureEncoder(16, 8, width=104, seed=1, max_feature_dim=128)
