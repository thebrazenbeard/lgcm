import numpy as np
import pytest

import lgcm


def test_rls_public_api_exists():
    assert hasattr(lgcm, "RLSRegressor")


def test_gamma_one_matches_closed_form_ridge():
    RLS = getattr(lgcm, "RLSRegressor", None)
    assert RLS is not None
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 4))
    y = rng.normal(size=(40, 2))
    lam = 0.75
    model = RLS(4, 2, regularization=lam, forgetting_factor=1.0)
    for phi, target in zip(x, y):
        model.update(phi, target)
    expected = np.linalg.solve(lam * np.eye(4) + x.T @ x, x.T @ y)
    np.testing.assert_allclose(model.weights, expected, rtol=1e-10, atol=1e-10)


def test_forgetting_rls_adapts_faster_after_coefficient_shift():
    RLS = getattr(lgcm, "RLSRegressor", None)
    assert RLS is not None
    stable = RLS(1, 1, regularization=1.0, forgetting_factor=1.0)
    adaptive = RLS(1, 1, regularization=1.0, forgetting_factor=0.8)
    phi = np.array([1.0])
    for _ in range(50):
        stable.update(phi, np.array([1.0]))
        adaptive.update(phi, np.array([1.0]))
    for _ in range(5):
        stable.update(phi, np.array([-1.0]))
        adaptive.update(phi, np.array([-1.0]))
    stable_err = abs(stable.predict(phi)[0] + 1.0)
    adaptive_err = abs(adaptive.predict(phi)[0] + 1.0)
    assert adaptive_err < stable_err


def test_rls_support_increases_with_repeated_evidence():
    RLS = getattr(lgcm, "RLSRegressor", None)
    assert RLS is not None
    model = RLS(2, 1, regularization=1.0, forgetting_factor=1.0)
    phi = np.array([1.0, 0.5])
    before = model.feature_support(phi)
    for _ in range(20):
        model.update(phi, np.array([0.25]))
    after = model.feature_support(phi)
    assert 0.0 < before < after <= 1.0


def test_rls_rejects_nonfinite_update_without_mutation():
    RLS = getattr(lgcm, "RLSRegressor", None)
    assert RLS is not None
    model = RLS(2, 1, regularization=1.0, forgetting_factor=1.0)
    w0 = model.weights.copy()
    p0 = model.covariance.copy()
    with pytest.raises(ValueError, match="finite"):
        model.update(np.array([1.0, np.nan]), np.array([0.0]))
    np.testing.assert_array_equal(model.weights, w0)
    np.testing.assert_array_equal(model.covariance, p0)


@pytest.mark.parametrize(
    "regularization,forgetting_factor",
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, 1.01)],
)
def test_rls_rejects_invalid_hyperparameters(regularization, forgetting_factor):
    RLS = getattr(lgcm, "RLSRegressor", None)
    assert RLS is not None
    with pytest.raises(ValueError):
        RLS(
            2,
            1,
            regularization=regularization,
            forgetting_factor=forgetting_factor,
        )
