import numpy as np

import lgcm


def _experience(sequence, observation, action, next_observation):
    return lgcm.Experience(
        sequence=sequence,
        observation=np.asarray(observation, dtype=np.float64),
        action=np.asarray(action, dtype=np.float64),
        next_observation=np.asarray(next_observation, dtype=np.float64),
        source="simulator",
    )


def test_baseline_public_api_exists():
    for name in (
        "PersistencePredictor",
        "ObservationOnlyWorldModel",
        "GlobalRidgeWorldModel",
        "EWRLSWorldModel",
    ):
        assert hasattr(lgcm, name)


def test_persistence_predictor_returns_current_observation():
    cls = getattr(lgcm, "PersistencePredictor", None)
    assert cls is not None
    model = cls(observation_dim=2, action_dim=1)
    prediction = model.predict(np.array([0.2, -0.4]), np.array([0.8]))
    np.testing.assert_allclose(prediction.mean, [0.2, -0.4])


def test_observation_only_baseline_is_action_invariant():
    cls = getattr(lgcm, "ObservationOnlyWorldModel", None)
    assert cls is not None
    model = cls(observation_dim=1, action_dim=1, regularization=0.1)
    for i in range(20):
        action = -1.0 if i % 2 else 1.0
        model.update(_experience(i, [0.2], [action], [0.4]))
    a = model.predict(np.array([0.2]), np.array([-1.0])).mean
    b = model.predict(np.array([0.2]), np.array([1.0])).mean
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_global_and_forgetting_world_models_diverge_after_shift():
    gcls = getattr(lgcm, "GlobalRidgeWorldModel", None)
    ecls = getattr(lgcm, "EWRLSWorldModel", None)
    assert gcls is not None and ecls is not None
    global_model = gcls(1, 1, regularization=0.1)
    adaptive = ecls(1, 1, regularization=0.1, forgetting_factor=0.8)
    seq = 0
    for _ in range(40):
        event = _experience(seq, [0.0], [1.0], [0.5])
        global_model.update(event)
        adaptive.update(event)
        seq += 1
    for _ in range(5):
        event = _experience(seq, [0.0], [1.0], [-0.5])
        global_model.update(event)
        adaptive.update(event)
        seq += 1
    gerr = abs(global_model.predict(np.array([0.0]), np.array([1.0])).mean[0] + 0.5)
    aerr = abs(adaptive.predict(np.array([0.0]), np.array([1.0])).mean[0] + 0.5)
    assert aerr < gerr
