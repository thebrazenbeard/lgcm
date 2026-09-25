import numpy as np

import lgcm


def test_causal_control_is_action_conditioned_and_bounded():
    env = lgcm.CausalControlEnv(seed=7, noise_std=0.0)
    state = np.array([0.2, -0.1])
    plus = env.transition(state, np.array([0.8]))
    minus = env.transition(state, np.array([-0.8]))
    assert not np.allclose(plus, minus)
    assert np.all(np.abs(plus) <= 1.0)
    assert np.all(np.abs(minus) <= 1.0)


def test_causal_control_is_seed_reproducible():
    a = lgcm.CausalControlEnv(seed=11, noise_std=0.01)
    b = lgcm.CausalControlEnv(seed=11, noise_std=0.01)
    actions = [np.array([x]) for x in (-0.5, 0.2, 0.9, -0.1)]
    for action in actions:
        np.testing.assert_allclose(a.step(action), b.step(action))


def test_continual_stream_hides_regime_from_experience_and_repeats_a():
    stream = lgcm.make_continual_regime_stream(
        seed=3,
        lengths={"A": 6, "B": 5, "A_RETURN": 4, "C": 5},
        gradual=False,
    )
    labels = [event.regime for event in stream]
    assert labels[:6] == ["A"] * 6
    assert labels[6:11] == ["B"] * 5
    assert labels[11:15] == ["A"] * 4
    assert labels[15:] == ["C"] * 5
    for event in stream:
        assert not hasattr(event.experience, "regime")
        assert event.experience.source == "simulator"


def test_gradual_variant_contains_intermediate_dynamics_without_learner_label():
    stream = lgcm.make_continual_regime_stream(
        seed=5,
        lengths={"A": 4, "B": 4, "A_RETURN": 4, "C": 4},
        gradual=True,
        drift_steps=5,
    )
    assert any(event.regime == "A_TO_B_DRIFT" for event in stream)
    assert all(not hasattr(event.experience, "regime") for event in stream)
