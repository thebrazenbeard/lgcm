"""Independent behavior checks beyond unit-level model-structure tests."""

import numpy as np

import lgcm


def _experience(sequence, action, target):
    return lgcm.Experience(
        sequence=sequence,
        observation=np.array([0.0]),
        action=np.array([action]),
        next_observation=np.array([target]),
        source="heldout-control",
    )


def test_action_conditioned_learning_beats_untrained_negative_control():
    trained = lgcm.ContextualWorldModel(lgcm.LGCMConfig(1, 1))
    untrained = lgcm.ContextualWorldModel(lgcm.LGCMConfig(1, 1))
    for i in range(80):
        action = 1.0 if i % 2 else -1.0
        trained.update(_experience(i, action, 0.5 * action))

    # Interpolation controls: neither test action appeared in training.
    outcomes = [(0.7, 0.35), (-0.4, -0.2)]
    trained_errors = []
    control_errors = []
    for action, target in outcomes:
        observation = np.array([0.0])
        current_action = np.array([action])
        learned = trained.predict(observation, current_action).mean[0]
        untouched = untrained.predict(observation, current_action).mean[0]
        trained_errors.append((learned - target) ** 2)
        control_errors.append((untouched - target) ** 2)
    assert np.mean(trained_errors) < 0.01 * np.mean(control_errors)


def test_regime_labels_cannot_affect_prequential_prediction_or_gate():
    stream = lgcm.make_continual_regime_stream(
        seed=41,
        lengths={"A": 30, "B": 30, "A_RETURN": 30, "C": 30},
        gradual=True,
    )
    decoys = tuple(
        lgcm.EvaluatorEvent(experience=event.experience, regime="false-label")
        for event in stream
    )
    left = lgcm.evaluate_prequential(
        lgcm.ContextualWorldModel(lgcm.LGCMConfig(2, 1)), stream
    )
    right = lgcm.evaluate_prequential(
        lgcm.ContextualWorldModel(lgcm.LGCMConfig(2, 1)), decoys
    )
    np.testing.assert_allclose(
        [step.mse for step in left.steps],
        [step.mse for step in right.steps],
        rtol=0,
        atol=0,
    )
    assert [step.active_expert for step in left.steps] == [
        step.active_expert for step in right.steps
    ]
    assert [step.adequacy for step in left.steps] == [
        step.adequacy for step in right.steps
    ]


def test_restart_equivalence_on_multivariate_unseen_regime_stream(tmp_path):
    stream = lgcm.make_continual_regime_stream(
        seed=59,
        lengths={"A": 40, "B": 40, "A_RETURN": 40, "C": 40},
        gradual=True,
    )
    uninterrupted = lgcm.ContextualWorldModel(lgcm.LGCMConfig(2, 1))
    cut = 85
    for event in stream[:cut]:
        uninterrupted.update(event.experience)
    lgcm.save_snapshot(uninterrupted, tmp_path / "heldout-state")
    resumed = lgcm.load_snapshot(tmp_path / "heldout-state")

    for event in stream[cut:]:
        experience = event.experience
        left = uninterrupted.predict(experience.observation, experience.action)
        right = resumed.predict(experience.observation, experience.action)
        np.testing.assert_allclose(left.mean, right.mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(left.residual_scale, right.residual_scale, rtol=0, atol=1e-12)
        assert left.active_expert == right.active_expert
        assert left.adequacy is right.adequacy
        lreceipt = uninterrupted.update(experience)
        rreceipt = resumed.update(experience)
        assert lreceipt.decision is rreceipt.decision
        assert lreceipt.active_expert == rreceipt.active_expert
        assert lreceipt.expert_count == rreceipt.expert_count
    assert uninterrupted.generation == resumed.generation == len(stream)
