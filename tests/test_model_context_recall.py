import numpy as np

import lgcm


def _model(max_experts=4):
    return lgcm.ContextualWorldModel(
        lgcm.LGCMConfig(
            observation_dim=1,
            action_dim=1,
            regularization=0.1,
            shared_forgetting_factor=1.0,
            expert_forgetting_factor=0.75,
            residual_decay=0.8,
            residual_floor=0.02,
            min_expert_evidence=3,
            max_experts=max_experts,
            switch_margin=0.1,
            support_penalty=0.0,
            spawn_patience=1,
            mismatch_delta=0.0,
            mismatch_threshold=0.8,
            mismatch_min_evidence=2,
            bootstrap_size=4,
            new_expert_init="neutral",
        )
    )


def _event(sequence, value):
    return lgcm.Experience(
        sequence=sequence,
        observation=np.array([0.0]),
        action=np.array([0.0]),
        next_observation=np.array([value]),
        source="simulator",
    )


def _train(model, seq, value, steps):
    for _ in range(steps):
        model.update(_event(seq, value))
        seq += 1
    return seq


def test_a_b_a_return_reuses_dormant_expert_without_parameter_drift():
    model = _model()
    seq = _train(model, 0, 0.6, 20)
    assert model.expert_ids == (0,)

    # Shift to B until a second expert is created.
    for _ in range(12):
        model.update(_event(seq, -0.6))
        seq += 1
        if len(model.expert_ids) == 2:
            break
    assert model.expert_ids == (0, 1)
    assert model.active_expert_id == 1
    a_frozen = model.expert_weights(0).copy()

    seq = _train(model, seq, -0.6, 20)
    np.testing.assert_allclose(model.expert_weights(0), a_frozen, rtol=0.0, atol=0.0)

    # Return to A. The retained A expert should be selected instead of a third spawn.
    for _ in range(12):
        model.update(_event(seq, 0.6))
        seq += 1
        if model.active_expert_id == 0:
            break
    assert model.active_expert_id == 0
    assert model.expert_ids == (0, 1)
    pred = model.predict(np.array([0.0]), np.array([0.0]))
    assert abs(pred.mean[0] - 0.6) < 0.2
