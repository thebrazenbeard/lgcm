import numpy as np
import pytest

import lgcm


def _cfg(**overrides):
    values = dict(
        observation_dim=1,
        action_dim=1,
        regularization=0.1,
        shared_forgetting_factor=1.0,
        expert_forgetting_factor=0.8,
        residual_decay=0.8,
        residual_floor=0.05,
        min_expert_evidence=3,
        max_experts=4,
        switch_margin=0.2,
        support_penalty=0.0,
        spawn_patience=1,
        mismatch_delta=0.0,
        mismatch_threshold=1.0,
        mismatch_min_evidence=2,
        bootstrap_size=4,
        new_expert_init="neutral",
    )
    values.update(overrides)
    return lgcm.LGCMConfig(**values)


def _event(sequence: int, next_value: float):
    return lgcm.Experience(
        sequence=sequence,
        observation=np.array([0.0]),
        action=np.array([0.0]),
        next_observation=np.array([next_value]),
        source="simulator",
    )


def test_contextual_model_public_api_exists():
    assert hasattr(lgcm, "LGCMConfig")
    assert hasattr(lgcm, "ContextualWorldModel")
    assert hasattr(lgcm, "UpdateReceipt")


def test_predict_is_read_only_and_first_expert_is_neutral_internal_context():
    model = lgcm.ContextualWorldModel(_cfg())
    before = model.expert_weights(0)
    generation = model.generation
    prediction = model.predict(np.array([0.0]), np.array([0.0]))
    assert prediction.active_expert == 0
    assert model.active_expert_id == 0
    assert model.expert_ids == (0,)
    assert model.generation == generation
    np.testing.assert_array_equal(model.expert_weights(0), before)


def test_update_commits_once_and_rejects_duplicate_or_older_sequence():
    model = lgcm.ContextualWorldModel(_cfg())
    receipt = model.update(_event(10, 0.5))
    assert receipt.generation == 1
    assert model.generation == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        model.update(_event(10, 0.5))
    with pytest.raises(ValueError, match="strictly increasing"):
        model.update(_event(9, 0.5))
    assert model.generation == 1
