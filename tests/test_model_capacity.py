import numpy as np
import pytest

import lgcm


def _event(sequence, value):
    return lgcm.Experience(
        sequence=sequence,
        observation=np.array([0.0]),
        action=np.array([0.0]),
        next_observation=np.array([value]),
        source="simulator",
    )


def test_capacity_exhaustion_is_explicit_and_terminal_for_learning():
    model = lgcm.ContextualWorldModel(
        lgcm.LGCMConfig(
            observation_dim=1,
            action_dim=1,
            regularization=0.1,
            expert_forgetting_factor=0.8,
            residual_decay=0.8,
            residual_floor=0.02,
            min_expert_evidence=2,
            max_experts=1,
            switch_margin=0.1,
            support_penalty=0.0,
            spawn_patience=1,
            mismatch_delta=0.0,
            mismatch_threshold=0.5,
            mismatch_min_evidence=2,
            bootstrap_size=3,
            new_expert_init="neutral",
        )
    )
    seq = 0
    for _ in range(12):
        model.update(_event(seq, 0.5))
        seq += 1

    exhausted = None
    for _ in range(10):
        exhausted = model.update(_event(seq, -0.5))
        seq += 1
        if exhausted.decision is lgcm.ContextDecisionKind.CAPACITY_EXHAUSTED:
            break
    assert exhausted is not None
    assert exhausted.decision is lgcm.ContextDecisionKind.CAPACITY_EXHAUSTED
    assert model.predict(np.array([0.0]), np.array([0.0])).adequacy is lgcm.AdequacyState.CAPACITY_EXHAUSTED
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        model.update(_event(seq, -0.5))
