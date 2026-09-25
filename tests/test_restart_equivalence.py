from pathlib import Path

import numpy as np

import lgcm


def _event(sequence, value):
    return lgcm.Experience(
        sequence=sequence,
        observation=np.array([0.0]),
        action=np.array([0.0]),
        next_observation=np.array([value]),
        source="simulator",
    )


def test_restart_preserves_predictions_gating_and_continuation(tmp_path: Path):
    config = lgcm.LGCMConfig(
        observation_dim=1,
        action_dim=1,
        regularization=0.1,
        expert_forgetting_factor=0.75,
        residual_decay=0.8,
        residual_floor=0.02,
        min_expert_evidence=3,
        max_experts=4,
        switch_margin=0.1,
        support_penalty=0.0,
        spawn_patience=1,
        mismatch_delta=0.0,
        mismatch_threshold=0.8,
        mismatch_min_evidence=2,
        bootstrap_size=4,
        new_expert_init="neutral",
    )
    original = lgcm.ContextualWorldModel(config)
    seq = 0
    for value, steps in ((0.6, 20), (-0.6, 20)):
        for _ in range(steps):
            original.update(_event(seq, value))
            seq += 1

    path = tmp_path / "snapshot"
    lgcm.save_snapshot(original, path)
    restored = lgcm.load_snapshot(path)

    assert restored.generation == original.generation
    assert restored.active_expert_id == original.active_expert_id
    assert restored.expert_ids == original.expert_ids

    probes = [
        (np.array([0.0]), np.array([0.0])),
        (np.array([0.2]), np.array([-0.3])),
    ]
    for obs, action in probes:
        left = original.predict(obs, action)
        right = restored.predict(obs, action)
        np.testing.assert_allclose(left.mean, right.mean, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            left.residual_scale,
            right.residual_scale,
            rtol=0.0,
            atol=1e-12,
        )
        assert left.active_expert == right.active_expert
        assert abs(left.feature_support - right.feature_support) <= 1e-12
        assert abs(left.mismatch_score - right.mismatch_score) <= 1e-12

    for value in [0.6] * 8 + [-0.6] * 8:
        l = original.update(_event(seq, value))
        r = restored.update(_event(seq, value))
        assert l.decision is r.decision
        assert l.active_expert == r.active_expert
        assert l.expert_count == r.expert_count
        np.testing.assert_allclose(
            original.predict(np.array([0.0]), np.array([0.0])).mean,
            restored.predict(np.array([0.0]), np.array([0.0])).mean,
            rtol=0.0,
            atol=1e-10,
        )
        seq += 1
