from pathlib import Path

import numpy as np
import pytest

import lgcm


def _model():
    return lgcm.ContextualWorldModel(
        lgcm.LGCMConfig(
            observation_dim=1,
            action_dim=1,
            regularization=0.1,
            expert_forgetting_factor=0.8,
            residual_decay=0.8,
            residual_floor=0.02,
            min_expert_evidence=2,
            mismatch_threshold=0.8,
            mismatch_min_evidence=2,
            spawn_patience=1,
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


def test_persistence_public_api_exists():
    assert hasattr(lgcm, "save_snapshot")
    assert hasattr(lgcm, "load_snapshot")


def test_snapshot_refuses_overwrite_by_default(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    with pytest.raises(FileExistsError):
        lgcm.save_snapshot(model, path)


def test_digest_mismatch_is_rejected_before_model_construction(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    payload = path / "shared_weights.npy"
    payload.write_bytes(payload.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="digest"):
        lgcm.load_snapshot(path)
