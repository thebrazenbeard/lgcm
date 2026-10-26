"""A persisted generation must represent a whole committed update."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np

import lgcm
from lgcm import persistence


def test_snapshot_waits_for_in_progress_update(tmp_path, monkeypatch):
    model = lgcm.ContextualWorldModel(lgcm.LGCMConfig(1, 1))
    experience = lgcm.Experience(
        sequence=0,
        observation=np.array([0.0]),
        action=np.array([0.0]),
        next_observation=np.array([0.5]),
        source="simulator",
    )
    shared_updated = Event()
    allow_commit = Event()
    snapshot_started = Event()
    snapshot_payload_started = Event()
    original_shared_update = model._shared.update
    original_save_array = persistence._save_array

    def pause_after_shared_update(*args, **kwargs):
        result = original_shared_update(*args, **kwargs)
        shared_updated.set()
        if not allow_commit.wait(timeout=5):
            raise TimeoutError("timed out waiting for update continuation")
        return result

    monkeypatch.setattr(model._shared, "update", pause_after_shared_update)

    def observe_snapshot_payload(*args, **kwargs):
        snapshot_payload_started.set()
        return original_save_array(*args, **kwargs)

    monkeypatch.setattr(persistence, "_save_array", observe_snapshot_payload)

    def write_snapshot():
        snapshot_started.set()
        return lgcm.save_snapshot(model, tmp_path / "snapshot")

    with ThreadPoolExecutor(max_workers=2) as pool:
        update_future = pool.submit(model.update, experience)
        assert shared_updated.wait(timeout=5), "update did not reach shared-model step"
        snapshot_future = pool.submit(write_snapshot)
        try:
            assert snapshot_started.wait(timeout=5)
            # A snapshot requested during the update may not even start
            # reading payload arrays; every hash can validate a torn state.
            assert not snapshot_payload_started.wait(timeout=1)
        finally:
            allow_commit.set()
        update_future.result(timeout=5)
        snapshot_future.result(timeout=5)

    restored = lgcm.load_snapshot(tmp_path / "snapshot")
    assert restored.generation == 1
    assert restored.last_sequence == 0
    assert restored._shared.evidence_count == 1
    assert restored._experts[0].regressor.evidence_count == 1
    np.testing.assert_array_equal(restored.shared_weights, model.shared_weights)
    np.testing.assert_array_equal(restored.expert_weights(0), model.expert_weights(0))
