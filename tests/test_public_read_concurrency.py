"""Public model-state reads must see only committed update boundaries."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import Event

import numpy as np
import pytest

import lgcm


@pytest.mark.parametrize(
    "reader_name",
    ("active_expert_id", "expert_ids", "expert_weights", "shared_weights"),
)
def test_public_state_read_waits_for_in_progress_update(monkeypatch, reader_name):
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
    read_started = Event()
    original_shared_update = model._shared.update

    def pause_after_shared_update(*args, **kwargs):
        result = original_shared_update(*args, **kwargs)
        shared_updated.set()
        if not allow_commit.wait(timeout=5):
            raise TimeoutError("timed out waiting for update continuation")
        return result

    monkeypatch.setattr(model._shared, "update", pause_after_shared_update)

    def read():
        read_started.set()
        if reader_name == "expert_weights":
            return model.expert_weights(0)
        return getattr(model, reader_name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        update_future = pool.submit(model.update, experience)
        assert shared_updated.wait(timeout=5), "update did not reach shared model"
        read_future = pool.submit(read)
        try:
            assert read_started.wait(timeout=5)
            with pytest.raises(FutureTimeoutError):
                read_future.result(timeout=0.2)
        finally:
            allow_commit.set()
        update_future.result(timeout=5)
        observed = read_future.result(timeout=5)

    committed = read()
    if isinstance(observed, np.ndarray):
        np.testing.assert_array_equal(observed, committed)
        assert not observed.flags.writeable
    else:
        assert observed == committed
