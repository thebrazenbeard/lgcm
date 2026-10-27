"""Predictions must not observe an update before its commit boundary."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np

import lgcm


def test_predict_waits_for_in_progress_update(monkeypatch):
    model = lgcm.ContextualWorldModel(lgcm.LGCMConfig(1, 1))
    observation = np.array([0.0])
    action = np.array([0.0])
    experience = lgcm.Experience(
        sequence=0,
        observation=observation,
        action=action,
        next_observation=np.array([0.5]),
        source="simulator",
    )
    shared_updated = Event()
    allow_commit = Event()
    prediction_requested = Event()
    original_shared_update = model._shared.update

    def pause_after_shared_update(*args, **kwargs):
        result = original_shared_update(*args, **kwargs)
        shared_updated.set()
        if not allow_commit.wait(timeout=5):
            raise TimeoutError("timed out waiting for update continuation")
        return result

    monkeypatch.setattr(model._shared, "update", pause_after_shared_update)

    def read_prediction():
        prediction_requested.set()
        return model.predict(observation, action)

    with ThreadPoolExecutor(max_workers=2) as pool:
        update_future = pool.submit(model.update, experience)
        assert shared_updated.wait(timeout=5), "update did not reach shared-model step"
        prediction_future = pool.submit(read_prediction)
        try:
            assert prediction_requested.wait(timeout=5)
            premature = prediction_future.done() or prediction_future_done_within(
                prediction_future, timeout=0.5
            )
        finally:
            allow_commit.set()
        update_future.result(timeout=5)
        prediction = prediction_future.result(timeout=5)

    assert not premature, "prediction read an incompletely committed update"
    complete_prediction = model.predict(observation, action)
    np.testing.assert_array_equal(prediction.mean, complete_prediction.mean)
    assert prediction.active_expert == complete_prediction.active_expert
    assert prediction.adequacy == complete_prediction.adequacy


def prediction_future_done_within(future, *, timeout):
    """Wait briefly without cancelling a legitimately blocked prediction."""
    from concurrent.futures import TimeoutError

    try:
        future.result(timeout=timeout)
    except TimeoutError:
        return False
    return True
