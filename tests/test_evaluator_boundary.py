import numpy as np

import lgcm


class SpyModel:
    def __init__(self):
        self.value = 0.0
        self.calls = []

    def predict(self, observation, action):
        self.calls.append(("predict", self.value))
        return lgcm.Prediction(
            mean=np.array([self.value]),
            feature_support=1.0,
            residual_scale=np.array([1.0]),
            mismatch_score=0.0,
            adequacy=lgcm.AdequacyState.ADEQUATE_WITHIN_EVIDENCE,
            active_expert=0,
        )

    def update(self, experience):
        self.calls.append(("update", self.value))
        self.value = float(experience.next_observation[0])
        return None


def test_evaluator_scores_prediction_before_update_and_never_passes_regime():
    events = [
        lgcm.EvaluatorEvent(
            experience=lgcm.Experience(
                sequence=0,
                observation=np.array([0.0]),
                action=np.array([0.0]),
                next_observation=np.array([0.5]),
                source="simulator",
            ),
            regime="A",
        ),
        lgcm.EvaluatorEvent(
            experience=lgcm.Experience(
                sequence=1,
                observation=np.array([0.5]),
                action=np.array([0.0]),
                next_observation=np.array([0.25]),
                source="simulator",
            ),
            regime="B",
        ),
    ]
    model = SpyModel()
    trace = lgcm.evaluate_prequential(model, events)
    assert model.calls == [
        ("predict", 0.0),
        ("update", 0.0),
        ("predict", 0.5),
        ("update", 0.5),
    ]
    assert [step.regime for step in trace.steps] == ["A", "B"]
    assert [step.mse for step in trace.steps] == [0.25, 0.0625]
