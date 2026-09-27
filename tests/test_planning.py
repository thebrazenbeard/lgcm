import numpy as np
import pytest

import lgcm


class IntegratorModel:
    action_dim = 1

    def predict(self, observation, action):
        mean = np.asarray(observation, dtype=np.float64) + np.asarray(
            action,
            dtype=np.float64,
        )
        return lgcm.Prediction(
            mean=mean,
            feature_support=1.0,
            residual_scale=np.ones_like(mean),
            mismatch_score=0.0,
            adequacy=lgcm.AdequacyState.ADEQUATE_WITHIN_EVIDENCE,
            active_expert=None,
        )


def quadratic_target(target):
    def objective(state):
        return float(np.sum((state - target) ** 2))

    return objective


def test_planning_public_api_exists():
    assert hasattr(lgcm, "plan_action")
    assert hasattr(lgcm, "PlanResult")


def test_random_shooting_is_seed_deterministic_and_bounded():
    model = IntegratorModel()
    kwargs = dict(
        model=model,
        observation=np.array([0.0]),
        objective=quadratic_target(np.array([1.0])),
        horizon=3,
        candidates=64,
        seed=17,
        lower_bound=-0.4,
        upper_bound=0.4,
    )

    left = lgcm.plan_action(**kwargs)
    right = lgcm.plan_action(**kwargs)

    np.testing.assert_array_equal(left.action, right.action)
    assert left.score == right.score
    assert left.candidate_index == right.candidate_index
    assert not left.action.flags.writeable
    assert np.all(left.action >= -0.4)
    assert np.all(left.action <= 0.4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"horizon": 0},
        {"candidates": 0},
        {"lower_bound": 1.0, "upper_bound": 1.0},
        {"lower_bound": np.nan},
        {"upper_bound": np.inf},
    ],
)
def test_random_shooting_rejects_invalid_controls(overrides):
    kwargs = dict(
        model=IntegratorModel(),
        observation=np.array([0.0]),
        objective=quadratic_target(np.array([1.0])),
        horizon=2,
        candidates=8,
        seed=1,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        lgcm.plan_action(**kwargs)


def test_plan_result_rejects_invalid_payload():
    with pytest.raises(ValueError, match="action"):
        lgcm.PlanResult(
            action=np.array([[0.0]]),
            score=0.0,
            candidate_index=0,
            seed=0,
            horizon=1,
            candidates=1,
        )
    with pytest.raises(ValueError, match="score"):
        lgcm.PlanResult(
            action=np.array([0.0]),
            score=np.inf,
            candidate_index=0,
            seed=0,
            horizon=1,
            candidates=1,
        )


def test_planning_rejects_model_without_action_dimension():
    class MissingActionDim:
        def predict(self, observation, action):
            raise AssertionError("predict should not be called")

    with pytest.raises(ValueError, match="action dimension"):
        lgcm.plan_action(
            MissingActionDim(),
            np.array([0.0]),
            quadratic_target(np.array([0.0])),
            horizon=1,
            candidates=1,
            seed=0,
            lower_bound=-1.0,
            upper_bound=1.0,
        )


def test_planning_rejects_invalid_observation_and_prediction_shape():
    with pytest.raises(ValueError, match="observation"):
        lgcm.plan_action(
            IntegratorModel(),
            np.array([[0.0]]),
            quadratic_target(np.array([0.0])),
            horizon=1,
            candidates=1,
            seed=0,
            lower_bound=-1.0,
            upper_bound=1.0,
        )

    class WrongShapeModel:
        action_dim = 1

        def predict(self, observation, action):
            return lgcm.Prediction(
                mean=np.array([0.0, 0.0]),
                feature_support=1.0,
                residual_scale=np.ones(2),
                mismatch_score=0.0,
                adequacy=lgcm.AdequacyState.ADEQUATE_WITHIN_EVIDENCE,
                active_expert=None,
            )

    with pytest.raises(ValueError, match="preserve"):
        lgcm.plan_action(
            WrongShapeModel(),
            np.array([0.0]),
            quadratic_target(np.array([0.0])),
            horizon=1,
            candidates=1,
            seed=0,
            lower_bound=-1.0,
            upper_bound=1.0,
        )


def test_planning_rejects_nonfinite_objective():
    with pytest.raises(ValueError, match="objective"):
        lgcm.plan_action(
            IntegratorModel(),
            np.array([0.0]),
            lambda state: np.nan,
            horizon=2,
            candidates=4,
            seed=1,
            lower_bound=-1.0,
            upper_bound=1.0,
        )


def test_planning_does_not_mutate_contextual_model():
    model = lgcm.ContextualWorldModel(
        lgcm.LGCMConfig(observation_dim=1, action_dim=1)
    )
    model.update(
        lgcm.Experience(
            sequence=0,
            observation=np.array([0.0]),
            action=np.array([0.1]),
            next_observation=np.array([0.2]),
            source="test",
        )
    )
    generation = model.generation
    active_expert = model.active_expert_id
    expert_ids = model.expert_ids
    shared = model.shared_weights.copy()
    expert = model.expert_weights(active_expert).copy()
    mismatch = model._mismatch.state
    evidence = model._experts[active_expert].regressor.evidence_count

    lgcm.plan_action(
        model,
        np.array([0.1]),
        quadratic_target(np.array([0.0])),
        horizon=4,
        candidates=32,
        seed=9,
        lower_bound=-0.5,
        upper_bound=0.5,
    )

    assert model.generation == generation
    assert model.active_expert_id == active_expert
    assert model.expert_ids == expert_ids
    np.testing.assert_array_equal(model.shared_weights, shared)
    np.testing.assert_array_equal(model.expert_weights(active_expert), expert)
    assert model._mismatch.state == mismatch
    assert model._experts[active_expert].regressor.evidence_count == evidence


def test_random_shooting_beats_neutral_action_on_simple_model():
    model = IntegratorModel()
    objective = quadratic_target(np.array([1.0]))
    horizon = 3
    neutral_score = horizon * objective(np.array([0.0]))

    result = lgcm.plan_action(
        model,
        np.array([0.0]),
        objective,
        horizon=horizon,
        candidates=256,
        seed=23,
        lower_bound=-0.5,
        upper_bound=0.5,
    )

    assert result.score < neutral_score
    assert result.action[0] > 0.0
