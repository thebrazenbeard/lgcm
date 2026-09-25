import numpy as np

import lgcm


def _trained_expert(expert_id: int, target: float):
    Expert = getattr(lgcm, "ContextExpert", None)
    RLS = getattr(lgcm, "RLSRegressor", None)
    Tracker = getattr(lgcm, "ResidualScaleTracker", None)
    assert Expert is not None and RLS is not None and Tracker is not None
    regressor = RLS(2, 1, regularization=0.1, forgetting_factor=0.9)
    tracker = Tracker(1, floor=0.05, min_evidence=3)
    phi = np.array([1.0, 0.25])
    for _ in range(12):
        pred = regressor.predict(phi)
        residual = np.array([target]) - pred
        regressor.update(phi, np.array([target]))
        tracker.update(residual)
    return Expert(expert_id, regressor, tracker)


def test_context_public_api_exists():
    for name in (
        "ContextExpert",
        "ContextGate",
        "ContextDecisionKind",
        "ResidualScaleTracker",
    ):
        assert hasattr(lgcm, name)


def test_expert_scoring_prefers_matching_retained_expert():
    Gate = getattr(lgcm, "ContextGate", None)
    assert Gate is not None
    a = _trained_expert(0, 1.0)
    b = _trained_expert(1, -1.0)
    gate = Gate(max_experts=4, switch_margin=0.2, support_penalty=0.0, spawn_patience=2)
    phi = np.array([1.0, 0.25])
    scores = gate.score_experts([a, b], phi, np.array([1.0]), active_expert_id=1)
    by_id = {score.expert_id: score for score in scores}
    assert by_id[0].total_score < by_id[1].total_score


def test_gate_switches_only_after_mismatch_and_material_improvement():
    Gate = getattr(lgcm, "ContextGate", None)
    Kind = getattr(lgcm, "ContextDecisionKind", None)
    assert Gate is not None and Kind is not None
    a = _trained_expert(0, 1.0)
    b = _trained_expert(1, -1.0)
    gate = Gate(max_experts=4, switch_margin=0.2, support_penalty=0.0, spawn_patience=2)
    phi = np.array([1.0, 0.25])
    scores = gate.score_experts([a, b], phi, np.array([1.0]), active_expert_id=1)

    quiet = gate.decide(
        scores,
        active_expert_id=1,
        mismatch_triggered=False,
        expert_count=2,
    )
    assert quiet.kind is Kind.KEEP
    assert quiet.expert_id == 1

    shifted = gate.decide(
        scores,
        active_expert_id=1,
        mismatch_triggered=True,
        expert_count=2,
    )
    assert shifted.kind is Kind.SWITCH
    assert shifted.expert_id == 0


def test_gate_spawns_only_after_persistent_unexplained_mismatch():
    Gate = getattr(lgcm, "ContextGate", None)
    Kind = getattr(lgcm, "ContextDecisionKind", None)
    assert Gate is not None and Kind is not None
    a = _trained_expert(0, 0.0)
    gate = Gate(max_experts=2, switch_margin=0.2, support_penalty=0.0, spawn_patience=2)
    phi = np.array([1.0, 0.25])
    scores = gate.score_experts([a], phi, np.array([5.0]), active_expert_id=0)

    first = gate.decide(scores, active_expert_id=0, mismatch_triggered=True, expert_count=1)
    assert first.kind is Kind.MISMATCH
    second = gate.decide(scores, active_expert_id=0, mismatch_triggered=True, expert_count=1)
    assert second.kind is Kind.SPAWN


def test_gate_reports_capacity_exhaustion_instead_of_eviction():
    Gate = getattr(lgcm, "ContextGate", None)
    Kind = getattr(lgcm, "ContextDecisionKind", None)
    assert Gate is not None and Kind is not None
    a = _trained_expert(0, 0.0)
    gate = Gate(max_experts=1, switch_margin=0.2, support_penalty=0.0, spawn_patience=1)
    scores = gate.score_experts(
        [a],
        np.array([1.0, 0.25]),
        np.array([5.0]),
        active_expert_id=0,
    )
    decision = gate.decide(
        scores,
        active_expert_id=0,
        mismatch_triggered=True,
        expert_count=1,
    )
    assert decision.kind is Kind.CAPACITY_EXHAUSTED
