"""Reject false context recall when every retained expert fits poorly."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lgcm import (
    ContextDecisionKind,
    ContextGate,
    ContextualWorldModel,
    ExpertScore,
    LGCMConfig,
    load_snapshot,
    save_snapshot,
)


def _score(expert_id, error):
    return ExpertScore(
        expert_id=expert_id,
        standardized_error=error,
        feature_support=1.0,
        total_score=error,
        adequate=True,
    )


class ContextRecallConfidenceTests(unittest.TestCase):
    def test_two_bad_models_cannot_misidentify_novel_context(self):
        gate = ContextGate(
            max_experts=3,
            switch_margin=0.25,
            support_penalty=0.0,
            spawn_patience=2,
        )
        # Candidate is relatively better yet nine historical residual scales
        # away from the observed transition: neither known model explains it.
        scores = (_score(0, 16.0), _score(1, 9.0))
        first = gate.decide(
            scores,
            active_expert_id=0,
            mismatch_triggered=True,
            expert_count=2,
        )
        self.assertEqual(first.kind, ContextDecisionKind.MISMATCH)
        second = gate.decide(
            scores,
            active_expert_id=0,
            mismatch_triggered=True,
            expert_count=2,
        )
        self.assertEqual(second.kind, ContextDecisionKind.SPAWN)

    def test_well_explained_return_still_recalls_retained_model(self):
        gate = ContextGate(
            max_experts=3,
            switch_margin=0.25,
            support_penalty=0.0,
            spawn_patience=2,
        )
        decision = gate.decide(
            (_score(0, 16.0), _score(1, 1.5)),
            active_expert_id=0,
            mismatch_triggered=True,
            expert_count=2,
        )
        self.assertEqual(decision.kind, ContextDecisionKind.SWITCH)
        self.assertEqual(decision.expert_id, 1)

    def test_recall_ceiling_is_configurable_and_restored(self):
        config = LGCMConfig(
            observation_dim=1,
            action_dim=1,
            max_recall_standardized_error=0.5,
        )
        model = ContextualWorldModel(config)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot"
            save_snapshot(model, path)
            restored = load_snapshot(path)
        self.assertEqual(restored._gate.max_recall_standardized_error, 0.5)
        other = ContextGate(
            max_experts=3,
            switch_margin=0.25,
            support_penalty=0.0,
            spawn_patience=2,
            max_recall_standardized_error=0.5,
        )
        decision = other.decide(
            (_score(0, 16.0), _score(1, 1.5)),
            active_expert_id=0,
            mismatch_triggered=True,
            expert_count=2,
        )
        self.assertEqual(decision.kind, ContextDecisionKind.MISMATCH)

    def test_invalid_recall_ceiling_is_rejected(self):
        for value in (-1.0, 0.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LGCMConfig(1, 1, max_recall_standardized_error=value)


if __name__ == "__main__":
    unittest.main()
