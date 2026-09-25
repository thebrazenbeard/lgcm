import lgcm


def test_samples_to_criterion_requires_consecutive_successes():
    errors = [1.0, 0.4, 0.6, 0.3, 0.2, 0.1]
    assert lgcm.samples_to_criterion(errors, threshold=0.5, consecutive=3) == 6
    assert lgcm.samples_to_criterion(errors, threshold=0.05, consecutive=2) is None


def test_transfer_and_forgetting_metrics_have_explicit_direction():
    assert lgcm.forgetting_delta(pre_error=0.1, return_error=0.4) == 0.3
    assert lgcm.forward_transfer_ratio(fresh_samples=20, experienced_samples=10) == 2.0


def test_interval_mse_uses_evaluator_labels_only_for_scoring():
    steps = [
        lgcm.EvaluationStep(0, "A", mse=0.1, active_expert=0, adequacy="ok"),
        lgcm.EvaluationStep(1, "B", mse=0.9, active_expert=1, adequacy="ok"),
        lgcm.EvaluationStep(2, "A", mse=0.3, active_expert=0, adequacy="ok"),
    ]
    trace = lgcm.EvaluationTrace(tuple(steps))
    assert abs(lgcm.interval_mse(trace, regime="A") - 0.2) < 1e-12
    assert abs(lgcm.interval_mse(trace) - (1.3 / 3.0)) < 1e-12
