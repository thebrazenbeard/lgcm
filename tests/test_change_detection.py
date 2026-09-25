import lgcm


def test_change_detector_public_api_exists():
    assert hasattr(lgcm, "PageHinkleyDetector")


def test_page_hinkley_ignores_stationary_stream_and_detects_sustained_shift():
    cls = getattr(lgcm, "PageHinkleyDetector", None)
    assert cls is not None
    detector = cls(delta=0.05, threshold=4.0, min_evidence=10)
    for value in [1.0, 0.9, 1.1, 1.0] * 10:
        state = detector.update(value)
        assert not state.triggered

    triggered = False
    for _ in range(20):
        state = detector.update(3.0)
        if state.triggered:
            triggered = True
            break
    assert triggered


def test_page_hinkley_reset_removes_old_shift_evidence():
    cls = getattr(lgcm, "PageHinkleyDetector", None)
    assert cls is not None
    detector = cls(delta=0.0, threshold=1.0, min_evidence=2)
    detector.update(0.0)
    detector.update(5.0)
    assert detector.state.triggered
    detector.reset()
    assert detector.state.count == 0
    assert detector.state.score == 0.0
    assert not detector.state.triggered
