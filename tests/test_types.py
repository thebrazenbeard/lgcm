import lgcm


def test_public_contracts_are_exported():
    assert hasattr(lgcm, "Experience")
    assert hasattr(lgcm, "AdequacyState")
    assert hasattr(lgcm, "Prediction")
