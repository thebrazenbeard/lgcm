import lgcm


def test_public_encoders_are_exported():
    assert hasattr(lgcm, "IdentityFeatureEncoder")
    assert hasattr(lgcm, "RandomFeatureEncoder")
