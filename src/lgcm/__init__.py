"""LGCM: persistent continual-learning research substrate."""

from .encoders import IdentityFeatureEncoder, RandomFeatureEncoder
from .types import AdequacyState, Experience, Prediction

__version__ = "0.1.0"

__all__ = [
    "AdequacyState",
    "Experience",
    "IdentityFeatureEncoder",
    "Prediction",
    "RandomFeatureEncoder",
]
