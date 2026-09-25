"""LGCM: persistent continual-learning research substrate."""

from .baselines import (
    EWRLSWorldModel,
    GlobalRidgeWorldModel,
    ObservationOnlyWorldModel,
    PersistencePredictor,
)
from .encoders import IdentityFeatureEncoder, RandomFeatureEncoder
from .rls import RLSRegressor
from .types import AdequacyState, Experience, Prediction

__version__ = "0.1.0"

__all__ = [
    "AdequacyState",
    "EWRLSWorldModel",
    "GlobalRidgeWorldModel",
    "Experience",
    "IdentityFeatureEncoder",
    "ObservationOnlyWorldModel",
    "PersistencePredictor",
    "Prediction",
    "RLSRegressor",
    "RandomFeatureEncoder",
]
