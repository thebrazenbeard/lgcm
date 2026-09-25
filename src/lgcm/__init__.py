"""LGCM: persistent continual-learning research substrate."""

from .change_detection import PageHinkleyDetector, PageHinkleyState
from .context import (
    ContextDecision,
    ContextDecisionKind,
    ContextExpert,
    ContextGate,
    ExpertScore,
)
from .baselines import (
    EWRLSWorldModel,
    GlobalRidgeWorldModel,
    ObservationOnlyWorldModel,
    PersistencePredictor,
)
from .encoders import IdentityFeatureEncoder, RandomFeatureEncoder
from .model import (
    BootstrapRecord,
    ContextualWorldModel,
    LGCMConfig,
    UpdateReceipt,
)
from .residuals import ResidualScaleTracker
from .rls import RLSRegressor
from .types import AdequacyState, Experience, Prediction

__version__ = "0.1.0"

__all__ = [
    "AdequacyState",
    "ContextDecision",
    "ContextDecisionKind",
    "ContextExpert",
    "ContextGate",
    "ContextualWorldModel",
    "EWRLSWorldModel",
    "ExpertScore",
    "GlobalRidgeWorldModel",
    "Experience",
    "IdentityFeatureEncoder",
    "LGCMConfig",
    "ObservationOnlyWorldModel",
    "PageHinkleyDetector",
    "PageHinkleyState",
    "PersistencePredictor",
    "Prediction",
    "RLSRegressor",
    "ResidualScaleTracker",
    "RandomFeatureEncoder",
    "BootstrapRecord",
    "UpdateReceipt",
]
