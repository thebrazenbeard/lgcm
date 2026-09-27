"""LGCM: persistent continual-learning research substrate."""

from .baselines import (
    EWRLSWorldModel,
    GlobalRidgeWorldModel,
    ObservationOnlyWorldModel,
    PersistencePredictor,
)
from .change_detection import PageHinkleyDetector, PageHinkleyState
from .context import (
    ContextDecision,
    ContextDecisionKind,
    ContextExpert,
    ContextGate,
    ExpertScore,
)
from .encoders import IdentityFeatureEncoder, RandomFeatureEncoder
from .envs import CausalControlEnv, make_continual_regime_stream
from .evaluation import (
    EvaluationStep,
    EvaluationTrace,
    EvaluatorEvent,
    evaluate_prequential,
)
from .metrics import (
    adaptation_error_auc,
    forgetting_delta,
    forward_transfer_ratio,
    interval_mse,
    samples_to_criterion,
)
from .model import (
    BootstrapRecord,
    ContextualWorldModel,
    LGCMConfig,
    UpdateReceipt,
)
from .persistence import SnapshotReceipt, load_snapshot, save_snapshot
from .planning import PlanResult, plan_action
from .qualification import (
    QualificationAggregate,
    QualificationReceipt,
    QualificationResource,
    QualificationResult,
    QualificationRun,
    SummaryStats,
    run_qualification,
)
from .residuals import ResidualScaleTracker
from .rls import RLSRegressor
from .types import AdequacyState, Experience, Prediction

__version__ = "0.1.0"

__all__ = [
    "AdequacyState",
    "BootstrapRecord",
    "CausalControlEnv",
    "ContextDecision",
    "ContextDecisionKind",
    "ContextExpert",
    "ContextGate",
    "ContextualWorldModel",
    "EWRLSWorldModel",
    "EvaluationStep",
    "EvaluationTrace",
    "EvaluatorEvent",
    "Experience",
    "ExpertScore",
    "GlobalRidgeWorldModel",
    "IdentityFeatureEncoder",
    "LGCMConfig",
    "ObservationOnlyWorldModel",
    "PageHinkleyDetector",
    "PageHinkleyState",
    "PersistencePredictor",
    "PlanResult",
    "Prediction",
    "QualificationAggregate",
    "QualificationReceipt",
    "QualificationResource",
    "QualificationResult",
    "QualificationRun",
    "RLSRegressor",
    "RandomFeatureEncoder",
    "ResidualScaleTracker",
    "SnapshotReceipt",
    "SummaryStats",
    "UpdateReceipt",
    "adaptation_error_auc",
    "evaluate_prequential",
    "forgetting_delta",
    "forward_transfer_ratio",
    "interval_mse",
    "load_snapshot",
    "make_continual_regime_stream",
    "plan_action",
    "run_qualification",
    "samples_to_criterion",
    "save_snapshot",
]
