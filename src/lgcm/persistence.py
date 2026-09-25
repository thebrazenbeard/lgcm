from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .change_detection import PageHinkleyDetector
from .context import ContextDecisionKind, ContextExpert, ContextGate
from .encoders import IdentityFeatureEncoder, RandomFeatureEncoder
from .model import BootstrapRecord, ContextualWorldModel, LGCMConfig
from .residuals import ResidualScaleTracker
from .rls import RLSRegressor

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SnapshotReceipt:
    path: Path
    generation: int
    manifest_sha256: str


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_array(root: Path, name: str, value) -> str:
    path = root / name
    np.save(path, np.ascontiguousarray(value, dtype=np.float64), allow_pickle=False)
    return name


def _encoder_state(model: ContextualWorldModel, root: Path) -> dict:
    encoder = model.encoder
    if isinstance(encoder, IdentityFeatureEncoder):
        return {
            "type": "identity",
            "observation_dim": encoder.observation_dim,
            "action_dim": encoder.action_dim,
            "lower_bound": encoder.lower_bound,
            "upper_bound": encoder.upper_bound,
        }
    if isinstance(encoder, RandomFeatureEncoder):
        _save_array(root, "encoder_projection.npy", encoder.projection)
        return {
            "type": "random",
            "observation_dim": encoder.observation_dim,
            "action_dim": encoder.action_dim,
            "width": encoder.width,
            "seed": encoder.seed,
            "max_feature_dim": encoder.max_feature_dim,
            "lower_bound": encoder.lower_bound,
            "upper_bound": encoder.upper_bound,
            "projection_file": "encoder_projection.npy",
        }
    raise TypeError(f"unsupported encoder type: {type(encoder).__name__}")


def save_snapshot(
    model: ContextualWorldModel,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> SnapshotReceipt:
    if getattr(model, "_update_in_progress", False):
        raise RuntimeError("snapshot requires a committed generation boundary")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(target)

    temp = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    backup: Path | None = None
    try:
        encoder = _encoder_state(model, temp)
        arrays: list[str] = []
        arrays.append(_save_array(temp, "shared_weights.npy", model._shared.weights))
        arrays.append(
            _save_array(temp, "shared_covariance.npy", model._shared.covariance)
        )

        experts_meta = []
        for expert_id in model.expert_ids:
            expert = model._experts[expert_id]
            weight_file = f"expert_{expert_id}_weights.npy"
            covariance_file = f"expert_{expert_id}_covariance.npy"
            residual_file = f"expert_{expert_id}_residual_mse.npy"
            arrays.extend(
                [
                    _save_array(temp, weight_file, expert.regressor.weights),
                    _save_array(temp, covariance_file, expert.regressor.covariance),
                    _save_array(
                        temp,
                        residual_file,
                        expert.residual_scale._mean_square,
                    ),
                ]
            )
            experts_meta.append(
                {
                    "expert_id": expert_id,
                    "regressor_evidence_count": expert.regressor.evidence_count,
                    "residual_evidence_count": expert.residual_scale.evidence_count,
                    "residual_decay": expert.residual_scale.decay,
                    "residual_floor": expert.residual_scale.floor,
                    "residual_min_evidence": expert.residual_scale.min_evidence,
                    "weights_file": weight_file,
                    "covariance_file": covariance_file,
                    "residual_file": residual_file,
                }
            )

        if model._bootstrap:
            bootstrap_features = np.stack(
                [record.encoded_features for record in model._bootstrap]
            )
            bootstrap_targets = np.stack(
                [record.target_delta for record in model._bootstrap]
            )
        else:
            bootstrap_features = np.empty(
                (0, model.encoder.feature_dim), dtype=np.float64
            )
            bootstrap_targets = np.empty(
                (0, model.config.observation_dim), dtype=np.float64
            )
        arrays.append(
            _save_array(temp, "bootstrap_features.npy", bootstrap_features)
        )
        arrays.append(
            _save_array(temp, "bootstrap_targets.npy", bootstrap_targets)
        )

        mismatch_state = model._mismatch.state
        state = {
            "schema_version": SCHEMA_VERSION,
            "model_type": "ContextualWorldModel",
            "config": asdict(model.config),
            "encoder": encoder,
            "generation": model.generation,
            "last_sequence": model.last_sequence,
            "active_expert_id": model._active_expert_id,
            "next_expert_id": model._next_expert_id,
            "capacity_exhausted": model._capacity_exhausted,
            "last_decision": model._last_decision.value,
            "shared_evidence_count": model._shared.evidence_count,
            "experts": experts_meta,
            "gate": {
                "max_experts": model._gate.max_experts,
                "switch_margin": model._gate.switch_margin,
                "support_penalty": model._gate.support_penalty,
                "spawn_patience": model._gate.spawn_patience,
                "unexplained_streak": model._gate.unexplained_streak,
            },
            "mismatch": {
                "delta": model._mismatch.delta,
                "threshold": model._mismatch.threshold,
                "min_evidence": model._mismatch.min_evidence,
                "count": mismatch_state.count,
                "mean": mismatch_state.mean,
                "cumulative": mismatch_state.cumulative,
                "minimum_cumulative": mismatch_state.minimum_cumulative,
            },
            "bootstrap_count": len(model._bootstrap),
        }
        state_path = temp / "state.json"
        state_path.write_text(
            json.dumps(state, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        files = ["state.json", *arrays]
        if encoder["type"] == "random":
            files.append(encoder["projection_file"])
        # Preserve first occurrence while keeping deterministic order.
        files = list(dict.fromkeys(files))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_type": "ContextualWorldModel",
            "files": {name: _digest(temp / name) for name in sorted(files)},
        }
        manifest_path = temp / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        if target.exists():
            backup = target.with_name(target.name + ".backup")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        os.replace(temp, target)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)

        return SnapshotReceipt(
            path=target,
            generation=model.generation,
            manifest_sha256=_digest(target / "manifest.json"),
        )
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def _load_array(root: Path, name: str) -> np.ndarray:
    value = np.load(root / name, allow_pickle=False)
    value = np.ascontiguousarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"snapshot array {name} contains non-finite values")
    return value


def _restore_regressor(
    feature_dim: int,
    output_dim: int,
    *,
    regularization: float,
    forgetting_factor: float,
    weights: np.ndarray,
    covariance: np.ndarray,
    evidence_count: int,
) -> RLSRegressor:
    model = RLSRegressor(
        feature_dim,
        output_dim,
        regularization=regularization,
        forgetting_factor=forgetting_factor,
        initial_weights=weights,
    )
    if covariance.shape != (feature_dim, feature_dim):
        raise ValueError("snapshot covariance shape mismatch")
    model._covariance = covariance.copy()
    model.evidence_count = int(evidence_count)
    return model


def load_snapshot(path: str | Path) -> ContextualWorldModel:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("snapshot manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema version")
    if manifest.get("model_type") != "ContextualWorldModel":
        raise ValueError("unsupported snapshot model type")
    for name, expected in manifest.get("files", {}).items():
        payload = root / name
        if not payload.is_file():
            raise ValueError(f"snapshot payload missing: {name}")
        actual = _digest(payload)
        if actual != expected:
            raise ValueError(f"snapshot digest mismatch: {name}")

    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported snapshot state schema version")
    config = LGCMConfig(**state["config"])
    encoder_meta = state["encoder"]
    if encoder_meta["type"] == "identity":
        encoder = IdentityFeatureEncoder(
            encoder_meta["observation_dim"],
            encoder_meta["action_dim"],
            lower_bound=encoder_meta["lower_bound"],
            upper_bound=encoder_meta["upper_bound"],
        )
    elif encoder_meta["type"] == "random":
        encoder = RandomFeatureEncoder(
            encoder_meta["observation_dim"],
            encoder_meta["action_dim"],
            width=encoder_meta["width"],
            seed=encoder_meta["seed"],
            max_feature_dim=encoder_meta["max_feature_dim"],
            lower_bound=encoder_meta["lower_bound"],
            upper_bound=encoder_meta["upper_bound"],
        )
        projection = _load_array(root, encoder_meta["projection_file"])
        if projection.shape != encoder.projection.shape:
            raise ValueError("snapshot encoder projection shape mismatch")
        encoder._projection = projection.copy()
        encoder._projection.setflags(write=False)
    else:
        raise ValueError("unsupported snapshot encoder type")

    model = ContextualWorldModel(config, encoder=encoder)
    model._shared = _restore_regressor(
        encoder.feature_dim,
        config.observation_dim,
        regularization=config.regularization,
        forgetting_factor=config.shared_forgetting_factor,
        weights=_load_array(root, "shared_weights.npy"),
        covariance=_load_array(root, "shared_covariance.npy"),
        evidence_count=state["shared_evidence_count"],
    )

    experts: dict[int, ContextExpert] = {}
    for meta in state["experts"]:
        expert_id = int(meta["expert_id"])
        regressor = _restore_regressor(
            encoder.feature_dim,
            config.observation_dim,
            regularization=config.regularization,
            forgetting_factor=config.expert_forgetting_factor,
            weights=_load_array(root, meta["weights_file"]),
            covariance=_load_array(root, meta["covariance_file"]),
            evidence_count=meta["regressor_evidence_count"],
        )
        tracker = ResidualScaleTracker(
            config.observation_dim,
            decay=meta["residual_decay"],
            floor=meta["residual_floor"],
            min_evidence=meta["residual_min_evidence"],
        )
        residual_mse = _load_array(root, meta["residual_file"])
        if residual_mse.shape != (config.observation_dim,):
            raise ValueError("snapshot residual scale shape mismatch")
        tracker._mean_square = residual_mse.copy()
        tracker.evidence_count = int(meta["residual_evidence_count"])
        experts[expert_id] = ContextExpert(expert_id, regressor, tracker)
    if not experts:
        raise ValueError("snapshot contains no context experts")
    model._experts = experts

    gate_meta = state["gate"]
    model._gate = ContextGate(
        max_experts=gate_meta["max_experts"],
        switch_margin=gate_meta["switch_margin"],
        support_penalty=gate_meta["support_penalty"],
        spawn_patience=gate_meta["spawn_patience"],
    )
    model._gate._unexplained_streak = int(gate_meta["unexplained_streak"])

    mismatch_meta = state["mismatch"]
    model._mismatch = PageHinkleyDetector(
        delta=mismatch_meta["delta"],
        threshold=mismatch_meta["threshold"],
        min_evidence=mismatch_meta["min_evidence"],
    )
    model._mismatch._count = int(mismatch_meta["count"])
    model._mismatch._mean = float(mismatch_meta["mean"])
    model._mismatch._cumulative = float(mismatch_meta["cumulative"])
    model._mismatch._minimum_cumulative = float(
        mismatch_meta["minimum_cumulative"]
    )

    bootstrap_features = _load_array(root, "bootstrap_features.npy")
    bootstrap_targets = _load_array(root, "bootstrap_targets.npy")
    if bootstrap_features.shape[0] != bootstrap_targets.shape[0]:
        raise ValueError("snapshot bootstrap length mismatch")
    model._bootstrap = deque(maxlen=config.bootstrap_size)
    for features, target in zip(bootstrap_features, bootstrap_targets):
        model._bootstrap.append(BootstrapRecord(features, target))

    model._active_expert_id = int(state["active_expert_id"])
    if model._active_expert_id not in model._experts:
        raise ValueError("snapshot active expert is missing")
    model._next_expert_id = int(state["next_expert_id"])
    model.generation = int(state["generation"])
    model.last_sequence = int(state["last_sequence"])
    model._capacity_exhausted = bool(state["capacity_exhausted"])
    model._last_decision = ContextDecisionKind(state["last_decision"])
    return model
