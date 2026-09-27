import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import lgcm


def _model():
    return lgcm.ContextualWorldModel(
        lgcm.LGCMConfig(
            observation_dim=1,
            action_dim=1,
            regularization=0.1,
            expert_forgetting_factor=0.8,
            residual_decay=0.8,
            residual_floor=0.02,
            min_expert_evidence=2,
            mismatch_threshold=0.8,
            mismatch_min_evidence=2,
            spawn_patience=1,
            bootstrap_size=4,
            new_expert_init="neutral",
        )
    )


def _event(sequence, value):
    return lgcm.Experience(
        sequence=sequence,
        observation=np.array([0.0]),
        action=np.array([0.0]),
        next_observation=np.array([value]),
        source="simulator",
    )


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rewrite_state(path: Path, mutate) -> dict:
    state_path = path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mutate(state)
    state_path.write_text(
        json.dumps(state, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["state.json"] = _digest(state_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return state


def _rehash_payload(path: Path, name: str) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = _digest(path / name)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_persistence_public_api_exists():
    assert hasattr(lgcm, "save_snapshot")
    assert hasattr(lgcm, "load_snapshot")


def test_snapshot_refuses_overwrite_by_default(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    with pytest.raises(FileExistsError):
        lgcm.save_snapshot(model, path)


def test_digest_mismatch_is_rejected_before_model_construction(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    payload = path / "shared_weights.npy"
    payload.write_bytes(payload.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="digest"):
        lgcm.load_snapshot(path)


def test_snapshot_rejects_parent_directory_payload_escape(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    source = path / state["experts"][0]["weights_file"]
    np.save(tmp_path / "outside.npy", np.load(source, allow_pickle=False), allow_pickle=False)

    _rewrite_state(
        path,
        lambda value: value["experts"][0].__setitem__(
            "weights_file",
            "../outside.npy",
        ),
    )

    with pytest.raises(ValueError, match="snapshot payload"):
        lgcm.load_snapshot(path)


def test_snapshot_rejects_absolute_payload_reference(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    source = path / state["experts"][0]["weights_file"]
    outside = tmp_path / "absolute.npy"
    np.save(outside, np.load(source, allow_pickle=False), allow_pickle=False)

    _rewrite_state(
        path,
        lambda value: value["experts"][0].__setitem__(
            "weights_file",
            str(outside.resolve()),
        ),
    )

    with pytest.raises(ValueError, match="snapshot payload"):
        lgcm.load_snapshot(path)


def test_snapshot_rejects_state_payload_not_bound_by_manifest(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    source = path / state["experts"][0]["weights_file"]
    np.save(path / "unbound.npy", np.load(source, allow_pickle=False), allow_pickle=False)

    _rewrite_state(
        path,
        lambda value: value["experts"][0].__setitem__(
            "weights_file",
            "unbound.npy",
        ),
    )

    with pytest.raises(ValueError, match="manifest"):
        lgcm.load_snapshot(path)


def test_snapshot_rejects_asymmetric_covariance(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    name = state["experts"][0]["covariance_file"]
    payload = path / name
    covariance = np.load(payload, allow_pickle=False)
    covariance[0, 1] += 0.25
    np.save(payload, covariance, allow_pickle=False)
    _rehash_payload(path, name)

    with pytest.raises(ValueError, match="symmetric"):
        lgcm.load_snapshot(path)


def test_snapshot_rejects_negative_evidence_count(tmp_path: Path):
    model = _model()
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)

    _rewrite_state(
        path,
        lambda value: value["experts"][0].__setitem__(
            "regressor_evidence_count",
            -1,
        ),
    )

    with pytest.raises(ValueError, match="evidence_count"):
        lgcm.load_snapshot(path)


def test_snapshot_is_blocked_during_model_update(tmp_path: Path):
    model = _model()
    original_update = model._shared.update
    observed = {"blocked": False}

    def update_with_snapshot_probe(phi, target):
        with pytest.raises(RuntimeError, match="committed generation boundary"):
            lgcm.save_snapshot(model, tmp_path / "during-update")
        observed["blocked"] = True
        return original_update(phi, target)

    model._shared.update = update_with_snapshot_probe
    model.update(_event(0, 0.2))

    assert observed["blocked"]
    assert not (tmp_path / "during-update").exists()
