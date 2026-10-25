"""Snapshot overwrite and untrusted-path security regressions."""

from hashlib import sha256
import json
import os

import pytest

import lgcm
from lgcm import persistence


def _snapshot(tmp_path):
    model = lgcm.ContextualWorldModel(lgcm.LGCMConfig(1, 1))
    path = tmp_path / "snapshot"
    lgcm.save_snapshot(model, path)
    return model, path


def _refresh_state_digest(path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["state.json"] = sha256(
        (path / "state.json").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_overwrite_must_not_delete_a_preexisting_sibling_backup(tmp_path):
    model, path = _snapshot(tmp_path)
    unrelated = tmp_path / "snapshot.backup"
    unrelated.mkdir()
    sentinel = unrelated / "do-not-delete.txt"
    sentinel.write_text("owner data", encoding="utf-8")

    lgcm.save_snapshot(model, path, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "owner data"
    assert lgcm.load_snapshot(path).generation == model.generation


def test_failed_overwrite_restores_original_snapshot(tmp_path, monkeypatch):
    model, path = _snapshot(tmp_path)
    old_manifest = (path / "manifest.json").read_bytes()
    original_replace = os.replace

    def fail_new_snapshot_publish(source, destination):
        if ".tmp-" in os.fspath(source) and os.fspath(destination) == str(path):
            raise OSError("simulated snapshot publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_new_snapshot_publish)
    with pytest.raises(OSError, match="publish failure"):
        lgcm.save_snapshot(model, path, overwrite=True)

    assert (path / "manifest.json").read_bytes() == old_manifest
    assert lgcm.load_snapshot(path).generation == model.generation
    assert not list(tmp_path.glob(".snapshot.backup-*"))


def test_manifest_cannot_hash_a_file_outside_snapshot(tmp_path):
    _, path = _snapshot(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("outside snapshot", encoding="utf-8")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../external.txt"] = sha256(external.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot.*(path|name)"):
        lgcm.load_snapshot(path)


def test_state_metadata_cannot_load_unlisted_parent_path(tmp_path):
    _, path = _snapshot(tmp_path)
    external = tmp_path / "outside.npy"
    external.write_bytes((path / "shared_weights.npy").read_bytes())
    state_path = path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["experts"][0]["weights_file"] = "../outside.npy"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _refresh_state_digest(path)

    with pytest.raises(ValueError, match="snapshot.*(path|name|manifest)"):
        lgcm.load_snapshot(path)


@pytest.mark.parametrize("unsafe_name", ["..\\outside.npy", "C:outside.npy"])
def test_windows_style_payload_paths_are_rejected(tmp_path, unsafe_name):
    _, path = _snapshot(tmp_path)
    state_path = path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["experts"][0]["weights_file"] = unsafe_name
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _refresh_state_digest(path)

    with pytest.raises(ValueError, match="snapshot.*(path|name|manifest)"):
        lgcm.load_snapshot(path)


def test_snapshot_payload_symlink_must_not_escape_root(tmp_path):
    _, path = _snapshot(tmp_path)
    external = tmp_path / "outside.npy"
    payload = path / "shared_weights.npy"
    external.write_bytes(payload.read_bytes())
    payload.unlink()
    try:
        payload.symlink_to(external)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="snapshot.*(path|link)"):
        lgcm.load_snapshot(path)


def test_snapshot_payload_hardlink_must_not_alias_outside_root(tmp_path):
    _, path = _snapshot(tmp_path)
    external = tmp_path / "outside.npy"
    payload = path / "shared_weights.npy"
    external.write_bytes(payload.read_bytes())
    payload.unlink()
    try:
        os.link(external, payload)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="snapshot.*(path|link)"):
        lgcm.load_snapshot(path)
