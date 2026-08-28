from __future__ import annotations

import sys
from pathlib import Path

from ai_photogrammetry.engineering import runtime_paths


def test_resource_root_uses_bundle_root(monkeypatch, tmp_path: Path):
    bundle = tmp_path / "_internal"
    bundle.mkdir()
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert runtime_paths.is_frozen()
    assert runtime_paths.resource_root() == bundle.resolve()


def test_user_paths_are_writable_and_overridable(monkeypatch, tmp_path: Path):
    documents = tmp_path / "documents"
    state = tmp_path / "state"
    monkeypatch.setenv("AI_PHOTOGRAMMETRY_DATA_ROOT", str(documents))
    monkeypatch.setenv("AI_PHOTOGRAMMETRY_STATE_ROOT", str(state))

    paths = runtime_paths.ensure_user_directories()

    assert paths["projects"] == documents.resolve() / "项目"
    assert paths["outputs"] == documents.resolve() / "成果"
    assert paths["logs"] == state.resolve() / "logs"
    assert all(path.is_dir() for path in paths.values())


def test_executable_root_uses_frozen_executable(monkeypatch, tmp_path: Path):
    executable = tmp_path / "install" / "RockVision.exe"
    executable.parent.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert runtime_paths.executable_root() == executable.parent.resolve()
