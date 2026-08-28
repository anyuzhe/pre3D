"""Runtime paths shared by source checkouts and frozen Windows releases."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PRODUCT_DIRECTORY = "岩创科技/岩土影像三维重建工作台"


def is_frozen() -> bool:
    """Return whether the application is running from a PyInstaller bundle."""

    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Return the read-only root containing bundled tools, models, and docs."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parents[2]


def executable_root() -> Path:
    """Return the directory containing the installed executable."""

    if is_frozen():
        return Path(sys.executable).resolve().parent
    return resource_root()


def user_documents_root() -> Path:
    """Return a user-writable location for projects and exported results."""

    override = os.environ.get("AI_PHOTOGRAMMETRY_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "Documents" / Path(PRODUCT_DIRECTORY)).resolve()


def user_state_root() -> Path:
    """Return a user-writable location for logs and application state."""

    override = os.environ.get("AI_PHOTOGRAMMETRY_STATE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return (base / Path(PRODUCT_DIRECTORY)).resolve()


def ensure_user_directories() -> dict[str, Path]:
    """Create and return the standard writable application directories."""

    documents = user_documents_root()
    state = user_state_root()
    paths = {
        "documents": documents,
        "projects": documents / "项目",
        "outputs": documents / "成果",
        "logs": state / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
