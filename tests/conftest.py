from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_desktop_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep UI tests out of the production recent-project registry."""

    monkeypatch.setenv(
        "AI_PHOTOGRAMMETRY_SETTINGS_PATH",
        str(tmp_path / "desktop_test_settings.ini"),
    )
