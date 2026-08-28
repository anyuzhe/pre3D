"""Exercise QProcess tree cancellation against a real photo scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer

from ai_photogrammetry.engineering.desktop_widgets import ProcessTaskController
from ai_photogrammetry.engineering.project_store import ProjectStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--cancel-after-ms", type=int, default=1500)
    args = parser.parse_args()
    store = (
        ProjectStore.open(args.project_root)
        if (args.project_root / "project.json").is_file()
        else ProjectStore.create(args.project_root, "取消测试")
    )
    application = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    controller = ProcessTaskController()
    outcome = {"code": 3}

    def cancelled(message: str) -> None:
        print(f"cancelled: {message}", flush=True)
        outcome["code"] = 0
        application.quit()

    def failed(message: str, details: str) -> None:
        print(f"failed: {message}\n{details}", flush=True)
        outcome["code"] = 1
        application.quit()

    def unexpected(_result: object) -> None:
        print("worker completed before cancellation", flush=True)
        outcome["code"] = 2
        application.quit()

    controller.cancelled.connect(cancelled)
    controller.failed.connect(failed)
    controller.succeeded.connect(unexpected)
    controller.start(
        {
            "task": "scan_photos",
            "project_root": str(store.root),
            "source_root": str(Path(args.source).resolve()),
            "max_keyframes": 500,
        },
        store.root / "cache" / "checkpoints" / "tasks",
    )
    QTimer.singleShot(args.cancel_after_ms, controller.cancel)
    QTimer.singleShot(30_000, application.quit)
    application.exec()
    return outcome["code"]


if __name__ == "__main__":
    raise SystemExit(main())
