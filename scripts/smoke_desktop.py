"""Launch the native window with a synthetic MVS point cloud for UI verification."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ai_photogrammetry.engineering.desktop import APP_STYLE, EngineeringMainWindow  # noqa: E402
from ai_photogrammetry.engineering.exporters import write_binary_ply  # noqa: E402


def _populate(
    window: EngineeringMainWindow,
    sparse_ply: Path | None = None,
    camera_images: Path | None = None,
) -> None:
    if sparse_ply:
        window.session.sparse_result = {
            "sparse_pointcloud": str(sparse_ply),
            "sparse_images_txt": str(camera_images or ""),
            "image_count": 0,
            "registered_images": 0,
            "sparse_point_count": 0,
        }
        window.cloud_view.load_pointcloud(
            str(sparse_ply),
            unit="模型单位",
            camera_images_txt=str(camera_images or ""),
            label="Sparse BA",
        )
        window._update_project_status()
        return
    height, width = 96, 128
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx - width / 2) / 35.0
    y = (yy - height / 2) / 35.0
    z = 2.5 + 0.12 * np.sin(x * 3) * np.cos(y * 2)
    points = np.stack([x, -y, z], axis=-1).reshape(-1, 3).astype(np.float64)
    colors = np.stack(
        [
            np.clip((x - x.min()) / np.ptp(x), 0, 1),
            np.clip((y - y.min()) / np.ptp(y), 0, 1),
            np.full_like(x, 0.65),
        ],
        axis=-1,
    ).reshape(-1, 3)
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    cloud = Path.cwd() / ".cache" / "smoke_desktop_ai_cloud.ply"
    write_binary_ply(cloud, points, colors, ["desktop smoke test"])
    window.session.project_name = "桌面界面验收"
    window.session.photogrammetry_result = {
        "pointcloud": str(cloud),
        "image_count": 3,
        "registered_images": 3,
        "point_count": len(points),
        "unit": "模型单位",
    }
    window.cloud_view.load_pointcloud(str(cloud), unit=window.session.unit)
    window._update_project_status()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--tab", type=int, default=0, choices=range(6))
    parser.add_argument("--settings-tab", type=int, choices=range(3))
    parser.add_argument("--sparse-ply", type=Path)
    parser.add_argument("--camera-images", type=Path)
    args = parser.parse_args()
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    application.setStyleSheet(APP_STYLE)
    window = EngineeringMainWindow()
    _populate(window, args.sparse_ply, args.camera_images)
    window.tabs.setCurrentIndex(args.tab)
    if args.settings_tab is not None:
        window.settings_tabs.setCurrentIndex(args.settings_tab)
    window.show()

    def finish() -> None:
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            vtk_path = args.screenshot.with_name(f"{args.screenshot.stem}_vtk.png")
            window.cloud_view.plotter.screenshot(str(vtk_path))
        window.close()
        application.quit()

    QTimer.singleShot(max(200, int(args.seconds * 1000)), finish)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
