"""Reusable Qt widgets for the native engineering point-cloud application."""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QPointF, QProcess, QProcessEnvironment, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from .calibration import SimilarityTransform
from .pointcloud_io import load_ply_preview
from .runtime_paths import executable_root, is_frozen, resource_root


def _colmap_camera_poses(images_txt: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(images_txt)
    if not path.is_file():
        return np.empty((0, 3)), np.empty((0, 3))
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    centers: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for index in range(0, len(lines), 2):
        parts = lines[index].split(maxsplit=9)
        if len(parts) != 10:
            continue
        try:
            qw, qx, qy, qz = (float(value) for value in parts[1:5])
            translation = np.asarray(parts[5:8], dtype=np.float64)
        except ValueError:
            continue
        rotation = np.asarray(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )
        centers.append(-(rotation.T @ translation))
        directions.append(rotation.T @ np.asarray([0.0, 0.0, 1.0]))
    return np.asarray(centers), np.asarray(directions)


class ProcessTaskController(QObject):
    """Run heavy work outside Qt and consume the worker's JSON-lines events."""

    progress_changed = Signal(int, str)
    telemetry_changed = Signal(object)
    status_changed = Signal(str, str)
    log_received = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._buffer = ""
        self._tail: list[str] = []
        self._result_received = False
        self._result_event: dict[str, Any] | None = None
        self._terminal_emitted = False
        self._cancel_requested = False
        self._config_path: Path | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, config: dict[str, Any], checkpoint_dir: str | Path) -> Path:
        if self.running:
            raise RuntimeError("已有后台任务正在运行")
        directory = Path(checkpoint_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = directory / f"task_{config.get('task', 'worker')}_{stamp}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        self._config_path = path
        self._buffer = ""
        self._tail.clear()
        self._result_received = False
        self._result_event = None
        self._terminal_emitted = False
        self._cancel_requested = False

        project_root = resource_root()
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        if not is_frozen():
            environment.insert("PYTHONPATH", str(project_root))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(executable_root()))
        self.process.setProgram(sys.executable)
        if is_frozen():
            self.process.setArguments(["--worker", "--config", str(path)])
        else:
            self.process.setArguments(
                ["-m", "ai_photogrammetry.engineering.worker", "--config", str(path)]
            )
        self.process.start()
        if not self.process.waitForStarted(5000):
            raise RuntimeError(f"无法启动独立任务进程：{self.process.errorString()}")
        return path

    def cancel(self) -> None:
        if not self.running:
            return
        self._cancel_requested = True
        process_id = int(self.process.processId())
        self.status_changed.emit("cancelling", "正在取消任务并释放 GPU")
        if sys.platform == "win32" and process_id:
            QProcess.startDetached(
                "taskkill.exe",
                ["/PID", str(process_id), "/T", "/F"],
            )
        else:
            self.process.kill()

    def _read_output(self) -> None:
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        self._tail.append(line)
        self._tail = self._tail[-30:]
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self.log_received.emit(line)
            return
        event_type = event.get("type")
        if event_type == "progress":
            self.progress_changed.emit(
                int(np.clip(float(event.get("progress", 0)), 0, 100)),
                str(event.get("message", "")),
            )
        elif event_type == "telemetry":
            self.telemetry_changed.emit(event)
        elif event_type == "status":
            self.status_changed.emit(
                str(event.get("status", "")),
                str(event.get("message", "")),
            )
        elif event_type == "result":
            self._result_received = True
            # The worker prints its result just before it exits.  Keep the
            # terminal event until QProcess has reached NotRunning; otherwise
            # an automatic next stage can race the still-running worker.
            self._result_event = dict(event)
        else:
            self.log_received.emit(line)

    def _finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        if self._buffer.strip():
            self._handle_line(self._buffer.strip())
        self._buffer = ""
        if self._result_event is not None and not self._terminal_emitted:
            self._terminal_emitted = True
            event = self._result_event
            self._result_event = None
            status = str(event.get("status", "failed"))
            if status == "completed":
                self.succeeded.emit(event.get("result") or {})
            elif status == "cancelled":
                self.cancelled.emit(str(event.get("error", "任务已取消")))
            else:
                self.failed.emit(
                    str(event.get("error", "后台任务失败")),
                    str(event.get("traceback", "")),
                )
        elif not self._result_received and not self._terminal_emitted:
            self._terminal_emitted = True
            if self._cancel_requested:
                self.cancelled.emit("任务已取消")
            else:
                tail = "\n".join(self._tail[-12:])
                self.failed.emit(
                    f"独立任务进程异常退出（退出码 {exit_code}）",
                    tail,
                )

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if (
            not self.running
            and not self._terminal_emitted
            and not self._cancel_requested
        ):
            self._terminal_emitted = True
            self.failed.emit(self.process.errorString(), "\n".join(self._tail[-12:]))


class PhotoPointPicker(QLabel):
    """Aspect-ratio-safe image viewer that reports source-image pixels."""

    point_clicked = Signal(float, float)

    def __init__(self, empty_text: str = "重建后在这里选择照片上的点") -> None:
        super().__init__(empty_text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(460, 340)
        self.setStyleSheet(
            "PhotoPointPicker { background:#101820; color:#9fb1b8;"
            " border:1px solid #40515a; border-radius:6px; }"
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._image = QImage()
        self._marks: list[tuple[float, float]] = []

    def set_array(self, image: np.ndarray | None) -> None:
        if image is None:
            self._image = QImage()
            self._marks.clear()
            self.update()
            return
        array = np.asarray(image)
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("照片必须是灰度或 RGB 数组")
        array = np.ascontiguousarray(array[..., :3])
        height, width = array.shape[:2]
        self._image = QImage(
            array.data,
            width,
            height,
            int(array.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._marks.clear()
        self.update()

    def set_marks(self, marks: list[tuple[float, float]]) -> None:
        self._marks = list(marks)
        self.update()

    def add_mark(self, x: float, y: float) -> None:
        self._marks.append((float(x), float(y)))
        self.update()

    def clear_marks(self) -> None:
        self._marks.clear()
        self.update()

    def _image_rect(self) -> QRect:
        if self._image.isNull():
            return QRect()
        scaled = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        left = (self.width() - scaled.width()) // 2
        top = (self.height() - scaled.height()) // 2
        return QRect(left, top, scaled.width(), scaled.height())

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._image.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = self._image_rect()
        painter.drawPixmap(target, QPixmap.fromImage(self._image))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, (x, y) in enumerate(self._marks, 1):
            screen_x = target.left() + x / self._image.width() * target.width()
            screen_y = target.top() + y / self._image.height() * target.height()
            painter.setPen(QPen(QColor("#ffca3a"), 3))
            painter.setBrush(QColor(255, 202, 58, 80))
            painter.drawEllipse(QPointF(screen_x, screen_y), 7, 7)
            painter.setPen(QPen(QColor("white"), 1))
            painter.drawText(QPointF(screen_x + 9, screen_y - 7), str(index))
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._image.isNull():
            super().mousePressEvent(event)
            return
        target = self._image_rect()
        position = event.position()
        if not target.contains(position.toPoint()):
            return
        x = (position.x() - target.left()) / target.width() * self._image.width()
        y = (position.y() - target.top()) / target.height() * self._image.height()
        x = float(np.clip(x, 0, self._image.width() - 1))
        y = float(np.clip(y, 0, self._image.height() - 1))
        self.point_clicked.emit(x, y)


class CloudView(QWidget):
    """Native VTK point-cloud view embedded into the Qt main window."""

    point_picked = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#0b1117", top="#172832")
        self.plotter.add_text(
            "Point cloud appears after AI photogrammetry\nLeft click: pick a 3D point",
            position="upper_left",
            color="#b8c8cf",
            font_size=10,
            name="hint",
        )
        self.plotter.show_axes()
        self._display_origin = np.zeros(3, dtype=np.float64)
        self._enable_picking()

    def _enable_picking(self) -> None:
        try:
            self.plotter.disable_picking()
        except RuntimeError:
            pass
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            show_message=False,
            show_point=True,
            color="#ffca3a",
            point_size=12,
            picker="point",
            left_clicking=True,
        )

    def _on_pick(self, point: np.ndarray | None) -> None:
        if point is None:
            return
        value = np.asarray(point, dtype=np.float64)
        if value.shape == (3,) and np.isfinite(value).all():
            self.point_picked.emit(value + self._display_origin)

    def clear(self) -> None:
        self.plotter.clear()
        self.plotter.add_text(
            "Point cloud appears after AI photogrammetry\nLeft click: pick a 3D point",
            position="upper_left",
            color="#b8c8cf",
            font_size=10,
            name="hint",
        )
        self.plotter.show_axes()
        self._enable_picking()


    def load_pointcloud(
        self,
        path: str,
        *,
        unit: str,
        transform: SimilarityTransform | None = None,
        max_points: int = 1_000_000,
        camera_images_txt: str | None = None,
        label: str = "BA/MVS",
    ) -> None:
        """Display an exported PLY while preserving large engineering coordinates."""

        points, colors, total_points = load_ply_preview(
            path,
            max_points=max_points,
        )
        if not len(points):
            raise ValueError("PLY 中没有可显示的点")
        if transform is not None:
            points = transform.apply(points)
        finite = np.isfinite(points).all(axis=1)
        indices = np.flatnonzero(finite)
        selected = points[indices]
        self._display_origin = np.median(selected, axis=0)
        cloud = pv.PolyData(selected - self._display_origin)
        cloud["rgb"] = colors[indices]
        self.plotter.clear()
        self.plotter.add_points(
            cloud,
            scalars="rgb",
            rgb=True,
            point_size=2,
            render_points_as_spheres=False,
            name="point_cloud",
        )
        camera_count = 0
        if camera_images_txt:
            centers, directions = _colmap_camera_poses(camera_images_txt)
            if transform is not None and len(centers):
                centers = transform.apply(centers)
                directions = directions @ transform.rotation.T
            if len(centers):
                camera_count = len(centers)
                local_centers = centers - self._display_origin
                extent = float(np.linalg.norm(np.ptp(selected, axis=0)))
                line_length = max(extent * 0.035, np.finfo(np.float64).eps)
                camera_cloud = pv.PolyData(local_centers)
                self.plotter.add_points(
                    camera_cloud,
                    color="#ff5a5f",
                    point_size=8,
                    render_points_as_spheres=True,
                    name="cameras",
                )
                line_points = np.empty((camera_count * 2, 3), dtype=np.float64)
                line_points[0::2] = local_centers
                line_points[1::2] = local_centers + directions * line_length
                lines = np.column_stack(
                    (
                        np.full(camera_count, 2, dtype=np.int64),
                        np.arange(0, camera_count * 2, 2, dtype=np.int64),
                        np.arange(1, camera_count * 2, 2, dtype=np.int64),
                    )
                ).ravel()
                direction_cloud = pv.PolyData(line_points, lines=lines)
                self.plotter.add_mesh(
                    direction_cloud,
                    color="#ff5a5f",
                    line_width=2,
                    name="camera_directions",
                )
        display_unit = "m" if unit == "m" else "model units"
        camera_text = f" · {camera_count} cameras" if camera_count else ""
        self.plotter.add_text(
            f"{label} · preview {len(selected):,}/{total_points:,} points"
            f"{camera_text} · {display_unit}\n"
            "Left: pick · Mouse: orbit/zoom",
            position="upper_left",
            color="white",
            font_size=10,
            name="hint",
        )
        self.plotter.show_axes()
        self._enable_picking()
        self.plotter.reset_camera()
        self.plotter.render()

    def load_model(
        self,
        mesh_path: str,
        texture_path: str,
        *,
        unit: str,
        transform: SimilarityTransform | None = None,
        label: str = "Textured model",
    ) -> None:
        """Display a COLMAP textured PLY mesh and its texture atlas."""

        self.load_models(
            [{"mesh": mesh_path, "texture": texture_path}],
            unit=unit,
            transform=transform,
            label=label,
        )

    def load_models(
        self,
        blocks: list[dict[str, object]],
        *,
        unit: str,
        transform: SimilarityTransform | None = None,
        label: str = "Textured model",
    ) -> None:
        """Display spatial mesh blocks with one independent atlas per block."""

        if not blocks:
            raise ValueError("多图集模型没有可显示的纹理块")
        loaded: list[tuple[pv.PolyData, object, np.ndarray]] = []
        finite_points: list[np.ndarray] = []
        total_vertices = 0
        total_faces = 0
        for block in blocks:
            mesh_path = str(block.get("mesh", ""))
            texture_path = str(block.get("texture", ""))
            mesh = pv.read(mesh_path).extract_surface().triangulate()
            if mesh.n_points == 0 or mesh.n_cells == 0:
                raise ValueError(f"模型纹理块没有有效三角网格：{mesh_path}")
            if mesh.active_texture_coordinates is None:
                coordinates = mesh.point_data.get("TCoords")
                if coordinates is None:
                    raise ValueError(f"模型纹理块缺少UV坐标：{mesh_path}")
                mesh.active_texture_coordinates = coordinates
            points = np.asarray(mesh.points, dtype=np.float64)
            if transform is not None:
                points = transform.apply(points)
            finite = np.isfinite(points).all(axis=1)
            if not np.any(finite):
                raise ValueError(f"模型纹理块顶点坐标无效：{mesh_path}")
            finite_points.append(points[finite])
            loaded.append((mesh, pv.read_texture(texture_path), points))
            total_vertices += int(mesh.n_points)
            total_faces += int(mesh.n_cells)
        self._display_origin = np.median(
            np.concatenate(finite_points, axis=0),
            axis=0,
        )
        self.plotter.clear()
        for index, (mesh, texture, points) in enumerate(loaded, 1):
            mesh.points = points - self._display_origin
            self.plotter.add_mesh(
                mesh,
                texture=texture,
                smooth_shading=True,
                show_edges=False,
                name=f"textured_model_{index:04d}",
            )
        display_unit = "m" if unit == "m" else "model units"
        self.plotter.add_text(
            f"{label} · {len(blocks)} atlases · {total_vertices:,} vertices · "
            f"{total_faces:,} faces · "
            f"{display_unit}\nLeft: pick · Mouse: orbit/zoom",
            position="upper_left",
            color="white",
            font_size=10,
            name="hint",
        )
        self.plotter.show_axes()
        self._enable_picking()
        self.plotter.reset_camera()
        self.plotter.render()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.plotter.close()
        super().closeEvent(event)


class TaskThread(QThread):
    """Run one callable without blocking Qt's GUI thread."""

    progress_changed = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, function: Callable[[Callable[[float, str], None]], Any]) -> None:
        super().__init__()
        self.function = function

    def run(self) -> None:
        def progress(value: float, text: str) -> None:
            self.progress_changed.emit(int(np.clip(value, 0, 1) * 100), str(text))

        try:
            result = self.function(progress)
        except Exception as exc:  # pragma: no cover - exercised by GUI integration
            self.failed.emit(str(exc), traceback.format_exc())
        else:
            self.succeeded.emit(result)
