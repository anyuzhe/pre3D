"""Native PySide6 + PyVista/VTK application for engineering point clouds."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .calibration import SimilarityTransform
from .colmap_pipeline import find_colmap
from .coordinate_systems import CoordinateReference
from .desktop_widgets import CloudView, ProcessTaskController, TaskThread
from .exporters import export_project
from .measurements import calculate
from .model_pipeline import find_osgconv
from .photo_selection import (
    PHOTO_EXTENSIONS,
    PhotoRecord,
    _orientation_display_info,
    discover_photos,
)
from .project_store import ProjectStore
from .runtime_paths import ensure_user_directories, resource_root
from .session import Measurement, ProjectSession

_logger = logging.getLogger(__name__)
_resource_root = resource_root()
_user_directories = ensure_user_directories()
_projects_root = _user_directories["projects"]
_outputs_root = _user_directories["outputs"]
_logs_root = _user_directories["logs"]

PHOTO_COLUMNS = [
    "预览",
    "状态",
    "照片",
    "尺寸",
    "清晰度",
    "曝光",
    "GPS",
    "焦距mm",
    "拍摄时间",
    "镜头",
    "提示",
]

RECONSTRUCTION_PRESETS: dict[str, dict[str, object]] = {
    "快速预览": {
        "feature_max_image_size": 2048,
        "max_image_size": 2048,
        "max_num_features": 2048,
        "geometric_consistency": False,
        "patch_match_filter": True,
        "patch_match_source_images": 8,
        "patch_match_iterations": 3,
        "mvs_reference_strategy": "all",
        "mvs_reference_ratio": 1.0,
        "spatial_block_target_images": 140,
    },
    "标准工程模式": {
        "feature_max_image_size": 3072,
        "max_image_size": 3072,
        "max_num_features": 4096,
        "geometric_consistency": True,
        "patch_match_filter": True,
        "patch_match_source_images": 12,
        "patch_match_iterations": 4,
        "mvs_reference_strategy": "all",
        "mvs_reference_ratio": 1.0,
        "spatial_block_target_images": 120,
    },
    "高精度模式": {
        "feature_max_image_size": 4096,
        "max_image_size": 4096,
        "max_num_features": 8192,
        "geometric_consistency": True,
        "patch_match_filter": True,
        "patch_match_source_images": 18,
        "patch_match_iterations": 5,
        "mvs_reference_strategy": "all",
        "mvs_reference_ratio": 1.0,
        "spatial_block_target_images": 90,
    },
}

PRESET_NAME_ALIASES = {
    "标准重建": "标准工程模式",
    "高精度重建": "高精度模式",
}

BUSINESS_STAGES = [
    ("photo_scan", "1", "照片检查"),
    ("ai_feature_extraction", "2", "提取照片特征"),
    ("ai_feature_matching", "3", "建立照片连接"),
    ("sparse_mapping", "4", "计算相机位置"),
    ("bundle_adjustment", "5", "优化空三结果"),
    ("image_undistortion", "6", "准备高精度照片"),
    ("patch_match", "7", "计算稠密深度"),
    ("stereo_fusion", "8", "生成稠密点云"),
    ("point_conditioning", "9", "点云去噪与法向修复"),
    ("surface_reconstruction", "10", "表面与三角网格"),
    ("mesh_repair", "11", "网格修补与简化"),
    ("texture_mapping", "12", "原图纹理与接缝融合"),
    ("model_export", "13", "模型格式导出"),
]

GUIDE = """
# 现场拍摄与成果等级

## 快速预览

只上传照片即可生成彩色点云、相机位置、相对深度和场景相对形状。成果处在任意尺度和任意坐标系，
只能用于浏览、展示、覆盖检查和粗略形态判断，不能输出带“米、厘米、毫米”的测量值。

## 尺寸测量

在不同位置布置 2～3 根刚性标尺，或使用已知边长的 ArUco/AprilTag 标志。一个已知距离可恢复统一
比例；多个分散尺度约束更容易发现粗差。统一尺度不能纠正长隧道、大边坡中的局部弯曲或漂移。

## 工程测量

建议至少 5～8 个控制点，分布在区域四周、中部、高处和低处；另留 2～4 个独立检查点，绝不能
参与拟合。数学上的最低数量是 3 个不共线三维点，但最低数量不等于工程可靠。

## 拍摄检查表

- 前后照片充分重叠，同一区域同时有正视和斜视；
- 不只沿一条直线平移，大区域形成回环；
- 固定焦距，不使用数码变焦，不频繁切换镜头；
- 尽量固定曝光和白平衡，保留原始 EXIF；
- 避免模糊、反光、阴影突变、动态人员和车辆；
- 保留原始高分辨率照片，工程成果使用“ALIKED → LightGlue → SfM → BA → 原图MVS”。

传统良好摄影测量中，点位误差常以约 1～3 倍GSD作为经验参考，但不是本软件对任何项目的
精度承诺；最终以独立检查点残差和现场验收为准。
"""


def _sparse_requires_stable_retry(result: dict[str, Any]) -> bool:
    """Retry only when sparse geometry is too weak for useful dense MVS."""

    gate = dict(result.get("quality_gate") or {})
    if str(gate.get("status", "review")) == "blocked":
        return True
    registered = int(result.get("registered_images", 0))
    registration_ratio = float(result.get("registration_ratio", 0.0))
    sparse_points = int(result.get("sparse_point_count", 0))
    error_value = result.get("mean_reprojection_error_px")
    reprojection_error = (
        float(error_value) if error_value is not None else None
    )
    return (
        registered < 3
        or registration_ratio < 0.8
        or sparse_points < max(100, registered * 25)
        or reprojection_error is None
        or reprojection_error > 3.0
    )


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item),
    )


def _table(headers: list[str], minimum_height: int = 150) -> QTableWidget:
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setAlternatingRowColors(True)
    widget.setMinimumHeight(minimum_height)
    header = widget.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(True)
    return widget


def _fill_table(widget: QTableWidget, rows: list[list[Any]]) -> None:
    widget.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column_index in range(widget.columnCount()):
            value = row[column_index] if column_index < len(row) else ""
            item = QTableWidgetItem("" if value is None else str(value))
            widget.setItem(row_index, column_index, item)
    widget.resizeRowsToContents()


def _read_only_text(lines: int = 8) -> QTextEdit:
    widget = QTextEdit()
    widget.setReadOnly(True)
    widget.setMinimumHeight(lines * 20)
    widget.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
    return widget


def _double_spin(
    minimum: float,
    maximum: float,
    value: float,
    decimals: int = 3,
    step: float = 1.0,
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    return widget


def _scroll_page(content: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(content)
    return area


class EngineeringMainWindow(QMainWindow):
    """Main native desktop window."""

    def __init__(self) -> None:
        super().__init__()
        self.session = ProjectSession()
        self.project_store: ProjectStore | None = None
        self.input_images: list[str] = []
        self.selected_images: list[str] = []
        self.source_root: str | None = None
        self.scale_points: list[np.ndarray] = []
        self.scale_pixels: list[tuple[float, float]] = []
        self.measurement_points: list[np.ndarray] = []
        self.measurement_pixels: list[tuple[float, float]] = []
        self.current_control_pick: dict[str, Any] | None = None
        self._active_task: TaskThread | None = None
        self._task_completion = None
        self._process_completion = None
        self._process_task_kind = ""
        self._colmap_target_stage = ""
        self._applying_reconstruction_preset = False
        self._advanced_only_widgets: list[QWidget] = []
        self._one_click_active = False
        self._one_click_retry_used = False
        self._one_click_wants_model = False
        self._result_view_mode = "pointcloud"
        self._interrupted_task_notice = ""
        settings_path = os.environ.get(
            "AI_PHOTOGRAMMETRY_SETTINGS_PATH",
            "",
        ).strip()
        self.settings = (
            QSettings(settings_path, QSettings.Format.IniFormat)
            if settings_path
            else QSettings(
                "AI Photogrammetry Engineering",
                "Point Cloud Workbench",
            )
        )
        self.process_task = ProcessTaskController(self)
        self.process_task.progress_changed.connect(self._task_progress)
        self.process_task.telemetry_changed.connect(self._gpu_telemetry)
        self.process_task.status_changed.connect(self._process_status)
        self.process_task.log_received.connect(
            lambda line: _logger.info("worker: %s", line)
        )
        self.process_task.succeeded.connect(self._process_succeeded)
        self.process_task.failed.connect(self._process_failed)
        self.process_task.cancelled.connect(self._process_cancelled)

        self.setWindowTitle("岩土影像三维重建工作台")
        self.resize(1680, 980)
        self.setMinimumSize(1200, 760)
        self.setAcceptDrops(True)
        self._build_window()
        self._load_recent_projects()
        self._update_project_status()

    def _build_window(self) -> None:
        self._build_menu()
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        self.project_status = QLabel()
        self.project_status.setObjectName("projectStatus")
        self.project_status.setWordWrap(True)
        outer.addWidget(self.project_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        self.project_tab = _scroll_page(self._build_project_tab())
        self.reconstruction_tab = _scroll_page(self._build_reconstruction_tab())
        self.colmap_tab = _scroll_page(self._build_colmap_tab())
        self.dense_tab = _scroll_page(self._build_dense_tab())

        control_page = QWidget()
        control_layout = QVBoxLayout(control_page)
        self.control_subtabs = QTabWidget()
        self.scale_tab = _scroll_page(self._build_scale_tab())
        self.control_tab = _scroll_page(self._build_control_tab())
        self.measurement_tab = _scroll_page(self._build_measurement_tab())
        self.control_subtabs.addTab(self.scale_tab, "标尺")
        self.control_subtabs.addTab(self.control_tab, "控制点与检查点")
        self.control_subtabs.addTab(self.measurement_tab, "测量")
        control_layout.addWidget(self.control_subtabs)
        self.control_accuracy_tab = control_page

        self.export_tab = _scroll_page(self._build_export_tab())
        for title, tab in (
            ("1 · 项目", self.project_tab),
            ("2 · 照片", self.reconstruction_tab),
            ("3 · 空三", self.colmap_tab),
            ("4 · 稠密重建", self.dense_tab),
            ("5 · 控制与精度", self.control_accuracy_tab),
            ("6 · 成果", self.export_tab),
        ):
            self.tabs.addTab(tab, title)
        self._set_advanced_ui(False)

        self.cloud_view = CloudView()
        self.cloud_view.setMinimumWidth(520)
        self.cloud_view.point_picked.connect(self._on_cloud_point)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.cloud_view)
        splitter.setSizes([790, 890])
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        outer.addWidget(splitter, 1)

        progress_row = QHBoxLayout()
        self.progress_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.gpu_label = QLabel("GPU：待命")
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.process_task.cancel)
        progress_row.addWidget(self.progress_label, 1)
        progress_row.addWidget(self.progress_bar, 2)
        progress_row.addWidget(self.gpu_label)
        progress_row.addWidget(self.cancel_button)
        outer.addLayout(progress_row)
        self.setCentralWidget(central)
        self.brand_mark = QLabel("岩创科技")
        self.brand_mark.setObjectName("brandMark")
        self.statusBar().addPermanentWidget(self.brand_mark)
        self.statusBar().showMessage("本机桌面模式，不启动 Web 服务")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        new_project = QAction("新建工程…", self)
        new_project.setShortcut("Ctrl+N")
        new_project.triggered.connect(self._new_project)
        file_menu.addAction(new_project)
        open_project = QAction("打开工程…", self)
        open_project.setShortcut("Ctrl+O")
        open_project.triggered.connect(self._open_project)
        file_menu.addAction(open_project)
        save_project = QAction("保存工程", self)
        save_project.setShortcut("Ctrl+S")
        save_project.triggered.connect(self._save_project)
        file_menu.addAction(save_project)
        close_project = QAction("关闭工程", self)
        close_project.triggered.connect(self._close_project)
        file_menu.addAction(close_project)
        file_menu.addSeparator()
        add_photos = QAction("添加照片…", self)
        add_photos.triggered.connect(self._choose_images)
        file_menu.addAction(add_photos)
        open_outputs = QAction("打开成果目录", self)
        open_outputs.triggered.connect(lambda: self._open_path(_outputs_root))
        file_menu.addAction(open_outputs)
        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("帮助")
        about = QAction("关于", self)
        about.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "关于",
                "AI 摄影测量工程点云工作台\n"
                "PySide6 + PyVista/VTK 原生桌面版\n\n"
                "ALIKED / LightGlue → GLOMAP / COLMAP → BA → 原图 MVS",
            )
        )
        help_menu.addAction(about)

        advanced_menu = self.menuBar().addMenu("高级功能")
        self.advanced_ui_action = QAction(
            "显示专业页面与参数",
            self,
            checkable=True,
        )
        self.advanced_ui_action.toggled.connect(self._set_advanced_ui)
        advanced_menu.addAction(self.advanced_ui_action)

    def _set_advanced_ui(self, enabled: bool) -> None:
        if hasattr(self, "tabs"):
            for index in (2, 3, 4):
                self.tabs.setTabVisible(index, enabled)
            if enabled:
                titles = (
                    "1 · 项目",
                    "2 · 照片",
                    "3 · 空三",
                    "4 · 稠密重建",
                    "5 · 控制与精度",
                    "6 · 成果",
                )
            else:
                titles = (
                    "1 · 项目",
                    "2 · 一键重建",
                    "3 · 空三",
                    "4 · 稠密重建",
                    "5 · 控制与精度",
                    "3 · 成果",
                )
                if self.tabs.currentIndex() in (2, 3, 4):
                    self.tabs.setCurrentWidget(self.reconstruction_tab)
            for index, title in enumerate(titles):
                self.tabs.setTabText(index, title)
        for widget in self._advanced_only_widgets:
            widget.setVisible(enabled)

    def _build_project_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("岩体、边坡、隧道及工程现场点云重建")
        title.setObjectName("homeTitle")
        subtitle = QLabel(
            "简单流程：创建项目 → 导入照片 → 选择精度 → 一键生成点云"
        )
        subtitle.setObjectName("homeSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        current_group = QGroupBox("当前工程")
        current_layout = QHBoxLayout(current_group)
        self.current_project_details = QLabel("尚未打开工程")
        self.current_project_details.setWordWrap(True)
        save_current = QPushButton("保存")
        save_current.clicked.connect(self._save_project)
        close_current = QPushButton("关闭")
        close_current.clicked.connect(self._close_project)
        current_layout.addWidget(self.current_project_details, 1)
        current_layout.addWidget(save_current)
        current_layout.addWidget(close_current)
        layout.addWidget(current_group)

        create_group = QGroupBox("新建重建任务")
        create = QGridLayout(create_group)
        self.home_project_name = QLineEdit()
        self.home_project_name.setPlaceholderText("例如：北侧边坡 2026-07")
        self.home_project_root = QLineEdit(
            str(_projects_root)
        )
        browse_root = QPushButton("选择保存位置…")
        browse_root.clicked.connect(self._choose_new_project_root)
        self.home_project_type = QComboBox()
        self.home_project_type.addItems(
            ["近景岩体重建", "普通照片三维重建", "无人机航测重建"]
        )
        self.home_coordinate_system = QComboBox()
        self.home_coordinate_system.setEditable(True)
        self.home_coordinate_system.addItems(
            [
                "本地模型坐标（后续标定）",
                "现场独立坐标系",
                "导入控制点后确定",
            ]
        )
        self.home_precision = QComboBox()
        self.home_precision.addItems(list(RECONSTRUCTION_PRESETS))
        self.home_precision.setCurrentText("标准工程模式")
        create_button = QPushButton("创建项目，下一步添加照片")
        create_button.setObjectName("primaryButton")
        create_button.clicked.connect(self._create_project_from_home)
        create.addWidget(QLabel("项目名称"), 0, 0)
        create.addWidget(self.home_project_name, 0, 1, 1, 3)
        create.addWidget(QLabel("保存位置"), 1, 0)
        create.addWidget(self.home_project_root, 1, 1, 1, 2)
        create.addWidget(browse_root, 1, 3)
        create.addWidget(QLabel("成果精度"), 2, 0)
        create.addWidget(self.home_precision, 2, 1)
        create.addWidget(create_button, 3, 0, 1, 4)
        layout.addWidget(create_group)

        recent_group = QGroupBox("最近项目")
        recent_layout = QVBoxLayout(recent_group)
        self.recent_projects_table = _table(
            ["项目", "位置", "最后更新"],
            180,
        )
        self.recent_projects_table.doubleClicked.connect(
            lambda _index: self._open_selected_recent()
        )
        recent_actions = QHBoxLayout()
        continue_button = QPushButton("继续处理")
        continue_button.setObjectName("primaryButton")
        continue_button.clicked.connect(self._open_selected_recent)
        open_button = QPushButton("打开已有项目…")
        open_button.clicked.connect(self._open_project)
        recent_actions.addWidget(continue_button)
        recent_actions.addWidget(open_button)
        recent_layout.addWidget(self.recent_projects_table)
        recent_layout.addLayout(recent_actions)
        layout.addWidget(recent_group)
        layout.addStretch(1)
        return tab

    def _build_reconstruction_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        source_group = QGroupBox("项目与照片")
        form = QGridLayout(source_group)
        self.project_name = QLineEdit("未命名项目")
        self.intended_mode = QComboBox()
        self.intended_mode.addItems(list(RECONSTRUCTION_PRESETS))
        self.intended_mode.setCurrentText("标准工程模式")
        self.intended_mode.currentTextChanged.connect(
            self._apply_beginner_precision
        )
        self.choose_images_button = QPushButton("添加照片…")
        self.choose_images_button.clicked.connect(self._choose_images)
        self.choose_folder_button = QPushButton("添加文件夹…")
        self.choose_folder_button.clicked.connect(self._choose_image_folder)
        self.clear_sources_button = QPushButton("清空")
        self.clear_sources_button.clicked.connect(self._clear_sources)
        self.source_summary = QLabel("尚未选择照片")
        self.source_summary.setWordWrap(True)
        form.addWidget(QLabel("项目名称"), 0, 0)
        form.addWidget(self.project_name, 0, 1, 1, 3)
        form.addWidget(QLabel("成果精度"), 1, 0)
        form.addWidget(self.intended_mode, 1, 1)
        form.addWidget(self.choose_images_button, 1, 2)
        form.addWidget(self.choose_folder_button, 1, 3)
        form.addWidget(self.clear_sources_button, 1, 4)
        form.addWidget(self.source_summary, 2, 0, 1, 5)
        layout.addWidget(source_group)

        drop_hint = QLabel(
            "可将照片或照片文件夹直接拖入窗口。原始文件不会被修改。"
        )
        drop_hint.setObjectName("dropZone")
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(drop_hint)

        stats_group = QGroupBox("照片概况")
        stats = QGridLayout(stats_group)
        self.photo_stat_total = QLabel("0")
        self.photo_stat_valid = QLabel("0")
        self.photo_stat_blur = QLabel("0")
        self.photo_stat_duplicate = QLabel("0")
        self.photo_stat_gps = QLabel("0")
        self.photo_stat_camera = QLabel("—")
        for index, (title, widget) in enumerate(
            (
                ("照片数量", self.photo_stat_total),
                ("有效照片", self.photo_stat_valid),
                ("模糊照片", self.photo_stat_blur),
                ("疑似重复", self.photo_stat_duplicate),
                ("GPS / RTK", self.photo_stat_gps),
                ("相机型号", self.photo_stat_camera),
            )
        ):
            column = index % 3
            row = (index // 3) * 2
            title_label = QLabel(title)
            title_label.setObjectName("statCaption")
            widget.setObjectName("statValue")
            stats.addWidget(title_label, row, column)
            stats.addWidget(widget, row + 1, column)
        layout.addWidget(stats_group)

        output_group = QGroupBox("选择需要的成果")
        output_layout = QGridLayout(output_group)
        self.output_dense_cloud = QCheckBox("稠密点云 / 深度图")
        self.output_dense_cloud.setChecked(True)
        self.output_dense_cloud.setEnabled(False)
        self.output_textured_model = QCheckBox("纹理三维模型（可选，耗时更长）")
        self.output_model_obj = QCheckBox("OBJ")
        self.output_model_obj.setChecked(True)
        self.output_model_fbx = QCheckBox("FBX")
        self.output_model_fbx.setChecked(True)
        self.output_model_gltf = QCheckBox("glTF / GLB")
        self.output_model_gltf.setChecked(True)
        self.output_model_osgb = QCheckBox("OSGB")
        osgconv = find_osgconv()
        self.output_model_osgb.setEnabled(bool(osgconv))
        self.output_model_osgb.setChecked(bool(osgconv))
        self.output_model_osgb.setToolTip(
            "已检测到OpenSceneGraph osgconv，可输出真实OSGB。"
            if osgconv
            else "本机未安装OpenSceneGraph osgconv，暂不可选择OSGB。"
        )
        self.output_textured_model.toggled.connect(self._toggle_model_outputs)
        output_layout.addWidget(self.output_dense_cloud, 0, 0)
        output_layout.addWidget(self.output_textured_model, 0, 1, 1, 3)
        output_layout.addWidget(QLabel("模型格式"), 1, 0)
        output_layout.addWidget(self.output_model_obj, 1, 1)
        output_layout.addWidget(self.output_model_fbx, 1, 2)
        output_layout.addWidget(self.output_model_gltf, 1, 3)
        output_layout.addWidget(self.output_model_osgb, 1, 4)
        model_hint = QLabel(
            "生成模型时会继续执行：点云去噪与法向修复 → 表面重建 → 网格修补/简化 → "
            "UV展开 → 原始照片纹理投影 → 多照片色彩与接缝融合 → 纹理图集。"
        )
        model_hint.setWordWrap(True)
        output_layout.addWidget(model_hint, 2, 0, 1, 5)
        layout.addWidget(output_group)
        self._toggle_model_outputs(False)

        one_click_group = QGroupBox("一键生成三维成果")
        one_click_layout = QVBoxLayout(one_click_group)
        one_click_description = QLabel(
            "软件会自动完成照片检查、照片匹配、相机位置计算、空三优化、"
            "原图稠密重建和点云融合；勾选模型后还会自动完成网格与纹理。"
            "三档模式都使用全部注册照片生成深度图。"
        )
        one_click_description.setWordWrap(True)
        self.one_click_status = QLabel(
            "添加照片并选择成果精度后，点击下方按钮即可。"
        )
        self.one_click_status.setObjectName("infoLabel")
        self.one_click_status.setWordWrap(True)
        self.one_click_button = QPushButton("开始一键处理")
        self.one_click_button.setObjectName("primaryButton")
        self.one_click_button.clicked.connect(self._run_one_click)
        one_click_layout.addWidget(one_click_description)
        one_click_layout.addWidget(self.one_click_status)
        one_click_layout.addWidget(self.one_click_button)
        layout.addWidget(one_click_group)

        selection_group = QGroupBox("重复检测与关键帧")
        selection = QGridLayout(selection_group)
        self.max_keyframes = QSpinBox()
        self.max_keyframes.setRange(0, 10_000)
        self.max_keyframes.setValue(0)
        self.auto_segment = QCheckBox("自动排除拍摄断层，保留最长连续段")
        self.auto_segment.setChecked(False)
        self.max_keyframes.setSpecialValueText("全部有效照片")
        self.photo_selection_policy = QComboBox()
        self.photo_selection_policy.addItems(
            [
                "自动排除不合格照片",
                "仅排除损坏与完全重复",
                "全部有效照片保留",
            ]
        )
        self.include_near_duplicates = QCheckBox("保留近重复照片")
        self.scan_photos_button = QPushButton("扫描质量、重复项并选择关键帧")
        self.scan_photos_button.clicked.connect(self._scan_photos)
        selection.addWidget(QLabel("最多关键帧"), 0, 0)
        selection.addWidget(self.max_keyframes, 0, 1)
        selection.addWidget(self.photo_selection_policy, 0, 2)
        selection.addWidget(self.include_near_duplicates, 0, 3)
        selection.addWidget(self.auto_segment, 1, 0, 1, 3)
        selection.addWidget(self.scan_photos_button, 1, 3)
        layout.addWidget(selection_group)

        workflow = QLabel(
            "绿色为质量合格，黄色表示需复核，红色表示损坏或严重异常，灰色表示已排除。"
            "建议在耗时空三前处理模糊、重复、镜头切换和拍摄断层。"
        )
        workflow.setObjectName("warningLabel")
        workflow.setWordWrap(True)
        layout.addWidget(workflow)
        next_button = QPushButton("进入空三设置")
        next_button.setObjectName("primaryButton")
        next_button.clicked.connect(lambda: self.tabs.setCurrentWidget(self.colmap_tab))
        layout.addWidget(next_button)

        self.photo_table = _table(PHOTO_COLUMNS, 190)
        layout.addWidget(self.photo_table)
        self.quality_text = _read_only_text(9)
        self.quality_text.setPlaceholderText("照片质量、重复检测和连续性诊断")
        layout.addWidget(self.quality_text)
        self._advanced_only_widgets.extend(
            [
                selection_group,
                workflow,
                next_button,
                self.photo_table,
                self.quality_text,
            ]
        )
        return tab

    def _build_scale_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        instruction = QLabel(
            "在右侧稠密点云中依次左键拾取已知距离的两个端点。"
            "建议在场景不同位置加入多根标尺，以便发现异常约束。"
        )
        instruction.setObjectName("warningLabel")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self.scale_selected_table = _table(["序号", "模型 X", "模型 Y", "模型 Z"], 90)
        layout.addWidget(self.scale_selected_table)
        self.scale_distance_label = QLabel("请依次选择两个端点；也可以在右侧三维视图拾点。")
        layout.addWidget(self.scale_distance_label)

        row = QHBoxLayout()
        self.scale_label = QLineEdit()
        self.scale_label.setPlaceholderText("标尺名称")
        self.actual_distance = _double_spin(0.000001, 1_000_000.0, 2.0, 6, 0.1)
        add_constraint = QPushButton("添加标尺约束")
        add_constraint.clicked.connect(self._add_scale_constraint)
        clear_selection = QPushButton("清空当前端点")
        clear_selection.clicked.connect(self._clear_scale_selection)
        row.addWidget(self.scale_label)
        row.addWidget(QLabel("实际距离（米）"))
        row.addWidget(self.actual_distance)
        row.addWidget(add_constraint)
        row.addWidget(clear_selection)
        layout.addLayout(row)

        self.distance_table = _table(
            ["名称", "来源", "模型距离", "实际距离 m", "单项比例"],
            125,
        )
        layout.addWidget(self.distance_table)
        action_row = QHBoxLayout()
        calibrate = QPushButton("按全部标尺恢复米制尺度")
        calibrate.setObjectName("primaryButton")
        calibrate.clicked.connect(self._calibrate_scale)
        clear_constraints = QPushButton("清空全部标尺")
        clear_constraints.clicked.connect(self._clear_distance_constraints)
        action_row.addWidget(calibrate)
        action_row.addWidget(clear_constraints)
        layout.addLayout(action_row)

        self.scale_report = _read_only_text(7)
        layout.addWidget(self.scale_report)
        return tab

    def _build_control_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        instruction = QLabel(
            "先在右侧稠密点云中拾取控制点中心，再输入对应工程坐标。"
            "控制点用于拟合；检查点只用于独立精度验证。"
        )
        instruction.setObjectName("warningLabel")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self.control_pick_text = QLabel("尚未选择模型点")
        self.control_pick_text.setWordWrap(True)
        layout.addWidget(self.control_pick_text)

        form_group = QGroupBox("坐标观测")
        form = QGridLayout(form_group)
        self.control_id = QLineEdit()
        self.control_id.setPlaceholderText("例如 GCP01")
        self.control_role = QComboBox()
        self.control_role.addItems(["控制点", "检查点"])
        self.coordinate_input_mode = QComboBox()
        self.coordinate_input_mode.addItems(
            ["工程坐标 XYZ", "WGS84 经度/纬度/椭球高"]
        )
        self.control_target_crs = QLineEdit()
        self.control_target_crs.setPlaceholderText(
            "如 EPSG:4547；WGS84输入时留空使用Local ENU"
        )
        self.control_sigma = _double_spin(0.0, 1000.0, 0.0, 4, 0.01)
        self.control_sigma.setSpecialValueText("未提供")
        self.control_ransac_threshold = _double_spin(0.001, 1000.0, 0.10, 3, 0.01)
        self.target_x = _double_spin(-1e9, 1e9, 0.0, 6, 1.0)
        self.target_y = _double_spin(-1e9, 1e9, 0.0, 6, 1.0)
        self.target_z = _double_spin(-1e9, 1e9, 0.0, 6, 1.0)
        add_button = QPushButton("添加坐标观测")
        add_button.clicked.connect(self._add_coordinate)
        import_button = QPushButton("导入 CSV…")
        import_button.clicked.connect(self._import_coordinate_csv)
        form.addWidget(QLabel("编号"), 0, 0)
        form.addWidget(self.control_id, 0, 1)
        form.addWidget(QLabel("类型"), 0, 2)
        form.addWidget(self.control_role, 0, 3)
        for column, (name, widget) in enumerate(
            (("X", self.target_x), ("Y", self.target_y), ("Z", self.target_z))
        ):
            form.addWidget(QLabel(name), 1, column * 2)
            form.addWidget(widget, 1, column * 2 + 1)
        form.addWidget(QLabel("输入坐标"), 2, 0)
        form.addWidget(self.coordinate_input_mode, 2, 1)
        form.addWidget(QLabel("目标CRS"), 2, 2)
        form.addWidget(self.control_target_crs, 2, 3)
        form.addWidget(QLabel("坐标σ"), 2, 4)
        form.addWidget(self.control_sigma, 2, 5)
        form.addWidget(add_button, 3, 0, 1, 3)
        form.addWidget(import_button, 3, 3, 1, 3)
        layout.addWidget(form_group)

        self.coordinate_table = _table(
            ["编号", "类型", "照片", "模型 X", "模型 Y", "模型 Z", "目标 X", "目标 Y", "目标 Z"],
            150,
        )
        layout.addWidget(self.coordinate_table)
        row = QHBoxLayout()
        calibrate = QPushButton("拟合工程坐标并计算检查点误差")
        calibrate.setObjectName("primaryButton")
        calibrate.clicked.connect(self._calibrate_engineering)
        clear = QPushButton("清空坐标观测")
        clear.clicked.connect(self._clear_coordinates)
        row.addWidget(QLabel("异常点阈值（目标坐标单位）"))
        row.addWidget(self.control_ransac_threshold)
        row.addWidget(calibrate)
        row.addWidget(clear)
        layout.addLayout(row)
        self.engineering_report = _read_only_text(10)
        layout.addWidget(self.engineering_report)
        return tab

    def _build_measurement_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        gate = QLabel(
            "所有测量均受真实尺度门禁约束：快速预览点云不能输出米、平方米或立方米。"
        )
        gate.setObjectName("warningLabel")
        gate.setWordWrap(True)
        layout.addWidget(gate)
        instruction = QLabel("按测量顺序在右侧三维点云中连续拾点。")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self.measurement_selected_table = _table(["序号", "X", "Y", "Z"], 110)
        layout.addWidget(self.measurement_selected_table)

        controls = QGridLayout()
        self.measurement_kind = QComboBox()
        self.measurement_kind.addItems(
            ["直线距离", "折线长度", "多边形面积", "凸包体积（近似）"]
        )
        self.measurement_label = QLineEdit()
        self.measurement_label.setPlaceholderText("测量名称")
        self.first_plane_count = QSpinBox()
        self.first_plane_count.setRange(3, 10_000)
        self.first_plane_count.setValue(3)
        calculate_button = QPushButton("计算并记录")
        calculate_button.setObjectName("primaryButton")
        calculate_button.clicked.connect(self._calculate_measurement)
        clear_points = QPushButton("清空当前选点")
        clear_points.clicked.connect(self._clear_measurement_points)
        controls.addWidget(QLabel("类型"), 0, 0)
        controls.addWidget(self.measurement_kind, 0, 1)
        controls.addWidget(QLabel("名称"), 0, 2)
        controls.addWidget(self.measurement_label, 0, 3)
        controls.addWidget(QLabel("第一平面点数"), 1, 0)
        controls.addWidget(self.first_plane_count, 1, 1)
        controls.addWidget(calculate_button, 1, 2)
        controls.addWidget(clear_points, 1, 3)
        layout.addLayout(controls)
        self.measurement_result = QLabel("尚未计算")
        self.measurement_result.setObjectName("resultLabel")
        layout.addWidget(self.measurement_result)
        self.measurement_table = _table(["名称", "类型", "数值", "单位", "点数"], 130)
        layout.addWidget(self.measurement_table)
        clear_results = QPushButton("清空测量记录")
        clear_results.clicked.connect(self._clear_measurements)
        layout.addWidget(clear_results)
        return tab

    def _build_legacy_colmap_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        explanation = QLabel(
            "默认工程管线：原始高分辨率照片 → ALIKED AI特征 → LightGlue匹配 → "
            "GLOMAP/COLMAP几何SfM → Bundle Adjustment → CUDA PatchMatch稠密点云。"
            "AI只增强最容易失败的特征提取与照片匹配，最终相机、尺度一致性和点云"
            "都由多视几何与原始高分辨率照片计算。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.colmap_path = QLineEdit(find_colmap() or "")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._choose_colmap)
        path_layout.addWidget(self.colmap_path)
        path_layout.addWidget(browse)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.photogrammetry_output_root = QLineEdit()
        self.photogrammetry_output_root.setPlaceholderText("默认：工程目录/colmap")
        output_browse = QPushButton("选择…")
        output_browse.clicked.connect(self._choose_photogrammetry_output)
        output_layout.addWidget(self.photogrammetry_output_root)
        output_layout.addWidget(output_browse)
        self.feature_method = QComboBox()
        self.feature_method.addItems(
            [
                "ALIKED‑N16Rot＋LightGlue（推荐）",
                "SIFT＋LightGlue（兼容备选）",
            ]
        )
        self.matcher = QComboBox()
        self.matcher.addItems(
            [
                "自动选择（≤120张全连接，较大项目顺序匹配）",
                "全连接匹配（无序照片/中小项目）",
                "顺序匹配（隧道/沿路线连续拍摄）",
            ]
        )
        self.sfm_mapper = QComboBox()
        self.sfm_mapper.addItems(
            [
                "GLOMAP 全局SfM（推荐）",
                "COLMAP 增量SfM（兼容备选）",
            ]
        )
        self.camera_model = QComboBox()
        self.camera_model.addItems(
            ["SIMPLE_RADIAL", "RADIAL", "OPENCV", "PINHOLE", "SIMPLE_PINHOLE"]
        )
        self.single_camera = QCheckBox("同一相机与固定镜头：共享一组内参")
        self.single_camera.setChecked(True)
        self.feature_size = QSpinBox()
        self.feature_size.setRange(1024, 16_000)
        self.feature_size.setValue(4096)
        self.mvs_size = QSpinBox()
        self.mvs_size.setRange(512, 16_000)
        self.mvs_size.setValue(4096)
        self.max_features = QSpinBox()
        self.max_features.setRange(1024, 32_768)
        self.max_features.setValue(4096)
        self.sequential_overlap = QSpinBox()
        self.sequential_overlap.setRange(2, 100)
        self.sequential_overlap.setValue(20)
        self.colmap_gpu = QCheckBox("使用 CUDA")
        self.colmap_gpu.setChecked(True)
        self.colmap_resume = QCheckBox("从已完成阶段继续")
        self.colmap_resume.setChecked(True)
        form.addRow("COLMAP", path_row)
        form.addRow("摄影测量工作目录", output_row)
        form.addRow("AI特征与匹配", self.feature_method)
        form.addRow("匹配模式", self.matcher)
        form.addRow("SfM求解器", self.sfm_mapper)
        form.addRow("相机模型", self.camera_model)
        form.addRow("", self.single_camera)
        form.addRow("AI特征最大边长", self.feature_size)
        form.addRow("MVS 最大图像尺寸", self.mvs_size)
        form.addRow("每图最大特征数", self.max_features)
        form.addRow("顺序匹配前后窗口", self.sequential_overlap)
        form.addRow("", self.colmap_gpu)
        form.addRow("", self.colmap_resume)
        layout.addLayout(form)
        self.run_colmap_button = QPushButton("运行 AI特征摄影测量 / 原图 MVS")
        self.run_colmap_button.setObjectName("primaryButton")
        self.run_colmap_button.clicked.connect(self._run_colmap)
        layout.addWidget(self.run_colmap_button)
        self.colmap_result = _read_only_text(18)
        layout.addWidget(self.colmap_result)
        self.colmap_output = QLineEdit()
        self.colmap_output.setReadOnly(True)
        layout.addWidget(self.colmap_output)
        open_button = QPushButton("打开优化成果所在目录")
        open_button.clicked.connect(lambda: self._open_path(Path(self.colmap_output.text()).parent))
        layout.addWidget(open_button)
        layout.addStretch(1)
        return tab

    def _build_colmap_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        explanation = QLabel(
            "先完成照片连接、相机位置和空三优化。软件会在这里暂停，"
            "让你检查注册率、未注册照片、重投影误差、相机位置和稀疏点云；"
            "确认几何正确后再进入稠密重建。"
        )
        explanation.setObjectName("warningLabel")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.settings_tabs = QTabWidget()
        simple = QWidget()
        simple_form = QFormLayout(simple)
        self.reconstruction_preset = QComboBox()
        self.reconstruction_preset.addItems(
            [*RECONSTRUCTION_PRESETS, "自定义参数"]
        )
        self.reconstruction_preset.setCurrentText("标准工程模式")
        self.reconstruction_preset.currentTextChanged.connect(
            self._apply_reconstruction_preset
        )
        preset_description = QLabel(
            "快速预览：2048 px，全部注册照片，几何一致性关闭，3 次迭代。\n"
            "标准工程：3072 px，全部注册照片，几何一致性开启，4 次迭代（默认）。\n"
            "高精度：4096 px，全部注册照片，几何一致性开启，5 次迭代。\n"
            "切换到工程/专家模式可修改全部参数；修改后自动标记为“自定义参数”。"
        )
        preset_description.setWordWrap(True)
        simple_form.addRow("重建精度", self.reconstruction_preset)
        simple_form.addRow("", preset_description)

        engineering = QWidget()
        engineering_form = QFormLayout(engineering)
        self.feature_method = QComboBox()
        self.feature_method.addItems(
            [
                "自动 / ALIKED‑N16Rot＋LightGlue（推荐）",
                "SIFT＋LightGlue（兼容备选）",
            ]
        )
        self.matcher = QComboBox()
        self.matcher.addItems(
            [
                "自动选择",
                "全局匹配（无序/中小项目）",
                "相邻照片（隧道/连续拍摄）",
            ]
        )
        self.sfm_mapper = QComboBox()
        self.sfm_mapper.addItems(
            [
                "自动 / GLOMAP 全局SfM（推荐）",
                "COLMAP 增量SfM（兼容备选）",
            ]
        )
        self.point_quality = QComboBox()
        self.point_quality.addItems(["低", "中", "高", "极高"])
        self.point_quality.setCurrentText("中")
        self.point_quality.currentTextChanged.connect(self._apply_point_quality)
        self.mvs_reference_strategy = QComboBox()
        self.mvs_reference_strategy.addItem(
            "全部注册照片生成深度图（固定100%）"
        )
        self.mvs_reference_ratio = QSpinBox()
        self.mvs_reference_ratio.setRange(100, 100)
        self.mvs_reference_ratio.setSuffix("%")
        self.mvs_reference_ratio.setValue(100)
        self.camera_model = QComboBox()
        self.camera_model.addItems(
            ["SIMPLE_RADIAL", "RADIAL", "OPENCV", "PINHOLE", "SIMPLE_PINHOLE"]
        )
        self.single_camera = QCheckBox("同一相机、同一镜头且未变焦：共享内参")
        self.single_camera.setChecked(True)
        self.geometric_consistency = QCheckBox("启用稠密深度几何一致性")
        self.geometric_consistency.setChecked(True)
        self.geometric_consistency.toggled.connect(
            lambda _checked: self._refresh_dense_quality_summary()
        )
        self.patch_match_filter = QCheckBox("启用 PatchMatch 深度过滤")
        self.patch_match_filter.setChecked(True)
        self.patch_match_filter.toggled.connect(
            lambda _checked: self._refresh_dense_quality_summary()
        )
        self.generate_quality_report = QCheckBox("生成空三与成果质量报告")
        self.generate_quality_report.setChecked(True)
        self.spatial_blocking = QCheckBox(
            "超过阈值时自动使用Core/Halo空间分块（推荐）"
        )
        self.spatial_blocking.setChecked(True)
        engineering_form.addRow("特征算法", self.feature_method)
        engineering_form.addRow("匹配范围", self.matcher)
        engineering_form.addRow("SfM 模式", self.sfm_mapper)
        engineering_form.addRow("点云质量", self.point_quality)
        engineering_form.addRow("MVS参考帧", self.mvs_reference_strategy)
        engineering_form.addRow("相机模型", self.camera_model)
        engineering_form.addRow("", self.single_camera)
        engineering_form.addRow("", self.geometric_consistency)
        engineering_form.addRow("", self.patch_match_filter)
        engineering_form.addRow("", self.generate_quality_report)
        engineering_form.addRow("", self.spatial_blocking)

        expert = QWidget()
        expert_form = QFormLayout(expert)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.colmap_path = QLineEdit(find_colmap() or "")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._choose_colmap)
        path_layout.addWidget(self.colmap_path)
        path_layout.addWidget(browse)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.photogrammetry_output_root = QLineEdit()
        self.photogrammetry_output_root.setPlaceholderText("默认：工程目录/colmap")
        output_browse = QPushButton("选择…")
        output_browse.clicked.connect(self._choose_photogrammetry_output)
        output_layout.addWidget(self.photogrammetry_output_root)
        output_layout.addWidget(output_browse)
        self.feature_size = QSpinBox()
        self.feature_size.setRange(1024, 16_000)
        self.mvs_size = QSpinBox()
        self.mvs_size.setRange(512, 16_000)
        self.max_features = QSpinBox()
        self.max_features.setRange(1024, 32_768)
        self.sequential_overlap = QSpinBox()
        self.sequential_overlap.setRange(2, 100)
        self.sequential_overlap.setValue(20)
        self.min_num_inliers = QSpinBox()
        self.min_num_inliers.setRange(8, 500)
        self.min_num_inliers.setValue(20)
        self.ransac_max_error = _double_spin(0.1, 50.0, 4.0, 2, 0.5)
        self.fusion_min_views = QSpinBox()
        self.fusion_min_views.setRange(1, 20)
        self.fusion_min_views.setValue(2)
        self.patch_match_source_images = QSpinBox()
        self.patch_match_source_images.setRange(1, 100)
        self.patch_match_source_images.setValue(12)
        self.patch_match_iterations = QSpinBox()
        self.patch_match_iterations.setRange(1, 20)
        self.patch_match_iterations.setValue(4)
        self.spatial_block_threshold = QSpinBox()
        self.spatial_block_threshold.setRange(20, 5000)
        self.spatial_block_threshold.setValue(180)
        self.spatial_block_target_images = QSpinBox()
        self.spatial_block_target_images.setRange(8, 500)
        self.spatial_block_target_images.setValue(120)
        self.spatial_block_halo_ratio = _double_spin(0.0, 1.0, 0.20, 2, 0.05)
        self.colmap_gpu = QCheckBox("使用 CUDA")
        self.colmap_gpu.setChecked(True)
        self.colmap_resume = QCheckBox("从已完成阶段继续")
        self.colmap_resume.setChecked(True)
        expert_form.addRow("COLMAP", path_row)
        expert_form.addRow("摄影测量工作目录", output_row)
        expert_form.addRow("特征最大边长", self.feature_size)
        expert_form.addRow("MVS 最大图像尺寸", self.mvs_size)
        expert_form.addRow("每图最大特征数", self.max_features)
        expert_form.addRow("顺序匹配前后窗口", self.sequential_overlap)
        expert_form.addRow("两视图最少内点", self.min_num_inliers)
        expert_form.addRow("RANSAC 最大误差（px）", self.ransac_max_error)
        expert_form.addRow("PatchMatch 源照片数", self.patch_match_source_images)
        expert_form.addRow("PatchMatch 迭代次数", self.patch_match_iterations)
        expert_form.addRow("空间分块启用阈值（照片）", self.spatial_block_threshold)
        expert_form.addRow("每块目标照片数", self.spatial_block_target_images)
        expert_form.addRow("Core外围Halo比例", self.spatial_block_halo_ratio)
        expert_form.addRow("融合最少一致视图", self.fusion_min_views)
        expert_form.addRow("", self.colmap_gpu)
        expert_form.addRow("", self.colmap_resume)
        self.settings_tabs.addTab(simple, "简易模式")
        self.settings_tabs.addTab(engineering, "工程模式")
        self.settings_tabs.addTab(expert, "专家模式")
        layout.addWidget(self.settings_tabs)
        for widget in (
            self.feature_size,
            self.mvs_size,
            self.max_features,
            self.patch_match_source_images,
            self.patch_match_iterations,
            self.mvs_reference_ratio,
            self.spatial_block_target_images,
        ):
            widget.valueChanged.connect(self._mark_custom_mvs_parameters)
        self.geometric_consistency.toggled.connect(self._mark_custom_mvs_parameters)
        self.patch_match_filter.toggled.connect(self._mark_custom_mvs_parameters)
        self.spatial_blocking.toggled.connect(self._refresh_dense_quality_summary)
        self.mvs_reference_strategy.currentIndexChanged.connect(
            self._mark_custom_mvs_parameters
        )
        self._apply_reconstruction_preset("标准工程模式")

        self.run_colmap_button = QPushButton("开始空三")
        self.run_colmap_button.setObjectName("primaryButton")
        self.run_colmap_button.clicked.connect(self._run_sparse)
        layout.addWidget(self.run_colmap_button)

        progress_group = QGroupBox("处理阶段")
        progress_layout = QVBoxLayout(progress_group)
        self.pipeline_stage_table = _table(["步骤", "业务阶段", "状态"], 220)
        _fill_table(
            self.pipeline_stage_table,
            [[number, label, "等待中"] for _key, number, label in BUSINESS_STAGES],
        )
        progress_layout.addWidget(self.pipeline_stage_table)
        layout.addWidget(progress_group)

        result_group = QGroupBox("空三检查")
        result_layout = QGridLayout(result_group)
        self.sparse_registered = QLabel("—")
        self.sparse_unregistered = QLabel("—")
        self.sparse_points = QLabel("—")
        self.sparse_error = QLabel("—")
        self.sparse_weak = QLabel("—")
        for column, (title, widget) in enumerate(
            (
                ("成功注册照片", self.sparse_registered),
                ("未注册照片", self.sparse_unregistered),
                ("稀疏点数量", self.sparse_points),
                ("平均重投影误差", self.sparse_error),
                ("质量提示", self.sparse_weak),
            )
        ):
            title_label = QLabel(title)
            title_label.setObjectName("statCaption")
            widget.setObjectName("statValue")
            widget.setWordWrap(True)
            result_layout.addWidget(title_label, 0, column)
            result_layout.addWidget(widget, 1, column)
        self.colmap_result = _read_only_text(8)
        result_layout.addWidget(self.colmap_result, 2, 0, 1, 5)
        layout.addWidget(result_group)

        continue_button = QPushButton("确认空三结果，进入稠密重建")
        continue_button.setObjectName("primaryButton")
        continue_button.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.dense_tab)
        )
        layout.addWidget(continue_button)
        return tab

    def _build_dense_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.dense_sparse_gate = QLabel(
            "尚未完成空三。请先在“3 · 空三”中检查注册照片和稀疏模型。"
        )
        self.dense_sparse_gate.setObjectName("warningLabel")
        self.dense_sparse_gate.setWordWrap(True)
        layout.addWidget(self.dense_sparse_gate)

        summary_group = QGroupBox("稠密重建设置")
        summary_form = QFormLayout(summary_group)
        self.dense_quality_summary = QLabel(
            "标准工程模式 · MVS 3072 px · 几何一致性开启 · "
            "过滤开启 · 源照片 12 · 迭代 4"
        )
        self.dense_quality_summary.setWordWrap(True)
        self.dense_disk_hint = QLabel(
            "实际显存、磁盘需求由照片数量与 MVS 分辨率决定；启动前会自动检查磁盘空间。"
        )
        self.dense_disk_hint.setWordWrap(True)
        self.dense_reference_summary = QLabel(
            "全部注册照片都会生成MVS深度图，三档模式均固定为100%参考帧。"
        )
        self.dense_reference_summary.setWordWrap(True)
        summary_form.addRow("当前预设", self.dense_quality_summary)
        summary_form.addRow("MVS工作集", self.dense_reference_summary)
        summary_form.addRow("资源检查", self.dense_disk_hint)
        layout.addWidget(summary_group)

        self.run_dense_button = QPushButton("确认并开始稠密重建")
        self.run_dense_button.setObjectName("primaryButton")
        self.run_dense_button.clicked.connect(self._run_dense)
        layout.addWidget(self.run_dense_button)

        result_group = QGroupBox("稠密点云成果")
        result_layout = QVBoxLayout(result_group)
        self.dense_result = _read_only_text(12)
        result_layout.addWidget(self.dense_result)
        self.colmap_output = QLineEdit()
        self.colmap_output.setReadOnly(True)
        self.colmap_output.setPlaceholderText("完成后显示稠密点云文件")
        result_layout.addWidget(self.colmap_output)
        open_button = QPushButton("打开摄影测量工作目录")
        open_button.clicked.connect(
            lambda: self._open_path(
                self.session.photogrammetry_result.get("folder", "")
                or self.session.sparse_result.get("folder", "")
            )
        )
        result_layout.addWidget(open_button)
        layout.addWidget(result_group)
        layout.addStretch(1)
        return tab

    def _build_export_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        view_group = QGroupBox("成果显示")
        view_layout = QHBoxLayout(view_group)
        self.result_pointcloud_button = QPushButton("点云")
        self.result_pointcloud_button.setObjectName("resultViewButton")
        self.result_pointcloud_button.setCheckable(True)
        self.result_pointcloud_button.clicked.connect(self._show_pointcloud_result)
        self.result_model_button = QPushButton("3D 模型")
        self.result_model_button.setObjectName("resultViewButton")
        self.result_model_button.setCheckable(True)
        self.result_model_button.clicked.connect(self._show_textured_model)
        self.result_view_status = QLabel("尚无可显示成果")
        self.result_view_status.setObjectName("resultViewStatus")
        view_layout.addWidget(self.result_pointcloud_button)
        view_layout.addWidget(self.result_model_button)
        view_layout.addWidget(self.result_view_status, 1)
        layout.addWidget(view_group)

        ready_group = QGroupBox("三维成果")
        ready_layout = QVBoxLayout(ready_group)
        self.simple_result_title = QLabel("尚未生成点云")
        self.simple_result_title.setObjectName("homeTitle")
        self.simple_result_summary = QLabel(
            "完成一键重建后，这里会显示照片注册数、点云数量和成果位置。"
        )
        self.simple_result_summary.setWordWrap(True)
        self.simple_result_path = QLineEdit()
        self.simple_result_path.setReadOnly(True)
        self.simple_result_path.setPlaceholderText("完成后显示点云文件")
        result_actions = QHBoxLayout()
        open_raw = QPushButton("打开点云所在文件夹")
        open_raw.clicked.connect(
            lambda: self._open_path(
                Path(self.simple_result_path.text()).parent
                if self.simple_result_path.text()
                else ""
            )
        )
        self.simple_export_button = QPushButton("导出成果包（点云 / 模型 / 报告 / ZIP）")
        self.simple_export_button.setObjectName("primaryButton")
        self.simple_export_button.clicked.connect(self._run_export)
        result_actions.addWidget(open_raw)
        result_actions.addWidget(self.simple_export_button)
        ready_layout.addWidget(self.simple_result_title)
        ready_layout.addWidget(self.simple_result_summary)
        ready_layout.addWidget(self.simple_result_path)
        ready_layout.addLayout(result_actions)
        layout.addWidget(ready_group)

        model_group = QGroupBox("纹理三维模型")
        model_layout = QVBoxLayout(model_group)
        self.simple_model_title = QLabel("尚未生成三维模型")
        self.simple_model_title.setObjectName("homeTitle")
        self.simple_model_summary = QLabel(
            "可在一键处理前勾选“纹理三维模型”，也可从已有稠密点云继续生成。"
        )
        self.simple_model_summary.setWordWrap(True)
        self.simple_model_path = QLineEdit()
        self.simple_model_path.setReadOnly(True)
        self.simple_model_path.setPlaceholderText("完成后显示纹理三角网格文件")
        model_actions = QHBoxLayout()
        self.generate_model_button = QPushButton("从当前点云生成 / 恢复模型")
        self.generate_model_button.setObjectName("primaryButton")
        self.generate_model_button.clicked.connect(self._run_model)
        self.show_model_detail_button = QPushButton("在视窗显示模型")
        self.show_model_detail_button.clicked.connect(self._show_textured_model)
        open_model = QPushButton("打开模型成果目录")
        open_model.clicked.connect(
            lambda: self._open_path(
                self.session.model_result.get("folder", "")
                or (
                    Path(self.simple_model_path.text()).parent
                    if self.simple_model_path.text()
                    else ""
                )
            )
        )
        model_actions.addWidget(self.generate_model_button)
        model_actions.addWidget(self.show_model_detail_button)
        model_actions.addWidget(open_model)
        model_layout.addWidget(self.simple_model_title)
        model_layout.addWidget(self.simple_model_summary)
        model_layout.addWidget(self.simple_model_path)
        model_layout.addLayout(model_actions)
        layout.addWidget(model_group)

        filter_group = QGroupBox("点云清理与降采样")
        filter_form = QGridLayout(filter_group)
        self.filter_distance_mad = _double_spin(0.0, 50.0, 0.0, 1, 0.5)
        self.filter_distance_mad.setSpecialValueText("关闭")
        self.filter_voxel = _double_spin(0.0, 1_000_000.0, 0.0, 6, 0.01)
        self.filter_voxel.setSpecialValueText("关闭")
        self.filter_neighbors = QSpinBox()
        self.filter_neighbors.setRange(0, 500)
        self.filter_neighbors.setValue(0)
        self.filter_neighbors.setSpecialValueText("关闭")
        self.filter_std_ratio = _double_spin(0.1, 20.0, 2.0, 2, 0.1)
        self.filter_radius = _double_spin(0.0, 1_000_000.0, 0.0, 6, 0.01)
        self.filter_radius.setSpecialValueText("关闭")
        self.filter_radius_neighbors = QSpinBox()
        self.filter_radius_neighbors.setRange(0, 10_000)
        self.filter_radius_neighbors.setValue(0)
        self.filter_radius_neighbors.setSpecialValueText("关闭")
        self.run_filter_button = QPushButton("执行过滤并缓存结果")
        self.run_filter_button.setObjectName("primaryButton")
        self.run_filter_button.clicked.connect(self._run_filter)
        filter_form.addWidget(QLabel("全局距离 MAD 倍数"), 0, 0)
        filter_form.addWidget(self.filter_distance_mad, 0, 1)
        filter_form.addWidget(QLabel("体素尺寸（当前单位）"), 1, 0)
        filter_form.addWidget(self.filter_voxel, 1, 1)
        filter_form.addWidget(QLabel("统计邻居数"), 1, 2)
        filter_form.addWidget(self.filter_neighbors, 1, 3)
        filter_form.addWidget(QLabel("统计标准差倍数"), 2, 0)
        filter_form.addWidget(self.filter_std_ratio, 2, 1)
        filter_form.addWidget(QLabel("半径 / 最少邻居"), 2, 2)
        radius_row = QWidget()
        radius_layout = QHBoxLayout(radius_row)
        radius_layout.setContentsMargins(0, 0, 0, 0)
        radius_layout.addWidget(self.filter_radius)
        radius_layout.addWidget(self.filter_radius_neighbors)
        filter_form.addWidget(radius_row, 2, 3)
        filter_form.addWidget(self.run_filter_button, 3, 0, 1, 4)
        self.filter_result = _read_only_text(8)
        filter_form.addWidget(self.filter_result, 4, 0, 1, 4)
        layout.addWidget(filter_group)

        export_options_group = QGroupBox("高级导出设置")
        form = QFormLayout(export_options_group)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_dir = QLineEdit(str(_outputs_root))
        browse = QPushButton("选择…")
        browse.clicked.connect(self._choose_output_dir)
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(browse)
        self.export_max_points = QSpinBox()
        self.export_max_points.setRange(0, 100_000_000)
        self.export_max_points.setValue(5_000_000)
        self.export_max_points.setSpecialValueText("不限制")
        self.include_las = QCheckBox("标定后同时导出 LAS")
        self.include_las.setChecked(True)
        form.addRow("成果根目录", output_row)
        form.addRow("最大点数", self.export_max_points)
        form.addRow("", self.include_las)
        formats = QLabel(
            "成果包括彩色PLY、标定后LAS、已有的OBJ/FBX/glTF/GLB/OSGB模型、"
            "控制/检查点CSV、测量CSV、JSON/HTML质量报告和ZIP成果包。"
        )
        formats.setWordWrap(True)
        form.addRow("输出内容", formats)
        self.export_button = QPushButton("生成成果包与精度报告")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self._run_export)
        form.addRow(self.export_button)
        layout.addWidget(export_options_group)
        self.export_result = _read_only_text(18)
        layout.addWidget(self.export_result)
        self.export_folder = QLineEdit()
        self.export_folder.setReadOnly(True)
        layout.addWidget(self.export_folder)
        open_folder = QPushButton("打开成果目录")
        open_folder.clicked.connect(lambda: self._open_path(self.export_folder.text()))
        layout.addWidget(open_folder)
        guide_group = QGroupBox("现场与精度指南")
        guide_layout = QVBoxLayout(guide_group)
        guide = QTextBrowser()
        guide.setMarkdown(GUIDE)
        guide.setOpenExternalLinks(True)
        guide.setMinimumHeight(260)
        guide_layout.addWidget(guide)
        layout.addWidget(guide_group)
        self._advanced_only_widgets.extend(
            [
                filter_group,
                export_options_group,
                self.export_result,
                self.export_folder,
                open_folder,
                guide_group,
            ]
        )
        self._update_result_view_controls()
        layout.addStretch(1)
        return tab

    def _build_guide_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        guide = QTextBrowser()
        guide.setMarkdown(GUIDE)
        guide.setOpenExternalLinks(True)
        layout.addWidget(guide)
        return tab

    def _new_project(self) -> bool:
        if self.process_task.running:
            self._error("后台任务运行时不能切换工程")
            return False
        self.tabs.setCurrentWidget(self.project_tab)
        self.home_project_name.setFocus()
        return False

    def _choose_new_project_root(self) -> None:
        root = QFileDialog.getExistingDirectory(
            self,
            "选择工程保存位置",
            self.home_project_root.text()
            or str(_projects_root),
        )
        if root:
            self.home_project_root.setText(root)

    def _create_project_from_home(self) -> bool:
        if self.process_task.running:
            self._error("后台任务运行时不能切换工程")
            return False
        name = self.home_project_name.text().strip()
        if not name:
            self._error("请输入项目名称")
            return False
        parent = Path(
            self.home_project_root.text().strip()
            or _projects_root
        ).expanduser()
        safe_name = "".join(
            "_" if character in '<>:"/\\|?*' else character
            for character in name
        ).strip(" .")
        root = parent / (safe_name or "photogrammetry_project")
        try:
            store = ProjectStore.create(
                root,
                name,
                project_type=self.home_project_type.currentText(),
                output_coordinate_system=self.home_coordinate_system.currentText(),
                precision_mode=self.home_precision.currentText(),
            )
        except Exception as exc:
            self._error(str(exc))
            return False
        self._activate_project(store)
        self.reconstruction_preset.setCurrentText(self.home_precision.currentText())
        self.intended_mode.setCurrentText(self.home_precision.currentText())
        self.tabs.setCurrentWidget(self.reconstruction_tab)
        self.statusBar().showMessage(f"已创建工程：{store.root}", 8000)
        return True

    def _open_project(self) -> None:
        if self.process_task.running:
            self._error("后台任务运行时不能切换工程")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 AI 摄影测量工程",
            str(_projects_root),
            "摄影测量工程 (project.json);;JSON 文件 (*.json)",
        )
        if not path:
            return
        self._open_project_path(path)

    def _open_project_path(self, path: str | Path) -> None:
        try:
            store = ProjectStore.open(path)
        except Exception as exc:
            self._error(f"工程打开失败：{exc}")
            return
        self._activate_project(store)
        if self.session is not None and self.session.photogrammetry_result:
            self.tabs.setCurrentWidget(self.export_tab)
        else:
            self.tabs.setCurrentWidget(self.reconstruction_tab)
        self.statusBar().showMessage(f"已打开工程：{store.root}", 8000)

    def _activate_project(self, store: ProjectStore) -> None:
        recovered_stages = store.recover_interrupted_stages()
        self._interrupted_task_notice = (
            f"检测到上次任务中断（{len(recovered_stages)}个阶段），缓存已保留，可断点继续"
            if recovered_stages
            else ""
        )
        session = store.load_session()
        self.project_store = store
        self.session = session
        manifest = store.read_manifest()
        saved_view_mode = str(
            self.settings.value(
                f"result_view_mode/{session.project_id}",
                "pointcloud",
            )
        )
        self._result_view_mode = (
            "model" if saved_view_mode == "model" else "pointcloud"
        )
        self.project_name.setText(session.project_name)
        self.input_images = [str(value) for value in manifest.get("source_images", [])]
        self.selected_images = [str(value) for value in manifest.get("selected_images", [])]
        self.source_root = None
        self.output_dir.setText(str((store.root / "export").resolve()))
        previous_photogrammetry = session.photogrammetry_result or session.sparse_result
        photogrammetry_options = session.photogrammetry_options
        previous_folder = Path(str(previous_photogrammetry.get("folder", "")))
        saved_photogrammetry_root = (
            photogrammetry_options.get("output_root")
            or (
                previous_folder.parent
                if previous_folder.name
                else (store.root / "colmap").resolve()
            )
        )
        self.photogrammetry_output_root.setText(
            str(saved_photogrammetry_root)
        )
        self.feature_method.setCurrentIndex(
            0 if photogrammetry_options.get("feature_type", "aliked") == "aliked" else 1
        )
        matcher_index = {
            "auto": 0,
            "exhaustive": 1,
            "sequential": 2,
        }.get(str(photogrammetry_options.get("matcher", "auto")), 0)
        self.matcher.setCurrentIndex(matcher_index)
        self.sfm_mapper.setCurrentIndex(
            0 if photogrammetry_options.get("mapper", "global") == "global" else 1
        )
        camera_index = self.camera_model.findText(
            str(photogrammetry_options.get("camera_model", "SIMPLE_RADIAL"))
        )
        if camera_index >= 0:
            self.camera_model.setCurrentIndex(camera_index)
        self.single_camera.setChecked(
            bool(photogrammetry_options.get("single_camera", True))
        )
        self.feature_size.setValue(
            int(photogrammetry_options.get("feature_max_image_size", 3072))
        )
        self.mvs_size.setValue(
            int(photogrammetry_options.get("max_image_size", 3072))
        )
        self.max_features.setValue(
            int(photogrammetry_options.get("max_num_features", 4096))
        )
        self.sequential_overlap.setValue(
            int(photogrammetry_options.get("sequential_overlap", 20))
        )
        self.colmap_gpu.setChecked(bool(photogrammetry_options.get("use_gpu", True)))
        self.colmap_resume.setChecked(bool(photogrammetry_options.get("resume", True)))
        self.geometric_consistency.setChecked(
            bool(photogrammetry_options.get("geometric_consistency", True))
        )
        self.patch_match_filter.setChecked(
            bool(photogrammetry_options.get("patch_match_filter", True))
        )
        self.patch_match_source_images.setValue(
            int(photogrammetry_options.get("patch_match_source_images", 12))
        )
        self.patch_match_iterations.setValue(
            int(photogrammetry_options.get("patch_match_iterations", 4))
        )
        self.mvs_reference_strategy.setCurrentIndex(0)
        self.mvs_reference_ratio.setValue(100)
        self.spatial_blocking.setChecked(
            bool(photogrammetry_options.get("spatial_blocking", True))
        )
        self.spatial_block_threshold.setValue(
            int(photogrammetry_options.get("spatial_block_threshold", 180))
        )
        self.spatial_block_target_images.setValue(
            int(photogrammetry_options.get("spatial_block_target_images", 120))
        )
        self.spatial_block_halo_ratio.setValue(
            float(photogrammetry_options.get("spatial_block_halo_ratio", 0.20))
        )
        reference = session.coordinate_reference
        self.coordinate_input_mode.setCurrentIndex(
            1 if reference.mode in {"wgs84_enu", "wgs84_projected"} else 0
        )
        self.control_target_crs.setText(
            reference.target_crs
            if reference.target_crs not in {"LOCAL_CARTESIAN", "LOCAL_ENU"}
            else ""
        )
        self.generate_quality_report.setChecked(
            bool(photogrammetry_options.get("generate_quality_report", True))
        )
        self.min_num_inliers.setValue(
            int(photogrammetry_options.get("min_num_inliers", 20))
        )
        self.ransac_max_error.setValue(
            float(photogrammetry_options.get("ransac_max_error", 4.0))
        )
        self.fusion_min_views.setValue(
            int(photogrammetry_options.get("fusion_min_num_pixels", 2))
        )
        precision_mode = PRESET_NAME_ALIASES.get(
            str(manifest.get("precision_mode", "标准工程模式")),
            str(manifest.get("precision_mode", "标准工程模式")),
        )
        if not photogrammetry_options and precision_mode in RECONSTRUCTION_PRESETS:
            selected_mode = precision_mode
            self._apply_reconstruction_preset(precision_mode)
        else:
            selected_mode = (
                precision_mode
                if self._current_mvs_parameters_match_preset(precision_mode)
                else "自定义参数"
            )
        self.reconstruction_preset.blockSignals(True)
        self.reconstruction_preset.setCurrentText(selected_mode)
        self.reconstruction_preset.blockSignals(False)
        beginner_mode = (
            selected_mode
            if selected_mode in RECONSTRUCTION_PRESETS
            else precision_mode
            if precision_mode in RECONSTRUCTION_PRESETS
            else "标准工程模式"
        )
        self.intended_mode.blockSignals(True)
        self.intended_mode.setCurrentText(beginner_mode)
        self.intended_mode.blockSignals(False)
        model_options = session.model_options or {}
        requested_model_formats = {
            str(value).lower() for value in model_options.get("formats", [])
        }
        wants_model = bool(model_options.get("generate_model", session.has_model))
        self.output_textured_model.setChecked(wants_model)
        self.output_model_obj.setChecked(
            not requested_model_formats or "obj" in requested_model_formats
        )
        self.output_model_fbx.setChecked(
            not requested_model_formats or "fbx" in requested_model_formats
        )
        self.output_model_gltf.setChecked(
            not requested_model_formats
            or bool(requested_model_formats & {"gltf", "glb"})
        )
        if self.output_model_osgb.isEnabled():
            self.output_model_osgb.setChecked("osgb" in requested_model_formats)
        self._toggle_model_outputs(wants_model)
        self._refresh_dense_quality_summary()
        self._load_photo_scan(manifest.get("photo_scan") or {})
        self._update_source_summary()
        self._refresh_all_tables()
        self.cloud_view.clear()
        scan_payload = manifest.get("photo_scan") or {}
        if scan_payload:
            self.quality_text.setPlainText(_json(scan_payload))
        self.filter_result.setPlainText(_json(session.filter_report) if session.filter_report else "")
        sparse = session.sparse_result
        photogrammetry = session.photogrammetry_result
        self.colmap_result.setPlainText(_json(sparse) if sparse else "")
        self.dense_result.setPlainText(_json(photogrammetry) if photogrammetry else "")
        self.colmap_output.setText(str(photogrammetry.get("pointcloud", "")))
        self._show_sparse_result(sparse)
        self._update_simple_result()
        try:
            pointcloud_source, _transform = self._pointcloud_display_source()
            if self._result_view_mode == "model" and session.has_model:
                self._show_textured_model()
            elif pointcloud_source:
                self._refresh_cloud()
            elif session.has_model:
                self._show_textured_model()
            elif sparse:
                self.cloud_view.load_pointcloud(
                    str(sparse["sparse_pointcloud"]),
                    unit="模型单位",
                    camera_images_txt=str(sparse.get("sparse_images_txt", "")),
                    label="Sparse BA",
                )
            else:
                self.cloud_view.clear()
        except Exception as exc:
            _logger.warning("Could not restore photogrammetry point cloud: %s", exc)
        self._update_project_status()
        if recovered_stages:
            self.one_click_status.setText(
                "上次重建未正常结束，已有照片特征、匹配、空三或深度缓存均已保留。"
                "直接点击“一键生成点云”即可从有效断点继续。"
            )
            self.statusBar().showMessage(self._interrupted_task_notice, 12_000)
        self._remember_project(store)
        self._refresh_project_home()

    def _recent_project_paths(self) -> list[str]:
        raw = self.settings.value("recent_projects", "[]")
        try:
            values = json.loads(str(raw))
        except json.JSONDecodeError:
            values = []
        return [
            str(Path(value).resolve())
            for value in values
            if isinstance(value, str)
            and not any(
                part.casefold().startswith("pytest-")
                for part in Path(value).parts
            )
            and (Path(value) / "project.json").is_file()
        ][:12]

    def _remember_project(self, store: ProjectStore) -> None:
        current = str(store.root.resolve())
        values = [
            current,
            *[
                value
                for value in self._recent_project_paths()
                if value.casefold() != current.casefold()
            ],
        ][:12]
        self.settings.setValue(
            "recent_projects",
            json.dumps(values, ensure_ascii=False),
        )
        self._load_recent_projects()

    def _load_recent_projects(self) -> None:
        rows: list[list[str]] = []
        valid_paths: list[str] = []
        for value in self._recent_project_paths():
            try:
                store = ProjectStore.open(value)
                manifest = store.read_manifest()
            except Exception:
                continue
            valid_paths.append(value)
            rows.append(
                [
                    str(manifest.get("project_name") or Path(value).name),
                    value,
                    str(manifest.get("updated_at", "")),
                ]
            )
        self.settings.setValue(
            "recent_projects",
            json.dumps(valid_paths, ensure_ascii=False),
        )
        _fill_table(self.recent_projects_table, rows)
        if rows:
            self.recent_projects_table.selectRow(0)

    def _open_selected_recent(self) -> None:
        row = self.recent_projects_table.currentRow()
        if row < 0 and self.recent_projects_table.rowCount():
            row = 0
        if row < 0:
            self._error("没有可继续的最近项目")
            return
        item = self.recent_projects_table.item(row, 1)
        if item:
            self._open_project_path(item.text())

    def _refresh_project_home(self) -> None:
        if self.project_store is None:
            self.current_project_details.setText("尚未打开工程")
            return
        manifest = self.project_store.read_manifest()
        self.current_project_details.setText(
            f"{self.session.project_name}\n"
            f"{self.project_store.root}\n"
            f"{manifest.get('project_type', '照片三维重建')} · "
            f"{manifest.get('precision_mode', '标准工程模式')} · "
            f"{manifest.get('output_coordinate_system', '本地模型坐标')}"
        )

    def _save_project(self, *, silent: bool = False) -> bool:
        if self.project_store is None:
            if not silent:
                self._error("当前没有工程，请先新建或打开工程")
            return False
        try:
            self.session.project_name = self.project_name.text().strip() or "未命名项目"
            manifest = self.project_store.read_manifest()
            self.project_store.save_session(
                self.session,
                source_images=self.input_images,
                selected_images=self.selected_images,
                photo_scan=manifest.get("photo_scan") or {},
            )
            self._remember_project(self.project_store)
            self._refresh_project_home()
        except Exception as exc:
            if not silent:
                self._error(f"工程保存失败：{exc}")
            return False
        if not silent:
            self.statusBar().showMessage(f"工程已保存：{self.project_store.project_file}", 6000)
        return True

    def _close_project(self) -> None:
        if self.process_task.running:
            self._error("请先取消或等待后台任务完成")
            return
        self._save_project(silent=True)
        self.project_store = None
        self.session = ProjectSession()
        self._interrupted_task_notice = ""
        self.input_images.clear()
        self.selected_images.clear()
        self.source_root = None
        self.project_name.setText("未命名项目")
        self._update_source_summary()
        self._refresh_all_tables()
        _fill_table(self.photo_table, [])
        self.quality_text.clear()
        self.filter_result.clear()
        self.colmap_result.clear()
        self.dense_result.clear()
        self.colmap_output.clear()
        self.photogrammetry_output_root.clear()
        self.cloud_view.clear()
        self._show_sparse_result({})
        self._update_simple_result()
        self._refresh_photo_stats({})
        self._refresh_project_home()
        self._update_project_status()
        self.statusBar().showMessage("工程已关闭", 5000)

    def _ensure_project(self) -> bool:
        return self.project_store is not None or self._new_project()

    def _choose_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择重建照片",
            str(Path.home()),
            "照片 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;所有文件 (*)",
        )
        if paths:
            self.input_images = paths
            self.selected_images.clear()
            self.source_root = None
            _fill_table(self.photo_table, [])
            self._refresh_photo_stats({})
            self._update_source_summary()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if any(url.isLocalFile() for url in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        photos: list[str] = []
        for path in paths:
            if path.is_dir():
                photos.extend(discover_photos(path, recursive=True))
            elif path.is_file() and path.suffix.lower() in PHOTO_EXTENSIONS:
                photos.append(str(path.resolve()))
        if not photos:
            self._error("拖入内容中没有支持的照片")
            return
        self.input_images = sorted(set(photos), key=str.casefold)
        self.selected_images.clear()
        self.source_root = None
        _fill_table(self.photo_table, [])
        self._refresh_photo_stats({})
        self._update_source_summary()
        self.tabs.setCurrentWidget(self.reconstruction_tab)
        event.acceptProposedAction()

    def _choose_image_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择照片文件夹",
            self.source_root or str(Path.home()),
        )
        if path:
            self.source_root = path
            self.input_images.clear()
            self.selected_images.clear()
            _fill_table(self.photo_table, [])
            self._refresh_photo_stats({})
            self._update_source_summary()

    def _clear_sources(self) -> None:
        self.input_images.clear()
        self.selected_images.clear()
        self.source_root = None
        _fill_table(self.photo_table, [])
        self._refresh_photo_stats({})
        self._update_source_summary()

    def _update_source_summary(self) -> None:
        if self.input_images:
            names = "、".join(Path(path).name for path in self.input_images[:5])
            suffix = "…" if len(self.input_images) > 5 else ""
            selected = (
                f"；关键帧 {len(self.selected_images)} 张"
                if self.selected_images
                else "；尚未执行关键帧筛选"
            )
            self.source_summary.setText(
                f"已选择 {len(self.input_images)} 张照片{selected}：{names}{suffix}"
            )
        elif self.source_root:
            self.source_summary.setText(f"待扫描照片文件夹：{self.source_root}")
        else:
            self.source_summary.setText("尚未选择照片")

    def _scan_photos(self) -> None:
        self._start_photo_scan(self._photo_scan_finished, beginner=False)

    def _start_photo_scan(self, completion, *, beginner: bool) -> None:
        if not self.input_images and not self.source_root:
            self._error("请先选择照片或照片文件夹")
            return
        if not self._ensure_project():
            return
        assert self.project_store is not None
        self.project_store.update_manifest(project_name=self.project_name.text().strip() or "未命名项目")
        config: dict[str, Any] = {
            "task": "scan_photos",
            "project_root": str(self.project_store.root),
            "max_keyframes": 0 if beginner else self.max_keyframes.value(),
            "include_near_duplicates": self.include_near_duplicates.isChecked(),
            "selection_policy": "automatic"
            if beginner
            else (
                "keep_all"
                if self.photo_selection_policy.currentText().startswith("全部")
                else "duplicates_only"
                if self.photo_selection_policy.currentText().startswith("仅排除")
                else "automatic"
            ),
            "auto_segment": False
            if beginner
            else self.auto_segment.isChecked(),
            "recursive": True,
        }
        if self.input_images:
            config["source_images"] = self.input_images
        else:
            config["source_root"] = self.source_root
        self._run_process_task("自动检查照片", config, completion)

    def _load_photo_scan(self, payload: dict[str, Any]) -> None:
        rows: list[list[Any]] = []
        records: list[PhotoRecord] = []
        for item in payload.get("records", []):
            record = PhotoRecord.from_dict(item)
            records.append(record)
            warnings = [
                value
                for value in record.warning.split("；")
                if value and not value.startswith("需按 EXIF 方向")
            ]
            if record.duplicate_of:
                warnings.append(f"完全重复：{record.duplicate_of}")
            elif record.near_duplicate_of:
                warnings.append(f"近重复：{record.near_duplicate_of}")
            if not record.valid:
                status = "异常"
            elif not record.selected:
                status = "已排除"
            elif warnings:
                status = "需复核"
            else:
                status = "合格"
            exposure = (
                "合格"
                if record.dark_ratio <= 0.08 and record.bright_ratio <= 0.08
                else f"暗 {record.dark_ratio:.0%} / 亮 {record.bright_ratio:.0%}"
            )
            gps = (
                (
                    f"{record.rtk_status} "
                    if record.rtk_status not in {"", "NONE", "GPS"}
                    else ""
                )
                + f"{record.gps_latitude:.5f}, {record.gps_longitude:.5f}"
                + (
                    f" σ={record.sigma_x:.3f}/{record.sigma_y:.3f}/{record.sigma_z:.3f}m"
                    if record.sigma_x is not None
                    and record.sigma_y is not None
                    and record.sigma_z is not None
                    else ""
                )
                if record.gps_latitude is not None
                and record.gps_longitude is not None
                else ""
            )
            messages = list(warnings)
            orientation_info = _orientation_display_info(record.orientation)
            if orientation_info:
                messages.append(orientation_info)
            rows.append(
                [
                    "",
                    status,
                    record.name,
                    f"{record.width} × {record.height}",
                    round(record.sharpness, 1),
                    exposure,
                    gps,
                    record.focal_length_mm or "",
                    record.captured_at,
                    record.lens_model,
                    "；".join(value for value in messages if value),
                ]
            )
        _fill_table(self.photo_table, rows)
        self.photo_table.setIconSize(QSize(80, 60))
        colors = {
            "合格": QColor("#dff3e4"),
            "需复核": QColor("#fff1c7"),
            "异常": QColor("#ffd9d9"),
            "已排除": QColor("#e4e8ea"),
        }
        for row_index, record in enumerate(records):
            status_item = self.photo_table.item(row_index, 1)
            status = status_item.text() if status_item else "已排除"
            brush = QBrush(colors[status])
            for column in range(self.photo_table.columnCount()):
                item = self.photo_table.item(row_index, column)
                if item:
                    item.setBackground(brush)
            preview = self.photo_table.item(row_index, 0)
            if preview and record.thumbnail_path and Path(record.thumbnail_path).is_file():
                preview.setIcon(QIcon(record.thumbnail_path))
            self.photo_table.setRowHeight(row_index, 64)
        self._refresh_photo_stats(payload.get("summary") or {})

    def _refresh_photo_stats(self, summary: dict[str, Any]) -> None:
        self.photo_stat_total.setText(str(summary.get("photo_count", 0)))
        self.photo_stat_valid.setText(str(summary.get("valid_count", 0)))
        self.photo_stat_blur.setText(str(summary.get("blur_count", 0)))
        duplicate_count = int(summary.get("exact_duplicate_count", 0)) + int(
            summary.get("near_duplicate_count", 0)
        )
        self.photo_stat_duplicate.setText(str(duplicate_count))
        gps_count = int(summary.get("gps_count", 0))
        rtk_fix_count = int(summary.get("rtk_fix_count", 0))
        self.photo_stat_gps.setText(
            f"{gps_count} / Fix {rtk_fix_count}" if rtk_fix_count else str(gps_count)
        )
        camera_models = summary.get("camera_models") or []
        self.photo_stat_camera.setText("、".join(camera_models[:2]) or "—")

    def _photo_scan_finished(self, _result: dict) -> None:
        assert self.project_store is not None
        manifest = self.project_store.read_manifest()
        self.input_images = [str(value) for value in manifest.get("source_images", [])]
        self.selected_images = [str(value) for value in manifest.get("selected_images", [])]
        payload = manifest.get("photo_scan") or {}
        self._load_photo_scan(payload)
        self.quality_text.setPlainText(
            _json(
                {
                    "照片扫描": payload.get("summary", {}),
                    "序列连续性": payload.get("sequence_analysis", {}),
                    "自动分段": payload.get("segmentation", {}),
                    "已选择关键帧": len(self.selected_images),
                    "说明": "完全重复项自动排除；近重复项默认排除，可在扫描选项中保留。",
                }
            )
        )
        self._update_source_summary()
        self._set_business_stage("photo_scan", "已完成")
        self.statusBar().showMessage(
            f"照片扫描完成：{len(self.input_images)} 张，选择 {len(self.selected_images)} 张关键帧",
            8000,
        )

    def _toggle_model_outputs(self, enabled: bool) -> None:
        for widget in (
            self.output_model_obj,
            self.output_model_fbx,
            self.output_model_gltf,
        ):
            widget.setEnabled(bool(enabled))
        self.output_model_osgb.setEnabled(bool(enabled and find_osgconv()))

    def _selected_model_formats(self) -> list[str]:
        formats: list[str] = []
        if self.output_model_obj.isChecked():
            formats.append("obj")
        if self.output_model_fbx.isChecked():
            formats.append("fbx")
        if self.output_model_gltf.isChecked():
            formats.append("gltf")
        if self.output_model_osgb.isChecked() and self.output_model_osgb.isEnabled():
            formats.append("osgb")
        return formats

    def _run_one_click(self) -> None:
        if self.process_task.running:
            self._error("已有任务正在运行，请等待完成或取消")
            return
        if not self.input_images and not self.source_root:
            self._error("请先添加照片或照片文件夹")
            return
        if not self._ensure_project():
            return
        precision = self.intended_mode.currentText()
        if precision not in RECONSTRUCTION_PRESETS:
            precision = "标准工程模式"
        self.reconstruction_preset.setCurrentText(precision)
        self.feature_method.setCurrentIndex(0)
        self.matcher.setCurrentIndex(0)
        self.sfm_mapper.setCurrentIndex(0)
        self._one_click_wants_model = self.output_textured_model.isChecked()
        if self._one_click_wants_model and not self._selected_model_formats():
            self._error("已选择生成模型，请至少勾选一种模型格式")
            return
        self._one_click_active = True
        self._one_click_retry_used = False
        self._reset_business_stages()
        step_count = 4 if self._one_click_wants_model else 3
        self.one_click_status.setText(
            f"第1步/{step_count}：正在自动检查照片质量和连续性…"
        )
        self._start_photo_scan(
            self._one_click_photo_scan_finished,
            beginner=True,
        )

    def _one_click_photo_scan_finished(self, result: dict) -> None:
        self._photo_scan_finished(result)
        if len(self.selected_images) < 3:
            self._one_click_active = False
            self.one_click_status.setText(
                "可用照片不足3张。请补充清晰且相互重叠的照片后重试。"
            )
            self._error("自动检查后可用照片不足3张，无法进行三维重建")
            return
        step_count = 4 if self._one_click_wants_model else 3
        self.one_click_status.setText(
            f"第2步/{step_count}：{len(self.selected_images)}张照片正在计算相机位置…"
        )
        self._start_one_click_sparse()

    def _start_one_click_sparse(self) -> None:
        config = self._photogrammetry_config("sparse")
        if config is None:
            self._one_click_active = False
            return
        self._colmap_target_stage = "sparse"
        self._set_business_stage("photo_scan", "已完成")
        self._run_process_task(
            "自动建立照片连接与相机位置",
            config,
            self._one_click_sparse_finished,
        )

    def _one_click_sparse_finished(self, result: dict[str, Any]) -> None:
        self._sparse_finished(result)
        gate = dict(result.get("quality_gate") or {})
        gate_status = str(gate.get("status", "review"))
        should_retry = (
            _sparse_requires_stable_retry(result)
            and not self._one_click_retry_used
        )
        if should_retry:
            self._one_click_retry_used = True
            self.feature_method.setCurrentIndex(1)
            self.sfm_mapper.setCurrentIndex(1)
            self.matcher.setCurrentIndex(0)
            self.one_click_status.setText(
                "自动模式发现照片连接不足，正在切换稳定算法重新计算…"
            )
            self._start_one_click_sparse()
            return
        if gate_status == "blocked":
            self._one_click_active = False
            self.one_click_status.setText(
                "照片连接质量不足，已停止耗时的稠密计算。请补拍重叠照片后重试。"
            )
            self._error(
                "照片之间的有效重叠不足，自动稳定模式仍无法形成可靠模型。"
                "请补充相邻视角照片后重新运行。"
            )
            return
        step_count = 4 if self._one_click_wants_model else 3
        self.one_click_status.setText(
            f"第3步/{step_count}：相机位置计算完成，正在生成高分辨率稠密点云…"
        )
        config = self._photogrammetry_config("dense")
        if config is None:
            self._one_click_active = False
            return
        self._colmap_target_stage = "dense"
        self._run_process_task(
            "自动生成稠密点云",
            config,
            self._one_click_dense_finished,
        )

    def _one_click_dense_finished(self, result: dict[str, Any]) -> None:
        self._colmap_finished(result)
        if self._one_click_wants_model:
            self.one_click_status.setText(
                "第4步/4：点云已经生成，正在重建表面并融合原始照片纹理…"
            )
            self._start_one_click_model()
            return
        self._one_click_active = False
        self.one_click_status.setText(
            "重建完成。点云已经显示在右侧，可在“成果”页打开或导出。"
        )
        self._update_simple_result()
        self.tabs.setCurrentWidget(self.export_tab)

    def _model_config(self) -> dict[str, Any] | None:
        if self.project_store is None:
            self._error("请先新建或打开工程")
            return None
        result = self.session.photogrammetry_result
        dense_workspace = str(result.get("dense_workspace", "")).strip()
        pointcloud = str(
            result.get("raw_fused", "") or result.get("pointcloud", "")
        ).strip()
        if not dense_workspace or not pointcloud:
            self._error("当前成果缺少稠密MVS工作区，无法进行原始照片纹理投影")
            return None
        formats = self._selected_model_formats()
        if not formats:
            self._error("请至少选择一种模型格式")
            return None
        precision = self.intended_mode.currentText()
        if precision not in RECONSTRUCTION_PRESETS:
            precision = "标准工程模式"
        config = {
            "task": "model",
            "project_root": str(self.project_store.root),
            "dense_workspace": dense_workspace,
            "pointcloud": pointcloud,
            "colmap_path": self.colmap_path.text().strip() or None,
            "precision_mode": precision,
            "formats": formats,
            "osgconv_path": find_osgconv(),
            "resume": self.colmap_resume.isChecked(),
        }
        self.session.model_options = {
            "generate_model": True,
            "precision_mode": precision,
            "formats": formats,
            "resume": self.colmap_resume.isChecked(),
        }
        self._save_project(silent=True)
        return config

    def _start_one_click_model(self) -> None:
        config = self._model_config()
        if config is None:
            self._one_click_active = False
            return
        self._run_process_task(
            "自动生成纹理三维模型",
            config,
            self._one_click_model_finished,
        )

    def _one_click_model_finished(self, result: dict[str, Any]) -> None:
        self._model_finished(result)
        self._one_click_active = False
        self.one_click_status.setText(
            "处理完成。稠密点云和纹理三维模型均已生成，可在“成果”页查看。"
        )
        self.tabs.setCurrentWidget(self.export_tab)

    def _run_model(self) -> None:
        if self.process_task.running:
            self._error("已有任务正在运行，请等待完成或取消")
            return
        if not self.session.photogrammetry_result:
            self._error("请先生成稠密点云，再继续生成纹理三维模型")
            return
        self.output_textured_model.setChecked(True)
        config = self._model_config()
        if config is None:
            return
        self._run_process_task("生成纹理三维模型", config, self._model_finished)

    def _model_finished(self, result: dict[str, Any]) -> None:
        if self.project_store is not None:
            self.session = self.project_store.load_session()
        else:
            self.session.model_result = dict(result)
        self._update_simple_result()
        for key, _number, _label in BUSINESS_STAGES[8:]:
            self._set_business_stage(key, "已完成")
        try:
            self._show_textured_model()
        except Exception as exc:
            _logger.warning("Could not display textured model: %s", exc)
        self._update_project_status()
        self.statusBar().showMessage(
            f"纹理模型完成：{int(result.get('face_count', 0)):,} 个三角面",
            10_000,
        )

    def _pointcloud_display_source(
        self,
    ) -> tuple[str, SimilarityTransform | None]:
        """Return the preferred point-cloud file and its display transform."""

        filtered = (
            self.project_store.processed_cache / "filtered.ply"
            if self.project_store is not None
            else None
        )
        if (
            self.session.has_processed_cloud
            and filtered is not None
            and filtered.is_file()
        ):
            return str(filtered), None
        source = str(self.session.photogrammetry_result.get("pointcloud", "")).strip()
        if source and Path(source).is_file():
            return source, self.session.transform
        return "", None

    def _set_result_view_mode(self, mode: str, *, persist: bool = True) -> None:
        """Synchronize the result switch after a view loads successfully."""

        mode = "model" if mode == "model" else "pointcloud"
        self._result_view_mode = mode
        if hasattr(self, "result_pointcloud_button"):
            for button, checked in (
                (self.result_pointcloud_button, mode == "pointcloud"),
                (self.result_model_button, mode == "model"),
            ):
                button.blockSignals(True)
                button.setChecked(checked)
                button.blockSignals(False)
            self.result_view_status.setText(
                "当前显示：纹理 3D 模型"
                if mode == "model"
                else "当前显示：彩色点云"
            )
        if persist and self.project_store is not None:
            self.settings.setValue(
                f"result_view_mode/{self.session.project_id}",
                mode,
            )

    def _update_result_view_controls(self) -> None:
        """Enable only the result types that are actually available on disk."""

        if not hasattr(self, "result_pointcloud_button"):
            return
        cloud_available = bool(self._pointcloud_display_source()[0])
        model_available = self.session.has_model
        self.result_pointcloud_button.setEnabled(cloud_available)
        self.result_model_button.setEnabled(model_available)
        self.show_model_detail_button.setEnabled(model_available)
        if self._result_view_mode == "model" and model_available:
            self._set_result_view_mode("model", persist=False)
        elif cloud_available:
            self._set_result_view_mode("pointcloud", persist=False)
        elif model_available:
            self._set_result_view_mode("model", persist=False)
        else:
            self._result_view_mode = "pointcloud"
            for button in (
                self.result_pointcloud_button,
                self.result_model_button,
            ):
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
            self.result_view_status.setText("尚无可显示成果")

    def _show_pointcloud_result(self) -> None:
        source, _transform = self._pointcloud_display_source()
        if not source:
            self._update_result_view_controls()
            self._error("尚未生成可显示的彩色点云")
            return
        self._refresh_cloud()

    def _show_textured_model(self) -> None:
        result = self.session.model_result
        texture_blocks = [
            dict(block)
            for block in result.get("texture_blocks", [])
            if isinstance(block, dict)
        ]
        if texture_blocks:
            self.cloud_view.load_models(
                texture_blocks,
                unit=self.session.unit,
                transform=self.session.transform,
                label="3D model",
            )
            self._set_result_view_mode("model")
            return
        mesh = str(result.get("textured_mesh", "")).strip()
        texture = str(result.get("texture_atlas", "")).strip()
        if (
            not mesh
            or not texture
            or not Path(mesh).is_file()
            or not Path(texture).is_file()
        ):
            self._update_result_view_controls()
            self._error("尚未生成可显示的纹理三维模型")
            return
        self.cloud_view.load_model(
            mesh,
            texture,
            unit=self.session.unit,
            transform=self.session.transform,
            label="3D model",
        )
        self._set_result_view_mode("model")


    def _append_scale_point(
        self,
        model_point: np.ndarray,
        pixel: tuple[float, float] | None = None,
    ) -> None:
        if len(self.scale_points) >= 2:
            self.scale_points.clear()
            self.scale_pixels.clear()
        self.scale_points.append(np.asarray(model_point, dtype=np.float64))
        if pixel is not None:
            self.scale_pixels.append(pixel)
        self._refresh_scale_selection()

    def _refresh_scale_selection(self) -> None:
        rows = [
            [index, *(f"{value:.8f}" for value in point)]
            for index, point in enumerate(self.scale_points, 1)
        ]
        _fill_table(self.scale_selected_table, rows)
        if len(self.scale_points) == 2:
            distance = float(np.linalg.norm(self.scale_points[1] - self.scale_points[0]))
            self.scale_distance_label.setText(f"当前模型距离：{distance:.10g} 模型单位")
        else:
            self.scale_distance_label.setText(f"已选择 {len(self.scale_points)}/2 个端点")

    def _clear_scale_selection(self) -> None:
        self.scale_points.clear()
        self.scale_pixels.clear()
        self._refresh_scale_selection()

    def _add_scale_constraint(self) -> None:
        if len(self.scale_points) != 2:
            self._error("请先在右侧三维点云中选择两个标尺端点")
            return
        try:
            self.session.add_distance_constraint(
                self.scale_label.text(),
                self.scale_points[0],
                self.scale_points[1],
                self.actual_distance.value(),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self._clear_scale_selection()
        _fill_table(self.distance_table, self._distance_rows())
        self._save_project(silent=True)
        self._update_project_status()

    def _clear_distance_constraints(self) -> None:
        self.session.distance_constraints.clear()
        if self.session.transform.mode == "scaled":
            self.session.transform = SimilarityTransform.identity()
            self.session.calibration_report.clear()
            self.session.clear_processed()
            self._refresh_cloud()
        _fill_table(self.distance_table, [])
        self.scale_report.clear()
        self._save_project(silent=True)
        self._update_project_status()

    def _calibrate_scale(self) -> None:
        try:
            report = self.session.calibrate_scale()
            self.scale_report.setPlainText(_json(report))
            self._refresh_cloud()
            self._save_project(silent=True)
            self._update_project_status()
        except Exception as exc:
            self._error(str(exc))


    def _show_control_pick(self) -> None:
        if self.current_control_pick is None:
            self.control_pick_text.setText("尚未选择模型点")
            return
        point = self.current_control_pick["model_xyz"]
        source = self.current_control_pick["image_name"]
        self.control_pick_text.setText(
            f"来源：{source}　模型坐标："
            f"({point[0]:.8f}, {point[1]:.8f}, {point[2]:.8f})"
        )

    def _add_coordinate(self) -> None:
        if self.current_control_pick is None:
            self._error("请先在右侧三维点云中选择控制点中心")
            return
        try:
            role = "control" if self.control_role.currentText() == "控制点" else "check"
            sigma = (
                (self.control_sigma.value(),) * 3
                if self.control_sigma.value() > 0
                else None
            )
            if self.coordinate_input_mode.currentIndex() == 1:
                self.session.add_geographic_coordinate_observation(
                    point_id=self.control_id.text(),
                    model_xyz=self.current_control_pick["model_xyz"],
                    longitude=self.target_x.value(),
                    latitude=self.target_y.value(),
                    height=self.target_z.value(),
                    role=role,
                    image_name=self.current_control_pick["image_name"],
                    pixel_uv=self.current_control_pick["pixel_uv"],
                    sigma_xyz=sigma,
                    target_crs=self.control_target_crs.text(),
                )
            else:
                if (
                    self.session.coordinate_observations
                    and self.session.coordinate_reference.mode != "cartesian"
                ):
                    raise ValueError("当前工程已有WGS84坐标观测，不能混用工程XYZ")
                self.session.coordinate_reference = CoordinateReference(
                    mode="cartesian",
                    source_crs=self.control_target_crs.text().strip() or "LOCAL_CARTESIAN",
                    target_crs=self.control_target_crs.text().strip() or "LOCAL_CARTESIAN",
                )
                self.session.add_coordinate_observation(
                    point_id=self.control_id.text(),
                    model_xyz=self.current_control_pick["model_xyz"],
                    target_xyz=np.array(
                        [self.target_x.value(), self.target_y.value(), self.target_z.value()]
                    ),
                    role=role,
                    image_name=self.current_control_pick["image_name"],
                    pixel_uv=self.current_control_pick["pixel_uv"],
                    sigma_xyz=sigma,
                    source_crs=self.session.coordinate_reference.source_crs,
                )
        except Exception as exc:
            self._error(str(exc))
            return
        _fill_table(self.coordinate_table, self._coordinate_rows())
        self._save_project(silent=True)
        self._update_project_status()
        self.statusBar().showMessage(
            f"已添加{self.control_role.currentText()} {self.control_id.text()}。",
            7000,
        )

    def _import_coordinate_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入控制点/检查点 CSV",
            str(_resource_root / "docs"),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return
        added = 0
        try:
            with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    point_id = (row.get("point_id") or row.get("id") or "").strip()
                    role_raw = (row.get("role") or "control").strip().lower()
                    role = "check" if role_raw in {"check", "检查点"} else "control"
                    longitude_value = row.get("longitude", row.get("lon", ""))
                    latitude_value = row.get("latitude", row.get("lat", ""))
                    geographic = longitude_value not in ("", None) and latitude_value not in ("", None)
                    target = np.array(
                        [
                            float(
                                longitude_value
                                if geographic
                                else row.get("x", row.get("target_x", ""))
                            ),
                            float(
                                latitude_value
                                if geographic
                                else row.get("y", row.get("target_y", ""))
                            ),
                            float(
                                row.get("height", row.get("altitude", ""))
                                if geographic
                                else row.get("z", row.get("target_z", ""))
                            ),
                        ]
                    )
                    sigma_values = [
                        row.get("sigma_x", ""),
                        row.get("sigma_y", ""),
                        row.get("sigma_z", ""),
                    ]
                    sigma = (
                        tuple(float(value) for value in sigma_values)
                        if all(value not in ("", None) for value in sigma_values)
                        else None
                    )
                    has_model = all(
                        row.get(key, "") not in ("", None)
                        for key in ("model_x", "model_y", "model_z")
                    )
                    if has_model:
                        model = np.array(
                            [float(row["model_x"]), float(row["model_y"]), float(row["model_z"])]
                        )
                        uv = None
                    else:
                        raise ValueError(
                            "新摄影测量模式的 CSV 必须包含 model_x、model_y、model_z；"
                            "也可以直接在三维视图拾点后手工添加"
                        )
                    if geographic:
                        self.session.add_geographic_coordinate_observation(
                            point_id=point_id,
                            model_xyz=model,
                            longitude=float(target[0]),
                            latitude=float(target[1]),
                            height=float(target[2]),
                            role=role,
                            image_name="CSV",
                            pixel_uv=uv,
                            sigma_xyz=sigma,
                            target_crs=(
                                str(row.get("target_crs") or row.get("crs") or "").strip()
                                or self.control_target_crs.text().strip()
                            ),
                        )
                    else:
                        if (
                            self.session.coordinate_observations
                            and self.session.coordinate_reference.mode != "cartesian"
                        ):
                            raise ValueError("CSV不能混用WGS84和工程XYZ坐标")
                        crs = (
                            str(row.get("target_crs") or row.get("crs") or "").strip()
                            or self.control_target_crs.text().strip()
                            or "LOCAL_CARTESIAN"
                        )
                        self.session.coordinate_reference = CoordinateReference(
                            mode="cartesian",
                            source_crs=crs,
                            target_crs=crs,
                        )
                        self.session.add_coordinate_observation(
                            point_id=point_id,
                            model_xyz=model,
                            target_xyz=target,
                            role=role,
                            image_name="CSV",
                            pixel_uv=uv,
                            sigma_xyz=sigma,
                            source_crs=crs,
                        )
                    added += 1
        except Exception as exc:
            self._error(f"CSV 导入失败：{exc}")
            return
        _fill_table(self.coordinate_table, self._coordinate_rows())
        self._save_project(silent=True)
        self._update_project_status()
        self.statusBar().showMessage(f"已导入 {added} 条坐标观测", 6000)

    def _clear_coordinates(self) -> None:
        self.session.coordinate_observations.clear()
        self.session.coordinate_reference = CoordinateReference()
        if self.session.transform.mode == "engineering":
            self.session.transform = SimilarityTransform.identity()
            self.session.calibration_report.clear()
            self.session.clear_processed()
            self._refresh_cloud()
        _fill_table(self.coordinate_table, [])
        self.engineering_report.clear()
        self._save_project(silent=True)
        self._update_project_status()

    def _calibrate_engineering(self) -> None:
        try:
            report = self.session.calibrate_engineering(
                ransac_threshold=self.control_ransac_threshold.value()
            )
            self.engineering_report.setPlainText(_json(report))
            self._refresh_cloud()
            self._save_project(silent=True)
            self._update_project_status()
        except Exception as exc:
            self._error(str(exc))


    def _append_measurement_point(self, point: np.ndarray) -> None:
        self.measurement_points.append(np.asarray(point, dtype=np.float64))
        self._refresh_measurement_selection()

    def _refresh_measurement_selection(self) -> None:
        rows = []
        for index, model_point in enumerate(self.measurement_points, 1):
            point = (
                self.session.transform.apply(model_point)
                if self.session.calibrated
                else model_point
            )
            rows.append([index, *(f"{value:.8f}" for value in point)])
        _fill_table(self.measurement_selected_table, rows)

    def _clear_measurement_points(self) -> None:
        self.measurement_points.clear()
        self.measurement_pixels.clear()
        self._refresh_measurement_selection()

    def _calculate_measurement(self) -> None:
        try:
            self.session.require_metric(self.measurement_kind.currentText())
            points = [
                self.session.transform.apply(point) for point in self.measurement_points
            ]
            value, unit = calculate(
                self.measurement_kind.currentText(),
                points,
                self.first_plane_count.value(),
            )
            measurement = Measurement(
                kind=self.measurement_kind.currentText(),
                value=float(value),
                unit=unit,
                label=(
                    self.measurement_label.text().strip()
                    or f"{self.measurement_kind.currentText()}{len(self.session.measurements) + 1}"
                ),
                point_count=len(points),
            )
            self.session.measurements.append(measurement)
        except Exception as exc:
            self._error(str(exc))
            return
        self.measurement_result.setText(
            f"{measurement.label}：{measurement.value:.10g} {measurement.unit}"
        )
        _fill_table(self.measurement_table, self._measurement_rows())
        self._save_project(silent=True)

    def _clear_measurements(self) -> None:
        self.session.measurements.clear()
        _fill_table(self.measurement_table, [])
        self.measurement_result.setText("尚未计算")
        self._save_project(silent=True)

    def _choose_colmap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 COLMAP",
            str(_resource_root / "tools"),
            "COLMAP (colmap.exe COLMAP.bat);;程序 (*.exe *.bat);;所有文件 (*)",
        )
        if path:
            self.colmap_path.setText(path)

    def _choose_photogrammetry_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择摄影测量工作目录（建议空间充足的SSD）",
            self.photogrammetry_output_root.text()
            or str(self.project_store.root if self.project_store else _projects_root),
        )
        if path:
            self.photogrammetry_output_root.setText(path)

    def _apply_beginner_precision(self, name: str) -> None:
        if name not in RECONSTRUCTION_PRESETS:
            return
        if hasattr(self, "reconstruction_preset"):
            self.reconstruction_preset.setCurrentText(name)
        if hasattr(self, "home_precision"):
            self.home_precision.blockSignals(True)
            self.home_precision.setCurrentText(name)
            self.home_precision.blockSignals(False)

    def _apply_reconstruction_preset(self, name: str) -> None:
        preset = RECONSTRUCTION_PRESETS.get(name)
        if not hasattr(self, "feature_size"):
            return
        if not preset:
            self._refresh_dense_quality_summary()
            return
        self._applying_reconstruction_preset = True
        try:
            self.feature_size.setValue(int(preset["feature_max_image_size"]))
            self.mvs_size.setValue(int(preset["max_image_size"]))
            self.max_features.setValue(int(preset["max_num_features"]))
            self.geometric_consistency.setChecked(
                bool(preset["geometric_consistency"])
            )
            self.patch_match_filter.setChecked(bool(preset["patch_match_filter"]))
            self.patch_match_source_images.setValue(
                int(preset["patch_match_source_images"])
            )
            self.patch_match_iterations.setValue(
                int(preset["patch_match_iterations"])
            )
            self.mvs_reference_strategy.setCurrentIndex(
                0
            )
            self.mvs_reference_ratio.setValue(
                round(float(preset["mvs_reference_ratio"]) * 100)
            )
            self.spatial_blocking.setChecked(True)
            self.spatial_block_target_images.setValue(
                int(preset["spatial_block_target_images"])
            )
        finally:
            self._applying_reconstruction_preset = False
        quality = {
            "快速预览": "低",
            "标准工程模式": "中",
            "高精度模式": "高",
        }[name]
        self.point_quality.blockSignals(True)
        self.point_quality.setCurrentText(quality)
        self.point_quality.blockSignals(False)
        if hasattr(self, "intended_mode"):
            self.intended_mode.blockSignals(True)
            self.intended_mode.setCurrentText(name)
            self.intended_mode.blockSignals(False)
        if hasattr(self, "home_precision"):
            self.home_precision.blockSignals(True)
            self.home_precision.setCurrentText(name)
            self.home_precision.blockSignals(False)
        self._refresh_dense_quality_summary()

    def _current_mvs_parameters_match_preset(self, name: str) -> bool:
        preset = RECONSTRUCTION_PRESETS.get(name)
        if not preset or not hasattr(self, "patch_match_iterations"):
            return False
        return (
            self.feature_size.value() == int(preset["feature_max_image_size"])
            and self.mvs_size.value() == int(preset["max_image_size"])
            and self.max_features.value() == int(preset["max_num_features"])
            and self.geometric_consistency.isChecked()
            == bool(preset["geometric_consistency"])
            and self.patch_match_filter.isChecked()
            == bool(preset["patch_match_filter"])
            and self.patch_match_source_images.value()
            == int(preset["patch_match_source_images"])
            and self.patch_match_iterations.value()
            == int(preset["patch_match_iterations"])
            and str(preset["mvs_reference_strategy"]) == "all"
            and self.mvs_reference_ratio.value()
            == round(float(preset["mvs_reference_ratio"]) * 100)
            and self.spatial_blocking.isChecked()
            and self.spatial_block_target_images.value()
            == int(preset["spatial_block_target_images"])
        )

    def _mark_custom_mvs_parameters(self, *_args: object) -> None:
        if (
            self._applying_reconstruction_preset
            or not hasattr(self, "reconstruction_preset")
        ):
            return
        if self.reconstruction_preset.currentText() != "自定义参数":
            self.reconstruction_preset.blockSignals(True)
            self.reconstruction_preset.setCurrentText("自定义参数")
            self.reconstruction_preset.blockSignals(False)
        self._refresh_dense_quality_summary()

    def _apply_point_quality(self, quality: str) -> None:
        sizes = {"低": 2048, "中": 3072, "高": 4096, "极高": 6000}
        if quality in sizes:
            self.mvs_size.setValue(sizes[quality])
        self._refresh_dense_quality_summary()

    def _refresh_dense_quality_summary(self) -> None:
        if not hasattr(self, "dense_quality_summary"):
            return
        consistency = "开启" if self.geometric_consistency.isChecked() else "关闭"
        filtering = "开启" if self.patch_match_filter.isChecked() else "关闭"
        references = "全部注册照片均为参考帧"
        blocks = (
            f"；≥{self.spatial_block_threshold.value()}张时自动分块，"
            f"每块约{self.spatial_block_target_images.value()}张"
            if self.spatial_blocking.isChecked()
            else "；空间分块关闭"
        )
        self.dense_quality_summary.setText(
            f"{self.reconstruction_preset.currentText()} · "
            f"MVS {self.mvs_size.value()} px · 几何一致性{consistency} · "
            f"过滤{filtering} · 源照片 {self.patch_match_source_images.value()} · "
            f"迭代 {self.patch_match_iterations.value()} · {references}{blocks}"
        )

    def _photogrammetry_config(self, target_stage: str) -> dict[str, Any] | None:
        if self.project_store is None:
            self._error("请先新建或打开工程")
            return None
        image_paths = list(self.selected_images or self.input_images)
        if len(image_paths) < 3:
            self._error("请先导入、检查并选择至少三张有连续重叠的照片")
            return None
        matcher_text = self.matcher.currentText()
        if matcher_text.startswith("全局"):
            matcher = "exhaustive"
        elif matcher_text.startswith("相邻"):
            matcher = "sequential"
        else:
            matcher = "auto"
        config: dict[str, Any] = {
            "task": "colmap",
            "target_stage": target_stage,
            "project_root": str(self.project_store.root),
            "image_paths": image_paths,
            "output_root": (
                self.photogrammetry_output_root.text().strip()
                or str(self.project_store.root / "colmap")
            ),
            "colmap_path": self.colmap_path.text().strip() or None,
            "feature_type": (
                "aliked"
                if not self.feature_method.currentText().startswith("SIFT")
                else "sift_lightglue"
            ),
            "matcher": matcher,
            "mapper": (
                "incremental"
                if self.sfm_mapper.currentText().startswith("COLMAP")
                else "global"
            ),
            "camera_model": self.camera_model.currentText(),
            "single_camera": self.single_camera.isChecked(),
            "feature_max_image_size": self.feature_size.value(),
            "max_image_size": self.mvs_size.value(),
            "max_num_features": self.max_features.value(),
            "sequential_overlap": self.sequential_overlap.value(),
            "geometric_consistency": self.geometric_consistency.isChecked(),
            "patch_match_filter": self.patch_match_filter.isChecked(),
            "patch_match_source_images": self.patch_match_source_images.value(),
            "patch_match_iterations": self.patch_match_iterations.value(),
            "mvs_reference_strategy": "all",
            "mvs_reference_ratio": 1.0,
            "spatial_blocking": self.spatial_blocking.isChecked(),
            "spatial_block_threshold": self.spatial_block_threshold.value(),
            "spatial_block_target_images": self.spatial_block_target_images.value(),
            "spatial_block_halo_ratio": self.spatial_block_halo_ratio.value(),
            "min_num_inliers": self.min_num_inliers.value(),
            "ransac_max_error": self.ransac_max_error.value(),
            "fusion_min_num_pixels": self.fusion_min_views.value(),
            "generate_quality_report": self.generate_quality_report.isChecked(),
            "use_gpu": self.colmap_gpu.isChecked(),
            "resume": self.colmap_resume.isChecked(),
        }
        option_keys = (
            "output_root",
            "colmap_path",
            "feature_type",
            "matcher",
            "mapper",
            "camera_model",
            "single_camera",
            "feature_max_image_size",
            "max_image_size",
            "max_num_features",
            "sequential_overlap",
            "geometric_consistency",
            "patch_match_filter",
            "patch_match_source_images",
            "patch_match_iterations",
            "mvs_reference_strategy",
            "mvs_reference_ratio",
            "spatial_blocking",
            "spatial_block_threshold",
            "spatial_block_target_images",
            "spatial_block_halo_ratio",
            "min_num_inliers",
            "ransac_max_error",
            "fusion_min_num_pixels",
            "generate_quality_report",
            "use_gpu",
            "resume",
        )
        self.session.photogrammetry_options = {
            key: config[key] for key in option_keys
        }
        self.project_store.update_manifest(
            precision_mode=self.reconstruction_preset.currentText()
        )
        self._save_project(silent=True)
        return config

    def _run_sparse(self) -> None:
        config = self._photogrammetry_config("sparse")
        if config is None:
            return
        self._colmap_target_stage = "sparse"
        self._reset_business_stages()
        if self.selected_images:
            self._set_business_stage("photo_scan", "已完成")
        self._run_process_task(
            "照片连接、相机位置与空三优化",
            config,
            self._sparse_finished,
        )

    def _run_dense(self) -> None:
        if not self.session.sparse_result:
            self._error("请先完成空三并检查注册照片、相机位置和稀疏点云")
            self.tabs.setCurrentWidget(self.colmap_tab)
            return
        quality_gate = dict(
            self.session.sparse_result.get("quality_gate") or {}
        )
        gate_status = str(quality_gate.get("status", "review"))
        if gate_status == "blocked":
            self._error(
                "空三质量闸门未通过。请先修复注册率、重投影误差或稀疏覆盖问题，"
                "避免浪费数小时运行稠密重建。"
            )
            self.tabs.setCurrentWidget(self.colmap_tab)
            return
        if gate_status == "review":
            reply = QMessageBox.question(
                self,
                "空三结果需要复核",
                "部分空三质量指标未达到推荐值。请确认相机轨迹和稀疏点云"
                "没有分层、折叠、飞散或错误闭环。\n\n仍然开始稠密重建吗？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        config = self._photogrammetry_config("dense")
        if config is None:
            return
        self._colmap_target_stage = "dense"
        self._run_process_task(
            "原图去畸变、稠密深度与点云融合",
            config,
            self._colmap_finished,
        )

    def _sparse_finished(self, result: dict[str, Any]) -> None:
        previous_folder = str(self.session.sparse_result.get("folder", ""))
        if previous_folder and previous_folder != str(result.get("folder", "")):
            self.session.photogrammetry_result.clear()
            self.session.model_result.clear()
            self.session.transform = SimilarityTransform.identity()
            self.session.distance_constraints.clear()
            self.session.coordinate_observations.clear()
            self.session.measurements.clear()
            self.session.calibration_report.clear()
            self.session.clear_processed()
        self.session.sparse_result = dict(result)
        self._save_project(silent=True)
        self._show_sparse_result(result)
        self.colmap_result.setPlainText(_json(result))
        self._set_business_stage("bundle_adjustment", "已完成")
        try:
            self.cloud_view.load_pointcloud(
                str(result["sparse_pointcloud"]),
                unit="模型单位",
                camera_images_txt=str(result.get("sparse_images_txt", "")),
                label="Sparse BA",
            )
        except Exception as exc:
            _logger.warning("Could not display sparse inspection model: %s", exc)
        self._update_project_status()
        self.statusBar().showMessage(
            "空三完成：请检查注册率、相机位置和稀疏点云，再继续稠密重建。",
            10_000,
        )

    def _show_sparse_result(self, result: dict[str, Any]) -> None:
        if not result:
            self.sparse_registered.setText("—")
            self.sparse_unregistered.setText("—")
            self.sparse_points.setText("—")
            self.sparse_error.setText("—")
            self.sparse_weak.setText("—")
            self.dense_sparse_gate.setText(
                "尚未完成空三。请先在“3 · 空三”中检查注册照片和稀疏模型。"
            )
            self.dense_reference_summary.setText(
                "全部注册照片都会生成MVS深度图，三档模式均固定为100%参考帧。"
            )
            return
        registered = int(result.get("registered_images", 0))
        total = int(result.get("image_count", 0))
        unregistered = list(result.get("unregistered_images") or [])
        error = result.get("mean_reprojection_error_px")
        warnings = list(result.get("warnings") or [])
        quality_gate = dict(result.get("quality_gate") or {})
        gate_status = str(quality_gate.get("status", "review"))
        gate_label = {
            "passed": "质量闸门通过",
            "review": "质量闸门需人工复核",
            "blocked": "质量闸门未通过",
        }.get(gate_status, "质量闸门需人工复核")
        self.sparse_registered.setText(f"{registered} / {total}")
        self.sparse_unregistered.setText(str(len(unregistered)))
        self.sparse_points.setText(f"{int(result.get('sparse_point_count', 0)):,}")
        self.sparse_error.setText(
            f"{float(error):.3f} px" if error is not None else "无统计"
        )
        self.sparse_weak.setText(
            "；".join([gate_label, *warnings])
        )
        self.dense_sparse_gate.setText(
            f"空三已完成（{gate_label}）：注册 {registered}/{total} 张，"
            f"稀疏点 {int(result.get('sparse_point_count', 0)):,}。"
            "请确认右侧相机位置和稀疏几何正确后开始稠密重建。"
        )
        self.dense_reference_summary.setText(
            f"注册照片 {registered} 张；全部 {registered} 张都会生成深度图；"
            "不再通过减少参考帧换取速度。"
        )

    def _reset_business_stages(self) -> None:
        _fill_table(
            self.pipeline_stage_table,
            [[number, label, "等待中"] for _key, number, label in BUSINESS_STAGES],
        )

    def _set_business_stage(self, key: str, status: str) -> None:
        for row, (stage_key, _number, _label) in enumerate(BUSINESS_STAGES):
            if stage_key == key:
                item = self.pipeline_stage_table.item(row, 2)
                if item:
                    item.setText(status)
                    color = {
                        "已完成": "#dff3e4",
                        "进行中": "#fff1c7",
                        "失败": "#ffd9d9",
                    }.get(status)
                    if color:
                        item.setBackground(QBrush(QColor(color)))
                return

    def _run_colmap(self) -> None:
        if self.project_store is None:
            self._error("请先新建或打开工程")
            return
        image_paths = list(self.selected_images or self.input_images)
        if len(image_paths) < 3:
            self._error("请先选择并扫描至少三张照片")
            return
        matcher_text = self.matcher.currentText()
        if matcher_text.startswith("全连接"):
            matcher = "exhaustive"
        elif matcher_text.startswith("顺序"):
            matcher = "sequential"
        else:
            matcher = "auto"
        config = {
            "task": "colmap",
            "project_root": str(self.project_store.root),
            "image_paths": image_paths,
            "output_root": (
                self.photogrammetry_output_root.text().strip()
                or str(self.project_store.root / "colmap")
            ),
            "colmap_path": self.colmap_path.text().strip() or None,
            "feature_type": (
                "aliked"
                if self.feature_method.currentText().startswith("ALIKED")
                else "sift_lightglue"
            ),
            "matcher": matcher,
            "mapper": (
                "global"
                if self.sfm_mapper.currentText().startswith("GLOMAP")
                else "incremental"
            ),
            "camera_model": self.camera_model.currentText(),
            "single_camera": self.single_camera.isChecked(),
            "feature_max_image_size": self.feature_size.value(),
            "max_image_size": self.mvs_size.value(),
            "max_num_features": self.max_features.value(),
            "sequential_overlap": self.sequential_overlap.value(),
            "use_gpu": self.colmap_gpu.isChecked(),
            "resume": self.colmap_resume.isChecked(),
        }
        self.session.photogrammetry_options = {
            key: config[key]
            for key in (
                "output_root",
                "colmap_path",
                "feature_type",
                "matcher",
                "mapper",
                "camera_model",
                "single_camera",
                "feature_max_image_size",
                "max_image_size",
                "max_num_features",
                "sequential_overlap",
                "use_gpu",
                "resume",
            )
        }
        self._save_project(silent=True)
        self._run_process_task(
            "ALIKED/LightGlue + SfM + BA/MVS 独立进程",
            config,
            self._colmap_finished,
        )

    def _colmap_finished(self, result: dict) -> None:
        previous_cloud = str(self.session.photogrammetry_result.get("pointcloud", ""))
        if previous_cloud != str(result["pointcloud"]) and (
            previous_cloud
            or self.session.calibrated
            or self.session.distance_constraints
            or self.session.coordinate_observations
            or self.session.measurements
        ):
            self.session.transform = SimilarityTransform.identity()
            self.session.distance_constraints.clear()
            self.session.coordinate_observations.clear()
            self.session.measurements.clear()
            self.session.calibration_report.clear()
            self.session.clear_processed()
            self.session.model_result.clear()
        self.session.photogrammetry_result = dict(result)
        sparse_snapshot = dict(result)
        for key in (
            "pointcloud",
            "raw_fused",
            "pointcloud_metadata",
            "point_count",
            "unit",
            "estimated_workspace_gb",
            "disk_free_gb_at_start",
        ):
            sparse_snapshot.pop(key, None)
        sparse_snapshot["result_stage"] = "sparse"
        self.session.sparse_result = sparse_snapshot
        self._save_project(silent=True)
        self._refresh_all_tables()
        self._show_sparse_result(self.session.sparse_result)
        self.dense_result.setPlainText(_json(result))
        self.colmap_output.setText(str(result["pointcloud"]))
        for key, _number, _label in BUSINESS_STAGES[:8]:
            self._set_business_stage(key, "已完成")
        try:
            self._refresh_cloud()
        except Exception as exc:
            _logger.warning("Could not display BA/MVS point cloud: %s", exc)
        self._update_project_status()
        self._update_simple_result()
        self.statusBar().showMessage(
            "稠密重建完成，已生成原图高分辨率点云。",
            8000,
        )

    def _update_simple_result(self) -> None:
        if not hasattr(self, "simple_result_title"):
            return
        result = self.session.photogrammetry_result
        if not result:
            self.simple_result_title.setText("尚未生成点云")
            self.simple_result_summary.setText(
                "完成一键重建后，这里会显示照片注册数、点云数量和成果位置。"
            )
            self.simple_result_path.clear()
            self.simple_model_title.setText("尚未生成三维模型")
            self.simple_model_summary.setText(
                "可在一键处理前勾选“纹理三维模型”，也可从已有稠密点云继续生成。"
            )
            self.simple_model_path.clear()
            if hasattr(self, "one_click_status") and not self._one_click_active:
                self.one_click_status.setText(
                    "添加照片并选择成果精度后，点击下方按钮即可。"
                )
            self._update_result_view_controls()
            return
        registered = int(result.get("registered_images", 0))
        image_count = int(result.get("image_count", 0))
        point_count = int(result.get("point_count", 0))
        self.simple_result_title.setText("三维点云已经生成")
        self.simple_result_summary.setText(
            f"成功使用 {registered}/{image_count} 张照片，"
            f"生成 {point_count:,} 个彩色三维点。"
            "当前成果为模型坐标，可直接浏览和导出；只有需要真实米制测量时"
            "才需要在高级功能中添加标尺或控制点。"
        )
        self.simple_result_path.setText(str(result.get("pointcloud", "")))
        model = self.session.model_result
        if self.session.has_model:
            model_formats = model.get("formats") or {}
            delivered = [
                name.upper()
                for name in ("obj", "fbx", "gltf", "glb", "osgb")
                if model_formats.get(name)
            ]
            self.simple_model_title.setText("纹理三维模型已经生成")
            self.simple_model_summary.setText(
                f"模型包含 {int(model.get('vertex_count', 0)):,} 个顶点、"
                f"{int(model.get('face_count', 0)):,} 个三角面；"
                f"{int(model.get('texture_block_count', 1))}个纹理图集；"
                f"已输出：{' / '.join(delivered) or '纹理PLY'}。"
            )
            self.simple_model_path.setText(
                str(model.get("texture_manifest") or model.get("textured_mesh", ""))
            )
        else:
            self.simple_model_title.setText("尚未生成三维模型")
            self.simple_model_summary.setText(
                "点云已经可用。勾选模型格式后，可从当前点云继续生成网格和原始照片纹理。"
            )
            self.simple_model_path.clear()
        if hasattr(self, "one_click_status") and not self._one_click_active:
            self.one_click_status.setText(
                "该项目已经生成点云"
                + ("和纹理模型" if self.session.has_model else "")
                + "，可直接查看成果或重新选择精度运行。"
            )
        self._update_result_view_controls()

    def _refresh_cloud(self) -> None:
        source, transform = self._pointcloud_display_source()
        if not source:
            self.cloud_view.clear()
            self._update_result_view_controls()
            return
        self.cloud_view.load_pointcloud(
            source,
            unit=self.session.unit,
            transform=transform,
            label="Point cloud",
        )
        self._set_result_view_mode("pointcloud")

    def _run_filter(self) -> None:
        if not self.session.has_geometry:
            self._error("请先完成 AI 摄影测量")
            return
        if self.project_store is None:
            self._error("点云过滤需要工程缓存，请先新建或打开工程")
            return
        radius = self.filter_radius.value()
        radius_neighbors = self.filter_radius_neighbors.value()
        if bool(radius) != bool(radius_neighbors):
            self._error("半径过滤需同时设置半径和最少邻居数，或两项都设为关闭")
            return
        self._save_project(silent=True)
        config = {
            "task": "filter",
            "project_root": str(self.project_store.root),
            "options": {
                "distance_mad_multiplier": self.filter_distance_mad.value(),
                "voxel_size": self.filter_voxel.value(),
                "statistical_neighbors": self.filter_neighbors.value(),
                "statistical_std_ratio": self.filter_std_ratio.value(),
                "radius": radius,
                "radius_min_neighbors": radius_neighbors,
                "max_points": 0,
            },
        }
        self._run_process_task("点云过滤与体素降采样", config, self._filter_finished)

    def _filter_finished(self, result: dict) -> None:
        if self.project_store is None:
            return
        self.session = self.project_store.load_session()
        self.filter_result.setPlainText(_json(result.get("report") or self.session.filter_report))
        self._refresh_cloud()
        self._update_project_status()
        self.statusBar().showMessage(
            f"点云清理完成，保留 {result.get('point_count', 0):,} 点",
            8000,
        )

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择成果根目录",
            self.output_dir.text() or str(_outputs_root),
        )
        if path:
            self.output_dir.setText(path)

    def _run_export(self) -> None:
        if not self.session.has_geometry:
            self._error("请先完成 AI 摄影测量")
            return
        output_dir = self.output_dir.text() or str(_outputs_root)
        max_points = self.export_max_points.value() or None
        include_las = self.include_las.isChecked()

        def task(progress):
            progress(0.05, "筛选点云并写入工程文件")
            result = export_project(
                self.session,
                output_dir,
                max_points=max_points,
                include_las=include_las,
            )
            progress(1.0, "成果包与精度报告已生成")
            return result

        self._run_task("成果导出", task, self._export_finished)

    def _export_finished(self, result: dict) -> None:
        self.export_result.setPlainText(_json(result))
        self.export_folder.setText(str(result["folder"]))
        self.simple_result_summary.setText(
            self.simple_result_summary.text()
            + f"\n成果包已导出到：{result['folder']}"
        )
        self.statusBar().showMessage(f"成果包已生成：{result['zip']}", 10_000)

    def _on_cloud_point(self, transformed_point: object) -> None:
        if not self.session.has_geometry:
            return
        transformed = np.asarray(transformed_point, dtype=np.float64)
        model = self.session.transform.inverse().apply(transformed)
        current = (
            self.control_subtabs.currentWidget()
            if self.tabs.currentWidget() is self.control_accuracy_tab
            else self.tabs.currentWidget()
        )
        if current is self.scale_tab:
            self._append_scale_point(model)
            self.statusBar().showMessage(f"已从三维视图选择标尺点：{transformed}", 5000)
        elif current is self.control_tab:
            self.current_control_pick = {
                "model_xyz": model,
                "image_name": "三维视图",
                "pixel_uv": None,
            }
            self._show_control_pick()
        elif current is self.measurement_tab:
            self._append_measurement_point(model)
            self.statusBar().showMessage(f"已选择第 {len(self.measurement_points)} 个测量点", 5000)
        else:
            self.statusBar().showMessage(
                f"三维坐标：({transformed[0]:.6f}, {transformed[1]:.6f}, {transformed[2]:.6f}) "
                f"[{self.session.unit}]",
                7000,
            )

    def _run_task(self, name: str, function, completion) -> None:
        if self.process_task.running:
            self._error("已有独立后台任务正在运行，请等待完成或取消")
            return
        if self._active_task is not None and self._active_task.isRunning():
            self._error("已有后台任务正在运行，请等待完成")
            return
        task = TaskThread(function)
        self._active_task = task
        self._task_completion = completion
        task.progress_changed.connect(self._task_progress)
        task.succeeded.connect(self._task_succeeded)
        task.failed.connect(self._task_failed)
        task.finished.connect(lambda task=task: self._task_thread_finished(task))
        self._set_busy(True, name)
        task.start()

    def _run_process_task(self, name: str, config: dict[str, Any], completion) -> None:
        if self.process_task.running:
            self._error("已有独立后台任务正在运行，请等待完成或取消")
            return
        if self._active_task is not None and self._active_task.isRunning():
            self._error("已有后台任务正在运行，请等待完成")
            return
        if self.project_store is None:
            self._error("独立任务需要一个已保存工程")
            return
        self._process_completion = completion
        self._process_task_kind = str(config.get("task", ""))
        self._set_busy(True, name, cancellable=True)
        try:
            self.process_task.start(
                config,
                self.project_store.root / "cache" / "checkpoints" / "tasks",
            )
            self._interrupted_task_notice = ""
            self._update_project_status()
        except Exception as exc:
            self._process_completion = None
            self._process_task_kind = ""
            self._set_busy(False, "任务启动失败")
            self._update_project_status()
            self._error(str(exc))

    def _process_succeeded(self, result: object) -> None:
        completion = self._process_completion
        self._process_completion = None
        self._process_task_kind = ""
        self._set_busy(False, "完成")
        if completion:
            def continue_pipeline() -> None:
                try:
                    completion(result)
                except Exception as exc:
                    _logger.exception("Process completion handler failed")
                    self._error(str(exc))

            # Continue on the next event-loop turn, after QProcess.finished
            # and its state transition have been fully dispatched.
            QTimer.singleShot(0, continue_pipeline)

    def _process_failed(self, message: str, traceback_text: str) -> None:
        retry_one_click_sparse = (
            self._one_click_active
            and self._process_task_kind == "colmap"
            and self._colmap_target_stage == "sparse"
            and not self._one_click_retry_used
        )
        self._process_completion = None
        self._record_process_terminal("failed", message)
        self._process_task_kind = ""
        _logger.error("Independent worker failed:\n%s", traceback_text)
        self._set_busy(False, "任务失败")
        if retry_one_click_sparse:
            self._one_click_retry_used = True
            self.feature_method.setCurrentIndex(1)
            self.sfm_mapper.setCurrentIndex(1)
            self.matcher.setCurrentIndex(0)
            self.one_click_status.setText(
                "推荐算法未能完成，正在自动切换稳定算法重新计算…"
            )
            QTimer.singleShot(0, self._start_one_click_sparse)
            return
        if self._one_click_active:
            self._one_click_active = False
            concise_message = message.splitlines()[0].strip()
            self.one_click_status.setText(
                f"自动处理失败：{concise_message}"
            )
        for key, _number, _label in BUSINESS_STAGES:
            item = next(
                (
                    self.pipeline_stage_table.item(row, 2)
                    for row, (stage_key, _n, _l) in enumerate(BUSINESS_STAGES)
                    if stage_key == key
                ),
                None,
            )
            if item and item.text() == "进行中":
                self._set_business_stage(key, "失败")
        self._update_project_status()
        self._error(message)

    def _process_cancelled(self, message: str) -> None:
        self._process_completion = None
        self._record_process_terminal("cancelled", message or "用户取消任务")
        self._process_task_kind = ""
        self._set_busy(False, "任务已取消")
        if self._one_click_active:
            self._one_click_active = False
            self.one_click_status.setText("任务已取消，可以稍后重新开始。")
        self._update_project_status()
        self.statusBar().showMessage(message or "任务已取消", 7000)

    def _record_process_terminal(self, status: str, message: str) -> None:
        if self.project_store is None:
            return
        stage = {
            "scan_photos": "photo_scan",
            "filter": "point_filter",
            "model": "textured_model",
        }.get(self._process_task_kind)
        if self._process_task_kind == "colmap":
            stage = (
                "sparse_ba"
                if self._colmap_target_stage == "sparse"
                else "colmap_mvs"
            )
        if stage:
            self.project_store.stage_tracker.set(stage, status, message=message)

    def _process_status(self, status: str, message: str) -> None:
        if message:
            self.progress_label.setText(message)
        if status == "cancelling":
            self.cancel_button.setEnabled(False)

    def _gpu_telemetry(self, telemetry: object) -> None:
        values = dict(telemetry) if isinstance(telemetry, dict) else {}
        used = values.get("gpu_memory_used_gb")
        total = values.get("gpu_memory_total_gb")
        utilization = values.get("gpu_utilization_percent")
        allocated = values.get("process_cuda_allocated_gb")
        if used is None or total is None:
            self.gpu_label.setText("GPU：状态不可用")
            return
        text = f"GPU：{used:.2f}/{total:.2f} GB"
        if utilization is not None:
            text += f" · {utilization:.0f}%"
        if allocated is not None:
            text += f" · 本任务 {allocated:.2f} GB"
        self.gpu_label.setText(text)

    def _task_progress(self, value: int, text: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)
        self.statusBar().showMessage(text)
        if self._one_click_active:
            self.one_click_status.setText(text)
        if self._process_task_kind in {"colmap", "model"}:
            stage_key = self._business_stage_from_text(text)
            if stage_key:
                current_index = next(
                    index
                    for index, item in enumerate(BUSINESS_STAGES)
                    if item[0] == stage_key
                )
                for index, (key, _number, _label) in enumerate(BUSINESS_STAGES):
                    if index < current_index:
                        self._set_business_stage(key, "已完成")
                    elif index == current_index:
                        self._set_business_stage(key, "进行中")
            self._update_project_status()

    @staticmethod
    def _business_stage_from_text(text: str) -> str | None:
        mappings = (
            ("点云去噪", "point_conditioning"),
            ("法向修复", "point_conditioning"),
            ("表面重建", "surface_reconstruction"),
            ("三角网格", "surface_reconstruction"),
            ("网格修补", "mesh_repair"),
            ("统一法向", "mesh_repair"),
            ("网格简化", "mesh_repair"),
            ("UV展开", "texture_mapping"),
            ("接缝融合", "texture_mapping"),
            ("纹理", "texture_mapping"),
            ("模型格式", "model_export"),
            ("写出OBJ", "model_export"),
            ("检查照片", "photo_scan"),
            ("特征提取", "ai_feature_extraction"),
            ("特征匹配", "ai_feature_matching"),
            ("照片连接", "ai_feature_matching"),
            ("SfM", "sparse_mapping"),
            ("相机位置", "sparse_mapping"),
            ("Bundle", "bundle_adjustment"),
            ("空三", "bundle_adjustment"),
            ("去畸变", "image_undistortion"),
            ("高精度MVS工作区", "image_undistortion"),
            ("空间块", "patch_match"),
            ("PatchMatch", "patch_match"),
            ("稠密深度", "patch_match"),
            ("融合", "stereo_fusion"),
            ("点云", "stereo_fusion"),
        )
        return next((stage for phrase, stage in mappings if phrase in text), None)

    def _task_succeeded(self, result: object) -> None:
        completion = self._task_completion
        self._set_busy(False, "完成")
        if completion:
            try:
                completion(result)
            except Exception as exc:
                _logger.exception("GUI completion handler failed")
                self._error(str(exc))

    def _task_failed(self, message: str, traceback_text: str) -> None:
        _logger.error("Background task failed:\n%s", traceback_text)
        self._set_busy(False, "任务失败")
        self._error(message)

    def _task_thread_finished(self, task: TaskThread) -> None:
        task.deleteLater()
        if self._active_task is task:
            self._active_task = None
            self._task_completion = None

    def _set_busy(self, busy: bool, text: str, *, cancellable: bool = False) -> None:
        self.tabs.setEnabled(not busy)
        self.progress_label.setText(text)
        self.cancel_button.setEnabled(bool(busy and cancellable))
        if not busy:
            self.progress_bar.setValue(100 if text == "完成" else 0)


    def _distance_rows(self) -> list[list[Any]]:
        return [
            [
                item.label,
                item.source,
                f"{item.model_distance:.8f}",
                f"{item.actual_distance_m:.8f}",
                f"{item.actual_distance_m / item.model_distance:.10f}",
            ]
            for item in self.session.distance_constraints
        ]

    def _coordinate_rows(self) -> list[list[Any]]:
        return [
            [
                item.point_id,
                "控制点" if item.role == "control" else "检查点",
                item.image_name,
                *(f"{value:.8f}" for value in item.model_xyz),
                *(f"{value:.8f}" for value in item.target_xyz),
            ]
            for item in self.session.coordinate_observations
        ]

    def _measurement_rows(self) -> list[list[Any]]:
        return [
            [item.label, item.kind, f"{item.value:.10g}", item.unit, item.point_count]
            for item in self.session.measurements
        ]

    def _refresh_all_tables(self) -> None:
        _fill_table(self.distance_table, self._distance_rows())
        _fill_table(self.coordinate_table, self._coordinate_rows())
        _fill_table(self.measurement_table, self._measurement_rows())
        self._refresh_scale_selection()
        self._refresh_measurement_selection()

    def _update_project_status(self) -> None:
        project_prefix = (
            f"工程：{self.project_store.root}　｜　" if self.project_store is not None else "未打开工程　｜　"
        )
        photogrammetry = self.session.photogrammetry_result
        colmap_running = (
            self.process_task.running and self._process_task_kind == "colmap"
        )
        model_running = (
            self.process_task.running and self._process_task_kind == "model"
        )
        if model_running:
            progress_text = self.progress_label.text().strip() or "正在准备"
            text = project_prefix + f"纹理模型生成中　｜　{progress_text}"
            mode = self.session.transform.mode
        elif colmap_running:
            progress_text = self.progress_label.text().strip() or "正在准备"
            if self._colmap_target_stage == "dense":
                sparse = self.session.sparse_result
                registered_count = int(sparse.get("registered_images", 0))
                reference_count = registered_count
                text = project_prefix + (
                    f"稠密重建进行中　｜　{progress_text}　｜　"
                    f"MVS参考图 {reference_count}/{registered_count} 张"
                )
            else:
                text = project_prefix + f"空三计算进行中　｜　{progress_text}"
            mode = "preview"
        elif photogrammetry:
            labels = {
                "preview": "AI摄影测量点云（任意尺度；禁止米制测量）",
                "scaled": "米制点云（已恢复统一尺度）",
                "engineering": "工程坐标点云（已用控制点转换）",
            }
            mode = self.session.transform.mode
            processed = (
                f"　｜　已处理点 {len(self.session.processed_points):,}"
                if self.session.has_processed_cloud and self.session.processed_points is not None
                else ""
            )
            text = project_prefix + (
                f"{labels[mode]}　｜　注册照片 "
                f"{photogrammetry.get('registered_images', 0)}/"
                f"{photogrammetry.get('image_count', 0)} 张　｜　"
                f"稠密点 {int(photogrammetry.get('point_count', 0)):,}　｜　"
                f"标尺 {len(self.session.distance_constraints)} 个　｜　"
                f"坐标观测 {len(self.session.coordinate_observations)} 个{processed}"
                + (
                    f"　｜　纹理模型 {int(self.session.model_result.get('face_count', 0)):,} 面"
                    if self.session.has_model
                    else ""
                )
            )
        elif self.session.sparse_result:
            sparse = self.session.sparse_result
            text = project_prefix + (
                "空三已完成，等待确认稠密重建　｜　注册照片 "
                f"{sparse.get('registered_images', 0)}/"
                f"{sparse.get('image_count', 0)} 张　｜　"
                f"稀疏点 {int(sparse.get('sparse_point_count', 0)):,}"
            )
            mode = "preview"
        else:
            text = project_prefix + "尚未生成点云｜请先导入并检查照片，再开始空三。"
            mode = "preview"
        if self._interrupted_task_notice and not self.process_task.running:
            detail = text.removeprefix(project_prefix)
            text = (
                project_prefix
                + self._interrupted_task_notice
                + "　｜　"
                + detail
            )
        self._refresh_project_home()
        self.project_status.setProperty("mode", mode)
        self.project_status.setText(text)
        self.project_status.style().unpolish(self.project_status)
        self.project_status.style().polish(self.project_status)

    def _open_path(self, value: str | Path) -> None:
        if not value:
            self._error("尚无可打开的路径")
            return
        path = Path(value)
        if not path.exists():
            self._error(f"路径不存在：{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "AI 摄影测量工程点云工作台", message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.process_task.running:
            answer = QMessageBox.question(
                self,
                "后台任务运行中",
                "退出会取消当前 CUDA/COLMAP 任务。是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process_task.cancel()
        if self._active_task is not None and self._active_task.isRunning():
            QMessageBox.warning(self, "后台任务运行中", "请等待当前 CUDA/COLMAP 任务完成后再退出。")
            event.ignore()
            return
        self._save_project(silent=True)
        self.cloud_view.close()
        event.accept()


APP_STYLE = """
QMainWindow, QWidget { font-family: "Microsoft YaHei UI"; font-size: 10pt; }
QMainWindow { background: #edf2f3; }
QGroupBox {
    font-weight: 600; border: 1px solid #b8c5c8; border-radius: 7px;
    margin-top: 9px; padding-top: 10px; background: #f8fafb;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton {
    min-height: 29px; padding: 2px 10px; border: 1px solid #91a3a7;
    border-radius: 5px; background: #ffffff;
}
QPushButton:hover { background: #e4f2f0; border-color: #0b7068; }
QPushButton#primaryButton {
    min-height: 34px; color: white; font-weight: 700;
    background: #0b7068; border: 1px solid #07534e;
}
QPushButton#primaryButton:hover { background: #0f897f; }
QPushButton#resultViewButton {
    min-width: 120px; min-height: 36px; font-weight: 700;
    color: #25454a; background: #eef4f5; border: 1px solid #91a3a7;
}
QPushButton#resultViewButton:checked {
    color: white; background: #0b7068; border-color: #07534e;
}
QPushButton#resultViewButton:disabled {
    color: #8b999c; background: #e5eaeb; border-color: #c6d0d2;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
    background: white; border: 1px solid #aebdc0; border-radius: 4px;
    selection-background-color: #0b7068;
}
QHeaderView::section { background: #dfe9ea; padding: 5px; border: 0; border-right: 1px solid #c3ced0; }
QTabWidget::pane { border: 1px solid #aebdc0; background: #f6f8f9; }
QTabBar::tab { padding: 8px 12px; background: #dfe7e8; }
QTabBar::tab:selected { color: white; background: #0b7068; }
QLabel#projectStatus {
    padding: 10px 14px; color: white; font-size: 11pt; font-weight: 700;
    background: #52636a; border-radius: 7px;
}
QLabel#projectStatus[mode="scaled"] { background: #246b86; }
QLabel#projectStatus[mode="engineering"] { background: #0b7068; }
QLabel#warningLabel { padding: 8px; background: #fff0c7; border-left: 4px solid #d99400; }
QLabel#resultLabel { padding: 10px; background: #dff2ef; color: #07534e; font-size: 13pt; font-weight: 700; }
QLabel#homeTitle { color: #083f3b; font-size: 20pt; font-weight: 800; padding: 8px 0 2px 0; }
QLabel#homeSubtitle { color: #52636a; font-size: 11pt; padding-bottom: 8px; }
QLabel#resultViewStatus { color: #52636a; font-weight: 600; padding-left: 10px; }
QLabel#brandMark {
    color: #0b7068; font-size: 10pt; font-weight: 800;
    padding: 2px 10px 2px 14px; border-left: 1px solid #b8c5c8;
}
QLabel#dropZone {
    min-height: 48px; color: #0b7068; background: #e6f4f2;
    border: 2px dashed #65a9a3; border-radius: 8px;
}
QLabel#statCaption { color: #607178; font-size: 9pt; }
QLabel#statValue { color: #123d3a; font-size: 13pt; font-weight: 750; padding: 4px; }
QProgressBar { min-height: 18px; text-align: center; }
QProgressBar::chunk { background: #0b7068; }
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI 摄影测量 PySide6 工程点云工作台")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="创建主窗口后自动退出，用于安装检查",
    )
    args = parser.parse_args(argv)
    startup_trace = print if args.smoke_test else (lambda *_args, **_kwargs: None)
    startup_trace("startup: configuring logs", flush=True)
    log_dir = _logs_root
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "desktop.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    os.environ.setdefault("QT_API", "pyside6")
    startup_trace("startup: creating QApplication", flush=True)
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("AI 摄影测量工程点云工作台")
    application.setOrganizationName("AI Photogrammetry Engineering")
    application.setStyle("Fusion")
    application.setStyleSheet(APP_STYLE)
    startup_trace("startup: creating main window", flush=True)
    window = EngineeringMainWindow()
    startup_trace("startup: showing main window", flush=True)
    window.show()
    if args.smoke_test:
        QTimer.singleShot(700, application.quit)
    startup_trace("startup: entering event loop", flush=True)
    return application.exec()
