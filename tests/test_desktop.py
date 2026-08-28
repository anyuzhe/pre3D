import json
import os
from unittest.mock import PropertyMock, patch

import numpy as np

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QProcess  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ai_photogrammetry.engineering.desktop import (  # noqa: E402
    EngineeringMainWindow,
    _sparse_requires_stable_retry,
)
from ai_photogrammetry.engineering.desktop_widgets import (  # noqa: E402
    ProcessTaskController,
    _colmap_camera_poses,
)
from ai_photogrammetry.engineering.project_store import ProjectStore  # noqa: E402


def test_native_window_contains_all_workflow_tabs():
    application = QApplication.instance() or QApplication([])
    window = EngineeringMainWindow()

    assert window.tabs.count() == 6
    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "1 · 项目",
        "2 · 一键重建",
        "3 · 空三",
        "4 · 稠密重建",
        "5 · 控制与精度",
        "3 · 成果",
    ]
    assert [window.tabs.isTabVisible(index) for index in range(6)] == [
        True,
        True,
        False,
        False,
        False,
        True,
    ]
    window.advanced_ui_action.setChecked(True)
    assert all(window.tabs.isTabVisible(index) for index in range(6))
    assert window.tabs.tabText(5) == "6 · 成果"
    window.advanced_ui_action.setChecked(False)
    assert [window.settings_tabs.tabText(index) for index in range(3)] == [
        "简易模式",
        "工程模式",
        "专家模式",
    ]
    assert window.control_subtabs.count() == 3
    assert window.pipeline_stage_table.rowCount() == 13
    assert window.output_dense_cloud.isChecked()
    assert not window.output_textured_model.isChecked()
    assert window.brand_mark.text() == "岩创科技"
    assert window.brand_mark.isVisibleTo(window.statusBar())
    window.output_textured_model.setChecked(True)
    assert window.output_model_obj.isEnabled()
    assert window.output_model_fbx.isEnabled()
    assert window.output_model_gltf.isEnabled()
    window.reconstruction_preset.setCurrentText("快速预览")
    assert window.feature_size.value() == 2048
    assert window.mvs_size.value() == 2048
    assert not window.geometric_consistency.isChecked()
    assert window.patch_match_filter.isChecked()
    assert window.patch_match_source_images.value() == 8
    assert window.patch_match_iterations.value() == 3
    assert window.mvs_reference_strategy.currentIndex() == 0
    assert window.mvs_reference_ratio.value() == 100
    window.reconstruction_preset.setCurrentText("标准工程模式")
    assert window.mvs_size.value() == 3072
    assert window.geometric_consistency.isChecked()
    assert window.patch_match_source_images.value() == 12
    assert window.patch_match_iterations.value() == 4
    assert window.mvs_reference_strategy.currentIndex() == 0
    assert window.mvs_reference_ratio.value() == 100
    window.reconstruction_preset.setCurrentText("高精度模式")
    assert window.mvs_size.value() == 4096
    assert window.max_features.value() == 8192
    assert window.patch_match_source_images.value() == 18
    assert window.patch_match_iterations.value() == 5
    assert window.mvs_reference_strategy.currentIndex() == 0
    assert window.mvs_reference_ratio.value() == 100
    window.patch_match_iterations.setValue(6)
    assert window.reconstruction_preset.currentText() == "自定义参数"
    window.close()
    application.processEvents()


def test_result_page_switches_between_pointcloud_and_textured_model(
    tmp_path,
    monkeypatch,
):
    application = QApplication.instance() or QApplication([])
    window = EngineeringMainWindow()
    pointcloud = tmp_path / "dense.ply"
    mesh = tmp_path / "textured_mesh.ply"
    texture = tmp_path / "texture.png"
    for path in (pointcloud, mesh, texture):
        path.write_bytes(b"test")
    window.session.photogrammetry_result = {
        "pointcloud": str(pointcloud),
        "registered_images": 12,
        "image_count": 12,
        "point_count": 1234,
    }
    window.session.model_result = {
        "textured_mesh": str(mesh),
        "texture_atlas": str(texture),
        "vertex_count": 100,
        "face_count": 180,
    }
    displayed = []
    monkeypatch.setattr(
        window.cloud_view,
        "load_pointcloud",
        lambda path, **kwargs: displayed.append(("pointcloud", path)),
    )
    monkeypatch.setattr(
        window.cloud_view,
        "load_model",
        lambda path, texture_path, **kwargs: displayed.append(("model", path)),
    )

    window._update_simple_result()

    assert window.result_pointcloud_button.isEnabled()
    assert window.result_model_button.isEnabled()
    assert window.result_pointcloud_button.isChecked()
    window.result_model_button.click()
    assert displayed[-1] == ("model", str(mesh))
    assert window.result_model_button.isChecked()
    assert not window.result_pointcloud_button.isChecked()
    assert "3D 模型" in window.result_view_status.text()
    window.result_pointcloud_button.click()
    assert displayed[-1] == ("pointcloud", str(pointcloud))
    assert window.result_pointcloud_button.isChecked()
    assert not window.result_model_button.isChecked()
    assert "彩色点云" in window.result_view_status.text()
    window.close()
    application.processEvents()


def test_result_page_uses_multi_atlas_viewer_when_blocks_exist(
    tmp_path,
    monkeypatch,
):
    application = QApplication.instance() or QApplication([])
    window = EngineeringMainWindow()
    blocks = []
    for index in range(2):
        mesh = tmp_path / f"mesh_{index}.ply"
        texture = tmp_path / f"texture_{index}.png"
        mesh.write_bytes(b"mesh")
        texture.write_bytes(b"texture")
        blocks.append({"mesh": str(mesh), "texture": str(texture)})
    window.session.model_result = {"texture_blocks": blocks}
    displayed = []
    monkeypatch.setattr(
        window.cloud_view,
        "load_models",
        lambda selected, **kwargs: displayed.append(selected),
    )

    window._show_textured_model()

    assert displayed == [blocks]
    assert window.result_model_button.isChecked()
    window.close()
    application.processEvents()


def test_beginner_one_click_starts_automatic_photo_scan(
    tmp_path,
    monkeypatch,
):
    application = QApplication.instance() or QApplication([])
    store = ProjectStore.create(tmp_path / "project", "一键测试")
    window = EngineeringMainWindow()
    window._activate_project(store)
    window.input_images = [
        str(tmp_path / "01.jpg"),
        str(tmp_path / "02.jpg"),
        str(tmp_path / "03.jpg"),
    ]
    calls = {}

    def capture(completion, *, beginner):
        calls["completion"] = completion
        calls["beginner"] = beginner

    monkeypatch.setattr(window, "_start_photo_scan", capture)
    window.intended_mode.setCurrentText("快速预览")

    window._run_one_click()

    assert window._one_click_active
    assert calls["beginner"] is True
    assert calls["completion"] == window._one_click_photo_scan_finished
    assert window.reconstruction_preset.currentText() == "快速预览"
    window.close()
    application.processEvents()


def test_beginner_one_click_retries_sparse_failure_with_stable_route(
    monkeypatch,
):
    application = QApplication.instance() or QApplication([])
    window = EngineeringMainWindow()
    window._one_click_active = True
    window._one_click_retry_used = False
    window._process_task_kind = "colmap"
    window._colmap_target_stage = "sparse"
    retries = []
    monkeypatch.setattr(
        window,
        "_start_one_click_sparse",
        lambda: retries.append(True),
    )

    window._process_failed("GLOMAP failed", "traceback")
    application.processEvents()

    assert window._one_click_active
    assert window._one_click_retry_used
    assert window.feature_method.currentIndex() == 1
    assert window.sfm_mapper.currentIndex() == 1
    assert retries == [True]
    window.close()
    application.processEvents()


def test_usable_review_sparse_result_does_not_trigger_full_retry():
    project_16_style = {
        "registered_images": 503,
        "image_count": 565,
        "registration_ratio": 503 / 565,
        "sparse_point_count": 95_610,
        "mean_reprojection_error_px": 0.885,
        "quality_gate": {"status": "review"},
    }
    weak_result = {
        "registered_images": 350,
        "image_count": 565,
        "registration_ratio": 350 / 565,
        "sparse_point_count": 40_000,
        "mean_reprojection_error_px": 1.0,
        "quality_gate": {"status": "review"},
    }

    assert not _sparse_requires_stable_retry(project_16_style)
    assert _sparse_requires_stable_retry(weak_result)


def test_open_project_reports_recovered_interrupted_stage(tmp_path):
    application = QApplication.instance() or QApplication([])
    store = ProjectStore.create(tmp_path / "interrupted", "中断恢复")
    store.stage_tracker.set("sparse_ba", "running", message="空三计算")
    window = EngineeringMainWindow()

    window._activate_project(store)

    recovered = ProjectStore.open(store.root).stage_tracker
    assert recovered.status("sparse_ba") == "interrupted"
    assert "检测到上次任务中断" in window.project_status.text()
    assert "断点继续" in window.one_click_status.text()
    window.close()
    application.processEvents()


def test_process_result_is_emitted_only_after_worker_has_finished():
    application = QApplication.instance() or QApplication([])
    controller = ProcessTaskController()
    results = []
    controller.succeeded.connect(results.append)

    controller._handle_line(
        json.dumps(
            {
                "type": "result",
                "status": "completed",
                "result": {"selected_images": ["01.jpg", "02.jpg", "03.jpg"]},
            }
        )
    )

    assert results == []
    controller._finished(0, QProcess.ExitStatus.NormalExit)
    assert results == [
        {"selected_images": ["01.jpg", "02.jpg", "03.jpg"]}
    ]
    controller.deleteLater()
    application.processEvents()


def test_process_completion_continues_on_next_event_loop_turn():
    application = QApplication.instance() or QApplication([])
    window = EngineeringMainWindow()
    results = []
    window._process_completion = results.append

    window._process_succeeded({"stage": "photo_scan"})

    assert results == []
    application.processEvents()
    assert results == [{"stage": "photo_scan"}]
    window.close()
    application.processEvents()


def test_colmap_camera_pose_preview_parser(tmp_path):
    images = tmp_path / "images.txt"
    images.write_text(
        "1 1 0 0 0 1 2 3 1 image.jpg\n"
        "10 10 -1\n",
        encoding="utf-8",
    )

    centers, directions = _colmap_camera_poses(images)

    np.testing.assert_allclose(centers, [[-1, -2, -3]])
    np.testing.assert_allclose(directions, [[0, 0, 1]])


def test_new_project_precision_preset_is_applied_when_reopened(tmp_path):
    application = QApplication.instance() or QApplication([])
    store = ProjectStore.create(
        tmp_path / "high_precision",
        "高精度工程",
        precision_mode="高精度模式",
    )
    window = EngineeringMainWindow()

    window._activate_project(store)

    assert window.reconstruction_preset.currentText() == "高精度模式"
    assert window.mvs_size.value() == 4096
    assert window.patch_match_source_images.value() == 18
    assert window.patch_match_iterations.value() == 5
    window.close()
    application.processEvents()


def test_pytest_projects_are_never_added_to_recent_projects(tmp_path):
    application = QApplication.instance() or QApplication([])
    store = ProjectStore.create(
        tmp_path / "pytest-123" / "synthetic",
        "自动化测试工程",
    )
    window = EngineeringMainWindow()

    window._activate_project(store)

    assert window._recent_project_paths() == []
    assert window.recent_projects_table.rowCount() == 0
    window.close()
    application.processEvents()


def test_exif_orientation_information_does_not_mark_photo_for_review():
    application = QApplication.instance() or QApplication([])
    window = EngineeringMainWindow()
    window._load_photo_scan(
        {
            "records": [
                {
                    "path": "portrait.jpg",
                    "name": "portrait.jpg",
                    "size_bytes": 1,
                    "modified_ns": 1,
                    "width": 4032,
                    "height": 3024,
                    "orientation": 6,
                    "valid": True,
                    "selected": True,
                    "warning": "需按 EXIF 方向 6 纠正",
                }
            ],
            "summary": {"photo_count": 1, "valid_count": 1},
        }
    )

    assert window.photo_table.item(0, 1).text() == "合格"
    assert window.photo_table.item(0, 10).text() == (
        "显示方向已自动顺时针旋转90°（正常）"
    )
    window.close()
    application.processEvents()


def test_dense_run_status_shows_reference_view_progress():
    application = QApplication.instance() or QApplication([])
    window = EngineeringMainWindow()
    window.session.sparse_result = {
        "registered_images": 83,
        "image_count": 83,
        "sparse_point_count": 27_897,
        "mvs_reference_selection": {
            "reference_image_count": 63,
            "helper_source_image_count": 20,
        },
    }
    window._process_task_kind = "colmap"
    window._colmap_target_stage = "dense"
    window.progress_label.setText("PatchMatch 几何一致性：17/83 张参考图")

    with patch.object(
        type(window.process_task),
        "running",
        new_callable=PropertyMock,
        return_value=True,
    ):
        window._update_project_status()

    status = window.project_status.text()
    assert "稠密重建进行中" in status
    assert "17/83 张参考图" in status
    assert "MVS参考图 83/83 张" in status
    window.close()
    application.processEvents()
