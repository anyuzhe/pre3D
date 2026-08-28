from pathlib import Path

import numpy as np
import pytest

from ai_photogrammetry.engineering.exporters import export_project, write_binary_ply
from ai_photogrammetry.engineering.session import ProjectSession


def synthetic_session(tmp_path: Path) -> tuple[ProjectSession, np.ndarray]:
    session = ProjectSession(project_name="测试项目", project_id="testproject")
    colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cloud = tmp_path / "dense.ply"
    write_binary_ply(cloud, points, colors, ["test cloud"])
    session.photogrammetry_result = {
        "pointcloud": str(cloud),
        "image_count": 1,
        "registered_images": 1,
        "point_count": len(points),
        "unit": "模型单位",
    }
    return session, points


def test_metric_gate_and_scale(tmp_path: Path):
    session, points = synthetic_session(tmp_path)
    with pytest.raises(RuntimeError, match="真实尺度"):
        session.require_metric("距离")
    session.add_distance_constraint("one", np.array([0, 0, 0]), np.array([1, 0, 0]), 2.0)
    report = session.calibrate_scale()
    session.require_metric("距离")
    assert session.transform.apply(points)[1, 0] == pytest.approx(2.0)
    assert report["scale"] == pytest.approx(2.0)


def test_preview_export_has_explicit_nonmetric_outputs(tmp_path: Path):
    session, _ = synthetic_session(tmp_path)
    result = export_project(session, tmp_path, include_las=True)
    assert Path(result["ply"]).is_file()
    assert Path(result["zip"]).is_file()
    assert "las" not in result
    with Path(result["ply"]).open("rb") as stream:
        header = stream.read(512).split(b"end_header")[0]
    assert b"property double x" in header
    report = Path(result["report_json"]).read_text(encoding="utf-8")
    assert '"calibration_mode": "preview"' in report


def test_large_engineering_coordinates_retain_small_differences(tmp_path: Path):
    session, points = synthetic_session(tmp_path)
    session.add_coordinate_observation(
        point_id="A",
        model_xyz=np.array([0.0, 0.0, 1.0]),
        target_xyz=np.array([500000.0, 3300000.0, 1250.0]),
        role="control",
    )
    session.add_coordinate_observation(
        point_id="B",
        model_xyz=np.array([1.0, 0.0, 1.0]),
        target_xyz=np.array([500000.01, 3300000.0, 1250.0]),
        role="control",
    )
    session.add_coordinate_observation(
        point_id="C",
        model_xyz=np.array([0.0, 1.0, 1.0]),
        target_xyz=np.array([500000.0, 3300000.01, 1250.0]),
        role="control",
    )
    session.calibrate_engineering()
    transformed = session.transform.apply(points)
    assert transformed.dtype == np.float64
    assert transformed[1, 0] - transformed[0, 0] == pytest.approx(0.01)


def test_engineering_report_warns_about_weak_vertical_distribution(tmp_path: Path):
    session, _ = synthetic_session(tmp_path)
    for point_id, model_xyz in {
        "A": [0.0, 0.0, 1.0],
        "B": [0.1, 0.0, 1.0],
        "C": [0.0, 0.1, 1.0],
    }.items():
        session.add_coordinate_observation(
            point_id=point_id,
            model_xyz=np.asarray(model_xyz),
            target_xyz=np.asarray(model_xyz),
            role="control",
        )

    report = session.calibrate_engineering()

    assert any("高程分布" in warning for warning in report["warnings"])
