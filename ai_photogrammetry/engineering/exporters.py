"""Point-cloud, calibration observations, measurements and report export."""

from __future__ import annotations

import csv
import html
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .pointcloud_io import load_ply_preview, load_ply_vertices_colors
from .session import ProjectSession


def _safe_name(value: str) -> str:
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    return normalized.strip(" .") or "project"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def _filtered_cloud(
    session: ProjectSession,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    session.require_geometry()
    if session.has_processed_cloud:
        assert session.processed_points is not None
        assert session.processed_colors is not None
        points = session.processed_points
        colors = session.processed_colors
        indices = np.arange(len(points))
        if max_points is not None and len(indices) > max_points:
            indices = np.sort(
                np.random.default_rng(42).choice(indices, int(max_points), replace=False)
            )
        return points[indices], colors[indices]
    source = str(session.photogrammetry_result.get("pointcloud", "")).strip()
    if not source:
        raise ValueError("工程中没有可导出的 AI 摄影测量点云")
    if max_points is None:
        points, colors = load_ply_vertices_colors(source)
    else:
        points, colors, _total = load_ply_preview(
            source,
            max_points=max_points,
        )
    points = session.transform.apply(points)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        raise ValueError("没有可导出的有限三维点")
    indices = np.flatnonzero(finite)
    if max_points is not None and len(indices) > max_points:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(indices, int(max_points), replace=False))
    return points[indices], colors[indices]


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray, comments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        *[f"comment {line}" for line in comments],
        f"element vertex {len(points)}",
        "property double x",
        "property double y",
        "property double z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]
    dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.astype(np.float64).T
    vertices["red"], vertices["green"], vertices["blue"] = colors.astype(np.uint8).T
    with path.open("wb") as stream:
        # PLY headers are ASCII by specification. Full Unicode metadata is
        # retained in accuracy_report.json; replace only non-ASCII header text.
        stream.write("\n".join(header_lines).encode("ascii", errors="replace"))
        vertices.tofile(stream)


def write_las(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    crs: str = "",
) -> None:
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError("导出 LAS 需要安装 laspy") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    if crs and crs.upper() not in {"LOCAL_CARTESIAN", "LOCAL_ENU"}:
        try:
            from pyproj import CRS

            header.add_crs(CRS.from_user_input(crs))
        except Exception as exc:
            raise RuntimeError(f"无法把坐标系 {crs} 写入LAS：{exc}") from exc
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(points.min(axis=0))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = points[:, 0], points[:, 1], points[:, 2]
    rgb16 = colors.astype(np.uint16) * 257
    cloud.red, cloud.green, cloud.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    cloud.write(path)


def _write_constraints_csv(session: ProjectSession, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "kind",
                "id",
                "role_or_source",
                "image",
                "pixel_u",
                "pixel_v",
                "model_x",
                "model_y",
                "model_z",
                "target_x",
                "target_y",
                "target_z",
                "actual_distance_m",
                "sigma_x",
                "sigma_y",
                "sigma_z",
                "source_crs",
            ]
        )
        for item in session.distance_constraints:
            writer.writerow(
                [
                    "distance",
                    item.label,
                    item.source,
                    "",
                    "",
                    "",
                    *item.point_a.tolist(),
                    "",
                    "",
                    "",
                    item.actual_distance_m,
                    "",
                    "",
                    "",
                    "",
                ]
            )
            writer.writerow(
                [
                    "distance_endpoint_b",
                    item.label,
                    item.source,
                    "",
                    "",
                    "",
                    *item.point_b.tolist(),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
        for item in session.coordinate_observations:
            uv = item.pixel_uv or ("", "")
            writer.writerow(
                [
                    "coordinate",
                    item.point_id,
                    item.role,
                    item.image_name,
                    uv[0],
                    uv[1],
                    *item.model_xyz.tolist(),
                    *item.target_xyz.tolist(),
                    "",
                    *(
                        list(item.sigma_xyz)
                        if item.sigma_xyz is not None
                        else ["", "", ""]
                    ),
                    item.source_crs,
                ]
            )


def _write_measurements_csv(session: ProjectSession, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "kind", "value", "unit", "point_count"])
        for item in session.measurements:
            writer.writerow([item.label, item.kind, item.value, item.unit, item.point_count])


def _accuracy_payload(session: ProjectSession, point_count: int) -> dict[str, Any]:
    limitations = [
        "AI 特征匹配提高注册成功率，但软件不是经计量认证的测量仪器。",
        "单一比例只能恢复全局尺度，不能纠正大型场景的局部拉伸、弯曲和累积漂移。",
        "只有独立检查点残差才能用于评价绝对坐标精度；控制点拟合残差不能替代检查点。",
        "传统摄影测量的 1–3 倍 GSD 仅是良好拍摄、标定、BA/MVS 后的经验参考，不是本成果保证。",
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": session.to_summary(),
        "exported_point_count": point_count,
        "quality": session.quality_report,
        "sparse_photogrammetry": session.sparse_result,
        "photogrammetry": session.photogrammetry_result,
        "calibration": session.calibration_report,
        "coordinate_reference": session.coordinate_reference.to_dict(),
        "measurements": [item.__dict__ for item in session.measurements],
        "limitations": limitations,
    }


def _report_html(payload: dict[str, Any]) -> str:
    project = payload["project"]
    calibration = payload.get("calibration") or {}
    quality = payload.get("quality") or {}
    limitations = payload["limitations"]
    status = {
        "preview": "未标定（仅模型单位）",
        "scaled": "已恢复米制尺度",
        "engineering": "已转换到工程坐标系",
    }.get(project["calibration_mode"], project["calibration_mode"])
    check = calibration.get("check", {})
    control = calibration.get("control", {})
    coordinate_reference = payload.get("coordinate_reference") or {}
    crs_label = html.escape(
        str(coordinate_reference.get("target_crs") or "未设置")
    )
    control_html = "<p>尚未执行控制点工程坐标拟合。</p>"
    if control.get("control_count", 0):
        control_html = (
            f"<table><tr><th>控制点</th><th>采用</th><th>排除</th>"
            f"<th>RMSE XY</th><th>RMSE Z</th><th>P95 3D</th></tr><tr>"
            f"<td>{control['control_count']}</td>"
            f"<td>{control.get('inlier_count', control['control_count'])}</td>"
            f"<td>{control.get('outlier_count', 0)}</td>"
            f"<td>{control['rmse_xy']:.4f}</td><td>{control['rmse_z']:.4f}</td>"
            f"<td>{control.get('p95_3d', control.get('max_3d', 0.0)):.4f}</td>"
            "</tr></table>"
        )
    check_html = "<p>无独立检查点，不能声明绝对精度。</p>"
    if check.get("check_count", 0):
        check_html = (
            f"<table><tr><th>检查点数量</th><th>RMSE XY</th><th>RMSE Z</th>"
            f"<th>RMSE 3D</th><th>P95 3D</th><th>最大 3D</th></tr><tr>"
            f"<td>{check['check_count']}</td><td>{check['rmse_xy']:.4f} m</td>"
            f"<td>{check['rmse_z']:.4f} m</td><td>{check['rmse_3d']:.4f} m</td>"
            f"<td>{check.get('p95_3d', check['max_3d']):.4f} m</td>"
            f"<td>{check['max_3d']:.4f} m</td></tr></table>"
        )
    warnings = (
        quality.get("photo_summary", {}).get("warnings", [])
        + payload.get("photogrammetry", {}).get("warnings", [])
        + calibration.get("warnings", [])
    )
    warning_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings) or "<li>无自动警告</li>"
    limitation_items = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
    payload_text = html.escape(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>AI 摄影测量工程点云精度报告</title>
<style>
body{{font-family:"Microsoft YaHei",system-ui,sans-serif;max-width:1050px;
margin:40px auto;padding:0 24px;color:#172033}}
h1,h2{{color:#0b5b55}} .badge{{display:inline-block;padding:5px 10px;border-radius:12px;background:#e3f4f1}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5df;padding:8px;text-align:left}}
.warning{{background:#fff5dc;border-left:5px solid #e4a11b;padding:14px}}
pre{{white-space:pre-wrap;background:#f5f7fa;padding:14px}}
</style></head><body>
<h1>AI 摄影测量工程点云精度报告</h1>
<p><b>项目：</b>{html.escape(project['project_name'])}&emsp;
<b>照片：</b>{project['image_count']} 张　<b>点数：</b>{payload['exported_point_count']:,}</p>
  <p class="badge">{html.escape(status)}</p>
  <p><b>目标坐标系：</b>{crs_label}</p>
  <h2>控制点鲁棒拟合</h2>{control_html}
  <h2>独立检查点</h2>{check_html}
<h2>自动质量提示</h2><div class="warning"><ul>{warning_items}</ul></div>
<h2>适用边界</h2><ul>{limitation_items}</ul>
<h2>完整机器可读记录</h2><details><summary>展开 JSON</summary><pre>{payload_text}</pre></details>
</body></html>"""


def export_project(
    session: ProjectSession,
    base_output_dir: str | Path,
    max_points: int | None = None,
    include_las: bool = True,
) -> dict[str, str]:
    """Export a self-contained result folder and ZIP archive."""

    session.require_geometry()
    root = Path(base_output_dir).expanduser().resolve()
    project_dir = root / f"{_safe_name(session.project_name)}_{session.project_id}"
    project_dir.mkdir(parents=True, exist_ok=True)
    session.output_dir = project_dir

    points, colors = _filtered_cloud(session, max_points)
    ply_path = project_dir / "pointcloud.ply"
    write_binary_ply(
        ply_path,
        points,
        colors,
        [
            f"project {session.project_name}",
            f"coordinate_mode {session.transform.mode}",
            f"unit {session.unit}",
            "unscaled preview output must not be interpreted as metres",
        ],
    )
    constraints_path = project_dir / "calibration_observations.csv"
    _write_constraints_csv(session, constraints_path)
    measurements_path = project_dir / "measurements.csv"
    _write_measurements_csv(session, measurements_path)

    payload = _accuracy_payload(session, len(points))
    json_path = project_dir / "accuracy_report.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    html_path = project_dir / "accuracy_report.html"
    html_path.write_text(_report_html(payload), encoding="utf-8")

    result: dict[str, str] = {
        "folder": str(project_dir),
        "ply": str(ply_path),
        "report_json": str(json_path),
        "report_html": str(html_path),
    }
    if include_las and session.calibrated:
        las_path = project_dir / "pointcloud.las"
        write_las(
            las_path,
            points,
            colors,
            crs=(
                session.coordinate_reference.target_crs
                if session.engineering_calibrated
                else ""
            ),
        )
        result["las"] = str(las_path)

    model_formats = session.model_result.get("formats") or {}
    existing_model_files = {
        name: Path(value)
        for name, value in model_formats.items()
        if isinstance(value, str) and Path(value).is_file()
    }
    if existing_model_files:
        model_dir = project_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        exported_model: dict[str, str] = {}
        for name, source in existing_model_files.items():
            destination = model_dir / source.name
            if destination.exists():
                destination.unlink()
            try:
                destination.hardlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
            exported_model[name] = str(destination)
        model_manifest = model_dir / "model_manifest.json"
        model_manifest.write_text(
            json.dumps(
                {
                    "vertex_count": int(session.model_result.get("vertex_count", 0)),
                    "face_count": int(session.model_result.get("face_count", 0)),
                    "precision_mode": session.model_result.get("precision_mode", ""),
                    "formats": exported_model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result["model_folder"] = str(model_dir)

    zip_path = project_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file_path in sorted(project_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(project_dir))
    result["zip"] = str(zip_path)
    return result
