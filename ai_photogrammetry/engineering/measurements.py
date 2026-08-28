"""Metric-gated point measurements."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _points(value: list[np.ndarray] | np.ndarray, minimum: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < minimum:
        raise ValueError(f"该测量至少需要 {minimum} 个有效三维点")
    if not np.isfinite(array).all():
        raise ValueError("测量点包含非有限坐标")
    return array


def straight_distance(points: list[np.ndarray] | np.ndarray) -> float:
    array = _points(points, 2)
    return float(np.linalg.norm(array[1] - array[0]))


def polyline_length(points: list[np.ndarray] | np.ndarray) -> float:
    array = _points(points, 2)
    return float(np.sum(np.linalg.norm(np.diff(array, axis=0), axis=1)))


def polygon_area(points: list[np.ndarray] | np.ndarray) -> float:
    """Area of an ordered 3D polygon after best-fit-plane projection."""

    array = _points(points, 3)
    centered = array - array.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= 1e-12:
        raise ValueError("面积点近似共线")
    basis = vt[:2]
    uv = centered @ basis.T
    x, y = uv[:, 0], uv[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_hull_volume(points: list[np.ndarray] | np.ndarray) -> float:
    """Volume of the 3D convex hull of selected boundary points."""

    array = _points(points, 4)
    try:
        hull = ConvexHull(array)
    except QhullError as exc:
        raise ValueError("体积点共面或空间分布退化，无法形成三维凸包") from exc
    return float(hull.volume)


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= 1e-12:
        raise ValueError("拟合平面的点近似共线")
    return center, vt[-1]


def plane_spacing(points: list[np.ndarray] | np.ndarray, first_plane_count: int) -> float:
    """Distance between two best-fit near-parallel planes."""

    array = _points(points, 6)
    if first_plane_count < 3 or len(array) - first_plane_count < 3:
        raise ValueError("两个平面各至少需要 3 个点")
    center_a, normal_a = _fit_plane(array[:first_plane_count])
    center_b, normal_b = _fit_plane(array[first_plane_count:])
    if np.dot(normal_a, normal_b) < 0:
        normal_b = -normal_b
    angle = float(np.rad2deg(np.arccos(np.clip(np.dot(normal_a, normal_b), -1, 1))))
    if angle > 20:
        raise ValueError(f"两个拟合平面夹角为 {angle:.1f}°，不适合报告为平行间距")
    normal = normal_a + normal_b
    normal /= np.linalg.norm(normal)
    return float(abs(np.dot(center_b - center_a, normal)))


def calculate(kind: str, points: list[np.ndarray], first_plane_count: int = 3) -> tuple[float, str]:
    mapping = {
        "直线距离": (straight_distance, "m"),
        "裂隙开度": (straight_distance, "m"),
        "折线长度": (polyline_length, "m"),
        "多边形面积": (polygon_area, "m²"),
        "凸包体积（近似）": (convex_hull_volume, "m³"),
    }
    if kind == "结构面间距":
        return plane_spacing(points, first_plane_count), "m"
    if kind not in mapping:
        raise ValueError(f"不支持的测量类型：{kind}")
    function, unit = mapping[kind]
    return function(points), unit
