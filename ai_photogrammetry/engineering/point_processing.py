"""Deterministic point-cloud cleanup for engineering exports and viewing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from .pointcloud_io import load_ply_vertices_colors
from .session import ProjectSession


@dataclass
class FilterOptions:
    """Point filtering controls. Zero disables the corresponding operation."""

    distance_mad_multiplier: float = 0.0
    voxel_size: float = 0.0
    statistical_neighbors: int = 0
    statistical_std_ratio: float = 2.0
    radius: float = 0.0
    radius_min_neighbors: int = 0
    max_points: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "FilterOptions":
        source = value or {}
        return cls(
            **{
                name: source[name]
                for name in cls.__dataclass_fields__
                if name in source
            }
        )

    def validate(self) -> None:
        if self.distance_mad_multiplier < 0:
            raise ValueError("距离 MAD 倍数不能小于 0")
        if self.voxel_size < 0:
            raise ValueError("体素尺寸不能小于 0")
        if self.statistical_neighbors < 0:
            raise ValueError("统计邻居数不能小于 0")
        if self.statistical_neighbors and self.statistical_neighbors < 3:
            raise ValueError("统计离群过滤至少需要 3 个邻居")
        if self.statistical_std_ratio <= 0:
            raise ValueError("统计离群标准差倍数必须大于 0")
        if self.radius < 0 or self.radius_min_neighbors < 0:
            raise ValueError("半径和最少邻居数不能小于 0")
        if bool(self.radius) != bool(self.radius_min_neighbors):
            raise ValueError("半径离群过滤需同时设置半径和最少邻居数")
        if self.max_points < 0:
            raise ValueError("最大点数不能小于 0")


def _notify(
    callback: Callable[[float, str], None] | None,
    value: float,
    message: str,
) -> None:
    if callback:
        callback(value, message)


def _apply_mask(
    points: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return points[mask], colors[mask]


def _voxel_representatives(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Keep one deterministic representative from every occupied voxel."""

    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    return np.sort(order[first])


def _statistical_mask(
    points: np.ndarray,
    neighbors: int,
    std_ratio: float,
    *,
    chunk_size: int = 250_000,
) -> tuple[np.ndarray, float]:
    count = len(points)
    k = min(neighbors + 1, count)
    if k <= 2:
        return np.ones(count, dtype=bool), float("inf")
    tree = cKDTree(points)
    mean_distances = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        distances, _ = tree.query(points[start:end], k=k, workers=-1)
        mean_distances[start:end] = np.mean(distances[:, 1:], axis=1)
    finite = np.isfinite(mean_distances)
    center = float(np.mean(mean_distances[finite]))
    spread = float(np.std(mean_distances[finite]))
    threshold = center + std_ratio * spread
    return finite & (mean_distances <= threshold), threshold


def _radius_mask(
    points: np.ndarray,
    radius: float,
    minimum_neighbors: int,
    *,
    chunk_size: int = 250_000,
) -> np.ndarray:
    tree = cKDTree(points)
    counts = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        try:
            values = tree.query_ball_point(
                points[start:end],
                radius,
                return_length=True,
                workers=-1,
            )
        except TypeError:  # Older SciPy fallback.
            values = [
                len(item)
                for item in tree.query_ball_point(points[start:end], radius)
            ]
        counts[start:end] = np.asarray(values, dtype=np.int32)
    # The result includes the point itself.
    return counts >= minimum_neighbors + 1


def process_session_cloud(
    session: ProjectSession,
    options: FilterOptions | dict[str, Any] | None = None,
    *,
    source_pointcloud: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run the configured cleanup chain and attach its result to ``session``."""

    session.require_geometry()
    settings = options if isinstance(options, FilterOptions) else FilterOptions.from_dict(options)
    settings.validate()

    _notify(progress_callback, 0.02, "读取稠密点云")
    if not source_pointcloud:
        raise ValueError("缺少 AI 摄影测量点云路径")
    points, colors = load_ply_vertices_colors(source_pointcloud)
    points = session.transform.apply(points)
    original_count = len(points)
    steps: list[dict[str, Any]] = []

    finite = np.isfinite(points).all(axis=1)
    points, colors = _apply_mask(points, colors, finite)
    steps.append(
        {
            "stage": "finite",
            "before": original_count,
            "after": len(points),
            "removed": original_count - len(points),
        }
    )
    if not len(points):
        raise ValueError("没有有限的三维点可供过滤")

    if settings.distance_mad_multiplier > 0 and len(points) >= 10:
        _notify(progress_callback, 0.27, "剔除全局距离异常点")
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        distance_median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - distance_median)))
        robust_sigma = max(1.4826 * mad, np.finfo(np.float64).eps)
        distance_limit = distance_median + settings.distance_mad_multiplier * robust_sigma
        keep = distances <= distance_limit
        before = len(points)
        points, colors = _apply_mask(points, colors, keep)
        steps.append(
            {
                "stage": "distance_mad",
                "before": before,
                "after": len(points),
                "removed": before - len(points),
                "distance_limit": distance_limit,
            }
        )

    if settings.voxel_size > 0 and len(points):
        _notify(progress_callback, 0.43, "体素降采样")
        before = len(points)
        indices = _voxel_representatives(points, settings.voxel_size)
        points, colors = points[indices], colors[indices]
        steps.append(
            {
                "stage": "voxel",
                "before": before,
                "after": len(points),
                "removed": before - len(points),
                "voxel_size": settings.voxel_size,
            }
        )

    if settings.statistical_neighbors > 0 and len(points) >= 4:
        _notify(progress_callback, 0.62, "统计离群点过滤")
        keep, statistical_threshold = _statistical_mask(
            points,
            settings.statistical_neighbors,
            settings.statistical_std_ratio,
        )
        before = len(points)
        points, colors = _apply_mask(points, colors, keep)
        steps.append(
            {
                "stage": "statistical",
                "before": before,
                "after": len(points),
                "removed": before - len(points),
                "neighbors": settings.statistical_neighbors,
                "distance_threshold": statistical_threshold,
            }
        )

    if settings.radius > 0 and settings.radius_min_neighbors > 0 and len(points):
        _notify(progress_callback, 0.78, "半径离群点过滤")
        keep = _radius_mask(points, settings.radius, settings.radius_min_neighbors)
        before = len(points)
        points, colors = _apply_mask(points, colors, keep)
        steps.append(
            {
                "stage": "radius",
                "before": before,
                "after": len(points),
                "removed": before - len(points),
                "radius": settings.radius,
                "minimum_neighbors": settings.radius_min_neighbors,
            }
        )

    if settings.max_points and len(points) > settings.max_points:
        _notify(progress_callback, 0.9, "限制最终点数")
        before = len(points)
        indices = np.sort(
            np.random.default_rng(42).choice(
                len(points),
                int(settings.max_points),
                replace=False,
            )
        )
        points, colors = points[indices], colors[indices]
        steps.append(
            {
                "stage": "max_points",
                "before": before,
                "after": len(points),
                "removed": before - len(points),
            }
        )

    if not len(points):
        raise ValueError("过滤参数过强，最终没有保留任何三维点")
    session.processed_points = np.asarray(points, dtype=np.float64)
    session.processed_colors = np.asarray(colors, dtype=np.uint8)
    session.processed_coordinate_mode = session.transform.mode
    session.filter_report = {
        "options": asdict(settings),
        "input_point_count": original_count,
        "output_point_count": int(len(points)),
        "removed_point_count": int(original_count - len(points)),
        "coordinate_mode": session.transform.mode,
        "unit": session.unit,
        "steps": steps,
    }
    _notify(progress_callback, 1.0, f"点云过滤完成，保留 {len(points):,} 点")
    return session.filter_report
