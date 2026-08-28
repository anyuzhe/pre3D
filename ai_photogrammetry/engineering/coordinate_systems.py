"""Coordinate-reference helpers for engineering photogrammetry.

The reconstruction itself stays in a numerically stable local Cartesian
system.  Survey coordinates can either already be Cartesian, be converted
from WGS84 to a local East/North/Up frame without extra dependencies, or be
projected to an EPSG CRS through ``pyproj`` when it is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


@dataclass
class CoordinateReference:
    """Serializable description of the coordinates used by one project."""

    mode: str = "cartesian"
    source_crs: str = "LOCAL_CARTESIAN"
    target_crs: str = "LOCAL_CARTESIAN"
    origin_longitude: float | None = None
    origin_latitude: float | None = None
    origin_height: float | None = None
    vertical_datum: str = "unknown"
    axis_order: str = "E,N,Z"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "CoordinateReference":
        if not payload:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: payload[key] for key in fields if key in payload})

    @property
    def label(self) -> str:
        if self.mode == "wgs84_enu":
            return "Local ENU（由WGS84转换）"
        if self.mode == "wgs84_projected":
            return self.target_crs
        return self.target_crs or "LOCAL_CARTESIAN"


def _geodetic_array(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("经纬高必须是N×3数组，顺序为经度、纬度、高度")
    if not np.isfinite(array).all():
        raise ValueError("经纬高包含空值或非有限数")
    if np.any(np.abs(array[:, 0]) > 180.0) or np.any(np.abs(array[:, 1]) > 90.0):
        raise ValueError("经纬度超出有效范围")
    return array


def wgs84_to_ecef(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Convert longitude/latitude/ellipsoidal-height to WGS84 ECEF metres."""

    array = _geodetic_array(values)
    longitude = np.deg2rad(array[:, 0])
    latitude = np.deg2rad(array[:, 1])
    height = array[:, 2]
    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    radius = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_latitude**2)
    return np.column_stack(
        (
            (radius + height) * cos_latitude * np.cos(longitude),
            (radius + height) * cos_latitude * np.sin(longitude),
            (radius * (1.0 - _WGS84_E2) + height) * sin_latitude,
        )
    )


def wgs84_to_local_enu(
    values: Sequence[Sequence[float]] | np.ndarray,
    origin: Sequence[float],
) -> np.ndarray:
    """Convert WGS84 longitude/latitude/height to local ENU metres."""

    array = _geodetic_array(values)
    origin_array = _geodetic_array(np.asarray(origin, dtype=np.float64).reshape(1, 3))[0]
    ecef = wgs84_to_ecef(array)
    origin_ecef = wgs84_to_ecef(origin_array.reshape(1, 3))[0]
    longitude = np.deg2rad(origin_array[0])
    latitude = np.deg2rad(origin_array[1])
    rotation = np.array(
        [
            [-np.sin(longitude), np.cos(longitude), 0.0],
            [
                -np.sin(latitude) * np.cos(longitude),
                -np.sin(latitude) * np.sin(longitude),
                np.cos(latitude),
            ],
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ],
        ],
        dtype=np.float64,
    )
    return (ecef - origin_ecef) @ rotation.T


def wgs84_to_projected(
    values: Sequence[Sequence[float]] | np.ndarray,
    target_crs: str,
) -> np.ndarray:
    """Project WGS84 longitude/latitude/height using PROJ/pyproj."""

    array = _geodetic_array(values)
    normalized = str(target_crs).strip()
    if not normalized:
        raise ValueError("目标EPSG/CRS不能为空")
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise RuntimeError("EPSG坐标转换需要安装pyproj；Local ENU转换不需要") from exc
    try:
        target = CRS.from_user_input(normalized)
        if target.is_geographic:
            raise ValueError(
                "目标CRS仍是经纬度坐标，不能作为米制工程坐标；"
                "请选择投影坐标系或使用Local ENU"
            )
        horizontal_axes = list(target.axis_info[:2])
        if horizontal_axes and any(
            not np.isclose(float(axis.unit_conversion_factor or 0.0), 1.0)
            for axis in horizontal_axes
        ):
            raise ValueError("目标CRS的水平单位不是米，请选择米制投影坐标系")
        transformer = Transformer.from_crs("EPSG:4979", target, always_xy=True)
        x, y, z = transformer.transform(array[:, 0], array[:, 1], array[:, 2])
    except Exception as exc:
        raise ValueError(f"无法使用目标坐标系 {normalized}：{exc}") from exc
    result = np.column_stack((x, y, z)).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"坐标转换到 {normalized} 后出现非有限数")
    return result


def transform_wgs84(
    values: Sequence[Sequence[float]] | np.ndarray,
    reference: CoordinateReference,
) -> np.ndarray:
    """Transform WGS84 observations according to a project reference."""

    if reference.mode == "wgs84_projected":
        return wgs84_to_projected(values, reference.target_crs)
    if reference.mode != "wgs84_enu":
        raise ValueError("当前工程未配置WGS84坐标输入")
    origin_values = (
        reference.origin_longitude,
        reference.origin_latitude,
        reference.origin_height,
    )
    if any(value is None for value in origin_values):
        raise ValueError("Local ENU尚未设置经纬高原点")
    return wgs84_to_local_enu(values, origin_values)  # type: ignore[arg-type]
