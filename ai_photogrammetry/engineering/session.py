"""In-memory project state for the local engineering application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np

from .calibration import (
    SimilarityTransform,
    fit_scale_from_distances,
    fit_similarity_robust,
    residual_report,
)
from .coordinate_systems import CoordinateReference, transform_wgs84


@dataclass
class DistanceConstraint:
    label: str
    point_a: np.ndarray
    point_b: np.ndarray
    actual_distance_m: float
    source: str = "manual"

    @property
    def model_distance(self) -> float:
        return float(np.linalg.norm(self.point_a - self.point_b))


@dataclass
class CoordinateObservation:
    point_id: str
    model_xyz: np.ndarray
    target_xyz: np.ndarray
    role: str = "control"
    image_name: str = ""
    pixel_uv: tuple[float, float] | None = None
    sigma_xyz: tuple[float, float, float] | None = None
    source_crs: str = ""


@dataclass
class Measurement:
    kind: str
    value: float
    unit: str
    label: str
    point_count: int


@dataclass
class ProjectSession:
    """All state for one local AI-photogrammetry project."""

    project_name: str = "未命名项目"
    project_id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    image_paths: list[str] = field(default_factory=list)
    image_names: list[str] = field(default_factory=list)
    original_sizes: list[tuple[int, int]] = field(default_factory=list)
    transform: SimilarityTransform = field(default_factory=SimilarityTransform.identity)
    coordinate_reference: CoordinateReference = field(default_factory=CoordinateReference)
    distance_constraints: list[DistanceConstraint] = field(default_factory=list)
    coordinate_observations: list[CoordinateObservation] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    quality_report: dict[str, Any] = field(default_factory=dict)
    calibration_report: dict[str, Any] = field(default_factory=dict)
    photogrammetry_options: dict[str, Any] = field(default_factory=dict)
    model_options: dict[str, Any] = field(default_factory=dict)
    sparse_result: dict[str, Any] = field(default_factory=dict)
    photogrammetry_result: dict[str, Any] = field(default_factory=dict)
    model_result: dict[str, Any] = field(default_factory=dict)
    processed_points: np.ndarray | None = None
    processed_colors: np.ndarray | None = None
    processed_coordinate_mode: str = ""
    filter_report: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    selected_points: list[np.ndarray] = field(default_factory=list)
    lock: RLock = field(default_factory=RLock, repr=False)

    @property
    def has_geometry(self) -> bool:
        """Whether the project has a usable SfM/MVS or processed point cloud."""

        pointcloud = str(self.photogrammetry_result.get("pointcloud", "")).strip()
        return bool(pointcloud) or self.has_processed_cloud

    @property
    def has_model(self) -> bool:
        """Whether a textured triangle model is available on disk."""

        texture_blocks = self.model_result.get("texture_blocks") or []
        if texture_blocks:
            return all(
                isinstance(block, dict)
                and Path(str(block.get("mesh", ""))).is_file()
                and Path(str(block.get("texture", ""))).is_file()
                for block in texture_blocks
            )
        mesh = str(self.model_result.get("textured_mesh", "")).strip()
        texture = str(self.model_result.get("texture_atlas", "")).strip()
        return (
            bool(mesh)
            and bool(texture)
            and Path(mesh).is_file()
            and Path(texture).is_file()
        )

    @property
    def calibrated(self) -> bool:
        return self.transform.mode in {"scaled", "engineering"}

    @property
    def engineering_calibrated(self) -> bool:
        return self.transform.mode == "engineering"

    @property
    def unit(self) -> str:
        return "m" if self.calibrated else "模型单位"

    def require_geometry(self) -> None:
        if not self.has_geometry:
            raise RuntimeError("请先完成 AI 特征摄影测量并生成稠密点云")

    def require_metric(self, operation: str = "测量") -> None:
        self.require_geometry()
        if not self.calibrated:
            raise RuntimeError(f"{operation}需要真实尺度；请先添加已知距离或控制点并完成标定")

    def clear_processed(self) -> None:
        self.processed_points = None
        self.processed_colors = None
        self.processed_coordinate_mode = ""
        self.filter_report.clear()

    @property
    def has_processed_cloud(self) -> bool:
        return (
            self.processed_points is not None
            and self.processed_colors is not None
            and self.processed_coordinate_mode == self.transform.mode
        )

    def add_distance_constraint(
        self,
        label: str,
        point_a: np.ndarray,
        point_b: np.ndarray,
        actual_distance_m: float,
        source: str = "manual",
    ) -> DistanceConstraint:
        self.require_geometry()
        if not np.isfinite(actual_distance_m) or actual_distance_m <= 0:
            raise ValueError("实际距离必须是大于 0 的米制数值")
        constraint = DistanceConstraint(
            label=label.strip() or f"标尺{len(self.distance_constraints) + 1}",
            point_a=np.asarray(point_a, dtype=np.float64),
            point_b=np.asarray(point_b, dtype=np.float64),
            actual_distance_m=float(actual_distance_m),
            source=source,
        )
        if constraint.model_distance <= 1e-12:
            raise ValueError("两个标尺点重合")
        self.distance_constraints.append(constraint)
        return constraint

    def calibrate_scale(self) -> dict:
        self.clear_processed()
        transform, report = fit_scale_from_distances(
            [item.model_distance for item in self.distance_constraints],
            [item.actual_distance_m for item in self.distance_constraints],
        )
        self.transform = transform
        report["mode"] = "scaled"
        report["warning"] = "仅恢复统一比例；未纠正大型场景的局部拉伸、弯曲或累积漂移。"
        self.calibration_report = report
        return report

    def add_coordinate_observation(
        self,
        *,
        point_id: str,
        model_xyz: np.ndarray,
        target_xyz: np.ndarray,
        role: str,
        image_name: str = "",
        pixel_uv: tuple[float, float] | None = None,
        sigma_xyz: tuple[float, float, float] | None = None,
        source_crs: str = "",
    ) -> CoordinateObservation:
        normalized_role = role.lower()
        if normalized_role not in {"control", "check"}:
            raise ValueError("点类型必须是 control 或 check")
        if not point_id.strip():
            raise ValueError("控制点/检查点编号不能为空")
        observation = CoordinateObservation(
            point_id=point_id.strip(),
            model_xyz=np.asarray(model_xyz, dtype=np.float64),
            target_xyz=np.asarray(target_xyz, dtype=np.float64),
            role=normalized_role,
            image_name=image_name,
            pixel_uv=pixel_uv,
            sigma_xyz=sigma_xyz,
            source_crs=source_crs,
        )
        if observation.model_xyz.shape != (3,) or observation.target_xyz.shape != (3,):
            raise ValueError("控制点坐标必须为三维坐标")
        if not np.isfinite(observation.model_xyz).all() or not np.isfinite(observation.target_xyz).all():
            raise ValueError("控制点坐标包含空值或非有限数")
        if observation.sigma_xyz is not None:
            sigma = np.asarray(observation.sigma_xyz, dtype=np.float64)
            if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError("坐标精度σX/σY/σZ必须是三个有限正数")
        self.coordinate_observations.append(observation)
        return observation

    def configure_wgs84_coordinates(
        self,
        *,
        target_crs: str = "",
        origin: tuple[float, float, float] | None = None,
        vertical_datum: str = "ellipsoidal",
    ) -> CoordinateReference:
        """Configure WGS84 input as projected coordinates or local ENU."""

        normalized_target = target_crs.strip()
        if normalized_target:
            reference = CoordinateReference(
                mode="wgs84_projected",
                source_crs="EPSG:4979",
                target_crs=normalized_target,
                vertical_datum=vertical_datum,
            )
        else:
            if origin is None:
                raise ValueError("Local ENU需要经度、纬度、椭球高原点")
            reference = CoordinateReference(
                mode="wgs84_enu",
                source_crs="EPSG:4979",
                target_crs="LOCAL_ENU",
                origin_longitude=float(origin[0]),
                origin_latitude=float(origin[1]),
                origin_height=float(origin[2]),
                vertical_datum=vertical_datum,
            )
        self.coordinate_reference = reference
        return reference

    def add_geographic_coordinate_observation(
        self,
        *,
        point_id: str,
        model_xyz: np.ndarray,
        longitude: float,
        latitude: float,
        height: float,
        role: str,
        image_name: str = "",
        pixel_uv: tuple[float, float] | None = None,
        sigma_xyz: tuple[float, float, float] | None = None,
        target_crs: str = "",
    ) -> CoordinateObservation:
        """Convert one WGS84 observation before adding it to calibration."""

        value = (float(longitude), float(latitude), float(height))
        if self.coordinate_reference.mode not in {"wgs84_enu", "wgs84_projected"}:
            self.configure_wgs84_coordinates(
                target_crs=target_crs,
                origin=value if not target_crs.strip() else None,
            )
        elif target_crs.strip() and self.coordinate_reference.target_crs.casefold() != target_crs.strip().casefold():
            raise ValueError("当前工程已经使用另一目标坐标系，请清空坐标观测后再修改")
        target = transform_wgs84([value], self.coordinate_reference)[0]
        return self.add_coordinate_observation(
            point_id=point_id,
            model_xyz=model_xyz,
            target_xyz=target,
            role=role,
            image_name=image_name,
            pixel_uv=pixel_uv,
            sigma_xyz=sigma_xyz,
            source_crs="EPSG:4979",
        )

    def _unique_coordinate_points(
        self,
        role: str,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        grouped: dict[str, list[CoordinateObservation]] = {}
        for item in self.coordinate_observations:
            if item.role == role:
                grouped.setdefault(item.point_id, []).append(item)
        ids: list[str] = []
        model: list[np.ndarray] = []
        target: list[np.ndarray] = []
        weights: list[float] = []
        for point_id, values in grouped.items():
            targets = np.stack([value.target_xyz for value in values])
            if np.max(np.linalg.norm(targets - targets.mean(axis=0), axis=1), initial=0.0) > 1e-6:
                raise ValueError(f"点 {point_id} 的工程坐标在多次观测中不一致")
            ids.append(point_id)
            model.append(np.median(np.stack([value.model_xyz for value in values]), axis=0))
            target.append(targets.mean(axis=0))
            sigmas = [
                np.asarray(value.sigma_xyz, dtype=np.float64)
                for value in values
                if value.sigma_xyz is not None
            ]
            if sigmas:
                sigma = np.median(np.stack(sigmas), axis=0)
                weights.append(float(1.0 / max(np.mean(sigma**2), 1e-12)))
            else:
                weights.append(1.0)
        if not model:
            return ids, np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
        return ids, np.stack(model), np.stack(target), np.asarray(weights)

    def calibrate_engineering(self, *, ransac_threshold: float = 0.10) -> dict:
        self.clear_processed()
        control_ids, model, target, weights = self._unique_coordinate_points("control")
        transform, control_report = fit_similarity_robust(
            model,
            target,
            weights=weights,
            threshold=float(ransac_threshold),
        )
        control_report["point_ids"] = control_ids
        control_report["outlier_point_ids"] = [
            point_id
            for point_id, inlier in zip(
                control_ids,
                control_report.get("inlier_mask", [True] * len(control_ids)),
                strict=True,
            )
            if not inlier
        ]
        distribution: dict[str, float] = {}
        report: dict[str, Any] = {
            "mode": "engineering",
            "control_point_ids": control_ids,
            "control": control_report,
            "distribution": distribution,
            "coordinate_reference": self.coordinate_reference.to_dict(),
        }
        check_ids, check_model, check_target, _ = self._unique_coordinate_points("check")
        if len(check_model):
            check_report = residual_report(transform, check_model, check_target)
            check_report["point_ids"] = check_ids
            report["check"] = check_report
        else:
            report["check"] = {
                "check_count": 0,
                "warning": "尚未提供独立检查点，不能独立评价绝对精度。",
            }
        warnings: list[str] = []
        if len(control_ids) < 5:
            warnings.append("控制点少于建议的 5 个；最低数量可解算，但工程可靠性不足。")
        if control_report.get("outlier_count", 0):
            warnings.append(
                "已自动排除异常控制点："
                + "、".join(control_report.get("outlier_point_ids", []))
            )
        if len(check_ids) < 2:
            warnings.append("检查点少于建议的 2 个。")
        target_extent = float(
            np.max(np.linalg.norm(target - target.mean(axis=0), axis=1), initial=0.0)
        )
        vertical_ratio = float(np.ptp(target[:, 2]) / max(target_extent, 1e-12))
        distribution["vertical_spread_ratio"] = vertical_ratio
        if target_extent > 1e-9 and vertical_ratio < 0.1:
            warnings.append("控制点高程分布较弱；建议增加高处和低处控制点。")
        report["warnings"] = warnings
        self.transform = transform
        self.calibration_report = report
        return report

    def clear_selected_points(self) -> None:
        self.selected_points.clear()

    def to_summary(self) -> dict[str, Any]:
        photo_count = int(
            self.photogrammetry_result.get("image_count", len(self.image_names))
        )
        return {
            "project_name": self.project_name,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "image_count": photo_count,
            "has_geometry": self.has_geometry,
            "has_model": self.has_model,
            "calibration_mode": self.transform.mode,
            "unit": self.unit,
            "distance_constraint_count": len(self.distance_constraints),
            "coordinate_observation_count": len(self.coordinate_observations),
            "measurement_count": len(self.measurements),
            "processed_point_count": (
                int(len(self.processed_points)) if self.has_processed_cloud else 0
            ),
            "transform": self.transform.to_dict(),
            "coordinate_reference": self.coordinate_reference.to_dict(),
        }
