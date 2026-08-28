"""Scale and engineering-coordinate calibration utilities.

Photogrammetric SfM produces geometry with an arbitrary origin, orientation and
scale unless survey constraints are supplied. This module keeps scale-only
calibration separate from a full 7-parameter similarity transformation so the
UI can enforce honest units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12


def _as_points(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} 必须是 N×3 坐标数组")
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"{name} 包含空值或非有限数")
    return points


@dataclass(frozen=True)
class SimilarityTransform:
    """Transform model coordinates to calibrated coordinates.

    ``target = scale * (rotation @ model) + translation``
    """

    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    mode: str = "preview"

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("尺度必须是有限正数")
        if rotation.shape != (3, 3):
            raise ValueError("旋转矩阵必须为 3×3")
        if translation.shape != (3,):
            raise ValueError("平移向量必须为 3 维")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("变换包含非有限数")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("旋转矩阵不是正交矩阵")
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("旋转矩阵包含反射或退化")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> "SimilarityTransform":
        return cls(mode="preview")

    def apply(self, points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.shape[-1] != 3:
            raise ValueError("点坐标最后一维必须为 3")
        return self.scale * (array @ self.rotation.T) + self.translation

    def apply_vectors(self, vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float64)
        return self.scale * (array @ self.rotation.T)

    def apply_poses(self, camera_to_world: np.ndarray) -> np.ndarray:
        poses = np.asarray(camera_to_world, dtype=np.float64)
        if poses.shape[-2:] != (4, 4):
            raise ValueError("相机位姿必须为 …×4×4")
        result = poses.copy()
        result[..., :3, :3] = self.rotation @ poses[..., :3, :3]
        result[..., :3, 3] = self.apply(poses[..., :3, 3])
        return result

    def inverse(self) -> "SimilarityTransform":
        inverse_rotation = self.rotation.T
        inverse_scale = 1.0 / self.scale
        inverse_translation = -inverse_scale * (inverse_rotation @ self.translation)
        return SimilarityTransform(
            scale=inverse_scale,
            rotation=inverse_rotation,
            translation=inverse_translation,
            mode=self.mode,
        )

    def then(self, after: "SimilarityTransform", mode: str | None = None) -> "SimilarityTransform":
        """Compose this transform followed by ``after``."""

        return SimilarityTransform(
            scale=after.scale * self.scale,
            rotation=after.rotation @ self.rotation,
            translation=after.scale * (after.rotation @ self.translation) + after.translation,
            mode=mode or after.mode,
        )

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "mode": self.mode,
        }


def fit_scale_from_distances(
    model_distances: Iterable[float],
    actual_distances: Iterable[float],
) -> tuple[SimilarityTransform, dict]:
    """Fit one global scale using zero-intercept least squares."""

    model = np.asarray(list(model_distances), dtype=np.float64)
    actual = np.asarray(list(actual_distances), dtype=np.float64)
    if model.ndim != 1 or actual.ndim != 1 or len(model) != len(actual):
        raise ValueError("模型距离与实际距离数量不一致")
    if len(model) < 1:
        raise ValueError("至少需要一个有效的已知距离")
    valid = np.isfinite(model) & np.isfinite(actual) & (model > EPS) & (actual > EPS)
    if not valid.all():
        raise ValueError("已知距离必须为有限正数，且两模型点不能重合")

    scale = float(np.dot(model, actual) / np.dot(model, model))
    fitted = scale * model
    residuals = fitted - actual
    relative = residuals / actual
    report = {
        "constraint_count": int(len(model)),
        "scale": scale,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "mean_abs_relative_error": float(np.mean(np.abs(relative))),
        "residuals": residuals.tolist(),
        "fitted_distances": fitted.tolist(),
    }
    return SimilarityTransform(scale=scale, mode="scaled"), report


def fit_similarity(
    model_points: np.ndarray | Sequence[Sequence[float]],
    target_points: np.ndarray | Sequence[Sequence[float]],
    weights: np.ndarray | Sequence[float] | None = None,
    mode: str = "engineering",
) -> tuple[SimilarityTransform, dict]:
    """Fit a no-reflection 7-parameter similarity transform (Umeyama).

    At least three non-collinear 3D correspondences are required.
    """

    source = _as_points(model_points, "模型控制点")
    target = _as_points(target_points, "工程控制点")
    if source.shape != target.shape:
        raise ValueError("模型控制点与工程控制点数量不一致")
    if len(source) < 3:
        raise ValueError("三维坐标转换至少需要 3 个控制点")

    centered_rank = np.linalg.matrix_rank(source - source.mean(axis=0), tol=1e-9)
    target_rank = np.linalg.matrix_rank(target - target.mean(axis=0), tol=1e-9)
    if centered_rank < 2 or target_rank < 2:
        raise ValueError("控制点共线或过于集中，无法稳定求解三维相似变换")

    if weights is None:
        w = np.ones(len(source), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(source),) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("控制点权重必须是与控制点等长的有限正数")
    w /= w.sum()

    source_mean = np.sum(source * w[:, None], axis=0)
    target_mean = np.sum(target * w[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (target_centered * w[:, None]).T @ source_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt

    source_variance = float(np.sum(w * np.sum(source_centered**2, axis=1)))
    if source_variance <= EPS:
        raise ValueError("控制点空间尺度退化")
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)

    transform = SimilarityTransform(scale, rotation, translation, mode=mode)
    fitted = transform.apply(source)
    residual_vectors = fitted - target
    residual_3d = np.linalg.norm(residual_vectors, axis=1)
    residual_xy = np.linalg.norm(residual_vectors[:, :2], axis=1)
    report = {
        "control_count": int(len(source)),
        "scale": scale,
        "rmse_x": float(np.sqrt(np.mean(residual_vectors[:, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(residual_vectors[:, 1] ** 2))),
        "rmse_3d": float(np.sqrt(np.mean(residual_3d**2))),
        "rmse_xy": float(np.sqrt(np.mean(residual_xy**2))),
        "rmse_z": float(np.sqrt(np.mean(residual_vectors[:, 2] ** 2))),
        "median_3d": float(np.median(residual_3d)),
        "p95_3d": float(np.percentile(residual_3d, 95)),
        "max_3d": float(np.max(residual_3d)),
        "residual_vectors": residual_vectors.tolist(),
        "residual_3d": residual_3d.tolist(),
        "source_rank": int(centered_rank),
    }
    return transform, report


def fit_similarity_robust(
    model_points: np.ndarray | Sequence[Sequence[float]],
    target_points: np.ndarray | Sequence[Sequence[float]],
    weights: np.ndarray | Sequence[float] | None = None,
    *,
    threshold: float = 0.10,
    max_trials: int = 512,
    random_seed: int = 0,
    mode: str = "engineering",
) -> tuple[SimilarityTransform, dict]:
    """Robustly fit Sim3 with deterministic RANSAC followed by weighted LS.

    The threshold is expressed in target-coordinate units, normally metres.
    Three-point minimal samples are used only to find an inlier set; the final
    transform is always refitted from every accepted control point.
    """

    source = _as_points(model_points, "模型控制点")
    target = _as_points(target_points, "工程控制点")
    if source.shape != target.shape:
        raise ValueError("模型控制点与工程控制点数量不一致")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("控制点RANSAC阈值必须是有限正数")
    if max_trials < 1:
        raise ValueError("控制点RANSAC迭代次数必须大于0")
    if len(source) <= 3:
        transform, report = fit_similarity(source, target, weights=weights, mode=mode)
        report.update(
            {
                "robust": False,
                "ransac_threshold": float(threshold),
                "inlier_mask": [True] * len(source),
                "inlier_count": int(len(source)),
                "outlier_count": 0,
            }
        )
        return transform, report

    raw_weights = (
        np.ones(len(source), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if raw_weights.shape != (len(source),) or not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0):
        raise ValueError("控制点权重必须是与控制点等长的有限正数")

    generator = np.random.default_rng(random_seed)
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float, float] | None = None
    for _ in range(int(max_trials)):
        sample = generator.choice(len(source), size=3, replace=False)
        try:
            candidate, _ = fit_similarity(source[sample], target[sample], mode=mode)
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(candidate.apply(source) - target, axis=1)
        mask = residuals <= float(threshold)
        count = int(mask.sum())
        if count < 3:
            continue
        inlier_residuals = residuals[mask]
        score = (
            count,
            -float(np.median(inlier_residuals)),
            -float(np.mean(inlier_residuals)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask
            if count == len(source) and float(np.max(inlier_residuals)) <= threshold * 0.25:
                break

    if best_mask is None or int(best_mask.sum()) < 3:
        raise ValueError("控制点RANSAC无法找到至少3个一致控制点，请检查编号、坐标和单位")

    transform, report = fit_similarity(
        source[best_mask],
        target[best_mask],
        weights=raw_weights[best_mask],
        mode=mode,
    )
    all_vectors = transform.apply(source) - target
    all_residuals = np.linalg.norm(all_vectors, axis=1)
    # Re-evaluate once after the all-inlier least-squares fit.  This prevents a
    # weak minimal sample from retaining a point that is no longer consistent.
    refined_mask = all_residuals <= float(threshold)
    if int(refined_mask.sum()) >= 3 and not np.array_equal(refined_mask, best_mask):
        best_mask = refined_mask
        transform, report = fit_similarity(
            source[best_mask],
            target[best_mask],
            weights=raw_weights[best_mask],
            mode=mode,
        )
        all_vectors = transform.apply(source) - target
        all_residuals = np.linalg.norm(all_vectors, axis=1)

    report.update(
        {
            "control_count": int(len(source)),
            "robust": True,
            "ransac_threshold": float(threshold),
            "ransac_trials": int(max_trials),
            "inlier_mask": best_mask.tolist(),
            "inlier_count": int(best_mask.sum()),
            "outlier_count": int((~best_mask).sum()),
            "all_residual_vectors": all_vectors.tolist(),
            "all_residual_3d": all_residuals.tolist(),
            "all_median_3d": float(np.median(all_residuals)),
            "all_p95_3d": float(np.percentile(all_residuals, 95)),
            "all_max_3d": float(np.max(all_residuals)),
        }
    )
    return transform, report


def residual_report(
    transform: SimilarityTransform,
    model_points: np.ndarray | Sequence[Sequence[float]],
    target_points: np.ndarray | Sequence[Sequence[float]],
) -> dict:
    """Evaluate independent check points without refitting."""

    source = _as_points(model_points, "模型检查点")
    target = _as_points(target_points, "工程检查点")
    if source.shape != target.shape:
        raise ValueError("模型检查点与工程检查点数量不一致")
    fitted = transform.apply(source)
    vectors = fitted - target
    distance_3d = np.linalg.norm(vectors, axis=1)
    distance_xy = np.linalg.norm(vectors[:, :2], axis=1)
    return {
        "check_count": int(len(source)),
        "rmse_x": float(np.sqrt(np.mean(vectors[:, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(vectors[:, 1] ** 2))),
        "rmse_3d": float(np.sqrt(np.mean(distance_3d**2))),
        "rmse_xy": float(np.sqrt(np.mean(distance_xy**2))),
        "rmse_z": float(np.sqrt(np.mean(vectors[:, 2] ** 2))),
        "mean_3d": float(np.mean(distance_3d)),
        "median_3d": float(np.median(distance_3d)),
        "p95_3d": float(np.percentile(distance_3d, 95)),
        "max_3d": float(np.max(distance_3d)),
        "residual_vectors": vectors.tolist(),
        "residual_3d": distance_3d.tolist(),
        "fitted_points": fitted.tolist(),
    }
