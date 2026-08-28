import numpy as np
import pytest

from ai_photogrammetry.engineering.calibration import (
    SimilarityTransform,
    fit_scale_from_distances,
    fit_similarity,
    fit_similarity_robust,
    residual_report,
)


def test_scale_fit_exact_and_residuals():
    transform, report = fit_scale_from_distances([1.0, 2.0, 4.0], [2.5, 5.0, 10.0])
    assert transform.mode == "scaled"
    assert np.isclose(transform.scale, 2.5)
    assert np.isclose(report["rmse"], 0.0)


def test_similarity_fit_recovers_known_transform():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    expected = SimilarityTransform(
        scale=3.25,
        rotation=rotation,
        translation=np.array([100.0, -40.0, 12.5]),
        mode="engineering",
    )
    target = expected.apply(source)
    fitted, report = fit_similarity(source, target)
    assert np.allclose(fitted.apply(source), target, atol=1e-10)
    assert np.isclose(fitted.scale, expected.scale)
    assert np.allclose(fitted.rotation, expected.rotation)
    assert report["rmse_3d"] < 1e-10


def test_independent_check_report_is_not_refit():
    transform = SimilarityTransform(scale=2.0, mode="engineering")
    model = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    target = transform.apply(model)
    target[1, 2] += 0.1
    report = residual_report(transform, model, target)
    assert report["check_count"] == 2
    assert report["max_3d"] == pytest.approx(0.1)


def test_collinear_control_points_rejected():
    source = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="共线"):
        fit_similarity(source, source * 2)


def test_robust_similarity_rejects_one_bad_control_point():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.5],
        ]
    )
    expected = SimilarityTransform(
        scale=2.5,
        translation=np.array([100.0, 50.0, 10.0]),
        mode="engineering",
    )
    target = expected.apply(source)
    target[-1] += np.array([20.0, -15.0, 8.0])

    fitted, report = fit_similarity_robust(
        source,
        target,
        threshold=0.05,
        max_trials=256,
    )

    assert report["inlier_count"] == 5
    assert report["outlier_count"] == 1
    assert report["inlier_mask"][-1] is False
    assert np.allclose(fitted.apply(source[:-1]), target[:-1], atol=1e-9)
