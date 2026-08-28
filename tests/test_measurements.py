import numpy as np
import pytest

from ai_photogrammetry.engineering.measurements import (
    convex_hull_volume,
    plane_spacing,
    polygon_area,
    polyline_length,
    straight_distance,
)


def test_basic_metric_measurements():
    assert straight_distance([[0, 0, 0], [3, 4, 0]]) == pytest.approx(5.0)
    assert polyline_length([[0, 0, 0], [3, 4, 0], [3, 4, 2]]) == pytest.approx(7.0)
    assert polygon_area([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]]) == pytest.approx(6.0)


def test_convex_hull_volume():
    tetrahedron = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert convex_hull_volume(tetrahedron) == pytest.approx(1 / 6)


def test_plane_spacing():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 2],
            [1, 0, 2],
            [0, 1, 2],
        ],
        dtype=float,
    )
    assert plane_spacing(points, 3) == pytest.approx(2.0)
