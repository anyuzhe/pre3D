import numpy as np
import pytest

from ai_photogrammetry.engineering.coordinate_systems import (
    CoordinateReference,
    transform_wgs84,
    wgs84_to_local_enu,
    wgs84_to_projected,
)
from ai_photogrammetry.engineering.session import ProjectSession


def test_wgs84_to_local_enu_has_metric_axes():
    origin = (120.0, 30.0, 100.0)
    points = np.array(
        [
            origin,
            (120.00001, 30.0, 100.0),
            (120.0, 30.00001, 100.0),
            (120.0, 30.0, 102.0),
        ]
    )
    enu = wgs84_to_local_enu(points, origin)

    assert np.allclose(enu[0], 0.0, atol=1e-8)
    assert 0.9 < enu[1, 0] < 1.1
    assert 1.0 < enu[2, 1] < 1.2
    np.testing.assert_allclose(enu[3, 2], 2.0, atol=1e-5)


def test_session_converts_geographic_control_points_to_one_enu_frame():
    session = ProjectSession()
    session.add_geographic_coordinate_observation(
        point_id="GCP01",
        model_xyz=np.zeros(3),
        longitude=120.0,
        latitude=30.0,
        height=100.0,
        role="control",
    )
    second = session.add_geographic_coordinate_observation(
        point_id="GCP02",
        model_xyz=np.ones(3),
        longitude=120.00001,
        latitude=30.0,
        height=100.0,
        role="control",
    )

    assert session.coordinate_reference.mode == "wgs84_enu"
    assert np.allclose(session.coordinate_observations[0].target_xyz, 0.0, atol=1e-8)
    assert second.target_xyz[0] > 0.9


def test_coordinate_reference_round_trip():
    reference = CoordinateReference(
        mode="wgs84_enu",
        source_crs="EPSG:4979",
        target_crs="LOCAL_ENU",
        origin_longitude=120.0,
        origin_latitude=30.0,
        origin_height=50.0,
    )
    restored = CoordinateReference.from_dict(reference.to_dict())
    transformed = transform_wgs84([(120.0, 30.0, 50.0)], restored)
    assert restored == reference
    assert np.allclose(transformed, 0.0, atol=1e-8)


def test_geographic_target_crs_is_rejected_as_non_metric():
    with pytest.raises(ValueError, match="不能作为米制工程坐标"):
        wgs84_to_projected([(120.0, 30.0, 50.0)], "EPSG:4326")
