from pathlib import Path

import numpy as np

from ai_photogrammetry.engineering.mvs_selection import SparseView, read_sparse_views
from ai_photogrammetry.engineering.spatial_blocks import (
    _augment_reference_coverage,
    crop_fused_ply_to_core,
    plan_spatial_blocks,
    read_sparse_points,
    write_block_patch_match_config,
)

FUSED_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def _write_corridor_model(root: Path) -> tuple[Path, Path]:
    images = root / "images.txt"
    points = root / "points3D.txt"
    point_rows = ["# synthetic corridor"]
    for point_id in range(1, 81):
        point_rows.append(
            f"{point_id} {point_id:.4f} {0.03 * (point_id % 3):.4f} 0 "
            "100 120 140 0.2"
        )
    points.write_text("\n".join(point_rows) + "\n", encoding="utf-8")

    image_rows = ["# synthetic views"]
    for image_id in range(1, 17):
        start = max(1, (image_id - 1) * 5 - 2)
        observed = list(range(start, min(81, start + 13)))
        image_rows.append(
            f"{image_id} 1 0 0 0 {image_id} 0 0 1 image_{image_id:02d}.jpg"
        )
        image_rows.append(
            " ".join(f"{point_id}.0 0.0 {point_id}" for point_id in observed)
        )
    images.write_text("\n".join(image_rows) + "\n", encoding="utf-8")
    return images, points


def test_corridor_plan_uses_core_halo_and_visibility(tmp_path: Path):
    images, points = _write_corridor_model(tmp_path)
    plan = plan_spatial_blocks(
        images,
        points,
        target_images=8,
        halo_ratio=0.20,
        min_core_points=3,
    )

    assert plan["strategy"] == "pca_corridor"
    assert plan["block_count"] >= 2
    assert plan["block_count"] <= int(np.ceil(16 / 8))
    assert plan["assigned_image_count"] == 16
    assert all(block["image_count"] >= 3 for block in plan["blocks"])
    assert plan["total_block_image_assignments"] > plan["registered_image_count"]
    references = [
        name
        for block in plan["blocks"]
        for name in block["reference_images"]
    ]
    assert plan["version"] == 3
    assert plan["total_reference_image_assignments"] >= 16
    assert len(set(references)) == plan["unique_reference_image_count"] == 16
    assert all(
        block["core_reference_coverage_ratio"] >= 0.95
        for block in plan["blocks"]
    )
    # Corridor blocks only partition PCA axis 0. The other Core dimensions
    # must extend beyond sparse bounds so dense rock thickness is not clipped.
    _, xyz = read_sparse_points(points)
    frame = plan["coordinate_frame"]
    local = (
        xyz - np.asarray(frame["center"], dtype=np.float64)
    ) @ np.asarray(frame["basis"], dtype=np.float64)
    assert all(
        all(
            block["core_lower"][axis] < float(np.min(local[:, axis]))
            and block["core_upper"][axis] > float(np.max(local[:, axis]))
            for axis in (1, 2)
        )
        for block in plan["blocks"]
    )


def test_reference_coverage_promotes_halo_views_for_two_view_support():
    views = [
        SparseView(1, "owner.jpg", frozenset({1, 2})),
        SparseView(2, "halo_a.jpg", frozenset({1, 2, 3})),
        SparseView(3, "halo_b.jpg", frozenset({3, 4})),
        SparseView(4, "halo_c.jpg", frozenset({4})),
    ]

    selected, coverage, zero_ratio, added = _augment_reference_coverage(
        ["owner.jpg"],
        [view.name for view in views],
        {1, 2, 3, 4},
        {view.name.casefold(): view for view in views},
        minimum_observations=2,
        target_coverage=1.0,
    )

    assert selected == ["owner.jpg", "halo_a.jpg", "halo_b.jpg", "halo_c.jpg"]
    assert coverage == 1.0
    assert zero_ratio == 0.0
    assert added == 3


def test_block_patch_match_config_never_uses_remote_auto_sources(tmp_path: Path):
    images, _ = _write_corridor_model(tmp_path)
    views = read_sparse_views(images)
    selected = [view.name for view in views[:8]]
    config = tmp_path / "patch-match.cfg"

    count = write_block_patch_match_config(
        config,
        selected,
        views,
        source_count=4,
    )

    lines = config.read_text(encoding="utf-8").splitlines()
    assert count == 8
    assert "__auto__" not in "\n".join(lines)
    assert set(lines[::2]) == set(selected)
    assert all(
        source.strip() in selected
        for row in lines[1::2]
        for source in row.split(",")
    )


def test_halo_sources_are_not_promoted_to_reference_views(tmp_path: Path):
    images, _ = _write_corridor_model(tmp_path)
    views = read_sparse_views(images)
    references = [view.name for view in views[:4]]
    source_pool = [view.name for view in views[:8]]
    config = tmp_path / "patch-match.cfg"

    count = write_block_patch_match_config(
        config,
        references,
        views,
        source_count=6,
        source_image_names=source_pool,
        stable_source_image_names=references,
        minimum_stable_sources=3,
    )

    lines = config.read_text(encoding="utf-8").splitlines()
    assert count == 4
    assert set(lines[::2]) == set(references)
    assert any(
        source.strip() in source_pool[4:]
        for row in lines[1::2]
        for source in row.split(",")
    )
    reference_set = set(references)
    for reference, row in zip(lines[::2], lines[1::2]):
        stable_sources = {
            source.strip() for source in row.split(",")
        } & (reference_set - {reference})
        assert len(stable_sources) >= 3


def test_patch_match_prefers_a_full_set_of_available_dense_map_sources(
    tmp_path: Path,
):
    views = [
        SparseView(1, "reference.jpg", frozenset({1, 2, 3, 4})),
        SparseView(2, "stable_1.jpg", frozenset({1, 2})),
        SparseView(3, "stable_2.jpg", frozenset({1, 3})),
        SparseView(4, "stable_3.jpg", frozenset({1, 4})),
        SparseView(5, "stable_4.jpg", frozenset({2, 3})),
        SparseView(6, "missing_map.jpg", frozenset({1, 2, 3, 4})),
    ]
    config = tmp_path / "patch-match.cfg"
    stable = [view.name for view in views[1:5]]

    write_block_patch_match_config(
        config,
        ["reference.jpg"],
        views,
        source_count=4,
        source_image_names=[view.name for view in views],
        stable_source_image_names=["reference.jpg", *stable],
        minimum_stable_sources=3,
    )

    sources = {
        value.strip()
        for value in config.read_text(encoding="utf-8").splitlines()[1].split(",")
    }
    assert sources == set(stable)
    assert "missing_map.jpg" not in sources


def test_patch_match_config_skips_reference_without_sparse_depth_support(
    tmp_path: Path,
):
    images, _ = _write_corridor_model(tmp_path)
    with images.open("a", encoding="utf-8") as stream:
        stream.write("17 1 0 0 0 17 0 0 1 pose_only.jpg\n")
        stream.write("10.0 20.0 -1 30.0 40.0 -1\n")
    views = read_sparse_views(images)
    references = [view.name for view in views]
    config = tmp_path / "patch-match.cfg"

    count = write_block_patch_match_config(
        config,
        references,
        views,
        source_count=4,
    )

    lines = config.read_text(encoding="utf-8").splitlines()
    assert count == 16
    assert "pose_only.jpg" not in lines[::2]


def test_binary_fused_cloud_is_stream_cropped_to_core(tmp_path: Path):
    source = tmp_path / "fused.ply"
    output = tmp_path / "core.ply"
    records = np.zeros(10, dtype=FUSED_DTYPE)
    records["x"] = np.arange(10, dtype=np.float32)
    records["nz"] = 1.0
    records["red"] = 100
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 10\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with source.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)

    count = crop_fused_ply_to_core(
        source,
        output,
        center=[0.0, 0.0, 0.0],
        basis=np.eye(3),
        lower=[2.0, -1.0, -1.0],
        upper=[5.0, 1.0, 1.0],
    )

    assert count == 4
    assert b"element vertex 4" in output.read_bytes()[:256]
