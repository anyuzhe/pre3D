import sqlite3
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from ai_photogrammetry.engineering.colmap_pipeline import (
    _configure_patch_match,
    _cuda_out_of_memory,
    _database_matches_images,
    _dense_map_valid,
    _estimate_dense_workspace_bytes,
    _evaluate_sparse_quality_gate,
    _fusion_resources,
    _merge_colmap_fused_plys,
    _plan_fusion_batches,
    _patch_match_dependency_arguments,
    _prepare_colmap_image_paths,
    _ply_vertex_count,
    _remove_invalid_dense_maps,
    _remove_photometric_maps_after_geometric_patchmatch,
    _remove_stereo_fusion_output,
    _set_patch_match_source_count,
    _sparse_text_quality,
    _stereo_fusion_arguments,
)


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8)).save(path)


def test_patch_match_source_count_is_applied_to_every_reference_image(
    tmp_path: Path,
):
    config = tmp_path / "patch-match.cfg"
    config.write_text(
        "first.jpg\n__auto__, 20\nsecond.jpg\nfirst.jpg\n",
        encoding="utf-8",
    )

    count = _set_patch_match_source_count(config, 12)

    assert count == 2
    assert config.read_text(encoding="utf-8") == (
        "first.jpg\n__auto__, 12\nsecond.jpg\n__auto__, 12\n"
    )


def test_patch_match_configuration_can_keep_only_selected_reference_images(
    tmp_path: Path,
):
    config = tmp_path / "patch-match.cfg"
    config.write_text(
        "first.jpg\n__auto__, 20\n"
        "second.jpg\n__auto__, 20\n"
        "third.jpg\n__auto__, 20\n",
        encoding="utf-8",
    )

    count = _configure_patch_match(
        config,
        10,
        ["first.jpg", "third.jpg"],
    )

    assert count == 2
    assert config.read_text(encoding="utf-8") == (
        "first.jpg\n__auto__, 10\nthird.jpg\n__auto__, 10\n"
    )


def test_subset_geometric_mvs_allows_missing_helper_depth_maps():
    assert _patch_match_dependency_arguments(
        registered_images=58,
        reference_images=44,
        geometric_consistency=True,
    ) == ["--PatchMatchStereo.allow_missing_files", "1"]
    assert not _patch_match_dependency_arguments(
        registered_images=58,
        reference_images=58,
        geometric_consistency=True,
    )
    assert not _patch_match_dependency_arguments(
        registered_images=58,
        reference_images=44,
        geometric_consistency=False,
    )


def test_sparse_quality_gate_distinguishes_pass_review_and_block():
    passed = _evaluate_sparse_quality_gate(
        {
            "registered_images": 80,
            "registration_ratio": 0.98,
            "sparse_point_count": 20_000,
            "mean_reprojection_error_px": 1.2,
        }
    )
    review = _evaluate_sparse_quality_gate(
        {
            "registered_images": 70,
            "registration_ratio": 0.9,
            "sparse_point_count": 10_000,
            "mean_reprojection_error_px": 2.5,
        }
    )
    blocked = _evaluate_sparse_quality_gate(
        {
            "registered_images": 10,
            "registration_ratio": 0.2,
            "sparse_point_count": 50,
            "mean_reprojection_error_px": 8.0,
        }
    )

    assert passed["status"] == "passed"
    assert review["status"] == "review"
    assert blocked["status"] == "blocked"


def test_colmap_run_directory_contains_only_current_session_images(tmp_path: Path):
    source = tmp_path / "source"
    first = source / "first.jpg"
    second = source / "second.jpg"
    stale = source / "stale.jpg"
    for index, path in enumerate((first, second, stale)):
        _image(path, index)
    run_root = tmp_path / "run"

    expected = _prepare_colmap_image_paths(
        [str(first), str(second)],
        [first.name, second.name],
        run_root / "images",
        run_root / "input_images.json",
    )

    assert expected == ["first.jpg", "second.jpg"]
    assert {path.name for path in (run_root / "images").iterdir()} == set(expected)
    assert not (run_root / "images" / stale.name).exists()


def test_colmap_database_resume_requires_exact_image_names(tmp_path: Path):
    database = tmp_path / "database.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE images(image_id INTEGER, camera_id INTEGER, name TEXT)"
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, ?, ?)",
            [
                (1, 1, "first.jpg"),
                (2, 2, "second.jpg"),
                (3, 3, "stale.jpg"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    assert not _database_matches_images(database, ["first.jpg", "second.jpg"])
    assert _database_matches_images(
        database,
        ["first.jpg", "second.jpg", "stale.jpg"],
    )

def test_corrupt_dense_map_is_detected_and_removed(tmp_path: Path):
    depth_maps = tmp_path / "dense" / "stereo" / "depth_maps"
    depth_maps.mkdir(parents=True)
    valid = depth_maps / "valid.geometric.bin"
    header = b"3&2&1&"
    valid.write_bytes(header + struct.pack("<6f", *range(6)))
    corrupt = depth_maps / "corrupt.photometric.bin"
    corrupt.write_bytes(b"\x00")

    assert _dense_map_valid(valid)
    assert not _dense_map_valid(corrupt)
    removed = _remove_invalid_dense_maps(tmp_path / "dense")
    assert removed == [str(corrupt)]
    assert valid.is_file()
    assert not corrupt.exists()


def test_dense_workspace_estimate_scales_with_resolution(tmp_path: Path):
    image = tmp_path / "large.jpg"
    Image.fromarray(np.zeros((1200, 1600, 3), dtype=np.uint8)).save(image)

    small = _estimate_dense_workspace_bytes([str(image)], 800)
    large = _estimate_dense_workspace_bytes([str(image)], 1600)

    assert large > small * 3


def test_dense_workspace_estimate_accounts_for_geometric_maps(tmp_path: Path):
    image = tmp_path / "large.jpg"
    Image.fromarray(np.zeros((1200, 1600, 3), dtype=np.uint8)).save(image)

    photometric = _estimate_dense_workspace_bytes(
        [str(image)],
        1600,
        False,
    )
    geometric = _estimate_dense_workspace_bytes(
        [str(image)],
        1600,
        True,
    )

    assert geometric > photometric * 1.4


def test_photometric_maps_are_removed_after_geometric_patchmatch(
    tmp_path: Path,
):
    stereo = tmp_path / "dense" / "stereo"
    depth = stereo / "depth_maps"
    normal = stereo / "normal_maps"
    depth.mkdir(parents=True)
    normal.mkdir()
    files = {
        depth / "a.photometric.bin": b"photo-depth",
        depth / "a.geometric.bin": b"geo-depth",
        normal / "a.photometric.bin": b"photo-normal",
        normal / "a.geometric.bin": b"geo-normal",
    }
    for path, payload in files.items():
        path.write_bytes(payload)

    removed, freed = _remove_photometric_maps_after_geometric_patchmatch(
        tmp_path / "dense"
    )

    assert removed == 2
    assert freed == len(b"photo-depth") + len(b"photo-normal")
    assert not (depth / "a.photometric.bin").exists()
    assert not (normal / "a.photometric.bin").exists()
    assert (depth / "a.geometric.bin").is_file()
    assert (normal / "a.geometric.bin").is_file()


def test_cuda_oom_detection_is_specific():
    assert _cuda_out_of_memory(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert _cuda_out_of_memory(RuntimeError("failed to allocate device buffer"))
    assert not _cuda_out_of_memory(RuntimeError("input image is missing"))


def test_stereo_fusion_uses_bounded_disk_cache(tmp_path: Path):
    cache_size_gb, num_threads = _fusion_resources()
    arguments = _stereo_fusion_arguments(
        tmp_path / "dense",
        tmp_path / "fused.partial.ply",
        geometric_consistency=True,
        min_num_pixels=2,
        check_num_images=18,
        use_cache=True,
        cache_size_gb=cache_size_gb,
        num_threads=num_threads,
    )

    def option(name: str) -> str:
        return arguments[arguments.index(name) + 1]

    assert option("--StereoFusion.use_cache") == "1"
    assert option("--StereoFusion.check_num_images") == "18"
    assert 2 <= int(option("--StereoFusion.cache_size")) <= 8
    assert 1 <= int(option("--StereoFusion.num_threads")) <= 12
    assert option("--input_type") == "geometric"
    assert option("--output_path").endswith("fused.partial.ply")


def test_large_dense_workspace_is_split_into_full_resolution_batches(tmp_path: Path):
    dense = tmp_path / "dense"
    names = [f"image_{index:03d}.jpg" for index in range(100)]
    for name in names:
        for folder, channels in (("depth_maps", 1), ("normal_maps", 3)):
            path = dense / "stereo" / folder / f"{name}.geometric.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            header = f"4&3&{channels}&".encode()
            path.write_bytes(header + bytes(4 * 3 * channels * 4))

    batches, total_bytes = _plan_fusion_batches(
        dense,
        names,
        input_type="geometric",
        memory_capacity_bytes=1,
    )

    assert total_bytes > 0
    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [48, 48, 4]
    assert [name for batch in batches for name in batch] == names


def test_binary_colmap_ply_batches_are_stream_merged(tmp_path: Path):
    def write(path: Path, points: list[tuple[float, ...]]) -> None:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        payload = b"".join(struct.pack("<6f3B", *point) for point in points)
        path.write_bytes(header + payload)

    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"
    output = tmp_path / "merged.ply"
    write(first, [(0, 0, 0, 0, 0, 1, 255, 0, 0)])
    write(second, [(1, 0, 0, 0, 0, 1, 0, 255, 0), (2, 0, 0, 0, 0, 1, 0, 0, 255)])

    assert _merge_colmap_fused_plys([first, second], output) == 3
    assert _ply_vertex_count(output) == 3


def test_fusion_batch_cleanup_removes_visibility_sidecar(tmp_path: Path):
    pointcloud = tmp_path / "batch_0001.ply"
    visibility = tmp_path / "batch_0001.ply.vis"
    pointcloud.write_bytes(b"pointcloud")
    visibility.write_bytes(b"visibility")

    assert _remove_stereo_fusion_output(pointcloud) == 20
    assert not pointcloud.exists()
    assert not visibility.exists()


def test_sparse_quality_reports_registered_and_unregistered_images(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 first.jpg\n"
        "10 20 1\n"
        "2 1 0 0 0 1 0 0 1 second.jpg\n"
        "11 21 2\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text(
        "# 3D point list\n"
        "1 0 0 0 255 0 0 0.5 1 0\n"
        "2 1 0 0 0 255 0 1.5 2 0\n",
        encoding="utf-8",
    )

    result = _sparse_text_quality(
        model,
        ["first.jpg", "second.jpg", "third.jpg"],
    )

    assert result["registered_images"] == 2
    assert result["unregistered_images"] == ["third.jpg"]
    assert result["sparse_point_count"] == 2
    assert result["mean_reprojection_error_px"] == 1.0
