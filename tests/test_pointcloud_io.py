from pathlib import Path

import numpy as np

from ai_photogrammetry.engineering.exporters import write_binary_ply
from ai_photogrammetry.engineering.pointcloud_io import load_ply_preview


def test_binary_ply_preview_reads_distributed_blocks(tmp_path: Path):
    count = 10_000
    points = np.column_stack(
        (
            np.arange(count, dtype=np.float64),
            np.zeros(count),
            np.ones(count),
        )
    )
    colors = np.column_stack(
        (
            np.arange(count, dtype=np.uint32) % 255,
            np.full(count, 20),
            np.full(count, 30),
        )
    ).astype(np.uint8)
    path = tmp_path / "large.ply"
    write_binary_ply(path, points, colors, [])

    preview, preview_colors, total = load_ply_preview(
        path,
        max_points=100,
        block_count=10,
    )

    assert total == count
    assert len(preview) == 100
    assert preview[0, 0] == 0
    assert preview[-1, 0] == count - 1
    assert preview_colors.shape == (100, 3)


def test_binary_ply_preview_reads_all_small_cloud_points(tmp_path: Path):
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    path = tmp_path / "small.ply"
    write_binary_ply(path, points, colors, [])

    preview, preview_colors, total = load_ply_preview(path, max_points=10)

    assert total == 2
    np.testing.assert_allclose(preview, points)
    np.testing.assert_array_equal(preview_colors, colors)


def test_binary_ply_preview_supports_colmap_crlf_header(tmp_path: Path):
    path = tmp_path / "colmap.ply"
    dtype = np.dtype(
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
    records = np.zeros(2, dtype=dtype)
    records["x"] = [1, 4]
    records["y"] = [2, 5]
    records["z"] = [3, 6]
    records["red"] = [7, 10]
    records["green"] = [8, 11]
    records["blue"] = [9, 12]
    header = (
        "ply\r\nformat binary_little_endian 1.0\r\n"
        "element vertex 2\r\n"
        "property float x\r\nproperty float y\r\nproperty float z\r\n"
        "property float nx\r\nproperty float ny\r\nproperty float nz\r\n"
        "property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\n"
        "end_header\r\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)

    preview, colors, total = load_ply_preview(path, max_points=10)

    assert total == 2
    np.testing.assert_allclose(preview, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(colors, [[7, 8, 9], [10, 11, 12]])
