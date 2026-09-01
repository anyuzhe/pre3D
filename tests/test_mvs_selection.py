from pathlib import Path

from ai_photogrammetry.engineering.mvs_selection import (
    read_sparse_views,
    select_mvs_references,
)


def _write_sparse_images(path: Path) -> None:
    rows = ["# synthetic sparse model"]
    point_sets = (
        (1, 2, 3),
        (2, 3, 4),
        (4, 5, 6),
        (6, 7, 8),
        (8, 9, 10),
    )
    for index, points in enumerate(point_sets, start=1):
        rows.append(f"{index} 1 0 0 0 {index} 0 0 1 image_{index:02d}.jpg")
        rows.append(
            " ".join(
                f"{point * 10}.0 {point * 5}.0 {point}"
                for point in points
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_sparse_view_parser_and_covisibility_reference_selection(tmp_path: Path):
    images = tmp_path / "images.txt"
    _write_sparse_images(images)

    views = read_sparse_views(images)
    selection = select_mvs_references(
        images,
        reference_ratio=0.6,
        strategy="covisibility",
    )

    assert len(views) == 5
    assert views[0].point_ids == frozenset({1, 2, 3})
    assert selection["reference_image_count"] == 3
    assert selection["helper_source_image_count"] == 2
    assert selection["reference_images"][0] == "image_01.jpg"
    assert selection["reference_images"][-1] == "image_05.jpg"
    assert 0.7 <= selection["sparse_point_coverage_ratio"] <= 1.0


def test_all_reference_strategy_keeps_every_registered_image(tmp_path: Path):
    images = tmp_path / "images.txt"
    _write_sparse_images(images)

    selection = select_mvs_references(
        images,
        reference_ratio=0.5,
        strategy="all",
    )

    assert selection["reference_image_count"] == 5
    assert selection["helper_source_image_count"] == 0
    assert selection["sparse_point_coverage_ratio"] == 1.0


def test_all_reference_strategy_skips_pose_only_images(tmp_path: Path):
    images = tmp_path / "images.txt"
    _write_sparse_images(images)
    with images.open("a", encoding="utf-8") as stream:
        stream.write("6 1 0 0 0 6 0 0 1 pose_only.jpg\n")
        stream.write("10.0 20.0 -1 30.0 40.0 -1\n")

    selection = select_mvs_references(
        images,
        reference_ratio=1.0,
        strategy="all",
    )

    assert selection["registered_image_count"] == 6
    assert selection["depth_supported_image_count"] == 5
    assert selection["reference_image_count"] == 5
    assert selection["helper_source_image_count"] == 1
    assert selection["excluded_reference_images"] == ["pose_only.jpg"]
    assert "pose_only.jpg" not in selection["reference_images"]
