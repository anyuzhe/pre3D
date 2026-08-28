import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ai_photogrammetry.engineering.exporters import write_binary_ply
from ai_photogrammetry.engineering.photo_selection import (
    _dji_xmp_metadata,
    _exif_value,
    _gps_altitude_reference,
    _gps_decimal,
    _orientation_display_info,
    analyze_sequence_continuity,
    recommended_continuous_segment,
    scan_photos,
    select_keyframes,
)
from ai_photogrammetry.engineering.point_processing import FilterOptions, process_session_cloud
from ai_photogrammetry.engineering.project_store import ProjectStore, StageTracker
from ai_photogrammetry.engineering.session import ProjectSession


def _image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    Image.fromarray(array).save(path, "PNG")


def test_gps_exif_coordinates_are_converted_to_decimal_degrees():
    assert _gps_decimal((30, 15, 0), "N") == 30.25
    assert _gps_decimal((120, 30, 0), "W") == -120.5
    assert _gps_altitude_reference(b"\x00") == 0
    assert _gps_altitude_reference(b"\x01") == 1
    assert _gps_altitude_reference(1) == 1
    assert _gps_altitude_reference("invalid") is None


def test_dji_xmp_rtk_metadata_is_classified_with_sigmas(tmp_path: Path):
    photo = tmp_path / "dji.jpg"
    photo.write_bytes(
        b'jpeg-prefix drone-dji:GpsStatus="RTK" '
        b'drone-dji:GpsLatitude="+29.934832960" '
        b'drone-dji:GpsLongitude="+120.621124486" '
        b'drone-dji:AbsoluteAltitude="+47.972" '
        b'drone-dji:RtkFlag="50" '
        b'drone-dji:RtkStdLon="0.00231" '
        b'drone-dji:RtkStdLat="0.00328" '
        b'drone-dji:RtkStdHgt="0.01001" '
        b'drone-dji:GimbalYawDegree="55.80"'
    )

    metadata = _dji_xmp_metadata(photo)

    assert metadata["rtk_status"] == "RTK_FIX"
    assert metadata["gps_source"] == "DJI_XMP"
    assert metadata["sigma_x"] == 0.00231
    assert metadata["yaw"] == 55.8


def test_exif_orientation_is_normal_information_not_a_quality_warning():
    assert _orientation_display_info(1) == ""
    assert _orientation_display_info(3) == "显示方向已自动旋转180°（正常）"
    assert _orientation_display_info(6) == "显示方向已自动顺时针旋转90°（正常）"
    assert _orientation_display_info(8) == "显示方向已自动逆时针旋转90°（正常）"


def test_project_manifest_keeps_first_version_creation_choices(tmp_path: Path):
    store = ProjectStore.create(
        tmp_path / "slope",
        "北侧边坡",
        project_type="近景岩体重建",
        output_coordinate_system="现场独立坐标系",
        precision_mode="高精度模式",
    )

    manifest = store.read_manifest()

    assert manifest["project_type"] == "近景岩体重建"
    assert manifest["output_coordinate_system"] == "现场独立坐标系"
    assert manifest["precision_mode"] == "高精度模式"


def test_resetting_stage_to_pending_clears_old_terminal_timestamps(
    tmp_path: Path,
):
    tracker = StageTracker(tmp_path / "stages.json")
    tracker.set("patch_match", "running", message="running")
    tracker.set("patch_match", "failed", message="failed")
    tracker.set("patch_match", "pending", message="resume later")

    record = tracker.data["stages"]["patch_match"]
    assert record["status"] == "pending"
    assert "started_at" not in record
    assert "finished_at" not in record


def test_project_store_recovers_all_orphaned_pipeline_stages(tmp_path: Path):
    store = ProjectStore.create(tmp_path / "project", "断点恢复")
    store.stage_tracker.set("sparse_ba", "running", message="空三")
    sparse_state = (
        store.root
        / "colmap"
        / "photogrammetry_test"
        / "pipeline_state.json"
    )
    dense_state = (
        store.root
        / "colmap"
        / "photogrammetry_test"
        / "dense_test"
        / "pipeline_state.json"
    )
    sparse_tracker = StageTracker(sparse_state)
    sparse_tracker.set("ai_feature_extraction", "completed")
    sparse_tracker.set("ai_feature_matching", "running", message="匹配")
    dense_tracker = StageTracker(dense_state)
    dense_tracker.set("patch_match", "running", message="深度")

    recovered = store.recover_interrupted_stages()

    assert {item["stage"] for item in recovered} == {
        "sparse_ba",
        "ai_feature_matching",
        "patch_match",
    }
    assert StageTracker(store.stage_tracker.path).status("sparse_ba") == "interrupted"
    assert StageTracker(sparse_state).status("ai_feature_matching") == "interrupted"
    assert StageTracker(sparse_state).status("ai_feature_extraction") == "completed"
    assert StageTracker(dense_state).status("patch_match") == "interrupted"


def test_project_roundtrip_ai_cloud_and_processed_cache(tmp_path: Path):
    source = tmp_path / "source.png"
    _image(source, 1)
    store = ProjectStore.create(tmp_path / "project", "持久化测试")
    session = ProjectSession(project_name="持久化测试", project_id="persistent")
    yy, xx = np.mgrid[:10, :10]
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3).astype(float)
    points[-1] = [1000, 1000, 1000]
    dense = tmp_path / "dense.ply"
    write_binary_ply(
        dense,
        points,
        np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(points), 1)),
        ["test cloud"],
    )
    session.photogrammetry_result = {
        "pipeline": "AI局部特征 + 传统摄影测量",
        "pointcloud": str(dense),
        "image_count": 1,
        "registered_images": 1,
    }
    session.photogrammetry_options = {
        "feature_type": "aliked",
        "mapper": "global",
        "output_root": str(tmp_path / "photogrammetry"),
    }
    session.sparse_result = {
        "result_stage": "sparse",
        "registered_images": 1,
        "image_count": 1,
        "sparse_point_count": 42,
    }
    model = tmp_path / "textured_mesh.ply"
    model.write_bytes(b"test model")
    texture = tmp_path / "texture.png"
    texture.write_bytes(b"test texture")
    session.model_options = {
        "generate_model": True,
        "formats": ["obj", "gltf"],
    }
    session.model_result = {
        "textured_mesh": str(model),
        "texture_atlas": str(texture),
        "face_count": 120,
        "formats": {"obj": str(tmp_path / "model.obj")},
    }
    report = process_session_cloud(
        session,
        FilterOptions(distance_mad_multiplier=4),
        source_pointcloud=str(dense),
    )
    assert report["output_point_count"] < report["input_point_count"]
    assert np.max(session.processed_points) < 100
    store.save_processed_cache(session)
    store.save_session(
        session,
        source_images=[str(source)],
        selected_images=[str(source)],
    )

    restored = ProjectStore.open(store.project_file).load_session()

    assert restored.has_geometry
    assert restored.has_processed_cloud
    assert restored.photogrammetry_options == session.photogrammetry_options
    assert restored.sparse_result == session.sparse_result
    assert restored.photogrammetry_result == session.photogrammetry_result
    assert restored.has_model
    assert restored.model_options == session.model_options
    assert restored.model_result == session.model_result
    assert restored.filter_report["output_point_count"] == len(restored.processed_points)


def test_duplicate_detection_and_keyframe_limit(tmp_path: Path):
    paths = []
    for index in range(6):
        path = tmp_path / f"{index:02d}.png"
        _image(path, index)
        paths.append(str(path))
    duplicate = tmp_path / "06_duplicate.png"
    shutil.copy2(paths[2], duplicate)
    paths.append(str(duplicate))

    records, summary = scan_photos(paths)
    selected = select_keyframes(records, max_count=3)

    assert summary["exact_duplicate_count"] == 1
    assert len(selected) == 3
    assert str(duplicate.resolve()) not in selected


def test_scan_worker_isolated_process(tmp_path: Path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for index in range(4):
        _image(photos / f"{index:02d}.png", index)
    store = ProjectStore.create(tmp_path / "project", "worker")
    config = {
        "task": "scan_photos",
        "project_root": str(store.root),
        "source_root": str(photos),
        "max_keyframes": 2,
    }
    config_path = tmp_path / "worker.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_photogrammetry.engineering.worker",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    events = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    terminal = [event for event in events if event.get("type") == "result"]

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert terminal[-1]["status"] == "completed"
    manifest = store.read_manifest()
    assert len(manifest["source_images"]) == 4
    assert len(manifest["selected_images"]) == 2


def test_nested_phone_exif_values_are_read():
    class NestedExif(dict):
        def get_ifd(self, ifd: int):
            assert ifd == 34665
            return {
                36867: "2025:08:07 16:37:32",
                37386: 5.7,
                42036: "Phone back camera 5.7mm",
            }

    exif = NestedExif({272: "Phone"})

    assert _exif_value(exif, "DateTimeOriginal") == "2025:08:07 16:37:32"
    assert _exif_value(exif, "FocalLength") == 5.7
    assert _exif_value(exif, "LensModel") == "Phone back camera 5.7mm"
    assert _exif_value(exif, "Model") == "Phone"


def test_sequence_preflight_splits_disconnected_capture(tmp_path: Path):
    rng = np.random.default_rng(123)
    first_texture = rng.integers(0, 256, (420, 760, 3), dtype=np.uint8)
    second_texture = rng.integers(0, 256, (420, 760, 3), dtype=np.uint8)
    paths: list[str] = []
    for group, texture in enumerate((first_texture, second_texture)):
        for index, offset in enumerate((0, 55, 110)):
            path = tmp_path / f"{group}_{index}.png"
            Image.fromarray(texture[:, offset : offset + 520]).save(path)
            paths.append(str(path))

    analysis = analyze_sequence_continuity(paths, max_size=600)
    selected, segmentation = recommended_continuous_segment(
        paths,
        analysis,
        minimum_images=3,
    )

    assert any(
        item["first_index"] == 2 and item["second_index"] == 3
        for item in analysis["breaks"]
    )
    assert [item["image_count"] for item in analysis["segments"]] == [3, 3]
    assert segmentation["applied"]
    assert selected == paths[:3]
