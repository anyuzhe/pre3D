"""AI local-feature photogrammetry with geometric SfM, BA, and dense MVS.

COLMAP/GLOMAP owns all camera geometry. Learned ALIKED/SIFT-LightGlue is
used only to make feature extraction and image matching more robust.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import psutil
from PIL import Image

from .mvs_selection import read_sparse_views, select_mvs_references
from .project_store import StageTracker, source_fingerprint
from .runtime_paths import executable_root, resource_root
from .session import ProjectSession
from .spatial_blocks import (
    crop_fused_ply_to_core,
    plan_spatial_blocks,
    save_spatial_plan,
    write_block_patch_match_config,
)

_logger = logging.getLogger(__name__)

_COLMAP_AI_MODELS = {
    "aliked-n16rot.onnx": "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    "aliked-lightglue.onnx": "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    "bruteforce-matcher.onnx": "3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47",
    "sift-lightglue.onnx": "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
}

_COLMAP_PAIR_ID_MAX = 2_147_483_647


def _runtime_library_directories(executable: str) -> list[str]:
    """Find optional CUDA/cuDNN DLL locations without requiring PyTorch."""

    candidates = [
        Path(executable).resolve().parent,
        resource_root(),
        executable_root(),
    ]
    for variable in ("CUDA_PATH", "CUDNN_PATH", "ONNXRUNTIME_CUDA_DLL_DIR"):
        value = os.environ.get(variable)
        if value:
            root = Path(value)
            candidates.extend((root, root / "bin"))
    for value in sys.path:
        if not value:
            continue
        root = Path(value)
        candidates.extend(
            (
                root / "nvidia" / "cudnn" / "bin",
                root / "nvidia" / "cublas" / "bin",
                # Some existing Python environments bundle the exact cuDNN
                # runtime required by ONNX Runtime. This is an optional DLL
                # source only; the application never imports or requires torch.
                root / "torch" / "lib",
            )
        )
    unique: list[str] = []
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except OSError:
            continue
        if candidate.is_dir() and resolved.casefold() not in {
            value.casefold() for value in unique
        }:
            unique.append(resolved)
    return unique


def _verified_colmap_ai_model(name: str) -> Path:
    expected_hash = _COLMAP_AI_MODELS[name]
    path = resource_root() / "checkpoints" / "colmap_ai" / name
    if not path.is_file():
        raise RuntimeError(
            f"缺少AI匹配权重 {name}。请运行 scripts/download_colmap_ai_models.ps1"
        )
    with path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"AI匹配权重 {name} 的SHA‑256校验失败，请重新下载"
        )
    return path


def find_colmap(explicit_path: str | None = None) -> str | None:
    candidates = [explicit_path] if explicit_path else []
    project_root = resource_root()
    candidates.extend(
        [
            str(project_root / "tools" / "colmap" / "bin" / "colmap.exe"),
            str(project_root / "tools" / "colmap" / "COLMAP.bat"),
        ]
    )
    candidates.extend(["colmap", "COLMAP.bat", "colmap.exe"])
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            path = Path(resolved)
        else:
            path = Path(candidate).expanduser()
        if path.is_file():
            if path.suffix.lower() in {".bat", ".cmd"}:
                direct_executable = path.parent / "bin" / "colmap.exe"
                if direct_executable.is_file():
                    return str(direct_executable.resolve())
            return str(path.resolve())
    return None


def _run(
    executable: str,
    arguments: list[str],
    log_path: Path,
    callback: Callable[[str], None] | None = None,
) -> None:
    command = [executable, *arguments]
    _logger.info("Running %s", subprocess.list2cmdline(command))
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    environment = os.environ.copy()
    runtime_directories = _runtime_library_directories(executable)
    existing_path = environment.get("PATH", "")
    environment["PATH"] = os.pathsep.join(
        [*runtime_directories, existing_path]
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {subprocess.list2cmdline(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
            env=environment,
        )
        assert process.stdout is not None
        tail: list[str] = []
        for line in process.stdout:
            log.write(line)
            log.flush()
            clean = line.strip()
            if clean:
                tail.append(clean)
                tail = tail[-25:]
                if callback:
                    callback(clean)
        return_code = process.wait()
    if return_code:
        message = "\n".join(tail[-10:])
        raise RuntimeError(
            f"COLMAP 子命令 {arguments[0]} 失败（退出码 {return_code}）。\n{message}\n完整日志：{log_path}"
        )



def _database_images(database_path: Path) -> list[tuple[int, int, str]]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT image_id, camera_id, name FROM images ORDER BY image_id"
        ).fetchall()
    finally:
        connection.close()
    return [(int(image_id), int(camera_id), str(name)) for image_id, camera_id, name in rows]


def _decode_colmap_pair_id(pair_id: int) -> tuple[int, int]:
    image_id2 = int(pair_id) % _COLMAP_PAIR_ID_MAX
    image_id1 = (int(pair_id) - image_id2) // _COLMAP_PAIR_ID_MAX
    return image_id1, image_id2


def _encode_colmap_pair_id(image_id1: int, image_id2: int) -> int:
    first, second = sorted((int(image_id1), int(image_id2)))
    return first * _COLMAP_PAIR_ID_MAX + second


def _database_verified_match_components(
    database_path: Path,
    *,
    min_num_inliers: int,
) -> list[list[int]]:
    """Return verified image-graph components, largest first."""

    connection = sqlite3.connect(database_path)
    try:
        image_ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT image_id FROM images ORDER BY image_id"
            )
        ]
        edges = connection.execute(
            "SELECT pair_id FROM two_view_geometries WHERE rows >= ?",
            (int(min_num_inliers),),
        ).fetchall()
    finally:
        connection.close()
    image_set = set(image_ids)
    adjacency: dict[int, set[int]] = {image_id: set() for image_id in image_ids}
    for (pair_id,) in edges:
        first, second = _decode_colmap_pair_id(int(pair_id))
        if first in image_set and second in image_set:
            adjacency[first].add(second)
            adjacency[second].add(first)
    components: list[list[int]] = []
    visited: set[int] = set()
    for image_id in image_ids:
        if image_id in visited:
            continue
        component: list[int] = []
        stack = [image_id]
        visited.add(image_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values[0]))


def _database_verified_pair_count(
    database_path: Path,
    *,
    min_num_inliers: int,
) -> int:
    connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE rows >= ?",
            (int(min_num_inliers),),
        ).fetchone()
    finally:
        connection.close()
    return int(row[0]) if row else 0


def _database_global_descriptor_means(
    connection: sqlite3.Connection,
) -> dict[int, np.ndarray]:
    """Build compact retrieval vectors from already cached local features."""

    vectors: dict[int, np.ndarray] = {}
    query = "SELECT image_id, type, rows, cols, data FROM descriptors"
    for image_id, descriptor_type, rows, cols, blob in connection.execute(query):
        row_count = int(rows)
        column_count = int(cols)
        if not blob or row_count <= 0 or column_count <= 0:
            continue
        if int(descriptor_type) == 1:
            dimension = column_count // np.dtype("<f4").itemsize
            if dimension <= 0:
                continue
            values = np.frombuffer(blob, dtype="<f4")
        else:
            dimension = column_count
            values = np.frombuffer(blob, dtype=np.uint8)
        if len(values) != row_count * dimension:
            continue
        descriptors = values.reshape(row_count, dimension)
        vector = descriptors.mean(axis=0, dtype=np.float64).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if math.isfinite(norm) and norm > np.finfo(np.float32).eps:
            vectors[int(image_id)] = vector / norm
    return vectors


def _geodetic_to_ecef(position: tuple[float, float, float]) -> np.ndarray:
    """Convert WGS84 latitude/longitude/altitude to metric ECEF."""

    latitude, longitude, altitude = position
    latitude_rad = math.radians(latitude)
    longitude_rad = math.radians(longitude)
    semi_major = 6_378_137.0
    eccentricity_squared = 6.69437999014e-3
    sin_latitude = math.sin(latitude_rad)
    prime_vertical = semi_major / math.sqrt(
        1.0 - eccentricity_squared * sin_latitude * sin_latitude
    )
    return np.asarray(
        [
            (prime_vertical + altitude)
            * math.cos(latitude_rad)
            * math.cos(longitude_rad),
            (prime_vertical + altitude)
            * math.cos(latitude_rad)
            * math.sin(longitude_rad),
            (
                prime_vertical * (1.0 - eccentricity_squared)
                + altitude
            )
            * sin_latitude,
        ],
        dtype=np.float64,
    )


def _database_pose_prior_positions(
    connection: sqlite3.Connection,
) -> dict[int, np.ndarray]:
    """Read metric camera positions from COLMAP pose priors when available."""

    try:
        rows = connection.execute(
            "SELECT corr_data_id, position, coordinate_system "
            "FROM pose_priors WHERE position IS NOT NULL"
        ).fetchall()
    except sqlite3.Error:
        return {}
    positions: dict[int, np.ndarray] = {}
    for image_id, blob, coordinate_system in rows:
        if blob is None or len(blob) != struct.calcsize("<3d"):
            continue
        raw = tuple(float(value) for value in struct.unpack("<3d", blob))
        if not all(math.isfinite(value) for value in raw):
            continue
        # COLMAP coordinate-system value 0 stores WGS84 latitude, longitude,
        # altitude. Other systems are already Cartesian metric coordinates.
        positions[int(image_id)] = (
            _geodetic_to_ecef(raw)
            if int(coordinate_system) == 0
            else np.asarray(raw, dtype=np.float64)
        )
    return positions


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(max(0, int(count)), len(values))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == len(values):
        return np.argsort(values)[::-1]
    selected = np.argpartition(values, len(values) - count)[-count:]
    return selected[np.argsort(values[selected])[::-1]]


def _bridge_candidate_pairs(
    database_path: Path,
    *,
    min_num_inliers: int,
    retrieval_neighbors: int = 3,
    gps_neighbors: int = 3,
    gps_max_distance_m: float = 150.0,
) -> tuple[list[tuple[str, str]], dict[str, object]]:
    """Propose targeted cross-component pairs from retrieval and GPS priors."""

    components = _database_verified_match_components(
        database_path,
        min_num_inliers=min_num_inliers,
    )
    image_rows = _database_images(database_path)
    names = {image_id: name for image_id, _camera_id, name in image_rows}
    if len(components) <= 1:
        return [], {
            "component_sizes": [len(component) for component in components],
            "retrieval_pairs": 0,
            "gps_pairs": 0,
            "candidate_pairs": 0,
        }

    connection = sqlite3.connect(database_path)
    try:
        descriptors = _database_global_descriptor_means(connection)
        positions = _database_pose_prior_positions(connection)
    finally:
        connection.close()

    main = components[0]
    main_descriptors = [image_id for image_id in main if image_id in descriptors]
    descriptor_matrix = (
        np.stack([descriptors[image_id] for image_id in main_descriptors])
        if main_descriptors
        else np.empty((0, 0), dtype=np.float32)
    )
    main_positions = [image_id for image_id in main if image_id in positions]
    position_matrix = (
        np.stack([positions[image_id] for image_id in main_positions])
        if main_positions
        else np.empty((0, 3), dtype=np.float64)
    )

    retrieval_pairs: set[tuple[int, int]] = set()
    gps_pairs: set[tuple[int, int]] = set()
    for component in components[1:]:
        for image_id in component:
            vector = descriptors.get(image_id)
            if vector is not None and descriptor_matrix.shape[1:] == vector.shape:
                similarities = descriptor_matrix @ vector
                for index in _top_indices(similarities, retrieval_neighbors):
                    retrieval_pairs.add(
                        tuple(sorted((image_id, main_descriptors[int(index)])))
                    )
            position = positions.get(image_id)
            if position is not None and len(position_matrix):
                distances = np.linalg.norm(position_matrix - position, axis=1)
                nearby = np.flatnonzero(distances <= float(gps_max_distance_m))
                if len(nearby):
                    order = nearby[np.argsort(distances[nearby])]
                    for index in order[: max(0, int(gps_neighbors))]:
                        gps_pairs.add(
                            tuple(sorted((image_id, main_positions[int(index)])))
                        )

    combined = sorted(retrieval_pairs | gps_pairs)
    pairs = [(names[first], names[second]) for first, second in combined]
    return pairs, {
        "component_sizes": [len(component) for component in components],
        "retrieval_pairs": len(retrieval_pairs),
        "gps_pairs": len(gps_pairs),
        "candidate_pairs": len(combined),
        "gps_max_distance_m": float(gps_max_distance_m),
    }


def _clear_database_image_pairs(
    database_path: Path,
    pairs: list[tuple[str, str]],
) -> None:
    """Remove failed raw rows only for verified bridge candidates before retry."""

    if not pairs:
        return
    connection = sqlite3.connect(database_path)
    try:
        name_to_id = {
            str(name): int(image_id)
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
        }
        pair_ids = [
            _encode_colmap_pair_id(name_to_id[first], name_to_id[second])
            for first, second in pairs
            if first in name_to_id and second in name_to_id
        ]
        connection.executemany(
            "DELETE FROM matches WHERE pair_id = ?",
            [(pair_id,) for pair_id in pair_ids],
        )
        connection.executemany(
            "DELETE FROM two_view_geometries WHERE pair_id = ?",
            [(pair_id,) for pair_id in pair_ids],
        )
        connection.commit()
    finally:
        connection.close()


def _normalized_image_name(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")




def _database_matches_images(database_path: Path, expected_names: list[str]) -> bool:
    """Require the resumed COLMAP database to contain this exact image set."""

    if not database_path.is_file():
        return False
    try:
        actual = [
            _normalized_image_name(row[2]).casefold()
            for row in _database_images(database_path)
        ]
    except (sqlite3.Error, OSError):
        return False
    expected = [name.casefold() for name in expected_names]
    return len(actual) == len(expected) and set(actual) == set(expected)



def _prepare_colmap_image_paths(
    image_paths: list[str],
    image_names: list[str],
    image_dir: Path,
    manifest_path: Path,
) -> list[str]:
    """Build a run-private flat directory containing exactly one input image set."""

    expected_names = [_normalized_image_name(Path(value).name) for value in image_names]
    folded = [name.casefold() for name in expected_names]
    if len(set(folded)) != len(folded):
        raise RuntimeError("摄影测量输入中存在同名照片，请先使用不同文件名")
    if len(expected_names) != len(image_paths):
        raise RuntimeError("摄影测量的照片路径与照片名称数量不一致")
    entries: list[dict[str, str | int]] = []
    for source_value, name in zip(image_paths, expected_names):
        source = Path(source_value).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"COLMAP 输入照片不存在：{source}")
        stat = source.stat()
        entries.append(
            {
                "name": name,
                "source": str(source),
                "size": int(stat.st_size),
                "modified_ns": int(stat.st_mtime_ns),
            }
        )
    payload = {"version": 1, "images": entries}
    reusable = False
    if manifest_path.is_file() and image_dir.is_dir():
        try:
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual_files = {
                path.name.casefold()
                for path in image_dir.iterdir()
                if path.is_file()
            }
            reusable = (
                saved == payload
                and actual_files == {name.casefold() for name in expected_names}
                and all(
                    (image_dir / name).is_file()
                    and (image_dir / name).stat().st_size == int(entry["size"])
                    for name, entry in zip(expected_names, entries)
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            reusable = False
    if reusable:
        return expected_names

    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=False)
    try:
        for source_value, name in zip(image_paths, expected_names):
            source = Path(source_value).resolve()
            target = image_dir / name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    except Exception:
        shutil.rmtree(image_dir, ignore_errors=True)
        raise
    return expected_names


def _invalidate_colmap_outputs(root: Path) -> None:
    """Remove only derived files in one verified COLMAP run directory."""

    for name in (
        "sparse_initial",
        "sparse_triangulated",
        "sparse_mapped",
        "sparse_ba",
        "sparse_preview",
        "dense",
    ):
        path = root / name
        if path.is_dir():
            shutil.rmtree(path)
    for path in root.glob("dense_*"):
        if path.is_dir():
            shutil.rmtree(path)
    for name in (
        "database.db",
        "database.db-shm",
        "database.db-wal",
        "pointcloud_ba_mvs_calibrated.ply",
        "pointcloud_ai_photogrammetry.ply",
        "sparse_ba.ply",
        "sparse_quality.json",
        "matching_connectivity.json",
        "cross_sequence_pairs.txt",
        "mapping.json",
        "pointcloud_output.json",
    ):
        path = root / name
        if path.is_file():
            path.unlink()




def _ply_vertex_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("rb") as stream:
            header = stream.read(64 * 1024).split(b"end_header", 1)[0]
        for line in header.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[:2] == [b"element", b"vertex"]:
                return int(parts[2])
    except (OSError, ValueError):
        return 0
    return 0


_COLMAP_FUSED_PROPERTIES = (
    ("float", "x"),
    ("float", "y"),
    ("float", "z"),
    ("float", "nx"),
    ("float", "ny"),
    ("float", "nz"),
    ("uchar", "red"),
    ("uchar", "green"),
    ("uchar", "blue"),
)
_COLMAP_FUSED_RECORD_SIZE = 27


def _colmap_fused_ply_info(path: Path) -> tuple[int, int]:
    """Return vertex count and payload offset for a complete COLMAP PLY."""

    if not path.is_file():
        raise RuntimeError(f"融合分块不存在：{path}")
    with path.open("rb") as stream:
        prefix = stream.read(64 * 1024)
    marker = prefix.find(b"end_header")
    if marker < 0:
        raise RuntimeError(f"融合分块缺少PLY头：{path}")
    newline = prefix.find(b"\n", marker)
    if newline < 0:
        raise RuntimeError(f"融合分块PLY头不完整：{path}")
    payload_offset = newline + 1
    lines = prefix[:payload_offset].decode("ascii", errors="strict").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise RuntimeError(f"融合分块不是COLMAP二进制PLY：{path}")
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertices = False
    for line in lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3:
            vertex_count = int(parts[2])
            in_vertices = True
            continue
        if parts[:1] == ["element"]:
            in_vertices = False
            continue
        if in_vertices and parts[:1] == ["property"] and len(parts) == 3:
            properties.append((parts[1], parts[2]))
    if vertex_count <= 0 or tuple(properties) != _COLMAP_FUSED_PROPERTIES:
        raise RuntimeError(f"融合分块PLY顶点格式异常：{path}")
    expected_size = payload_offset + vertex_count * _COLMAP_FUSED_RECORD_SIZE
    if path.stat().st_size < expected_size:
        raise RuntimeError(f"融合分块PLY被截断：{path}")
    return vertex_count, payload_offset


def _merge_colmap_fused_plys(inputs: list[Path], output: Path) -> int:
    """Stream binary COLMAP PLY batches into one atomic point cloud."""

    if not inputs:
        raise RuntimeError("没有可合并的融合分块")
    layouts = [_colmap_fused_ply_info(path) for path in inputs]
    total_vertices = sum(count for count, _ in layouts)
    payload_bytes = total_vertices * _COLMAP_FUSED_RECORD_SIZE
    if shutil.disk_usage(output.parent).free < payload_bytes + 512 * 1024**2:
        raise RuntimeError(
            "磁盘空间不足以合并稠密点云；请至少再释放 "
            f"{(payload_bytes + 512 * 1024**2) / 1024**3:.1f} GB"
        )
    temporary = output.with_name(f"{output.name}.merging")
    temporary.unlink(missing_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {total_vertices}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    try:
        with temporary.open("wb") as target:
            target.write(header)
            for path, (count, offset) in zip(inputs, layouts, strict=True):
                remaining = count * _COLMAP_FUSED_RECORD_SIZE
                with path.open("rb") as source:
                    source.seek(offset)
                    while remaining:
                        block = source.read(min(16 * 1024**2, remaining))
                        if not block:
                            raise RuntimeError(f"读取融合分块时提前结束：{path}")
                        target.write(block)
                        remaining -= len(block)
        merged_count, _ = _colmap_fused_ply_info(temporary)
        if merged_count != total_vertices:
            raise RuntimeError("稠密点云分块合并后的点数校验失败")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total_vertices


def _remove_stereo_fusion_output(path: Path) -> int:
    """Remove one derived fusion PLY and its COLMAP visibility sidecar."""

    removed_bytes = 0
    for candidate in (path, Path(f"{path}.vis")):
        if candidate.is_file():
            removed_bytes += candidate.stat().st_size
            candidate.unlink()
    return removed_bytes


def _dense_map_valid(path: Path) -> bool:
    """Validate COLMAP's width&height&channels& float32 dense-map format."""

    if not path.is_file() or path.stat().st_size < 16:
        return False
    try:
        with path.open("rb") as stream:
            header = bytearray()
            while header.count(b"&") < 3 and len(header) < 100:
                value = stream.read(1)
                if not value:
                    return False
                header.extend(value)
        width, height, channels = (
            int(value) for value in bytes(header).rstrip(b"&").split(b"&")
        )
        if width <= 0 or height <= 0 or channels <= 0:
            return False
        expected_size = len(header) + width * height * channels * 4
        return path.stat().st_size >= expected_size
    except (OSError, TypeError, ValueError):
        return False


def _remove_invalid_dense_maps(dense: Path) -> list[str]:
    removed: list[str] = []
    stereo = dense / "stereo"
    for folder_name in ("depth_maps", "normal_maps"):
        folder = stereo / folder_name
        if not folder.is_dir():
            continue
        for path in folder.glob("*.bin"):
            if not _dense_map_valid(path):
                removed.append(str(path))
                path.unlink()
    if removed:
        consistency = stereo / "consistency_graphs"
        if consistency.is_dir():
            shutil.rmtree(consistency)
    return removed


def _configure_patch_match(
    config_path: Path,
    source_count: int,
    reference_images: list[str] | None = None,
) -> int:
    """Configure selected reference views and automatic source-view counts."""

    if source_count < 1:
        raise ValueError("PatchMatch 源照片数必须大于 0")
    if not config_path.is_file():
        raise RuntimeError(f"缺少 COLMAP PatchMatch 配置文件：{config_path}")
    lines = [
        line.strip()
        for line in config_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]
    if not lines or len(lines) % 2:
        raise RuntimeError(f"COLMAP PatchMatch 配置格式异常：{config_path}")
    available = {
        lines[index].casefold(): lines[index]
        for index in range(0, len(lines), 2)
    }
    if reference_images is not None:
        missing = [
            name for name in reference_images if name.casefold() not in available
        ]
        if missing:
            raise RuntimeError(
                "MVS参考照片不在去畸变工作区："
                + "、".join(missing[:5])
            )
        selected = [available[name.casefold()] for name in reference_images]
    else:
        selected = list(available.values())
    configured: list[str] = []
    for name in selected:
        configured.extend([name, f"__auto__, {int(source_count)}"])
    config_path.write_text(
        "\n".join(configured) + "\n",
        encoding="utf-8",
    )
    return len(configured) // 2


def _set_patch_match_source_count(config_path: Path, source_count: int) -> int:
    """Backward-compatible helper that keeps every image as a reference."""

    return _configure_patch_match(config_path, source_count)


def _patch_match_dependency_arguments(
    *,
    registered_images: int,
    reference_images: int,
    geometric_consistency: bool,
) -> list[str]:
    """Allow helper source views without requiring their own depth maps."""

    if geometric_consistency and reference_images < registered_images:
        return ["--PatchMatchStereo.allow_missing_files", "1"]
    return []


def _estimate_dense_workspace_bytes(
    image_paths: list[str],
    max_image_size: int,
    geometric_consistency: bool = True,
) -> int:
    """Estimate peak dense-MVS disk use without summing sequential stages.

    COLMAP stores float32 depth (4 B/px) and three-channel normals
    (12 B/px). Geometric consistency temporarily needs both photometric and
    geometric maps, i.e. 32 B/px. Photometric maps are removed before
    geometric fusion, so the fused PLY must not be added to that peak again.
    """

    total_pixels = 0
    for value in image_paths:
        with Image.open(value) as image:
            width, height = image.size
        scale = min(1.0, float(max_image_size) / max(width, height))
        total_pixels += max(1, round(width * scale)) * max(1, round(height * scale))
    source_bytes = sum(Path(value).stat().st_size for value in image_paths)
    bytes_per_pixel = 34 if geometric_consistency else 24
    return int(total_pixels * bytes_per_pixel + source_bytes)


def _mvs_cache_size_gb() -> int:
    """Reserve enough RAM for Qt/VTK while keeping COLMAP I/O efficient."""

    available_gb = psutil.virtual_memory().available / 1024**3
    return max(4, min(16, int(available_gb * 0.3)))


def _fusion_resources() -> tuple[int, int]:
    """Return a conservative cache/thread budget for dense-map fusion.

    StereoFusion defaults to loading every depth and normal map into memory.
    A few hundred 4K views can therefore require well over 100 GB even though
    PatchMatch itself fits in GPU memory.  Cached fusion keeps only a bounded
    working set in RAM.  Limiting threads also avoids simultaneous map reads
    causing a large transient allocation on Windows.
    """

    available_gb = max(1.0, psutil.virtual_memory().available / 1024**3)
    physical_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    cache_size_gb = max(2, min(8, int(available_gb * 0.15)))
    memory_limited_threads = max(2, int(available_gb / 2.0))
    num_threads = max(1, min(12, int(physical_cores), memory_limited_threads))
    return cache_size_gb, num_threads


def _write_fusion_config(path: Path, image_names: list[str]) -> None:
    if not image_names:
        raise RuntimeError("融合照片列表为空")
    temporary = path.with_name(f"{path.name}.writing")
    temporary.write_text("\n".join(image_names) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fusion_reference_names(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"缺少COLMAP融合配置：{path}")
    names = [
        line.strip().lstrip("\ufeff")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not names or len(set(names)) != len(names):
        raise RuntimeError(f"COLMAP融合配置为空或包含重复照片：{path}")
    return names


def _plan_fusion_batches(
    dense: Path,
    image_names: list[str],
    *,
    input_type: str,
    memory_capacity_bytes: int | None = None,
) -> tuple[list[list[str]], int]:
    """Plan full-resolution batches when all dense maps cannot fit in RAM."""

    if input_type not in {"geometric", "photometric"}:
        raise ValueError("融合输入类型必须是 geometric 或 photometric")
    total_map_bytes = 0
    for image_name in image_names:
        for folder in ("depth_maps", "normal_maps"):
            path = dense / "stereo" / folder / f"{image_name}.{input_type}.bin"
            if not _dense_map_valid(path):
                raise RuntimeError(f"融合所需深度/法向图缺失或损坏：{path}")
            total_map_bytes += path.stat().st_size
    memory_capacity = int(
        memory_capacity_bytes
        if memory_capacity_bytes is not None
        else psutil.virtual_memory().total
    )
    direct_limit = min(24 * 1024**3, int(max(1, memory_capacity) * 0.35))
    if total_map_bytes <= direct_limit:
        return [list(image_names)], total_map_bytes
    average_bytes = max(1, total_map_bytes // len(image_names))
    target_bytes = min(
        8 * 1024**3,
        max(3 * 1024**3, int(memory_capacity * 0.15)),
    )
    batch_size = max(8, min(48, target_bytes // average_bytes))
    batches = [
        image_names[offset : offset + batch_size]
        for offset in range(0, len(image_names), batch_size)
    ]
    return batches, total_map_bytes


def _stereo_fusion_arguments(
    dense: Path,
    output_path: Path,
    *,
    geometric_consistency: bool,
    min_num_pixels: int,
    check_num_images: int,
    use_cache: bool,
    cache_size_gb: int,
    num_threads: int,
) -> list[str]:
    """Build memory-safe COLMAP StereoFusion arguments."""

    return [
        "stereo_fusion",
        "--workspace_path",
        str(dense),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric" if geometric_consistency else "photometric",
        "--output_path",
        str(output_path),
        "--StereoFusion.min_num_pixels",
        str(int(min_num_pixels)),
        "--StereoFusion.check_num_images",
        str(max(1, int(check_num_images))),
        "--StereoFusion.use_cache",
        "1" if use_cache else "0",
        "--StereoFusion.cache_size",
        str(int(cache_size_gb)),
        "--StereoFusion.num_threads",
        str(int(num_threads)),
    ]


def _retryable_fusion_failure(error: BaseException) -> bool:
    """Recognise native Windows resource failures worth retrying safely."""

    message = str(error).casefold()
    return _cuda_out_of_memory(error) or any(
        marker in message
        for marker in (
            "3221225495",  # 0xC0000017: STATUS_NO_MEMORY
            "-1073741801",
            "0xc0000017",
            "3221226505",  # 0xC0000409: native fast-fail / BEX64
            "-1073740791",
            "0xc0000409",
            "bad allocation",
            "std::bad_alloc",
        )
    )


def _cuda_out_of_memory(error: BaseException) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cudaerrormemoryallocation",
            "cuda_error_out_of_memory",
            "cuda error 2",
            "failed to allocate",
        )
    )


def _remove_photometric_maps_after_geometric_patchmatch(
    dense: Path,
) -> tuple[int, int]:
    """Free maps that geometric StereoFusion will never read."""

    removed_count = 0
    freed_bytes = 0
    for folder_name in ("depth_maps", "normal_maps"):
        folder = dense / "stereo" / folder_name
        if not folder.is_dir():
            continue
        for path in folder.glob("*.photometric.bin"):
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            removed_count += 1
            freed_bytes += size
    return removed_count, freed_bytes


def _remove_dense_maps_for_images(dense: Path, image_names: list[str]) -> int:
    """Remove completed block maps after Core fusion to keep disk bounded."""

    removed_bytes = 0
    stereo = dense / "stereo"
    for folder_name in ("depth_maps", "normal_maps", "consistency_graphs"):
        folder = stereo / folder_name
        if not folder.is_dir():
            continue
        for image_name in image_names:
            for suffix in ("photometric.bin", "geometric.bin"):
                path = folder / f"{image_name}.{suffix}"
                if not path.is_file():
                    continue
                try:
                    removed_bytes += path.stat().st_size
                    path.unlink()
                except OSError:
                    pass
    return removed_bytes


def _spatial_dense_workspace_estimate(
    image_paths: list[str],
    max_image_size: int,
    geometric_consistency: bool,
    target_images: int,
) -> int:
    """Estimate peak disk for one active MVS block plus the merged cloud."""

    full = _estimate_dense_workspace_bytes(
        image_paths,
        max_image_size,
        geometric_consistency,
    )
    source_bytes = sum(Path(value).stat().st_size for value in image_paths)
    map_bytes = max(0, full - source_bytes)
    active_ratio = min(
        1.0,
        max(3, int(target_images * 1.5)) / max(len(image_paths), 1),
    )
    active_maps = int(map_bytes * active_ratio)
    # A fused COLMAP record is 27 bytes and is typically much smaller than all
    # float depth/normal maps.  Keep a conservative whole-scene allowance.
    fused_allowance = max(2 * 1024**3, int(map_bytes * 0.16))
    return int(source_bytes * 2 + active_maps + fused_allowance)


def _run_spatial_block_mvs(
    *,
    executable: str,
    dense: Path,
    sparse_preview: Path,
    log_path: Path,
    fused: Path,
    geometric_consistency: bool,
    patch_match_filter: bool,
    patch_match_source_images: int,
    patch_match_iterations: int,
    max_image_size: int,
    fusion_min_num_pixels: int,
    use_gpu: bool,
    target_images: int,
    halo_ratio: float,
    mvs_cache_size_gb: int,
    progress_callback: Callable[[float, str], None] | None,
    tracker: StageTracker,
) -> dict[str, object]:
    """Run PatchMatch and fusion one PCA Core/Halo block at a time."""

    block_root = dense / "spatial_blocks"
    block_root.mkdir(parents=True, exist_ok=True)
    plan_path = block_root / "block_plan.json"
    plan = plan_spatial_blocks(
        sparse_preview / "images.txt",
        sparse_preview / "points3D.txt",
        target_images=int(target_images),
        halo_ratio=float(halo_ratio),
    )
    save_spatial_plan(plan_path, plan)
    blocks = list(plan["blocks"])
    if not blocks:
        raise RuntimeError("空间分块计划为空")
    frame = dict(plan["coordinate_frame"])
    center = frame["center"]
    basis = frame["basis"]
    views = read_sparse_views(sparse_preview / "images.txt")
    registered_count = len(views)
    block_tracker = StageTracker(block_root / "pipeline_state.json")
    fusion_config = dense / "stereo" / "fusion.cfg"
    patch_config = dense / "stereo" / "patch-match.cfg"
    input_type = "geometric" if geometric_consistency else "photometric"
    fusion_cache_size_gb, fusion_num_threads = _fusion_resources()
    core_outputs: list[Path] = []
    effective_sources: list[int] = []

    def update(value: float, text: str) -> None:
        if progress_callback:
            progress_callback(value, text)

    for block_index, raw_block in enumerate(blocks, start=1):
        block = dict(raw_block)
        block_id = str(block["id"])
        image_names = [str(value) for value in block["image_names"]]
        reference_images = [
            str(value) for value in block.get("reference_images", image_names)
        ]
        directory = block_root / block_id
        directory.mkdir(parents=True, exist_ok=True)
        raw_fused = directory / "halo_fused.ply"
        core_fused = directory / "core_fused.ply"
        core_outputs.append(core_fused)
        try:
            reusable = _colmap_fused_ply_info(core_fused)[0] > 0
        except RuntimeError:
            reusable = False
            _remove_stereo_fusion_output(core_fused)
        if reusable and block_tracker.status(block_id) == "completed":
            _remove_dense_maps_for_images(dense, reference_images)
            details = dict(
                block_tracker.data.get("stages", {}).get(block_id, {}).get("details")
                or {}
            )
            effective_sources.append(int(details.get("source_images", patch_match_source_images)))
            update(
                0.64 + 0.28 * block_index / len(blocks),
                f"恢复空间块 {block_index}/{len(blocks)}：已完成",
            )
            continue

        block_tracker.set(
            block_id,
            "running",
            message=f"空间块 {block_index}/{len(blocks)}",
            details={
                "image_count": len(image_names),
                "reference_image_count": len(reference_images),
            },
        )
        source_attempts = list(
            dict.fromkeys(
                [
                    min(int(patch_match_source_images), max(1, len(image_names) - 1)),
                    min(12, max(1, len(image_names) - 1)),
                    min(8, max(1, len(image_names) - 1)),
                    min(6, max(1, len(image_names) - 1)),
                ]
            )
        )
        effective_source_count = source_attempts[0]
        try:
            _remove_invalid_dense_maps(dense)
            for attempt_index, source_count in enumerate(source_attempts):
                effective_source_count = source_count
                configured = write_block_patch_match_config(
                    patch_config,
                    reference_images,
                    views,
                    source_count=source_count,
                    source_image_names=image_names,
                )
                patch_args = [
                    "patch_match_stereo",
                    "--workspace_path",
                    str(dense),
                    "--workspace_format",
                    "COLMAP",
                    "--PatchMatchStereo.geom_consistency",
                    "1" if geometric_consistency else "0",
                    "--PatchMatchStereo.filter",
                    "1" if patch_match_filter else "0",
                    "--PatchMatchStereo.num_iterations",
                    str(int(patch_match_iterations)),
                    "--PatchMatchStereo.max_image_size",
                    str(int(max_image_size)),
                    "--PatchMatchStereo.cache_size",
                    str(int(mvs_cache_size_gb)),
                    *_patch_match_dependency_arguments(
                        registered_images=registered_count,
                        reference_images=configured,
                        geometric_consistency=bool(geometric_consistency),
                    ),
                ]
                if use_gpu:
                    patch_args += ["--PatchMatchStereo.gpu_index", "0"]

                def patch_progress(line: str) -> None:
                    match = re.search(r"Processing view\s+(\d+)\s*/\s*(\d+)", line)
                    if not match:
                        return
                    current, total = (int(value) for value in match.groups())
                    local_fraction = current / max(total, 1)
                    overall = (block_index - 1 + local_fraction) / len(blocks)
                    update(
                        0.64 + 0.22 * overall,
                        f"空间块 {block_index}/{len(blocks)} PatchMatch：{current}/{total}",
                    )

                try:
                    _run(executable, patch_args, log_path, callback=patch_progress)
                    break
                except Exception as exc:
                    _remove_invalid_dense_maps(dense)
                    has_retry = attempt_index + 1 < len(source_attempts)
                    if not (use_gpu and has_retry and _cuda_out_of_memory(exc)):
                        raise
                    next_count = source_attempts[attempt_index + 1]
                    update(
                        0.64 + 0.22 * (block_index - 1) / len(blocks),
                        f"空间块显存不足，源照片数 {source_count}→{next_count} 后重试",
                    )

            expected_maps = [
                dense / "stereo" / "depth_maps" / f"{name}.{input_type}.bin"
                for name in reference_images
            ]
            if not all(_dense_map_valid(path) for path in expected_maps):
                missing = [path.name for path in expected_maps if not _dense_map_valid(path)]
                raise RuntimeError(
                    f"空间块 {block_id} 深度图不完整：" + "、".join(missing[:5])
                )
            if geometric_consistency:
                _remove_photometric_maps_after_geometric_patchmatch(dense)
            _write_fusion_config(fusion_config, reference_images)
            _remove_stereo_fusion_output(raw_fused)
            update(
                0.86 + 0.05 * (block_index - 1) / len(blocks),
                f"融合空间块 {block_index}/{len(blocks)}（{len(reference_images)}张）",
            )
            try:
                _run(
                    executable,
                    _stereo_fusion_arguments(
                        dense,
                        raw_fused,
                        geometric_consistency=bool(geometric_consistency),
                        min_num_pixels=int(fusion_min_num_pixels),
                        check_num_images=int(effective_source_count),
                        use_cache=False,
                        cache_size_gb=fusion_cache_size_gb,
                        num_threads=fusion_num_threads,
                    ),
                    log_path,
                )
            except RuntimeError as exc:
                _remove_stereo_fusion_output(raw_fused)
                if not _retryable_fusion_failure(exc):
                    raise
                _run(
                    executable,
                    _stereo_fusion_arguments(
                        dense,
                        raw_fused,
                        geometric_consistency=bool(geometric_consistency),
                        min_num_pixels=int(fusion_min_num_pixels),
                        check_num_images=int(effective_source_count),
                        use_cache=True,
                        cache_size_gb=min(4, fusion_cache_size_gb),
                        num_threads=min(4, fusion_num_threads),
                    ),
                    log_path,
                )
            raw_count = _colmap_fused_ply_info(raw_fused)[0]
            core_count = crop_fused_ply_to_core(
                raw_fused,
                core_fused,
                center=center,
                basis=basis,
                lower=block["core_lower"],
                upper=block["core_upper"],
            )
            block_tracker.set(
                block_id,
                "completed",
                message=f"空间块 {block_index}/{len(blocks)}完成",
                details={
                    "image_count": len(image_names),
                    "reference_image_count": len(reference_images),
                    "source_images": int(effective_source_count),
                    "halo_point_count": int(raw_count),
                    "core_point_count": int(core_count),
                    "output": str(core_fused),
                },
            )
            effective_sources.append(effective_source_count)
        except Exception as exc:
            block_tracker.set(block_id, "failed", message=str(exc))
            raise
        finally:
            _remove_stereo_fusion_output(raw_fused)
            if core_fused.is_file():
                _remove_dense_maps_for_images(dense, reference_images)

    update(0.925, f"合并 {len(core_outputs)} 个空间Core点云")
    partial = fused.with_suffix(".spatial-merging.ply")
    _merge_colmap_fused_plys(core_outputs, partial)
    partial.replace(fused)
    _write_fusion_config(fusion_config, [view.name for view in views])
    point_count = _ply_vertex_count(fused)
    if point_count <= 0:
        raise RuntimeError("空间分块合并点云为空")
    report = {
        **plan,
        "plan": str(plan_path),
        "output": str(fused),
        "point_count": int(point_count),
        "effective_source_images": (
            int(min(effective_sources)) if effective_sources else int(patch_match_source_images)
        ),
        "workspace_strategy": "shared_undistorted_workspace_core_halo_blocks",
        "depth_maps_retained": False,
    }
    save_spatial_plan(block_root / "spatial_mvs_report.json", report)
    tracker.set("patch_match", "completed", message="空间分块PatchMatch完成", details=report)
    tracker.set("stereo_fusion", "completed", message="空间Core点云合并完成", details=report)
    return report


def _model_exists(path: Path) -> bool:
    return all(
        any((path / f"{name}.{suffix}").is_file() for suffix in ("bin", "txt"))
        for name in ("cameras", "images", "points3D")
    )


def _registered_image_count(path: Path) -> int:
    binary = path / "images.bin"
    if binary.is_file():
        try:
            with binary.open("rb") as stream:
                payload = stream.read(8)
            return int(struct.unpack("<Q", payload)[0]) if len(payload) == 8 else 0
        except (OSError, struct.error):
            return 0
    text = path / "images.txt"
    if text.is_file():
        try:
            count = 0
            for line in text.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split(maxsplit=9)
                if len(parts) < 10 or line.lstrip().startswith("#"):
                    continue
                try:
                    int(parts[0])
                    [float(value) for value in parts[1:8]]
                except ValueError:
                    continue
                count += 1
            return count
        except OSError:
            return 0
    return 0


def _best_sparse_model(root: Path) -> Path | None:
    candidates: set[Path] = set()
    if _model_exists(root):
        candidates.add(root)
    for pattern in ("images.bin", "images.txt"):
        candidates.update(
            item.parent for item in root.rglob(pattern) if _model_exists(item.parent)
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (_registered_image_count(path), str(path).casefold()),
    )


def _database_count(path: Path, table: str) -> int:
    if not path.is_file():
        return 0
    try:
        connection = sqlite3.connect(path)
        try:
            return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return 0


def _sparse_text_quality(
    text_model: Path,
    expected_names: list[str],
) -> dict[str, object]:
    image_path = text_model / "images.txt"
    points_path = text_model / "points3D.txt"
    registered_names: list[str] = []
    image_lines = [
        line.strip()
        for line in image_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_lookup = {name.casefold(): name for name in expected_names}
    for line in image_lines:
        parts = line.split(maxsplit=9)
        if len(parts) == 10 and parts[9].casefold() in expected_lookup:
            registered_names.append(expected_lookup[parts[9].casefold()])

    point_count = 0
    errors: list[float] = []
    track_lengths: list[int] = []
    for line in points_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        parts = clean.split()
        if len(parts) >= 8:
            point_count += 1
            try:
                errors.append(float(parts[7]))
            except ValueError:
                pass
            track_lengths.append(max(0, (len(parts) - 8) // 2))
    registered_set = {name.casefold() for name in registered_names}
    unregistered = [
        name for name in expected_names if name.casefold() not in registered_set
    ]
    mean_error = sum(errors) / len(errors) if errors else None
    sorted_errors = sorted(errors)
    sorted_tracks = sorted(track_lengths)

    def percentile(values: list[float] | list[int], fraction: float) -> float | None:
        if not values:
            return None
        position = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
        return float(values[position])

    return {
        "registered_images": len(registered_names),
        "registration_ratio": len(registered_names) / max(len(expected_names), 1),
        "unregistered_images": unregistered,
        "sparse_point_count": point_count,
        "mean_reprojection_error_px": mean_error,
        "median_reprojection_error_px": percentile(sorted_errors, 0.5),
        "p95_reprojection_error_px": percentile(sorted_errors, 0.95),
        "mean_track_length": (
            float(sum(track_lengths) / len(track_lengths)) if track_lengths else None
        ),
        "median_track_length": percentile(sorted_tracks, 0.5),
    }


def _evaluate_sparse_quality_gate(
    quality: dict[str, object],
) -> dict[str, object]:
    """Classify sparse geometry before expensive dense reconstruction."""

    registered = int(quality.get("registered_images", 0))
    registration_ratio = float(quality.get("registration_ratio", 0.0))
    sparse_points = int(quality.get("sparse_point_count", 0))
    error_value = quality.get("mean_reprojection_error_px")
    reprojection_error = (
        float(error_value) if error_value is not None else None
    )
    checks = [
        {
            "name": "注册照片比例",
            "passed": registration_ratio >= 0.95,
            "value": registration_ratio,
            "target": "≥ 95%",
        },
        {
            "name": "平均重投影误差",
            "passed": (
                reprojection_error is not None
                and reprojection_error <= 2.0
            ),
            "value": reprojection_error,
            "target": "≤ 2.0 px",
        },
        {
            "name": "稀疏点密度",
            "passed": sparse_points >= max(100, registered * 50),
            "value": sparse_points,
            "target": f"≥ {max(100, registered * 50)}",
        },
    ]
    blocking = (
        registered < 3
        or registration_ratio < 0.5
        or sparse_points < max(30, registered * 10)
        or (
            reprojection_error is not None
            and reprojection_error > 5.0
        )
    )
    status = (
        "blocked"
        if blocking
        else "passed"
        if all(bool(item["passed"]) for item in checks)
        else "review"
    )
    return {
        "status": status,
        "checks": checks,
        "manual_confirmation_required": status != "passed",
    }


def run_colmap_ba_mvs(
    session: ProjectSession,
    *,
    colmap_path: str | None,
    output_root: str | Path,
    source_image_paths: list[str] | None = None,
    feature_type: str = "aliked",
    matcher: str = "auto",
    mapper: str = "global",
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    feature_max_image_size: int = 4096,
    max_image_size: int = 4096,
    max_num_features: int = 4096,
    sequential_overlap: int = 20,
    geometric_consistency: bool = True,
    patch_match_filter: bool = True,
    patch_match_source_images: int = 12,
    patch_match_iterations: int = 4,
    mvs_reference_strategy: str = "all",
    mvs_reference_ratio: float = 1.0,
    spatial_blocking: bool = True,
    spatial_block_threshold: int = 180,
    spatial_block_target_images: int = 120,
    spatial_block_halo_ratio: float = 0.20,
    min_num_inliers: int = 20,
    ransac_max_error: float = 4.0,
    fusion_min_num_pixels: int = 2,
    generate_quality_report: bool = True,
    target_stage: str = "dense",
    use_gpu: bool = True,
    progress_callback: Callable[[float, str], None] | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """Run learned matching and stop after sparse BA or continue through dense MVS."""

    executable = find_colmap(colmap_path)
    if executable is None:
        raise RuntimeError("未找到 COLMAP。AI摄影测量需要带 CUDA/ONNX 的 COLMAP 4.x。")
    image_paths = [
        str(Path(value).resolve())
        for value in (source_image_paths or session.image_paths)
    ]
    if len(image_paths) < 3:
        raise RuntimeError("AI摄影测量至少需要三张有连续重叠的照片")
    if feature_type not in {"aliked", "sift_lightglue"}:
        raise ValueError("特征方案必须是 aliked 或 sift_lightglue")
    if matcher not in {"auto", "exhaustive", "sequential"}:
        raise ValueError("匹配模式必须是 auto、exhaustive 或 sequential")
    if mapper not in {"global", "incremental"}:
        raise ValueError("SfM求解器必须是 global 或 incremental")
    if target_stage not in {"sparse", "dense"}:
        raise ValueError("目标阶段必须是 sparse 或 dense")
    if min_num_inliers < 8:
        raise ValueError("两视图几何最少内点不能少于 8")
    if ransac_max_error <= 0:
        raise ValueError("RANSAC 最大误差必须大于 0")
    if fusion_min_num_pixels < 1:
        raise ValueError("融合最少一致视图数必须大于 0")
    if not 1 <= patch_match_source_images <= 100:
        raise ValueError("PatchMatch 源照片数必须在 1～100 之间")
    if not 1 <= patch_match_iterations <= 20:
        raise ValueError("PatchMatch 迭代次数必须在 1～20 之间")
    if mvs_reference_strategy not in {"covisibility", "all"}:
        raise ValueError("MVS参考帧策略必须是 covisibility 或 all")
    if not 0.1 <= mvs_reference_ratio <= 1.0:
        raise ValueError("MVS参考帧比例必须在 0.1～1.0 之间")
    if spatial_block_threshold < 20:
        raise ValueError("空间分块启用阈值不能少于20张照片")
    if spatial_block_target_images < 8:
        raise ValueError("每个空间块的目标照片数不能少于8")
    if not 0.0 <= spatial_block_halo_ratio <= 1.0:
        raise ValueError("空间分块Halo比例必须在0～1之间")
    if camera_model not in {
        "SIMPLE_PINHOLE",
        "PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "OPENCV",
    }:
        raise ValueError("不支持的 COLMAP 相机模型")

    raw_names = [Path(value).name for value in image_paths]
    image_names = (
        [f"{index:06d}_{Path(value).name}" for index, value in enumerate(image_paths)]
        if len({name.casefold() for name in raw_names}) != len(raw_names)
        else raw_names
    )
    resolved_matcher = (
        "exhaustive"
        if matcher == "auto" and len(image_paths) <= 120
        else "sequential"
        if matcher == "auto"
        else matcher
    )
    sparse_cache_options = {
        "version": 8,
        "feature_type": feature_type,
        "matcher": resolved_matcher,
        "mapper": mapper,
        "camera_model": camera_model,
        "single_camera": bool(single_camera),
        "feature_max_image_size": int(feature_max_image_size),
        "max_num_features": int(max_num_features),
        "sequential_overlap": int(sequential_overlap),
        "min_num_inliers": int(min_num_inliers),
        "ransac_max_error": float(ransac_max_error),
        "use_gpu": bool(use_gpu),
    }
    dense_cache_options = {
        "dense_version": 4,
        "max_image_size": int(max_image_size),
        "geometric_consistency": bool(geometric_consistency),
        "patch_match_filter": bool(patch_match_filter),
        "patch_match_source_images": int(patch_match_source_images),
        "patch_match_iterations": int(patch_match_iterations),
        "mvs_reference_strategy": mvs_reference_strategy,
        "mvs_reference_ratio": float(mvs_reference_ratio),
        "spatial_blocking": bool(spatial_blocking),
        "spatial_block_threshold": int(spatial_block_threshold),
        "spatial_block_target_images": int(spatial_block_target_images),
        "spatial_block_halo_ratio": float(spatial_block_halo_ratio),
        "fusion_min_num_pixels": int(fusion_min_num_pixels),
        "use_gpu": bool(use_gpu),
    }
    pipeline_options = {
        **sparse_cache_options,
        **dense_cache_options,
    }
    sparse_fingerprint = hashlib.sha256(
        json.dumps(
            sparse_cache_options,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    dense_fingerprint = hashlib.sha256(
        json.dumps(
            dense_cache_options,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    images_fingerprint = source_fingerprint(image_paths)[:12]
    if resume:
        root = (
            Path(output_root).resolve()
            / f"photogrammetry_{session.project_id}_{images_fingerprint}_{sparse_fingerprint}"
        )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (
            Path(output_root).resolve()
            / f"photogrammetry_{session.project_id}_{stamp}_{sparse_fingerprint}"
        )
    root.mkdir(parents=True, exist_ok=resume)
    image_dir = root / "images"
    image_manifest = root / "input_images.json"
    database = root / "database.db"
    mapped_root = root / "sparse_mapped"
    mapping_path = root / "mapping.json"
    connectivity_path = root / "matching_connectivity.json"
    bridge_pairs_path = root / "cross_sequence_pairs.txt"
    adjusted = root / "sparse_ba"
    dense = root / f"dense_{dense_fingerprint}"
    log_path = root / "photogrammetry.log"
    runtime_path = root / "ai_runtime.json"
    sparse_stages = StageTracker(root / "pipeline_state.json")
    dense_stages = StageTracker(dense / "pipeline_state.json")
    stages = sparse_stages
    try:
        ai_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ai_runtime = {
            "feature_device": "cuda" if use_gpu else "cpu",
            "matching_device": "cuda" if use_gpu else "cpu",
            "fallbacks": [],
        }

    def save_ai_runtime() -> None:
        runtime_path.write_text(
            json.dumps(ai_runtime, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    spatial_candidate = bool(
        spatial_blocking and len(image_paths) >= int(spatial_block_threshold)
    )
    estimated_workspace_bytes = (
        _spatial_dense_workspace_estimate(
            image_paths,
            int(max_image_size),
            bool(geometric_consistency),
            int(spatial_block_target_images),
        )
        if spatial_candidate
        else _estimate_dense_workspace_bytes(
            image_paths,
            int(max_image_size),
            bool(geometric_consistency),
        )
    )
    existing_dense_bytes = (
        sum(path.stat().st_size for path in dense.rglob("*") if path.is_file())
        if dense.is_dir()
        else 0
    )
    disk_free_bytes = shutil.disk_usage(root).free
    remaining_workspace_bytes = max(
        0,
        estimated_workspace_bytes - existing_dense_bytes,
    )
    fusion_input_type = "geometric" if geometric_consistency else "photometric"
    existing_depth_maps = (
        list((dense / "stereo" / "depth_maps").glob(f"*.{fusion_input_type}.bin"))
        if (dense / "stereo" / "depth_maps").is_dir()
        else []
    )
    existing_normal_maps = (
        list((dense / "stereo" / "normal_maps").glob(f"*.{fusion_input_type}.bin"))
        if (dense / "stereo" / "normal_maps").is_dir()
        else []
    )
    dense_maps_ready = (
        dense_stages.status("patch_match") == "completed"
        and len(existing_depth_maps) >= 3
        and len(existing_depth_maps) == len(existing_normal_maps)
        and all(_dense_map_valid(path) for path in existing_depth_maps)
        and all(_dense_map_valid(path) for path in existing_normal_maps)
    )
    if dense_maps_ready:
        # PatchMatch is already complete, so restarting only needs room for
        # resumable fusion batches plus the atomically merged PLY.  Estimating
        # the whole MVS workspace here would incorrectly demand another copy
        # of the existing 100+ GB depth/normal maps.
        depth_bytes = sum(path.stat().st_size for path in existing_depth_maps)
        estimated_fused_bytes = max(512 * 1024**2, int(depth_bytes * 0.75))
        existing_batch_bytes = sum(
            path.stat().st_size
            for batch_root in dense.glob("fusion_batches_*")
            if batch_root.is_dir()
            for path in batch_root.glob("batch_*.ply")
            if path.is_file()
        )
        remaining_workspace_bytes = (
            max(0, estimated_fused_bytes - existing_batch_bytes)
            + estimated_fused_bytes
        )
    required_free_bytes = remaining_workspace_bytes + 512 * 1024**2
    if (
        target_stage == "dense"
        and dense_stages.status("stereo_fusion") != "completed"
        and disk_free_bytes < required_free_bytes
    ):
        raise RuntimeError(
            "磁盘空间不足，无法安全启动原图MVS："
            f"当前可用 {disk_free_bytes / 1024**3:.2f} GB，"
            f"预计至少还需 {required_free_bytes / 1024**3:.2f} GB。"
            "软件已按断点和节省磁盘模式估算；请清理磁盘、移动工程，"
            "或降低“MVS最大图像尺寸”。"
        )

    expected_names = _prepare_colmap_image_paths(
        image_paths,
        image_names,
        image_dir,
        image_manifest,
    )
    (root / "sparse_pipeline_config.json").write_text(
        json.dumps(sparse_cache_options, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / f"dense_pipeline_config_{dense_fingerprint}.json").write_text(
        json.dumps(dense_cache_options, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if database.is_file() and not _database_matches_images(database, expected_names):
        _logger.warning(
            "Discarding stale AI photogrammetry database in %s because its image "
            "list does not match the current project selection",
            root,
        )
        _invalidate_colmap_outputs(root)
        for stage_name in (
            "ai_feature_extraction",
            "ai_feature_matching",
            "cross_sequence_matching",
            "sparse_mapping",
            "bundle_adjustment",
            "sparse_output",
            "image_undistortion",
            "patch_match",
            "stereo_fusion",
            "pointcloud_output",
        ):
            stages.set(
                stage_name,
                "pending",
                message="输入照片已变化，自动废弃旧摄影测量断点",
            )
    for directory in (mapped_root, adjusted, dense):
        directory.mkdir(parents=True, exist_ok=True)

    def update(value: float, text: str) -> None:
        if progress_callback:
            progress_callback(value, text)

    def run_stage(
        name: str,
        value: float,
        text: str,
        valid: Callable[[], bool],
        operation: Callable[[], None],
    ) -> None:
        if resume and stages.status(name) == "completed" and valid():
            update(value, f"断点恢复：跳过已完成的{text}")
            return
        update(value, text)
        stages.set(name, "running", message=text)
        try:
            operation()
            if not valid():
                raise RuntimeError(f"{text}未生成完整输出")
        except Exception as exc:
            stages.set(name, "failed", message=str(exc))
            raise
        stages.set(name, "completed", message=text)

    if feature_type == "aliked":
        extractor_type = "ALIKED_N16ROT"
        matcher_type = "ALIKED_LIGHTGLUE"
        extractor_model = _verified_colmap_ai_model("aliked-n16rot.onnx")
        matcher_model = _verified_colmap_ai_model("aliked-lightglue.onnx")
        feature_limit_args = [
            "--AlikedExtraction.max_num_features",
            str(int(max_num_features)),
            "--AlikedExtraction.n16rot_model_path",
            str(extractor_model),
        ]
        matcher_model_args = [
            "--AlikedMatching.lightglue_model_path",
            str(matcher_model),
        ]
        feature_label = "ALIKED"
    else:
        extractor_type = "SIFT"
        matcher_type = "SIFT_LIGHTGLUE"
        matcher_model = _verified_colmap_ai_model("sift-lightglue.onnx")
        feature_limit_args = [
            "--SiftExtraction.max_num_features",
            str(int(max_num_features)),
        ]
        matcher_model_args = [
            "--SiftMatching.lightglue_model_path",
            str(matcher_model),
        ]
        feature_label = "SIFT"
    def feature_arguments(gpu: bool) -> list[str]:
        return [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera",
            "1" if single_camera else "0",
            "--FeatureExtraction.type",
            extractor_type,
            "--FeatureExtraction.use_gpu",
            "1" if gpu else "0",
            "--FeatureExtraction.max_image_size",
            str(int(feature_max_image_size)),
            *feature_limit_args,
        ]

    def run_ai_features() -> None:
        try:
            _run(executable, feature_arguments(use_gpu), log_path)
            ai_runtime["feature_device"] = "cuda" if use_gpu else "cpu"
        except RuntimeError as exc:
            if not use_gpu:
                raise
            _logger.warning(
                "AI feature CUDA provider failed; retrying on ONNX CPU provider: %s",
                exc,
            )
            ai_runtime["feature_device"] = "cpu_fallback"
            ai_runtime.setdefault("fallbacks", []).append(
                "ALIKED/SIFT特征的ONNX CUDA Provider不可用，已自动切换CPU"
            )
            save_ai_runtime()
            _run(executable, feature_arguments(False), log_path)
        save_ai_runtime()

    def feature_cache_valid() -> bool:
        return (
            _database_matches_images(database, expected_names)
            and _database_count(database, "keypoints") == len(expected_names)
            and _database_count(database, "descriptors") == len(expected_names)
        )

    run_stage(
        "ai_feature_extraction",
        0.03,
        f"原图 {feature_label} AI特征提取",
        feature_cache_valid,
        run_ai_features,
    )

    matcher_command = (
        "sequential_matcher"
        if resolved_matcher == "sequential"
        else "exhaustive_matcher"
    )
    def matching_arguments(gpu: bool) -> list[str]:
        arguments = [
            matcher_command,
            "--database_path",
            str(database),
            "--FeatureMatching.type",
            matcher_type,
            "--FeatureMatching.use_gpu",
            "1" if gpu else "0",
            "--TwoViewGeometry.min_num_inliers",
            str(int(min_num_inliers)),
            "--TwoViewGeometry.max_error",
            str(float(ransac_max_error)),
            *matcher_model_args,
        ]
        if resolved_matcher == "sequential":
            arguments += [
                "--SequentialMatching.overlap",
                str(max(2, int(sequential_overlap))),
                "--SequentialMatching.quadratic_overlap",
                "1",
            ]
        return arguments

    def run_ai_matching() -> None:
        try:
            _run(executable, matching_arguments(use_gpu), log_path)
            ai_runtime["matching_device"] = "cuda" if use_gpu else "cpu"
        except RuntimeError as exc:
            if not use_gpu:
                raise
            _logger.warning(
                "LightGlue CUDA provider failed; retrying on ONNX CPU provider: %s",
                exc,
            )
            ai_runtime["matching_device"] = "cpu_fallback"
            ai_runtime.setdefault("fallbacks", []).append(
                "LightGlue的ONNX CUDA Provider不可用，已自动切换CPU"
            )
            save_ai_runtime()
            _run(executable, matching_arguments(False), log_path)
        save_ai_runtime()
    run_stage(
        "ai_feature_matching",
        0.2,
        f"{feature_label}‑LightGlue AI特征匹配",
        lambda: _database_count(database, "two_view_geometries") > 0,
        run_ai_matching,
    )

    def connectivity_report_valid() -> bool:
        if not connectivity_path.is_file():
            return False
        try:
            report = json.loads(connectivity_path.read_text(encoding="utf-8"))
            return (
                int(report.get("algorithm_version", 0)) == 1
                and int(report.get("image_count", -1)) == len(expected_names)
                and int(report.get("verified_pair_count", -1))
                == _database_verified_pair_count(
                    database,
                    min_num_inliers=min_num_inliers,
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def invalidate_geometry_after_bridge(message: str) -> None:
        for stage_name in (
            "sparse_mapping",
            "bundle_adjustment",
            "sparse_output",
        ):
            sparse_stages.set(stage_name, "pending", message=message)
        for stage_name in (
            "image_undistortion",
            "spatial_block_mvs",
            "patch_match",
            "stereo_fusion",
            "pointcloud_output",
        ):
            dense_stages.set(stage_name, "pending", message=message)

    def run_cross_sequence_matching() -> None:
        before_components = _database_verified_match_components(
            database,
            min_num_inliers=min_num_inliers,
        )
        verified_before = _database_verified_pair_count(
            database,
            min_num_inliers=min_num_inliers,
        )
        pairs, candidate_report = _bridge_candidate_pairs(
            database,
            min_num_inliers=min_num_inliers,
        )
        if pairs:
            bridge_pairs_path.write_text(
                "".join(f"{first} {second}\n" for first, second in pairs),
                encoding="utf-8",
            )
            # Sequential/LightGlue may have cached a zero-match row at a
            # sequence boundary. Remove only these derived candidate rows so
            # the wide-baseline fallback can genuinely retry them.
            _clear_database_image_pairs(database, pairs)
            if feature_type == "aliked":
                bridge_matcher_type = "ALIKED_BRUTEFORCE"
                bridge_model = _verified_colmap_ai_model(
                    "bruteforce-matcher.onnx"
                )
                bridge_matcher_arguments = [
                    "--AlikedMatching.bruteforce_model_path",
                    str(bridge_model),
                    "--AlikedMatching.brute_force_min_cossim",
                    "0.50",
                    "--AlikedMatching.brute_force_max_ratio",
                    "0.98",
                    "--AlikedMatching.brute_force_cross_check",
                    "1",
                ]
            else:
                bridge_matcher_type = "SIFT_BRUTEFORCE"
                bridge_matcher_arguments = [
                    "--SiftMatching.max_ratio",
                    "0.90",
                    "--SiftMatching.max_distance",
                    "1.0",
                    "--SiftMatching.cross_check",
                    "1",
                ]

            def bridge_arguments(gpu: bool) -> list[str]:
                return [
                    "matches_importer",
                    "--database_path",
                    str(database),
                    "--match_list_path",
                    str(bridge_pairs_path),
                    "--match_type",
                    "pairs",
                    "--FeatureMatching.type",
                    bridge_matcher_type,
                    "--FeatureMatching.use_gpu",
                    "1" if gpu else "0",
                    "--TwoViewGeometry.min_num_inliers",
                    str(int(min_num_inliers)),
                    "--TwoViewGeometry.max_error",
                    str(float(ransac_max_error)),
                    *bridge_matcher_arguments,
                ]

            try:
                _run(executable, bridge_arguments(use_gpu), log_path)
                ai_runtime["bridge_matching_device"] = (
                    "cuda" if use_gpu else "cpu"
                )
            except RuntimeError as exc:
                if not use_gpu:
                    raise
                _logger.warning(
                    "Cross-sequence matching failed on CUDA; retrying on CPU: %s",
                    exc,
                )
                ai_runtime["bridge_matching_device"] = "cpu_fallback"
                ai_runtime.setdefault("fallbacks", []).append(
                    "跨航线桥接匹配的CUDA Provider不可用，已自动切换CPU"
                )
                save_ai_runtime()
                _run(executable, bridge_arguments(False), log_path)
            save_ai_runtime()
        elif bridge_pairs_path.is_file():
            bridge_pairs_path.unlink()

        after_components = _database_verified_match_components(
            database,
            min_num_inliers=min_num_inliers,
        )
        verified_after = _database_verified_pair_count(
            database,
            min_num_inliers=min_num_inliers,
        )
        report: dict[str, object] = {
            "algorithm_version": 1,
            "image_count": len(expected_names),
            "components_before": [
                len(component) for component in before_components
            ],
            "components_after": [len(component) for component in after_components],
            "verified_pairs_before": int(verified_before),
            "verified_pair_count": int(verified_after),
            "new_verified_pairs": max(0, int(verified_after - verified_before)),
            "connected": len(after_components) <= 1,
            "pair_list": str(bridge_pairs_path) if pairs else "",
            **candidate_report,
        }
        connectivity_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        mapped_registration = 0
        if mapping_path.is_file():
            try:
                mapped_registration = int(
                    json.loads(mapping_path.read_text(encoding="utf-8")).get(
                        "registered_images", 0
                    )
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                mapped_registration = 0
        largest_after = len(after_components[0]) if after_components else 0
        if verified_after > verified_before or (
            mapped_registration and mapped_registration < largest_after
        ):
            invalidate_geometry_after_bridge(
                "跨航线照片组已连接，自动废弃旧空三及其下游稠密成果"
            )

    run_stage(
        "cross_sequence_matching",
        0.28,
        "跨航线检索、GPS邻域匹配与照片组自动桥接",
        connectivity_report_valid,
        run_cross_sequence_matching,
    )

    def selected_mapped_model() -> Path | None:
        if not mapping_path.is_file():
            return None
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            candidate = (root / str(payload["relative_path"])).resolve()
            if root.resolve() not in (candidate, *candidate.parents):
                return None
            return candidate if _model_exists(candidate) else None
        except (KeyError, OSError, json.JSONDecodeError):
            return None

    def run_mapper() -> None:
        if mapped_root.is_dir():
            shutil.rmtree(mapped_root)
        mapped_root.mkdir(parents=True, exist_ok=False)
        if mapper == "global":
            arguments = [
                "global_mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(mapped_root),
                "--GlobalMapper.min_num_matches",
                "15",
                "--GlobalMapper.gp_use_gpu",
                "1" if use_gpu else "0",
                "--GlobalMapper.ba_ceres_use_gpu",
                "1" if use_gpu else "0",
            ]
        else:
            arguments = [
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(mapped_root),
                "--Mapper.min_num_matches",
                "15",
                "--Mapper.ba_use_gpu",
                "1" if use_gpu else "0",
            ]
        _run(executable, arguments, log_path)
        chosen = _best_sparse_model(mapped_root)
        if chosen is None:
            raise RuntimeError(
                "SfM没有生成可用稀疏模型；请检查相邻照片重叠、模糊、反光和拍摄断层"
            )
        registered = _registered_image_count(chosen)
        if registered < 3:
            raise RuntimeError(f"SfM仅注册 {registered} 张照片，无法继续稠密重建")
        mapping_path.write_text(
            json.dumps(
                {
                    "relative_path": str(chosen.relative_to(root)),
                    "registered_images": registered,
                    "input_images": len(expected_names),
                    "registration_ratio": registered / len(expected_names),
                    "mapper": mapper,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    run_stage(
        "sparse_mapping",
        0.38,
        "GLOMAP 全局SfM" if mapper == "global" else "COLMAP 增量SfM",
        lambda: selected_mapped_model() is not None,
        run_mapper,
    )
    mapped_model = selected_mapped_model()
    if mapped_model is None:
        raise RuntimeError("找不到已完成的稀疏模型")

    def run_bundle_adjustment() -> None:
        if adjusted.is_dir():
            shutil.rmtree(adjusted)
        adjusted.mkdir(parents=True, exist_ok=False)
        arguments = [
            "bundle_adjuster",
            "--input_path",
            str(mapped_model),
            "--output_path",
            str(adjusted),
            "--BundleAdjustment.refine_principal_point",
            "0",
        ]
        if use_gpu:
            arguments += ["--BundleAdjustmentCeres.use_gpu", "1"]
        _run(executable, arguments, log_path)

    run_stage(
        "bundle_adjustment",
        0.5,
        "传统几何全局 Bundle Adjustment",
        lambda: _model_exists(adjusted),
        run_bundle_adjustment,
    )

    sparse_preview = root / "sparse_preview"
    sparse_ply = root / "sparse_ba.ply"
    sparse_metadata_path = root / "sparse_quality.json"

    def write_sparse_output() -> None:
        if sparse_preview.is_dir():
            shutil.rmtree(sparse_preview)
        sparse_preview.mkdir(parents=True)
        if sparse_ply.is_file():
            sparse_ply.unlink()
        _run(
            executable,
            [
                "model_converter",
                "--input_path",
                str(adjusted),
                "--output_path",
                str(sparse_preview),
                "--output_type",
                "TXT",
            ],
            log_path,
        )
        _run(
            executable,
            [
                "model_converter",
                "--input_path",
                str(adjusted),
                "--output_path",
                str(sparse_ply),
                "--output_type",
                "PLY",
            ],
            log_path,
        )
        quality = _sparse_text_quality(sparse_preview, expected_names)
        quality["quality_report_generated"] = bool(generate_quality_report)
        sparse_metadata_path.write_text(
            json.dumps(quality, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def sparse_output_valid() -> bool:
        if (
            not sparse_ply.is_file()
            or _ply_vertex_count(sparse_ply) == 0
            or not (sparse_preview / "images.txt").is_file()
            or not (sparse_preview / "points3D.txt").is_file()
            or not sparse_metadata_path.is_file()
        ):
            return False
        try:
            payload = json.loads(sparse_metadata_path.read_text(encoding="utf-8"))
            return int(payload.get("registered_images", 0)) >= 3
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    run_stage(
        "sparse_output",
        0.54,
        "生成空三检查成果",
        sparse_output_valid,
        write_sparse_output,
    )

    sparse_quality = json.loads(sparse_metadata_path.read_text(encoding="utf-8"))
    try:
        connectivity_report = json.loads(
            connectivity_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        connectivity_report = {
            "components_before": [],
            "components_after": [],
            "connected": True,
        }
    quality_gate = _evaluate_sparse_quality_gate(sparse_quality)
    mvs_selection = select_mvs_references(
        sparse_preview / "images.txt",
        reference_ratio=float(mvs_reference_ratio),
        strategy=mvs_reference_strategy,
    )
    mvs_selection["source_images_per_reference"] = int(
        patch_match_source_images
    )
    mvs_selection_path = (
        root / f"mvs_reference_selection_{dense_fingerprint}.json"
    )
    mvs_selection_path.write_text(
        json.dumps(mvs_selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sparse_warnings: list[str] = []
    components_before = list(connectivity_report.get("components_before", []))
    components_after = list(connectivity_report.get("components_after", []))
    if len(components_before) > 1 and len(components_after) == 1:
        sparse_warnings.append(
            "已通过图像检索、GPS邻域和几何验证自动连接分离照片组"
        )
    elif len(components_after) > 1:
        sparse_warnings.append(
            "照片仍分成多个互不连接的组；缺少共同可见区域时无法可靠自动合并"
        )
    if float(sparse_quality["registration_ratio"]) < 0.8:
        sparse_warnings.append(
            f"仅注册 {sparse_quality['registered_images']}/{len(expected_names)} 张照片；"
            "建议检查拍摄断层、共享内参或更换 SfM 求解器"
        )
    if quality_gate["status"] == "review":
        sparse_warnings.append("空三质量闸门提示复核；请确认相机轨迹和稀疏覆盖后继续")
    elif quality_gate["status"] == "blocked":
        sparse_warnings.append("空三质量闸门未通过；不建议启动稠密重建")
    sparse_result: dict[str, object] = {
        "pipeline": "AI局部特征 + 传统摄影测量",
        "result_stage": "sparse",
        "feature": "ALIKED‑N16Rot" if feature_type == "aliked" else "SIFT",
        "matcher": f"{matcher_type} / {resolved_matcher}",
        "mapper": "GLOMAP全局SfM" if mapper == "global" else "COLMAP增量SfM",
        "folder": str(root),
        "sparse_model": str(adjusted),
        "sparse_pointcloud": str(sparse_ply),
        "sparse_images_txt": str(sparse_preview / "images.txt"),
        "database": str(database),
        "log": str(log_path),
        "image_count": len(expected_names),
        "registered_images": int(sparse_quality["registered_images"]),
        "registration_ratio": float(sparse_quality["registration_ratio"]),
        "unregistered_images": list(sparse_quality["unregistered_images"]),
        "sparse_point_count": int(sparse_quality["sparse_point_count"]),
        "mean_reprojection_error_px": sparse_quality["mean_reprojection_error_px"],
        "median_reprojection_error_px": sparse_quality.get(
            "median_reprojection_error_px"
        ),
        "p95_reprojection_error_px": sparse_quality.get(
            "p95_reprojection_error_px"
        ),
        "mean_track_length": sparse_quality.get("mean_track_length"),
        "median_track_length": sparse_quality.get("median_track_length"),
        "quality_gate": quality_gate,
        "matching_connectivity": connectivity_report,
        "matching_connectivity_report": str(connectivity_path),
        "mvs_reference_selection": mvs_selection,
        "mvs_reference_selection_report": str(mvs_selection_path),
        "matched_pairs": _database_count(database, "two_view_geometries"),
        "ai_runtime": ai_runtime,
        "pipeline_options": pipeline_options,
        "warnings": sparse_warnings,
    }
    if target_stage == "sparse":
        update(1.0, "空三计算完成，请检查后确认稠密重建")
        return sparse_result

    stages = dense_stages

    def run_undistortion() -> None:
        if dense.is_dir():
            shutil.rmtree(dense)
        dense.mkdir(parents=True, exist_ok=False)
        _run(
            executable,
            [
                "image_undistorter",
                "--image_path",
                str(image_dir),
                "--input_path",
                str(adjusted),
                "--output_path",
                str(dense),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(int(max_image_size)),
            ],
            log_path,
        )

    run_stage(
        "image_undistortion",
        0.58,
        "原图去畸变并建立高分辨率MVS工作区",
        lambda: _model_exists(dense / "sparse")
        and (dense / "images").is_dir()
        and any((dense / "images").iterdir()),
        run_undistortion,
    )

    spatial_enabled = bool(
        spatial_blocking
        and int(sparse_quality["registered_images"])
        >= int(spatial_block_threshold)
    )
    if spatial_enabled:
        fused = dense / "fused.ply"
        spatial_report_path = dense / "spatial_blocks" / "spatial_mvs_report.json"
        mvs_cache_size_gb = _mvs_cache_size_gb()
        spatial_report: dict[str, object] = {}

        def run_spatial_mvs() -> None:
            nonlocal spatial_report
            spatial_report = _run_spatial_block_mvs(
                executable=executable,
                dense=dense,
                sparse_preview=sparse_preview,
                log_path=log_path,
                fused=fused,
                geometric_consistency=bool(geometric_consistency),
                patch_match_filter=bool(patch_match_filter),
                patch_match_source_images=int(patch_match_source_images),
                patch_match_iterations=int(patch_match_iterations),
                max_image_size=int(max_image_size),
                fusion_min_num_pixels=int(fusion_min_num_pixels),
                use_gpu=bool(use_gpu),
                target_images=int(spatial_block_target_images),
                halo_ratio=float(spatial_block_halo_ratio),
                mvs_cache_size_gb=int(mvs_cache_size_gb),
                progress_callback=progress_callback,
                tracker=dense_stages,
            )

        run_stage(
            "spatial_block_mvs",
            0.64,
            "按空三共视关系执行Core/Halo空间分块MVS",
            lambda: _ply_vertex_count(fused) > 0 and spatial_report_path.is_file(),
            run_spatial_mvs,
        )
        if not spatial_report:
            spatial_report = json.loads(
                spatial_report_path.read_text(encoding="utf-8")
            )

        final_ply = dense / "pointcloud_ai_photogrammetry.ply"
        output_metadata_path = dense / "pointcloud_output.json"

        def write_spatial_pointcloud_output() -> None:
            point_count = _ply_vertex_count(fused)
            if point_count <= 0:
                raise RuntimeError("空间分块融合点云为空，无法写出最终成果")
            final_ply.unlink(missing_ok=True)
            try:
                os.link(fused, final_ply)
            except OSError as exc:
                _logger.warning(
                    "Could not hard-link spatial final PLY; using fused output: %s",
                    exc,
                )
            output_metadata_path.write_text(
                json.dumps(
                    {
                        "coordinate_source": "sfm_native",
                        "unit": "model_units",
                        "point_count": int(point_count),
                        "storage": (
                            "hardlink" if final_ply.is_file() else "spatial_fused_fallback"
                        ),
                        "dense_strategy": "core_halo_spatial_blocks",
                        "spatial_block_count": int(spatial_report.get("block_count", 0)),
                        "spatial_plan": str(spatial_report.get("plan", "")),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        run_stage(
            "pointcloud_output",
            0.95,
            "写出空间分块合并点云",
            lambda: (final_ply.is_file() or fused.is_file())
            and output_metadata_path.is_file(),
            write_spatial_pointcloud_output,
        )
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        output_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
        registered_images = int(mapping_payload["registered_images"])
        registration_ratio = float(mapping_payload["registration_ratio"])
        spatial_warnings: list[str] = []
        if spatial_report.get("unassigned_images"):
            spatial_warnings.append(
                "少量注册照片没有有效稀疏点落入任何空间块；已记录在分块报告中"
            )
        update(1.0, "空间分块AI摄影测量完成")
        pointcloud_path = final_ply if final_ply.is_file() else fused
        fusion_cache_size_gb, fusion_num_threads = _fusion_resources()
        return {
            **sparse_result,
            "result_stage": "dense",
            "pointcloud": str(pointcloud_path),
            "raw_fused": str(fused),
            "dense_workspace": str(dense),
            "dense_cache_fingerprint": dense_fingerprint,
            "pointcloud_metadata": str(output_metadata_path),
            "registered_images": registered_images,
            "registration_ratio": registration_ratio,
            "point_count": int(output_metadata.get("point_count", 0)),
            "spatial_mvs": spatial_report,
            "mvs_settings": {
                "max_image_size": int(max_image_size),
                "geometric_consistency": bool(geometric_consistency),
                "filter": bool(patch_match_filter),
                "source_images": int(
                    spatial_report.get(
                        "effective_source_images", patch_match_source_images
                    )
                ),
                "requested_source_images": int(patch_match_source_images),
                "iterations": int(patch_match_iterations),
                "cache_size_gb": int(mvs_cache_size_gb),
                "fusion_cache_size_gb": int(fusion_cache_size_gb),
                "fusion_num_threads": int(fusion_num_threads),
                "fusion_strategy": "core_halo_spatial_blocks",
                "configured_reference_images": registered_images,
                "reference_strategy": "all_registered_spatially_assigned",
                "requested_reference_ratio": 1.0,
                "sparse_point_coverage_ratio": float(
                    mvs_selection["sparse_point_coverage_ratio"]
                ),
                "spatial_block_count": int(spatial_report.get("block_count", 0)),
                "spatial_block_target_images": int(spatial_block_target_images),
                "spatial_block_halo_ratio": float(spatial_block_halo_ratio),
                "depth_maps_retained": False,
            },
            "unit": "模型单位",
            "estimated_workspace_gb": round(
                estimated_workspace_bytes / 1024**3,
                2,
            ),
            "disk_free_gb_at_start": round(disk_free_bytes / 1024**3, 2),
            "warnings": [*sparse_warnings, *spatial_warnings],
        }

    patch_match_config = dense / "stereo" / "patch-match.cfg"
    reference_images = [
        str(value)
        for value in mvs_selection.get("reference_images", [])
    ]
    configured_reference_images = _configure_patch_match(
        patch_match_config,
        patch_match_source_images,
        reference_images,
    )
    mvs_cache_size_gb = _mvs_cache_size_gb()
    effective_patch_match_source_images = int(patch_match_source_images)
    patch_args = [
        "patch_match_stereo",
        "--workspace_path",
        str(dense),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "1" if geometric_consistency else "0",
        "--PatchMatchStereo.filter",
        "1" if patch_match_filter else "0",
        "--PatchMatchStereo.num_iterations",
        str(int(patch_match_iterations)),
        "--PatchMatchStereo.max_image_size",
        str(int(max_image_size)),
        "--PatchMatchStereo.cache_size",
        str(mvs_cache_size_gb),
        *_patch_match_dependency_arguments(
            registered_images=int(
                mvs_selection["registered_image_count"]
            ),
            reference_images=configured_reference_images,
            geometric_consistency=bool(geometric_consistency),
        ),
    ]
    if use_gpu:
        patch_args += ["--PatchMatchStereo.gpu_index", "0"]
    depth_maps = dense / "stereo" / "depth_maps"

    def dense_maps_valid() -> bool:
        depth_type = "geometric" if geometric_consistency else "photometric"
        maps = (
            list(depth_maps.glob(f"*.{depth_type}.bin"))
            if depth_maps.is_dir()
            else []
        )
        return (
            len(maps) >= configured_reference_images
            and all(_dense_map_valid(path) for path in maps)
        )

    def run_patch_match() -> None:
        nonlocal effective_patch_match_source_images
        removed_before = _remove_invalid_dense_maps(dense)
        if removed_before:
            _logger.warning(
                "Removed %d corrupt COLMAP dense maps before resume",
                len(removed_before),
            )
        existing_photometric = (
            len(list(depth_maps.glob("*.photometric.bin")))
            if depth_maps.is_dir()
            else 0
        )
        existing_geometric = (
            len(list(depth_maps.glob("*.geometric.bin")))
            if depth_maps.is_dir()
            else 0
        )
        initial_phase = int(
            geometric_consistency
            and existing_photometric >= configured_reference_images
            and existing_geometric < configured_reference_images
        )
        progress_state = {"phase": initial_phase, "last_view": 0}

        def patch_progress(line: str) -> None:
            if "Writing geometric output" in line:
                progress_state["phase"] = 1
            elif "Writing photometric output" in line:
                progress_state["phase"] = 0
            match = re.search(
                r"Processing view\s+(\d+)\s*/\s*(\d+)",
                line,
            )
            if not match:
                return
            current, total = (int(value) for value in match.groups())
            if (
                geometric_consistency
                and current < int(progress_state["last_view"])
            ):
                progress_state["phase"] = 1
            progress_state["last_view"] = current
            phases = 2 if geometric_consistency else 1
            fraction = (
                int(progress_state["phase"]) + current / max(total, 1)
            ) / phases
            phase_label = (
                "几何一致性"
                if int(progress_state["phase"])
                else "光度深度"
            )
            update(
                0.68 + min(max(fraction, 0.0), 1.0) * 0.18,
                f"PatchMatch {phase_label}：{current}/{total} 张参考图",
            )

        source_attempts = list(
            dict.fromkeys(
                [
                    int(patch_match_source_images),
                    min(int(patch_match_source_images), 12),
                    min(int(patch_match_source_images), 8),
                    min(int(patch_match_source_images), 6),
                ]
            )
        )
        for attempt_index, source_count in enumerate(source_attempts):
            effective_patch_match_source_images = source_count
            _configure_patch_match(
                patch_match_config,
                source_count,
                reference_images,
            )
            try:
                _run(executable, patch_args, log_path, callback=patch_progress)
                return
            except Exception as exc:
                removed_after = _remove_invalid_dense_maps(dense)
                if removed_after:
                    _logger.warning(
                        "Removed %d incomplete COLMAP dense maps after failure",
                        len(removed_after),
                    )
                has_retry = attempt_index + 1 < len(source_attempts)
                if not (use_gpu and has_retry and _cuda_out_of_memory(exc)):
                    raise
                next_sources = source_attempts[attempt_index + 1]
                fallback = (
                    "PatchMatch显存不足：保持MVS分辨率不变，"
                    f"源照片数从 {source_count} 自动降为 {next_sources}"
                )
                _logger.warning(fallback)
                ai_runtime["fallbacks"].append(fallback)
                save_ai_runtime()
                update(0.68, fallback)

    run_stage(
        "patch_match",
        0.68,
        "CUDA PatchMatch Stereo 稠密深度",
        dense_maps_valid,
        run_patch_match,
    )

    if geometric_consistency:
        removed_maps, freed_bytes = _remove_photometric_maps_after_geometric_patchmatch(
            dense
        )
        if removed_maps:
            _logger.info(
                "Removed %d photometric maps after geometric PatchMatch "
                "and released %.2f GB",
                removed_maps,
                freed_bytes / 1024**3,
            )
            update(
                0.87,
                f"已释放光度中间图 {freed_bytes / 1024**3:.1f} GB，准备融合",
            )

    fused = dense / "fused.ply"
    partial_fused = dense / "fused.partial.ply"
    fusion_cache_size_gb, fusion_num_threads = _fusion_resources()

    def fuse_depth_maps() -> None:
        if fused.is_file() and _ply_vertex_count(fused) == 0:
            fused.unlink()
        input_type = "geometric" if geometric_consistency else "photometric"
        fusion_config = dense / "stereo" / "fusion.cfg"
        canonical_fusion_config = dense / "stereo" / "fusion.full.cfg"
        full_reference_names = list(reference_images)
        if not full_reference_names:
            full_reference_names = (
                _fusion_reference_names(canonical_fusion_config)
                if canonical_fusion_config.is_file()
                else _fusion_reference_names(fusion_config)
            )
        # Keep a canonical list outside COLMAP's mutable fusion.cfg. If the
        # worker is forcibly terminated mid-batch, the next resume restores
        # all reference images before starting another COLMAP subprocess.
        _write_fusion_config(canonical_fusion_config, full_reference_names)
        _write_fusion_config(fusion_config, full_reference_names)
        batches, dense_map_bytes = _plan_fusion_batches(
            dense,
            full_reference_names,
            input_type=input_type,
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "images": full_reference_names,
                    "input_type": input_type,
                    "min_num_pixels": int(fusion_min_num_pixels),
                    "check_num_images": int(effective_patch_match_source_images),
                    "batch_size": max(len(batch) for batch in batches),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        batch_root = dense / f"fusion_batches_{fingerprint}"
        batch_root.mkdir(parents=True, exist_ok=True)
        batch_outputs: list[Path] = []
        partial_fused.unlink(missing_ok=True)
        _logger.info(
            "StereoFusion plan: %d images, %.2f GB maps, %d batch(es)",
            len(full_reference_names),
            dense_map_bytes / 1024**3,
            len(batches),
        )

        try:
            for batch_index, batch_names in enumerate(batches, start=1):
                batch_output = batch_root / f"batch_{batch_index:04d}.ply"
                batch_outputs.append(batch_output)
                try:
                    if _colmap_fused_ply_info(batch_output)[0] > 0:
                        update(
                            0.88 + 0.06 * batch_index / len(batches),
                            f"复用稠密融合分块 {batch_index}/{len(batches)}",
                        )
                        continue
                except RuntimeError:
                    _remove_stereo_fusion_output(batch_output)
                _write_fusion_config(fusion_config, batch_names)
                update(
                    0.88 + 0.06 * (batch_index - 1) / len(batches),
                    f"多线程融合高分辨率分块 {batch_index}/{len(batches)}（{len(batch_names)}张）",
                )
                try:
                    _run(
                        executable,
                        _stereo_fusion_arguments(
                            dense,
                            batch_output,
                            geometric_consistency=bool(geometric_consistency),
                            min_num_pixels=int(fusion_min_num_pixels),
                            check_num_images=int(effective_patch_match_source_images),
                            use_cache=False,
                            cache_size_gb=fusion_cache_size_gb,
                            num_threads=fusion_num_threads,
                        ),
                        log_path,
                    )
                except RuntimeError as exc:
                    _remove_stereo_fusion_output(batch_output)
                    if not _retryable_fusion_failure(exc):
                        raise
                    update(
                        0.88 + 0.06 * (batch_index - 1) / len(batches),
                        f"分块 {batch_index}/{len(batches)} 内存异常，切换低内存缓存重试",
                    )
                    _run(
                        executable,
                        _stereo_fusion_arguments(
                            dense,
                            batch_output,
                            geometric_consistency=bool(geometric_consistency),
                            min_num_pixels=int(fusion_min_num_pixels),
                            check_num_images=int(effective_patch_match_source_images),
                            use_cache=True,
                            cache_size_gb=min(4, fusion_cache_size_gb),
                            num_threads=min(4, fusion_num_threads),
                        ),
                        log_path,
                    )
                _colmap_fused_ply_info(batch_output)
        finally:
            _write_fusion_config(fusion_config, full_reference_names)

        update(0.945, f"合并 {len(batch_outputs)} 个稠密点云分块")
        _merge_colmap_fused_plys(batch_outputs, partial_fused)
        partial_fused.replace(fused)
        for batch_output in batch_outputs:
            _remove_stereo_fusion_output(batch_output)
        if _ply_vertex_count(fused) == 0:
            stages.set(
                "patch_match",
                "pending",
                message="融合结果为空；下次继续时重新执行 PatchMatch",
            )
            raise RuntimeError("COLMAP融合点云为空")

    run_stage(
        "stereo_fusion",
        0.88,
        "融合原图高分辨率稠密点云",
        lambda: _ply_vertex_count(fused) > 0,
        fuse_depth_maps,
    )

    final_ply = dense / "pointcloud_ai_photogrammetry.ply"
    output_metadata_path = dense / "pointcloud_output.json"


    def write_pointcloud_output() -> None:
        point_count = _ply_vertex_count(fused)
        if point_count <= 0:
            raise RuntimeError("COLMAP融合点云为空，无法写出最终成果")
        if final_ply.exists():
            final_ply.unlink()
        try:
            # The native fused PLY is already the desired full-resolution
            # colored cloud. A hard link avoids loading huge projects into RAM
            # and avoids consuming a second copy on disk.
            os.link(fused, final_ply)
        except OSError as exc:
            _logger.warning(
                "Could not hard-link final PLY; using COLMAP fused output: %s",
                exc,
            )
        output_metadata_path.write_text(
            json.dumps(
                {
                    "coordinate_source": "sfm_native",
                    "unit": "model_units",
                    "point_count": int(point_count),
                    "storage": (
                        "hardlink"
                        if final_ply.is_file()
                        else "colmap_fused_fallback"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    run_stage(
        "pointcloud_output",
        0.95,
        "写出 SfM 原生坐标点云",
        lambda: (final_ply.is_file() or fused.is_file())
        and output_metadata_path.is_file(),
        write_pointcloud_output,
    )

    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    output_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
    registered_images = int(mapping_payload["registered_images"])
    registration_ratio = float(mapping_payload["registration_ratio"])
    warnings: list[str] = []
    if registration_ratio < 0.8:
        warnings.append(
            f"仅注册 {registered_images}/{len(expected_names)} 张照片；"
            "建议检查拍摄断层或改用另一种SfM求解器"
        )
    update(1.0, "AI特征摄影测量完成")
    pointcloud_path = final_ply if final_ply.is_file() else fused
    return {
        **sparse_result,
        "result_stage": "dense",
        "pointcloud": str(pointcloud_path),
        "raw_fused": str(fused),
        "dense_workspace": str(dense),
        "dense_cache_fingerprint": dense_fingerprint,
        "pointcloud_metadata": str(output_metadata_path),
        "registered_images": registered_images,
        "registration_ratio": registration_ratio,
        "point_count": int(output_metadata.get("point_count", 0)),
        "mvs_settings": {
            "max_image_size": int(max_image_size),
            "geometric_consistency": bool(geometric_consistency),
            "filter": bool(patch_match_filter),
            "source_images": int(effective_patch_match_source_images),
            "requested_source_images": int(patch_match_source_images),
            "iterations": int(patch_match_iterations),
            "cache_size_gb": int(mvs_cache_size_gb),
            "fusion_cache_size_gb": int(fusion_cache_size_gb),
            "fusion_num_threads": int(fusion_num_threads),
            "fusion_strategy": "adaptive_full_resolution_batches",
            "fusion_disk_cache_fallback": True,
            "configured_reference_images": int(configured_reference_images),
            "reference_strategy": mvs_reference_strategy,
            "requested_reference_ratio": float(mvs_reference_ratio),
            "sparse_point_coverage_ratio": float(
                mvs_selection["sparse_point_coverage_ratio"]
            ),
        },
        "unit": "模型单位",
        "estimated_workspace_gb": round(
            estimated_workspace_bytes / 1024**3,
            2,
        ),
        "disk_free_gb_at_start": round(disk_free_bytes / 1024**3, 2),
        "warnings": [*sparse_warnings, *warnings],
    }
