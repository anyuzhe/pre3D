"""Recover, repair, texture and export a triangle model from COLMAP MVS.

The dense reconstruction remains the source of truth.  This module adds a
restartable post-process which runs in the existing crash-isolated worker:

fused points -> conditioned oriented points -> Poisson surface -> repaired
triangle mesh -> simplified mesh -> COLMAP multi-view texture atlas -> common
delivery formats.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import psutil
import pyvista as pv
import trimesh
import vtk
from PIL import Image
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from .colmap_pipeline import _run, find_colmap
from .project_store import StageTracker
from .runtime_paths import resource_root

ProgressCallback = Callable[[float, str], None]


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "快速预览": {
        "point_limit": 2_000_000,
        "outlier_sample_size": 10,
        "outlier_std_ratio": 3.0,
        "poisson_depth": 10,
        "poisson_trim": 10.0,
        "target_faces": 250_000,
        "hole_size_ratio": 0.006,
        "texture_scale_factor": 0.5,
        "texture_block_target_faces": 80_000,
        "texture_atlas_max_dimension": 32_768,
        "texture_atlas_max_pixels": 120_000_000,
    },
    "标准工程模式": {
        "point_limit": 5_000_000,
        "outlier_sample_size": 12,
        "outlier_std_ratio": 2.75,
        "poisson_depth": 11,
        "poisson_trim": 10.0,
        "target_faces": 750_000,
        "hole_size_ratio": 0.004,
        "texture_scale_factor": 0.75,
        "texture_block_target_faces": 100_000,
        "texture_atlas_max_dimension": 32_768,
        "texture_atlas_max_pixels": 160_000_000,
    },
    "高精度模式": {
        "point_limit": 10_000_000,
        "outlier_sample_size": 16,
        "outlier_std_ratio": 2.5,
        "poisson_depth": 12,
        "poisson_trim": 9.0,
        "target_faces": 1_500_000,
        "hole_size_ratio": 0.0025,
        "texture_scale_factor": 1.0,
        "texture_block_target_faces": 125_000,
        "texture_atlas_max_dimension": 32_768,
        "texture_atlas_max_pixels": 160_000_000,
    },
}

_PRESET_ALIASES = {
    "标准重建": "标准工程模式",
    "高精度重建": "高精度模式",
}

_PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _notify(callback: ProgressCallback | None, value: float, text: str) -> None:
    if callback:
        callback(float(np.clip(value, 0.0, 1.0)), text)


def _normalized_preset(name: str) -> str:
    normalized = _PRESET_ALIASES.get(name, name)
    return normalized if normalized in MODEL_PRESETS else "标准工程模式"


def _ply_header(path: str | Path) -> dict[str, Any]:
    """Read enough of a PLY header for memory-mapped vertex access."""

    source = Path(path)
    elements: dict[str, int] = {}
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    ply_format = ""
    with source.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"不是PLY文件：{source}")
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"PLY头不完整：{source}")
            line = raw.decode("ascii", errors="strict").strip()
            parts = line.split()
            if parts[:1] == ["format"] and len(parts) >= 2:
                ply_format = parts[1]
            elif parts[:1] == ["element"] and len(parts) == 3:
                current_element = parts[1]
                elements[current_element] = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError("暂不支持带列表属性的PLY顶点")
                vertex_properties.append((parts[2], parts[1]))
            elif line == "end_header":
                offset = stream.tell()
                break
    return {
        "format": ply_format,
        "elements": elements,
        "vertex_properties": vertex_properties,
        "data_offset": offset,
    }


def ply_counts(path: str | Path) -> tuple[int, int]:
    header = _ply_header(path)
    elements = header["elements"]
    return int(elements.get("vertex", 0)), int(elements.get("face", 0))


def _vertex_memmap(path: Path) -> tuple[np.memmap, dict[str, Any]]:
    header = _ply_header(path)
    if header["format"] != "binary_little_endian":
        raise ValueError("模型点云必须为 binary_little_endian PLY")
    fields: list[tuple[str, str]] = []
    for name, ply_type in header["vertex_properties"]:
        dtype = _PLY_TYPES.get(ply_type)
        if dtype is None:
            raise ValueError(f"不支持的PLY属性类型：{ply_type}")
        fields.append((name, dtype))
    required = {"x", "y", "z"}
    if not required.issubset(name for name, _dtype in fields):
        raise ValueError("PLY缺少x/y/z顶点坐标")
    vertices = np.memmap(
        path,
        mode="r",
        dtype=np.dtype(fields),
        offset=int(header["data_offset"]),
        shape=(int(header["elements"].get("vertex", 0)),),
    )
    return vertices, header


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    # A regular stride is deterministic and avoids allocating a random index
    # array the size of very large (50M+) fused clouds.
    step = int(math.ceil(count / max(limit, 1)))
    return np.arange(0, count, step, dtype=np.int64)[:limit]


def _repair_normals(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
    repaired = np.zeros_like(normals, dtype=np.float32)
    repaired[valid] = normals[valid] / lengths[valid, None]
    invalid_indices = np.flatnonzero(~valid)
    if not len(invalid_indices):
        return repaired
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) >= 3:
        tree = cKDTree(points[valid_indices])
        k = min(8, len(valid_indices))
        _distance, neighbors = tree.query(points[invalid_indices], k=k, workers=-1)
        neighbors = np.asarray(neighbors)
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        estimates = repaired[valid_indices[neighbors]].mean(axis=1)
        estimate_lengths = np.linalg.norm(estimates, axis=1)
        good = estimate_lengths > 1e-8
        repaired[invalid_indices[good]] = estimates[good] / estimate_lengths[good, None]
    still_invalid = np.linalg.norm(repaired, axis=1) <= 1e-8
    if np.any(still_invalid):
        center = np.median(points, axis=0)
        fallback = points[still_invalid] - center
        lengths = np.linalg.norm(fallback, axis=1)
        good = lengths > 1e-8
        fallback[good] /= lengths[good, None]
        fallback[~good] = (0.0, 0.0, 1.0)
        repaired[still_invalid] = fallback.astype(np.float32)
    return repaired


def _statistical_filter(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    *,
    sample_size: int,
    std_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(vtk_points)
    normal_array = numpy_to_vtk(np.ascontiguousarray(normals), deep=True)
    normal_array.SetName("Normals")
    cloud.GetPointData().SetNormals(normal_array)
    color_array = numpy_to_vtk(
        np.ascontiguousarray(colors),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    color_array.SetName("RGB")
    cloud.GetPointData().AddArray(color_array)

    outlier = vtk.vtkStatisticalOutlierRemoval()
    outlier.SetInputData(cloud)
    outlier.SetSampleSize(int(sample_size))
    outlier.SetStandardDeviationFactor(float(std_ratio))
    outlier.Update()
    output = outlier.GetOutput()
    filtered_points = vtk_to_numpy(output.GetPoints().GetData()).astype(np.float32, copy=False)
    output_normals = output.GetPointData().GetNormals()
    output_colors = output.GetPointData().GetArray("RGB")
    if output_normals is None or output_colors is None:
        # VTK versions which do not forward arrays still provide the filtered
        # point ids only indirectly. Fall back to the repaired unfiltered set
        # instead of silently writing a cloud without oriented normals.
        return points, normals, colors
    return (
        filtered_points,
        vtk_to_numpy(output_normals).astype(np.float32, copy=False),
        vtk_to_numpy(output_colors).astype(np.uint8, copy=False),
    )


def _write_oriented_point_ply(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    records = np.empty(len(points), dtype=dtype)
    for column, name in enumerate(("x", "y", "z")):
        records[name] = points[:, column]
    for column, name in enumerate(("nx", "ny", "nz")):
        records[name] = normals[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, column]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment model_pipeline conditioned oriented points\n"
        f"element vertex {len(records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)
    os.replace(temporary, path)


def condition_point_cloud(
    source: str | Path,
    output: str | Path,
    *,
    point_limit: int,
    outlier_sample_size: int,
    outlier_std_ratio: float,
) -> dict[str, Any]:
    """Memory-map a huge fused PLY, downsample, denoise and repair normals."""

    source_path = Path(source)
    target_path = Path(output)
    vertices, _header = _vertex_memmap(source_path)
    input_count = len(vertices)
    if input_count < 3:
        raise RuntimeError("稠密点云不足3个点，无法重建表面")
    indices = _sample_indices(input_count, int(point_limit))
    points = np.column_stack([vertices[name][indices] for name in ("x", "y", "z")]).astype(
        np.float32,
        copy=False,
    )
    names = set(vertices.dtype.names or ())
    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.column_stack(
            [vertices[name][indices] for name in ("nx", "ny", "nz")]
        ).astype(np.float32, copy=False)
    else:
        normals = np.zeros_like(points)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack(
            [vertices[name][indices] for name in ("red", "green", "blue")]
        ).astype(np.uint8, copy=False)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points, normals, colors = points[finite], normals[finite], colors[finite]
    normals = _repair_normals(points, normals)
    sampled_count = len(points)
    if len(points) >= max(32, int(outlier_sample_size) + 2):
        points, normals, colors = _statistical_filter(
            points,
            normals,
            colors,
            sample_size=int(outlier_sample_size),
            std_ratio=float(outlier_std_ratio),
        )
        normals = _repair_normals(points, normals)
    if len(points) < 3:
        raise RuntimeError("点云去噪后没有足够的有效点")
    _write_oriented_point_ply(target_path, points, normals, colors)
    return {
        "input_point_count": int(input_count),
        "sampled_point_count": int(sampled_count),
        "output_point_count": int(len(points)),
        "removed_as_outliers": int(sampled_count - len(points)),
        "point_limit": int(point_limit),
        "normal_repair": "normalize_and_neighbor_repair",
        "statistical_sample_size": int(outlier_sample_size),
        "statistical_std_ratio": float(outlier_std_ratio),
    }


def repair_mesh(source: str | Path, output: str | Path, *, hole_size_ratio: float) -> dict[str, Any]:
    mesh = pv.read(source).extract_surface().triangulate().clean(tolerance=0.0)
    if mesh.n_cells == 0:
        raise RuntimeError("表面重建没有生成有效三角面")
    before_faces = int(mesh.n_cells)
    diagonal = float(np.linalg.norm(np.asarray(mesh.bounds)[1::2] - np.asarray(mesh.bounds)[0::2]))
    hole_size = max(diagonal * float(hole_size_ratio), np.finfo(np.float32).eps)
    repaired = mesh.fill_holes(hole_size).triangulate().clean(tolerance=0.0)
    repaired = repaired.compute_normals(
        cell_normals=True,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        split_vertices=False,
        inplace=False,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    repaired.save(output, binary=True, recompute_normals=False)
    return {
        "input_vertices": int(mesh.n_points),
        "input_faces": before_faces,
        "output_vertices": int(repaired.n_points),
        "output_faces": int(repaired.n_cells),
        "hole_size": hole_size,
        "hole_size_ratio": float(hole_size_ratio),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_triangle_mesh_block(
    path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[int, int]:
    used_vertices, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
    local_points = np.ascontiguousarray(points[used_vertices], dtype=np.float32)
    local_triangles = inverse.reshape(-1, 3).astype(np.int64, copy=False)
    cells = np.empty((len(local_triangles), 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = local_triangles
    block = pv.PolyData(local_points, cells.reshape(-1))
    path.parent.mkdir(parents=True, exist_ok=True)
    block.save(path, binary=True)
    return int(len(local_points)), int(len(local_triangles))


def partition_mesh_for_texturing(
    source: str | Path,
    output_root: str | Path,
    *,
    target_faces: int,
) -> dict[str, Any]:
    """Partition a mesh into spatially coherent, face-disjoint texture blocks."""

    source_path = Path(source).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    manifest_path = root / "partition_manifest.json"
    if target_faces < 1:
        raise ValueError("纹理分块目标面数必须大于0")
    if manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_stat = source_path.stat()
            if (
                int(cached.get("version", 0)) == 1
                and int(cached.get("target_faces", 0)) == int(target_faces)
                and int(cached.get("source_size", -1)) == source_stat.st_size
                and int(cached.get("source_mtime_ns", -1)) == source_stat.st_mtime_ns
                and bool(cached.get("blocks"))
                and all(
                    Path(str(block.get("mesh", ""))).is_file()
                    and ply_counts(Path(str(block["mesh"])))[1]
                    == int(block.get("face_count", -1))
                    for block in cached.get("blocks", [])
                )
            ):
                return cached
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    mesh = pv.read(source_path).extract_surface().triangulate()
    points = np.asarray(mesh.points, dtype=np.float64)
    face_rows = np.asarray(mesh.faces).reshape(-1, 4)
    if not len(face_rows) or not np.all(face_rows[:, 0] == 3):
        raise RuntimeError("待纹理网格没有有效三角面")
    triangles = np.ascontiguousarray(face_rows[:, 1:4], dtype=np.int64)
    centroids = points[triangles].mean(axis=1)
    desired_blocks = max(1, int(math.ceil(len(triangles) / int(target_faces))))

    def split(indices: np.ndarray, count: int) -> list[np.ndarray]:
        if count <= 1 or len(indices) <= 1:
            return [indices]
        values = centroids[indices]
        spans = np.ptp(values, axis=0)
        axis = int(np.argmax(spans))
        left_blocks = count // 2
        split_at = int(round(len(indices) * left_blocks / count))
        split_at = min(max(1, split_at), len(indices) - 1)
        order = np.argpartition(values[:, axis], split_at)
        left = indices[order[:split_at]]
        right = indices[order[split_at:]]
        return [
            *split(left, left_blocks),
            *split(right, count - left_blocks),
        ]

    leaves = split(np.arange(len(triangles), dtype=np.int64), desired_blocks)
    blocks: list[dict[str, Any]] = []
    root.mkdir(parents=True, exist_ok=True)
    for index, face_indices in enumerate(leaves, 1):
        block_id = f"block_{index:04d}"
        block_mesh = root / block_id / "input_mesh.ply"
        vertex_count, face_count = _write_triangle_mesh_block(
            block_mesh,
            points,
            triangles[face_indices],
        )
        block_centroids = centroids[face_indices]
        blocks.append(
            {
                "id": block_id,
                "mesh": str(block_mesh),
                "vertex_count": vertex_count,
                "face_count": face_count,
                "bounds_min": np.min(block_centroids, axis=0).tolist(),
                "bounds_max": np.max(block_centroids, axis=0).tolist(),
            }
        )
    if sum(int(block["face_count"]) for block in blocks) != len(triangles):
        raise RuntimeError("纹理分块面数校验失败")
    source_stat = source_path.stat()
    payload = {
        "version": 1,
        "strategy": "recursive_longest_axis_face_centroids",
        "source_mesh": str(source_path),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "source_vertex_count": int(len(points)),
        "source_face_count": int(len(triangles)),
        "target_faces": int(target_faces),
        "block_count": len(blocks),
        "blocks": blocks,
    }
    _atomic_json(manifest_path, payload)
    return payload


def _partition_manifest_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        blocks = list(payload.get("blocks") or [])
        return bool(blocks) and sum(
            ply_counts(Path(str(block["mesh"])))[1] for block in blocks
        ) == int(payload["source_face_count"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _texture_block_valid(
    mesh_path: Path,
    texture_path: Path,
    *,
    max_dimension: int,
    max_pixels: int,
) -> tuple[bool, tuple[int, int]]:
    if not mesh_path.is_file() or not texture_path.is_file():
        return False, (0, 0)
    try:
        if ply_counts(mesh_path)[1] <= 0:
            return False, (0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(texture_path) as image:
                image.verify()
            with Image.open(texture_path) as image:
                width, height = image.size
        valid = (
            width > 0
            and height > 0
            and max(width, height) <= int(max_dimension)
            and width * height <= int(max_pixels)
        )
        return valid, (int(width), int(height))
    except (OSError, ValueError, Image.DecompressionBombError):
        return False, (0, 0)


def _retryable_texture_failure(error: BaseException) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "3221226505",
            "0xc0000409",
            "atlas size:",
            "baking texture",
            "超过安全上限",
            "out of memory",
            "bad allocation",
            "std::bad_alloc",
        )
    )


def _reset_texture_output(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise RuntimeError(f"拒绝清理纹理工作区以外的目录：{resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def texture_mesh_blocks(
    *,
    executable: str,
    dense_workspace: Path,
    partition: dict[str, Any],
    output_root: Path,
    texture_scale_factor: float,
    atlas_max_dimension: int,
    atlas_max_pixels: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Texture each spatial mesh block into an independently bounded atlas."""

    output_root.mkdir(parents=True, exist_ok=True)
    block_tracker = StageTracker(output_root / "pipeline_state.json")
    manifest_path = output_root / "texture_manifest.json"
    source_blocks = list(partition.get("blocks") or [])
    if not source_blocks:
        raise RuntimeError("纹理分块计划为空")
    scales = []
    for ratio in (1.0, 0.75, 0.5, 0.35, 0.25, 0.125):
        value = max(0.0625, float(texture_scale_factor) * ratio)
        if all(not np.isclose(value, existing) for existing in scales):
            scales.append(value)
    completed_blocks: list[dict[str, Any]] = []
    for block_index, source_block in enumerate(source_blocks, 1):
        block_id = str(source_block["id"])
        block_root = output_root / block_id
        textured_root = block_root / "textured"
        textured_mesh = textured_root / "mesh.ply"
        texture_png = textured_root / "texture.png"
        block_log = block_root / "mesh_texturer.log"
        cached_details = dict(
            block_tracker.data.get("stages", {}).get(block_id, {}).get("details")
            or {}
        )
        cached_valid, cached_size = _texture_block_valid(
            textured_mesh,
            texture_png,
            max_dimension=atlas_max_dimension,
            max_pixels=atlas_max_pixels,
        )
        if block_tracker.status(block_id) == "completed" and cached_valid:
            details = {
                **cached_details,
                "id": block_id,
                "mesh": str(textured_mesh),
                "texture": str(texture_png),
                "atlas_width": cached_size[0],
                "atlas_height": cached_size[1],
            }
            completed_blocks.append(details)
            _notify(
                progress_callback,
                0.67 + 0.21 * block_index / len(source_blocks),
                f"恢复纹理块 {block_index}/{len(source_blocks)}",
            )
            continue

        block_tracker.set(
            block_id,
            "running",
            message=f"纹理块 {block_index}/{len(source_blocks)}",
            details={"input_mesh": str(source_block["mesh"])},
        )
        last_error: BaseException | None = None
        selected_scale = scales[-1]
        selected_size = (0, 0)
        for attempt, scale in enumerate(scales, 1):
            selected_scale = scale
            _reset_texture_output(textured_root, output_root)
            _notify(
                progress_callback,
                0.67 + 0.21 * (block_index - 1) / len(source_blocks),
                f"纹理块 {block_index}/{len(source_blocks)}，图集比例 {scale:.3g}",
            )
            try:
                _run(
                    executable,
                    [
                        "mesh_texturer",
                        "--workspace_path",
                        str(dense_workspace),
                        "--input_path",
                        str(source_block["mesh"]),
                        "--output_path",
                        str(textured_root),
                        "--MeshTextureMapping.view_selection_smoothing_iterations",
                        "3",
                        "--MeshTextureMapping.atlas_patch_padding",
                        "2",
                        "--MeshTextureMapping.inpaint_radius",
                        "5",
                        "--MeshTextureMapping.apply_color_correction",
                        "1",
                        "--MeshTextureMapping.texture_scale_factor",
                        f"{scale:.9g}",
                    ],
                    block_log,
                )
                valid, atlas_size = _texture_block_valid(
                    textured_mesh,
                    texture_png,
                    max_dimension=atlas_max_dimension,
                    max_pixels=atlas_max_pixels,
                )
                if not valid:
                    width, height = atlas_size
                    raise RuntimeError(
                        "纹理图集超过安全上限："
                        f"{width}×{height}，限制 {atlas_max_dimension}px / "
                        f"{atlas_max_pixels:,}像素"
                    )
                selected_size = atlas_size
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= len(scales) or not _retryable_texture_failure(exc):
                    break
        if last_error is not None:
            block_tracker.set(block_id, "failed", message=str(last_error))
            raise RuntimeError(f"纹理块 {block_id} 生成失败：{last_error}") from last_error
        vertices, faces = ply_counts(textured_mesh)
        details = {
            "id": block_id,
            "input_mesh": str(source_block["mesh"]),
            "mesh": str(textured_mesh),
            "texture": str(texture_png),
            "vertex_count": int(vertices),
            "face_count": int(faces),
            "texture_scale_factor": float(selected_scale),
            "atlas_width": int(selected_size[0]),
            "atlas_height": int(selected_size[1]),
            "atlas_pixels": int(selected_size[0] * selected_size[1]),
            "log": str(block_log),
        }
        block_tracker.set(block_id, "completed", details=details)
        completed_blocks.append(details)

    payload = {
        "version": 2,
        "strategy": "spatial_mesh_blocks_multi_atlas",
        "block_count": len(completed_blocks),
        "face_count": int(sum(int(block["face_count"]) for block in completed_blocks)),
        "vertex_count": int(
            sum(int(block["vertex_count"]) for block in completed_blocks)
        ),
        "atlas_max_dimension": int(atlas_max_dimension),
        "atlas_max_pixels": int(atlas_max_pixels),
        "blocks": completed_blocks,
    }
    _atomic_json(manifest_path, payload)
    payload["manifest"] = str(manifest_path)
    return payload


def _texture_manifest_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        blocks = list(payload.get("blocks") or [])
        return bool(blocks) and all(
            _texture_block_valid(
                Path(str(block["mesh"])),
                Path(str(block["texture"])),
                max_dimension=int(payload["atlas_max_dimension"]),
                max_pixels=int(payload["atlas_max_pixels"]),
            )[0]
            for block in blocks
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _mesh_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh = pv.read(path).extract_surface().triangulate()
    faces = np.asarray(mesh.faces).reshape(-1, 4)
    if len(faces) and not np.all(faces[:, 0] == 3):
        raise RuntimeError("纹理模型包含非三角面")
    triangles = np.ascontiguousarray(faces[:, 1:4], dtype=np.int64)
    uv = mesh.active_texture_coordinates
    if uv is None:
        uv = mesh.point_data.get("TCoords")
    if uv is None or len(uv) != mesh.n_points:
        raise RuntimeError("纹理网格缺少有效UV坐标")
    normals = mesh.point_normals
    if normals is None or len(normals) != mesh.n_points:
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=False,
        )
        normals = mesh.point_normals
    return (
        np.ascontiguousarray(mesh.points, dtype=np.float32),
        triangles,
        np.ascontiguousarray(uv, dtype=np.float32),
        np.ascontiguousarray(normals, dtype=np.float32),
    )


def _write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    normals: np.ndarray,
    texture_name: str,
) -> None:
    mtl_path = path.with_suffix(".mtl")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"mtllib {mtl_path.name}\no PhotogrammetryModel\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uv:
            stream.write(f"vt {u:.9g} {v:.9g}\n")
        for nx, ny, nz in normals:
            stream.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
        stream.write("usemtl PhotogrammetryTexture\ns 1\n")
        for a, b, c in faces + 1:
            stream.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    mtl_path.write_text(
        "newmtl PhotogrammetryTexture\n"
        "Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
        f"map_Kd {texture_name}\n",
        encoding="utf-8",
    )


def _write_fbx_ascii(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    normals: np.ndarray,
    texture_name: str,
) -> None:
    """Write an interoperable FBX 7.4 ASCII mesh with one texture material."""

    def values(stream, prefix: str, array: Iterable[Any], per_line: int = 24) -> None:
        stream.write(prefix)
        first = True
        count = 0
        for value in array:
            if not first:
                stream.write(",")
            stream.write(str(value))
            first = False
            count += 1
            if count % per_line == 0:
                stream.write("\n        ")
        stream.write("\n")

    polygon_indices = faces.astype(np.int64, copy=True)
    polygon_indices[:, 2] = -polygon_indices[:, 2] - 1
    flat_vertices = (f"{float(value):.9g}" for value in vertices.ravel())
    flat_normals = (f"{float(value):.9g}" for value in normals[faces].reshape(-1))
    flat_uv = (f"{float(value):.9g}" for value in uv.ravel())
    uv_indices = faces.ravel()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "; FBX 7.4.0 project file\n"
            "FBXHeaderExtension:  {\n  FBXHeaderVersion: 1003\n  FBXVersion: 7400\n}\n"
            "GlobalSettings:  {\n  Version: 1000\n  Properties70:  {\n"
            "    P: \"UpAxis\", \"int\", \"Integer\", \"\",1\n"
            "    P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n"
            "    P: \"FrontAxis\", \"int\", \"Integer\", \"\",2\n"
            "    P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",-1\n"
            "    P: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n"
            "    P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n"
            "    P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1\n"
            "  }\n}\n"
            "Definitions:  {\n  Version: 100\n  Count: 4\n"
            "  ObjectType: \"Geometry\" { Count: 1 }\n"
            "  ObjectType: \"Model\" { Count: 1 }\n"
            "  ObjectType: \"Material\" { Count: 1 }\n"
            "  ObjectType: \"Texture\" { Count: 1 }\n}\n"
            "Objects:  {\n"
            "  Geometry: 1001, \"Geometry::PhotogrammetryModel\", \"Mesh\" {\n"
        )
        stream.write(f"    Vertices: *{vertices.size} {{\n      a: ")
        values(stream, "", flat_vertices)
        stream.write(f"    }}\n    PolygonVertexIndex: *{polygon_indices.size} {{\n      a: ")
        values(stream, "", polygon_indices.ravel())
        stream.write(
            "    }\n    LayerElementNormal: 0 {\n"
            "      Version: 101\n      Name: \"\"\n"
            "      MappingInformationType: \"ByPolygonVertex\"\n"
            "      ReferenceInformationType: \"Direct\"\n"
        )
        stream.write(f"      Normals: *{faces.size * 3} {{\n        a: ")
        values(stream, "", flat_normals)
        stream.write(
            "      }\n    }\n    LayerElementUV: 0 {\n"
            "      Version: 101\n      Name: \"UVMap\"\n"
            "      MappingInformationType: \"ByVertice\"\n"
            "      ReferenceInformationType: \"IndexToDirect\"\n"
        )
        stream.write(f"      UV: *{uv.size} {{\n        a: ")
        values(stream, "", flat_uv)
        stream.write(f"      }}\n      UVIndex: *{uv_indices.size} {{\n        a: ")
        values(stream, "", uv_indices)
        stream.write(
            "      }\n    }\n    LayerElementMaterial: 0 {\n"
            "      Version: 101\n      Name: \"\"\n"
            "      MappingInformationType: \"AllSame\"\n"
            "      ReferenceInformationType: \"IndexToDirect\"\n"
            "      Materials: *1 { a: 0 }\n    }\n"
            "    Layer: 0 {\n      Version: 100\n"
            "      LayerElement: { Type: \"LayerElementNormal\" TypedIndex: 0 }\n"
            "      LayerElement: { Type: \"LayerElementUV\" TypedIndex: 0 }\n"
            "      LayerElement: { Type: \"LayerElementMaterial\" TypedIndex: 0 }\n"
            "    }\n  }\n"
            "  Model: 1002, \"Model::PhotogrammetryModel\", \"Mesh\" {\n"
            "    Version: 232\n    Shading: T\n    Culling: \"CullingOff\"\n  }\n"
            "  Material: 1003, \"Material::PhotogrammetryTexture\", \"\" {\n"
            "    Version: 102\n    ShadingModel: \"lambert\"\n    MultiLayer: 0\n"
            "    Properties70: { P: \"DiffuseColor\", \"Color\", \"\", \"A\",1,1,1 }\n"
            "  }\n"
            "  Texture: 1004, \"Texture::PhotogrammetryTexture\", \"TextureVideoClip\" {\n"
            "    Type: \"TextureVideoClip\"\n    Version: 202\n"
            f"    TextureName: \"Texture::PhotogrammetryTexture\"\n    FileName: \"{texture_name}\"\n"
            f"    RelativeFilename: \"{texture_name}\"\n    UVSet: \"UVMap\"\n  }}\n"
            "}\nConnections:  {\n"
            "  C: \"OO\",1001,1002\n  C: \"OO\",1002,0\n"
            "  C: \"OO\",1003,1002\n"
            "  C: \"OP\",1004,1003,\"DiffuseColor\"\n}\n"
        )


def _write_gltf(
    folder: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    normals: np.ndarray,
    texture_path: Path,
) -> tuple[Path, Path]:
    image = Image.open(texture_path).convert("RGB")
    material = trimesh.visual.material.PBRMaterial(
        name="PhotogrammetryTexture",
        baseColorTexture=image,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        visual=visual,
        process=False,
        validate=False,
    )
    scene = trimesh.Scene(mesh)
    glb_path = folder / "model.glb"
    glb_path.write_bytes(
        trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    )
    gltf_path = folder / "model.gltf"
    payload = trimesh.exchange.gltf.export_gltf(
        scene,
        include_normals=True,
        merge_buffers=True,
        embed_buffers=True,
    )
    for name, data in payload.items():
        destination = gltf_path if name.lower().endswith(".gltf") else folder / name
        destination.write_bytes(data)
    if not gltf_path.is_file():
        candidates = list(folder.glob("*.gltf"))
        if not candidates:
            raise RuntimeError("glTF导出未生成主文件")
        candidates[0].replace(gltf_path)
    return gltf_path, glb_path


def find_osgconv(explicit_path: str | None = None) -> str | None:
    project_root = resource_root()
    candidates = [
        explicit_path,
        os.environ.get("OSGCONV_PATH"),
        str(project_root / "tools" / "openscenegraph" / "bin" / "osgconv.exe"),
        str(project_root / "tools" / "openscenegraph" / "Library" / "bin" / "osgconv.exe"),
        "osgconv.exe",
        "osgconv",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or candidate
        path = Path(resolved).expanduser()
        if path.is_file():
            return str(path.resolve())
    return None


def _model_exports_valid(manifest_path: Path, formats: Iterable[str]) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = set(formats)
    if "gltf" in required or "glb" in required:
        required.update({"gltf", "glb"})
    if "osgb" in required:
        required.add("obj")
    return all(
        isinstance(payload.get(name), str) and Path(payload[name]).is_file()
        for name in required
    )


def _convert_obj_to_osgb(
    obj_path: Path,
    osgb_path: Path,
    *,
    osgconv_path: str | None,
) -> None:
    converter = find_osgconv(osgconv_path)
    if not converter:
        raise RuntimeError(
            "选择了OSGB，但本机未找到OpenSceneGraph osgconv。"
            "请安装OpenSceneGraph并设置OSGCONV_PATH，其他模型格式不受影响。"
        )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    environment = os.environ.copy()
    runtime_dirs = [str(Path(converter).parent)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        runtime_dirs.append(str(Path(conda_prefix) / "Library" / "bin"))
    common_anaconda_runtime = Path("D:/anaconda3/Library/bin")
    if common_anaconda_runtime.is_dir():
        runtime_dirs.append(str(common_anaconda_runtime))
    environment["PATH"] = os.pathsep.join(
        [*runtime_dirs, environment.get("PATH", "")]
    )
    completed = subprocess.run(
        [converter, str(obj_path), str(osgb_path)],
        cwd=obj_path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
        env=environment,
        check=False,
    )
    if completed.returncode or not osgb_path.is_file():
        raise RuntimeError(
            "OSGB转换失败。\n" + (completed.stderr or completed.stdout)[-2000:]
        )


def export_textured_mesh(
    textured_ply: str | Path,
    texture_image: str | Path,
    output_dir: str | Path,
    *,
    formats: Iterable[str],
    osgconv_path: str | None = None,
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = {str(value).strip().lower() for value in formats}
    unsupported = requested - {"obj", "fbx", "gltf", "glb", "osgb"}
    if unsupported:
        raise ValueError(f"未知模型格式：{', '.join(sorted(unsupported))}")
    texture_source = Path(texture_image)
    texture_output = output / "model_texture.png"
    if texture_source.resolve() != texture_output.resolve():
        shutil.copy2(texture_source, texture_output)
    vertices, faces, uv, normals = _mesh_arrays(textured_ply)
    result: dict[str, str] = {"texture_atlas": str(texture_output)}
    obj_path = output / "model.obj"
    if requested & {"obj", "osgb"}:
        _write_obj(
            obj_path,
            vertices,
            faces,
            uv,
            normals,
            texture_output.name,
        )
        result["obj"] = str(obj_path)
        result["mtl"] = str(obj_path.with_suffix(".mtl"))
    if "fbx" in requested:
        fbx_path = output / "model.fbx"
        _write_fbx_ascii(
            fbx_path,
            vertices,
            faces,
            uv,
            normals,
            texture_output.name,
        )
        result["fbx"] = str(fbx_path)
    if requested & {"gltf", "glb"}:
        gltf_path, glb_path = _write_gltf(
            output,
            vertices,
            faces,
            uv,
            normals,
            texture_output,
        )
        result["gltf"] = str(gltf_path)
        result["glb"] = str(glb_path)
    if "osgb" in requested:
        osgb_path = output / "model.osgb"
        _convert_obj_to_osgb(obj_path, osgb_path, osgconv_path=osgconv_path)
        result["osgb"] = str(osgb_path)
    return result


def _write_multi_obj(
    path: Path,
    blocks: list[dict[str, Any]],
    texture_names: list[str],
) -> None:
    mtl_path = path.with_suffix(".mtl")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"mtllib {mtl_path.name}\n")
        offset = 0
        for block_index, (block, texture_name) in enumerate(
            zip(blocks, texture_names, strict=True),
            1,
        ):
            vertices, faces, uv, normals = _mesh_arrays(block["mesh"])
            stream.write(f"o TextureBlock_{block_index:04d}\n")
            for x, y, z in vertices:
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for u, v in uv:
                stream.write(f"vt {u:.9g} {v:.9g}\n")
            for nx, ny, nz in normals:
                stream.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
            stream.write(f"usemtl TextureBlock_{block_index:04d}\ns 1\n")
            for a, b, c in faces + offset + 1:
                stream.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            offset += len(vertices)
    with mtl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for block_index, texture_name in enumerate(texture_names, 1):
            stream.write(
                f"newmtl TextureBlock_{block_index:04d}\n"
                "Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
                f"map_Kd {texture_name}\n\n"
            )


def _write_multi_gltf(
    folder: Path,
    blocks: list[dict[str, Any]],
    texture_paths: list[Path],
) -> tuple[Path, Path]:
    scene = trimesh.Scene()
    for block_index, (block, texture_path) in enumerate(
        zip(blocks, texture_paths, strict=True),
        1,
    ):
        vertices, faces, uv, normals = _mesh_arrays(block["mesh"])
        with Image.open(texture_path) as source_image:
            image = source_image.convert("RGB").copy()
        material = trimesh.visual.material.PBRMaterial(
            name=f"TextureBlock_{block_index:04d}",
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            visual=visual,
            process=False,
            validate=False,
        )
        scene.add_geometry(
            mesh,
            node_name=f"TextureBlock_{block_index:04d}",
            geom_name=f"TextureBlock_{block_index:04d}",
        )
    glb_path = folder / "model.glb"
    glb_path.write_bytes(
        trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    )
    gltf_path = folder / "model.gltf"
    payload = trimesh.exchange.gltf.export_gltf(
        scene,
        include_normals=True,
        merge_buffers=True,
        embed_buffers=True,
    )
    for name, data in payload.items():
        destination = gltf_path if name.lower().endswith(".gltf") else folder / name
        destination.write_bytes(data)
    if not gltf_path.is_file():
        raise RuntimeError("多图集glTF导出未生成主文件")
    return gltf_path, glb_path


def export_textured_mesh_blocks(
    blocks: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    formats: Iterable[str],
    osgconv_path: str | None = None,
) -> dict[str, str]:
    """Export a spatially blocked model while retaining multiple atlases."""

    block_list = [dict(block) for block in blocks]
    if not block_list:
        raise ValueError("多图集模型没有纹理块")
    if len(block_list) == 1:
        return export_textured_mesh(
            block_list[0]["mesh"],
            block_list[0]["texture"],
            output_dir,
            formats=formats,
            osgconv_path=osgconv_path,
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = {str(value).strip().lower() for value in formats}
    unsupported = requested - {"obj", "fbx", "gltf", "glb", "osgb"}
    if unsupported:
        raise ValueError(f"未知模型格式：{', '.join(sorted(unsupported))}")

    texture_paths: list[Path] = []
    result: dict[str, str] = {}
    for block_index, block in enumerate(block_list, 1):
        destination = output / f"model_texture_{block_index:04d}.png"
        source = Path(str(block["texture"]))
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        texture_paths.append(destination)
        result[f"texture_atlas_{block_index:04d}"] = str(destination)
    texture_manifest = output / "texture_atlases.json"
    _atomic_json(
        texture_manifest,
        {
            "version": 1,
            "atlas_count": len(texture_paths),
            "atlases": [path.name for path in texture_paths],
        },
    )
    result["texture_atlas"] = str(texture_manifest)

    obj_path = output / "model.obj"
    if requested & {"obj", "osgb"}:
        _write_multi_obj(
            obj_path,
            block_list,
            [path.name for path in texture_paths],
        )
        result["obj"] = str(obj_path)
        result["mtl"] = str(obj_path.with_suffix(".mtl"))
    if "fbx" in requested:
        fbx_files: list[str] = []
        for block_index, (block, texture_path) in enumerate(
            zip(block_list, texture_paths, strict=True),
            1,
        ):
            vertices, faces, uv, normals = _mesh_arrays(block["mesh"])
            fbx_path = output / f"model_block_{block_index:04d}.fbx"
            _write_fbx_ascii(
                fbx_path,
                vertices,
                faces,
                uv,
                normals,
                texture_path.name,
            )
            fbx_files.append(fbx_path.name)
            result[f"fbx_block_{block_index:04d}"] = str(fbx_path)
        fbx_manifest = output / "model_fbx_blocks.json"
        _atomic_json(fbx_manifest, {"version": 1, "files": fbx_files})
        result["fbx"] = str(fbx_manifest)
    if requested & {"gltf", "glb"}:
        before_gltf = {path.resolve() for path in output.iterdir() if path.is_file()}
        gltf_path, glb_path = _write_multi_gltf(
            output,
            block_list,
            texture_paths,
        )
        result["gltf"] = str(gltf_path)
        result["glb"] = str(glb_path)
        extra_resources = [
            path
            for path in output.iterdir()
            if path.is_file()
            and path.resolve() not in before_gltf
            and path not in {gltf_path, glb_path}
        ]
        for index, path in enumerate(sorted(extra_resources), 1):
            result[f"gltf_resource_{index:04d}"] = str(path)
    if "osgb" in requested:
        osgb_path = output / "model.osgb"
        _convert_obj_to_osgb(obj_path, osgb_path, osgconv_path=osgconv_path)
        result["osgb"] = str(osgb_path)
    model_manifest = output / "model_blocks.json"
    _atomic_json(
        model_manifest,
        {
            "version": 1,
            "block_count": len(block_list),
            "formats": result,
        },
    )
    result["block_manifest"] = str(model_manifest)
    return result


def _fingerprint(paths: Iterable[Path], options: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        digest.update(str(resolved).casefold().encode("utf-8"))
        digest.update(struct.pack("<QQ", stat.st_size, stat.st_mtime_ns))
    digest.update(
        json.dumps(options, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return digest.hexdigest()[:12]


def run_model_pipeline(
    *,
    dense_workspace: str | Path,
    pointcloud: str | Path,
    output_root: str | Path | None = None,
    colmap_path: str | None = None,
    precision_mode: str = "标准工程模式",
    formats: Iterable[str] = ("obj", "fbx", "gltf"),
    osgconv_path: str | None = None,
    resume: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the restartable textured-model pipeline."""

    dense = Path(dense_workspace).expanduser().resolve()
    source = Path(pointcloud).expanduser().resolve()
    if not dense.is_dir() or not (dense / "images").is_dir() or not (dense / "sparse").is_dir():
        raise RuntimeError("缺少COLMAP去畸变照片或稀疏相机模型，无法投影原图纹理")
    if not source.is_file():
        raise FileNotFoundError(f"稠密点云不存在：{source}")
    executable = find_colmap(colmap_path)
    if not executable:
        raise RuntimeError("未找到COLMAP，无法执行表面重建和多照片纹理融合")
    preset_name = _normalized_preset(precision_mode)
    preset = dict(MODEL_PRESETS[preset_name])
    requested_formats = sorted({str(value).strip().lower() for value in formats})
    if not requested_formats:
        requested_formats = ["obj"]
    available_memory = int(psutil.virtual_memory().available)
    memory_safe_point_limit = max(250_000, available_memory // 450)
    preset["point_limit"] = min(
        int(preset["point_limit"]),
        int(memory_safe_point_limit),
    )
    options = {
        "model_pipeline_version": 1,
        "texture_pipeline_version": 2,
        "precision_mode": preset_name,
        "formats": requested_formats,
        **preset,
    }
    parent = Path(output_root).expanduser().resolve() if output_root else dense
    parent.mkdir(parents=True, exist_ok=True)
    source_point_count = ply_counts(source)[0]
    estimated_model_points = min(
        int(source_point_count),
        int(preset["point_limit"]),
    )
    estimated_model_faces = min(
        int(preset["target_faces"]),
        max(1, estimated_model_points * 4),
    )
    estimated_bytes = int(
        1024**3
        + estimated_model_points * 180
        + estimated_model_faces
        * (500 + (220 if "fbx" in requested_formats else 0))
    )
    disk_free = int(shutil.disk_usage(parent).free)
    reserve_bytes = max(1024**3, estimated_bytes // 4)
    if disk_free < estimated_bytes + reserve_bytes:
        raise RuntimeError(
            "模型生成磁盘空间不足："
            f"预计至少还需 {estimated_bytes / 1024**3:.1f} GB，"
            f"当前可用 {disk_free / 1024**3:.1f} GB。"
        )
    geometry_cache_options = {
        key: value
        for key, value in options.items()
        if key
        not in {
            "formats",
            "texture_pipeline_version",
            "texture_block_target_faces",
            "texture_atlas_max_dimension",
            "texture_atlas_max_pixels",
        }
    }
    cache_key = _fingerprint(
        [
            source,
            dense / "sparse" / "cameras.bin",
            dense / "sparse" / "images.bin",
        ],
        geometry_cache_options,
    )
    root = parent / f"model_{cache_key}"
    root.mkdir(parents=True, exist_ok=True)
    tracker = StageTracker(root / "pipeline_state.json")
    log_path = root / "model_pipeline.log"
    config_path = root / "model_config.json"
    config_path.write_text(json.dumps(options, ensure_ascii=False, indent=2), encoding="utf-8")

    conditioned = root / "01_conditioned_points.ply"
    raw_mesh = root / "02_surface_raw.ply"
    repaired_mesh = root / "03_mesh_repaired.ply"
    simplified_mesh = root / "04_mesh_simplified.ply"
    texture_partition_root = root / "05_texture_partitions"
    partition_manifest = texture_partition_root / "partition_manifest.json"
    textured_blocks_root = root / "05_textured_blocks"
    texture_manifest = textured_blocks_root / "texture_manifest.json"
    exports_dir = root / "06_exports"
    reports: dict[str, Any] = {}

    def stage(
        name: str,
        progress: float,
        message: str,
        valid: Callable[[], bool],
        action: Callable[[], Any],
    ) -> Any:
        if resume and tracker.status(name) == "completed" and valid():
            _notify(progress_callback, progress, f"恢复缓存：{message}")
            return dict(
                tracker.data.get("stages", {}).get(name, {}).get("details") or {}
            )
        tracker.set(name, "running", message=message)
        _notify(progress_callback, progress, message)
        try:
            value = action()
        except Exception as exc:
            tracker.set(name, "failed", message=str(exc))
            raise
        if not valid():
            error = f"{message}未生成有效成果"
            tracker.set(name, "failed", message=error)
            raise RuntimeError(error)
        details = value if isinstance(value, dict) else {}
        tracker.set(name, "completed", details=details)
        return value

    reports["conditioning"] = stage(
        "point_conditioning",
        0.04,
        "点云去噪、抽稀与法向修复",
        lambda: conditioned.is_file() and ply_counts(conditioned)[0] >= 3,
        lambda: condition_point_cloud(
            source,
            conditioned,
            point_limit=int(preset["point_limit"]),
            outlier_sample_size=int(preset["outlier_sample_size"]),
            outlier_std_ratio=float(preset["outlier_std_ratio"]),
        ),
    )

    stage(
        "surface_reconstruction",
        0.2,
        "泊松表面重建与三角网格生成",
        lambda: raw_mesh.is_file() and ply_counts(raw_mesh)[1] > 0,
        lambda: _run(
            executable,
            [
                "poisson_mesher",
                "--input_path",
                str(conditioned),
                "--output_path",
                str(raw_mesh),
                "--PoissonMeshing.depth",
                str(int(preset["poisson_depth"])),
                "--PoissonMeshing.trim",
                str(float(preset["poisson_trim"])),
            ],
            log_path,
        ),
    )

    reports["repair"] = stage(
        "mesh_repair",
        0.4,
        "清理退化面、修补小孔并统一法向",
        lambda: repaired_mesh.is_file() and ply_counts(repaired_mesh)[1] > 0,
        lambda: repair_mesh(
            raw_mesh,
            repaired_mesh,
            hole_size_ratio=float(preset["hole_size_ratio"]),
        ),
    )

    def simplify() -> dict[str, Any]:
        input_vertices, input_faces = ply_counts(repaired_mesh)
        target_faces = int(preset["target_faces"])
        if input_faces <= target_faces:
            if simplified_mesh.exists():
                simplified_mesh.unlink()
            try:
                os.link(repaired_mesh, simplified_mesh)
            except OSError:
                shutil.copy2(repaired_mesh, simplified_mesh)
        else:
            ratio = max(1e-4, min(1.0, target_faces / input_faces))
            _run(
                executable,
                [
                    "mesh_simplifier",
                    "--input_path",
                    str(repaired_mesh),
                    "--output_path",
                    str(simplified_mesh),
                    "--MeshSimplification.target_face_ratio",
                    f"{ratio:.9g}",
                ],
                log_path,
            )
        output_vertices, output_faces = ply_counts(simplified_mesh)
        return {
            "input_vertices": input_vertices,
            "input_faces": input_faces,
            "target_faces": target_faces,
            "output_vertices": output_vertices,
            "output_faces": output_faces,
        }

    reports["simplification"] = stage(
        "mesh_simplification",
        0.55,
        "网格简化并保留边界",
        lambda: simplified_mesh.is_file() and ply_counts(simplified_mesh)[1] > 0,
        simplify,
    )

    partition = stage(
        "texture_partition",
        0.61,
        "按空间划分网格纹理块",
        lambda: _partition_manifest_valid(partition_manifest),
        lambda: partition_mesh_for_texturing(
            simplified_mesh,
            texture_partition_root,
            target_faces=int(preset["texture_block_target_faces"]),
        ),
    )
    if not partition or not partition.get("blocks"):
        partition = json.loads(partition_manifest.read_text(encoding="utf-8"))
    reports["texture_partition"] = partition

    texture_report = stage(
        "texture_mapping",
        0.67,
        "分块UV展开、原始照片投影与多图集融合",
        lambda: _texture_manifest_valid(texture_manifest),
        lambda: texture_mesh_blocks(
            executable=executable,
            dense_workspace=dense,
            partition=partition,
            output_root=textured_blocks_root,
            texture_scale_factor=float(preset["texture_scale_factor"]),
            atlas_max_dimension=int(preset["texture_atlas_max_dimension"]),
            atlas_max_pixels=int(preset["texture_atlas_max_pixels"]),
            progress_callback=progress_callback,
        ),
    )
    if not texture_report or not texture_report.get("blocks"):
        texture_report = json.loads(texture_manifest.read_text(encoding="utf-8"))
        texture_report["manifest"] = str(texture_manifest)
    reports["texture_mapping"] = texture_report

    def write_exports() -> dict[str, str]:
        exported = export_textured_mesh_blocks(
            texture_report["blocks"],
            exports_dir,
            formats=requested_formats,
            osgconv_path=osgconv_path,
        )
        (exports_dir / "model_manifest.json").write_text(
            json.dumps(exported, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return exported

    export_result = stage(
        "model_export",
        0.9,
        "写出OBJ、FBX、glTF与OSGB模型",
        lambda: _model_exports_valid(
            exports_dir / "model_manifest.json",
            requested_formats,
        ),
        write_exports,
    )
    if not export_result:
        export_result = json.loads((exports_dir / "model_manifest.json").read_text(encoding="utf-8"))

    texture_blocks = [dict(block) for block in texture_report["blocks"]]
    first_block = texture_blocks[0]
    vertices = int(texture_report.get("vertex_count", 0))
    faces = int(texture_report.get("face_count", 0))
    result = {
        "folder": str(root),
        "dense_workspace": str(dense),
        "source_pointcloud": str(source),
        "conditioned_pointcloud": str(conditioned),
        "raw_mesh": str(raw_mesh),
        "repaired_mesh": str(repaired_mesh),
        "simplified_mesh": str(simplified_mesh),
        "textured_mesh": str(first_block["mesh"]),
        "texture_atlas": str(first_block["texture"]),
        "texture_manifest": str(texture_manifest),
        "texture_blocks": texture_blocks,
        "texture_block_count": len(texture_blocks),
        "texture_strategy": "spatial_mesh_blocks_multi_atlas",
        "vertex_count": int(vertices),
        "face_count": int(faces),
        "precision_mode": preset_name,
        "formats": dict(export_result),
        "reports": reports,
        "resources": {
            "available_memory_gb_at_start": round(available_memory / 1024**3, 2),
            "disk_free_gb_at_start": round(disk_free / 1024**3, 2),
            "estimated_additional_disk_gb": round(estimated_bytes / 1024**3, 2),
            "effective_point_limit": int(preset["point_limit"]),
            "source_point_count": int(source_point_count),
        },
        "pipeline_state": str(tracker.path),
        "log": str(log_path),
    }
    (root / "model_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _notify(progress_callback, 1.0, "纹理三维模型生成完成")
    return result
