"""Point-cloud I/O shared by photogrammetry, filtering, viewing, and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_SCALAR_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _binary_ply_layout(path: Path) -> tuple[int, int, np.dtype] | None:
    """Return data offset, vertex count and scalar dtype for binary PLY."""

    with path.open("rb") as stream:
        header = bytearray()
        while b"end_header" not in header and len(header) < 1024 * 1024:
            chunk = stream.read(4096)
            if not chunk:
                return None
            header.extend(chunk)
    marker = header.find(b"end_header")
    if marker < 0:
        return None
    newline = header.find(b"\n", marker)
    if newline < 0:
        return None
    header_end = newline + 1
    lines = header[:header_end].decode("ascii", errors="strict").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        return None
    vertex_count = 0
    in_vertices = False
    properties: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3:
            vertex_count = int(parts[2])
            in_vertices = True
            continue
        if parts and parts[0] == "element":
            in_vertices = False
            continue
        if in_vertices and parts[:1] == ["property"]:
            if len(parts) != 3 or parts[1] not in _PLY_SCALAR_TYPES:
                return None
            properties.append((parts[2], _PLY_SCALAR_TYPES[parts[1]]))
    if vertex_count <= 0 or not properties:
        return None
    return header_end, vertex_count, np.dtype(properties)


def load_ply_preview(
    path: str | Path,
    *,
    max_points: int = 1_000_000,
    block_count: int = 128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read a distributed PLY preview without loading a huge cloud in RAM."""

    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"点云不存在：{value}")
    layout = _binary_ply_layout(value)
    if layout is None:
        vertices, colors = load_ply_vertices_colors(value)
        total = len(vertices)
        if total > max_points:
            indices = np.linspace(0, total - 1, max_points, dtype=np.int64)
            vertices = vertices[indices]
            colors = colors[indices]
        return vertices, colors, total

    offset, total, dtype = layout
    names = set(dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise RuntimeError(f"PLY缺少XYZ坐标：{value}")
    target = min(total, max(1, int(max_points)))
    if total <= target:
        ranges = [(0, total)]
    else:
        blocks = min(max(1, int(block_count)), target)
        per_block = max(1, target // blocks)
        starts = np.linspace(
            0,
            max(0, total - per_block),
            blocks,
            dtype=np.int64,
        )
        ranges = [(int(start), per_block) for start in np.unique(starts)]

    chunks: list[np.ndarray] = []
    with value.open("rb") as stream:
        for start, count in ranges:
            stream.seek(offset + start * dtype.itemsize)
            chunk = np.fromfile(stream, dtype=dtype, count=count)
            if len(chunk):
                chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"PLY中没有可显示的点：{value}")
    records = np.concatenate(chunks)
    vertices = np.column_stack(
        (records["x"], records["y"], records["z"])
    ).astype(np.float64, copy=False)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack(
            (records["red"], records["green"], records["blue"])
        )
        colors = np.clip(colors, 0, 255).astype(np.uint8, copy=False)
    else:
        colors = np.full((len(vertices), 3), 185, dtype=np.uint8)
    return vertices, colors, total


def load_ply_vertices_colors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLY as float64 XYZ and uint8 RGB without altering geometry."""

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("读取摄影测量点云需要 trimesh") from exc
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"点云不存在：{value}")
    geometry = trimesh.load(value, process=False)
    if not hasattr(geometry, "vertices"):
        raise RuntimeError(f"点云没有任何顶点：{value}")
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if not len(vertices):
        raise RuntimeError(f"点云为空：{value}")
    raw_colors = getattr(geometry, "colors", None)
    if raw_colors is None and hasattr(geometry, "visual"):
        raw_colors = getattr(geometry.visual, "vertex_colors", None)
    if raw_colors is None or len(raw_colors) != len(vertices):
        colors = np.full((len(vertices), 3), 180, dtype=np.uint8)
    else:
        colors = np.asarray(raw_colors)[:, :3]
        if np.issubdtype(colors.dtype, np.floating) and colors.max(initial=0) <= 1:
            colors = colors * 255
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return vertices, colors
