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

_PREVIEW_SEED = np.uint64(0x6A09E667F3BCC909)


def _mix_uint64(values: np.ndarray | np.uint64) -> np.ndarray:
    """Return a stable SplitMix64 hash without touching global RNG state."""

    data = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        data = data + np.uint64(0x9E3779B97F4A7C15)
        data = (data ^ (data >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        data = (data ^ (data >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return data ^ (data >> np.uint64(31))


def _distributed_preview_indices(
    total: int,
    target: int,
    *,
    block_count: int,
) -> np.ndarray:
    """Sample the whole vertex stream without periodic or long-run aliasing.

    COLMAP writes fused vertices in image/depth-map batches. Reading a few long
    contiguous ranges therefore turns the preview into visible scan strips. We
    instead divide the stream into many strata, randomly position a short
    window inside every stratum and jitter the samples inside that window. This
    keeps disk reads reasonably local while removing the directional pattern.
    """

    total = max(0, int(total))
    target = min(total, max(0, int(target)))
    if target <= 0:
        return np.empty(0, dtype=np.int64)
    if target == total:
        return np.arange(total, dtype=np.int64)

    # No sampled run contains more than roughly 64 vertices. A million-point
    # preview consequently uses at least ~15k independently positioned windows
    # instead of the old 128 long scan strips.
    groups = min(
        target,
        max(max(1, int(block_count)), (target + 63) // 64),
    )
    group_ids = np.arange(groups, dtype=np.int64)
    starts = (group_ids * total) // groups
    ends = ((group_ids + 1) * total) // groups
    widths = ends - starts

    takes = np.full(groups, target // groups, dtype=np.int64)
    takes[: target % groups] += 1
    # Read from a window wider than the requested sample so adjacent selected
    # records do not form another miniature scan line.
    windows = np.minimum(widths, np.maximum(takes, takes * 8))
    free_space = widths - windows
    group_hashes = _mix_uint64(group_ids.astype(np.uint64) + _PREVIEW_SEED)
    window_offsets = (
        group_hashes % (free_space.astype(np.uint64) + np.uint64(1))
    ).astype(np.int64)
    window_starts = starts + window_offsets

    expanded_groups = np.repeat(group_ids, takes)
    take_starts = np.cumsum(takes) - takes
    local = np.arange(target, dtype=np.int64) - np.repeat(take_starts, takes)
    expanded_takes = takes[expanded_groups]
    expanded_windows = windows[expanded_groups]
    bin_starts = (local * expanded_windows) // expanded_takes
    bin_ends = ((local + 1) * expanded_windows) // expanded_takes
    bin_widths = np.maximum(1, bin_ends - bin_starts)
    sample_hashes = _mix_uint64(
        np.arange(target, dtype=np.uint64)
        + _PREVIEW_SEED
        + np.uint64(0x517CC1B727220A95)
    )
    jitter = (sample_hashes % bin_widths.astype(np.uint64)).astype(np.int64)
    return window_starts[expanded_groups] + bin_starts + jitter


def _spatially_balanced_subset(
    vertices: np.ndarray,
    source_indices: np.ndarray,
    target: int,
) -> np.ndarray:
    """Prefer different spatial cells, then fill remaining preview capacity."""

    count = len(vertices)
    target = min(count, max(0, int(target)))
    if target >= count:
        return np.arange(count, dtype=np.int64)
    finite = np.isfinite(vertices).all(axis=1)
    valid = np.flatnonzero(finite)
    if len(valid) <= target:
        return valid

    points = vertices[valid]
    # Robust bounds prevent a handful of flying points from collapsing the
    # useful scene into only a few spatial cells.
    probe_limit = 100_000
    if len(points) > probe_limit:
        probe_rows = np.linspace(0, len(points) - 1, probe_limit, dtype=np.int64)
        bounds_probe = points[probe_rows]
    else:
        bounds_probe = points
    lower = np.percentile(bounds_probe, 0.5, axis=0)
    upper = np.percentile(bounds_probe, 99.5, axis=0)
    span = upper - lower
    span[span <= np.finfo(np.float64).eps] = 1.0

    resolution = np.uint64(2048)
    normalized = np.clip((points - lower) / span, 0.0, 1.0)
    cells = np.floor(normalized * float(resolution - np.uint64(1))).astype(
        np.uint64
    )
    keys = cells[:, 0] + resolution * (
        cells[:, 1] + resolution * cells[:, 2]
    )
    unique_keys, first = np.unique(keys, return_index=True)
    representatives = valid[first]

    if len(representatives) >= target:
        scores = _mix_uint64(unique_keys + _PREVIEW_SEED)
        chosen = np.argpartition(scores, target - 1)[:target]
        return np.sort(representatives[chosen])

    needed = target - len(representatives)
    represented = np.zeros(count, dtype=bool)
    represented[representatives] = True
    remaining = valid[~represented[valid]]
    scores = _mix_uint64(
        source_indices[remaining].astype(np.uint64) + _PREVIEW_SEED
    )
    extra = remaining[np.argpartition(scores, needed - 1)[:needed]]
    return np.sort(np.concatenate((representatives, extra)))


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
    """Read a stable, direction-unbiased PLY preview with bounded memory use."""

    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"点云不存在：{value}")
    layout = _binary_ply_layout(value)
    if layout is None:
        vertices, colors = load_ply_vertices_colors(value)
        total = len(vertices)
        if total > max_points:
            target = max(1, int(max_points))
            candidate_count = min(total, target + min(target, 500_000))
            source_indices = _distributed_preview_indices(
                total,
                candidate_count,
                block_count=block_count,
            )
            candidate_vertices = vertices[source_indices]
            candidate_colors = colors[source_indices]
            selected = _spatially_balanced_subset(
                candidate_vertices,
                source_indices,
                target,
            )
            vertices = candidate_vertices[selected]
            colors = candidate_colors[selected]
        return vertices, colors, total

    offset, total, dtype = layout
    names = set(dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise RuntimeError(f"PLY缺少XYZ坐标：{value}")
    target = min(total, max(1, int(max_points)))
    candidate_count = min(total, target + min(target, 500_000))
    source_indices = _distributed_preview_indices(
        total,
        candidate_count,
        block_count=block_count,
    )
    mapped = np.memmap(
        value,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=(total,),
    )
    records = np.asarray(mapped[source_indices])
    del mapped
    if not len(records):
        raise RuntimeError(f"PLY中没有可显示的点：{value}")
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
    if len(vertices) > target:
        selected = _spatially_balanced_subset(
            vertices,
            source_indices,
            target,
        )
        vertices = vertices[selected]
        colors = colors[selected]
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
