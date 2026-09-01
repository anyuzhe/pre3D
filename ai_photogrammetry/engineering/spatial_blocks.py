"""Plan and materialize Core/Halo dense-MVS blocks from global SfM geometry."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .mvs_selection import SparseView, read_sparse_views

_FUSED_PROPERTIES = (
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
_FUSED_DTYPE = np.dtype(
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
    ],
    align=False,
)

SPATIAL_BLOCK_PLAN_VERSION = 3
_DENSE_CORE_MARGIN_RATIO = 0.10
_MINIMUM_CORE_REFERENCE_OBSERVATIONS = 2
_TARGET_CORE_REFERENCE_COVERAGE = 0.95


@dataclass(frozen=True)
class _Leaf:
    indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_sparse_points(points3d_txt: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return POINT3D_ID and XYZ arrays from a COLMAP text model."""

    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    for line in Path(points3d_txt).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        parts = clean.split()
        if len(parts) < 8:
            continue
        try:
            point_id = int(parts[0])
            point = tuple(float(value) for value in parts[1:4])
        except ValueError:
            continue
        if np.isfinite(point).all():
            ids.append(point_id)
            xyz.append(point)  # type: ignore[arg-type]
    if len(ids) < 3:
        raise RuntimeError("稀疏模型有效三维点少于3个，无法规划空间分块")
    return np.asarray(ids, dtype=np.int64), np.asarray(xyz, dtype=np.float64)


def _ranked_image_names_for_points(
    point_ids: np.ndarray,
    point_to_images: dict[int, set[str]],
    view_point_counts: dict[str, int],
    *,
    limit: int | None = None,
) -> list[str]:
    """Rank genuinely observing images while ignoring one-point bridges."""

    counts: dict[str, int] = {}
    for point_id in point_ids:
        for image_name in point_to_images.get(int(point_id), ()):
            counts[image_name] = counts.get(image_name, 0) + 1
    ranked = sorted(counts, key=lambda name: (-counts[name], name.casefold()))
    selected = [
        name
        for name in ranked
        if counts[name]
        >= max(3, min(25, int(np.ceil(view_point_counts.get(name, 0) * 0.01))))
    ]
    if len(selected) < min(3, len(ranked)):
        selected = ranked[:3]
    if limit is not None:
        selected = selected[: max(3, int(limit))]
    return selected


def _augment_reference_coverage(
    reference_names: Sequence[str],
    candidate_names: Sequence[str],
    core_point_ids: set[int],
    views_by_name: dict[str, SparseView],
    *,
    minimum_observations: int = _MINIMUM_CORE_REFERENCE_OBSERVATIONS,
    target_coverage: float = _TARGET_CORE_REFERENCE_COVERAGE,
) -> tuple[list[str], float, float, int]:
    """Add local references until most Core points have multi-view support.

    A globally unique depth-map owner is economical, but it is not sufficient
    for spatial MVS: an image can observe several neighbouring Core blocks and
    StereoFusion needs at least two reference depth maps in the block that owns
    a surface.  This greedy coverage pass promotes only the useful Halo views
    and therefore avoids falling back to recomputing every image in every
    block.
    """

    if minimum_observations < 1:
        raise ValueError("Core参考视图最少观测数必须大于0")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("Core参考视图覆盖率必须在0～1之间")
    if not core_point_ids:
        return list(dict.fromkeys(reference_names)), 0.0, 1.0, 0

    ordered_points = sorted(core_point_ids)
    point_positions = {
        point_id: index for index, point_id in enumerate(ordered_points)
    }
    selected = list(dict.fromkeys(reference_names))
    selected_keys = {name.casefold() for name in selected}
    observations = np.zeros(len(ordered_points), dtype=np.int16)
    candidate_indices: dict[str, np.ndarray] = {}

    for name in dict.fromkeys([*selected, *candidate_names]):
        view = views_by_name.get(name.casefold())
        if view is None or not view.point_ids:
            continue
        indices = np.fromiter(
            (
                point_positions[point_id]
                for point_id in view.point_ids
                if point_id in point_positions
            ),
            dtype=np.int64,
        )
        if len(indices):
            candidate_indices[name] = indices
            if name.casefold() in selected_keys:
                observations[indices] += 1

    def coverage_ratio() -> float:
        return float(np.mean(observations >= minimum_observations))

    base_count = len(selected)
    candidates = sorted(
        (
            name
            for name in candidate_names
            if name.casefold() not in selected_keys and name in candidate_indices
        ),
        key=str.casefold,
    )
    while candidates and coverage_ratio() < target_coverage:
        best_name = ""
        best_score = (-1, -1, -1)
        for name in candidates:
            indices = candidate_indices[name]
            score = (
                int(np.count_nonzero(observations[indices] == minimum_observations - 1)),
                int(np.count_nonzero(observations[indices] < minimum_observations)),
                int(len(indices)),
            )
            if score > best_score:
                best_name = name
                best_score = score
        if not best_name or best_score[1] <= 0:
            break
        observations[candidate_indices[best_name]] += 1
        selected.append(best_name)
        selected_keys.add(best_name.casefold())
        candidates.remove(best_name)

    return (
        selected,
        coverage_ratio(),
        float(np.mean(observations == 0)),
        len(selected) - base_count,
    )


def plan_spatial_blocks(
    images_txt: str | Path,
    points3d_txt: str | Path,
    *,
    target_images: int = 120,
    halo_ratio: float = 0.20,
    min_core_points: int = 500,
    max_blocks: int = 64,
) -> dict[str, Any]:
    """Build adaptive PCA-aligned Core/Halo blocks from a global sparse model.

    Long corridor-like scenes are split along their principal axis.  Surface
    scenes are recursively split in two PCA plane dimensions.  A photo is
    assigned from actual sparse-point visibility, never from filename order.
    """

    if target_images < 8:
        raise ValueError("每个空间块的目标照片数至少为8")
    if not 0.0 <= halo_ratio <= 1.0:
        raise ValueError("空间块Halo比例必须在0～1之间")
    if min_core_points < 3 or max_blocks < 1:
        raise ValueError("空间分块点数或最大块数设置无效")

    views = read_sparse_views(images_txt)
    point_ids, xyz = read_sparse_points(points3d_txt)
    point_lookup = {int(value): index for index, value in enumerate(point_ids)}
    point_to_images: dict[int, set[str]] = {}
    view_point_counts = {view.name: len(view.point_ids) for view in views}
    for view in views:
        for point_id in view.point_ids:
            if point_id in point_lookup:
                point_to_images.setdefault(point_id, set()).add(view.name)

    visible_mask = np.asarray(
        [int(point_id) in point_to_images for point_id in point_ids],
        dtype=bool,
    )
    point_ids = point_ids[visible_mask]
    xyz = xyz[visible_mask]
    if len(point_ids) < 3:
        raise RuntimeError("稀疏点没有有效照片观测，无法规划空间分块")

    center = np.median(xyz, axis=0)
    covariance = np.cov(xyz - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    local = (xyz - center) @ basis
    global_lower = np.min(local, axis=0)
    global_upper = np.max(local, axis=0)
    extent = np.maximum(global_upper - global_lower, 1e-9)
    corridor = bool(eigenvalues[0] > max(eigenvalues[1], 1e-12) * 4.0)
    split_dimensions = 1 if corridor else 2

    # A strict per-leaf visibility threshold can over-segment scenes that have
    # long tracks or common background points.  Use the theoretical reference
    # workload as the block-count ceiling; Halo source views do not own depth
    # maps and therefore must not force extra blocks.
    workload_blocks = max(1, int(np.ceil(len(views) / target_images)))
    effective_max_blocks = min(int(max_blocks), workload_blocks)
    leaves: list[_Leaf] = [
        _Leaf(
            indices=np.arange(len(point_ids), dtype=np.int64),
            lower=global_lower.copy(),
            upper=global_upper.copy(),
        )
    ]

    def split_leaf(leaf: _Leaf) -> tuple[_Leaf, _Leaf] | None:
        if len(leaf.indices) < 2 * min_core_points:
            return None
        lower = leaf.lower
        upper = leaf.upper
        spans = upper[:split_dimensions] - lower[:split_dimensions]
        axis = int(np.argmax(spans / extent[:split_dimensions]))
        values = local[leaf.indices, axis]
        split = float(np.median(values))
        left_mask = values <= split
        right_mask = ~left_mask
        if int(left_mask.sum()) < min_core_points or int(right_mask.sum()) < min_core_points:
            return None
        left_indices = leaf.indices[left_mask]
        right_indices = leaf.indices[right_mask]
        left_names = _ranked_image_names_for_points(
            point_ids[left_indices], point_to_images, view_point_counts
        )
        right_names = _ranked_image_names_for_points(
            point_ids[right_indices], point_to_images, view_point_counts
        )
        if len(left_names) < 3 or len(right_names) < 3:
            return None
        left_upper = upper.copy()
        left_upper[axis] = split
        right_lower = lower.copy()
        right_lower[axis] = split
        return (
            _Leaf(left_indices, lower.copy(), left_upper),
            _Leaf(right_indices, right_lower, upper.copy()),
        )

    while len(leaves) < effective_max_blocks:
        candidates = sorted(
            range(len(leaves)),
            key=lambda index: (
                len(
                    _ranked_image_names_for_points(
                        point_ids[leaves[index].indices],
                        point_to_images,
                        view_point_counts,
                    )
                ),
                len(leaves[index].indices),
            ),
            reverse=True,
        )
        split_result: tuple[_Leaf, _Leaf] | None = None
        selected_index = -1
        for candidate_index in candidates:
            names = _ranked_image_names_for_points(
                point_ids[leaves[candidate_index].indices],
                point_to_images,
                view_point_counts,
            )
            if len(names) <= target_images:
                continue
            split_result = split_leaf(leaves[candidate_index])
            if split_result is not None:
                selected_index = candidate_index
                break
        if split_result is None or selected_index < 0:
            break
        leaves[selected_index : selected_index + 1] = list(split_result)

    blocks: list[dict[str, Any]] = []
    assigned_images: set[str] = set()
    block_core_points: list[set[int]] = []
    block_halo_points: list[set[int]] = []
    for block_index, leaf in enumerate(leaves, start=1):
        core_lower = leaf.lower.copy()
        core_upper = leaf.upper.copy()
        # Only the PCA split dimensions partition ownership.  Dense geometry
        # regularly extends beyond the sparse model's thickness/height bounds,
        # so those non-partition dimensions must include the same safety margin
        # that is used while selecting Halo views.  Otherwise valid rock faces
        # are calculated and then silently clipped from every Core output.
        core_lower[split_dimensions:] = (
            global_lower[split_dimensions:]
            - extent[split_dimensions:] * _DENSE_CORE_MARGIN_RATIO
        )
        core_upper[split_dimensions:] = (
            global_upper[split_dimensions:]
            + extent[split_dimensions:] * _DENSE_CORE_MARGIN_RATIO
        )
        halo_lower = core_lower.copy()
        halo_upper = core_upper.copy()
        block_extent = np.maximum(core_upper - core_lower, extent * 1e-6)
        halo_lower[:split_dimensions] -= block_extent[:split_dimensions] * halo_ratio
        halo_upper[:split_dimensions] += block_extent[:split_dimensions] * halo_ratio
        halo_mask = np.all(
            (local >= halo_lower - 1e-9) & (local <= halo_upper + 1e-9),
            axis=1,
        )
        halo_indices = np.flatnonzero(halo_mask)
        image_names = _ranked_image_names_for_points(
            point_ids[halo_indices],
            point_to_images,
            view_point_counts,
            limit=max(target_images + 20, int(np.ceil(target_images * 1.5))),
        )
        if len(image_names) < 3:
            image_names = _ranked_image_names_for_points(
                point_ids[leaf.indices],
                point_to_images,
                view_point_counts,
            )
        if len(image_names) < 3:
            continue
        assigned_images.update(image_names)
        block_core_points.append({int(value) for value in point_ids[leaf.indices]})
        block_halo_points.append({int(value) for value in point_ids[halo_indices]})
        blocks.append(
            {
                "id": f"block_{block_index:04d}",
                "core_lower": core_lower.tolist(),
                "core_upper": core_upper.tolist(),
                "halo_lower": halo_lower.tolist(),
                "halo_upper": halo_upper.tolist(),
                "core_point_count": int(len(leaf.indices)),
                "halo_point_count": int(len(halo_indices)),
                "image_count": int(len(image_names)),
                "image_names": image_names,
            }
        )

    if not blocks:
        raise RuntimeError("自动空间分块没有产生可运行的MVS工作块")
    registered_names = {view.name for view in views}

    def nearest_block(view: SparseView, loads: list[int]) -> int:
        camera_center = np.asarray(view.camera_center, dtype=np.float64)
        if np.isfinite(camera_center).all():
            camera_local = (camera_center - center) @ basis

            def placement_cost(index: int) -> float:
                block = blocks[index]
                lower = np.asarray(block["core_lower"], dtype=np.float64)
                upper = np.asarray(block["core_upper"], dtype=np.float64)
                clipped = np.clip(camera_local, lower, upper)
                distance = float(
                    np.linalg.norm(
                        (camera_local[:split_dimensions] - clipped[:split_dimensions])
                        / extent[:split_dimensions]
                    )
                )
                load = loads[index] / max(target_images, 1)
                return distance + 0.20 * load

            return min(range(len(blocks)), key=placement_cost)
        return min(range(len(blocks)), key=lambda index: loads[index])

    # Each registered photo owns one and only one depth map globally. Halo
    # photos remain available as source views in neighbouring blocks but are
    # not recomputed there. This preserves the user's 100% reference coverage
    # without multiplying PatchMatch work by the number of overlapping blocks.
    reference_lists: list[list[str]] = [[] for _ in blocks]
    loads = [0] * len(blocks)
    scored_views = [
        (
            view,
            [len(view.point_ids & core_points) for core_points in block_core_points],
        )
        for view in views
    ]
    # Assign constrained views first, then balance flexible overlap views. The
    # coverage term keeps each owner geometrically relevant; the load term
    # prevents one central block from receiving most of the depth maps.
    scored_views.sort(
        key=lambda item: (
            sum(score > 0 for score in item[1]),
            -max(item[1], default=0),
            item[0].name.casefold(),
        )
    )
    ideal_load = max(1.0, len(views) / len(blocks))
    for view, core_scores in scored_views:
        maximum_score = max(core_scores, default=0)
        if maximum_score > 0:
            candidates = [
                index for index, score in enumerate(core_scores) if score > 0
            ]
            best_index = min(
                candidates,
                key=lambda index: (
                    1.0 - core_scores[index] / maximum_score
                    + loads[index] / ideal_load,
                    loads[index],
                    index,
                ),
            )
        else:
            best_index = nearest_block(view, loads)
        reference_lists[best_index].append(view.name)
        loads[best_index] += 1

    views_by_name = {view.name.casefold(): view for view in views}
    achieved_coverages: list[float] = []
    for index, references in enumerate(reference_lists):
        if len(references) < 3:
            raise RuntimeError(
                f"空间块 {index + 1} 仅分配到 {len(references)} 张参考照片；"
                "请提高每块目标照片数或Halo比例"
            )
        local_images = set(str(value) for value in blocks[index]["image_names"])
        local_images.update(references)
        (
            references,
            achieved_coverage,
            zero_coverage,
            augmentation_count,
        ) = _augment_reference_coverage(
            references,
            sorted(local_images, key=str.casefold),
            block_core_points[index],
            views_by_name,
        )
        achieved_coverages.append(achieved_coverage)
        blocks[index]["image_names"] = sorted(local_images, key=str.casefold)
        blocks[index]["image_count"] = len(local_images)
        blocks[index]["reference_images"] = sorted(references, key=str.casefold)
        blocks[index]["reference_image_count"] = len(references)
        blocks[index]["base_reference_image_count"] = (
            len(references) - augmentation_count
        )
        blocks[index]["reference_augmentation_count"] = augmentation_count
        blocks[index]["core_reference_coverage_ratio"] = achieved_coverage
        blocks[index]["core_reference_zero_ratio"] = zero_coverage
    assigned_images = set().union(
        *(set(str(value) for value in block["image_names"]) for block in blocks)
    )
    payload = {
        "version": SPATIAL_BLOCK_PLAN_VERSION,
        "strategy": "pca_corridor" if corridor else "pca_adaptive_surface",
        "coordinate_frame": {
            "center": center.tolist(),
            "basis": basis.tolist(),
            "eigenvalues": eigenvalues.tolist(),
        },
        "target_images": int(target_images),
        "maximum_planned_blocks": int(effective_max_blocks),
        "halo_ratio": float(halo_ratio),
        "registered_image_count": int(len(views)),
        "assigned_image_count": int(len(assigned_images)),
        "unassigned_images": sorted(registered_names - assigned_images, key=str.casefold),
        "sparse_point_count": int(len(point_ids)),
        "block_count": int(len(blocks)),
        "total_block_image_assignments": int(sum(block["image_count"] for block in blocks)),
        "total_reference_image_assignments": int(
            sum(block["reference_image_count"] for block in blocks)
        ),
        "unique_reference_image_count": int(
            len(
                set().union(
                    *(
                        set(str(value) for value in block["reference_images"])
                        for block in blocks
                    )
                )
            )
        ),
        "minimum_core_reference_observations": (
            _MINIMUM_CORE_REFERENCE_OBSERVATIONS
        ),
        "target_core_reference_coverage_ratio": (
            _TARGET_CORE_REFERENCE_COVERAGE
        ),
        "minimum_achieved_core_reference_coverage_ratio": (
            min(achieved_coverages) if achieved_coverages else 0.0
        ),
        "blocks": blocks,
    }
    return payload


def save_spatial_plan(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_json(Path(path), payload)


def write_block_patch_match_config(
    path: str | Path,
    reference_image_names: Sequence[str],
    views: Sequence[SparseView],
    *,
    source_count: int,
    source_image_names: Sequence[str] | None = None,
    stable_source_image_names: Sequence[str] | None = None,
    minimum_stable_sources: int = 3,
) -> int:
    """Write explicit Core-reference/Halo-source lists for one spatial block.

    Reference views own depth maps in this block. The wider source pool may
    include neighbouring Halo views, but those views are not promoted to
    references. ``stable_source_image_names`` identifies views whose
    photometric maps are guaranteed to exist during geometric consistency.
    Keeping a few of those sources in every row prevents a reference from
    losing all sources after completed blocks have released their dense maps.
    """

    view_lookup = {view.name.casefold(): view for view in views}
    reference_keys = list(
        dict.fromkeys(name.casefold() for name in reference_image_names)
    )
    source_names = source_image_names or reference_image_names
    source_keys = list(dict.fromkeys(name.casefold() for name in source_names))
    stable_names = stable_source_image_names or reference_image_names
    stable_keys = {
        name.casefold()
        for name in stable_names
        if name.casefold() in view_lookup
    }
    references = [
        view_lookup[key]
        for key in reference_keys
        if key in view_lookup and view_lookup[key].point_ids
    ]
    source_views = [view_lookup[key] for key in source_keys if key in view_lookup]
    missing_references = [
        name
        for name in reference_image_names
        if name.casefold() not in view_lookup
    ]
    if missing_references:
        raise RuntimeError(
            "空间块参考照片不在稀疏模型中：" + "、".join(missing_references[:5])
        )
    if not references:
        raise RuntimeError("空间块没有具备稀疏点深度约束的参考照片")
    if len(source_views) < 2:
        raise RuntimeError("空间块内可用的源照片少于2张")
    rows: list[str] = []
    source_positions = {
        source.name.casefold(): index for index, source in enumerate(source_views)
    }
    for reference in references:
        candidates: list[tuple[int, int, str]] = []
        reference_position = source_positions.get(reference.name.casefold(), 0)
        for candidate_index, candidate in enumerate(source_views):
            if candidate.name.casefold() == reference.name.casefold():
                continue
            shared = len(reference.point_ids & candidate.point_ids)
            sequence_distance = abs(reference_position - candidate_index)
            candidates.append((shared, -sequence_distance, candidate.name))
        candidates.sort(reverse=True)
        count = min(max(1, int(source_count)), len(candidates))
        stable_candidates = [
            value for value in candidates if value[2].casefold() in stable_keys
        ]
        # Prefer every source whose photometric map is guaranteed to exist,
        # not merely the minimum safety quota.  Filling the remaining slots
        # with higher-ranked Halo views that do not own a map makes COLMAP
        # silently skip the most useful cross-block sources during geometric
        # consistency and can leave visible holes at Core boundaries.
        stable_count = min(count, len(stable_candidates))
        selected = stable_candidates[:stable_count]
        selected_keys = {value[2].casefold() for value in selected}
        for value in candidates:
            if len(selected) >= count:
                break
            if value[2].casefold() in selected_keys:
                continue
            selected.append(value)
            selected_keys.add(value[2].casefold())
        required_stable = min(
            max(0, int(minimum_stable_sources)),
            count,
            len(stable_candidates),
        )
        if sum(
            value[2].casefold() in stable_keys for value in selected
        ) < required_stable:
            raise RuntimeError("空间块稳定源照片数不足")
        candidate_names = [value[2] for value in selected]
        rows.extend((reference.name, ", ".join(candidate_names)))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return len(references)


def _fused_ply_info(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise RuntimeError(f"融合点云不存在：{path}")
    with path.open("rb") as stream:
        prefix = stream.read(64 * 1024)
    marker = prefix.find(b"end_header")
    newline = prefix.find(b"\n", marker)
    if marker < 0 or newline < 0:
        raise RuntimeError(f"融合点云PLY头不完整：{path}")
    offset = newline + 1
    lines = prefix[:offset].decode("ascii", errors="strict").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise RuntimeError(f"融合点云不是二进制小端PLY：{path}")
    count = 0
    properties: list[tuple[str, str]] = []
    in_vertices = False
    for line in lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3:
            count = int(parts[2])
            in_vertices = True
        elif parts[:1] == ["element"]:
            in_vertices = False
        elif in_vertices and parts[:1] == ["property"] and len(parts) == 3:
            properties.append((parts[1], parts[2]))
    if count <= 0 or tuple(properties) != _FUSED_PROPERTIES:
        raise RuntimeError(f"融合点云PLY顶点格式异常：{path}")
    if path.stat().st_size < offset + count * _FUSED_DTYPE.itemsize:
        raise RuntimeError(f"融合点云PLY数据被截断：{path}")
    return count, offset


def crop_fused_ply_to_core(
    input_path: str | Path,
    output_path: str | Path,
    *,
    center: Sequence[float],
    basis: Sequence[Sequence[float]],
    lower: Sequence[float],
    upper: Sequence[float],
) -> int:
    """Stream-crop a COLMAP fused PLY to one PCA-aligned Core prism."""

    source = Path(input_path)
    target = Path(output_path)
    count, offset = _fused_ply_info(source)
    center_array = np.asarray(center, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if center_array.shape != (3,) or basis_array.shape != (3, 3):
        raise ValueError("空间块坐标框架格式异常")
    if lower_array.shape != (3,) or upper_array.shape != (3,) or np.any(upper_array < lower_array):
        raise ValueError("空间块Core范围格式异常")

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = target.with_suffix(target.suffix + ".payload")
    temporary = target.with_suffix(target.suffix + ".writing")
    payload.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    selected_count = 0
    records_per_chunk = max(1, 16 * 1024**2 // _FUSED_DTYPE.itemsize)
    try:
        with source.open("rb") as stream, payload.open("wb") as payload_stream:
            stream.seek(offset)
            remaining = count
            while remaining:
                current = min(records_per_chunk, remaining)
                raw = stream.read(current * _FUSED_DTYPE.itemsize)
                if len(raw) != current * _FUSED_DTYPE.itemsize:
                    raise RuntimeError(f"读取空间块融合点云时提前结束：{source}")
                records = np.frombuffer(raw, dtype=_FUSED_DTYPE, count=current)
                xyz = np.column_stack((records["x"], records["y"], records["z"]))
                local = (xyz.astype(np.float64) - center_array) @ basis_array
                mask = np.all(
                    (local >= lower_array - 1e-6) & (local <= upper_array + 1e-6),
                    axis=1,
                )
                if np.any(mask):
                    selected = records[mask]
                    payload_stream.write(selected.tobytes(order="C"))
                    selected_count += int(mask.sum())
                remaining -= current
        if selected_count <= 0:
            raise RuntimeError("空间块融合成功，但Core范围内没有稠密点")
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {selected_count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        with temporary.open("wb") as output, payload.open("rb") as payload_stream:
            output.write(header)
            shutil.copyfileobj(payload_stream, output, length=16 * 1024**2)
        os.replace(temporary, target)
    finally:
        payload.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return selected_count
