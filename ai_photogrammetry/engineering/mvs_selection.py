"""Select dense-MVS reference views after sparse geometry is known."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SparseView:
    image_id: int
    name: str
    point_ids: frozenset[int]
    camera_center: tuple[float, float, float] = (float("nan"),) * 3


def _camera_center(qvec: list[float], tvec: list[float]) -> tuple[float, float, float]:
    qw, qx, qy, qz = qvec
    rotation = (
        (
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ),
        (
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ),
        (
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ),
    )
    return tuple(
        -sum(rotation[row][column] * tvec[row] for row in range(3))
        for column in range(3)
    )


def read_sparse_views(images_txt: str | Path) -> list[SparseView]:
    """Read registered images and their observed sparse point IDs."""

    path = Path(images_txt)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    views: list[SparseView] = []
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        index += 1
        if not header or header.startswith("#"):
            continue
        parts = header.split(maxsplit=9)
        if len(parts) != 10:
            continue
        try:
            image_id = int(parts[0])
            pose = [float(value) for value in parts[1:8]]
            int(parts[8])
        except ValueError:
            continue
        observations = lines[index].strip() if index < len(lines) else ""
        index += 1
        values = observations.split()
        point_ids: set[int] = set()
        for offset in range(2, len(values), 3):
            try:
                point_id = int(values[offset])
            except ValueError:
                continue
            if point_id >= 0:
                point_ids.add(point_id)
        views.append(
            SparseView(
                image_id=image_id,
                name=parts[9],
                point_ids=frozenset(point_ids),
                camera_center=_camera_center(pose[:4], pose[4:7]),
            )
        )
    return views


def _evenly_spaced_indices(count: int, target: int) -> list[int]:
    if target >= count:
        return list(range(count))
    if target <= 1:
        return [count // 2]
    return sorted(
        {
            min(count - 1, round(position * (count - 1) / (target - 1)))
            for position in range(target)
        }
    )


def select_mvs_references(
    images_txt: str | Path,
    *,
    reference_ratio: float,
    strategy: str = "covisibility",
) -> dict[str, object]:
    """Greedily retain views that cover sparse geometry with low redundancy."""

    if strategy not in {"covisibility", "all"}:
        raise ValueError("MVS参考帧策略必须是 covisibility 或 all")
    if not 0.1 <= reference_ratio <= 1.0:
        raise ValueError("MVS参考帧比例必须在 0.1～1.0 之间")
    views = read_sparse_views(images_txt)
    count = len(views)
    if count < 3:
        raise RuntimeError("注册照片少于3张，无法选择MVS参考帧")
    target = (
        count
        if strategy == "all"
        else min(count, max(3, math.ceil(count * reference_ratio)))
    )
    all_points = set().union(*(view.point_ids for view in views))
    if target == count:
        selected_indices = list(range(count))
    elif not all_points:
        selected_indices = _evenly_spaced_indices(count, target)
    else:
        frequencies = Counter(
            point_id
            for view in views
            for point_id in view.point_ids
        )
        weights = {
            point_id: 1.0 / math.sqrt(max(frequency, 1))
            for point_id, frequency in frequencies.items()
        }
        selected: set[int] = {0, count - 1}
        if target >= 3:
            selected.add(
                max(
                    range(count),
                    key=lambda value: (
                        sum(weights[point_id] for point_id in views[value].point_ids),
                        len(views[value].point_ids),
                    ),
                )
            )
        covered = set().union(*(views[value].point_ids for value in selected))
        while len(selected) < target:
            best_index = -1
            best_score: tuple[float, float, float, float] | None = None
            for candidate_index, candidate in enumerate(views):
                if candidate_index in selected:
                    continue
                new_points = candidate.point_ids - covered
                new_weight = sum(weights[point_id] for point_id in new_points)
                total_weight = sum(
                    weights[point_id] for point_id in candidate.point_ids
                )
                max_redundancy = max(
                    (
                        len(candidate.point_ids & views[value].point_ids)
                        / max(
                            1,
                            min(
                                len(candidate.point_ids),
                                len(views[value].point_ids),
                            ),
                        )
                        for value in selected
                    ),
                    default=0.0,
                )
                sequence_gap = min(
                    abs(candidate_index - value) for value in selected
                ) / max(count - 1, 1)
                score = (
                    new_weight,
                    1.0 - max_redundancy,
                    sequence_gap,
                    total_weight,
                )
                if best_score is None or score > best_score:
                    best_index = candidate_index
                    best_score = score
            if best_index < 0:
                break
            selected.add(best_index)
            covered.update(views[best_index].point_ids)
        selected_indices = sorted(selected)

    selected_views = [views[index] for index in selected_indices]
    covered_points = set().union(*(view.point_ids for view in selected_views))
    selected_names = [view.name for view in selected_views]
    return {
        "strategy": strategy,
        "registered_image_count": count,
        "reference_image_count": len(selected_names),
        "helper_source_image_count": count - len(selected_names),
        "requested_reference_ratio": float(reference_ratio),
        "actual_reference_ratio": len(selected_names) / count,
        "sparse_point_count": len(all_points),
        "covered_sparse_point_count": len(covered_points),
        "sparse_point_coverage_ratio": (
            len(covered_points) / len(all_points) if all_points else 1.0
        ),
        "reference_images": selected_names,
    }
