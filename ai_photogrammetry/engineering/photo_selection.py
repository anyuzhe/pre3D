"""Photo integrity, duplicate detection and automatic keyframe selection."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class PhotoRecord:
    path: str
    name: str
    size_bytes: int
    modified_ns: int
    width: int = 0
    height: int = 0
    orientation: int = 1
    captured_at: str = ""
    camera_model: str = ""
    lens_model: str = ""
    focal_length_mm: float | None = None
    gps_latitude: float | None = None
    gps_longitude: float | None = None
    gps_altitude: float | None = None
    gps_source: str = ""
    rtk_status: str = "NONE"
    sigma_x: float | None = None
    sigma_y: float | None = None
    sigma_z: float | None = None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    sharpness: float = 0.0
    dark_ratio: float = 0.0
    bright_ratio: float = 0.0
    quick_hash: str = ""
    perceptual_hash: str = ""
    valid: bool = True
    duplicate_of: str = ""
    near_duplicate_of: str = ""
    selected: bool = True
    warning: str = ""
    thumbnail_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PhotoRecord":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


def discover_photos(root: str | Path, recursive: bool = True) -> list[str]:
    directory = Path(root).expanduser().resolve()
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        [
        str(path.resolve())
        for path in iterator
        if path.is_file() and path.suffix.lower() in PHOTO_EXTENSIONS
        ],
        key=str.casefold,
    )


def _quick_hash(path: Path) -> str:
    """Hash size plus first/middle/last blocks; full hash only for collisions."""

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(size.to_bytes(8, "little"))
    with path.open("rb") as stream:
        positions = sorted({0, max(0, size // 2 - 32768), max(0, size - 65536)})
        for position in positions:
            stream.seek(position)
            digest.update(stream.read(65536))
    return digest.hexdigest()


def _full_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dhash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for flag in bits.flat:
        value = (value << 1) | int(flag)
    return f"{value:016x}"


def hash_distance(first: str, second: str) -> int:
    if not first or not second:
        return 64
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _exif_value(exif: Any, name: str, default: Any = "") -> Any:
    tag = next((key for key, value in ExifTags.TAGS.items() if value == name), None)
    if tag is None:
        return default
    value = exif.get(tag, None)
    if value not in {None, ""}:
        return value
    # Phones commonly keep DateTimeOriginal, FocalLength and LensModel in the
    # nested Exif IFD instead of the top-level IFD exposed by ``getexif``.
    try:
        nested = exif.get_ifd(34665)
    except (AttributeError, KeyError, TypeError, ValueError):
        nested = {}
    value = nested.get(tag, default) if nested else default
    return default if value in {None, ""} else value


def _float_exif(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if np.isfinite(result) else None


def _gps_decimal(
    values: Any,
    reference: Any,
) -> float | None:
    try:
        degrees, minutes, seconds = (float(value) for value in values)
        result = degrees + minutes / 60.0 + seconds / 3600.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if str(reference).upper() in {"S", "W"}:
        result = -result
    return result if np.isfinite(result) else None


def _gps_altitude_reference(value: Any) -> int | None:
    """Normalize EXIF GPSAltitudeRef, including iPhone's single-byte form."""

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return int(payload[0]) if payload else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if len(stripped) == 1 and ord(stripped) in {0, 1}:
            return ord(stripped)
        value = stripped
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _gps_metadata(exif: Any) -> dict[str, float | None]:
    empty = {
        "gps_latitude": None,
        "gps_longitude": None,
        "gps_altitude": None,
    }
    try:
        gps = exif.get_ifd(34853)
        if not gps:
            return empty
        latitude = _gps_decimal(gps.get(2), gps.get(1, ""))
        longitude = _gps_decimal(gps.get(4), gps.get(3, ""))
        altitude = _float_exif(gps.get(6))
        if altitude is not None and _gps_altitude_reference(gps.get(5)) == 1:
            altitude = -altitude
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
    ):
        return empty
    return {
        "gps_latitude": latitude,
        "gps_longitude": longitude,
        "gps_altitude": altitude,
    }


def _dji_xmp_metadata(path: Path) -> dict[str, Any]:
    """Read DJI POS/RTK XMP attributes without loading the full photograph."""

    empty: dict[str, Any] = {
        "gps_source": "",
        "rtk_status": "NONE",
        "sigma_x": None,
        "sigma_y": None,
        "sigma_z": None,
        "roll": None,
        "pitch": None,
        "yaw": None,
    }
    try:
        with path.open("rb") as stream:
            payload = stream.read(1024 * 1024)
    except OSError:
        return empty
    if b"drone-dji:" not in payload:
        return empty

    def attribute(name: str) -> str:
        match = re.search(
            rb"drone-dji:" + re.escape(name.encode("ascii")) + rb'=(["\'])(.*?)\1',
            payload,
            flags=re.IGNORECASE,
        )
        return match.group(2).decode("utf-8", errors="replace").strip() if match else ""

    def number(name: str) -> float | None:
        return _float_exif(attribute(name))

    latitude = number("GpsLatitude")
    longitude = number("GpsLongitude")
    altitude = number("AbsoluteAltitude")
    sigma_x = number("RtkStdLon")
    sigma_y = number("RtkStdLat")
    sigma_z = number("RtkStdHgt")
    gps_status = attribute("GpsStatus").upper()
    rtk_flag = attribute("RtkFlag")
    if "RTK" in gps_status or rtk_flag:
        if (
            sigma_x is not None
            and sigma_y is not None
            and sigma_z is not None
            and max(sigma_x, sigma_y) <= 0.05
            and sigma_z <= 0.10
        ):
            rtk_status = "RTK_FIX"
        else:
            rtk_status = "RTK_FLOAT"
    elif latitude is not None and longitude is not None:
        rtk_status = "GPS"
    else:
        rtk_status = "NONE"
    return {
        "gps_latitude": latitude,
        "gps_longitude": longitude,
        "gps_altitude": altitude,
        "gps_source": "DJI_XMP" if latitude is not None and longitude is not None else "",
        "rtk_status": rtk_status,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "roll": number("GimbalRollDegree"),
        "pitch": number("GimbalPitchDegree"),
        "yaw": number("GimbalYawDegree"),
    }


def _orientation_display_info(orientation: int) -> str:
    """Describe the normal display correction implied by EXIF orientation."""

    descriptions = {
        2: "显示方向已自动水平镜像（正常）",
        3: "显示方向已自动旋转180°（正常）",
        4: "显示方向已自动垂直镜像（正常）",
        5: "显示方向已自动镜像并旋转90°（正常）",
        6: "显示方向已自动顺时针旋转90°（正常）",
        7: "显示方向已自动镜像并逆时针旋转90°（正常）",
        8: "显示方向已自动逆时针旋转90°（正常）",
    }
    return descriptions.get(int(orientation), "")


def _read_for_analysis(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    xmp = _dji_xmp_metadata(path)
    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        exif = source.getexif()
        metadata = {
            "original_size": source.size,
            "orientation": int(_exif_value(exif, "Orientation", 1) or 1),
            "captured_at": str(_exif_value(exif, "DateTimeOriginal", "") or ""),
            "camera_model": str(_exif_value(exif, "Model", "") or "").strip(),
            "lens_model": str(_exif_value(exif, "LensModel", "") or "").strip(),
            "focal_length_mm": _float_exif(_exif_value(exif, "FocalLength", None)),
            **_gps_metadata(exif),
        }
        for key, value in xmp.items():
            if value not in {None, "", "NONE"}:
                metadata[key] = value
        metadata.setdefault(
            "gps_source",
            "EXIF"
            if metadata.get("gps_latitude") is not None
            and metadata.get("gps_longitude") is not None
            else "",
        )
        metadata.setdefault("rtk_status", "GPS" if metadata["gps_source"] else "NONE")
        for key in ("sigma_x", "sigma_y", "sigma_z", "roll", "pitch", "yaw"):
            metadata.setdefault(key, None)
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        return image.copy(), metadata


def _write_thumbnail(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.jpg")
    image.copy().resize((240, 180), Image.Resampling.LANCZOS).save(
        temporary,
        "JPEG",
        quality=82,
        optimize=True,
    )
    os.replace(temporary, path)


def scan_photos(
    paths: list[str],
    *,
    thumbnail_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[list[PhotoRecord], dict[str, Any]]:
    """Validate and characterize images, then mark exact/near duplicates."""

    records: list[PhotoRecord] = []
    total = len(paths)
    thumbnail_root = Path(thumbnail_dir) if thumbnail_dir else None
    for index, value in enumerate(paths):
        if cancelled and cancelled():
            raise InterruptedError("照片扫描已取消")
        path = Path(value).resolve()
        stat = path.stat()
        record = PhotoRecord(
            path=str(path),
            name=path.name,
            size_bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
        )
        warnings: list[str] = []
        try:
            image, metadata = _read_for_analysis(path)
            record.width, record.height = metadata["original_size"]
            record.orientation = metadata["orientation"]
            record.captured_at = metadata["captured_at"]
            record.camera_model = metadata["camera_model"]
            record.lens_model = metadata["lens_model"]
            record.focal_length_mm = metadata["focal_length_mm"]
            record.gps_latitude = metadata["gps_latitude"]
            record.gps_longitude = metadata["gps_longitude"]
            record.gps_altitude = metadata["gps_altitude"]
            record.gps_source = metadata["gps_source"]
            record.rtk_status = metadata["rtk_status"]
            record.sigma_x = metadata["sigma_x"]
            record.sigma_y = metadata["sigma_y"]
            record.sigma_z = metadata["sigma_z"]
            record.roll = metadata["roll"]
            record.pitch = metadata["pitch"]
            record.yaw = metadata["yaw"]
            array = np.asarray(image)
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            record.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            record.dark_ratio = float(np.mean(gray <= 5))
            record.bright_ratio = float(np.mean(gray >= 250))
            record.quick_hash = _quick_hash(path)
            record.perceptual_hash = _dhash(gray)
            if record.sharpness < 80:
                warnings.append("疑似模糊")
            if record.dark_ratio > 0.08:
                warnings.append("暗部剪切")
            if record.bright_ratio > 0.08:
                warnings.append("高光剪切")
            if thumbnail_root:
                thumbnail_name = f"{index:06d}_{path.stem[:80]}.jpg"
                thumbnail_path = thumbnail_root / thumbnail_name
                _write_thumbnail(image, thumbnail_path)
                record.thumbnail_path = str(thumbnail_path.resolve())
        except Exception as exc:
            record.valid = False
            record.selected = False
            warnings.append(f"损坏或无法读取：{exc}")
        record.warning = "；".join(warnings)
        records.append(record)
        if progress_callback:
            progress_callback(
                (index + 1) / max(total, 1) * 0.82,
                f"检查照片 {index + 1}/{total}",
            )

    quick_groups: dict[tuple[int, str], list[int]] = {}
    for index, record in enumerate(records):
        if record.valid:
            quick_groups.setdefault((record.size_bytes, record.quick_hash), []).append(index)
    for group in quick_groups.values():
        if len(group) < 2:
            continue
        full_groups: dict[str, list[int]] = {}
        for index in group:
            full_groups.setdefault(_full_hash(Path(records[index].path)), []).append(index)
        for exact_group in full_groups.values():
            for duplicate_index in exact_group[1:]:
                records[duplicate_index].duplicate_of = records[exact_group[0]].name
                records[duplicate_index].selected = False

    valid_indices = [
        index
        for index, record in enumerate(records)
        if record.valid and not record.duplicate_of
    ]
    for position, index in enumerate(valid_indices):
        current = records[index]
        for previous_index in valid_indices[max(0, position - 4) : position]:
            previous = records[previous_index]
            if hash_distance(current.perceptual_hash, previous.perceptual_hash) <= 2:
                current.near_duplicate_of = previous.name
                break
    if progress_callback:
        progress_callback(0.95, "汇总重复照片与质量统计")

    valid_records = [record for record in records if record.valid]
    summary = {
        "photo_count": len(records),
        "valid_count": len(valid_records),
        "invalid_count": sum(not record.valid for record in records),
        "exact_duplicate_count": sum(bool(record.duplicate_of) for record in records),
        "near_duplicate_count": sum(bool(record.near_duplicate_of) for record in records),
        "blur_count": sum(record.valid and record.sharpness < 80 for record in records),
        "orientation_correction_count": sum(
            record.valid and record.orientation not in {0, 1} for record in records
        ),
        "gps_count": sum(
            record.gps_latitude is not None and record.gps_longitude is not None
            for record in records
        ),
        "rtk_fix_count": sum(record.rtk_status == "RTK_FIX" for record in records),
        "rtk_float_count": sum(record.rtk_status == "RTK_FLOAT" for record in records),
        "gps_only_count": sum(record.rtk_status == "GPS" for record in records),
        "rtk_median_sigma_xyz": (
            np.median(
                np.asarray(
                    [
                        [record.sigma_x, record.sigma_y, record.sigma_z]
                        for record in records
                        if record.rtk_status == "RTK_FIX"
                        and record.sigma_x is not None
                        and record.sigma_y is not None
                        and record.sigma_z is not None
                    ],
                    dtype=np.float64,
                ),
                axis=0,
            ).tolist()
            if any(
                record.rtk_status == "RTK_FIX"
                and record.sigma_x is not None
                and record.sigma_y is not None
                and record.sigma_z is not None
                for record in records
            )
            else None
        ),
        "camera_models": sorted({record.camera_model for record in valid_records if record.camera_model}),
        "lens_models": sorted({record.lens_model for record in valid_records if record.lens_model}),
        "resolution_distribution": _resolution_distribution(valid_records),
    }
    if progress_callback:
        progress_callback(1.0, "照片质量与重复检测完成")
    return records, summary


def _resolution_distribution(records: list[PhotoRecord]) -> dict[str, Any]:
    megapixels = np.asarray(
        [record.width * record.height / 1e6 for record in records],
        dtype=np.float64,
    )
    if not len(megapixels):
        return {}
    return {
        "minimum_mp": float(megapixels.min()),
        "median_mp": float(np.median(megapixels)),
        "maximum_mp": float(megapixels.max()),
    }


def _capture_sort_key(record: PhotoRecord) -> tuple[str, str]:
    value = record.captured_at
    if value:
        for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, pattern).isoformat(), record.name
            except ValueError:
                pass
    return "", record.name


def select_keyframes(
    records: list[PhotoRecord],
    *,
    max_count: int,
    include_near_duplicates: bool = False,
    exclude_quality_failures: bool = False,
) -> list[str]:
    """Select spatially/temporally distributed sharp frames deterministically."""

    candidates = [
        record
        for record in sorted(records, key=_capture_sort_key)
        if record.valid
        and not record.duplicate_of
        and (include_near_duplicates or not record.near_duplicate_of)
        and (
            not exclude_quality_failures
            or (
                record.sharpness >= 80
                and record.dark_ratio <= 0.08
                and record.bright_ratio <= 0.08
            )
        )
    ]
    if max_count <= 0 or len(candidates) <= max_count:
        selected = candidates
    else:
        sharpness = np.asarray([record.sharpness for record in candidates])
        floor = float(np.quantile(sharpness, 0.08))
        selected = []
        boundaries = np.linspace(0, len(candidates), max_count + 1, dtype=int)
        previous_hash = ""
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group = candidates[start:max(end, start + 1)]
            usable = [record for record in group if record.sharpness >= floor] or group
            best = max(
                usable,
                key=lambda record: (
                    np.log1p(record.sharpness)
                    * (1.0 + hash_distance(previous_hash, record.perceptual_hash) / 64.0)
                ),
            )
            selected.append(best)
            previous_hash = best.perceptual_hash
    selected_paths = {record.path for record in selected}
    for record in records:
        record.selected = record.path in selected_paths
    return [record.path for record in selected]


def records_payload(records: list[PhotoRecord], summary: dict[str, Any]) -> dict[str, Any]:
    return {"summary": summary, "records": [record.to_dict() for record in records]}


def _sequence_features(
    path: str,
    detector: Any,
    *,
    max_size: int,
) -> tuple[tuple[int, int], list[Any], np.ndarray | None]:
    with Image.open(path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(32, round(width * scale)), max(32, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return gray.shape, keypoints or [], descriptors


def _feature_coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    if len(points) < 2:
        return 0.0
    height, width = shape
    span = np.ptp(points, axis=0)
    return float((span[0] / max(width, 1)) * (span[1] / max(height, 1)))


def analyze_sequence_continuity(
    paths: list[str],
    *,
    max_size: int = 800,
    progress_callback: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Measure adjacent visual continuity and recommend safe sequence segments.

    The diagnostic intentionally uses only adjacent frames. It is fast enough
    for preflight checks and catches capture restarts before an expensive GPU
    reconstruction turns them into folded or duplicated geometry.
    """

    resolved = [str(Path(value).resolve()) for value in paths]
    if len(resolved) < 2:
        return {
            "image_count": len(resolved),
            "pair_count": 0,
            "pairs": [],
            "breaks": [],
            "segments": (
                [
                    {
                        "start_index": 0,
                        "end_index": 0,
                        "image_count": 1,
                        "first_image": Path(resolved[0]).name,
                        "last_image": Path(resolved[0]).name,
                    }
                ]
                if resolved
                else []
            ),
            "recommended_segment_index": 0 if resolved else None,
            "warnings": [],
        }

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=4_000)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        ratio_threshold = 0.75
        detector_name = "SIFT"
    else:  # pragma: no cover - current OpenCV wheels include SIFT.
        detector = cv2.ORB_create(nfeatures=4_000, fastThreshold=12)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio_threshold = 0.78
        detector_name = "ORB"

    features: list[tuple[tuple[int, int], list[Any], np.ndarray | None]] = []
    for index, path in enumerate(resolved):
        if cancelled and cancelled():
            raise InterruptedError("照片序列连续性检查已取消")
        features.append(_sequence_features(path, detector, max_size=max_size))
        if progress_callback:
            progress_callback(
                0.45 * (index + 1) / len(resolved),
                f"提取序列特征 {index + 1}/{len(resolved)}",
            )

    pair_rows: list[dict[str, Any]] = []
    break_indices: list[int] = []
    for index in range(len(resolved) - 1):
        if cancelled and cancelled():
            raise InterruptedError("照片序列连续性检查已取消")
        shape_a, keypoints_a, descriptors_a = features[index]
        shape_b, keypoints_b, descriptors_b = features[index + 1]
        good: list[Any] = []
        if (
            descriptors_a is not None
            and descriptors_b is not None
            and len(descriptors_a) >= 2
            and len(descriptors_b) >= 2
        ):
            for candidates in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
                if len(candidates) == 2 and candidates[0].distance < ratio_threshold * candidates[1].distance:
                    good.append(candidates[0])

        inlier_count = 0
        coverage = 0.0
        displacement = 0.0
        if len(good) >= 8:
            points_a = np.float32([keypoints_a[item.queryIdx].pt for item in good])
            points_b = np.float32([keypoints_b[item.trainIdx].pt for item in good])
            _, inlier_mask = cv2.findFundamentalMat(
                points_a,
                points_b,
                cv2.FM_RANSAC,
                1.5,
                0.995,
            )
            if inlier_mask is not None:
                inliers = inlier_mask.ravel().astype(bool)
                inlier_count = int(inliers.sum())
                if inlier_count:
                    coverage = min(
                        _feature_coverage(points_a[inliers], shape_a),
                        _feature_coverage(points_b[inliers], shape_b),
                    )
                    diagonal = max(float(np.hypot(*shape_a)), 1.0)
                    displacement = float(
                        np.median(np.linalg.norm(points_b[inliers] - points_a[inliers], axis=1))
                        / diagonal
                    )

        hard_break = inlier_count < 24 or coverage < 0.06
        large_weak_jump = inlier_count < 80 and displacement > 0.35
        recommended_break = bool(hard_break or large_weak_jump)
        if recommended_break:
            break_indices.append(index)
        status = "disconnected" if hard_break else ("weak" if large_weak_jump else "continuous")
        pair_rows.append(
            {
                "first_index": index,
                "second_index": index + 1,
                "first_image": Path(resolved[index]).name,
                "second_image": Path(resolved[index + 1]).name,
                "match_count": len(good),
                "inlier_count": inlier_count,
                "feature_coverage": round(coverage, 4),
                "median_displacement_normalized": round(displacement, 4),
                "status": status,
                "recommended_break": recommended_break,
            }
        )
        if progress_callback:
            progress_callback(
                0.45 + 0.55 * (index + 1) / (len(resolved) - 1),
                f"检查相邻覆盖 {index + 1}/{len(resolved) - 1}",
            )

    starts = [0, *(index + 1 for index in break_indices)]
    ends = [*break_indices, len(resolved) - 1]
    segments = [
        {
            "start_index": start,
            "end_index": end,
            "image_count": end - start + 1,
            "first_image": Path(resolved[start]).name,
            "last_image": Path(resolved[end]).name,
        }
        for start, end in zip(starts, ends)
    ]
    recommended_segment_index = (
        max(range(len(segments)), key=lambda value: segments[value]["image_count"])
        if segments
        else None
    )
    warnings: list[str] = []
    if break_indices:
        names = [
            f"{Path(resolved[index]).name} → {Path(resolved[index + 1]).name}"
            for index in break_indices
        ]
        warnings.append("检测到照片序列断层：" + "；".join(names))
    return {
        "image_count": len(resolved),
        "pair_count": len(pair_rows),
        "detector": detector_name,
        "pairs": pair_rows,
        "breaks": [pair_rows[index] for index in break_indices],
        "segments": segments,
        "recommended_segment_index": recommended_segment_index,
        "warnings": warnings,
    }


def recommended_continuous_segment(
    paths: list[str],
    analysis: dict[str, Any],
    *,
    minimum_images: int = 3,
) -> tuple[list[str], dict[str, Any]]:
    """Return the longest safe segment without silently reducing tiny inputs."""

    segments = analysis.get("segments") or []
    recommended_index = analysis.get("recommended_segment_index")
    if recommended_index is None or not 0 <= int(recommended_index) < len(segments):
        return list(paths), {"applied": False, "reason": "没有可用的连续段"}
    segment = segments[int(recommended_index)]
    start = int(segment["start_index"])
    end = int(segment["end_index"])
    selected = list(paths[start : end + 1])
    if len(selected) < minimum_images or len(selected) == len(paths):
        return list(paths), {
            "applied": False,
            "reason": "无需分段" if len(selected) == len(paths) else "最长连续段照片过少",
            "recommended_segment": segment,
        }
    return selected, {
        "applied": True,
        "reason": "已保留最长连续拍摄段",
        "input_count": len(paths),
        "output_count": len(selected),
        "excluded_count": len(paths) - len(selected),
        "recommended_segment": segment,
    }
