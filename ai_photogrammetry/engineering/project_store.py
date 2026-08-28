"""Persistent project layout, metadata, and filtered-cloud cache."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import psutil

from .calibration import SimilarityTransform
from .coordinate_systems import CoordinateReference
from .session import (
    CoordinateObservation,
    DistanceConstraint,
    Measurement,
    ProjectSession,
)

PROJECT_VERSION = 4
CACHE_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def source_fingerprint(paths: list[str]) -> str:
    """Stable cheap fingerprint based on resolved paths, sizes and mtimes."""

    import hashlib

    digest = hashlib.sha256()
    for value in paths:
        path = Path(value).resolve()
        stat = path.stat()
        digest.update(str(path).casefold().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


class StageTracker:
    """Atomically persist coarse task stages for crash recovery."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "stages": {},
        }
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except (OSError, json.JSONDecodeError):
                pass

    def status(self, stage: str) -> str:
        return str(self.data.get("stages", {}).get(stage, {}).get("status", "pending"))

    def set(
        self,
        stage: str,
        status: str,
        *,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        if status not in {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
            "interrupted",
        }:
            raise ValueError(f"未知阶段状态：{status}")
        now = datetime.now().isoformat(timespec="seconds")
        record = dict(self.data.setdefault("stages", {}).get(stage, {}))
        record.update({"status": status, "updated_at": now, "message": message})
        if status == "pending":
            record.pop("started_at", None)
            record.pop("finished_at", None)
            record.pop("details", None)
        if status == "running":
            record["started_at"] = now
            record.pop("finished_at", None)
        if status in {"completed", "failed", "cancelled", "interrupted"}:
            record["finished_at"] = now
        if details:
            record["details"] = details
        self.data["stages"][stage] = record
        self.data["updated_at"] = now
        _atomic_json(self.path, self.data)

    def recover_interrupted(
        self,
        message: str = "上次任务未正常结束；缓存已保留，可断点继续",
    ) -> list[str]:
        """Convert orphaned running stages into resumable terminal records."""

        recovered = [
            str(name)
            for name, record in self.data.get("stages", {}).items()
            if str(record.get("status", "")) == "running"
        ]
        for name in recovered:
            self.set(name, "interrupted", message=message)
        return recovered


class ProjectStore:
    """One durable project rooted at a user-selected folder."""

    DIRECTORY_NAMES = (
        "thumbnails",
        "cache/model_outputs",
        "cache/checkpoints",
        "colmap",
        "export",
        "logs",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.project_file = self.root / "project.json"
        self.cache_root = self.root / "cache" / "model_outputs"
        self.processed_cache = self.cache_root / "processed_cloud"
        self.stage_tracker = StageTracker(self.root / "cache" / "checkpoints" / "stages.json")

    @classmethod
    def create(
        cls,
        root: str | Path,
        name: str,
        *,
        project_type: str = "近景岩体重建",
        output_coordinate_system: str = "本地模型坐标（后续标定）",
        precision_mode: str = "标准工程模式",
    ) -> "ProjectStore":
        store = cls(root)
        store.root.mkdir(parents=True, exist_ok=True)
        if store.project_file.exists():
            raise FileExistsError(f"工程已经存在：{store.project_file}")
        for relative in cls.DIRECTORY_NAMES:
            (store.root / relative).mkdir(parents=True, exist_ok=True)
        now = datetime.now().isoformat(timespec="seconds")
        _atomic_json(
            store.project_file,
            {
                "version": PROJECT_VERSION,
                "project_name": name.strip() or store.root.name,
                "project_id": uuid4().hex[:12],
                "project_type": project_type,
                "output_coordinate_system": output_coordinate_system,
                "precision_mode": precision_mode,
                "created_at": now,
                "updated_at": now,
                "source_images": [],
                "selected_images": [],
                "photo_scan": {},
                "session": {},
            },
        )
        return store

    @classmethod
    def open(cls, value: str | Path) -> "ProjectStore":
        path = Path(value).expanduser().resolve()
        root = path.parent if path.name.lower() == "project.json" else path
        store = cls(root)
        if not store.project_file.is_file():
            raise FileNotFoundError(f"未找到工程文件：{store.project_file}")
        manifest = store.read_manifest()
        if int(manifest.get("version", 0)) > PROJECT_VERSION:
            raise RuntimeError("该工程由更高版本软件创建")
        for relative in cls.DIRECTORY_NAMES:
            (store.root / relative).mkdir(parents=True, exist_ok=True)
        return store

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.project_file.read_text(encoding="utf-8"))

    def _has_live_worker(self) -> bool:
        """Whether another process is actively working on this project."""

        project_token = str(self.root).casefold()
        for process in psutil.process_iter(["cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or []).casefold()
            except (psutil.Error, OSError):
                continue
            if (
                "ai_photogrammetry.engineering.worker" in command
                and project_token in command
            ):
                return True
        return False

    def recover_interrupted_stages(self) -> list[dict[str, str]]:
        """Mark stale running checkpoints as interrupted when no worker exists."""

        if self._has_live_worker():
            return []
        paths = {self.stage_tracker.path}
        colmap_root = self.root / "colmap"
        if colmap_root.is_dir():
            paths.update(colmap_root.glob("photogrammetry_*/pipeline_state.json"))
            paths.update(
                colmap_root.glob("photogrammetry_*/dense_*/pipeline_state.json")
            )
            paths.update(
                colmap_root.glob(
                    "photogrammetry_*/dense_*/model_*/pipeline_state.json"
                )
            )
        recovered: list[dict[str, str]] = []
        for path in sorted(paths, key=lambda item: str(item).casefold()):
            if not path.is_file():
                continue
            tracker = StageTracker(path)
            for stage in tracker.recover_interrupted():
                recovered.append({"stage": stage, "path": str(path)})
        self.stage_tracker = StageTracker(self.stage_tracker.path)
        return recovered

    def update_manifest(self, **updates: Any) -> dict[str, Any]:
        payload = self.read_manifest()
        payload.update(updates)
        payload["version"] = PROJECT_VERSION
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(self.project_file, payload)
        return payload

    def save_session(
        self,
        session: ProjectSession,
        *,
        source_images: list[str] | None = None,
        selected_images: list[str] | None = None,
        photo_scan: dict[str, Any] | None = None,
    ) -> None:
        session_payload = {
            "project_name": session.project_name,
            "project_id": session.project_id,
            "created_at": session.created_at,
            "transform": session.transform.to_dict(),
            "coordinate_reference": session.coordinate_reference.to_dict(),
            "distance_constraints": [
                {
                    "label": item.label,
                    "point_a": item.point_a,
                    "point_b": item.point_b,
                    "actual_distance_m": item.actual_distance_m,
                    "source": item.source,
                }
                for item in session.distance_constraints
            ],
            "coordinate_observations": [asdict(item) for item in session.coordinate_observations],
            "measurements": [asdict(item) for item in session.measurements],
            "quality_report": session.quality_report,
            "calibration_report": session.calibration_report,
            "photogrammetry_options": session.photogrammetry_options,
            "model_options": session.model_options,
            "sparse_result": session.sparse_result,
            "photogrammetry_result": session.photogrammetry_result,
            "model_result": session.model_result,
            "output_dir": str(session.output_dir) if session.output_dir else None,
            "filter_report": session.filter_report,
        }
        manifest = self.read_manifest()
        self.update_manifest(
            project_name=session.project_name,
            project_id=session.project_id,
            created_at=session.created_at,
            source_images=source_images if source_images is not None else manifest.get("source_images", []),
            selected_images=(
                selected_images if selected_images is not None else manifest.get("selected_images", [])
            ),
            photo_scan=photo_scan if photo_scan is not None else manifest.get("photo_scan", {}),
            session=session_payload,
        )

    def load_session(self) -> ProjectSession:
        manifest = self.read_manifest()
        saved = manifest.get("session") or {}
        session = ProjectSession(
            project_name=str(saved.get("project_name") or manifest.get("project_name") or self.root.name),
            project_id=str(saved.get("project_id") or manifest.get("project_id") or uuid4().hex[:12]),
            created_at=str(
                saved.get("created_at")
                or manifest.get("created_at")
                or datetime.now().isoformat(timespec="seconds")
            ),
        )
        transform = saved.get("transform") or {}
        if transform:
            session.transform = SimilarityTransform(
                scale=float(transform.get("scale", 1.0)),
                rotation=np.asarray(transform.get("rotation", np.eye(3)), dtype=np.float64),
                translation=np.asarray(transform.get("translation", np.zeros(3)), dtype=np.float64),
                mode=str(transform.get("mode", "preview")),
            )
        session.coordinate_reference = CoordinateReference.from_dict(
            saved.get("coordinate_reference")
        )
        session.distance_constraints = [
            DistanceConstraint(
                label=str(item["label"]),
                point_a=np.asarray(item["point_a"], dtype=np.float64),
                point_b=np.asarray(item["point_b"], dtype=np.float64),
                actual_distance_m=float(item["actual_distance_m"]),
                source=str(item.get("source", "manual")),
            )
            for item in saved.get("distance_constraints", [])
        ]
        session.coordinate_observations = [
            CoordinateObservation(
                point_id=str(item["point_id"]),
                model_xyz=np.asarray(item["model_xyz"], dtype=np.float64),
                target_xyz=np.asarray(item["target_xyz"], dtype=np.float64),
                role=str(item.get("role", "control")),
                image_name=str(item.get("image_name", "")),
                pixel_uv=tuple(item["pixel_uv"]) if item.get("pixel_uv") else None,
                sigma_xyz=(
                    tuple(float(value) for value in item["sigma_xyz"])
                    if item.get("sigma_xyz")
                    else None
                ),
                source_crs=str(item.get("source_crs", "")),
            )
            for item in saved.get("coordinate_observations", [])
        ]
        session.measurements = [
            Measurement(
                kind=str(item["kind"]),
                value=float(item["value"]),
                unit=str(item["unit"]),
                label=str(item["label"]),
                point_count=int(item["point_count"]),
            )
            for item in saved.get("measurements", [])
        ]
        session.quality_report = saved.get("quality_report") or session.quality_report
        session.calibration_report = saved.get("calibration_report") or {}
        session.photogrammetry_options = saved.get("photogrammetry_options") or {}
        session.model_options = saved.get("model_options") or {}
        session.sparse_result = saved.get("sparse_result") or {}
        session.photogrammetry_result = saved.get("photogrammetry_result") or {}
        session.model_result = saved.get("model_result") or {}
        session.filter_report = saved.get("filter_report") or {}
        output_dir = saved.get("output_dir")
        session.output_dir = Path(output_dir) if output_dir else None
        self.load_processed_cache(session)
        return session


    def save_processed_cache(self, session: ProjectSession) -> dict[str, Any]:
        if not session.has_processed_cloud:
            raise RuntimeError("当前会话没有已处理点云")
        assert session.processed_points is not None
        assert session.processed_colors is not None
        root = self.processed_cache
        _atomic_npy(root / "points.npy", session.processed_points)
        _atomic_npy(root / "colors.npy", session.processed_colors)
        payload = {
            "version": CACHE_VERSION,
            "complete": True,
            "coordinate_mode": session.processed_coordinate_mode,
            "transform": session.transform.to_dict(),
            "point_count": int(len(session.processed_points)),
            "filter_report": session.filter_report,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _atomic_json(root / "manifest.json", payload)
        return payload

    def load_processed_cache(self, session: ProjectSession) -> bool:
        manifest_path = self.processed_cache / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            saved_transform = payload.get("transform") or {}
            same_transform = (
                payload.get("coordinate_mode") == session.transform.mode
                and np.isclose(float(saved_transform.get("scale", np.nan)), session.transform.scale)
                and np.allclose(
                    np.asarray(saved_transform.get("rotation"), dtype=np.float64),
                    session.transform.rotation,
                )
                and np.allclose(
                    np.asarray(saved_transform.get("translation"), dtype=np.float64),
                    session.transform.translation,
                )
            )
            files = [
                self.processed_cache / "points.npy",
                self.processed_cache / "colors.npy",
            ]
            if not payload.get("complete") or not same_transform or not all(
                path.is_file() for path in files
            ):
                return False
            session.processed_points = np.load(files[0], allow_pickle=False)
            session.processed_colors = np.load(files[1], allow_pickle=False)
            session.processed_coordinate_mode = str(payload["coordinate_mode"])
            session.filter_report = payload.get("filter_report") or session.filter_report
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
