"""Independent AI-photogrammetry worker using JSON-lines IPC over stdout."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .colmap_pipeline import run_colmap_ba_mvs
from .exporters import write_binary_ply
from .model_pipeline import run_model_pipeline
from .photo_selection import (
    analyze_sequence_continuity,
    discover_photos,
    recommended_continuous_segment,
    records_payload,
    scan_photos,
    select_keyframes,
)
from .point_processing import FilterOptions, process_session_cloud
from .project_store import ProjectStore

_print_lock = threading.Lock()
_cancelled = threading.Event()
_monitor_stop = threading.Event()


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "time": time.time(), **payload}
    with _print_lock:
        print(json.dumps(message, ensure_ascii=False, default=str), flush=True)


def _signal_cancel(_signum, _frame) -> None:
    _cancelled.set()
    emit("status", status="cancelling", message="收到取消请求")


def _gpu_telemetry() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creation_flags,
            check=False,
        )
        line = completed.stdout.splitlines()[0]
        index, name, used, total, utilization = [value.strip() for value in line.split(",", 4)]
        result.update(
            {
                "gpu_index": int(index),
                "gpu_name": name,
                "gpu_memory_used_gb": round(float(used) / 1024, 3),
                "gpu_memory_total_gb": round(float(total) / 1024, 3),
                "gpu_utilization_percent": float(utilization),
            }
        )
    except Exception:
        pass
    return result


def _monitor() -> None:
    while not _monitor_stop.wait(2):
        emit("telemetry", **_gpu_telemetry())


def _progress(value: float, text: str) -> None:
    if _cancelled.is_set():
        raise InterruptedError("任务已取消")
    emit("progress", progress=round(float(value) * 100, 2), message=text)



def run_scan(config: dict[str, Any]) -> dict[str, Any]:
    store = ProjectStore.open(config["project_root"])
    paths = [str(Path(value).resolve()) for value in config.get("source_images", [])]
    if not paths:
        paths = discover_photos(config["source_root"], recursive=bool(config.get("recursive", True)))
    store.stage_tracker.set("photo_scan", "running", details={"count": len(paths)})
    records, summary = scan_photos(
        paths,
        thumbnail_dir=store.root / "thumbnails",
        progress_callback=lambda value, text: _progress(0.72 * value, text),
        cancelled=_cancelled.is_set,
    )
    selection_policy = str(config.get("selection_policy", "automatic"))
    if selection_policy == "keep_all":
        selected = [record.path for record in records if record.valid]
        selected_paths = set(selected)
        for record in records:
            record.selected = record.path in selected_paths
    else:
        selected = select_keyframes(
            records,
            max_count=(
                0
                if selection_policy == "duplicates_only"
                else int(config.get("max_keyframes", 0))
            ),
            include_near_duplicates=(
                selection_policy == "duplicates_only"
                or bool(config.get("include_near_duplicates", False))
            ),
            exclude_quality_failures=selection_policy == "automatic",
        )
    selected_before_segmentation = list(selected)
    sequence_analysis = analyze_sequence_continuity(
        selected,
        progress_callback=lambda value, text: _progress(0.72 + 0.27 * value, text),
        cancelled=_cancelled.is_set,
    )
    segmentation = {"applied": False, "reason": "自动分段未启用"}
    if selection_policy == "automatic" and bool(config.get("auto_segment", False)):
        selected, segmentation = recommended_continuous_segment(
            selected,
            sequence_analysis,
            minimum_images=int(config.get("minimum_segment_images", 3)),
        )
        selected_paths = set(selected)
        for record in records:
            record.selected = record.path in selected_paths
    summary["sequence_break_count"] = len(sequence_analysis.get("breaks") or [])
    summary["sequence_segment_count"] = len(sequence_analysis.get("segments") or [])
    summary["sequence_warnings"] = list(sequence_analysis.get("warnings") or [])
    summary["selection_policy"] = selection_policy
    payload = records_payload(records, summary)
    payload["sequence_analysis"] = sequence_analysis
    payload["segmentation"] = segmentation
    payload["selected_before_segmentation"] = selected_before_segmentation
    store.update_manifest(
        source_images=paths,
        selected_images=selected,
        photo_scan=payload,
    )
    store.stage_tracker.set(
        "photo_scan",
        "completed",
        details={**summary, "selected_count": len(selected)},
    )
    return {
        **summary,
        "selected_count": len(selected),
        "segmentation": segmentation,
        "project_root": str(store.root),
    }



def run_colmap(config: dict[str, Any]) -> dict[str, Any]:
    store = ProjectStore.open(config["project_root"])
    manifest = store.read_manifest()
    session = store.load_session()
    target_stage = str(config.get("target_stage", "dense"))
    project_stage = "sparse_ba" if target_stage == "sparse" else "colmap_mvs"
    store.stage_tracker.set(project_stage, "running")
    selected_images = [
        str(Path(value).resolve())
        for value in (
            config.get("image_paths")
            or manifest.get("selected_images")
            or manifest.get("source_images")
            or []
        )
    ]
    result = run_colmap_ba_mvs(
        session,
        colmap_path=config.get("colmap_path"),
        output_root=(
            Path(config["output_root"]).resolve()
            if config.get("output_root")
            else store.root / "colmap"
        ),
        source_image_paths=selected_images,
        feature_type=str(config.get("feature_type", "aliked")),
        matcher=str(config.get("matcher", "auto")),
        mapper=str(config.get("mapper", "global")),
        camera_model=str(config.get("camera_model", "SIMPLE_RADIAL")),
        single_camera=bool(config.get("single_camera", True)),
        feature_max_image_size=int(config.get("feature_max_image_size", 4096)),
        max_image_size=int(config.get("max_image_size", 4096)),
        max_num_features=int(config.get("max_num_features", 4096)),
        sequential_overlap=int(config.get("sequential_overlap", 20)),
        geometric_consistency=bool(config.get("geometric_consistency", True)),
        patch_match_filter=bool(config.get("patch_match_filter", True)),
        patch_match_source_images=int(config.get("patch_match_source_images", 12)),
        patch_match_iterations=int(config.get("patch_match_iterations", 4)),
        mvs_reference_strategy="all",
        mvs_reference_ratio=1.0,
        spatial_blocking=bool(config.get("spatial_blocking", True)),
        spatial_block_threshold=int(config.get("spatial_block_threshold", 180)),
        spatial_block_target_images=int(
            config.get("spatial_block_target_images", 120)
        ),
        spatial_block_halo_ratio=float(
            config.get("spatial_block_halo_ratio", 0.20)
        ),
        min_num_inliers=int(config.get("min_num_inliers", 20)),
        ransac_max_error=float(config.get("ransac_max_error", 4.0)),
        fusion_min_num_pixels=int(config.get("fusion_min_num_pixels", 2)),
        generate_quality_report=bool(config.get("generate_quality_report", True)),
        target_stage=target_stage,
        use_gpu=bool(config.get("use_gpu", True)),
        progress_callback=_progress,
        resume=bool(config.get("resume", True)),
    )
    if target_stage == "dense":
        session.photogrammetry_result = dict(result)
        sparse_snapshot = dict(result)
        for key in (
            "pointcloud",
            "raw_fused",
            "pointcloud_metadata",
            "point_count",
            "unit",
            "estimated_workspace_gb",
            "disk_free_gb_at_start",
        ):
            sparse_snapshot.pop(key, None)
        sparse_snapshot["result_stage"] = "sparse"
        session.sparse_result = sparse_snapshot
    else:
        session.sparse_result = dict(result)
    # Persist inside the isolated worker as well as in the GUI completion
    # callback.  A finished result then survives a GUI crash or a headless
    # resume and is immediately available after reopening the project.
    store.save_session(session)
    store.stage_tracker.set(project_stage, "completed", details=result)
    return {**result, "project_root": str(store.root)}


def run_filter(config: dict[str, Any]) -> dict[str, Any]:
    store = ProjectStore.open(config["project_root"])
    session = store.load_session()
    source_pointcloud = str(session.photogrammetry_result.get("pointcloud", "")).strip()
    if not source_pointcloud:
        raise RuntimeError("工程中没有可过滤的 AI 摄影测量点云")
    store.stage_tracker.set("point_filter", "running")
    report = process_session_cloud(
        session,
        FilterOptions.from_dict(config.get("options")),
        source_pointcloud=source_pointcloud,
        progress_callback=_progress,
    )
    cache = store.save_processed_cache(session)
    assert session.processed_points is not None
    assert session.processed_colors is not None
    filtered_ply = store.processed_cache / "filtered.ply"
    write_binary_ply(
        filtered_ply,
        session.processed_points,
        session.processed_colors,
        [
            "AI photogrammetry filtered point cloud",
            f"coordinate_mode {session.transform.mode}",
            f"unit {session.unit}",
        ],
    )
    store.save_session(session)
    store.stage_tracker.set("point_filter", "completed", details=report)
    return {
        "project_root": str(store.root),
        "point_count": int(cache["point_count"]),
        "pointcloud": str(filtered_ply),
        "report": report,
    }


def run_model(config: dict[str, Any]) -> dict[str, Any]:
    store = ProjectStore.open(config["project_root"])
    session = store.load_session()
    dense_workspace = str(
        config.get("dense_workspace")
        or session.photogrammetry_result.get("dense_workspace", "")
    ).strip()
    pointcloud = str(
        config.get("pointcloud")
        or session.photogrammetry_result.get("raw_fused", "")
        or session.photogrammetry_result.get("pointcloud", "")
    ).strip()
    if not dense_workspace or not pointcloud:
        raise RuntimeError("项目中没有可用于模型生成的稠密点云和MVS工作区")
    store.stage_tracker.set("textured_model", "running")
    result = run_model_pipeline(
        dense_workspace=dense_workspace,
        pointcloud=pointcloud,
        output_root=config.get("output_root"),
        colmap_path=config.get("colmap_path"),
        precision_mode=str(config.get("precision_mode", "标准工程模式")),
        formats=config.get("formats") or ["obj", "fbx", "gltf"],
        osgconv_path=config.get("osgconv_path"),
        resume=bool(config.get("resume", True)),
        progress_callback=_progress,
    )
    session.model_options = {
        "generate_model": True,
        "precision_mode": str(config.get("precision_mode", "标准工程模式")),
        "formats": list(config.get("formats") or ["obj", "fbx", "gltf"]),
        "output_root": config.get("output_root"),
        "resume": bool(config.get("resume", True)),
    }
    session.model_result = dict(result)
    store.save_session(session)
    store.stage_tracker.set("textured_model", "completed", details=result)
    return {**result, "project_root": str(store.root)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    def mark_terminal(status: str, message: str) -> None:
        stage = {
            "scan_photos": "photo_scan",
            "filter": "point_filter",
            "model": "textured_model",
        }.get(str(config.get("task")))
        if str(config.get("task")) == "colmap":
            stage = (
                "sparse_ba"
                if str(config.get("target_stage", "dense")) == "sparse"
                else "colmap_mvs"
            )
        if not stage or not config.get("project_root"):
            return
        try:
            ProjectStore.open(config["project_root"]).stage_tracker.set(
                stage,
                status,
                message=message,
            )
        except Exception:
            pass

    signal.signal(signal.SIGINT, _signal_cancel)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_cancel)
    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    emit("status", status="started", task=config.get("task"), pid=os.getpid())
    try:
        task = str(config["task"])
        if task == "scan_photos":
            result = run_scan(config)
        elif task == "colmap":
            result = run_colmap(config)
        elif task == "filter":
            result = run_filter(config)
        elif task == "model":
            result = run_model(config)
        else:
            raise ValueError(f"未知任务：{task}")
        if _cancelled.is_set():
            raise InterruptedError("任务已取消")
        emit("result", status="completed", result=result)
        return 0
    except InterruptedError as exc:
        mark_terminal("cancelled", str(exc))
        emit("result", status="cancelled", error=str(exc))
        return 2
    except Exception as exc:
        mark_terminal("failed", str(exc))
        emit(
            "result",
            status="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1
    finally:
        _monitor_stop.set()
        monitor.join(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
