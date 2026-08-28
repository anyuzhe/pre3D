"""Run reproducible photo-management and AI-photogrammetry stress tests."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from ai_photogrammetry.engineering.photo_selection import discover_photos
from ai_photogrammetry.engineering.project_store import ProjectStore


def _monitor_memory(process: subprocess.Popen, result: dict[str, Any], stop: threading.Event) -> None:
    peak = 0
    while not stop.wait(0.5):
        try:
            root = psutil.Process(process.pid)
            processes = [root, *root.children(recursive=True)]
            peak = max(peak, sum(item.memory_info().rss for item in processes if item.is_running()))
        except (psutil.Error, OSError):
            pass
    result["peak_process_tree_ram_gb"] = round(peak / 1024**3, 3)


def _run_worker(config: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / f"stress_{config['task']}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "ai_photogrammetry.engineering.worker",
        "--config",
        str(config_path),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
    )
    memory: dict[str, Any] = {}
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor_memory, args=(process, memory, stop), daemon=True)
    monitor.start()
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    peak_gpu = 0.0
    terminal: dict[str, Any] = {}
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(line, flush=True)
            continue
        events.append(event)
        if event.get("type") == "progress":
            print(
                f"[{config['task']}] {event.get('progress', 0):6.2f}% "
                f"{event.get('message', '')}",
                flush=True,
            )
        elif event.get("type") == "telemetry":
            peak_gpu = max(peak_gpu, float(event.get("gpu_memory_used_gb", 0) or 0))
        elif event.get("type") == "result":
            terminal = event
    return_code = process.wait()
    stop.set()
    monitor.join(timeout=3)
    elapsed = time.perf_counter() - started
    return {
        "status": terminal.get("status", "failed"),
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 3),
        "peak_gpu_memory_used_gb": round(peak_gpu, 3),
        **memory,
        "result": terminal.get("result"),
        "error": terminal.get("error"),
        "traceback": terminal.get("traceback"),
        "progress_event_count": sum(event.get("type") == "progress" for event in events),
        "telemetry_event_count": sum(event.get("type") == "telemetry" for event in events),
    }


def _evenly_sample(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    if count <= 1:
        return values[:count]
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--scan-count", type=int, default=500)
    parser.add_argument("--photogrammetry-count", type=int, default=100)
    parser.add_argument("--skip-photogrammetry", action="store_true")
    parser.add_argument("--colmap")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    photos = discover_photos(args.source)
    if len(photos) < args.scan_count:
        raise SystemExit(f"照片不足：找到 {len(photos)}，需要 {args.scan_count}")
    scan_photos = photos[: args.scan_count]
    if (args.project_root / "project.json").is_file():
        store = ProjectStore.open(args.project_root)
    else:
        store = ProjectStore.create(args.project_root, "真实照片压力测试")
    checkpoint_dir = store.root / "cache" / "checkpoints" / "stress"
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source.resolve()),
        "source_photo_count": len(photos),
        "scan_photo_count": len(scan_photos),
        "scan_total_bytes": sum(Path(value).stat().st_size for value in scan_photos),
        "platform": platform.platform(),
        "python": sys.version,
    }

    existing = store.read_manifest()
    existing_scan = existing.get("photo_scan") or {}
    same_sources = [str(value) for value in existing.get("source_images", [])] == scan_photos
    if same_sources and len(existing_scan.get("records", [])) == len(scan_photos):
        report["photo_scan"] = {
            "status": "completed",
            "cached": True,
            "elapsed_seconds": 0,
            "result": existing_scan.get("summary") or {},
        }
        print("[scan_photos] 使用已完成的 500 张扫描缓存", flush=True)
    else:
        report["photo_scan"] = _run_worker(
            {
                "task": "scan_photos",
                "project_root": str(store.root),
                "source_images": scan_photos,
                "max_keyframes": args.scan_count,
                "include_near_duplicates": False,
            },
            checkpoint_dir,
        )
    if report["photo_scan"]["status"] != "completed":
        output = store.root / "stress_report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"扫描压力测试失败，报告：{output}")
        return 1

    manifest = store.read_manifest()
    candidates = [str(value) for value in manifest.get("selected_images", [])]
    photogrammetry_photos = _evenly_sample(candidates, args.photogrammetry_count)
    report["selected_after_scan"] = len(candidates)
    report["photogrammetry_photo_count"] = len(photogrammetry_photos)
    if not args.skip_photogrammetry:
        report["photogrammetry"] = _run_worker(
            {
                "task": "colmap",
                "project_root": str(store.root),
                "image_paths": photogrammetry_photos,
                "output_root": str(args.output_root or (store.root / "colmap")),
                "colmap_path": args.colmap,
                "feature_type": "aliked",
                "matcher": "auto",
                "mapper": "global",
                "camera_model": "SIMPLE_RADIAL",
                "single_camera": True,
                "feature_max_image_size": 4096,
                "max_image_size": 4096,
                "max_num_features": 4096,
                "sequential_overlap": 20,
                "use_gpu": True,
                "resume": True,
            },
            checkpoint_dir,
        )

    output = store.root / "stress_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"压力测试报告：{output}")
    photogrammetry = report.get("photogrammetry")
    return 0 if photogrammetry is None or photogrammetry.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
