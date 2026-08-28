"""Verify the structure and both entry modes of a frozen release."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


REQUIRED_RESOURCES = (
    "_internal/tools/colmap/bin/colmap.exe",
    "_internal/tools/colmap/bin/onnxruntime_providers_cuda.dll",
    "_internal/checkpoints/colmap_ai/aliked-n16rot.onnx",
    "_internal/checkpoints/colmap_ai/aliked-lightglue.onnx",
    "_internal/checkpoints/colmap_ai/sift-lightglue.onnx",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    release_dir = args.release_dir.resolve()
    executable = release_dir / "岩土影像三维重建工作台.exe"
    missing = [value for value in REQUIRED_RESOURCES if not (release_dir / value).is_file()]
    cuda_dir = release_dir / "_internal" / "tools" / "colmap" / "bin"
    if not list(cuda_dir.glob("cudart64_*.dll")):
        missing.append("_internal/tools/colmap/bin/cudart64_*.dll")
    if not list(cuda_dir.glob("cudnn64_*.dll")):
        missing.append("_internal/tools/colmap/bin/cudnn64_*.dll")
    if not executable.is_file():
        missing.insert(0, executable.name)
    if missing:
        raise SystemExit("发布目录缺少必要文件：\n- " + "\n- ".join(missing))

    smoke = subprocess.run(
        [str(executable), "--smoke-test"],
        cwd=release_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if smoke.returncode != 0:
        raise SystemExit(
            f"GUI 冻结版冒烟测试失败：退出码 {smoke.returncode}，"
            f"stdout={smoke.stdout[-1000:]!r}，stderr={smoke.stderr[-1000:]!r}"
        )

    with tempfile.TemporaryDirectory(prefix="rockvision_release_check_") as temporary:
        config_path = Path(temporary) / "worker.json"
        config_path.write_text(
            json.dumps({"task": "release_self_test"}, ensure_ascii=False),
            encoding="utf-8",
        )
        worker = subprocess.run(
            [str(executable), "--worker", "--config", str(config_path)],
            cwd=release_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    events = []
    for line in worker.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if worker.returncode != 1 or not any(
        event.get("type") == "result" and event.get("status") == "failed"
        for event in events
    ):
        raise SystemExit(
            "冻结版后台任务通信测试失败。"
            f"退出码={worker.returncode}，stdout={worker.stdout[-1000:]!r}，"
            f"stderr={worker.stderr[-1000:]!r}"
        )
    print("冻结版 GUI、后台 worker 和发布资源检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
