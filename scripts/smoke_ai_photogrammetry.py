"""Run the learned-feature photogrammetry pipeline on a real project subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_photogrammetry.engineering.colmap_pipeline import find_colmap, run_colmap_ba_mvs
from ai_photogrammetry.engineering.project_store import ProjectStore


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--target-stage",
        choices=["sparse", "dense"],
        default="dense",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    project_path = (
        args.project.resolve()
        if args.project.is_absolute()
        else repository_root / args.project
    )
    store = ProjectStore.open(project_path)
    manifest = store.read_manifest()
    images = [
        str(Path(value).resolve())
        for value in (
            manifest.get("selected_images")
            or manifest.get("source_images")
            or []
        )
    ][: max(3, args.count)]
    if len(images) < 3:
        raise RuntimeError("工程中没有足够的真实照片")
    session = store.load_session()
    output_root = (
        (
            args.output_root.resolve()
            if args.output_root.is_absolute()
            else repository_root / args.output_root
        )
        if args.output_root
        else store.root / "colmap_smoke"
    )
    result = run_colmap_ba_mvs(
        session,
        colmap_path=find_colmap(),
        output_root=output_root,
        source_image_paths=images,
        feature_type="aliked",
        matcher="exhaustive",
        mapper="global",
        feature_max_image_size=args.max_size,
        max_image_size=args.max_size,
        max_num_features=args.features,
        target_stage=args.target_stage,
        use_gpu=True,
        resume=True,
        progress_callback=lambda value, text: print(
            f"{value:6.1%} {text}",
            flush=True,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if int(result["registered_images"]) != len(images):
        return 2
    point_key = "sparse_point_count" if args.target_stage == "sparse" else "point_count"
    if int(result[point_key]) <= 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
