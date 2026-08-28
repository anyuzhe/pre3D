# pre3D — AI-assisted Photogrammetry Workbench

[简体中文](README.md) | [English](README_EN.md)

[![Windows Build and Release](https://github.com/anyuzhe/pre3D/actions/workflows/windows-release.yml/badge.svg)](https://github.com/anyuzhe/pre3D/actions/workflows/windows-release.yml)
[![Latest Release](https://img.shields.io/github/v/release/anyuzhe/pre3D?display_name=tag)](https://github.com/anyuzhe/pre3D/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4)](docs/Windows安装包构建与发布.md)

pre3D is a native Windows desktop application for reconstructing dense point clouds and textured 3D models from overlapping photographs. It is designed for rock slopes, tunnel faces, drone surveys, building facades, outcrops, and close-range objects. The interface is built with PySide6 and PyVista/VTK; reconstruction runs locally without a browser or a cloud service.

The product combines learned local-feature matching with established multi-view geometry. AI improves correspondences in repetitive or difficult imagery, while COLMAP/GLOMAP estimates the camera model, optimizes geometry, and computes dense depth from the original photographs.

> **Measurement boundary:** a photo-only reconstruction has an arbitrary scale. It is suitable for visualization but not for absolute measurements in metres or centimetres. Use a known scale or distributed control points, and reserve independent check points, before treating a result as an engineering survey.

## What pre3D does

The default workflow is intentionally simple: create a project, add photographs, choose a quality preset and requested outputs, then start one-click processing. Advanced pages remain available for users who need camera, matching, dense-reconstruction, filtering, coordinate, or export controls.

The processing chain is:

```text
Original full-resolution photographs
→ quality checks, duplicate detection, keyframe diagnostics
→ ALIKED or SIFT local features
→ LightGlue image matching and geometric verification
→ cross-flight retrieval, GPS-neighbour matching, and disconnected-group bridging
→ GLOMAP global SfM or COLMAP incremental SfM
→ global bundle adjustment and sparse quality gate
→ image undistortion
→ CUDA PatchMatch Stereo and depth-map fusion
→ dense point-cloud filtering and normal repair
→ Poisson surface reconstruction and mesh repair/simplification
→ UV unwrapping and multi-view texture projection/blending
→ PLY / LAS / OBJ / FBX / glTF / GLB / OSGB results
```

VGG-T³ inference, checkpoints, training/evaluation code, and its direct point-map reconstruction path have been removed. pre3D does not ask a single feed-forward model to produce the final engineering geometry. Instead, learned matching is used where it provides the largest practical benefit, while camera calibration, bundle adjustment, epipolar verification, stereo reconstruction, and fusion retain explicit geometric constraints.

## Main capabilities

- A three-page beginner interface for project setup, one-click reconstruction, and result viewing.
- Native PySide6 desktop UI with interactive PyVista/VTK point-cloud and mesh viewing.
- Photo sharpness and EXIF inspection, exact/near-duplicate detection, sequence continuity checks, and keyframe diagnostics.
- ALIKED-N16Rot + LightGlue as the recommended feature pipeline, with SIFT + LightGlue as a compatibility fallback.
- Automatic disconnected match-graph detection with appearance retrieval, GPS-neighbour candidates, wide-baseline ALIKED fallback matching, and RANSAC verification across capture runs.
- GLOMAP global SfM and COLMAP incremental SfM, including automatic fallback when registration quality is insufficient.
- A sparse quality gate based on registered-image ratio, reprojection quality, camera layout, and point coverage.
- Original-resolution CUDA PatchMatch Stereo with configurable source views, geometric consistency, filtering, and iterations.
- Separate SfM and MVS caches, stage validation, cancellation, worker-process isolation, crash recovery, and resume support.
- GPU-memory monitoring and adaptive source-view reduction on dense reconstruction failures without silently lowering output resolution.
- Spatial Core/Halo MVS blocks for large high-resolution projects; each reference image computes depth once and neighbouring halo images act as source views.
- Memory-mapped preview and processing for very large PLY files.
- Statistical, MAD, radius, and voxel filters; normal repair; surface reconstruction; mesh repair and simplification.
- Multi-atlas/block texture generation to avoid a single 65,536-pixel texture-atlas limit, with per-block recovery.
- Scale calibration, robust control-point coordinate transformation, WGS84-to-local-ENU or selected CRS conversion, and independent check-point reports.
- Dense point cloud, textured mesh, camera/quality reports, CSV/JSON/HTML, and packaged project exports.

## Quality presets

All registered photographs remain MVS reference images in every preset. Presets change resolution and compute effort, not scene coverage.

| Preset | MVS maximum size | Reference images | Intended use |
|---|---:|---:|---|
| Fast preview | 2048 px | 100% of registered images | Confirm connectivity and coverage quickly |
| Standard engineering | 3072 px | 100% of registered images | Recommended balance for routine projects |
| High accuracy | 4096 px | 100% of registered images | Final high-detail output |

Large projects are partitioned after SfM using camera positions and co-visibility. This reduces peak memory, disk pressure, and the cost of restarting a failed dense stage while preserving overlap between blocks.

## Windows installation

End users should download the latest installer from [GitHub Releases](https://github.com/anyuzhe/pre3D/releases/latest), run the `.exe`, choose an install directory, and launch **岩土影像三维重建工作台** from the Start menu. The packaged release includes Python, PySide6/VTK, COLMAP, learned matching models, and the required CUDA/cuDNN runtime libraries. A separate Python, PyTorch, CUDA Toolkit, or developer environment is not required.

A compatible NVIDIA GPU and up-to-date NVIDIA display driver are required for the full CUDA dense-reconstruction pipeline. CPU, memory, free disk space, image count, and output resolution all affect processing time and project capacity.

## Run from source

Requirements:

- Windows 10/11 x64;
- Python 3.11 or newer;
- a compatible NVIDIA GPU and driver for dense reconstruction;
- COLMAP 4.1.1 CUDA build;
- optional OpenSceneGraph for OSGB export.

Prepare the Python environment:

```powershell
.\scripts\setup.ps1
```

Install the external runtime and matching models:

```powershell
.\scripts\install_colmap.ps1
.\scripts\download_colmap_ai_models.ps1
```

Start the application:

```powershell
.\scripts\start.ps1
```

or:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Build the Windows installer

The repository contains a reproducible PyInstaller onedir build and an Inno Setup installer definition:

```powershell
.\scripts\build_release.ps1
```

Outputs are written to `dist/RockVision` and `release/`. The builder verifies the frozen GUI entry point, worker-process protocol, COLMAP executable, AI models, and CUDA/cuDNN runtime files before producing a SHA-256 checksum.

GitHub Actions performs the same process on `windows-2022`. Every push to `main` builds and tests a Windows installer artifact. A `v*` tag additionally creates a GitHub Release and attaches the installer and checksum. See the [Windows build and release guide](docs/Windows安装包构建与发布.md) for details.

## Cache and resume layout

Each distinct photo set and parameter configuration receives an isolated workspace:

```text
photogrammetry_<project>_<photos>_<sparse-options>/
├── images/
├── database.db
├── sparse_mapped/
├── sparse_ba/
├── pipeline_state.json
└── dense_<dense-options>/
    ├── images/
    ├── sparse/
    ├── stereo/
    ├── fused.ply
    ├── pointcloud_ai_photogrammetry.ply
    └── model_<model-options>/
        ├── 01_conditioned_points.ply
        ├── 02_surface_raw.ply
        ├── 03_mesh_repaired.ply
        ├── 04_mesh_simplified.ply
        ├── 05_textured/
        └── 06_exports/
```

Completed stages are reused only after their inputs and outputs pass validation. Changing dense settings reuses existing features, matches, SfM, and bundle adjustment. Adding only a scale or check points does not restart PatchMatch.

## Capture guidance

Good reconstruction still depends on observable geometry:

- keep strong forward and side overlap between neighbouring photographs;
- include oblique views and a useful baseline instead of moving along only one straight line;
- keep focus and focal length stable, avoid digital zoom, motion blur, and abrupt exposure changes;
- cover top, bottom, sides, occlusions, and turning surfaces;
- avoid large moving objects, reflective surfaces, featureless sky, and repeated frames;
- form loops on long slopes or tunnels to reduce accumulated drift;
- preserve original files and EXIF metadata.

AI matching can recover more correspondences than a basic matcher, but it cannot reconstruct a surface that was never photographed or recover reliable geometry from nearly identical camera positions.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe app.py --smoke-test
```

The repository also includes a real-photo stress-test driver for 100–500 image projects:

```powershell
.\.venv\Scripts\python.exe scripts\stress_test.py `
  --source "D:\photo_dataset" `
  --project-root "D:\photogrammetry_stress" `
  --scan-count 500 `
  --photogrammetry-count 100 `
  --output-root "D:\photogrammetry_stress_work"
```

## Source layout

- `ai_photogrammetry/engineering/desktop.py` — beginner and advanced PySide6 UI.
- `ai_photogrammetry/engineering/worker.py` — isolated scan and reconstruction workers.
- `ai_photogrammetry/engineering/colmap_pipeline.py` — features, matching, SfM, BA, and dense MVS.
- `ai_photogrammetry/engineering/model_pipeline.py` — point conditioning, meshes, textures, and formats.
- `ai_photogrammetry/engineering/photo_selection.py` — photo quality, duplicates, and keyframes.
- `ai_photogrammetry/engineering/spatial_blocks.py` — co-visible Core/Halo MVS planning.
- `ai_photogrammetry/engineering/calibration.py` — scale and engineering-coordinate transforms.
- `ai_photogrammetry/engineering/exporters.py` — deliverables and accuracy reports.
- `packaging/` — PyInstaller and Inno Setup definitions.
- `scripts/` — environment, runtime, build, verification, smoke, and stress tools.
- `tests/` — automated unit and workflow tests.

## Third-party components and licensing

This repository currently does not grant a separate license for its own source code; the project owner must choose the distribution terms. ALIKED, LightGlue, COLMAP, PySide6/Qt, VTK/PyVista, CUDA/cuDNN, and optional OpenSceneGraph remain subject to their own licenses and redistribution conditions. Review [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) before commercial distribution.

The Windows installer is a distribution artifact, not a transfer of ownership or a replacement for upstream license obligations.
