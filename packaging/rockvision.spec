# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for the RockVision photogrammetry workstation."""

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(os.environ.get("ROCKVISION_PROJECT_ROOT", Path.cwd())).resolve()
RUNTIME_STAGE = Path(
    os.environ.get("ROCKVISION_RUNTIME_STAGE", ROOT / "build" / "release_runtime")
).resolve()
VERSION_FILE = Path(
    os.environ.get("ROCKVISION_VERSION_FILE", ROOT / "build" / "version_info.txt")
).resolve()


def add_tree(
    target: list[tuple[str, str]],
    source: Path,
    destination: str,
    *,
    skip_colmap_tests: bool = False,
) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if skip_colmap_tests and (
            path.name.lower().endswith("_test.exe")
            or path.name.lower() == "run_tests.bat"
        ):
            continue
        relative_parent = path.relative_to(source).parent
        target.append((str(path), str(Path(destination) / relative_parent)))


datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []
external_resources: list[Tree] = []

add_tree(datas, ROOT / "checkpoints" / "colmap_ai", "checkpoints/colmap_ai")
add_tree(datas, ROOT / "docs", "docs")
if (ROOT / "tools" / "colmap").is_dir():
    external_resources.append(
        Tree(
            str(ROOT / "tools" / "colmap"),
            prefix="tools/colmap",
            excludes=["*.exe", "RUN_TESTS.bat"],
        )
    )
    external_resources.append(
        [
            (
                "tools/colmap/bin/colmap.exe",
                str(ROOT / "tools" / "colmap" / "bin" / "colmap.exe"),
                "DATA",
            )
        ]
    )
if (ROOT / "tools" / "openscenegraph").is_dir():
    external_resources.append(
        Tree(str(ROOT / "tools" / "openscenegraph"), prefix="tools/openscenegraph")
    )
if (RUNTIME_STAGE / "cuda").is_dir():
    external_resources.append(
        Tree(str(RUNTIME_STAGE / "cuda"), prefix="tools/colmap/bin")
    )
if (RUNTIME_STAGE / "licenses").is_dir():
    external_resources.append(
        Tree(str(RUNTIME_STAGE / "licenses"), prefix="licenses")
    )

for filename in ("README.md", "README_EN.md", "THIRD_PARTY_LICENSES.md"):
    path = ROOT / filename
    if path.is_file():
        datas.append((str(path), "."))

for package in ("pyvista", "pyvistaqt", "trimesh", "pyproj"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += collect_submodules("laspy")

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "jupyter", "pytest", "torch"],
    noarchive=False,
    optimize=1,
)
# Qt 6 uses the Windows system ICU API. A Conda installation earlier on PATH
# can make PyInstaller collect an incompatible legacy, unversioned ICU DLL;
# that file would shadow System32 and make PySide6.QtCore fail with WinError
# 127. Versioned ICU DLLs used by COLMAP remain isolated under tools/colmap.
a.binaries = [
    entry
    for entry in a.binaries
    if Path(entry[0]).name.casefold() not in {"icuuc.dll", "icudt58.dll"}
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="岩土影像三维重建工作台",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    version=str(VERSION_FILE) if VERSION_FILE.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    *external_resources,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RockVision",
)
