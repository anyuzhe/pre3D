# Third-party components and models

The former NVIDIA VGG‑T³ source code, checkpoint integration, and its
noncommercial license file have been removed from this product.

This repository uses or interoperates with the following independently licensed
components. This list is a practical inventory, not legal advice and not a
substitute for the complete license texts distributed by each upstream project.

| Component | Upstream | License noted by upstream |
|---|---|---|
| ALIKED and ALIKED weights | https://github.com/Shiaoming/ALIKED | BSD-3-Clause |
| LightGlue and LightGlue weights | https://github.com/cvg/LightGlue | Apache-2.0 |
| COLMAP / integrated global mapper | https://github.com/colmap/colmap | BSD-3-Clause |
| OpenSceneGraph / osgconv (optional OSGB export runtime) | https://github.com/openscenegraph/OpenSceneGraph | LGPL-2.1-only in the conda-forge package used here |
| PySide6 / Qt | https://doc.qt.io/qtforpython-6/licenses.html | LGPLv3/GPLv3 or commercial, depending on distribution |
| VTK | https://gitlab.kitware.com/vtk/vtk | BSD-3-Clause |
| PyVista | https://github.com/pyvista/pyvista | MIT |
| trimesh | https://github.com/mikedh/trimesh | MIT |
| pyproj / PROJ | https://pyproj4.github.io/pyproj/ / https://proj.org/ | MIT |
| NumPy, SciPy, Pillow, OpenCV, laspy, psutil | Their respective upstream distributions | See each installed package |
| NVIDIA CUDA runtime libraries (binary release only) | https://developer.nvidia.com/cuda-toolkit | NVIDIA CUDA Toolkit EULA |
| NVIDIA cuDNN runtime libraries (binary release only) | https://developer.nvidia.com/cudnn | NVIDIA cuDNN Software License Agreement |

Before a commercial binary release, the product owner should archive the exact
source/version/weight hashes used in that build, reproduce all required notices,
and review Qt/PySide6/OpenSceneGraph distribution obligations and the license
of the specific COLMAP binary bundle.

The Windows release builder copies only the CUDA/cuDNN runtime DLLs required
to let end users run the application with a compatible NVIDIA driver and no
separate CUDA Toolkit/PyTorch installation. The builder also copies the local
CUDA EULA when it is available. The party distributing the installer remains
responsible for confirming that every selected runtime file is redistributable
under the NVIDIA terms applicable to that release.
