# pre3D — AI 摄影测量工程三维重建工作台

[简体中文](README.md) | [English](README_EN.md)

[![Windows Build and Release](https://github.com/anyuzhe/pre3D/actions/workflows/windows-release.yml/badge.svg)](https://github.com/anyuzhe/pre3D/actions/workflows/windows-release.yml)
[![Latest Release](https://img.shields.io/github/v/release/anyuzhe/pre3D?display_name=tag)](https://github.com/anyuzhe/pre3D/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4)](docs/Windows安装包构建与发布.md)

本项目是本机运行的 PySide6 + PyVista/VTK 桌面软件。当前唯一重建主线是：

```text
原始高分辨率照片
→ 照片质检、去重、关键帧筛选
→ ALIKED / SIFT 提取局部特征
→ LightGlue 匹配照片
→ 跨航线图像检索、GPS邻域匹配与分离照片组自动桥接
→ GLOMAP 全局 SfM 或 COLMAP 增量 SfM
→ Bundle Adjustment
→ 空三质量闸门
→ 全部注册照片作为 MVS 参考帧
→ 大项目按空三共视关系自动规划 Core/Halo 空间块
→ 原图去畸变
→ 逐块 CUDA PatchMatch Stereo、Core裁切与全局融合
→ 点云去噪与法向修复
→ 泊松表面重建、网格修补与简化
→ UV展开、原始照片纹理投影与多照片接缝融合
→ OBJ / FBX / glTF / GLB / OSGB
```

VGG‑T³ 推理、权重、下载器、训练/评测源码、旧点图重建入口和旧初始化分支均已从产品中移除。AI 负责增强照片匹配；最终相机几何和稠密点云由传统多视几何计算。

pre3D 面向边坡、岩壁、隧道、无人机测绘、建筑外立面和近景物体等照片三维重建任务。普通用户只需导入一组具有连续重叠的照片、选择成果精度与成果类型，软件便会在本地自动完成照片质检、相机求解、稠密点云和可选纹理网格。所有照片和工程成果均保留在用户电脑中，不依赖 Web 服务。

> **测量边界：** 未使用标尺或控制点时，成果处于任意尺度，只适合形态浏览，不应直接用于米、厘米等绝对量测。工程测量应加入分布合理的控制点，并使用独立检查点验证误差。

## 当前功能

- 默认“小白模式”：只显示项目、一键重建、成果三个页面；
- 导入照片、选择三档成果精度和成果类型后，一键完成点云或纹理模型；
- 默认先用 ALIKED + LightGlue + GLOMAP，照片注册不足时自动回退到
  SIFT + LightGlue + COLMAP 增量 SfM；
- 专业页面和参数保留在“高级功能”菜单中，不干扰普通用户；
- 工程新建、保存、关闭、重新打开；
- 照片清晰度/EXIF 检查、完全重复与近重复检测；
- 拍摄连续性诊断、关键帧筛选；
- ALIKED‑N16Rot + LightGlue 推荐模式；
- SIFT + LightGlue 兼容备选；
- 全连接、顺序和自动匹配；
- 自动检测分离照片组，并用外观检索、GPS邻域、ALIKED宽基线匹配和RANSAC几何验证进行跨航线桥接；
- GLOMAP 全局 SfM、COLMAP 增量 SfM；
- 相机模型、共享内参、特征数和原图 MVS 分辨率设置；
- AI 特征、匹配、SfM、BA、去畸变、PatchMatch、融合逐阶段缓存；
- SfM 与 MVS 分离缓存；修改稠密参数不会重新提取特征或重跑空三；
- 全部合格照片参加 SfM，全部注册照片均生成 MVS 深度图；
- 大图节省磁盘模式：几何深度完成后自动释放光度中间图，最终PLY使用硬链接；
- PatchMatch显存不足时保持分辨率不变，自动减少单张参考图的源照片数后续跑；
- 超大PLY按分布式数据块读取预览，不再把整份点云一次性载入内存；
- 一键处理前可选择“仅稠密点云”或“稠密点云＋纹理三维模型”；
- 大点云内存映射抽稀、统计离群清理、法向归一化与异常法向修复；
- 泊松表面重建、退化网格清理、小孔修补、统一法向和边界保护简化；
- 基于全部去畸变原图的面片选图、颜色校正、接缝平滑和空白修补；
- 大网格按空间递归分块并生成独立多纹理图集，限制单图集尺寸且支持逐块失败重试；
- 纹理模型PLY预览，以及OBJ、FBX、glTF、GLB、OSGB格式写出；
- 模型各阶段独立缓存，崩溃或取消后可从有效阶段继续；
- 取消任务、独立子进程、崩溃隔离和 GPU 显存显示；
- 稠密点云三维浏览与拾点；
- 标尺恢复米制尺度，控制点RANSAC加权转换工程坐标，独立检查点残差报告；
- WGS84经纬高转Local ENU或指定EPSG/CRS，并把CRS写入LAS；
- 大项目按空三共视关系自动执行Core/Halo空间分块MVS；每张注册照片只生成一次深度图，Halo仅作邻块源图；
- 空间块独立融合、Core裁切、全局合并和逐块断点恢复；
- MAD、统计离群、半径离群、体素降采样；
- PLY、LAS、OBJ、FBX、glTF/GLB、OSGB、CSV、JSON、HTML 和 ZIP 成果包。

## 启动

```powershell
.\scripts\setup.ps1
.\scripts\start.ps1
```

或：

```powershell
.\.venv\Scripts\python.exe app.py
```

程序不会启动 Web 服务。

## Windows 安装包

项目已提供 PyInstaller 目录版与 Inno Setup 安装包流程。开发机准备好
COLMAP、AI 权重、CUDA/cuDNN 运行库和 Inno Setup 6 后，运行：

```powershell
.\scripts\build_release.ps1
```

安装包输出到 `release`。它包含 Python、PySide6/VTK、COLMAP、AI 模型及
CUDA/cuDNN 运行库；最终用户只需安装兼容的 NVIDIA 显卡驱动。完整说明见
[`docs/Windows安装包构建与发布.md`](docs/Windows安装包构建与发布.md)。

## 推荐操作顺序

普通用户只需要：

1. 在“项目”页填写项目名称、保存位置和成果精度。
2. 在“一键重建”页添加照片或照片文件夹。
3. 在“选择需要的成果”中保留稠密点云，按需要勾选“纹理三维模型”和格式。
4. 点击“开始一键处理”，等待软件完成点云；勾选模型后会继续完成网格和纹理。
5. 在“成果”页切换查看点云/纹理模型并导出成果包。

三档精度只控制计算量和细节，不减少 MVS 参考照片：

| 模式 | MVS 最大尺寸 | 参考照片 | 适用场景 |
|---|---:|---:|---|
| 快速预览 | 2048 px | 全部注册照片 | 快速确认能否形成完整点云 |
| 标准工程模式 | 3072 px | 全部注册照片 | 默认模式，兼顾速度与完整性 |
| 高精度模式 | 4096 px | 全部注册照片 | 终版高细节成果 |

软件会先走推荐的 ALIKED + LightGlue + GLOMAP 路线；若空三质量不合格或
注册率低于 90%，会自动改用 SIFT + LightGlue + COLMAP 增量 SfM 再尝试一次。
两次都失败时才停止，并提示补拍重叠照片或检查镜头切换。

需要查看空三、调整算法参数、添加标尺/控制点或执行高级过滤时，在菜单中选择
“高级功能 → 显示专业页面与参数”。

## 模型与外部程序

AI 匹配模型存放在：

```text
checkpoints/colmap_ai/
├── aliked-n16rot.onnx
├── aliked-lightglue.onnx
├── bruteforce-matcher.onnx
└── sift-lightglue.onnx
```

软件会校验四个模型的 SHA‑256，缺失或损坏时拒绝启动对应的AI匹配阶段。下载命令：

```powershell
.\scripts\download_colmap_ai_models.ps1
```

重建需要支持相应 AI 特征类型、ONNX Runtime、GLOMAP 和 CUDA MVS 的 COLMAP 版本。程序可以自动查找，也可以在界面中指定 `colmap.exe`。

OSGB由真实的OpenSceneGraph `osgconv` 写出。本机运行时安装命令：

```powershell
.\scripts\install_openscenegraph.ps1
```

未检测到转换器时，界面会禁用OSGB选项，不会用改扩展名的文件冒充OSGB。

## 缓存与断点恢复

每组照片和参数使用独立工作目录：

```text
photogrammetry_<project>_<photos>_<sparse-options>/
├── images/
├── database.db
├── sparse_mapped/
├── sparse_ba/
├── sparse_preview/
├── pipeline_state.json
├── sparse_pipeline_config.json
├── ai_runtime.json
├── mapping.json
├── mvs_reference_selection_<dense-options>.json
└── dense_<dense-options>/
    ├── images/
    ├── sparse/
    ├── stereo/
    ├── pipeline_state.json
    ├── fused.ply
    ├── pointcloud_output.json
    ├── pointcloud_ai_photogrammetry.ply
    └── model_<model-options>/
        ├── 01_conditioned_points.ply
        ├── 02_surface_raw.ply
        ├── 03_mesh_repaired.ply
        ├── 04_mesh_simplified.ply
        ├── 05_textured/mesh.ply
        ├── 05_textured/texture.png
        ├── 06_exports/
        └── pipeline_state.json
```

启用“从已完成阶段继续”后，软件验证输入和阶段成果，只跳过已完成且有效的阶段。输入照片或关键参数改变时会使用新的缓存目录，不会把旧数据库混入当前重建。
只修改 MVS 分辨率、几何一致性、深度过滤、源照片数或迭代次数时，
软件会复用已有特征、匹配、SfM 和 BA，并建立单独的稠密缓存。因此可先跑快速
预览，再切换标准工程或高精度模式生成终版，而不重复空三。

几何一致性模式在PatchMatch完成后只保留融合必需的几何深度和法向图，自动删除
同尺寸的光度中间图。磁盘预检按各阶段峰值而不是把顺序阶段全部相加；对于
数百张4096像素照片，这能显著降低长期磁盘占用。COLMAP内存缓存也会根据
当前可用内存自动限制在4～16GB，给桌面界面和系统保留空间。

## 测量边界

仅照片得到的 SfM/MVS 点云仍是任意尺度。没有标尺或控制点时，界面只显示“模型单位”，禁止输出米、平方米或立方米。一个可靠已知距离可恢复统一比例；工程坐标应使用分布良好的控制点，并保留独立检查点评价绝对误差。

## 详细说明

完整的算法阶段、各模式选择、参数建议、VGG‑T³ 被移除的原因和产品路线分析见：

[AI摄影测量新模式技术与产品说明](docs/AI摄影测量新模式技术与产品说明.md)

默认三页的一键界面和高级功能操作见：

[第一版产品操作说明](docs/第一版产品操作说明.md)

模型生成、三档参数、纹理融合、格式用途和缓存说明见：

[纹理三维模型功能说明](docs/纹理三维模型功能说明.md)

现场拍摄仍需连续重叠、清晰成像、稳定焦距和多角度基线。AI 匹配能提高困难照片的对应成功率，但不能恢复没有拍到的表面，也不能替代可靠的几何约束。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe app.py --smoke-test
```

真实照片压力测试：

```powershell
.\.venv\Scripts\python.exe scripts\stress_test.py `
  --source "D:\BaiduNetdiskDownload\点云资源\集合" `
  --project-root "D:\photogrammetry_stress" `
  --scan-count 500 `
  --photogrammetry-count 100 `
  --output-root "D:\photogrammetry_stress_work"
```

## 主要代码

- `ai_photogrammetry/engineering/desktop.py`：PySide6 默认三页、可展开六页的产品界面；
- `ai_photogrammetry/engineering/worker.py`：照片扫描、摄影测量和过滤独立进程；
- `ai_photogrammetry/engineering/colmap_pipeline.py`：AI 特征、匹配、SfM、BA 与原图 MVS；
- `ai_photogrammetry/engineering/model_pipeline.py`：点云清理、表面网格、纹理图集与模型格式；
- `ai_photogrammetry/engineering/photo_selection.py`：照片质检、去重和关键帧；
- `ai_photogrammetry/engineering/project_store.py`：工程和阶段缓存；
- `ai_photogrammetry/engineering/point_processing.py`：点云过滤与降采样；
- `ai_photogrammetry/engineering/calibration.py`：尺度和工程坐标变换；
- `ai_photogrammetry/engineering/exporters.py`：成果包与精度报告。

## 第三方许可

本项目自身代码的分发许可由项目所有者决定。ALIKED、LightGlue、COLMAP、Qt/PySide6、VTK/PyVista 等第三方组件分别受各自许可约束，详见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。正式商业发布前应由产品方完成依赖、二进制和模型权重的许可复核。
