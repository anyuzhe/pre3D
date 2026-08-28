# Windows 安装包构建与发布

本项目使用 PyInstaller 生成 Windows 目录版程序，再由 Inno Setup 制作普通用户可双击安装的安装包。发布包会带上 Python、PySide6、VTK、COLMAP、三套 AI 特征模型以及所需 CUDA/cuDNN 运行库。最终用户不需要安装 Python、PyTorch 或 CUDA Toolkit，只需要安装兼容的 NVIDIA 显卡驱动。

## 一、开发机准备

开发机需要：

- Windows 10/11 x64；
- 项目已经通过 `scripts/setup.ps1` 建立 `.venv`；
- `tools/colmap` 中已有本项目使用的 CUDA/ONNX 版 COLMAP；
- `checkpoints/colmap_ai` 中已有 ALIKED、ALIKED-LightGlue、SIFT-LightGlue 三个 ONNX 文件；
- CUDA 12 运行库可从 `CUDA_PATH\bin` 找到；
- cuDNN 9 运行库可从 `CUDNN_PATH`、当前 Conda 环境或某个 Conda 环境的 `torch\lib` 找到；
- Inno Setup 6。

安装 Inno Setup：

```powershell
winget install JRSoftware.InnoSetup
```

## 二、一条命令生成安装包

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_release.ps1
```

构建脚本会依次执行：

1. 读取 `pyproject.toml` 版本号；
2. 清理本项目内的旧构建目录；
3. 自动收集 CUDA、cuDNN 和 zlib 运行 DLL；
4. 使用 PyInstaller 生成 `dist\RockVision` 目录版；
5. 启动冻结版 GUI 做冒烟测试；
6. 启动冻结版后台 worker 并验证 JSON 通信；
7. 检查 COLMAP、AI 权重及 GPU DLL 是否完整；
8. 使用 Inno Setup 生成安装包；
9. 在 `release` 中生成安装包和 `SHA256SUMS.txt`。

如果自动找不到 cuDNN，可明确指定：

```powershell
.\scripts\build_release.ps1 `
  -CudaBin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" `
  -CudnnBin "D:\anaconda3\envs\sam3d\Lib\site-packages\torch\lib"
```

只测试 PyInstaller 目录版、不制作安装器：

```powershell
.\scripts\build_release.ps1 -SkipInstaller
```

正式发布不要使用 `-AllowExternalCudaRuntime`。该开关只用于开发排错，会使目标电脑可能仍需自行安装 CUDA/cuDNN。

## 三、用户如何安装

用户收到 `岩土影像三维重建工作台-版本号-Windows-x64-安装包.exe` 后：

1. 先安装或更新 NVIDIA 官方显卡驱动；
2. 双击安装包并允许 Windows 管理员授权；
3. 选择安装目录，可选创建桌面快捷方式；
4. 安装结束后勾选“启动岩土影像三维重建工作台”；
5. 在软件中创建项目、添加照片并开始一键重建。

用户不需要安装 Python、PyTorch、CUDA Toolkit、COLMAP 或 Inno Setup。项目和成果默认保存在用户“文档\岩创科技\岩土影像三维重建工作台”下，不写入 `Program Files`。日志保存在本地应用数据目录。卸载软件时这些项目和成果会保留。

## 四、升级、卸载与空间要求

- 相同 `AppId` 的新版本安装包会覆盖升级原安装，用户项目不会被删除；
- 可从 Windows“已安装的应用”正常卸载；
- 安装目录需要预留数 GB 空间，实际大小以构建产物为准；
- 高分辨率摄影测量工作目录通常需要照片总容量数倍到十数倍的可用 SSD 空间，这部分不属于安装包占用。

## 五、正式发布检查

发布前至少完成：

- 在一台未安装 Python/CUDA Toolkit 的干净 Windows 电脑测试安装、启动和卸载；
- 在兼容 NVIDIA 驱动的电脑上跑一套 ALIKED/LightGlue、SfM、PatchMatch、融合和模型生成；
- 核对安装包 SHA-256；
- 审核并随包保留第三方许可和 NVIDIA 再分发条款；
- 如对外商业发布，为安装包和主程序配置可信代码签名证书，以减少 SmartScreen 提示。

## 六、GitHub Actions 自动发布

仓库的 `.github/workflows/windows-release.yml` 会在 GitHub 提供的
`windows-2022` 环境中重新建立干净的 Python 3.11 虚拟环境，运行全部测试，
下载并校验官方 COLMAP CUDA 版与三套 AI 匹配权重，再从固定版本的 PyTorch
Windows CUDA wheel 中提取需要随包分发的 CUDA/cuDNN 运行 DLL。PyTorch 本体
不会进入最终安装包。

- 推送到 `main`：构建并上传保留 14 天的 Windows 安装包 Actions Artifact；
- 推送与 `pyproject.toml` 版本一致的 `v*` 标签：完成相同构建，并创建 GitHub
  Release，附加安装包和 `SHA256SUMS.txt`；
- 手工运行：可在 GitHub 的 Actions 页面选择 “Windows Build and Release” 后
  点击 “Run workflow”，只生成测试用 Artifact，不创建正式 Release。

例如发布 `0.3.0`：

```powershell
git tag -a v0.3.0 -m "pre3D 0.3.0"
git push origin v0.3.0
```

标签版本与 `pyproject.toml` 不一致时，流水线会主动失败，避免安装包名称、程序
版本和 GitHub Release 标签相互矛盾。
