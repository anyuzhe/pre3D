[CmdletBinding()]
param(
    [string]$ReleaseDirectory = "",
    [string]$PhotoDirectory = "D:\BaiduNetdiskDownload\点云资源\集合"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (-not $ReleaseDirectory) {
    $ReleaseDirectory = Join-Path $ProjectRoot "dist\RockVision"
}
$ReleaseDirectory = [IO.Path]::GetFullPath($ReleaseDirectory)
$InternalRoot = Join-Path $ReleaseDirectory "_internal"
$ColmapBin = Join-Path $InternalRoot "tools\colmap\bin"
$Colmap = Join-Path $ColmapBin "colmap.exe"
$FeatureModel = Join-Path $InternalRoot "checkpoints\colmap_ai\aliked-n16rot.onnx"
$MatcherModel = Join-Path $InternalRoot "checkpoints\colmap_ai\aliked-lightglue.onnx"
foreach ($required in ($Colmap, $FeatureModel, $MatcherModel)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "发布验证缺少文件：$required"
    }
}

$BuildRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "build"))
$TestRoot = [IO.Path]::GetFullPath((Join-Path $BuildRoot ("gpu_runtime_" + [guid]::NewGuid().ToString("N"))))
$allowedPrefix = $BuildRoot.TrimEnd('\') + '\'
if (-not $TestRoot.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "拒绝使用工作区 build 目录以外的临时路径：$TestRoot"
}
$Images = Join-Path $TestRoot "images"
New-Item -ItemType Directory -Path $Images -Force | Out-Null
$photos = @(
    Get-ChildItem -LiteralPath $PhotoDirectory -File |
        Where-Object { $_.Extension -match '^\.(jpg|jpeg|png)$' } |
        Select-Object -First 2
)
if ($photos.Count -lt 2) {
    throw "GPU 发布验证至少需要两张照片：$PhotoDirectory"
}
$photos | Copy-Item -Destination $Images
$Database = Join-Path $TestRoot "database.db"
$OldPath = $env:PATH
$nativePreference = $ErrorActionPreference

try {
    # Deliberately remove the developer CUDA/Conda paths. Only the packaged
    # runtime DLL directory plus Windows system directories remain.
    $env:PATH = "$ColmapBin;$env:WINDIR\System32;$env:WINDIR"
    $ErrorActionPreference = "Continue"
    $featureOutput = & $Colmap feature_extractor `
        --database_path $Database `
        --image_path $Images `
        --ImageReader.camera_model SIMPLE_RADIAL `
        --ImageReader.single_camera 1 `
        --FeatureExtraction.type ALIKED_N16ROT `
        --FeatureExtraction.use_gpu 1 `
        --FeatureExtraction.max_image_size 1024 `
        --AlikedExtraction.max_num_features 2048 `
        --AlikedExtraction.n16rot_model_path $FeatureModel 2>&1
    $featureExitCode = $LASTEXITCODE
    $ErrorActionPreference = $nativePreference
    if ($featureExitCode -ne 0) {
        $featureOutput | Select-Object -Last 80
        throw "安装包内 ALIKED CUDA 特征提取失败：退出码 $featureExitCode"
    }

    $ErrorActionPreference = "Continue"
    $matchOutput = & $Colmap exhaustive_matcher `
        --database_path $Database `
        --FeatureMatching.type ALIKED_LIGHTGLUE `
        --FeatureMatching.use_gpu 1 `
        --TwoViewGeometry.min_num_inliers 10 `
        --TwoViewGeometry.max_error 4 `
        --AlikedMatching.lightglue_model_path $MatcherModel 2>&1
    $matchExitCode = $LASTEXITCODE
    $ErrorActionPreference = $nativePreference
    if ($matchExitCode -ne 0) {
        $matchOutput | Select-Object -Last 80
        throw "安装包内 LightGlue CUDA 匹配失败：退出码 $matchExitCode"
    }

    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    & $VenvPython -c @"
import sqlite3
from pathlib import Path
database = sqlite3.connect(Path(r'$Database'))
counts = {
    'images': database.execute('select count(*) from images').fetchone()[0],
    'keypoints': database.execute('select count(*) from keypoints').fetchone()[0],
    'matches': database.execute('select count(*) from two_view_geometries').fetchone()[0],
}
assert counts['images'] == 2 and counts['keypoints'] == 2 and counts['matches'] >= 1, counts
print(counts)
"@
    if ($LASTEXITCODE -ne 0) { throw "AI 特征数据库结果检查失败。" }
    Write-Host "安装包内 ALIKED + LightGlue CUDA 运行验证通过。"
} finally {
    $env:PATH = $OldPath
    $ErrorActionPreference = $nativePreference
    if (Test-Path -LiteralPath $TestRoot) {
        $resolvedTestRoot = [IO.Path]::GetFullPath($TestRoot)
        if (-not $resolvedTestRoot.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "拒绝清理工作区 build 目录以外的路径：$resolvedTestRoot"
        }
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
