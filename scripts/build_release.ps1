[CmdletBinding()]
param(
    [string]$Version = "",
    [string]$CudaBin = "",
    [string]$CudnnBin = "",
    [string]$InnoSetupPath = "",
    [switch]$SkipInstaller,
    [switch]$AllowExternalCudaRuntime,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BuildRoot = Join-Path $ProjectRoot "build"
$RuntimeStage = Join-Path $BuildRoot "release_runtime"
$VersionFile = Join-Path $BuildRoot "version_info.txt"
$DistRoot = Join-Path $ProjectRoot "dist"
$ReleaseRoot = Join-Path $ProjectRoot "release"
$SpecPath = Join-Path $ProjectRoot "packaging\rockvision.spec"
$InstallerScript = Join-Path $ProjectRoot "packaging\installer.iss"

function Assert-WorkspaceChild([string]$PathValue) {
    $resolved = [System.IO.Path]::GetFullPath($PathValue)
    $prefix = $ProjectRoot.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝清理工作区以外的路径：$resolved"
    }
}

function Reset-Directory([string]$PathValue) {
    Assert-WorkspaceChild $PathValue
    if (Test-Path -LiteralPath $PathValue) {
        Remove-Item -LiteralPath $PathValue -Recurse -Force
    }
    New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
}

function Add-SearchDirectory([System.Collections.Generic.List[string]]$List, [string]$Value) {
    if (-not $Value) { return }
    $candidate = [System.IO.Path]::GetFullPath($Value)
    if ((Test-Path -LiteralPath $candidate -PathType Container) -and -not $List.Contains($candidate)) {
        $List.Add($candidate)
    }
}

function Copy-FirstPattern(
    [string]$Pattern,
    [System.Collections.Generic.List[string]]$SearchDirectories,
    [string]$Destination,
    [bool]$Required = $true
) {
    foreach ($directory in $SearchDirectories) {
        $matches = @(Get-ChildItem -LiteralPath $directory -Filter $Pattern -File -ErrorAction SilentlyContinue)
        if ($matches.Count -gt 0) {
            foreach ($match in $matches) {
                Copy-Item -LiteralPath $match.FullName -Destination $Destination -Force
            }
            return $matches.Count
        }
    }
    if ($Required -and -not $AllowExternalCudaRuntime) {
        throw "缺少发布所需运行库：$Pattern。请通过 -CudaBin 或 -CudnnBin 指定目录。"
    }
    Write-Warning "未打包 $Pattern，目标电脑可能仍需安装 CUDA/cuDNN。"
    return 0
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "未找到 .venv。请先运行 scripts\setup.ps1。"
}
if (-not $Version) {
    $pyproject = Get-Content -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -Raw -Encoding UTF8
    $match = [regex]::Match($pyproject, '(?m)^version\s*=\s*"([^"]+)"')
    if (-not $match.Success) { throw "无法从 pyproject.toml 读取版本号。" }
    $Version = $match.Groups[1].Value
}
if ($Version -notmatch '^\d+\.\d+\.\d+(\.\d+)?$') {
    throw "Inno Setup 版本号必须是数字形式，例如 0.2.0：$Version"
}

if (-not $NoClean) {
    Reset-Directory $RuntimeStage
    Reset-Directory $DistRoot
    Reset-Directory $ReleaseRoot
} else {
    New-Item -ItemType Directory -Path $RuntimeStage,$DistRoot,$ReleaseRoot -Force | Out-Null
}
$CudaStage = Join-Path $RuntimeStage "cuda"
$LicenseStage = Join-Path $RuntimeStage "licenses"
New-Item -ItemType Directory -Path $CudaStage,$LicenseStage -Force | Out-Null

$cudaSearch = [System.Collections.Generic.List[string]]::new()
if ($CudaBin) { Add-SearchDirectory $cudaSearch $CudaBin }
Add-SearchDirectory $cudaSearch (Join-Path $ProjectRoot "tools\colmap\bin")
Add-SearchDirectory $cudaSearch (Join-Path $ProjectRoot ".venv\Lib\site-packages\torch\lib")
if ($env:CUDA_PATH) { Add-SearchDirectory $cudaSearch (Join-Path $env:CUDA_PATH "bin") }
if ($env:CUDA_PATH_V12_9) { Add-SearchDirectory $cudaSearch (Join-Path $env:CUDA_PATH_V12_9 "bin") }

$cudnnSearch = [System.Collections.Generic.List[string]]::new()
if ($CudnnBin) { Add-SearchDirectory $cudnnSearch $CudnnBin }
Add-SearchDirectory $cudnnSearch (Join-Path $ProjectRoot "tools\colmap\bin")
if ($env:CUDNN_PATH) {
    Add-SearchDirectory $cudnnSearch $env:CUDNN_PATH
    Add-SearchDirectory $cudnnSearch (Join-Path $env:CUDNN_PATH "bin")
}
if ($env:CONDA_PREFIX) {
    Add-SearchDirectory $cudnnSearch (Join-Path $env:CONDA_PREFIX "Lib\site-packages\torch\lib")
}
Add-SearchDirectory $cudnnSearch (Join-Path $ProjectRoot ".venv\Lib\site-packages\torch\lib")
$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCommand) {
    $condaParent = Split-Path -Parent $condaCommand.Source
    $condaRoot = if ((Split-Path $condaParent -Leaf) -in @("condabin", "Scripts")) {
        Split-Path -Parent $condaParent
    } else { $condaParent }
    $condaEnvs = Join-Path $condaRoot "envs"
    if (Test-Path -LiteralPath $condaEnvs) {
        foreach ($environment in Get-ChildItem -LiteralPath $condaEnvs -Directory) {
            Add-SearchDirectory $cudnnSearch (Join-Path $environment.FullName "Lib\site-packages\torch\lib")
        }
    }
}

Copy-FirstPattern "cudart64_*.dll" $cudaSearch $CudaStage | Out-Null
Copy-FirstPattern "cublas64_*.dll" $cudaSearch $CudaStage | Out-Null
Copy-FirstPattern "cublasLt64_*.dll" $cudaSearch $CudaStage | Out-Null
Copy-FirstPattern "cufft64_*.dll" $cudaSearch $CudaStage | Out-Null
Copy-FirstPattern "nvJitLink_*.dll" $cudaSearch $CudaStage $false | Out-Null
Copy-FirstPattern "nvrtc64_*.dll" $cudaSearch $CudaStage $false | Out-Null
Copy-FirstPattern "nvrtc-builtins64_*.dll" $cudaSearch $CudaStage $false | Out-Null
Copy-FirstPattern "cudnn*.dll" $cudnnSearch $CudaStage | Out-Null
Copy-FirstPattern "zlibwapi.dll" $cudnnSearch $CudaStage $false | Out-Null

if ($env:CUDA_PATH) {
    $cudaEula = Join-Path $env:CUDA_PATH "EULA.txt"
    if (Test-Path -LiteralPath $cudaEula) {
        Copy-Item -LiteralPath $cudaEula -Destination (Join-Path $LicenseStage "NVIDIA_CUDA_EULA.txt") -Force
    }
}

$versionParts = $Version.Split('.')
while ($versionParts.Count -lt 4) { $versionParts += '0' }
$numericVersion = ($versionParts[0..3] -join ', ')
$versionText = @"
VSVersionInfo(
  ffi=FixedFileInfo(filevers=($numericVersion), prodvers=($numericVersion), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('080404b0', [
    StringStruct('CompanyName', '岩创科技'),
    StringStruct('FileDescription', '岩土影像三维重建工作台'),
    StringStruct('FileVersion', '$Version'),
    StringStruct('InternalName', 'RockVision'),
    StringStruct('OriginalFilename', '岩土影像三维重建工作台.exe'),
    StringStruct('ProductName', '岩土影像三维重建工作台'),
    StringStruct('ProductVersion', '$Version')
  ])]), VarFileInfo([VarStruct('Translation', [2052, 1200])])]
)
"@
Set-Content -LiteralPath $VersionFile -Value $versionText -Encoding UTF8

$strictPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $VenvPython -m PyInstaller --version 2>&1 | Out-Null
$pyinstallerCheckExitCode = $LASTEXITCODE
$ErrorActionPreference = $strictPreference
if ($pyinstallerCheckExitCode -ne 0) {
    $ErrorActionPreference = "Continue"
    & $VenvPython -m pip install "pyinstaller>=6.11,<7"
    $pipExitCode = $LASTEXITCODE
    $ErrorActionPreference = $strictPreference
    if ($pipExitCode -ne 0) { throw "PyInstaller 安装失败。" }
}

$env:ROCKVISION_RUNTIME_STAGE = $RuntimeStage
$env:ROCKVISION_VERSION_FILE = $VersionFile
$env:ROCKVISION_APP_VERSION = $Version
$env:ROCKVISION_PROJECT_ROOT = $ProjectRoot
$ErrorActionPreference = "Continue"
if ($NoClean) {
    & $VenvPython -m PyInstaller --noconfirm $SpecPath
} else {
    & $VenvPython -m PyInstaller --noconfirm --clean $SpecPath
}
$pyinstallerExitCode = $LASTEXITCODE
$ErrorActionPreference = $strictPreference
if ($pyinstallerExitCode -ne 0) { throw "PyInstaller 构建失败，退出码 $pyinstallerExitCode。" }

$ReleaseDirectory = Join-Path $DistRoot "RockVision"
$ErrorActionPreference = "Continue"
& $VenvPython (Join-Path $ProjectRoot "scripts\verify_release.py") $ReleaseDirectory
$verifyExitCode = $LASTEXITCODE
$ErrorActionPreference = $strictPreference
if ($verifyExitCode -ne 0) { throw "冻结版发布自检失败。" }

if (-not $SkipInstaller) {
    $iscc = $null
    if ($InnoSetupPath) {
        $iscc = $InnoSetupPath
    } else {
        $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
        if ($command) { $iscc = $command.Source }
        if (-not $iscc) {
            $standardPaths = @(
                (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
                (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
                (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
            )
            $iscc = $standardPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        }
    }
    if (-not $iscc -or -not (Test-Path -LiteralPath $iscc -PathType Leaf)) {
        throw "未找到 Inno Setup 6。请运行 winget install JRSoftware.InnoSetup，或使用 -InnoSetupPath 指定 ISCC.exe。"
    }
    $ErrorActionPreference = "Continue"
    & $iscc "/DAppVersion=$Version" "/DSourceDir=$ReleaseDirectory" "/DOutputDir=$ReleaseRoot" $InstallerScript
    $innoExitCode = $LASTEXITCODE
    $ErrorActionPreference = $strictPreference
    if ($innoExitCode -ne 0) { throw "Inno Setup 编译失败，退出码 $innoExitCode。" }
    $installer = Get-ChildItem -LiteralPath $ReleaseRoot -Filter "*.exe" -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $installer) { throw "Inno Setup 未生成安装包。" }
    $hash = Get-FileHash -LiteralPath $installer.FullName -Algorithm SHA256
    "$($hash.Hash)  $($installer.Name)" | Set-Content -LiteralPath (Join-Path $ReleaseRoot "SHA256SUMS.txt") -Encoding ASCII
    Write-Host "安装包：$($installer.FullName)"
    Write-Host "SHA-256：$($hash.Hash)"
} else {
    Write-Host "已跳过 Inno Setup，仅生成目录版：$ReleaseDirectory"
}
