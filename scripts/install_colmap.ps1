param(
    [string]$Version = "4.1.1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ToolsRoot = Join-Path $ProjectRoot "tools"
$InstallPath = Join-Path $ToolsRoot "colmap"
$CacheRoot = Join-Path $ProjectRoot ".cache\downloads"
$ArchivePath = Join-Path $CacheRoot "colmap-$Version-windows-cuda.zip"
$DownloadUrl = "https://github.com/colmap/colmap/releases/download/$Version/colmap-x64-windows-cuda.zip"

if (Test-Path -LiteralPath (Join-Path $InstallPath "COLMAP.bat")) {
    Write-Host "COLMAP is already installed: $InstallPath"
    exit 0
}

New-Item -ItemType Directory -Path $ToolsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $CacheRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $ArchivePath)) {
    Write-Host "Downloading COLMAP $Version CUDA build from the official GitHub release..."
    curl.exe -L --fail --retry 5 --retry-delay 3 -o $ArchivePath $DownloadUrl
    if ($LASTEXITCODE -ne 0) {
        throw "COLMAP download failed (curl exit code $LASTEXITCODE)"
    }
}

$ExpectedMinimumBytes = 300MB
$Archive = Get-Item -LiteralPath $ArchivePath
if ($Archive.Length -lt $ExpectedMinimumBytes) {
    throw "Downloaded archive is unexpectedly small: $($Archive.Length) bytes"
}

$StagingPath = Join-Path $ToolsRoot "colmap-$Version-staging"
if (Test-Path -LiteralPath $StagingPath) {
    throw "Staging directory already exists: $StagingPath"
}
New-Item -ItemType Directory -Path $StagingPath | Out-Null
Expand-Archive -LiteralPath $ArchivePath -DestinationPath $StagingPath

$Batch = Get-ChildItem -LiteralPath $StagingPath -Filter "COLMAP.bat" -File -Recurse | Select-Object -First 1
if ($null -eq $Batch) {
    throw "COLMAP.bat was not found in the archive"
}
$PayloadRoot = $Batch.Directory.FullName
if (Test-Path -LiteralPath $InstallPath) {
    throw "Install target already exists: $InstallPath"
}
Move-Item -LiteralPath $PayloadRoot -Destination $InstallPath

$ColmapBatch = Join-Path $InstallPath "COLMAP.bat"
& $ColmapBatch -h | Select-Object -First 8
if ($LASTEXITCODE -ne 0) {
    throw "COLMAP post-install self-check failed"
}
Write-Host "COLMAP $Version CUDA build installed: $ColmapBatch"
