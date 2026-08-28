param(
    [string]$CondaExe = "D:\anaconda3\Scripts\conda.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$InstallRoot = Join-Path $ProjectRoot "tools\openscenegraph"
$OsgConv = Join-Path $InstallRoot "Library\bin\osgconv.exe"

if (Test-Path -LiteralPath $OsgConv) {
    Write-Host "OpenSceneGraph is already installed: $OsgConv"
    exit 0
}
if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "conda.exe was not found: $CondaExe"
}

& $CondaExe create `
    --prefix $InstallRoot `
    --override-channels `
    --channel conda-forge `
    --solver libmamba `
    "openscenegraph=3.6.5" `
    --yes
if ($LASTEXITCODE -ne 0) {
    throw "OpenSceneGraph installation failed. conda exit code: $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $OsgConv)) {
    throw "osgconv.exe was not found after installation: $OsgConv"
}

Write-Host "OpenSceneGraph OSGB converter installed: $OsgConv"
