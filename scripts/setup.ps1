param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $PythonCommand -m venv $VenvPath
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -e "${ProjectRoot}[dev]"

Write-Host ""
Write-Host "环境安装完成。启动命令："
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$ProjectRoot\scripts\start.ps1`""
