$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "尚未安装项目环境。请先运行 scripts\setup.ps1"
}

$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = $env:NO_PROXY
$env:QT_API = "pyside6"

Set-Location -LiteralPath $ProjectRoot
& $VenvPython app.py
