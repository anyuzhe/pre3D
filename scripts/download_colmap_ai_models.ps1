$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TargetRoot = Join-Path $ProjectRoot "checkpoints\colmap_ai"
New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null

$Models = @(
    @{
        Name = "aliked-n16rot.onnx"
        Url = "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx"
        Sha256 = "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547"
    },
    @{
        Name = "aliked-lightglue.onnx"
        Url = "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx"
        Sha256 = "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d"
    },
    @{
        Name = "sift-lightglue.onnx"
        Url = "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx"
        Sha256 = "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e"
    }
)

foreach ($Model in $Models) {
    $Destination = Join-Path $TargetRoot $Model.Name
    $Valid = $false
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $Actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        $Valid = $Actual -eq $Model.Sha256
    }
    if ($Valid) {
        Write-Host "已校验：$($Model.Name)"
        continue
    }

    $Temporary = "$Destination.download"
    Write-Host "下载：$($Model.Name)"
    Invoke-WebRequest `
        -Uri $Model.Url `
        -OutFile $Temporary `
        -MaximumRedirection 10 `
        -TimeoutSec 180
    $Actual = (Get-FileHash -LiteralPath $Temporary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Model.Sha256) {
        Remove-Item -LiteralPath $Temporary -Force
        throw "SHA-256 校验失败：$($Model.Name)"
    }
    Move-Item -LiteralPath $Temporary -Destination $Destination -Force
}

Write-Host "COLMAP AI特征与LightGlue权重准备完成：$TargetRoot"
