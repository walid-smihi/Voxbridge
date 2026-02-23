param(
    [switch]$WithLoopback
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

$py = Join-Path $projectRoot ".venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

if ($WithLoopback) {
    & $py -m pip install -r requirements-loopback.txt
}

Write-Host "Dependencies installed."
Write-Host "Activate venv: .\.venv\Scripts\Activate.ps1"
if ($WithLoopback) {
    Write-Host "Loopback mode ready. Use: python .\traductor.py --loopback"
}
Write-Host "Next step (required): .\install_whisper.ps1 -Model tiny.en"
Write-Host "Optional CUDA build: .\install_whisper.ps1 -Model tiny.en -BuildCuda"
