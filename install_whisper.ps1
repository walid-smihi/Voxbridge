param(
    [string]$RepoUrl = "https://github.com/ggml-org/whisper.cpp",
    [string]$Model = "tiny.en",
    [switch]$BuildCuda,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Commande manquante: $Name"
    }
}

Require-Command git
Require-Command cmake

$whisperDir = Join-Path $projectRoot "whisper.cpp"

if (-not (Test-Path $whisperDir)) {
    Write-Host "Clonage de whisper.cpp depuis $RepoUrl ..."
    git clone --depth 1 $RepoUrl whisper.cpp
} elseif (-not (Test-Path (Join-Path $whisperDir ".git"))) {
    throw "Le dossier whisper.cpp existe deja mais n'est pas un repo git. Supprime-le ou renomme-le puis relance."
} else {
    Write-Host "whisper.cpp deja present: $whisperDir"
}

if (-not $SkipBuild) {
    $buildDir = if ($BuildCuda) { Join-Path $whisperDir "build-cuda" } else { Join-Path $whisperDir "build" }

    $cfgArgs = @("-S", $whisperDir, "-B", $buildDir)
    if ($BuildCuda) {
        $cfgArgs += "-DGGML_CUDA=ON"
        Write-Host "Configuration CMake (CUDA)..."
    } else {
        Write-Host "Configuration CMake (CPU)..."
    }

    & cmake @cfgArgs

    Write-Host "Build whisper-cli..."
    & cmake --build $buildDir --config Release --target whisper-cli
}

$modelsDir = Join-Path $whisperDir "models"
$downloadScript = Join-Path $modelsDir "download-ggml-model.cmd"

if (Test-Path $downloadScript) {
    Push-Location $modelsDir
    try {
        Write-Host "Telechargement du modele: $Model"
        & .\download-ggml-model.cmd $Model
    }
    finally {
        Pop-Location
    }
} else {
    Write-Warning "Script de download introuvable: $downloadScript"
    Write-Warning "Telecharge un modele manuellement depuis https://huggingface.co/ggerganov/whisper.cpp/tree/main"
}

Write-Host "Setup whisper.cpp termine."
if ($BuildCuda) {
    Write-Host "Binaire attendu: whisper.cpp\\build-cuda\\bin\\Release\\whisper-cli.exe"
} else {
    Write-Host "Binaire attendu: whisper.cpp\\build\\bin\\Release\\whisper-cli.exe"
}
Write-Host "Modeles: whisper.cpp\\models\\"
