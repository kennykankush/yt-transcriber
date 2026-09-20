#requires -Version 5.1
<#
.SYNOPSIS
Set up native Windows transcription. Existing models and environments are reused.
.EXAMPLE
.\bootstrap.ps1 -WithWhisperModel -Cuda -InstallCommand
#>
[CmdletBinding()]
param(
    [switch]$WithWhisperModel,
    [switch]$Cuda,
    [switch]$InstallCommand,
    [switch]$SkipModels,
    [string]$DownloadCache
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitProcess) {
    throw 'Run this script in 64-bit Windows PowerShell. macOS/Linux: use bootstrap.sh.'
}
if ($WithWhisperModel -and $SkipModels) {
    throw '-WithWhisperModel and -SkipModels cannot be combined.'
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw 'Install FFmpeg first: winget install --id Gyan.FFmpeg -e. Then open a new terminal.'
}

$ytxRepo = $PSScriptRoot
$ytxTools = Join-Path $ytxRepo '.tools'
$ytxModels = Join-Path $ytxRepo 'models'
$ytxPython = Join-Path $ytxRepo '.venv\Scripts\python.exe'
if (-not $DownloadCache) { $DownloadCache = Join-Path $ytxTools 'downloads' }
New-Item -ItemType Directory -Path $ytxTools, $ytxModels, $DownloadCache -Force | Out-Null

function Get-Download {
    param([string]$Url, [string]$Destination, [string]$Sha256)
    if (Test-Path -LiteralPath $Destination) {
        if ($Sha256) {
            if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $Sha256) {
                return
            }
        } elseif ((Get-Item -LiteralPath $Destination).Length -gt 0) {
            return
        }
    }
    Write-Host "Downloading $(Split-Path -Leaf $Destination)..."
    $partial = "$Destination.part"
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 --silent --show-error -o $partial $Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
    }
    if ($Sha256 -and (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $Sha256) {
        throw "Checksum mismatch: $Url"
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

Write-Host 'Setting up Python packages...'
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $ytxPython)) {
    if ($uv) {
        & uv venv (Join-Path $ytxRepo '.venv') --python 3.12
    } else {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            throw 'Install Python 3.12 or uv, then run this script again.'
        }
        & python -m venv (Join-Path $ytxRepo '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
if ($uv) {
    & uv pip install --python $ytxPython -r (Join-Path $ytxRepo 'requirements-windows.txt')
} else {
    & $ytxPython -m pip install -r (Join-Path $ytxRepo 'requirements-windows.txt')
}
if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed.' }

# Pin both the release and its SHA-256; never execute an unchecked binary download.
$release = 'b5130'
if ($Cuda) {
    $asset = 'whisper-cublas-12.4.0-bin-x64.zip'
    $checksum = 'af520ddd034d985b55dfeea3e465ed93653ba2aee1a55e865033edc548c272a7'
    $backend = 'cuda12'
} else {
    $asset = 'whisper-bin-x64.zip'
    $checksum = 'f9ec6c52a2e949b62ab51fa21d0d497958f9e41c3010c157c4e42932d5316f3c'
    $backend = 'cpu'
}
$whisperDir = Join-Path $ytxTools "whisper-$backend"
$whisperExe = Join-Path $whisperDir 'Release\whisper-cli.exe'
$marker = Join-Path $whisperDir '.archive-sha256'
if (-not ((Test-Path -LiteralPath $whisperExe) -and
          (Test-Path -LiteralPath $marker) -and
          ((Get-Content -LiteralPath $marker -Raw).Trim() -eq $checksum))) {
    $archive = Join-Path $DownloadCache "whisper-$release-$backend.zip"
    Get-Download "https://github.com/ggml-org/whisper.cpp/releases/download/$release/$asset" $archive $checksum
    Expand-Archive -LiteralPath $archive -DestinationPath $whisperDir -Force
    if (-not (Test-Path -LiteralPath $whisperExe)) { throw 'Whisper archive is missing whisper-cli.exe.' }
    Set-Content -LiteralPath $marker -Value $checksum -Encoding ascii
}
Write-Host "Whisper installed ($backend)."

if (-not $SkipModels) {
    $searchDirs = @($env:YTX_MODELS, $ytxModels, (Join-Path $env:USERPROFILE 'dev\whisperccp\models')) |
        Where-Object { $_ }
    $foundModel = $false
    foreach ($directory in $searchDirs) {
        foreach ($name in @('ggml-large-v3-turbo.bin', 'ggml-medium.bin', 'ggml-small.bin')) {
            if (Test-Path -LiteralPath (Join-Path $directory $name)) { $foundModel = $true }
        }
    }
    if (-not $foundModel) {
        if ($WithWhisperModel) {
            Get-Download 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin' `
                (Join-Path $ytxModels 'ggml-large-v3-turbo.bin')
        } else {
            Write-Warning 'No Whisper model found. Re-run with -WithWhisperModel (about 1.6 GB).'
        }
    }
    Get-Download 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin' `
        (Join-Path $ytxModels 'ggml-silero-v6.2.0.bin')
    $sherpaDir = Join-Path $ytxModels 'sherpa'
    New-Item -ItemType Directory -Path $sherpaDir -Force | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $sherpaDir 'sherpa-onnx-pyannote-segmentation-3-0\model.onnx'))) {
        if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
            throw 'tar.exe is required to unpack the speaker model (included in current Windows 10/11).'
        }
        $segArchive = Join-Path $DownloadCache 'sherpa-segmentation.tar.bz2'
        Get-Download 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2' $segArchive
        & tar.exe xjf $segArchive -C $sherpaDir
        if ($LASTEXITCODE -ne 0) { throw 'Speaker model extraction failed.' }
    }
    Get-Download 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx' `
        (Join-Path $sherpaDir 'nemo_en_titanet_large.onnx')
}

if ($InstallCommand) {
    $commandDir = Join-Path $ytxRepo 'bin'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($commandDir -notin ($userPath -split ';')) {
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$commandDir", 'User')
    }
    if ($commandDir -notin ($env:Path -split ';')) { $env:Path = "$commandDir;$env:Path" }
    Write-Host 'Installed transcribe. Open a new terminal if it is not found in an existing one.'
}

if (-not (Get-Command node -ErrorAction SilentlyContinue) -and
    -not (Get-Command deno -ErrorAction SilentlyContinue)) {
    Write-Warning 'For YouTube, install Node.js 22+ or Deno and put it on PATH. Local files do not need it.'
}
Write-Host 'Setup complete. Try: .\bin\transcribe.cmd --help'
