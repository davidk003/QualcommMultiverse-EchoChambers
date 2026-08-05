#Requires -Version 5.1
<#
.SYNOPSIS
    Dependency bootstrap helper -- installs all required packages via uv.

.DESCRIPTION
    Creates a local .venv and installs every package listed in
    requirements.txt using uv.  Run this once before running the app.

    Pass -Python to target a specific interpreter (default: 3.10).

.EXAMPLE
    # One-time setup
    .\run.ps1

    # Train the spectral classifier (synthesizes data, no hardware needed)
    .venv\Scripts\python.exe train_classifier.py

    # Run the detector end-to-end
    .venv\Scripts\python.exe run_echo_chamber.py
#>

[CmdletBinding()]
param(
    [string]$Python = "3.10"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Ok($msg)   { Write-Host "  [OK]    $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "  [INFO]  $msg" }

$ReqFile = Join-Path $PSScriptRoot "requirements.txt"
$VenvDir = Join-Path $PSScriptRoot ".venv"

if (-not (Test-Path $ReqFile)) {
    Write-Error "requirements.txt not found: $ReqFile"
    exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info "uv not found -- installing ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv install failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    }
}
Write-Ok "uv $(uv --version)"

if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating .venv (Python $Python) ..."
    uv venv "$VenvDir" --python $Python
    if ($LASTEXITCODE -ne 0) { Write-Error "uv venv failed."; exit 1 }
    Write-Ok ".venv created"
} else {
    Write-Ok ".venv already exists -- skipping creation"
}

Write-Info "Installing dependencies from requirements.txt ..."
uv pip install --system-certs --python "$VenvDir\Scripts\python.exe" -r "$ReqFile"
if ($LASTEXITCODE -ne 0) { Write-Error "uv pip install failed."; exit 1 }
Write-Ok "All dependencies installed"

Write-Host ""
Write-Host "  Setup complete." -ForegroundColor Cyan
Write-Host ""
Write-Host "    .venv\Scripts\python.exe train_classifier.py     # synthesize data + train + export ONNX"
Write-Host "    .venv\Scripts\python.exe run_echo_chamber.py     # capture + DSP + inference + alert"
Write-Host "    .venv\Scripts\python.exe run_echo_chamber.py --self-test   # offline smoke test, no mic needed"
Write-Host ""
