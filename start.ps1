# Start SwissDocAI Redact from this repository, without the installer: the desktop app
# (Tauri window, hot reload of the React UI) with the Python engine from .venv. Whatever
# is missing is set up first; the first start also compiles the app (a few minutes).
#
#   powershell -ExecutionPolicy Bypass -File start.ps1
#
# Needs uv, Node.js and Rust (README, "Ohne Installer starten"). macOS and Linux: start.sh.

$root = $PSScriptRoot
Set-Location $root

function Fail($message) {
    Write-Host ""
    Write-Host $message -ForegroundColor Red
    exit 1
}

function Need($command, $hint) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { Fail "$command was not found. $hint" }
}

# rustup's and uv's folders, for a shell opened before they were installed
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:USERPROFILE\.local\bin;$env:PATH"
Need git 'Install Git (with Git LFS): https://git-scm.com'
Need uv 'Install uv: https://docs.astral.sh/uv/getting-started/installation/'
Need npm 'Install Node.js (LTS): https://nodejs.org'
Need cargo 'Install Rust and the C++ build tools: https://v2.tauri.app/start/prerequisites/'

# The Swiss recognizer lives in Git LFS: a checkout without LFS has a small pointer file instead.
$model = Join-Path $root 'swissdocai\models\parseq_swiss.onnx'
if (-not (Test-Path $model) -or (Get-Item $model).Length -lt 1MB) {
    git lfs version *> $null
    if ($LASTEXITCODE -ne 0) { Fail 'Git LFS was not found. Install it (https://git-lfs.com), then start again.' }
    Write-Host 'Fetching the Swiss recognizer (Git LFS) ...'
    git lfs install --local
    git lfs pull
    if ($LASTEXITCODE -ne 0) { Fail 'git lfs pull failed.' }
}

# The engine's packages (no PyTorch), again whenever pyproject.toml changes
$python = Join-Path $root '.venv\Scripts\python.exe'
$installed = Join-Path $root '.venv\.engine-installed'
if (-not (Test-Path $python)) {
    Write-Host 'Creating the Python environment (.venv) ...'
    uv venv .venv
    if ($LASTEXITCODE -ne 0) { Fail 'uv venv failed.' }
}
if (-not (Test-Path $installed) -or (Get-Item pyproject.toml).LastWriteTime -gt (Get-Item $installed).LastWriteTime) {
    Write-Host "Installing the engine's Python packages ..."
    uv pip install --python $python -r pyproject.toml
    if ($LASTEXITCODE -ne 0) { Fail "Installing the engine's Python packages failed." }
    New-Item -ItemType File -Force $installed | Out-Null
}

# The app's Node.js packages, again whenever package-lock.json changes
Set-Location (Join-Path $root 'desktop')
if (-not (Test-Path 'node_modules\.package-lock.json') -or
    (Get-Item 'package-lock.json').LastWriteTime -gt (Get-Item 'node_modules\.package-lock.json').LastWriteTime) {
    Write-Host "Installing the app's Node.js packages ..."
    npm ci
    if ($LASTEXITCODE -ne 0) { Fail 'npm ci failed.' }
}

$env:PYTHONPATH = "$root"
$env:SWISSDOCAI_ENGINE = "$python|-m|swissdocai.app.api"
Write-Host 'Starting SwissDocAI Redact. The first start compiles the app, which takes a few minutes.'
npm run tauri dev
exit $LASTEXITCODE
