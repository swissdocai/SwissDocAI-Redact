#!/usr/bin/env bash
# Start SwissDocAI Redact from this repository, without the installer: the desktop app
# (Tauri window, hot reload of the React UI) with the Python engine from .venv. Whatever
# is missing is set up first; the first start also compiles the app (a few minutes).
#
#   bash start.sh
#
# Needs uv, Node.js and Rust (README, "Ohne Installer starten"). Windows: start.ps1.
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"

fail() { printf '\n\033[31m%s\033[0m\n' "$1" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 was not found. $2"; }

# rustup's and uv's folders, for a shell opened before they were installed
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
need git "Install Git (with Git LFS): https://git-scm.com"
need uv "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
need npm "Install Node.js (LTS): https://nodejs.org"
need cargo "Install Rust and Tauri's system packages: https://v2.tauri.app/start/prerequisites/"
case "$(command -v npm)" in
    /mnt/*) fail "npm is Windows' (WSL shares its PATH). Install Node.js inside Linux, e.g. with nvm." ;;
esac

# The Swiss recognizer lives in Git LFS: a checkout without LFS has a small pointer file instead.
model=swissdocai/models/parseq_swiss.onnx
if [ ! -f "$model" ] || [ "$(wc -c < "$model" | tr -d ' ')" -lt 1000000 ]; then
    git lfs version >/dev/null 2>&1 || fail "Git LFS was not found. Install it (https://git-lfs.com), then start again."
    echo "Fetching the Swiss recognizer (Git LFS) ..."
    git lfs install --local
    git lfs pull
fi

# The engine's packages (no PyTorch), again whenever pyproject.toml changes
python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
    echo "Creating the Python environment (.venv) ..."
    uv venv .venv
fi
if [ ! -f .venv/.engine-installed ] || [ pyproject.toml -nt .venv/.engine-installed ]; then
    echo "Installing the engine's Python packages ..."
    uv pip install --python "$python" -r pyproject.toml
    touch .venv/.engine-installed
fi

# The app's Node.js packages, again whenever package-lock.json changes
cd desktop
if [ ! -f node_modules/.package-lock.json ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
    echo "Installing the app's Node.js packages ..."
    npm ci
fi

export PYTHONPATH="$root"
export SWISSDOCAI_ENGINE="$python|-m|swissdocai.app.api"
echo "Starting SwissDocAI Redact. The first start compiles the app, which takes a few minutes."
exec npm run tauri dev
