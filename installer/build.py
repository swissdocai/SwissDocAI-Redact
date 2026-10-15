"""Build the desktop app's installer for the system this runs on.

    uv run --no-project python installer/build.py              # Windows: setup .exe (NSIS)
    uv run --no-project python installer/build.py --bundles dmg   # macOS; Linux: appimage,deb

1. A clean Python environment with the engine's dependencies only (build/engine-venv, no
   PyTorch), never the development venv.
2. The engine frozen with PyInstaller (build/engine/swissdocai-engine), then started once to
   check that it answers and finds its hardware.
3. The Tauri app with the engine as a resource, bundled into the installer
   (desktop/src-tauri/target/release/bundle/).

The installer holds no models and is the same for every machine: the app detects the hardware
on its first start and downloads what runs there (see swissdocai/engine.py).
Needs uv, Node.js and Rust; on Linux also Tauri's system packages (webkit2gtk, see the workflow).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
VENV = BUILD / "engine-venv"
WINDOWS = sys.platform == "win32"
PYTHON = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")
ENGINE = BUILD / "engine" / "swissdocai-engine" / ("swissdocai-engine.exe" if WINDOWS else "swissdocai-engine")
DEFAULT_BUNDLES = {"win32": "nsis", "darwin": "dmg"}.get(sys.platform, "appimage,deb")


def run(cmd: list, **kwargs) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def check_models() -> None:
    """The Swiss recognizer lives in Git LFS: a checkout without LFS has a pointer file instead."""
    model = ROOT / "swissdocai" / "models" / "parseq_swiss.onnx"
    if not model.exists() or model.stat().st_size < 1_000_000:
        sys.exit(f"{model} is missing or an LFS pointer: run `git lfs pull` first")


def engine_venv(python: str) -> None:
    if not PYTHON.exists():
        run(["uv", "venv", VENV, "--python", python])
    # the engine's dependencies from pyproject.toml (the package itself is frozen from the source)
    run(["uv", "pip", "install", "--python", PYTHON, "-r", ROOT / "pyproject.toml", "pyinstaller>=6.10"])


def freeze() -> None:
    shutil.rmtree(BUILD / "engine", ignore_errors=True)
    run([PYTHON, "-m", "PyInstaller", "--noconfirm", "--clean", "--log-level", "WARN",
         "--distpath", BUILD / "engine", "--workpath", BUILD / "pyinstaller",
         ROOT / "installer" / "swissdocai-engine.spec"])


# Stands in for the app: starts the engine as the app does, passes on its port, and ends when its
# stdin closes, without stopping the engine — which has to notice and end by itself.
_APP_STAND_IN = """
import os, subprocess, sys
engine = subprocess.Popen([sys.argv[1], "--port", "0", "--token", "check", "--parent-pid", str(os.getpid())],
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
for line in engine.stdout:
    if line.startswith("SWISSDOCAI_ENGINE_READY "):
        print(line.split()[1], flush=True)
        break
sys.stdin.read()
"""


def check_engine() -> None:
    """Start the frozen engine as the app does, ask it what it would download and run here, and
    check that it ends when the app ends."""
    cache = BUILD / "check-cache"  # nothing from this machine's real cache
    shutil.rmtree(cache, ignore_errors=True)
    app = subprocess.Popen([sys.executable, "-c", _APP_STAND_IN, str(ENGINE)], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, text=True, env={**os.environ, "SWISSDOCAI_CACHE": str(cache)})
    try:
        port = int(app.stdout.readline() or 0)
        if not port:
            sys.exit("the frozen engine did not get ready")

        def get(path: str) -> dict:
            request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"X-Token": "check"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)

        assert get("/api/health")["ok"]
        setup = get("/api/setup")
        print("engine check: hardware", json.dumps(setup["hardware"]), flush=True)
        print("engine check: first start downloads", ", ".join(f"{n['name']} ({n['gb']:.2f} GB)"
                                                             for n in setup["needed"]), flush=True)
        assert setup["needed"], "a fresh cache needs downloads"
        assert get("/api/settings")["language"] == "de"
    finally:
        app.stdin.close()
        app.wait(timeout=30)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            get("/api/health")
        except OSError:  # the engine has ended with its app
            break
        time.sleep(0.5)
    else:
        sys.exit("the frozen engine kept running after the app had ended")
    time.sleep(1)  # its log file closes
    shutil.rmtree(cache, ignore_errors=True)


def bundle(bundles: str) -> None:
    desktop = ROOT / "desktop"
    npm = shutil.which("npm") or sys.exit("npm (Node.js) is needed")
    cargo_bin = Path.home() / ".cargo" / "bin"
    env = dict(os.environ)
    if not shutil.which("cargo") and cargo_bin.exists():
        env["PATH"] = f"{cargo_bin}{os.pathsep}{env['PATH']}"
    if not (desktop / "node_modules").exists():
        run([npm, "ci"], cwd=desktop, env=env)
    run([npm, "run", "tauri", "--", "build", "--config", "src-tauri/tauri.release.conf.json",
         "--bundles", bundles], cwd=desktop, env=env)
    out = desktop / "src-tauri" / "target" / "release" / "bundle"
    for path in sorted(out.glob("*/*")):
        if path.suffix in (".exe", ".msi", ".dmg", ".AppImage", ".deb", ".rpm"):
            print(f"installer: {path} ({path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundles", default=DEFAULT_BUNDLES, help=f"Tauri bundle types (default here: {DEFAULT_BUNDLES})")
    parser.add_argument("--python", default="3.12", help="Python version of the frozen engine")
    parser.add_argument("--skip-engine", action="store_true", help="Reuse build/engine from an earlier run")
    parser.add_argument("--engine-only", action="store_true", help="Freeze and check the engine, no installer")
    args = parser.parse_args()
    check_models()
    if not args.skip_engine:
        engine_venv(args.python)
        freeze()
        check_engine()
    if not ENGINE.exists():
        sys.exit(f"{ENGINE} is missing: build without --skip-engine")
    if not args.engine_only:
        bundle(args.bundles)


if __name__ == "__main__":
    main()
