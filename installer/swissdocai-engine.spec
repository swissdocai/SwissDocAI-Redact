# PyInstaller: the desktop app's engine (swissdocai.app.api) as one folder, which the installer
# puts next to the app (desktop/src-tauri/tauri.release.conf.json). Built by installer/build.py
# in a clean environment with the engine's dependencies only: no PyTorch.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 (PyInstaller defines SPECPATH)

hiddenimports = (
    collect_submodules("swissdocai")
    + collect_submodules("onnxtr")  # its model families are imported by name (words._onnx_model_urls)
    + collect_submodules("uvicorn")  # protocols and loops chosen at run time
)
datas = [(str(ROOT / "swissdocai" / "models"), "swissdocai/models")]  # the Swiss recognizer
for package in ("onnxtr", "anyascii", "langdetect"):
    datas += collect_data_files(package)

binaries = []
if sys.platform == "win32":
    # The Visual C++ runtime, as one matching set: ONNX Runtime needs it, and the engine puts it
    # next to llama.cpp's Windows builds, which need it too (engine._add_vc_runtime). A PC without
    # the Visual C++ Redistributable then runs both.
    system32 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
    binaries += [(str(system32 / name), ".") for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")]

# The engine runs without these; they come in only through optional code paths (docTR with
# PyTorch, transformers readers) or development tools.
excludes = ["torch", "torchvision", "transformers", "doctr", "datasets", "onnx", "matplotlib", "pandas",
            "tkinter", "IPython", "pytest", "PySide6", "PyQt5", "PyQt6"]

a = Analysis(
    [str(ROOT / "installer" / "engine.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="swissdocai-engine",
    console=True,  # stdout carries the "ready" line to the app; the app starts it without a window
    upx=False,
    icon=str(ROOT / "desktop" / "src-tauri" / "icons" / "icon.ico") if sys.platform == "win32" else None,
)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="swissdocai-engine")
