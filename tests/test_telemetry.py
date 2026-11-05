"""ONNX Runtime's telemetry: switched off before anything can load it, on every way in.

Each check runs in a new Python process: in this one, the module is long loaded."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# Records the variable at the moment `onnxruntime` is imported, whoever imports it.
WATCH = """
import importlib.abc, os, sys
seen = []
class Watch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "onnxruntime":
            seen.append(os.environ.get("ORT_DISABLE_TELEMETRY"))
        return None
sys.meta_path.insert(0, Watch())
"""


def run(code: str, env_value: str | None = None) -> None:
    import os

    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    if env_value is not None:
        env["ORT_DISABLE_TELEMETRY"] = env_value
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{ROOT / 'installer'}"
    done = subprocess.run([sys.executable, "-c", WATCH + code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("module", ["swissdocai", "swissdocai.cli", "swissdocai.redact", "swissdocai.saas",
                                    "swissdocai.app.api", "engine"])
def test_every_entry_point_sets_the_variable_and_does_not_load_onnx_runtime(module):
    pytest.importorskip("fastapi")
    run(f"""
import {module}
assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
assert "onnxruntime" not in sys.modules and not seen, seen
""")


@pytest.mark.parametrize("value", ["0", "", "false"])
def test_nobody_can_switch_it_on_again_by_accident(value):
    run("""
import swissdocai
assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
""", env_value=value)


def test_onnx_runtime_loads_after_the_variable_also_through_onnxtr():
    pytest.importorskip("onnxtr")
    run("""
import swissdocai
import onnxtr.models
assert "onnxruntime" in sys.modules and seen and set(seen) == {"1"}, seen
""")


def test_the_reader_switches_the_python_api_off_too(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from swissdocai import words

    calls = []
    monkeypatch.setattr(ort, "disable_telemetry_events", lambda: calls.append(1))
    assert words.onnxruntime() is ort and calls == [1]


def test_nothing_in_the_package_imports_onnx_runtime_past_that_switch():
    package = ROOT / "swissdocai"
    lines = [f"{path.name}: {line.strip()}" for path in package.rglob("*.py")
             for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip().startswith(("import onnxruntime", "from onnxruntime"))]
    assert lines == ["words.py: import onnxruntime as ort"]
