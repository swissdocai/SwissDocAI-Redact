"""Local engine: backend and build choice, GPU offload (no downloads, no GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from swissdocai import engine  # noqa: E402
from swissdocai.engine import GPU, Hardware  # noqa: E402


def machine(system="Windows", gpus=(), ram=32.0, arch="x86_64"):
    return Hardware(system, arch, ram, list(gpus))


RTX_4070 = GPU("nvidia", "RTX 4070", 12, "581.57", 8.9)
GTX_1080 = GPU("nvidia", "GTX 1080", 8, "581.57", 6.1)
RTX_5070_OLD_DRIVER = GPU("nvidia", "RTX 5070", 12, "572.16", 12.0)
RX_7900 = GPU("amd", "RX 7900 XTX", 24)


@pytest.fixture(autouse=True)
def no_override(monkeypatch, tmp_path):
    for name in ("SWISSDOCAI_LLAMA_BACKEND", "SWISSDOCAI_ROCM_LIBS", "HIP_PATH", "ROCM_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "no-rocm"))
    monkeypatch.setenv("SWISSDOCAI_CACHE", str(tmp_path / "cache"))  # no backend chosen yet


def test_backend_follows_the_hardware(monkeypatch, tmp_path):
    assert engine.backend_candidates(machine(gpus=[RTX_4070])) == ["cuda", "vulkan", "cpu"]
    assert engine.backend_candidates(machine(gpus=[GPU("nvidia", "GT 1030", 2, "581.57", 6.1)])) == ["vulkan", "cpu"]
    assert engine.backend_candidates(machine(gpus=[GPU("nvidia", "Quadro", 8, "470.10", 6.1)])) == ["vulkan", "cpu"]
    assert engine.backend_candidates(machine(gpus=[RX_7900])) == ["vulkan", "cpu"]  # no ROCm libraries
    assert engine.backend_candidates(machine(gpus=[GPU("intel", "Arc A770", 16)])) == ["vulkan", "cpu"]
    assert engine.backend_candidates(machine(gpus=[GPU("other", "Basic Display", 0)])) == ["cpu"]
    assert engine.backend_candidates(machine()) == ["cpu"]
    assert engine.backend_candidates(machine("Darwin", [GPU("apple", "M2", 16)], arch="arm64")) == ["metal", "cpu"]
    assert engine.backend_candidates(machine("Darwin")) == ["cpu"]  # Intel Mac
    assert engine.choose_backend(machine(gpus=[RTX_4070])) == "cuda"

    hip = tmp_path / "hip" / "bin"
    hip.mkdir(parents=True)
    (hip / "hipblas.dll").write_bytes(b"")
    monkeypatch.setenv("HIP_PATH", str(hip.parent))
    assert engine.backend_candidates(machine(gpus=[RX_7900])) == ["rocm", "vulkan", "cpu"]
    assert engine.backend_candidates(machine(gpus=[GPU("amd", "Radeon 780M", 0.5)])) == ["vulkan", "cpu"]
    rocm = tmp_path / "rocm" / "lib"
    rocm.mkdir(parents=True)
    (rocm / "libhipblas.so.3").write_bytes(b"")
    monkeypatch.setenv("ROCM_PATH", str(rocm.parent))
    assert engine.backend_candidates(machine("Linux", [RX_7900])) == ["rocm", "vulkan", "cpu"]

    monkeypatch.setenv("SWISSDOCAI_LLAMA_BACKEND", "sycl")
    assert engine.choose_backend(machine(gpus=[RTX_4070])) == "sycl"


def test_the_cuda_build_follows_driver_and_gpu():
    assert engine.cuda_version(machine(gpus=[RTX_4070])) == "13"
    assert engine.cuda_version(machine(gpus=[GTX_1080])) == "12"  # CUDA 13 dropped Pascal
    assert engine.cuda_version(machine(gpus=[GPU("nvidia", "RTX 4070", 12, "551.86", 8.9)])) == "12"
    assert engine.cuda_version(machine(gpus=[RTX_5070_OLD_DRIVER])) is None  # Blackwell needs CUDA 13
    assert engine.cuda_version(machine(gpus=[GPU("nvidia", "RTX 4070", 12)])) is None  # driver unknown
    assert engine.cuda_version(machine(gpus=[RX_7900])) is None


def test_release_build_names():
    assert engine.release_assets(machine(), "vulkan", "b1") == ["llama-b1-bin-win-vulkan-x64.zip"]
    assert engine.release_assets(machine(), "rocm", "b1") == ["llama-b1-bin-win-rocm-10.0-x64.zip"]
    assert engine.release_assets(machine(gpus=[RTX_4070]), "cuda", "b1") == [
        "llama-b1-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"]
    assert engine.release_assets(machine(gpus=[GTX_1080]), "cuda", "b1") == [
        "llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    assert engine.release_assets(machine("Linux"), "cpu", "b1") == ["llama-b1-bin-ubuntu-x64.tar.gz"]
    assert engine.release_assets(machine("Linux", [GTX_1080]), "cuda", "b1") == [
        "llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz"]
    assert engine.release_asset(machine("Darwin", arch="arm64"), "metal", "b1") == "llama-b1-bin-macos-arm64.tar.gz"
    assert engine.release_asset(machine(arch="arm64"), "vulkan", "b1") == "llama-b1-bin-win-vulkan-arm64.zip"


def test_every_build_the_app_may_choose_is_pinned():
    """The first start downloads with checksums from the table, not from GitHub's API."""
    gpus = [[], [RTX_4070], [GTX_1080], [RX_7900]]
    for system, arch in [("Windows", "x86_64"), ("Linux", "x86_64"), ("Linux", "arm64"), ("Windows", "arm64")]:
        for gpu in gpus:
            hw = machine(system, gpu, arch=arch)
            for backend in ["cuda", "rocm", "vulkan", "cpu"] if arch == "x86_64" else ["vulkan", "cpu"]:
                for asset in engine.release_assets(hw, backend):
                    assert asset in engine._ASSETS, asset
    for arch in ("arm64", "x86_64"):
        assert engine.release_asset(machine("Darwin", arch=arch), "metal") in engine._ASSETS
    assert 0.6 < engine.server_download_gb(machine(gpus=[GTX_1080]), "cuda") < 0.7  # build + CUDA runtime


def test_the_first_build_that_sees_a_gpu_is_kept(monkeypatch):
    hw = machine(gpus=[RTX_4070])
    installed, probed = [], []

    def install(hw, backend, build=engine.LLAMA_CPP_BUILD, progress=None):
        installed.append(backend)
        server = engine._build_dir(hw, backend, build) / "llama-server.exe"
        server.parent.mkdir(parents=True, exist_ok=True)
        server.write_bytes(b"")
        (server.parent / ".complete").write_text(build)
        return server

    def probe(server, backend):
        probed.append(backend)
        return {"vulkan": ["Vulkan0: NVIDIA GeForce RTX 4070 (12282 MiB, 11000 MiB free)"]}.get(backend, [])

    monkeypatch.setattr(engine, "_install", install)
    monkeypatch.setattr(engine, "probe_devices", probe)
    server, backend = engine.ensure_server(hw)  # e.g. a driver the CUDA build does not run on
    assert backend == "vulkan" and installed == probed == ["cuda", "vulkan"]
    assert not engine._build_dir(hw, "cuda").exists()  # the useless build is deleted again
    assert engine.chosen_backend()["devices"][0].startswith("Vulkan0")
    assert engine.choose_backend(hw) == "vulkan" and engine.server_ready(hw)
    assert engine.ensure_server(hw) == (server, "vulkan") and probed == ["cuda", "vulkan"]  # kept, not probed again
    assert not engine.server_ready(machine(gpus=[RX_7900]))  # another GPU: chosen again

    engine._choice_file().unlink()
    monkeypatch.setattr(engine, "probe_devices", lambda server, backend: [])
    assert engine.ensure_server(machine(gpus=[RX_7900]))[1] == "cpu"  # no build sees the GPU: the CPU


def test_the_cuda_runtime_lands_next_to_the_server(monkeypatch, tmp_path):
    import io
    import tarfile

    def tarball(path, files):
        with tarfile.open(path, "w:gz") as t:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))

    archives = tmp_path / "archives"
    archives.mkdir()
    hw = machine("Linux", [GTX_1080])
    main, runtime = engine.release_assets(hw, "cuda", "b1")
    tarball(archives / main, {"llama-b1/llama-server": b"#!", "llama-b1/libggml-cuda.so": b"so"})
    tarball(archives / runtime, {"llama-b1/libcudart.so.12": b"so", "llama-b1/libcublas.so.12": b"so"})

    def download(url, target, sha256="", progress=None, label=""):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((archives / target.name).read_bytes())
        return target

    monkeypatch.setattr(engine, "download", download)
    monkeypatch.setattr(engine, "_asset_digest", lambda build, asset: None)  # build "b1": no GitHub API
    server = engine._install(hw, "cuda", "b1")
    assert sorted(p.name for p in server.parent.iterdir()) == [
        "libcublas.so.12", "libcudart.so.12", "libggml-cuda.so", "llama-server"]
    assert engine._installed(hw, "cuda", "b1") == server
    assert not list(engine.cache_dir().joinpath("llama.cpp").glob("*.tar.gz"))  # archives removed


def test_probe_reads_the_device_list(monkeypatch, tmp_path):
    import subprocess

    out = ("ggml_vulkan: Found 2 Vulkan devices:\nAvailable devices:\n"
           "  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23749 MiB free)\n  Vulkan1: llvmpipe (LLVM 19.1.1) (0 MiB)\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, out, ""))
    assert engine.probe_devices(tmp_path / "llama-server", "vulkan") == [
        "Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23749 MiB free)"]
    assert engine.probe_devices(tmp_path / "llama-server", "rocm") == []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(
        a, 0, "Available devices:\n  (none)\n", ""))
    assert engine.probe_devices(tmp_path / "llama-server", "vulkan") == []


def test_the_model_goes_on_the_gpu_when_it_fits():
    def layers(*gpus, ram=32.0):
        hw = machine(gpus=gpus, ram=ram)
        return engine.LocalEngine(hardware=hw, backend=engine.choose_backend(hw)).gpu_layers()

    assert layers(RX_7900) == 999
    assert layers(GPU("nvidia", "GTX 1650", 4)) == 0  # 4 GB: too tight, runs on the CPU
    assert layers(GPU("nvidia", "RTX 3050", 6)) == 999
    assert layers() == 0  # no GPU
    assert layers(GPU("intel", "Iris Xe", 0), ram=16) == 999  # integrated: shares system memory
    assert layers(GPU("intel", "UHD", 0), ram=8) == 0


def test_unknown_model_is_refused():
    with pytest.raises(ValueError):
        engine.LocalEngine("qwen3.5-4b", hardware=machine())


class FakeStream:
    """httpx.stream stand-in serving ``data`` (honouring a Range header)."""

    def __init__(self, data: bytes, calls: list):
        self.data, self.calls = data, calls

    def __call__(self, method, url, headers=None, **kw):
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers and "Range" in headers else 0
        self.calls.append(start)
        body, status = self.data[start:], 206 if start else 200

        class Response:
            status_code = status
            headers = {"content-length": str(len(body))}

            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                for i in range(0, len(body), 4):
                    yield body[i:i + 4]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

        return Response()


def test_download_checks_resumes_and_reports(monkeypatch, tmp_path):
    import hashlib

    import httpx

    data = b"swissdocai model bytes"
    sha = hashlib.sha256(data).hexdigest()
    calls, seen = [], []
    monkeypatch.setattr(httpx, "stream", FakeStream(data, calls))
    target = tmp_path / "m.gguf"
    target.with_name("m.gguf.part").write_bytes(data[:8])  # an earlier, broken download
    engine.download("https://x/m.gguf", target, sha, lambda what, done, total: seen.append((done, total)))
    assert target.read_bytes() == data and calls == [8] and seen[-1] == (len(data), len(data))
    engine.download("https://x/m.gguf", target, sha)  # checked before: no second request
    assert calls == [8]
    with pytest.raises(RuntimeError):
        engine.download("https://x/m.gguf", tmp_path / "bad.gguf", "0" * 64)
    assert not (tmp_path / "bad.gguf").exists()
