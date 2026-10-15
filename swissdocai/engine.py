"""Local LLM engine: run a GGUF model with llama.cpp's server — no Ollama needed.

The part of Ollama we need, nothing more: detect the hardware, pick the
matching prebuilt llama.cpp server, download it and the model once into a
cache, start ``llama-server`` on a free local port, and talk to it through the
same OpenAI-compatible API (``llm.ChatClient``).

    with LocalEngine() as engine:
        client = engine.client()          # a llm.ChatClient
        ...

Backends (llama.cpp release builds), chosen from the hardware on first use:
CUDA on NVIDIA GPUs (the build brings its CUDA runtime, the driver is enough);
ROCm on AMD GPUs where AMD's ROCm libraries are installed (llama.cpp's ROCm
builds do not ship hipBLAS/rocBLAS); Metal on Apple silicon; Vulkan on any
other GPU, and as the fallback for NVIDIA and AMD; the CPU without a usable
GPU. Each build is asked which GPUs it sees (``llama-server --list-devices``)
before it is kept, and the next one is tried when it sees none.
``SWISSDOCAI_LLAMA_BACKEND`` (cuda / rocm / vulkan / sycl / metal / cpu)
forces one.

The model is Google's quantization-aware 4-bit Gemma 4 E4B: on the PII
benchmark it found 99.7% of the personal values in a
single pass, in 5.5 s per document on a 7900 XTX, with 3.3 GB on the GPU — the
best of the small models tried (Qwen3.5 2–9B, Gemma 4 E2B, Ministral 3 3B/8B,
Granite 4.2 3B/8B). Its per-layer embeddings stay in system memory, so it
needs much less GPU memory than its 5 GB file.

The page reader (transcription for the PII step) is GLM-OCR, also through
llama.cpp, with its vision projector (``READER_MODEL``).
"""

import atexit
import ctypes
import functools
import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LLAMA_CPP_BUILD = "b11541"  # pinned: a server release whose API and chat templates we tested
_RELEASES = "https://github.com/ggml-org/llama.cpp/releases"

# SHA-256 and size of the release files the app may download, so that the first start needs no
# GitHub API (rate-limited behind a company's shared address) and shows the real download size.
# From api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11541; other builds ask the API.
_ASSETS: dict[str, tuple[str, int]] = {
    "cudart-llama-b11541-bin-ubuntu-cuda-12.8-x64.tar.gz": ("6c8acf749cb80a5932b8c5ba16476c3626ce72b09bcfe9078c27b1087ae39c4f", 594377950),
    "cudart-llama-b11541-bin-ubuntu-cuda-13.4-x64.tar.gz": ("4d6dba50d1d16b3987b6a8cff73a58338ea1eba7ded1d3295ea174b4f515a746", 440236543),
    "cudart-llama-bin-win-cuda-12.4-x64.zip": ("8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", 391443627),
    "cudart-llama-bin-win-cuda-13.4-x64.zip": ("738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668", 423535356),
    "llama-b11541-bin-macos-arm64.tar.gz": ("226929037cfcb956f50ae1d54bac13d3d9446d8d819c439580e1e2c00197da5e", 12078846),
    "llama-b11541-bin-macos-x64.tar.gz": ("f74c3e79f68e9d0099bd2adc45c72208ec9748cd39e4098eeef7e49011445565", 11602041),
    "llama-b11541-bin-ubuntu-arm64.tar.gz": ("fcd44afd8ecdd2068d0c8505c0f96fa32014b4814d0dc74947a471deddcb9998", 13799565),
    "llama-b11541-bin-ubuntu-cuda-12.8-x64.tar.gz": ("551ee13d02dd42599861c8ffbd71c7a110f379a51cb9c5947cb799cdbbc610b2", 171990585),
    "llama-b11541-bin-ubuntu-cuda-13.4-x64.tar.gz": ("2d6f7589eb46f80905cd8fadcc41c67a0a5f5d6fe96ddd4fb37fd7a73a0872d5", 152841931),
    "llama-b11541-bin-ubuntu-rocm-10.0-x64.tar.gz": ("22d3c8165192f88bbf9c77e2a72d44dede698faf5679a738b610291e3d0755ee", 244931286),
    "llama-b11541-bin-ubuntu-vulkan-arm64.tar.gz": ("cfc991be4e09135ab00870582cbb0f403b48683b9f5d33d547ee327a2a604c78", 24967480),
    "llama-b11541-bin-ubuntu-vulkan-x64.tar.gz": ("bf91ce14ddfda55c01c8b8846aa7d6093a07b0213913b6216ed4a2d6ea78b1f8", 31770592),
    "llama-b11541-bin-ubuntu-x64.tar.gz": ("36ca310be4405acb7ba59b32dbdce32bb9cd7b1ef95358997b7eaed2c6f0188b", 17809316),
    "llama-b11541-bin-win-cpu-arm64.zip": ("8f112d39c0bde44451ae5878709fdf2196e8d680cb8cb9354ca5332337f0942b", 12343725),
    "llama-b11541-bin-win-cpu-x64.zip": ("cdc0535d11038bb337dfba3c15682050eaa9c07aedd0f915b4c09e8a8e8c0a5c", 19518702),
    "llama-b11541-bin-win-cuda-12.4-x64.zip": ("c5ccdd22e6b11c013efd6a931f93c3707e6fa5ab2f90df5c7d1838785540ece3", 265224711),
    "llama-b11541-bin-win-cuda-13.4-x64.zip": ("927672eae5bb9cf4dde89a00a764c33e69e98db4b3d8be42f8990bf383cfdf61", 153464749),
    "llama-b11541-bin-win-rocm-10.0-x64.zip": ("cdcb0b4482fe80ede8c561e9e52406a8ea7d3b6a6586b77f5a9564882ec32ce5", 257974995),
    "llama-b11541-bin-win-vulkan-arm64.zip": ("c31124767a0c61dc78e021f574dfb2d0b5a8a0837525a72bf0a0dfacc1ffcade", 26018413),
    "llama-b11541-bin-win-vulkan-x64.zip": ("37bba511d120222f13c3d520dd875f739a530cc124e78f7cbce0026de0173d8a", 33464528),
}

# Windows: helper processes without a console window of their own (the app's engine has none)
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


@dataclass
class ModelSpec:
    repo: str  # Hugging Face repo with GGUF files
    file: str  # the quantization to use
    size_gb: float  # file size
    vram_gb: float  # GPU memory with every layer on the GPU, at ``ctx``
    sha256: str = ""  # of ``file``, checked once after the download
    revision: str = "main"  # repo commit the files are taken from
    ctx: int = 16384
    mmproj: str = ""  # vision projector file in the same repo (image input)
    mmproj_sha256: str = ""
    mmproj_size_gb: float = 0.0


DEFAULT_MODEL = "gemma4-e4b"
READER_MODEL = "glm-ocr"
MODELS: dict[str, ModelSpec] = {
    DEFAULT_MODEL: ModelSpec("google/gemma-4-E4B-it-qat-q4_0-gguf", "gemma-4-E4B_q4_0-it.gguf", 5.15, 4.0,
                             "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
                             "4b4a2c1d584be7264f87aac328a1bc739ce81b6c"),
    READER_MODEL: ModelSpec("ggml-org/GLM-OCR-GGUF", "GLM-OCR-Q8_0.gguf", 0.95, 2.5,
                            "45bc244a6446aff850521dc41f18bc8d7105ad5f0c2c8c28af04e7cc4f4d50b1",
                            "65a42de1148dbed2297e922b5dbc7d9b70c36578",
                            mmproj="mmproj-GLM-OCR-Q8_0.gguf",
                            mmproj_sha256="9c4b58e33e316ed142eb5dcb41abec3844d3e6e5dc361ffb782c3fa9d175141f",
                            mmproj_size_gb=0.48),
}


@dataclass
class GPU:
    vendor: str  # nvidia | amd | intel | apple | other
    name: str
    vram_gb: float  # 0 when unknown or shared with the system (integrated GPUs)
    driver: str = ""  # NVIDIA driver version, e.g. "581.57"
    compute: float = 0.0  # NVIDIA compute capability, e.g. 8.9


@dataclass
class Hardware:
    system: str  # Windows | Linux | Darwin
    machine: str  # x86_64 | arm64 ...
    ram_gb: float
    gpus: list[GPU] = field(default_factory=list)

    @property
    def best_gpu(self) -> GPU | None:
        usable = [g for g in self.gpus if g.vendor != "other"]
        return max(usable, key=lambda g: g.vram_gb, default=None)


# --- hardware detection ----------------------------------------------------------------

_VENDOR_IDS = {"10de": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel"}


def _ram_gb() -> float:
    system = platform.system()
    try:
        if system == "Windows":
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_avail", ctypes.c_ulonglong), ("virt_total", ctypes.c_ulonglong),
                            ("virt_avail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 2**30
        if system == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except Exception as error:  # noqa: BLE001 — detection must never break the run
        logger.debug("RAM detection failed: %s", error)
    return 0.0


def _nvidia_gpus() -> list[GPU]:
    """NVIDIA GPUs with memory, driver version and compute capability (which CUDA build runs)."""
    smi = shutil.which("nvidia-smi") or next(
        (p for p in [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation/NVSMI/nvidia-smi.exe"]
         if platform.system() == "Windows" and p.exists()), None)
    if not smi:
        return []
    gpus = []
    # compute_cap needs a driver from 2021 on; older ones answer the shorter query
    for fields in ("name,memory.total,driver_version,compute_cap", "name,memory.total,driver_version"):
        try:
            out = subprocess.check_output([str(smi), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                          text=True, timeout=10, stderr=subprocess.DEVNULL, **_NO_WINDOW)
        except Exception as error:  # noqa: BLE001 — detection must never break the run
            logger.debug("nvidia-smi --query-gpu=%s: %s", fields, error)
            continue
        for line in out.strip().splitlines():
            values = [v.strip() for v in line.rsplit(",", fields.count(","))]
            try:
                compute = float(values[3]) if len(values) > 3 else 0.0
                gpus.append(GPU("nvidia", values[0], float(values[1]) / 1024, values[2], compute))
            except (ValueError, IndexError):
                continue
        break
    return gpus


def _windows_gpus() -> list[GPU]:
    """Display adapters from the registry: vendor id and dedicated memory (64-bit value)."""
    import winreg

    gpus = []
    path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
    except OSError:
        return []
    for i in range(64):
        try:
            sub = winreg.OpenKey(root, f"{i:04d}")
        except OSError:
            break

        def value(name):
            try:
                return winreg.QueryValueEx(sub, name)[0]
            except OSError:
                return None

        device = str(value("MatchingDeviceId") or "").lower()
        if "ven_" not in device:
            continue
        vendor = _VENDOR_IDS.get(device.split("ven_")[1][:4], "other")
        memory = value("HardwareInformation.qwMemorySize") or value("HardwareInformation.MemorySize") or 0
        if isinstance(memory, bytes):
            memory = int.from_bytes(memory[:8], "little")
        gpus.append(GPU(vendor, str(value("DriverDesc") or device), float(memory) / 2**30))
    return gpus


def _linux_gpus() -> list[GPU]:
    gpus = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        try:
            vendor = _VENDOR_IDS.get(card.joinpath("device/vendor").read_text().strip()[2:], "other")
        except OSError:
            continue
        vram = card / "device/mem_info_vram_total"
        size = int(vram.read_text()) / 2**30 if vram.exists() else 0.0
        if vendor != "nvidia":  # NVIDIA comes from nvidia-smi
            gpus.append(GPU(vendor, card.name, size))
    return gpus


@functools.cache
def detect_hardware() -> Hardware:
    """This machine's system, memory and GPUs (detected once per process)."""
    system, machine = platform.system(), platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    hw = Hardware(system, machine, _ram_gb())
    hw.gpus = _nvidia_gpus()
    if system == "Windows":
        hw.gpus += [g for g in _windows_gpus() if g.vendor != "nvidia" or not hw.gpus]
    elif system == "Linux":
        hw.gpus += _linux_gpus()
    elif system == "Darwin" and machine == "arm64":
        hw.gpus.append(GPU("apple", "Apple silicon (unified memory)", hw.ram_gb))
    return hw


# --- llama.cpp builds ------------------------------------------------------------------


BACKEND_NAMES = {"cuda": "NVIDIA CUDA", "rocm": "AMD ROCm", "vulkan": "Vulkan", "metal": "Apple Metal",
                 "sycl": "Intel SYCL", "cpu": "CPU"}
# how each build names its GPUs in ``llama-server --list-devices`` ("Vulkan0: AMD Radeon ...")
_DEVICE_PREFIXES = {"cuda": ("CUDA",), "rocm": ("ROCm",), "vulkan": ("Vulkan",), "metal": ("MTL", "Metal"),
                    "sycl": ("SYCL",)}
_SOFTWARE_GPUS = ("llvmpipe", "lavapipe", "swiftshader", "basic render")  # CPU emulations, slower than the CPU


def cuda_version(hw: Hardware) -> str | None:
    """The llama.cpp CUDA build this NVIDIA GPU and driver run: "13" (driver 580 or newer, Turing
    or newer), "12" (older drivers, GPUs back to Maxwell), None (no NVIDIA GPU, a driver too old,
    or a GPU only CUDA 13 knows with a driver older than that)."""
    gpu = hw.best_gpu
    if gpu is None or gpu.vendor != "nvidia":
        return None
    try:
        driver = int(gpu.driver.split(".")[0])
    except ValueError:
        return None
    if driver >= 580 and gpu.compute >= 7.5:
        return "13"
    if driver >= (528 if hw.system == "Windows" else 525) and 5.0 <= gpu.compute < 10:
        return "12"
    return None


def rocm_libraries(system: str | None = None) -> Path | None:
    """The folder with AMD's hipBLAS and rocBLAS, which llama.cpp's ROCm builds need but do not
    ship (the Radeon driver brings only the HIP runtime): SWISSDOCAI_ROCM_LIBS, else the HIP SDK
    (HIP_PATH) on Windows, ROCm (ROCM_PATH, /opt/rocm) on Linux. None: use Vulkan."""
    system = system or platform.system()
    folders = [os.environ.get("SWISSDOCAI_ROCM_LIBS")]
    if system == "Windows":
        folders.append(os.environ.get("HIP_PATH") and str(Path(os.environ["HIP_PATH"]) / "bin"))
        pattern = "hipblas.dll"
    else:
        folders.append(str(Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "lib"))
        pattern = "libhipblas.so*"
    return next((Path(f) for f in folders if f and next(Path(f).glob(pattern), None)), None)


def _fits_on_gpu(gpu: GPU) -> bool:
    """Whether the smaller of the two models fits in this GPU's own memory (else a large CUDA or
    ROCm download would gain nothing)."""
    return gpu.vram_gb >= min(spec.vram_gb for spec in MODELS.values()) + 1


def backend_candidates(hw: Hardware) -> list[str]:
    """The llama.cpp builds worth trying on this machine, fastest first; the first that sees a GPU
    is kept (``ensure_server``). Every build also runs on the CPU, which is the last resort."""
    gpu = hw.best_gpu
    if hw.system == "Darwin":
        return ["metal", "cpu"] if gpu and gpu.vendor == "apple" else ["cpu"]
    if gpu is None:
        return ["cpu"]
    fast = []
    if hw.machine == "x86_64" and _fits_on_gpu(gpu):
        if gpu.vendor == "nvidia" and cuda_version(hw):
            fast.append("cuda")
        if gpu.vendor == "amd" and rocm_libraries(hw.system):
            fast.append("rocm")
    return fast + ["vulkan", "cpu"]


def choose_backend(hw: Hardware) -> str:
    """The backend llama.cpp runs with: SWISSDOCAI_LLAMA_BACKEND, else the one kept on first use,
    else (nothing tried yet) the first candidate."""
    forced = os.environ.get("SWISSDOCAI_LLAMA_BACKEND")
    if forced:
        return forced
    choice = chosen_backend(hw)
    if choice and _installed(hw, choice["backend"]):
        return choice["backend"]
    return backend_candidates(hw)[0]


def release_assets(hw: Hardware, backend: str, build: str = LLAMA_CPP_BUILD) -> list[str]:
    """File names of the llama.cpp release build for this machine and backend: the build, then
    (CUDA) the CUDA runtime it needs, unpacked next to it."""
    arch = "arm64" if hw.machine == "arm64" else "x64"
    if hw.system == "Darwin":
        return [f"llama-{build}-bin-macos-{arch}.tar.gz"]
    cuda = cuda_version(hw) or "12"
    if hw.system == "Windows":
        cuda_flavor = {"12": "cuda-12.4", "13": "cuda-13.4"}[cuda]
        flavor = {"cpu": "cpu", "vulkan": "vulkan", "cuda": cuda_flavor, "rocm": "rocm-10.0", "sycl": "sycl"}[backend]
        assets = [f"llama-{build}-bin-win-{flavor}-{arch}.zip"]
        return assets + ([f"cudart-llama-bin-win-{flavor}-{arch}.zip"] if backend == "cuda" else [])
    cuda_flavor = {"12": "cuda-12.8", "13": "cuda-13.4"}[cuda]
    flavor = {"cpu": "", "vulkan": "vulkan-", "cuda": f"{cuda_flavor}-", "rocm": "rocm-10.0-",
              "sycl": "sycl-fp16-"}[backend]
    assets = [f"llama-{build}-bin-ubuntu-{flavor}{arch}.tar.gz"]
    return assets + ([f"cudart-llama-{build}-bin-ubuntu-{flavor}{arch}.tar.gz"] if backend == "cuda" else [])


def release_asset(hw: Hardware, backend: str, build: str = LLAMA_CPP_BUILD) -> str:
    """File name of the llama.cpp release build for this machine and backend."""
    return release_assets(hw, backend, build)[0]


def server_download_gb(hw: Hardware, backend: str, build: str = LLAMA_CPP_BUILD) -> float:
    """What the build for ``backend`` downloads, in GB."""
    return sum(_ASSETS.get(asset, ("", 50_000_000))[1] for asset in release_assets(hw, backend, build)) / 1e9


def cache_dir() -> Path:
    base = os.environ.get("SWISSDOCAI_CACHE")
    if base:
        return Path(base)
    if platform.system() == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "swissdocai"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "swissdocai"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(target: Path, sha256: str) -> bool:
    """Whether ``target`` was downloaded and checked against ``sha256`` before."""
    mark = target.with_name(target.name + ".ok")
    return target.exists() and (not sha256 or (mark.exists() and mark.read_text().strip() == sha256))


def download(url: str, target: Path, sha256: str = "", progress=None, label: str = "") -> Path:
    """``url`` into ``target`` once: resumes a broken download, checks the SHA-256 (when given),
    reports ``progress(label, done, total)`` in bytes."""
    import httpx

    if _checked(target, sha256):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    logger.info("downloading %s", url)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                      timeout=httpx.Timeout(60, read=300)) as response:
        if response.status_code == 200:
            done = 0  # the server sends the whole file
        elif response.status_code != 206:
            response.raise_for_status()
        total = done + int(response.headers.get("content-length", 0))
        with part.open("ab" if done else "wb") as f:
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(label or target.name, done, total)
    if sha256 and _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(f"{target.name}: checksum mismatch, download discarded")
    part.replace(target)
    target.with_name(target.name + ".ok").write_text(sha256)
    return target


def _asset_digest(build: str, asset: str) -> str | None:
    """SHA-256 GitHub records for a release asset (None when the API does not say)."""
    import httpx

    try:
        release = httpx.get(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{build}", timeout=30).json()
    except Exception:  # noqa: BLE001
        return None
    for item in release.get("assets", []):
        if item.get("name") == asset and str(item.get("digest", "")).startswith("sha256:"):
            return item["digest"].split(":", 1)[1]
    return None


def _asset_sha256(build: str, asset: str) -> str:
    if asset in _ASSETS and build == LLAMA_CPP_BUILD:
        return _ASSETS[asset][0]
    return _asset_digest(build, asset) or ""


def _build_dir(hw: Hardware, backend: str, build: str = LLAMA_CPP_BUILD) -> Path:
    asset = release_asset(hw, backend, build)
    return cache_dir() / "llama.cpp" / asset.rsplit(".zip", 1)[0].rsplit(".tar.gz", 1)[0]


def _server_name(hw: Hardware) -> str:
    return "llama-server.exe" if hw.system == "Windows" else "llama-server"


def _installed(hw: Hardware, backend: str, build: str = LLAMA_CPP_BUILD) -> Path | None:
    """``llama-server`` of this build when it was unpacked completely before."""
    target = _build_dir(hw, backend, build)
    if not (target / ".complete").exists():
        return None
    return next(target.rglob(_server_name(hw)), None)


# The Visual C++ runtime that llama.cpp's Windows builds need but do not ship
_VC_RUNTIME = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def _add_vc_runtime(folder: Path) -> None:
    """Put this program's own Visual C++ runtime (the app's engine bundles it) next to llama.cpp:
    a PC without the Visual C++ Redistributable then runs it, and one with an outdated runtime
    does not crash in it (Windows looks next to the program first)."""
    sources = [Path(p) for p in (getattr(sys, "_MEIPASS", None), Path(sys.executable).parent, sys.base_prefix) if p]
    # only a complete set: an msvcp140 newer than its vcruntime140 fails to load
    source = next((d for d in sources if all((d / name).exists() for name in _VC_RUNTIME)), None)
    for name in _VC_RUNTIME if source else ():
        if not (folder / name).exists():
            shutil.copy2(source / name, folder / name)


def _install(hw: Hardware, backend: str, build: str = LLAMA_CPP_BUILD, progress=None) -> Path:
    """``llama-server`` of the build for ``backend``, downloaded and unpacked once."""
    found = _installed(hw, backend, build)
    if found:
        return found
    target = _build_dir(hw, backend, build)
    shutil.rmtree(target, ignore_errors=True)  # a build unpacked only partly before
    for i, asset in enumerate(release_assets(hw, backend, build)):
        archive = download(f"{_RELEASES}/download/{build}/{asset}", cache_dir() / "llama.cpp" / asset,
                           _asset_sha256(build, asset), progress, f"llama.cpp ({BACKEND_NAMES.get(backend, backend)})")
        into = target if i == 0 else target / "_runtime"
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(into)
        else:
            with tarfile.open(archive) as t:
                t.extractall(into, filter="data")
        archive.unlink()
        archive.with_name(archive.name + ".ok").unlink(missing_ok=True)
    server = next(target.rglob(_server_name(hw)), None)
    if not server:
        raise RuntimeError(f"{release_asset(hw, backend, build)} contains no {_server_name(hw)}")
    runtime = target / "_runtime"
    if runtime.exists():  # the CUDA runtime goes next to the server, where its libraries look for it
        for path in sorted(runtime.rglob("*")):
            if path.is_symlink() or path.is_file():
                path.replace(server.parent / path.name)
        shutil.rmtree(runtime)
    if hw.system == "Windows":
        _add_vc_runtime(server.parent)
    else:
        server.chmod(0o755)
    (target / ".complete").write_text(build)
    return server


def _server_env(backend: str) -> dict | None:
    """The environment for llama-server: ROCm builds find AMD's libraries (``rocm_libraries``)."""
    libs = rocm_libraries() if backend == "rocm" else None
    if libs is None:
        return None
    var = "PATH" if platform.system() == "Windows" else "LD_LIBRARY_PATH"
    return {**os.environ, var: os.pathsep.join(p for p in (str(libs), os.environ.get(var, "")) if p)}


def probe_devices(server: Path, backend: str) -> list[str]:
    """The GPUs this llama.cpp build can use here, as ``llama-server --list-devices`` names them
    ("Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23749 MiB free)"); CPU emulations left out."""
    try:
        done = subprocess.run([str(server), "--list-devices"], capture_output=True, text=True, timeout=120,
                              check=False, cwd=server.parent, env=_server_env(backend), **_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("llama.cpp (%s) could not list its devices: %s", backend, error)
        return []
    lines = [line.strip() for line in (done.stdout + "\n" + done.stderr).splitlines()]
    devices = [line for line in lines if line.startswith(_DEVICE_PREFIXES.get(backend, ("-",))) and ":" in line]
    return [d for d in devices if not any(s in d.lower() for s in _SOFTWARE_GPUS)]


def _choice_file() -> Path:
    return cache_dir() / "llama.cpp" / "backend.json"


def _gpu_name(hw: Hardware) -> str:
    return hw.best_gpu.name if hw.best_gpu else ""


def chosen_backend(hw: Hardware | None = None, build: str = LLAMA_CPP_BUILD) -> dict | None:
    """The backend kept on first use, {"build", "backend", "gpu", "devices"}; None when nothing
    was chosen yet, or for another llama.cpp build or (with ``hw``) another GPU: choose again."""
    try:
        choice = json.loads(_choice_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(choice, dict) or choice.get("build") != build:
        return None
    return None if hw is not None and choice.get("gpu") != _gpu_name(hw) else choice


def ensure_server(hw: Hardware, backend: str | None = None, build: str = LLAMA_CPP_BUILD,
                  progress=None) -> tuple[Path, str]:
    """``llama-server`` and the backend it runs with. ``backend`` (or SWISSDOCAI_LLAMA_BACKEND)
    forces one. Otherwise the builds of ``backend_candidates`` are tried in order, each downloaded
    once and asked which GPUs it sees; the first that sees one is kept for every later start, and
    a build that sees none is deleted again. The CPU build needs no GPU."""
    backend = backend or os.environ.get("SWISSDOCAI_LLAMA_BACKEND")
    if backend:
        return _install(hw, backend, build, progress), backend
    choice = chosen_backend(hw, build)
    if choice:
        server = _installed(hw, choice["backend"], build)
        if server:
            return server, choice["backend"]
    candidates = backend_candidates(hw)
    for i, candidate in enumerate(candidates):
        server = _install(hw, candidate, build, progress)
        devices = probe_devices(server, candidate) if candidate != "cpu" else []
        if devices or candidate == "cpu":
            _choice_file().write_text(json.dumps({"build": build, "backend": candidate, "gpu": _gpu_name(hw),
                                                  "devices": devices}), encoding="utf-8")
            logger.info("llama.cpp runs with %s on %s", candidate, devices or "the CPU")
            return server, candidate
        logger.warning("llama.cpp's %s build sees no GPU here; trying %s", candidate, candidates[i + 1:])
        if all(_build_dir(hw, later, build) != _build_dir(hw, candidate, build) for later in candidates[i + 1:]):
            shutil.rmtree(_build_dir(hw, candidate, build), ignore_errors=True)
    raise RuntimeError("no llama.cpp build runs on this machine")


def _hf_file(spec: ModelSpec, file: str, sha256: str, progress=None) -> Path:
    url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{file}"
    return download(url, cache_dir() / "models" / file, sha256, progress, file)


def ensure_model(spec: ModelSpec, progress=None) -> tuple[Path, Path | None]:
    """The model file and its vision projector (None for text-only models), downloaded once."""
    model = _hf_file(spec, spec.file, spec.sha256, progress)
    return model, _hf_file(spec, spec.mmproj, spec.mmproj_sha256, progress) if spec.mmproj else None


def server_ready(hw: Hardware | None = None) -> bool:
    """Whether llama.cpp is on disk with its backend chosen (no download, no probing needed)."""
    hw = hw or detect_hardware()
    forced = os.environ.get("SWISSDOCAI_LLAMA_BACKEND")
    if forced:
        return _installed(hw, forced) is not None
    choice = chosen_backend(hw)
    return choice is not None and _installed(hw, choice["backend"]) is not None


def model_ready(spec: ModelSpec) -> bool:
    """Whether the model (and its vision projector) were downloaded and checked."""
    files = [(spec.file, spec.sha256)] + ([(spec.mmproj, spec.mmproj_sha256)] if spec.mmproj else [])
    return all(_checked(cache_dir() / "models" / f, sha) for f, sha in files)


def is_ready(hw: Hardware | None = None) -> bool:
    """Whether the server build and both models are on disk (no download needed)."""
    return server_ready(hw) and all(model_ready(spec) for spec in MODELS.values())


# --- the server ------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_job = None


def _end_with_this_process(process: subprocess.Popen) -> None:
    """Windows: end ``process`` together with this one, even when this one is killed or crashes
    (a job object that kills its processes when its last handle, ours, closes) — a model server
    left behind would hold GPU memory until the next restart."""
    global _job
    if sys.platform != "win32":
        return
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    if _job is None:
        class Limits(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            _fields_ = [("user_time", ctypes.c_int64 * 2), ("flags", wintypes.DWORD),
                        ("working_set", ctypes.c_size_t * 2), ("process_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD),
                        ("io", ctypes.c_uint64 * 6), ("memory", ctypes.c_size_t * 4)]

        job = kernel32.CreateJobObjectW(None, None)
        limits = Limits(flags=0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            logger.debug("no job object: %s", ctypes.get_last_error())
            return
        _job = job
    if not kernel32.AssignProcessToJobObject(_job, int(process._handle)):
        logger.debug("process not in the job object: %s", ctypes.get_last_error())


_running: set["LocalEngine"] = set()


def stop_all() -> None:
    """Stop every model server this process started (the app's engine on exit)."""
    for engine in list(_running):
        engine.stop()


class LocalEngine:
    """A llama.cpp server for one model on this machine (started lazily, stopped at exit).
    ``backend`` forces a llama.cpp build; by default the hardware chooses (``ensure_server``)."""

    def __init__(self, model: str = DEFAULT_MODEL, backend: str | None = None, hardware: Hardware | None = None):
        if model not in MODELS:
            raise ValueError(f"unknown local model {model!r}; choose from {sorted(MODELS)}")
        self.name = model
        self.spec = MODELS[model]
        self.hw = hardware or detect_hardware()
        self.forced_backend = backend
        self.backend = backend or choose_backend(self.hw)
        self.process: subprocess.Popen | None = None
        self.port = 0
        self._log = None

    def gpu_layers(self) -> int:
        """All layers on the GPU when the model fits comfortably, none otherwise (llama.cpp then
        runs on the CPU); 999 = all."""
        gpu = self.hw.best_gpu
        if self.backend == "cpu" or gpu is None:
            return 0
        memory = gpu.vram_gb or self.hw.ram_gb / 2  # integrated GPUs share system memory
        return 999 if memory >= self.spec.vram_gb + 1 else 0

    def start(self) -> "LocalEngine":
        if self.process and self.process.poll() is None:
            return self
        server, self.backend = ensure_server(self.hw, self.forced_backend)
        model, mmproj = ensure_model(self.spec)
        layers = self.gpu_layers()
        try:
            self._launch(server, model, mmproj, layers)
        except RuntimeError as error:
            if not layers:
                raise
            # e.g. GPU memory taken by another program, or a driver the build does not get along with
            logger.warning("llama.cpp failed on the GPU (%s), starting %s on the CPU: %s", self.backend, self.name,
                           error)
            self.stop()
            self._launch(server, model, mmproj, 0, cpu_only=True)
        return self

    def _launch(self, server: Path, model: Path, mmproj: Path | None, layers: int, cpu_only: bool = False) -> None:
        self.port = _free_port()
        cmd = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.spec.ctx),
               "-ngl", str(layers), "--parallel", "1", "--jinja", "--no-webui", "--reasoning-budget", "0"]
        if mmproj:
            cmd += ["--mmproj", str(mmproj)]
        if cpu_only:
            cmd += ["--device", "none"]
        logger.info("starting llama.cpp (%s, %s) with %s", self.backend, "GPU" if layers else "CPU", self.name)
        # the CPU retry appends: the log keeps why the GPU failed
        self._log = (cache_dir() / f"llama-server-{self.name}.log").open("a" if cpu_only else "w", encoding="utf-8")
        self.process = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, cwd=server.parent,
                                        env=_server_env(self.backend), **_NO_WINDOW)
        _end_with_this_process(self.process)
        _running.add(self)
        atexit.register(self.stop)
        self._wait_ready()

    def _wait_ready(self, timeout: float = 300.0) -> None:
        import httpx

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.process.returncode}); "
                                   f"see {cache_dir() / f'llama-server-{self.name}.log'}")
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=2).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("llama-server did not become ready")

    def client(self) -> "_EngineChat":
        """A chat client (like llm.ChatClient) that starts the server on its first request."""
        return _EngineChat(self)

    def reader(self) -> "_EngineReader":
        """A page reader on this engine (for READER_MODEL) whose ``close()`` stops the server."""
        return _EngineReader(self)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        _running.discard(self)
        if self._log:
            self._log.close()
            self._log = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class _EngineChat:
    def __init__(self, engine: LocalEngine):
        self.engine, self.model = engine, engine.name

    def chat(self, *args, **kwargs) -> str:
        from .llm import ChatClient

        self.engine.start()
        return ChatClient(self.model, base_url=f"http://127.0.0.1:{self.engine.port}/v1", api_key="").chat(
            *args, **kwargs)


class _EngineReader:
    def __init__(self, engine: LocalEngine):
        from .llm import ApiMarkdownModel
        from .prompts import prompt_for

        self.engine = engine
        self.model = ApiMarkdownModel(engine.client(), prompt=prompt_for("zai-org/GLM-OCR"))
        self.model_id = f"{engine.spec.repo}/{engine.spec.file}"

    def __call__(self, image) -> str:
        return self.model(image)

    def close(self) -> None:
        self.engine.stop()


def describe(hw: Hardware | None = None) -> dict:
    """What ``swissdocai doctor`` and the app's first start show: the hardware and what runs on it.
    ``chosen``: the backend was tried on this machine and kept (else it is the expected one)."""
    hw = hw or detect_hardware()
    backend = choose_backend(hw)
    engine = LocalEngine(hardware=hw, backend=backend)
    choice = chosen_backend(hw)
    gpu = hw.best_gpu
    try:
        from .words import onnx_providers

        ocr = onnx_providers()[0].removesuffix("ExecutionProvider")
    except ImportError:  # no ONNX Runtime
        ocr = None
    return {
        "system": f"{hw.system} {hw.machine}",
        "ramGB": round(hw.ram_gb, 1),
        "gpus": [{"vendor": g.vendor, "name": g.name, "vramGB": round(g.vram_gb, 1),
                  **({"driver": g.driver, "compute": g.compute} if g.driver else {})} for g in hw.gpus],
        "gpu": gpu.name if gpu else None,
        "vramGB": round(gpu.vram_gb, 1) if gpu else 0,
        "backend": backend,
        "backendName": BACKEND_NAMES.get(backend, backend),
        "candidates": backend_candidates(hw),
        "chosen": bool(choice) and choice["backend"] == backend,
        "devices": choice["devices"] if choice else [],
        "llamaCppBuild": release_assets(hw, backend),
        "model": f"{engine.spec.repo}/{engine.spec.file}",
        "runsOn": "GPU" if engine.gpu_layers() else "CPU",
        "ocr": ocr,
        "cache": str(cache_dir()),
    }


if __name__ == "__main__":
    print(json.dumps(describe(), indent=1))
