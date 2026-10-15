"""The SwissDocAI API token, kept out of the settings file.

Windows: encrypted with DPAPI for the current user (only this Windows account can
read it back). Elsewhere: a file only the user may read. SWISSDOCAI_API_KEY in
the environment or .env wins (development).
"""

import os
import sys
from pathlib import Path

from ..engine import cache_dir


def _path() -> Path:
    return cache_dir() / "swissdocai-token.bin"


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_char))]

    source = Blob(len(data), ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_char)))
    target = Blob()
    crypt = ctypes.windll.crypt32
    call = crypt.CryptProtectData if protect else crypt.CryptUnprotectData
    if not call(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(target)):  # 0x1: no UI
        raise OSError("DPAPI failed")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        ctypes.windll.kernel32.LocalFree(target.data)


def save_token(token: str) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not token:
        path.unlink(missing_ok=True)
        return
    data = token.encode("utf-8")
    if sys.platform == "win32":
        data = _dpapi(data, protect=True)
    path.write_bytes(data)
    if sys.platform != "win32":
        path.chmod(0o600)


def load_token() -> str | None:
    from ..saas import api_key

    from_env = api_key()
    if from_env:
        return from_env
    path = _path()
    if not path.exists():
        return None
    data = path.read_bytes()
    try:
        if sys.platform == "win32":
            data = _dpapi(data, protect=False)
        return data.decode("utf-8") or None
    except (OSError, UnicodeDecodeError):
        return None


def token_hint() -> str | None:
    """The token's last characters, to show which one is set (never the token)."""
    token = load_token()
    return f"…{token[-4:]}" if token else None


def from_environment() -> bool:
    return bool(os.environ.get("SWISSDOCAI_API_KEY"))
