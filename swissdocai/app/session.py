"""What the app runs: settings, the models (created once, reused for
every document), the first-start download, and analyze / save of one document."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from ..pii import DEFAULT_CATEGORIES
from ..policy import Policy, default_policy
from ..redact import RedactionJob, analyze_document, render_job

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    engine: str = "local"  # PII step: local (Gemma on this machine) | swissdocai (the SwissDocAI PII service)
    reader: str = "local"  # page transcription for the PII step: local (GLM-OCR) | swissdocai (the SaaS OCR) | none
    dpi: int = 200
    language: str = "de"  # de | fr | it | en
    policy: dict | None = None  # the user's black-out / keep lists (policy.Policy); None: the standard ones
    always_redact_contacts: bool = True  # safety rule: every phone number and e-mail address, whatever the lists say
    always_redact_titled_names: bool = True  # safety rule: the name after Dr., Prof., PD
    redact_qr_codes: bool = True  # every QR code, whatever it holds (a QR-bill's: names, addresses, IBAN)

    def effective_policy(self) -> Policy:
        return Policy.from_dict(self.policy) if self.policy else default_policy(self.language)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        settings = cls()
        if "use_reader" in data and "reader" not in data:  # settings written before the reader choice
            data = {**data, "reader": "local" if data["use_reader"] else "none"}
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        if settings.engine not in ("local", "swissdocai"):  # an engine no longer offered
            settings.engine = "local"
        return settings

    def to_dict(self) -> dict:
        return asdict(self)


class Session:
    """The models behind the app. The OCR model and the local PII model stay loaded between
    documents; the page reader runs per document and frees its memory after the last page."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ocr = None
        self._pii_engine = None

    # --- first start ---------------------------------------------------------------------

    def downloads_needed(self) -> list[tuple[str, float]]:
        """(what, GB) still to download for local use; empty when everything is there."""
        from .. import engine
        from ..words import _onnx_model_urls

        needed = []
        hw = engine.detect_hardware()
        local = self._local_models()
        if local and not engine.server_ready(hw):
            backend = engine.choose_backend(hw)
            needed.append((f"llama.cpp ({engine.BACKEND_NAMES.get(backend, backend)})",
                           engine.server_download_gb(hw, backend)))
        for spec in local:
            if not engine.model_ready(spec):
                needed.append((spec.file, spec.size_gb + spec.mmproj_size_gb))
        from ..words import OnnxWordReader

        ocr_dir = engine.cache_dir() / "onnxtr" / "models"
        reader = OnnxWordReader()
        for url in _onnx_model_urls(reader.det_arch, reader.reco_arch):
            if not (ocr_dir / url.rpartition("/")[2]).exists():
                needed.append((url.rpartition("/")[2], 0.1))
        return needed

    def download_all(self, progress) -> None:
        from .. import engine
        from ..words import ensure_ocr_models

        hw = engine.detect_hardware()
        from ..words import OnnxWordReader

        reader = OnnxWordReader()
        ensure_ocr_models(reader.det_arch, reader.reco_arch, progress=progress)
        local = self._local_models()
        if local:  # the llama.cpp build for this hardware, tried before it is kept
            engine.ensure_server(hw, progress=progress)
        for spec in local:
            engine.ensure_model(spec, progress)

    def hardware(self) -> dict:
        """What the local models run on here: the GPU, the llama.cpp backend (tried and kept, or
        the one expected before the first download), GPU or CPU, and the OCR's device."""
        from ..engine import describe

        info = describe()
        return {key: info[key] for key in ("gpu", "vramGB", "backend", "backendName", "chosen", "runsOn", "ocr")}

    def _local_models(self) -> list:
        """The models that run on this machine with these settings (PII model, page reader)."""
        from .. import engine

        wanted = {engine.DEFAULT_MODEL: self.settings.engine == "local",
                  engine.READER_MODEL: self.settings.reader == "local"}
        return [engine.MODELS[name] for name, on in wanted.items() if on]

    # --- the models ----------------------------------------------------------------------

    def ocr(self):
        if self._ocr is None:
            from ..words import OnnxWordReader

            self._ocr = OnnxWordReader()
        return self._ocr

    def pii_client(self):
        """What finds the personal data: the local model's chat client, or the SwissDocAI PII service."""
        if self.settings.engine == "swissdocai":
            from ..saas import SaasClient, SwissDocAIPii
            from .secret import load_token

            return SwissDocAIPii(SaasClient(token=load_token()))
        if self._pii_engine is None:
            from ..engine import DEFAULT_MODEL, LocalEngine

            self._pii_engine = LocalEngine(DEFAULT_MODEL)
        return self._pii_engine.client()

    def reader(self):
        if self.settings.reader == "swissdocai":
            from ..saas import SaasClient, SwissDocAIReader
            from .secret import load_token

            return SwissDocAIReader(SaasClient(token=load_token()))
        if self.settings.reader != "local":
            return None
        from ..engine import READER_MODEL, LocalEngine

        return LocalEngine(READER_MODEL).reader()

    # --- one document --------------------------------------------------------------------

    def analyze(self, path: str | Path, progress) -> RedactionJob:
        off = {c for c, on in (("contact", self.settings.always_redact_contacts),
                               ("person", self.settings.always_redact_titled_names)) if not on}
        rules = tuple(c for c in DEFAULT_CATEGORIES if c not in off)
        return analyze_document(path, self.pii_client(), reader=self.reader(), ocr=self.ocr(),
                                categories=DEFAULT_CATEGORIES, dpi=self.settings.dpi, progress=progress,
                                policy=self.settings.effective_policy(), rule_categories=rules,
                                qr_codes=self.settings.redact_qr_codes)

    def save(self, job: RedactionJob, out_path: str | Path, progress, write_report: bool = True) -> dict:
        report = render_job(job, out_path, verify=True, progress=progress)
        if write_report:
            Path(out_path).with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                                  encoding="utf-8")
        return report

    def close(self) -> None:
        """Stop the model servers, also a page reader's that is still running."""
        from ..engine import stop_all

        if self._pii_engine is not None:
            self._pii_engine.stop()
        stop_all()
