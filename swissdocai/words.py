"""Word layer: every word on a page with its box and confidence.

Sources:
- ``pdf_page_words`` reads the text layer of born-digital PDFs (exact
  boxes, confidence 1.0) via pdfplumber;
- ``OnnxWordReader`` runs docTR's text detection and our Swiss recognizer
  through ONNX Runtime on page images (scans, photos, PDFs without a usable
  text layer) — what the app uses, no PyTorch;
- ``DoctrWordReader`` runs the same with docTR and PyTorch.

Coordinates are always pixels of the page image (top-left origin), so word
boxes, the page reader's text and the image stay in one frame.
"""

import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Word:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in page pixels
    confidence: float = 1.0
    line: int = -1  # index into PageWords.lines
    font: str = ""  # PDF font name (text layer only)


@dataclass
class PageWords:
    width: float  # page image size in pixels
    height: float
    words: list[Word] = field(default_factory=list)
    lines: list[list[int]] = field(default_factory=list)  # word indices per line
    source: str = "pdf"  # "pdf" | "ocr"
    marks: list = field(default_factory=list)  # SelectionMark (selection_marks.py)


# --- PDF text layer ---------------------------------------------------------------


def pdf_page_words(page, scale: float) -> list[Word]:
    """Words of one pdfplumber page, scaled from PDF points to pixels.

    ``use_text_flow`` keeps the content-stream order, which for generated
    PDFs (Chromium, Word, LaTeX) is the reading order — far better than a
    geometric sort for multi-column layouts.
    """
    raw = page.extract_words(use_text_flow=True, keep_blank_chars=False, return_chars=True)
    words = []
    for w in raw:
        if not w.get("upright", True):
            continue
        text = w["text"].strip()
        if not text:
            continue
        words.append(
            Word(
                text=text,
                bbox=(w["x0"] * scale, w["top"] * scale, w["x1"] * scale, w["bottom"] * scale),
                font=w["chars"][0].get("fontname", "").split("+")[-1] if w.get("chars") else "",
            )
        )
    return words


def text_layer_usable(words: list[Word], min_words: int = 3, max_bad_ratio: float = 0.05) -> bool:
    """Reject empty text layers and broken font encodings.

    Fonts without a ToUnicode map surface as "(cid:123)", U+FFFD or
    private-use code points — such a layer has boxes but wrong text, and
    OCR is the better source.
    """
    if len(words) < min_words:
        return False
    chars = "".join(w.text for w in words)
    if not chars:
        return False
    bad = chars.count("(cid:") * 5
    bad += sum(1 for ch in chars if ch == "�" or unicodedata.category(ch) in ("Co", "Cn"))
    return bad / len(chars) <= max_bad_ratio


def group_lines(words: list[Word]) -> list[list[int]]:
    """Group reading-ordered words into text lines.

    A new line starts when the next word leaves the current line's
    vertical band, jumps backwards, or jumps far right (a column gap).
    Sets ``Word.line`` and returns word indices per line.
    """
    lines: list[list[int]] = []
    current: list[int] = []
    for idx, word in enumerate(words):
        if current:
            last = words[current[-1]]
            top = min(words[k].bbox[1] for k in current)
            bottom = max(words[k].bbox[3] for k in current)
            height = max(1.0, bottom - top)
            center = (word.bbox[1] + word.bbox[3]) / 2
            same_band = top - 0.2 * height <= center <= bottom + 0.2 * height
            gap = word.bbox[0] - last.bbox[2]
            if not same_band or gap < -0.5 * height or gap > 3.0 * height:
                lines.append(current)
                current = []
        current.append(idx)
    if current:
        lines.append(current)
    for line_idx, members in enumerate(lines):
        for k in members:
            words[k].line = line_idx
    return lines


# --- OCR (docTR) --------------------------------------------------------------------


def load_doctr_model(spec: str, task: str):
    """A docTR architecture name, a hub repo ("user/model"), or a fine-tuned
    ``<name>.pt`` with its ``<name>.json`` config (scripts/train_doctr.py)."""
    if spec.endswith(".pt"):
        import doctr.models
        import torch

        config = json.loads(Path(spec).with_suffix(".json").read_text(encoding="utf-8"))
        builder = getattr(doctr.models, config["arch"])
        kwargs = {"vocab": config["vocab"]} if task == "reco" else {}
        model = builder(pretrained=False, pretrained_backbone=False, **kwargs)
        model.load_state_dict(torch.load(spec, map_location="cpu"))
        return model
    if "/" in spec:
        from doctr.models import from_hub

        return from_hub(spec)
    return spec


class DoctrWordReader:
    """docTR detection + recognition, returning word boxes in pixels.

    Models load lazily on first use. ``det_arch``/``reco_arch`` may be docTR
    architecture names, Hugging Face hub repos, or fine-tuned ``.pt`` files
    from scripts/train_doctr.py (e.g. the Swiss-vocabulary recognizer).
    """

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "parseq",
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.device = device
        self.batch_size = batch_size
        self._predictor = None

    def _load(self):
        import torch
        from doctr.models import ocr_predictor

        predictor = ocr_predictor(
            det_arch=load_doctr_model(self.det_arch, "det"),
            reco_arch=load_doctr_model(self.reco_arch, "reco"),
            pretrained=True,
            assume_straight_pages=True,
            preserve_aspect_ratio=True,
            symmetric_pad=True,
            reco_bs=self.batch_size,
        )
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor = predictor.to(device)
        vocab = getattr(self._predictor.reco_predictor.model, "vocab", "")
        missing = [ch for ch in "äöüÄÖÜàéèêçôûî" if ch not in vocab]
        if missing:
            logger.warning(
                "docTR recognizer %s lacks characters %s — consider a multilingual model",
                self.reco_arch, "".join(missing),
            )
        logger.info("docTR loaded (%s + %s) on %s", self.det_arch, self.reco_arch, device)

    def __call__(self, image) -> PageWords:
        import numpy as np

        if self._predictor is None:
            self._load()
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        return _page_words(self._predictor([rgb]).pages[0], width, height)


def _page_words(page, width: int, height: int) -> PageWords:
    """A docTR / OnnxTR result page as PageWords in pixels."""
    words: list[Word] = []
    lines: list[list[int]] = []
    for block in page.blocks:
        for line in block.lines:
            members = []
            for w in line.words:
                (x0, y0), (x1, y1) = w.geometry
                if not w.value.strip():
                    continue
                members.append(len(words))
                words.append(
                    Word(
                        text=w.value,
                        bbox=(x0 * width, y0 * height, x1 * width, y1 * height),
                        confidence=float(w.confidence),
                        line=len(lines),
                    )
                )
            if members:
                lines.append(members)
    return PageWords(width, height, words, lines, source="ocr")


# --- OCR without PyTorch (OnnxTR) -----------------------------------------------------


# The recognizer fine-tuned on Swiss documents (umlauts, accents, Swiss vocabulary), exported
# from output/doctr_models/parseq_swiss.pt by scripts/export_onnx.py; shipped with the package.
SWISS_RECOGNIZER = Path(__file__).parent / "models" / "parseq_swiss.onnx"


CPU_ONLY = ["CPUExecutionProvider"]


def onnx_providers() -> list:
    """ONNX Runtime execution providers, fastest first: DirectML (any GPU on Windows),
    CUDA, then the CPU. Only the CPU without a GPU (or with SWISSDOCAI_OCR_DEVICE=cpu):
    DirectML would run on Windows' software renderer, much slower than the CPU itself.
    Never CoreML (Apple): it cannot build the PARSeq recognizer ("Unable to get shape
    for output: /Squeeze_output_0") and would compute in half precision, which the
    recognizer was not measured with; Apple's CPUs read a page in about a second."""
    import onnxruntime as ort

    from .engine import detect_hardware

    if os.environ.get("SWISSDOCAI_OCR_DEVICE", "").lower() == "cpu" or detect_hardware().best_gpu is None:
        return CPU_ONLY
    available = ort.get_available_providers()
    return [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider") if p in available] + CPU_ONLY


def _onnx_model_urls(det_arch: str, reco_arch: str) -> list[str]:
    import importlib

    families = {"detection": ("differentiable_binarization", "fast", "linknet"),
                "recognition": ("crnn", "master", "parseq", "sar", "viptr", "vitstr")}
    urls = []
    for arch, task in ((det_arch, "detection"), (reco_arch, "recognition")):
        if arch.endswith(".onnx"):
            continue
        for family in families[task]:
            cfgs = importlib.import_module(f"onnxtr.models.{task}.models.{family}").default_cfgs
            if arch in cfgs:
                urls.append(cfgs[arch]["url"])
                break
        else:
            raise ValueError(f"unknown OnnxTR architecture {arch!r}")
    return urls


def ensure_ocr_models(det_arch: str = "db_resnet50", reco_arch: str = "parseq", progress=None) -> None:
    """Download the OCR models once (OnnxTR would fetch them silently on first use), with
    ``progress(what, done, total)``. OnnxTR checks each file against the hash in its name."""
    from .engine import cache_dir, download

    for url in _onnx_model_urls(det_arch, reco_arch):
        name = url.rpartition("/")[2]
        download(url, cache_dir() / "onnxtr" / "models" / name, progress=progress, label=name)


class OnnxWordReader:
    """docTR's models through OnnxTR and ONNX Runtime: the same words and boxes as
    DoctrWordReader, without PyTorch (what the desktop app ships).

    The recognizer defaults to the Swiss one (SWISS_RECOGNIZER). ``det_arch``/``reco_arch``
    are OnnxTR architecture names (their ONNX files
    download once, into SWISSDOCAI_CACHE/onnxtr) or exported ``.onnx`` files
    with their ``.json`` (architecture, vocab) from scripts/export_onnx.py.
    """

    def __init__(self, det_arch: str = "db_resnet50", reco_arch: str | None = None, batch_size: int = 64):
        self.det_arch = det_arch
        self.reco_arch = reco_arch or (str(SWISS_RECOGNIZER) if SWISS_RECOGNIZER.exists() else "parseq")
        self.batch_size = batch_size
        self._predictor = None
        self.providers: list[str] = []

    def _build(self, providers: list[str]):
        import onnxruntime as ort
        from onnxtr import models
        from onnxtr.models import EngineConfig, ocr_predictor

        from .engine import cache_dir

        os.environ.setdefault("ONNXTR_CACHE_DIR", str(cache_dir() / "onnxtr"))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if "DmlExecutionProvider" in providers:  # DirectML: no memory patterns, sequential only
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        config = EngineConfig(providers=providers, session_options=options)

        def model(spec: str):
            if not spec.endswith(".onnx"):
                return spec
            meta = json.loads(Path(spec).with_suffix(".json").read_text(encoding="utf-8"))
            kwargs = {"vocab": meta["vocab"]} if meta.get("task") == "reco" else {}
            return getattr(models, meta["arch"])(spec, engine_cfg=config, **kwargs)

        return ocr_predictor(
            det_arch=model(self.det_arch), reco_arch=model(self.reco_arch),
            assume_straight_pages=True, preserve_aspect_ratio=True, symmetric_pad=True,
            reco_bs=self.batch_size, det_engine_cfg=config, reco_engine_cfg=config,
        )

    def _load(self, providers: list[str] | None = None):
        self.providers = providers or onnx_providers()
        try:
            self._predictor = self._build(self.providers)
        except Exception as error:  # noqa: BLE001 — ONNX Runtime's errors are no RuntimeError
            if self.providers == CPU_ONLY:
                raise
            logger.warning("OCR cannot load on %s, using the CPU: %s", self.providers[0], error)
            self.providers = CPU_ONLY
            self._predictor = self._build(self.providers)
        logger.info("OnnxTR loaded (%s + %s) on %s", self.det_arch, self.reco_arch, self.providers[0])

    def __call__(self, image) -> PageWords:
        import numpy as np

        if self._predictor is None:
            self._load()
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        try:
            result = self._predictor([rgb])
        except Exception as error:  # noqa: BLE001 — a GPU provider that fails only when it runs
            if self.providers == CPU_ONLY:
                raise
            logger.warning("OCR failed on %s, reading again on the CPU: %s", self.providers[0], error)
            self._load(CPU_ONLY)
            result = self._predictor([rgb])
        return _page_words(result.pages[0], width, height)
