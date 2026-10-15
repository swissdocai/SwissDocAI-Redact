"""End-to-end document analysis: file → pages → words + markdown → result.

    analyzer = DocumentAnalyzer(markdown_model=MarkdownModel("path/to/model"))
    result = analyzer.analyze("invoice.pdf")   # Document-Intelligence-style dict

Word layer per page (``word_mode``):
- "auto": the PDF text layer when present and sane, else OCR;
- "pdf":  text layer only (pages without one get no words);
- "ocr":  always OCR, even for born-digital PDFs;
- "hybrid": both, for completeness (redaction): digital pages keep the
  text layer and gain the OCR words where it has none (text in images,
  form field values, outlined text); scanned pages are OCR'd and keep
  only those text-layer words the OCR did not find.
"""

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageSequence

from .result import PageInput, build_analyze_result
from .selection_marks import (
    SelectionMark,
    detect_selection_marks,
    pdf_selection_marks,
    remove_mark_words,
)
from .words import PageWords, Word, group_lines, pdf_page_words, text_layer_usable

logger = logging.getLogger(__name__)

_PAGE_BREAK_SPLIT = re.compile(r"\s*<!--\s*PageBreak\s*-->\s*")


@dataclass
class SourcePage:
    page_number: int
    image: Image.Image
    pdf_words: list[Word] | None  # text layer words (PDF inputs only)
    unit: str  # result unit: inch for PDFs, pixel for images
    scale: float  # result units per image pixel
    pdf_marks: list[SelectionMark] | None = None  # checkboxes from the PDF itself
    scanned: bool = False  # page is mainly one embedded image (a scan)
    size_pt: tuple[float, float] | None = None  # page size in PDF points (as displayed)
    images: list[tuple[float, float, float, float]] | None = None  # embedded images, pixel boxes


def merge_word_layers(primary: list[Word], extra: list[Word], max_overlap: float = 0.3) -> list[Word]:
    """``primary`` plus every ``extra`` word whose box is not already covered
    (more than ``max_overlap`` of its area overlapping a primary word)."""
    added = []
    for word in extra:
        x0, y0, x1, y1 = word.bbox
        area = max(1e-6, (x1 - x0) * (y1 - y0))
        covered = 0.0
        for other in primary:
            ix = min(x1, other.bbox[2]) - max(x0, other.bbox[0])
            iy = min(y1, other.bbox[3]) - max(y0, other.bbox[1])
            if ix > 0 and iy > 0:
                covered = max(covered, ix * iy / area)
        if covered <= max_overlap:
            added.append(word)
    return list(primary) + added


def load_pages(path: str | Path, dpi: int = 150, page_numbers: set[int] | None = None) -> list[SourcePage]:
    """Render/load every requested page of a PDF or (multi-page) image file."""
    path = Path(path)
    pages: list[SourcePage] = []
    if path.suffix.lower() == ".pdf":
        import pdfplumber
        import pypdfium2 as pdfium

        data = path.read_bytes()
        pdf = pdfium.PdfDocument(data)
        pdf.init_forms()  # draw form field values as a viewer shows them
        try:
            with pdfplumber.open(io.BytesIO(data)) as plumber:
                for i in range(len(pdf)):
                    number = i + 1
                    if page_numbers and number not in page_numbers:
                        continue
                    page = plumber.pages[i]
                    image = pdf[i].render(scale=dpi / 72).to_pil().convert("RGB")
                    words = pdf_page_words(page, dpi / 72)
                    marks, words = pdf_selection_marks(page, dpi / 72, words)
                    page_area = float(page.width * page.height) or 1.0
                    scanned = any(
                        img["width"] * img["height"] > 0.5 * page_area for img in page.images
                    )
                    scale = dpi / 72  # as for the words
                    images = [(img["x0"] * scale, img["top"] * scale, img["x1"] * scale, img["bottom"] * scale)
                              for img in page.images if page.rotation % 360 == 0]
                    size_pt = (image.width * 72 / dpi, image.height * 72 / dpi)
                    pages.append(SourcePage(number, image, words, "inch", 1 / dpi, marks, scanned, size_pt, images))
        finally:
            pdf.close()
        return pages

    with Image.open(path) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            number = i + 1
            if page_numbers and number not in page_numbers:
                continue
            frame = frame.convert("RGB")
            # images carry no reliable size: lay them out at 150 dpi
            size_pt = (frame.width * 72 / 150, frame.height * 72 / 150)
            pages.append(SourcePage(number, frame, None, "pixel", 1.0, size_pt=size_pt))
    return pages


def split_markdown_pages(markdown: str) -> list[str]:
    """Split a multi-page markdown file at ``<!-- PageBreak -->`` markers."""
    return [part.strip() for part in _PAGE_BREAK_SPLIT.split(markdown)]


class DocumentAnalyzer:
    """Runs the word layer and the markdown model and merges them.

    Args:
        markdown_model: Callable image → markdown (e.g. MarkdownModel). Not
            needed when ``analyze`` gets precomputed ``markdown_pages``.
        word_mode: "auto" | "pdf" | "ocr" (see module docstring).
        ocr: Callable image → PageWords (default: DoctrWordReader, created
            on first use).
        dpi: Render resolution for PDF pages.
        model_id: Reported as ``analyzeResult.modelId``.
    """

    def __init__(self, markdown_model=None, word_mode: str = "auto", ocr=None, dpi: int = 150,
                 model_id: str | None = None):
        if word_mode not in ("auto", "pdf", "ocr", "hybrid"):
            raise ValueError(f"word_mode must be auto, pdf, ocr or hybrid — got {word_mode!r}")
        self.markdown_model = markdown_model
        self.word_mode = word_mode
        self._ocr = ocr
        self.dpi = dpi
        self.model_id = model_id or getattr(markdown_model, "model_id", None) or "swissdocai-layout"

    @property
    def ocr(self):
        if self._ocr is None:
            from .words import DoctrWordReader

            self._ocr = DoctrWordReader()
        return self._ocr

    def page_words(self, page: SourcePage) -> PageWords:
        """Words and selection marks of one page, from the PDF or from OCR."""
        width, height = page.image.size
        if self.word_mode == "hybrid":
            return self._hybrid_words(page)
        if self.word_mode != "ocr" and page.pdf_words is not None:
            if self.word_mode == "pdf" or text_layer_usable(page.pdf_words):
                words = list(page.pdf_words)
                marks = list(page.pdf_marks or [])
                if not marks and page.scanned:  # scan with an OCR text layer
                    marks = detect_selection_marks(page.image, self.dpi)
                    words = remove_mark_words(words, marks)
                return PageWords(width, height, words, group_lines(words), source="pdf", marks=marks)
        if self.word_mode == "pdf":
            return PageWords(width, height, [], [], source="none")
        page_words = self.ocr(page.image)
        # Image inputs carry no reliable resolution: assume the long side
        # spans an A4 page (11.69 in) to size the checkbox search.
        dpi = self.dpi if page.unit == "inch" else max(width, height) / 11.69
        marks = detect_selection_marks(page.image, dpi)
        kept = remove_mark_words(page_words.words, marks)
        if len(kept) != len(page_words.words):
            page_words = PageWords(width, height, kept, group_lines(kept), source="ocr")
        page_words.marks = marks
        return page_words

    def _hybrid_words(self, page: SourcePage) -> PageWords:
        width, height = page.image.size
        ocr = self.ocr(page.image)
        text_layer = list(page.pdf_words or [])
        usable = bool(text_layer) and text_layer_usable(text_layer)
        if usable and not page.scanned:
            words, source, marks = merge_word_layers(text_layer, ocr.words), "pdf+ocr", list(page.pdf_marks or [])
        else:
            words = merge_word_layers(ocr.words, text_layer if usable else [])
            source, marks = "ocr+pdf" if usable else "ocr", list(page.pdf_marks or [])
        if not marks:
            dpi = self.dpi if page.unit == "inch" else max(width, height) / 11.69
            marks = detect_selection_marks(page.image, dpi)
            words = remove_mark_words(words, marks)
        return PageWords(width, height, words, group_lines(words), source=source, marks=marks)

    def analyze(self, path: str | Path, page_numbers: set[int] | None = None,
                markdown_pages: list[str] | None = None) -> dict:
        pages = load_pages(path, self.dpi, page_numbers)
        if not pages:
            raise ValueError(f"No pages loaded from {path}")
        if markdown_pages is not None and len(markdown_pages) != len(pages):
            raise ValueError(
                f"{len(markdown_pages)} markdown pages given for {len(pages)} document pages"
            )
        if markdown_pages is None and self.markdown_model is None:
            raise ValueError("Need a markdown_model or precomputed markdown_pages")

        inputs = []
        for k, page in enumerate(pages):
            words = self.page_words(page)
            markdown = markdown_pages[k] if markdown_pages is not None else self.markdown_model(page.image)
            logger.info(
                "page %d: %d words (%s), %d markdown chars",
                page.page_number, len(words.words), words.source, len(markdown),
            )
            inputs.append(PageInput(page.page_number, markdown, words, page.unit, page.scale))

        return build_analyze_result(
            inputs,
            self.model_id,
            extra={"wordMode": self.word_mode, "renderDpi": self.dpi, "source": Path(path).name},
        )
