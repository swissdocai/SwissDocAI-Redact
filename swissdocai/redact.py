"""Automatic redaction: find personal data, black it out, rebuild the PDF.

    engine = LocalEngine(DEFAULT_MODEL)
    report = redact_document("letter.pdf", "letter.redacted.pdf", client=engine.client(),
                             reader=LocalEngine(READER_MODEL).reader())

1. Word layer per page (hybrid: PDF text layer + OCR where it has no words).
2. A reader model (optional) transcribes each page: the text the PII step
   reads. Every page word missing from that transcription is added as plain
   lines, so the PII step always sees all words.
3. Personal data: the chat model (pii.detect_with_llm), or the SwissDocAI PII
   service (saas.SwissDocAIPii), plus rules (pii.detect_with_rules) on both the
   page words and the transcription.
4. Every found string is located on every page, so a name found once is
   redacted everywhere: on a letters-and-digits stream of the page words
   (immune to spacing, line breaks and punctuation), through the
   transcription's links to the words (catches OCR misreadings), and only
   then fuzzily.
5. Black bars over the words (padded, one bar per run of words on a line),
   over signature images next to a redacted name, and — unless switched off — over
   every QR code (``find_qr_codes``); the pages are rasterized and written
   as a new PDF of the original page sizes — no text layer, metadata,
   attachments or annotations survive.
6. Verification: the redacted pages are OCR'd again and searched for every
   found string.

Steps 1–4 are ``analyze_document`` (a ``RedactionJob``), 5–6 ``render_job``; in
between a reviewer may keep words readable, redact more words or draw boxes
(the desktop app). ``redact_document`` runs both without review.
"""

import io
import logging
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from .pii import DEFAULT_CATEGORIES, Entity, detect_with_llm, detect_with_rules
from .saas import new_document_id
from .pipeline import DocumentAnalyzer, SourcePage, load_pages
from .result import PageInput, build_analyze_result
from .words import PageWords, Word

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


# --- letters-and-digits streams --------------------------------------------------------


@dataclass
class Stream:
    """Casefolded letters and digits of a text, each char mapped to its owner."""

    chars: str = ""
    owner: list[int] = field(default_factory=list)  # word index or text offset per char
    boundary: list[bool] = field(default_factory=list)  # char starts a token


def _fold(text: str):
    """(char, source index) for every letter/digit of ``text`` after NFKC + casefold."""
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).casefold():
            if c.isalnum():
                yield c, i


def _append(stream: Stream, text: str, owner_of, chars: list[str]) -> None:
    prev_src, prev_char = None, None
    for c, i in _fold(text):
        starts = prev_src is None or i > prev_src + 1 or c.isdigit() != prev_char.isdigit()
        chars.append(c)
        stream.owner.append(owner_of(i))
        stream.boundary.append(starts)
        prev_src, prev_char = i, c


def word_stream(words: list[Word]) -> Stream:
    stream, chars = Stream(), []
    for j, word in enumerate(words):
        _append(stream, word.text, lambda i, j=j: j, chars)
    stream.chars = "".join(chars)
    return stream


def text_stream(text: str) -> Stream:
    stream, chars = Stream(), []
    _append(stream, text, lambda i: i, chars)
    stream.chars = "".join(chars)
    return stream


def find_all(stream: Stream, needle: str) -> list[tuple[int, int]]:
    """Occurrences of ``needle`` that start and end on token boundaries."""
    hits, start = [], stream.chars.find(needle)
    while start != -1:
        end = start + len(needle)
        if stream.boundary[start] and (end == len(stream.chars) or stream.boundary[end]):
            hits.append((start, end))
        start = stream.chars.find(needle, start + 1)
    return hits


# --- locating found strings ------------------------------------------------------------


@dataclass
class Occurrence:
    """One place where a found value sits on a page, and the words under it."""

    entity: Entity
    needle: str  # the value's letters and digits
    words: list[int]
    how: str  # exact | reader | fuzzy
    missing: int = 0  # characters of the value no word on the page accounts for


@dataclass
class PageState:
    source: SourcePage
    words: PageWords
    reader_text: str = ""
    stream: Stream | None = None
    word_spans: list[tuple[int, int] | None] = field(default_factory=list)  # content span per word
    redact: dict[int, set] = field(default_factory=lambda: defaultdict(set))  # word -> {(category, how)}
    occurrences: list[Occurrence] = field(default_factory=list)
    qr_codes: list[Box] = field(default_factory=list)  # blacked out whole (find_qr_codes)
    # review: words kept readable, boxes drawn by hand, signature and QR code boxes switched off (pixels)
    removed: set[int] = field(default_factory=set)
    manual_boxes: list[Box] = field(default_factory=list)
    hidden_boxes: set[Box] = field(default_factory=set)


def locate(entity: Entity, pages: list[PageState], content: Stream | None, fuzzy_threshold: float = 88.0) -> int:
    """Mark every occurrence of ``entity`` on every page; returns the number of words it covers."""
    from rapidfuzz import fuzz

    needle = "".join(c for c, _ in _fold(entity.text))
    if len(needle) < 2:
        return 0
    marked = 0

    def occurrence(page: PageState, indices, how: str):
        nonlocal marked
        indices = sorted(set(indices))
        if not indices:
            return
        for j in indices:
            page.redact[j].add((entity.category, how))
        marked += len(indices)
        page.occurrences.append(Occurrence(entity, needle, indices, how))

    for page in pages:
        for start, end in find_all(page.stream, needle):
            occurrence(page, page.stream.owner[start:end], "exact")
    if content is not None:
        for start, end in find_all(content, needle):
            lo, hi = content.owner[start], content.owner[end - 1] + 1
            for page in pages:
                occurrence(page, [j for j, span in enumerate(page.word_spans)
                                  if span and span[0] < hi and lo < span[1]], "reader")
    if marked == 0 and len(needle) >= 5:
        for page in pages:
            if not page.stream.chars:
                continue
            hit = fuzz.partial_ratio_alignment(needle, page.stream.chars, score_cutoff=fuzzy_threshold)
            if hit:
                occurrence(page, page.stream.owner[hit.dest_start:hit.dest_end], "fuzzy")
    return marked


# --- completeness: every character of a found value needs a box -------------------------


def _folded(text: str) -> str:
    return "".join(c for c, _ in _fold(text))


def missing_segments(needle: str, covered: str) -> list[str]:
    """Pieces of ``needle`` that ``covered`` (the words' letters and digits) lacks — deletions
    in the edit script, so a misread character ("l2" for "12") does not count as missing."""
    from rapidfuzz.distance import Levenshtein

    segments, current, last = [], "", None
    for op in Levenshtein.editops(needle, covered):
        if op.tag != "delete":
            continue
        if last is not None and op.src_pos != last + 1:
            segments.append(current)
            current = ""
        current += needle[op.src_pos]
        last = op.src_pos
    if current:
        segments.append(current)
    return segments


def _covered(page: PageState, indices: list[int]) -> str:
    words = page.words.words
    ordered = sorted(indices, key=lambda j: (words[j].line, words[j].bbox[0]))
    return "".join(_folded(words[j].text) for j in ordered)


def _part_of_value(text: str, box: Box, span: Box, needle: str, segments: list[str]) -> bool:
    """Does a word with ``text`` at ``box`` belong to the found value whose words span ``span``?

    Its text must be part of the value — never just its position: inside the
    value's span, any piece of the value; right next to it, a missing piece
    (or a piece containing one, "415" for a missing "41")."""
    x0, y0, x1, y1 = span
    h = max(1.0, y1 - y0)
    bx0, by0, bx1, by1 = box
    if not text or not (by0 < y1 and y0 < by1):
        return False
    centre = (bx0 + bx1) / 2
    if x0 < centre < x1:
        return text in needle
    if x0 - 1.5 * h <= bx1 and bx0 <= x1 + 1.5 * h:
        return any(text == s or (s in text and text in needle) or (len(text) >= 2 and text in s) for s in segments)
    return False


def complete_occurrence(page: PageState, occ: Occurrence, ocr, max_zoom: float = 4.0) -> None:
    """Find boxes for the characters of ``occ`` that no linked page word covers.

    1. Page words on the value's line that the transcription did not link —
       typically misread by OCR ("66" for "266") — are taken if their text is
       part of the value (``_part_of_value``).
    2. If characters are still missing, the line is cut out, enlarged and run
       through the OCR again; new words are taken under the same rule.
    Nothing is taken for its position alone, so no unrelated word on the line
    can be blacked out. Whatever is still missing stays in ``occ.missing``
    for review."""
    words = page.words.words
    if not missing_segments(occ.needle, _covered(page, occ.words)):
        return

    def take(j: int, how: str) -> None:
        page.redact[j].add((occ.entity.category, how))
        occ.words.append(j)

    image = page.source.image
    # the value's words per visual line — not per word-layer line, which breaks at wide gaps:
    # "077 [455 74 missed] 56" would become two crops that each cut the missing digits in half
    bands: list[list[int]] = []
    for j in sorted(occ.words, key=lambda j: (words[j].bbox[1] + words[j].bbox[3]) / 2):
        top, bottom = words[j].bbox[1], words[j].bbox[3]
        for band in bands:
            b_top, b_bottom = min(words[k].bbox[1] for k in band), max(words[k].bbox[3] for k in band)
            if min(bottom, b_bottom) - max(top, b_top) > 0.5 * min(bottom - top, b_bottom - b_top):
                band.append(j)
                break
        else:
            bands.append([j])
    def span_of(members: list[int]) -> Box:
        return (min(words[j].bbox[0] for j in members), min(words[j].bbox[1] for j in members),
                max(words[j].bbox[2] for j in members), max(words[j].bbox[3] for j in members))

    def take_all(members: list[int], candidates: list, how: str) -> None:
        """Take every candidate (page word index, or a Word read again) that is part of the
        value, again as the value's span grows: in "078 858 [21 13]" first "858" next to
        "21", then "078" next to "858". A Word read again joins the page only when taken."""
        pending = [c for c in candidates if not (isinstance(c, int) and c in occ.words)]

        def word(c) -> Word:
            return words[c] if isinstance(c, int) else c

        def boxed(box) -> bool:  # already under one of the value's words (several crops read the same digits)
            return any(min(box[2], words[j].bbox[2]) - max(box[0], words[j].bbox[0]) > 0.5 * (box[2] - box[0])
                       for j in occ.words)

        while pending:
            segments = missing_segments(occ.needle, _covered(page, occ.words))
            if not segments:
                return
            span = span_of(members)
            taken = [c for c in pending if _part_of_value(_folded(word(c).text), word(c).bbox, span, occ.needle,
                                                          segments)]
            if not taken:
                return
            for c in taken:
                pending.remove(c)
                if not isinstance(c, int):
                    if boxed(c.bbox):
                        continue
                    words.append(c)
                    c = len(words) - 1
                take(c, how)
                members.append(c)

    for members in bands:
        take_all(members, list(range(len(words))), "linked-by-text")
        segments = missing_segments(occ.needle, _covered(page, occ.words))
        if not segments:
            continue
        x0, y0, x1, y1 = span_of(members)
        h = max(1.0, y1 - y0)
        # The OCR reads a cut-out line differently depending on where it is cut and how far it is
        # enlarged, so look several times — close around the value and far enough to the sides
        # for every missing character, at two sizes — and pool what is read.
        char_width = (x1 - x0) / max(1, len(_covered(page, members)))
        reaches = sorted({round(3 * h), round(max(3 * h, 1.4 * len("".join(segments)) * char_width + 2 * h))})
        base_zoom = min(max_zoom, max(1.0, 40 / h))
        found_words = []
        for reach in reaches:
            region = (max(0, x0 - reach), max(0, y0 - 0.6 * h), min(image.width, x1 + reach),
                      min(image.height, y1 + 0.6 * h))
            for zoom in sorted({base_zoom, min(max_zoom, 2 * base_zoom)}):
                crop = image.crop(tuple(round(v) for v in region))
                crop = crop.resize((round(crop.width * zoom), round(crop.height * zoom)), Image.LANCZOS)
                crop.info["page_region"], crop.info["zoom"] = region, zoom  # where the crop sits (tests)
                for found in ocr(crop).words:
                    box = (region[0] + found.bbox[0] / zoom, region[1] + found.bbox[1] / zoom,
                           region[0] + found.bbox[2] / zoom, region[1] + found.bbox[3] / zoom)
                    found_words.append(Word(found.text, box, found.confidence, words[members[0]].line))
        take_all(members, found_words, "redetected")
    occ.missing = len("".join(missing_segments(occ.needle, _covered(page, occ.words))))



# --- page texts for the PII step ------------------------------------------------------


def _line_text(words: list[Word], indices: list[int]) -> str:
    return " ".join(words[j].text for j in sorted(indices, key=lambda j: words[j].bbox[0]))


def missing_text(page: PageState) -> str:
    """Page words the transcription lacks, as lines grouped into regions (columns stay apart)."""
    words = page.words.words
    missing = [j for j, span in enumerate(page.word_spans) if span is None]
    lines: dict[int, list[int]] = defaultdict(list)
    for j in missing:
        lines[words[j].line].append(j)
    regions: list[dict] = []
    for members in sorted(lines.values(), key=lambda m: min(words[j].bbox[1] for j in m)):
        x0 = min(words[j].bbox[0] for j in members)
        x1 = max(words[j].bbox[2] for j in members)
        y0 = min(words[j].bbox[1] for j in members)
        y1 = max(words[j].bbox[3] for j in members)
        h = max(1.0, y1 - y0)
        for region in regions:
            if x0 < region["x1"] and region["x0"] < x1 and y0 - region["y1"] < 2.5 * h:
                region["lines"].append(members)
                region.update(x0=min(region["x0"], x0), x1=max(region["x1"], x1), y1=max(region["y1"], y1))
                break
        else:
            regions.append({"x0": x0, "x1": x1, "y1": y1, "lines": [members]})
    return "\n\n".join("\n".join(_line_text(words, m) for m in r["lines"]) for r in regions)


def page_text_for_pii(page: PageState) -> str:
    if not page.reader_text:
        return missing_text(page)
    extra = missing_text(page)
    if not extra:
        return page.reader_text
    return f"{page.reader_text}\n\n[Further text on this page, not in the transcription above]\n{extra}"


def _plain_lines(words: PageWords) -> str:
    return "\n".join(_line_text(words.words, line) for line in words.lines)


# --- bars, signatures, rendering -------------------------------------------------------


def redaction_bars(words: list[Word], indices, pad: float = 0.15) -> list[Box]:
    """One padded bar per run of neighbouring redacted words on a line."""
    by_line: dict[int, list[Word]] = defaultdict(list)
    for j in indices:
        by_line[words[j].line if words[j].line >= 0 else -1 - j].append(words[j])
    bars = []
    for line_words in by_line.values():
        line_words.sort(key=lambda w: w.bbox[0])
        run = [line_words[0]]
        for word in line_words[1:]:
            h = max(1.0, run[-1].bbox[3] - run[-1].bbox[1])
            if word.bbox[0] - run[-1].bbox[2] <= 1.5 * h:
                run.append(word)
            else:
                bars.append(run)
                run = [word]
        bars.append(run)
    out = []
    for run in bars:
        x0, y0 = min(w.bbox[0] for w in run), min(w.bbox[1] for w in run)
        x1, y1 = max(w.bbox[2] for w in run), max(w.bbox[3] for w in run)
        p = pad * max(1.0, y1 - y0)
        out.append((x0 - p, y0 - p, x1 + p, y1 + p))
    return out


def signature_images(page: PageState, person_bars: list[Box], max_area: float = 0.03) -> list[Box]:
    """Small embedded images (signatures) right above or below a redacted name."""
    if not page.source.images or not person_bars:
        return []
    width, height = page.source.image.size
    found = []
    for box in page.source.images:
        x0, y0, x1, y1 = box
        if (x1 - x0) * (y1 - y0) > max_area * width * height:
            continue
        for bx0, by0, bx1, by1 in person_bars:
            h = by1 - by0
            near = by0 - 3 * h <= y1 and y0 <= by1 + 3 * h
            if near and bx0 - 2 * h < x1 and x0 < bx1 + 2 * h:
                found.append(box)
                break
    return found


def find_qr_codes(image: Image.Image, pad: float = 0.06) -> list[Box]:
    """QR codes on the page, as boxes padded by ``pad`` of their size. A code can carry personal
    data that is not printed as text, and in any case all of it: the payment part of a Swiss
    QR-bill holds the names, addresses and IBAN. Codes are only located, not decoded, so an
    unreadable one is found too.

    OpenCV's ArUco-based detector finds codes in scans far more often than its classic one; it
    looks at the page and at the page enlarged 1.5x, which catches some small codes on
    low-resolution scans. Codes of about two pixels per module (10 mm at 150 dpi) can still be
    missed — a reviewer then draws a box."""
    import cv2
    import numpy as np

    gray = np.asarray(image.convert("L"))
    detector = cv2.QRCodeDetectorAruco()
    found: list[Box] = []
    for scale in (1.0, 1.5):
        scaled = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, points = detector.detectMulti(scaled)
        if not ok or points is None:
            continue
        for corners in points:
            xs, ys = corners[:, 0] / scale, corners[:, 1] / scale
            p = pad * max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
            box = (max(0.0, float(xs.min()) - p), max(0.0, float(ys.min()) - p),
                   min(float(image.width), float(xs.max()) + p), min(float(image.height), float(ys.max()) + p))
            for k, other in enumerate(found):  # the same code seen at both sizes: one box around both
                if min(box[2], other[2]) > max(box[0], other[0]) and min(box[3], other[3]) > max(box[1], other[1]):
                    found[k] = (min(box[0], other[0]), min(box[1], other[1]),
                                max(box[2], other[2]), max(box[3], other[3]))
                    break
            else:
                found.append(box)
    return sorted(found, key=lambda box: (box[1], box[0]))  # in reading order


def burn(image: Image.Image, boxes: list[Box]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle((max(0, x0), max(0, y0), min(out.width, x1), min(out.height, y1)), fill=(0, 0, 0))
    return out


def preview(image: Image.Image, labelled: list[tuple[Box, str]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for (x0, y0, x1, y1), label in labelled:
        draw.rectangle((x0, y0, x1, y1), outline=(220, 0, 0, 255), fill=(220, 0, 0, 70), width=2)
        draw.text((x0, max(0, y0 - 11)), label, fill=(220, 0, 0, 255))
    return out


def save_pdf(images: list[Image.Image], sizes_pt: list[tuple[float, float]], path: str | Path,
             quality: int = 92) -> None:
    """Image-only PDF, one page per image at the given size in points."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    try:
        for image, (w, h) in zip(images, sizes_pt):
            page = pdf.new_page(w, h)
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "JPEG", quality=quality, subsampling=0)
            buffer.seek(0)
            obj = pdfium.PdfImage.new(pdf)
            obj.load_jpeg(buffer, inline=True)
            obj.set_matrix(pdfium.PdfMatrix().scale(w, h))
            page.insert_obj(obj)
            page.gen_content()
        pdf.save(str(path))
    finally:
        pdf.close()


# --- the whole thing -------------------------------------------------------------------


@dataclass
class RedactionJob:
    """A document after detection and before the redacted PDF is written — what a reviewer edits."""

    path: Path
    pages: list[PageState]
    entities: list[Entity]
    not_found: list[Entity]
    categories: tuple
    dpi: int
    ocr: object = None  # word reader, for verification
    reader_id: str | None = None
    llm_ids: list = field(default_factory=list)
    seconds: float = 0.0
    qr_codes: bool = False  # QR codes were looked for


def _no_progress(stage: str, done: int, total: int) -> None:
    pass


def analyze_document(path: str | Path, client, reader=None, ocr=None, categories=DEFAULT_CATEGORIES, dpi: int = 200,
                     progress=_no_progress, policy=None, rule_categories=None, qr_codes: bool = True) -> RedactionJob:
    """Steps 1–4: words, transcription, personal data, where it sits. ``progress(stage, done,
    total)`` is called per page ("words", "reading") and per step ("personal data", "locating").
    ``policy``: the user's lists of what to black out and keep (policy.Policy); ``categories``
    then only switches the rules (AHV, IBAN, phone, e-mail …) on and off — or ``rule_categories``.
    ``qr_codes``: black out every QR code on the pages, whatever it holds (False: leave them readable).
    A reader with a ``close()`` (a local llama.cpp reader) is closed after the last page, so
    its memory is free for the PII model."""
    started = time.time()
    document_id = new_document_id()
    for service in (reader, client):  # the SwissDocAI services: one id for all calls about this document
        if hasattr(service, "document_id"):
            service.document_id = document_id
    analyzer = DocumentAnalyzer(word_mode="hybrid", ocr=ocr, dpi=dpi)
    sources = load_pages(path, dpi)
    if not sources:
        raise ValueError(f"No pages loaded from {path}")
    pages = []
    for i, source in enumerate(sources):
        progress("words", i, len(sources))
        pages.append(PageState(source, analyzer.page_words(source),
                               qr_codes=find_qr_codes(source.image) if qr_codes else []))
    for i, page in enumerate(pages):
        if reader:
            progress("reading", i, len(pages))
            page.reader_text = reader(page.source.image)
        logger.info("page %d: %d words (%s)%s%s", page.source.page_number, len(page.words.words), page.words.source,
                    f", {len(page.reader_text)} reader chars" if reader else "",
                    f", {len(page.qr_codes)} QR code(s)" if page.qr_codes else "")
    if hasattr(reader, "close"):
        reader.close()

    # link the transcription to the words; without a reader every word is "missing"
    result = build_analyze_result([PageInput(p.source.page_number, p.reader_text, p.words) for p in pages],
                                  getattr(reader, "model_id", "none"))["analyzeResult"]
    content = text_stream(result["content"]) if reader else None
    for page, out_page in zip(pages, result["pages"]):
        page.stream = word_stream(page.words.words)
        page.word_spans = [(w["span"]["offset"], w["span"]["offset"] + w["span"]["length"]) if "span" in w else None
                           for w in out_page["words"]]

    # personal data: LLM on the full page texts, rules on words and transcription
    progress("personal data", 0, 1)
    texts = [page_text_for_pii(p) for p in pages]
    if hasattr(client, "detect"):  # the SwissDocAI PII service (saas.SwissDocAIPii)
        entities = client.detect(texts, categories, policy)
    else:
        entities = detect_with_llm(client, texts, categories, policy=policy)
    for text in [_plain_lines(p.words) for p in pages] + ([result["content"]] if reader else []):
        entities += [e for _, _, e in detect_with_rules(text, rule_categories or categories)]
    unique = {}
    for e in entities:
        unique.setdefault(("".join(c for c, _ in _fold(e.text)), e.category), e)
    entities = list(unique.values())
    progress("locating", 0, 1)
    not_found = [e for e in entities if locate(e, pages, content) == 0]
    logger.info("%d personal data strings (%d not found on the pages)", len(entities), len(not_found))

    # every character of a found value needs a box (exact matches have one by construction)
    for page in pages:
        for occ in page.occurrences:
            if occ.how != "exact":
                complete_occurrence(page, occ, analyzer.ocr)
    redetected = sum(1 for p in pages for marks in p.redact.values() if any(h == "redetected" for _, h in marks))
    if redetected:
        logger.info("%d words found by looking again at incompletely boxed values", redetected)
    return RedactionJob(Path(path), pages, entities, not_found, tuple(categories), dpi, analyzer.ocr,
                        getattr(reader, "model_id", None), [getattr(client, "model", None)],
                        time.time() - started, qr_codes)


def _word_bars(page: PageState) -> list[tuple[Box, str]]:
    """(bar, category) over the redacted words of ``page`` except those a reviewer kept readable."""
    by_category: dict[str, set] = defaultdict(set)
    for j, marks in page.redact.items():
        if j not in page.removed:
            by_category[sorted(marks)[0][0]].add(j)
    return [(bar, category) for category, indices in by_category.items()
            for bar in redaction_bars(page.words.words, indices)]


def area_boxes(page: PageState, bars: list[tuple[Box, str]] | None = None) -> list[tuple[Box, str]]:
    """(box, category) of the areas blacked out whole: signature images next to a redacted name
    and QR codes — also those a reviewer switched off (``page.hidden_boxes``)."""
    person = [bar for bar, category in (_word_bars(page) if bars is None else bars) if category == "person"]
    return [(box, "signature") for box in signature_images(page, person)] + [(box, "qr_code") for box in page.qr_codes]


def page_boxes(page: PageState) -> list[tuple[Box, str]]:
    """(box, category) to black out on ``page``: bars over its redacted words except those a
    reviewer kept readable, signature images next to a redacted name and QR codes unless a
    reviewer switched them off, the reviewer's boxes."""
    labelled = _word_bars(page)
    labelled += [(box, category) for box, category in area_boxes(page, labelled)
                 if tuple(box) not in page.hidden_boxes]
    labelled += [(box, "manual") for box in page.manual_boxes]
    return labelled


def toggle_at(page: PageState, x: float, y: float) -> bool:
    """A reviewer's click at (x, y) in page pixels: removes a drawn box there, else switches a
    signature or QR code box there off (or on again), else switches the word there (a found word
    is kept readable, a kept one is redacted again, any other word is redacted by hand). False
    when nothing is there."""
    for box in list(page.manual_boxes):
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            page.manual_boxes.remove(box)
            return True
    for box, _ in area_boxes(page):
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            page.hidden_boxes.symmetric_difference_update({tuple(box)})
            return True
    j = next((j for j, w in enumerate(page.words.words)
              if w.bbox[0] <= x <= w.bbox[2] and w.bbox[1] <= y <= w.bbox[3]), None)
    if j is None:
        return False
    if j in page.redact and j not in page.removed:
        page.removed.add(j)
    elif j in page.removed:
        page.removed.discard(j)
    else:
        page.redact[j].add(("manual", "manual"))
    return True


def set_entity_redacted(job: RedactionJob, entity: Entity, redacted: bool) -> None:
    """Black out every occurrence of ``entity`` again, or keep them all readable."""
    for page in job.pages:
        for occ in page.occurrences:
            if occ.entity is entity:
                if redacted:
                    page.removed.difference_update(occ.words)
                else:
                    page.removed.update(occ.words)


def set_qr_code_redacted(page: PageState, index: int, redacted: bool) -> None:
    """Black out the ``index``-th QR code on ``page`` again, or keep it readable."""
    box = tuple(page.qr_codes[index])
    if redacted:
        page.hidden_boxes.discard(box)
    else:
        page.hidden_boxes.add(box)


def kept_readable(job: RedactionJob, entity: Entity) -> bool:
    """Whether a reviewer kept every occurrence of ``entity`` readable."""
    places = [(page, occ) for page in job.pages for occ in page.occurrences if occ.entity is entity]
    return bool(places) and all(set(occ.words) <= page.removed for page, occ in places)


def render_job(job: RedactionJob, out_path: str | Path, verify: bool = True, report_text: bool = False,
               preview_dir: str | Path | None = None, progress=_no_progress) -> dict:
    """Steps 5–6: burn the boxes in, write the image-only PDF, verify; returns the report."""
    started = time.time()
    report_pages, burned, previews = [], [], []
    for i, page in enumerate(job.pages):
        progress("writing", i, len(job.pages))
        words = page.words.words
        labelled = page_boxes(page)
        burned.append(burn(page.source.image, [box for box, _ in labelled]))
        if preview_dir:
            previews.append(preview(page.source.image, labelled))
        to_pt = 72 / job.dpi if page.source.unit == "inch" else 72 / 150
        findings = []
        for j, marks in sorted(page.redact.items()):
            if j in page.removed:
                continue
            item = {"box_pt": [round(v * to_pt, 2) for v in words[j].bbox],
                    "categories": sorted({c for c, _ in marks}), "how": sorted({h for _, h in marks})}
            if report_text:
                item["text"] = words[j].text
            findings.append(item)
        incomplete = []
        for occ in page.occurrences:
            if occ.missing and not set(occ.words) <= page.removed:
                boxes = [words[j].bbox for j in occ.words]
                box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                       max(b[3] for b in boxes))
                incomplete.append({"category": occ.entity.category, "missingChars": occ.missing,
                                   "box_pt": [round(v * to_pt, 2) for v in box],
                                   **({"text": occ.entity.text} if report_text else {})})
        report_pages.append({"page": page.source.page_number, "wordSource": page.words.source,
                             "redactedWords": len(findings), "bars": len(labelled),
                             "signatures": sum(1 for _, c in labelled if c == "signature"),
                             "qrCodes": sum(1 for _, c in labelled if c == "qr_code"),
                             "keptReadable": len(page.removed & set(page.redact)),
                             "manualBoxes": len(page.manual_boxes),
                             "needsReview": incomplete, "words": findings})

    save_pdf(burned, [p.source.size_pt for p in job.pages], out_path)
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        for page, image in zip(job.pages, previews):
            image.save(Path(preview_dir) / f"{job.path.stem}_page{page.source.page_number:03d}.png")

    def listed(entities):
        return [{"category": e.category, "source": e.source, **({"text": e.text} if report_text else {})}
                for e in entities]

    report = {
        "source": job.path.name,
        "output": Path(out_path).name,
        "reader": job.reader_id,
        "llm": job.llm_ids,
        "categories": list(job.categories),
        "qrCodes": job.qr_codes,
        "entities": listed(job.entities),
        "notFound": listed(job.not_found),
        "pages": report_pages,
    }
    if verify:
        progress("verifying", 0, 1)
        report["verification"] = verify_redaction(burned, [e for e in job.entities if not kept_readable(job, e)],
                                                   job.ocr, report_text)
    report["seconds"] = round(job.seconds + time.time() - started, 1)
    return report


def redact_document(path: str | Path, out_path: str | Path, client, reader=None, ocr=None,
                    categories=DEFAULT_CATEGORIES, dpi: int = 200, verify: bool = True,
                    report_text: bool = False, preview_dir: str | Path | None = None,
                    policy=None, qr_codes: bool = True) -> dict:
    """Redact ``path`` into ``out_path`` without review; returns the redaction report. ``client``:
    a chat client (the local engine's) or the SwissDocAI PII service; ``policy``: see
    pii.detect_with_llm; ``qr_codes``: black out every QR code."""
    job = analyze_document(path, client, reader, ocr, categories, dpi, policy=policy, qr_codes=qr_codes)
    return render_job(job, out_path, verify, report_text, preview_dir)


def verify_redaction(images: list[Image.Image], entities: list[Entity], ocr, report_text: bool = False) -> dict:
    """OCR the redacted pages and search them for every found string (4+ letters/digits)."""
    leaks = []
    for number, image in enumerate(images, 1):
        stream = word_stream(ocr(image).words)
        for e in entities:
            needle = "".join(c for c, _ in _fold(e.text))
            if len(needle) >= 4 and find_all(stream, needle):
                leaks.append({"page": number, "category": e.category, **({"text": e.text} if report_text else {})})
    return {"ok": not leaks, "stillReadable": leaks}
