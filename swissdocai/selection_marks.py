"""Selection marks (checkboxes): where they are and whether they are ticked.

Sources, best first:
- PDF form fields (AcroForm checkbox/radio widgets): exact box and state;
- PDF drawings: small stroked squares, ticked when a tick glyph ("X",
  "✓", ...) sits inside — this is how HTML/Word forms print checkboxes;
- PDF box glyphs ("☐", "☒", ...) in the text layer;
- page images (scans, photos): an OpenCV detector for square outlines,
  ticked when the inside holds ink.

The markdown model writes ☒/☐ at the matching places; result.py links
each detected mark to its glyph in the markdown.
"""

from dataclasses import dataclass

from .words import Word

Box = tuple[float, float, float, float]

# Glyphs that tick a drawn box when they sit inside it.
_TICK_GLYPHS = {"x", "X", "✓", "✔", "✗", "✘", "×", "■", "●"}
# Glyphs that are a checkbox on their own.
_BOX_GLYPHS = {"☐": False, "□": False, "❏": False, "☒": True, "☑": True, "▣": True}


@dataclass
class SelectionMark:
    bbox: Box  # pixels of the page image
    selected: bool
    confidence: float = 1.0
    source: str = "image"  # pdf-form | pdf-drawing | pdf-glyph | image


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _overlap_of(a: Box, b: Box) -> float:
    """Share of box a covered by box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def _drop_comb_fields(marks: list[SelectionMark]) -> list[SelectionMark]:
    """Remove rows of touching squares: comb fields for one character each
    (postal codes, IBAN boxes) look like checkboxes but are not."""
    combed = set()
    for i, a in enumerate(marks):
        w = a.bbox[2] - a.bbox[0]
        for j, b in enumerate(marks):
            if i == j:
                continue
            same_row = abs(a.bbox[1] - b.bbox[1]) < 0.3 * w and abs(a.bbox[3] - b.bbox[3]) < 0.3 * w
            touching = -0.3 * w < b.bbox[0] - a.bbox[2] < 0.3 * w
            if same_row and touching:
                combed.update((i, j))
    return [m for k, m in enumerate(marks) if k not in combed]


def _name(obj) -> str:
    """PDF name/string object → str (PSLiteral('Btn') → 'Btn')."""
    from pdfminer.pdftypes import resolve1

    obj = resolve1(obj)
    if obj is None:
        return ""
    if hasattr(obj, "name"):
        return str(obj.name)
    if isinstance(obj, bytes):
        return obj.decode("latin-1")
    return str(obj)


def _form_marks(page, scale: float) -> list[SelectionMark]:
    from pdfminer.pdftypes import resolve1

    marks = []
    for annot in page.annots or []:
        data = resolve1(annot.get("data")) or {}
        if _name(data.get("Subtype")) != "Widget":
            continue
        parent = resolve1(data.get("Parent")) or {}
        field_type = _name(data.get("FT")) or _name(parent.get("FT"))
        if field_type != "Btn":
            continue
        flags = int(resolve1(data.get("Ff")) or resolve1(parent.get("Ff")) or 0)
        if flags & (1 << 16):  # push button, not a checkbox/radio
            continue
        state = _name(data.get("AS")) or _name(data.get("V")) or _name(parent.get("V"))
        bbox = (annot["x0"] * scale, annot["top"] * scale, annot["x1"] * scale, annot["bottom"] * scale)
        marks.append(SelectionMark(bbox, state not in ("", "Off"), 1.0, "pdf-form"))
    return marks


def pdf_selection_marks(page, scale: float, words: list[Word]) -> tuple[list[SelectionMark], list[Word]]:
    """Checkboxes of a pdfplumber page, and the words that remain once tick
    and box glyphs (parts of checkboxes, not words) are taken out."""
    marks = _form_marks(page, scale)

    drawn = []
    for shape in list(page.rects) + list(page.curves):
        w, h = shape["width"], shape["height"]
        if not (5.5 <= w <= 20 and 5.5 <= h <= 20 and 0.8 <= w / h <= 1.25):
            continue
        if not shape.get("stroke"):
            continue  # filled squares are backgrounds or bullets
        bbox = (shape["x0"] * scale, shape["top"] * scale, shape["x1"] * scale, shape["bottom"] * scale)
        if any(_iou(bbox, m.bbox) > 0.5 for m in marks + drawn):
            continue
        drawn.append(SelectionMark(bbox, False, 0.95, "pdf-drawing"))
    drawn = _drop_comb_fields(drawn)

    remaining = []
    for word in words:
        cx, cy = (word.bbox[0] + word.bbox[2]) / 2, (word.bbox[1] + word.bbox[3]) / 2
        host = next(
            (m for m in drawn + marks if m.bbox[0] <= cx <= m.bbox[2] and m.bbox[1] <= cy <= m.bbox[3]),
            None,
        )
        if host is not None and word.text in _TICK_GLYPHS:
            host.selected = True
            continue
        if word.text in _BOX_GLYPHS:
            marks.append(SelectionMark(word.bbox, _BOX_GLYPHS[word.text], 1.0, "pdf-glyph"))
            continue
        remaining.append(word)
    return marks + drawn, remaining


def _straightness(band) -> float:
    """Best coverage of one straight line within a side band of a box.

    Rows of ``band`` run along the side; two adjacent rows are merged to
    tolerate anti-aliasing and slightly rotated scans.
    """
    if band.size == 0:
        return 0.0
    best = 0.0
    for r in range(band.shape[0]):
        line = band[r] | band[min(r + 1, band.shape[0] - 1)]
        best = max(best, float(line.mean()))
    return best


def _corner_distances(roi) -> list[float]:
    """Distance from each bounding-box corner to the nearest ink pixel.

    A box outline reaches into its corners even when rounded (radius r
    leaves 0.41 r); a round letter keeps ~0.2 of its width away.
    """
    import numpy as np

    h, w = roi.shape
    qh, qw = max(1, h // 3), max(1, w // 3)
    out = []
    for rows, cols, cy, cx in (
        (slice(0, qh), slice(0, qw), 0, 0),
        (slice(0, qh), slice(w - qw, w), 0, qw - 1),
        (slice(h - qh, h), slice(0, qw), qh - 1, 0),
        (slice(h - qh, h), slice(w - qw, w), qh - 1, qw - 1),
    ):
        ys, xs = np.nonzero(roi[rows, cols])
        out.append(float(np.min(np.hypot(ys - cy, xs - cx))) if len(ys) else float(max(h, w)))
    return out


def detect_selection_marks(image, dpi: int = 150, min_mm: float = 2.2,
                           max_mm: float = 8.0) -> list[SelectionMark]:
    """Square checkbox outlines in a page image (OpenCV).

    A candidate is an ink component whose bounding box is square, between
    ``min_mm`` and ``max_mm`` wide, with ink along all four sides of its
    outline. The inside (outline excluded) decides the state.
    """
    import cv2
    import numpy as np

    gray = np.asarray(image.convert("L"))
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    binary[gray > 180] = 0  # ink must be dark: pale backgrounds and shading are not outlines
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    lo, hi = min_mm / 25.4 * dpi, max_mm / 25.4 * dpi

    marks: list[SelectionMark] = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            continue  # a hole, not an ink component
        x, y, w, h = cv2.boundingRect(contour)
        if not (lo <= w <= hi and lo <= h <= hi and 0.8 <= w / h <= 1.25):
            continue
        roi = binary[y:y + h, x:x + w] > 0
        band = max(2, int(round(0.18 * min(w, h))))
        sides = [
            _straightness(roi[:band, band:w - band]),  # top
            _straightness(roi[h - band:, band:w - band]),  # bottom
            _straightness(roi[band:h - band, :band].T),  # left
            _straightness(roi[band:h - band, w - band:].T),  # right
        ]
        if min(sides) < 0.9:
            continue  # sides are not straight lines (round letters: o, a, e, Q)
        if max(_corner_distances(roi)) > 0.1 * min(w, h) + 0.5:
            continue  # empty corners: a round shape (bold O, o, 0), not a box
        # A checkbox stands apart from its label; a letter (D, O, 0 in bold
        # fonts pass the outline test) touches its neighbours in the word.
        gap = max(2, int(round(0.2 * w)))
        rows = slice(y + h // 5, y + h - h // 5)
        left = binary[rows, max(0, x - gap):x]
        right = binary[rows, x + w:x + w + gap]
        if (left.size and left.mean() > 0.02 * 255) or (right.size and right.mean() > 0.02 * 255):
            continue
        inner = roi[band + 1:h - band - 1, band + 1:w - band - 1]
        if inner.size == 0:
            continue
        fill = float(inner.mean())
        if fill > 0.75:
            continue  # solid square: bullet or decoration
        selected = fill > 0.08
        margin = abs(fill - 0.08) / 0.08
        confidence = round(min(0.99, 0.6 + 0.1 * min(sides) + 0.3 * min(1.0, margin)), 3)
        box = (float(x), float(y), float(x + w), float(y + h))
        if any(_iou(box, m.bbox) > 0.5 for m in marks):
            continue
        marks.append(SelectionMark(box, selected, confidence, "image"))
    return _drop_comb_fields(marks)


def remove_mark_words(words: list[Word], marks: list[SelectionMark]) -> list[Word]:
    """Drop OCR words that are really checkboxes ("O", "X", "[x]", "口" ...)."""
    if not marks:
        return words
    return [w for w in words if not any(_overlap_of(w.bbox, m.bbox) > 0.5 for m in marks)]
