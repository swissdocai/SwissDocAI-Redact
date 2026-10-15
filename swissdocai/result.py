"""Assemble a Document-Intelligence-style analyze result.

The JSON follows the shape of Azure AI Document Intelligence's
prebuilt-layout response, so existing consumers can switch with few
changes:

    {"status": "succeeded", "analyzeResult": {
        "content": "<markdown of all pages>", "contentFormat": "markdown",
        "pages": [{"pageNumber", "width", "height", "unit", "angle", "spans",
                   "words": [{"content", "polygon", "confidence", "span"}],
                   "lines": [{"content", "polygon", "spans"}]}],
        "paragraphs": [{"role", "content", "boundingRegions", "spans"}],
        "tables": [{"rowCount", "columnCount", "cells", "boundingRegions", "spans"}],
        "figures": [{"id", "boundingRegions", "spans"}]}}

Offsets are Unicode code points into ``content`` (stringIndexType
"unicodeCodePoint"). Polygons are the 4 corners clockwise from top-left
in the page ``unit``: inch for PDFs, pixel for images.

Deliberate differences from Document Intelligence:
- a word's ``content`` is the markdown text its span covers, so
  ``content[span] == word.content`` always holds; the word layer's own
  reading is kept in ``ocrContent`` when it differs. Words the markdown
  does not contain (e.g. a page number the model left out) have no span;
- ``paragraphs[].grounding`` and ``analyzeResult.swissdocai`` report how
  much of the markdown is backed by words on the page. Low values flag
  text the model may have hallucinated or misread.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from .align import align_tokens, word_char_spans
from .markdown_doc import parse_markdown
from .text import normalize_token
from .words import PageWords, group_lines

API_VERSION = "2026-10-01-preview"
PAGE_BREAK = "\n\n<!-- PageBreak -->\n\n"

_PARAGRAPH_KINDS = {"heading", "paragraph", "list_item", "quote", "code", "formula", "comment"}


@dataclass
class PageInput:
    page_number: int
    markdown: str
    words: PageWords  # boxes in pixels of the page image
    unit: str = "pixel"  # output unit: "inch" (PDF) or "pixel" (image)
    scale: float = 1.0  # output units per pixel


Box = tuple[float, float, float, float]


def _union(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _merge_spans(spans: list[tuple[int, int]], content: str) -> list[dict]:
    """Merge word spans that are only separated by whitespace/markup."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged:
            gap = content[merged[-1][1]:start]
            if start <= merged[-1][1] or (len(gap) <= 4 and not any(ch.isalnum() for ch in gap)):
                merged[-1][1] = max(merged[-1][1], end)
                continue
        merged.append([start, end])
    return [{"offset": s, "length": e - s} for s, e in merged]


class _PageBuilder:
    """Everything for one page; offsets shifted by the page's content base."""

    def __init__(self, page: PageInput, base: int, full_content: str):
        self.page = page
        self.base = base
        self.full_content = full_content
        self.digits = 4 if page.unit == "inch" else 1
        self.doc = parse_markdown(page.markdown)
        self.words = page.words.words
        if not page.words.lines and self.words:
            page.words.lines = group_lines(self.words)

        norms = [normalize_token(w.text) for w in self.words]
        matches = align_tokens([t.norm for t in self.doc.tokens], norms)
        self.word_spans = word_char_spans(
            matches, [(t.start, t.end) for t in self.doc.tokens], norms, self.doc.content
        )
        self.word_score: dict[int, float] = {}
        self.token_words: dict[int, list[int]] = {}
        for m in matches:
            for j in range(m.word_start, m.word_end):
                self.word_score[j] = m.score
            for t in range(m.md_start, m.md_end):
                self.token_words.setdefault(t, []).extend(range(m.word_start, m.word_end))
        self.alignable = [j for j, n in enumerate(norms) if n]

    def polygon(self, box: Box) -> list[float]:
        s, d = self.page.scale, self.digits
        x0, y0, x1, y1 = (v * s for v in box)
        return [round(v, d) for v in (x0, y0, x1, y0, x1, y1, x0, y1)]

    def region(self, box: Box | None) -> list[dict]:
        if box is None:
            return []
        return [{"pageNumber": self.page.page_number, "polygon": self.polygon(box)}]

    def span(self, start: int, end: int) -> list[dict]:
        return [{"offset": self.base + start, "length": end - start}] if end > start else []

    def tokens_box(self, token_ids) -> Box | None:
        return _union(
            [self.words[j].bbox for t in token_ids for j in self.token_words.get(t, [])]
        )

    # --- page-level layers -------------------------------------------------------

    def page_entry(self) -> dict:
        md = self.page.markdown
        words_out = []
        for j, word in enumerate(self.words):
            entry = {
                "content": word.text,
                "polygon": self.polygon(word.bbox),
                "confidence": round(word.confidence * self.word_score.get(j, 1.0), 3),
            }
            if j in self.word_spans:
                start, end = self.word_spans[j]
                entry["content"] = md[start:end]
                if md[start:end] != word.text:
                    entry["ocrContent"] = word.text
                entry["span"] = {"offset": self.base + start, "length": end - start}
            words_out.append(entry)

        lines_out = []
        for members in self.page.words.lines:
            spans = [
                (self.base + s, self.base + e)
                for j in members
                if j in self.word_spans
                for s, e in [self.word_spans[j]]
            ]
            lines_out.append(
                {
                    "content": " ".join(words_out[j]["content"] for j in members),
                    "polygon": self.polygon(_union([self.words[j].bbox for j in members])),
                    "spans": _merge_spans(spans, self.full_content),
                }
            )

        s = self.page.scale
        return {
            "pageNumber": self.page.page_number,
            "angle": 0,
            "width": round(self.page.words.width * s, self.digits),
            "height": round(self.page.words.height * s, self.digits),
            "unit": self.page.unit,
            "spans": self.span(0, len(md)),
            "words": words_out,
            "selectionMarks": self.selection_marks(),
            "lines": lines_out,
        }

    def link_marks(self) -> dict[int, int]:
        """Detected mark index → markdown mark index (☒/☐ glyph or [x]).

        A detected box belongs to a markdown checkbox when it sits on the
        same line as the word right after the glyph (box left of its label)
        or right before it (box after the label, end of a block).
        """
        page_marks = self.page.words.marks
        doc_marks = self.doc.marks
        if not page_marks or not doc_marks:
            return {}
        tokens = self.doc.tokens
        candidates = []
        for di, mark in enumerate(doc_marks):
            block = self.doc.blocks[mark.block]
            max_gap_factor = 20 if block.kind == "table" else 6
            nxt = next((t for t in range(len(tokens)) if tokens[t].start >= mark.end
                        and tokens[t].block == mark.block and t in self.token_words), None)
            prv = next((t for t in reversed(range(len(tokens))) if tokens[t].end <= mark.start
                        and tokens[t].block == mark.block and t in self.token_words), None)
            for pi, pm in enumerate(page_marks):
                x0, y0, x1, y1 = pm.bbox
                w, h = x1 - x0, y1 - y0
                cost = None
                for t, after in ((nxt, True), (prv, False)):
                    if t is None:
                        continue
                    wb = self.tokens_box([t])
                    overlap = min(y1, wb[3]) - max(y0, wb[1])
                    if overlap < 0.3 * min(h, wb[3] - wb[1]):
                        continue
                    gap = (wb[0] - x1) if after else (x0 - wb[2])
                    if -0.5 * w <= gap <= max_gap_factor * w:
                        cost = abs(gap) if cost is None else min(cost, abs(gap))
                if cost is not None:
                    candidates.append((cost, di, pi))
        linked: dict[int, int] = {}
        used_doc: set[int] = set()
        for cost, di, pi in sorted(candidates):
            if pi in linked or di in used_doc:
                continue
            linked[pi] = di
            used_doc.add(di)
        return linked

    def selection_marks(self) -> list[dict]:
        linked = self.link_marks()
        self.marks_linked = len(linked)
        out = []
        for pi, pm in enumerate(self.page.words.marks):
            entry = {
                "state": "selected" if pm.selected else "unselected",
                "polygon": self.polygon(pm.bbox),
                "confidence": round(pm.confidence, 3),
            }
            if pi in linked:
                mark = self.doc.marks[linked[pi]]
                if mark.selected != pm.selected:
                    # the page and the markdown disagree: trust the pixels,
                    # but say so through the confidence
                    entry["confidence"] = round(pm.confidence * 0.5, 3)
                entry["span"] = {"offset": self.base + mark.start, "length": mark.end - mark.start}
            out.append(entry)
        return out

    def structure(self, title_seen: bool) -> tuple[list, list, list, bool]:
        paragraphs, tables, figures = [], [], []
        block_tokens: dict[int, list[int]] = {}
        for t_idx, tok in enumerate(self.doc.tokens):
            block_tokens.setdefault(tok.block, []).append(t_idx)

        for b_idx, block in enumerate(self.doc.blocks):
            toks = block_tokens.get(b_idx, [])
            if block.kind == "table":
                tables.append(self.table_entry(b_idx, block, toks))
                continue
            if block.kind == "figure":
                figures.append(
                    {
                        "id": f"{self.page.page_number}.{len(figures) + 1}",
                        "boundingRegions": self.region(self.tokens_box(toks)),
                        "spans": self.span(block.start, block.end),
                    }
                )
                continue
            if block.kind not in _PARAGRAPH_KINDS:
                continue
            role = block.role
            if role == "title":
                role = "sectionHeading" if title_seen else "title"
                title_seen = True
            para = {}
            if role:
                para["role"] = role
            para["content"] = block.text
            para["boundingRegions"] = self.region(self.tokens_box(toks))
            para["spans"] = self.span(block.start, block.end)
            if toks:
                grounded = sum(1 for t in toks if t in self.token_words)
                para["grounding"] = round(grounded / len(toks), 3)
            paragraphs.append(para)
        return paragraphs, tables, figures, title_seen

    def table_entry(self, b_idx: int, block, toks: list[int]) -> dict:
        cell_tokens: dict[int, list[int]] = {}
        for t in toks:
            cell_tokens.setdefault(self.doc.tokens[t].cell, []).append(t)
        cell_boxes = {ci: self.tokens_box(ts) for ci, ts in cell_tokens.items()}

        # Row and column extents from cells that have words; empty cells get
        # the intersection of their row and column bands.
        row_ext: dict[int, list[float]] = {}
        col_ext: dict[int, list[float]] = {}
        for ci, cell in enumerate(block.cells):
            box = cell_boxes.get(ci)
            if box is None:
                continue
            if cell.row_span == 1:
                ext = row_ext.setdefault(cell.row, [box[1], box[3]])
                ext[0], ext[1] = min(ext[0], box[1]), max(ext[1], box[3])
            if cell.col_span == 1:
                ext = col_ext.setdefault(cell.col, [box[0], box[2]])
                ext[0], ext[1] = min(ext[0], box[0]), max(ext[1], box[2])

        cells_out, all_boxes = [], []
        for ci, cell in enumerate(block.cells):
            box = cell_boxes.get(ci)
            rows = range(cell.row, cell.row + cell.row_span)
            cols = range(cell.col, cell.col + cell.col_span)
            if box is None and all(r in row_ext for r in rows) and all(c in col_ext for c in cols):
                box = (
                    min(col_ext[c][0] for c in cols),
                    min(row_ext[r][0] for r in rows),
                    max(col_ext[c][1] for c in cols),
                    max(row_ext[r][1] for r in rows),
                )
            entry = {}
            if cell.header:
                entry["kind"] = "columnHeader"
            entry.update(rowIndex=cell.row, columnIndex=cell.col)
            if cell.row_span > 1:
                entry["rowSpan"] = cell.row_span
            if cell.col_span > 1:
                entry["columnSpan"] = cell.col_span
            entry["content"] = cell.text
            entry["boundingRegions"] = self.region(box)
            entry["spans"] = self.span(cell.start, cell.end)
            cells_out.append(entry)
            if box is not None:
                all_boxes.append(box)

        return {
            "rowCount": block.row_count,
            "columnCount": block.col_count,
            "cells": cells_out,
            "boundingRegions": self.region(_union(all_boxes)),
            "spans": self.span(block.start, block.end),
        }

    def stats(self) -> dict:
        tokens = self.doc.tokens
        md = self.page.markdown
        # Ungrounded markdown: runs of tokens without a word on the page.
        runs, current = [], []
        for t_idx in range(len(tokens)):
            if t_idx in self.token_words:
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(t_idx)
        if current:
            runs.append(current)
        ungrounded = [
            self.span(tokens[r[0]].start, tokens[r[-1]].end)[0]
            for r in runs
            if len(r) >= 2 or len(tokens[r[0]].norm) >= 4
        ]
        matched_words = sum(1 for j in self.alignable if j in self.word_spans)
        return {
            "pageNumber": self.page.page_number,
            "wordSource": self.page.words.source,
            "wordCount": len(self.words),
            "wordCoverage": round(matched_words / len(self.alignable), 4) if self.alignable else None,
            "markdownGrounded": (
                round(len(self.token_words) / len(tokens), 4) if tokens else None
            ),
            "ungroundedSpans": ungrounded[:50],
            "markdownChars": len(md),
            "selectionMarks": len(self.page.words.marks),
            "selectionMarksLinked": getattr(self, "marks_linked", 0),
        }


def build_analyze_result(pages: list[PageInput], model_id: str, extra: dict | None = None) -> dict:
    """Combine per-page markdown and words into one analyze result."""
    parts, bases, pos = [], [], 0
    for i, page in enumerate(pages):
        if i:
            parts.append(PAGE_BREAK)
            pos += len(PAGE_BREAK)
        bases.append(pos)
        parts.append(page.markdown)
        pos += len(page.markdown)
    content = "".join(parts)

    out_pages, paragraphs, tables, figures, stats = [], [], [], [], []
    title_seen = False
    for page, base in zip(pages, bases):
        builder = _PageBuilder(page, base, content)
        out_pages.append(builder.page_entry())
        page_paragraphs, page_tables, page_figures, title_seen = builder.structure(title_seen)
        paragraphs += page_paragraphs
        tables += page_tables
        figures += page_figures
        stats.append(builder.stats())

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "status": "succeeded",
        "createdDateTime": now,
        "lastUpdatedDateTime": now,
        "analyzeResult": {
            "apiVersion": API_VERSION,
            "modelId": model_id,
            "stringIndexType": "unicodeCodePoint",
            "content": content,
            "contentFormat": "markdown",
            "pages": out_pages,
            "paragraphs": paragraphs,
            "tables": tables,
            "figures": figures,
            "swissdocai": {"pages": stats, **(extra or {})},
        },
    }
