"""Parse markdown into blocks and word tokens with character offsets.

The parser is line-based and lenient. It targets the dialect our models
emit and our datasets contain: ATX headings, pipe and HTML tables, lists
and task items, block quotes, fenced code, ``$$`` math, ``<figure>`` and
Document-Intelligence-style ``<!-- PageHeader="..." -->`` comments.

Its job is to answer, for every character of the content string, which
block (paragraph, table cell, ...) it belongs to and which word tokens it
contains — the offsets later become the ``span`` fields of the analysis
result. Faithful markdown rendering is not a goal.
"""

import re
from dataclasses import dataclass, field

from .text import normalize_token


@dataclass
class Token:
    """One word of the markdown, as a character range of the content."""

    start: int
    end: int
    norm: str
    block: int
    cell: int = -1  # index into the block's cells (tables only)


@dataclass
class Cell:
    row: int
    col: int
    start: int  # text span within the content
    end: int
    row_span: int = 1
    col_span: int = 1
    header: bool = False
    text: str = ""


@dataclass
class Block:
    """A paragraph-level unit: heading, paragraph, list item, table, ..."""

    kind: str  # heading|paragraph|list_item|quote|code|formula|table|comment|figure
    start: int  # text span within the content (leading syntax excluded)
    end: int
    role: str | None = None  # Document Intelligence paragraph role
    level: int = 0  # heading level / list nesting depth
    checked: bool | None = None  # task-list items: [x] / [ ]
    text: str = ""  # plain text: syntax removed, whitespace collapsed
    cells: list[Cell] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0


@dataclass
class Mark:
    """A checkbox in the markdown: a ☒/☐ glyph or a [x]/[ ] task marker."""

    start: int
    end: int
    selected: bool
    block: int


@dataclass
class MarkdownDoc:
    content: str
    blocks: list[Block]
    tokens: list[Token]
    marks: list[Mark] = field(default_factory=list)


# --- block-level patterns ---------------------------------------------------

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HR = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_LIST = re.compile(r"^([ \t]*)([-*+]|\d{1,9}[.)])[ \t]+(?:\[([ xX])\][ \t]+)?")
_QUOTE = re.compile(r"^ {0,3}>[ \t]?")
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
_MATH_OPEN = re.compile(r"^[ \t]*(\$\$|\\\[)")
_COMMENT = re.compile(r"^[ \t]*<!--(.*?)-->[ \t]*$")
_DI_COMMENT = re.compile(r'^\s*(PageHeader|PageFooter|PageNumber)\s*=\s*"(.*)"\s*$', re.S)
_HTML_TABLE_OPEN = re.compile(r"^[ \t]*<table\b", re.I)
_FIGURE_OPEN = re.compile(r"^[ \t]*<figure\b", re.I)

_DI_ROLES = {"PageHeader": "pageHeader", "PageFooter": "pageFooter", "PageNumber": "pageNumber"}

# --- inline patterns ----------------------------------------------------------

# Opaque: kept in plain text but never tokenized (math has no word-level
# counterpart on the page that would compare equal to its LaTeX source).
_INLINE_OPAQUE = re.compile(
    r"(?<![\\$])\$(?![\s$])(?:\\.|[^$\\\n])+?(?<!\s)\$(?!\d)"  # $...$
    r"|\\\(.+?\\\)"  # \( ... \)
    r"|\\\[.+?\\\]"  # \[ ... \]
    r"|<!--.*?-->",
    re.S,
)
# Syntax: removed from plain text and never tokenized.
_INLINE_SYNTAX = re.compile(
    r"!\[[^\]\n]*\]\([^)\n]*\)"  # image
    r"|\]\([^)\n]*\)"  # link target
    r"|\[(?=[^\]\n]*\]\()"  # link text opener (literal "[ ]" stays text)
    r"|\*+|~~|`+"  # emphasis / strike / stray code delimiters
    r"|<[A-Za-z/!][^>\n]*>"  # html tags
    # _emphasis_ delimiters touch exactly one alphanumeric side; snake_case
    # (both sides) and ____ form fields (neither side) are text. [^\W_] is
    # "alphanumeric" — plain \w would count the underscores themselves.
    r"|(?<![^\W_])_+(?=[^\W_])|(?<=[^\W_])_+(?![^\W_])"
)

# Inline code: the delimiters are syntax, the content is literal text.
_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")
# Backslash escapes (CommonMark: before ASCII punctuation). The backslash
# is invisible — dropped without splitting the word ("snake\_case").
_ESCAPE = re.compile(r"\\(?=[!-/:-@\[-`{-~])")

_HTML_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr\s*>", re.S | re.I)
_HTML_CELL = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1\s*>", re.S | re.I)
_SPAN_ATTR = re.compile(r"""(rowspan|colspan)\s*=\s*["']?(\d+)""", re.I)

_SYNTAX, _OPAQUE, _INVISIBLE = 1, 2, 3

# Checkbox glyphs: selection marks, never word tokens.
_MARK_GLYPHS = {"☒": True, "☑": True, "☐": False}


class _Parser:
    def __init__(self, content: str):
        self.content = content
        self.mask = bytearray(len(content))
        self.blocks: list[Block] = []
        self.tokens: list[Token] = []
        self.marks: list[Mark] = []
        # (start, end) offsets of every line, newline excluded
        self.lines: list[tuple[int, int]] = []
        pos = 0
        for raw in content.split("\n"):
            self.lines.append((pos, pos + len(raw)))
            pos += len(raw) + 1

    # --- helpers --------------------------------------------------------------

    def line(self, i: int) -> str:
        s, e = self.lines[i]
        return self.content[s:e]

    def is_blank(self, i: int) -> bool:
        return not self.line(i).strip()

    def is_table_start(self, i: int) -> bool:
        return (
            i + 1 < len(self.lines)
            and "|" in self.line(i)
            and "-" in self.line(i + 1)
            and bool(_TABLE_SEP.match(self.line(i + 1)))
        )

    def is_block_start(self, i: int) -> bool:
        text = self.line(i)
        return bool(
            _HEADING.match(text)
            or _FENCE.match(text)
            or _HR.match(text)
            or _LIST.match(text)
            or _QUOTE.match(text)
            or _MATH_OPEN.match(text)
            or _COMMENT.match(text)
            or _HTML_TABLE_OPEN.match(text)
            or _FIGURE_OPEN.match(text)
            or self.is_table_start(i)
        )

    def mask_inline(self, start: int, end: int) -> None:
        segment = self.content[start:end]
        protected: set[int] = set()  # inline code content: literal text
        for m in _CODE_SPAN.finditer(segment):
            a, b, ticks = start + m.start(), start + m.end(), len(m.group(1))
            self.mask_range(a, a + ticks)
            self.mask_range(b - ticks, b)
            protected.update(range(a + ticks, b - ticks))
        for m in _ESCAPE.finditer(segment):  # "\_" → literal "_"
            k = start + m.start()
            if k not in protected and self.mask[k] != _INVISIBLE:
                self.mask[k] = _INVISIBLE
                protected.add(k + 1)
        for m in _INLINE_OPAQUE.finditer(segment):
            if start + m.start() in protected:
                continue
            for k in range(start + m.start(), start + m.end()):
                self.mask[k] = _OPAQUE
        for m in _INLINE_SYNTAX.finditer(segment):
            for k in range(start + m.start(), start + m.end()):
                if not self.mask[k] and k not in protected:
                    self.mask[k] = _SYNTAX

    def mask_range(self, start: int, end: int, kind: int = _SYNTAX) -> None:
        for k in range(start, end):
            self.mask[k] = kind

    def plain_text(self, start: int, end: int) -> str:
        chars = [
            self.content[k]
            for k in range(start, end)
            if self.mask[k] not in (_SYNTAX, _INVISIBLE)
        ]
        return " ".join("".join(chars).split())

    def tokenize(self, start: int, end: int, block: int, cell: int = -1) -> None:
        def in_word(k: int) -> bool:
            return not self.content[k].isspace() and self.mask[k] in (0, _INVISIBLE)

        k = start
        while k < end:
            if in_word(k) and self.content[k] in _MARK_GLYPHS:
                self.marks.append(Mark(k, k + 1, _MARK_GLYPHS[self.content[k]], block))
                k += 1
                continue
            if not in_word(k) or self.mask[k] == _INVISIBLE:
                k += 1
                continue
            j = k
            while j < end and in_word(j) and self.content[j] not in _MARK_GLYPHS:
                j += 1
            text = "".join(self.content[p] for p in range(k, j) if self.mask[p] != _INVISIBLE)
            norm = normalize_token(text)
            if norm:
                self.tokens.append(Token(k, j, norm, block, cell))
            k = j

    def add_block(self, block: Block, tokenize: bool = True) -> int:
        idx = len(self.blocks)
        self.blocks.append(block)
        if block.kind != "table":
            block.text = self.plain_text(block.start, block.end)
            if tokenize:
                self.tokenize(block.start, block.end, idx)
        return idx

    @staticmethod
    def strip_span(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    # --- block parsers ----------------------------------------------------------

    def parse(self) -> MarkdownDoc:
        i, n = 0, len(self.lines)
        while i < n:
            if self.is_blank(i):
                i += 1
                continue
            text = self.line(i)
            if _FENCE.match(text):
                i = self.parse_fence(i)
            elif _HR.match(text):
                i += 1
            elif (m := _HEADING.match(text)) is not None:
                i = self.parse_heading(i, m)
            elif _MATH_OPEN.match(text):
                i = self.parse_math(i)
            elif _COMMENT.match(text):
                i = self.parse_comment(i)
            elif _HTML_TABLE_OPEN.match(text):
                i = self.parse_html_table(i)
            elif _FIGURE_OPEN.match(text):
                i = self.parse_figure(i)
            elif self.is_table_start(i):
                i = self.parse_pipe_table(i)
            elif (m := _LIST.match(text)) is not None:
                i = self.parse_list_item(i, m)
            elif _QUOTE.match(text):
                i = self.parse_quote(i)
            else:
                i = self.parse_paragraph(i)
        self.marks.sort(key=lambda mark: mark.start)
        return MarkdownDoc(self.content, self.blocks, self.tokens, self.marks)

    def parse_heading(self, i: int, m: re.Match) -> int:
        line_start = self.lines[i][0]
        start, end = line_start + m.start(2), line_start + m.end(2)
        self.mask_inline(start, end)
        level = len(m.group(1))
        role = "title" if level == 1 else "sectionHeading"
        self.add_block(Block("heading", start, end, role=role, level=level))
        return i + 1

    def continuation_end(self, i: int) -> int:
        """Index after the last lazy-continuation line following line i."""
        j = i + 1
        while j < len(self.lines) and not self.is_blank(j) and not self.is_block_start(j):
            j += 1
        return j

    def parse_paragraph(self, i: int) -> int:
        j = self.continuation_end(i)
        start, end = self.strip_span(self.content, self.lines[i][0], self.lines[j - 1][1])
        self.mask_inline(start, end)
        self.add_block(Block("paragraph", start, end))
        return j

    def parse_list_item(self, i: int, m: re.Match) -> int:
        j = self.continuation_end(i)
        line_start = self.lines[i][0]
        # Ordered-list numbers are printed text ("1."), bullets are not.
        text_start = line_start + (m.start(2) if m.group(2)[0].isdigit() else m.end())
        start, end = self.strip_span(self.content, text_start, self.lines[j - 1][1])
        self.mask_inline(start, end)
        checked = None if m.group(3) is None else m.group(3).lower() == "x"
        depth = len(m.group(1).expandtabs(4)) // 2
        idx = self.add_block(Block("list_item", start, end, level=depth, checked=checked))
        if checked is not None:  # "[x]" / "[ ]" task marker
            marker = line_start + m.start(3) - 1
            self.marks.append(Mark(marker, marker + 3, checked, idx))
        return j

    def parse_quote(self, i: int) -> int:
        j = i
        while j < len(self.lines) and _QUOTE.match(self.line(j)):
            m = _QUOTE.match(self.line(j))
            self.mask_range(self.lines[j][0], self.lines[j][0] + m.end())
            j += 1
        start, end = self.strip_span(
            self.content, self.lines[i][0] + _QUOTE.match(self.line(i)).end(), self.lines[j - 1][1]
        )
        self.mask_inline(start, end)
        self.add_block(Block("quote", start, end))
        return j

    def parse_fence(self, i: int) -> int:
        marker = _FENCE.match(self.line(i)).group(1)
        j = i + 1
        while j < len(self.lines) and not self.line(j).lstrip().startswith(marker):
            j += 1
        if j > i + 1:
            start, end = self.strip_span(self.content, self.lines[i + 1][0], self.lines[j - 1][1])
        else:
            start = end = self.lines[i][1]  # empty code block
        self.add_block(Block("code", start, end))
        return min(j + 1, len(self.lines))

    def parse_math(self, i: int) -> int:
        opener = _MATH_OPEN.match(self.line(i)).group(1)
        closer = "$$" if opener == "$$" else "\\]"
        first = self.line(i)
        after_open = first.find(opener) + len(opener)
        j = i
        if closer not in first[after_open:]:
            j = i + 1
            while j < len(self.lines) and closer not in self.line(j):
                j += 1
        j = min(j, len(self.lines) - 1)
        start, end = self.strip_span(self.content, self.lines[i][0], self.lines[j][1])
        self.mask_range(start, end, _OPAQUE)
        self.add_block(Block("formula", start, end, role="formulaBlock"), tokenize=False)
        return j + 1

    def parse_comment(self, i: int) -> int:
        line_start = self.lines[i][0]
        m = _COMMENT.match(self.line(i))
        di = _DI_COMMENT.match(m.group(1))
        if di is None:
            return i + 1  # unknown comment (e.g. <!-- PageBreak -->): no text
        value_start = line_start + m.start(1) + di.start(2)
        value_end = line_start + m.start(1) + di.end(2)
        self.mask_range(line_start, self.lines[i][1])
        self.mask_range(value_start, value_end, 0)
        self.mask_inline(value_start, value_end)
        self.add_block(Block("comment", value_start, value_end, role=_DI_ROLES[di.group(1)]))
        return i + 1

    def block_until(self, i: int, closing: str) -> int:
        j = i
        while j < len(self.lines) and closing not in self.line(j).lower():
            j += 1
        return min(j, len(self.lines) - 1)

    def parse_figure(self, i: int) -> int:
        j = self.block_until(i, "</figure")
        start, end = self.strip_span(self.content, self.lines[i][0], self.lines[j][1])
        self.mask_inline(start, end)
        self.add_block(Block("figure", start, end))
        return j + 1

    def parse_html_table(self, i: int) -> int:
        j = self.block_until(i, "</table")
        start, end = self.strip_span(self.content, self.lines[i][0], self.lines[j][1])
        block = Block("table", start, end)
        idx = self.add_block(block)
        segment = self.content[start:end]
        thead_end = segment.lower().find("</thead")

        occupied: set[tuple[int, int]] = set()
        row = 0
        for row_match in _HTML_ROW.finditer(segment):
            col = 0
            for cell_match in _HTML_CELL.finditer(row_match.group(1)):
                while (row, col) in occupied:
                    col += 1
                attrs = dict(
                    (k.lower(), int(v)) for k, v in _SPAN_ATTR.findall(cell_match.group(2))
                )
                row_span, col_span = attrs.get("rowspan", 1), attrs.get("colspan", 1)
                for dr in range(row_span):
                    for dc in range(col_span):
                        occupied.add((row + dr, col + dc))
                cell_start = start + row_match.start(1) + cell_match.start(3)
                cell_end = start + row_match.start(1) + cell_match.end(3)
                cell_start, cell_end = self.strip_span(self.content, cell_start, cell_end)
                self.mask_inline(cell_start, cell_end)
                in_thead = 0 <= thead_end and row_match.start() < thead_end
                self.add_cell(block, idx, row, col, cell_start, cell_end, row_span, col_span,
                              header=cell_match.group(1).lower() == "th" or in_thead)
                col += col_span
            row += 1
        self.finish_table(block)
        # Everything in the table outside cell texts is markup.
        cell_ranges = sorted((c.start, c.end) for c in block.cells)
        k = start
        for cs, ce in cell_ranges + [(end, end)]:
            if k < cs:
                self.mask_range(k, cs)
            k = max(k, ce)
        return j + 1

    def parse_pipe_table(self, i: int) -> int:
        j = i
        while j < len(self.lines) and not self.is_blank(j) and "|" in self.line(j):
            j += 1
        start, end = self.strip_span(self.content, self.lines[i][0], self.lines[j - 1][1])
        block = Block("table", start, end)
        idx = self.add_block(block)
        row = 0
        for k in range(i, j):
            line_start, line_end = self.lines[k]
            if k == i + 1:  # separator row
                self.mask_range(line_start, line_end)
                continue
            for cs, ce in _split_pipe_row(self.content, line_start, line_end):
                self.mask_inline(cs, ce)
            col = 0
            for cs, ce in _split_pipe_row(self.content, line_start, line_end):
                cs, ce = self.strip_span(self.content, cs, ce)
                self.add_cell(block, idx, row, col, cs, ce, header=(k == i))
                col += 1
            row += 1
            for p in range(line_start, line_end):
                if self.content[p] == "|" and self.mask[p] == 0:
                    self.mask[p] = _SYNTAX
        self.finish_table(block)
        return j

    def add_cell(self, block, idx, row, col, start, end, row_span=1, col_span=1, header=False):
        cell = Cell(row, col, start, end, row_span, col_span, header)
        cell.text = self.plain_text(start, end)
        block.cells.append(cell)
        self.tokenize(start, end, idx, len(block.cells) - 1)

    @staticmethod
    def finish_table(block: Block) -> None:
        if block.cells:
            block.row_count = max(c.row + c.row_span for c in block.cells)
            block.col_count = max(c.col + c.col_span for c in block.cells)
        block.text = " | ".join(c.text for c in block.cells if c.text)


def _split_pipe_row(content: str, start: int, end: int) -> list[tuple[int, int]]:
    """Cell ranges of one pipe-table row (unescaped pipes outside code spans)."""
    pipes = []
    in_code = False
    for k in range(start, end):
        ch = content[k]
        if ch == "`":
            in_code = not in_code
        elif ch == "|" and not in_code and (k == start or content[k - 1] != "\\"):
            pipes.append(k)
    bounds = [start - 1] + pipes + [end]
    cells = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    # A leading/trailing pipe produces an empty edge segment that is not a cell.
    if pipes and not content[start:pipes[0]].strip():
        cells = cells[1:]
    if pipes and not content[pipes[-1] + 1:end].strip():
        cells = cells[:-1]
    return cells


def parse_markdown(content: str) -> MarkdownDoc:
    """Parse ``content`` into blocks and word tokens (offsets into ``content``)."""
    return _Parser(content).parse()
