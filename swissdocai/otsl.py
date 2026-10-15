"""OTSL tables (as written by OCR models like Xiaomi-OCR-0 or Docling) → HTML.

OTSL writes a table row by row: every grid position is one token followed by
its text, ``<nl>`` ends a row. ``<fcel>``/``<ecel>`` start a filled/empty
cell (``<ched>``/``<rhed>``/``<srow>``: header and section cells), ``<lcel>``
continues the cell to the left (colspan), ``<ucel>`` the cell above (rowspan),
``<xcel>`` both. The result uses the DI dialect: one ``<tr>`` per line.
"""

import html
import re

_TOKEN = re.compile(r"<(fcel|ecel|ched|rhed|srow|lcel|ucel|xcel|nl)>")
_STARTS = ("fcel", "ecel", "ched", "rhed", "srow")
_TABLE = re.compile(r"(?:<(?:fcel|ecel|ched|rhed|srow|lcel|ucel|xcel)>[^\n]*?)+<nl>(?:[^\n]*?<nl>)*")
_WRAPPER = re.compile(r"</?otsl>")


def _grid(table: str) -> list[list[tuple[str, str]]]:
    rows, row = [], []
    parts = _TOKEN.split(table)
    # parts: [text_before, token, text, token, text, ...]
    for token, text in zip(parts[1::2], parts[2::2]):
        if token == "nl":
            if row:
                rows.append(row)
            row = []
        else:
            row.append((token, text.strip()))
    if row:
        rows.append(row)
    return rows


def otsl_table_to_html(table: str) -> str:
    rows = _grid(table)
    lines = ["<table>"]
    for r, row in enumerate(rows):
        cells = []
        for c, (token, text) in enumerate(row):
            if token not in _STARTS:
                continue
            colspan = 1
            while c + colspan < len(row) and row[c + colspan][0] == "lcel":
                colspan += 1
            rowspan = 1
            while (r + rowspan < len(rows) and c < len(rows[r + rowspan])
                   and rows[r + rowspan][c][0] in ("ucel", "xcel")):
                rowspan += 1
            tag = "th" if token in ("ched", "rhed") else "td"
            attrs = (f' colspan="{colspan}"' if colspan > 1 else "") + (f' rowspan="{rowspan}"' if rowspan > 1 else "")
            cells.append(f"<{tag}{attrs}>{html.escape(text, quote=False)}</{tag}>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def convert_otsl_tables(markdown: str) -> str:
    """Replace every OTSL table in ``markdown`` by an HTML table."""
    if "<nl>" not in markdown:
        return markdown
    return _TABLE.sub(lambda m: otsl_table_to_html(m.group(0)), _WRAPPER.sub("", markdown))
