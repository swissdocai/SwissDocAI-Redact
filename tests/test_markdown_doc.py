"""Markdown block/token parsing with character offsets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swissdocai.markdown_doc import parse_markdown  # noqa: E402


def tokens(md: str) -> list[str]:
    return [md[t.start:t.end] for t in parse_markdown(md).tokens]


def norms(md: str) -> list[str]:
    return [t.norm for t in parse_markdown(md).tokens]


def test_heading_and_paragraph_spans():
    md = "# Rechnung Nr. 17\n\n**Kunde:** Müller AG, *Zürich*\nBetrag: CHF 1'234.50"
    doc = parse_markdown(md)
    heading, para = doc.blocks
    assert heading.kind == "heading" and heading.role == "title"
    assert md[heading.start:heading.end] == "Rechnung Nr. 17"
    assert para.text == "Kunde: Müller AG, Zürich Betrag: CHF 1'234.50"
    assert norms(md) == ["rechnung", "nr", "17", "kunde", "müller", "ag", "zürich",
                         "betrag", "chf", "1'234.50"]


def test_inline_code_content_is_literal():
    md = "- **Français :** `[ ] A1 [x] C2`"
    assert tokens(md) == ["Français", ":", "[", "]", "A1", "[x]", "C2"]


def test_escapes_are_invisible():
    md = "Feld: \\_\\_\\_\\_ und snake\\_case \\*lit"
    assert norms(md) == ["feld", "_", "und", "snake_case", "lit"]
    assert parse_markdown(md).blocks[0].text == "Feld: ____ und snake_case *lit"


def test_form_field_underscores_survive():
    assert norms("**Name:** `______`\nOrt: ________") == ["name", "_", "ort", "_"]


def test_math_and_links_are_not_tokens():
    md = "Varianz $\\sigma^2$ siehe [Link](https://x.ch) und ![Bild](a.png)"
    assert tokens(md) == ["Varianz", "siehe", "Link", "und"]


def test_list_numbers_are_tokens_bullets_are_not():
    md = "1. Erstens\n2. Zweitens\n\n- frei"
    assert norms(md) == ["1", "erstens", "2", "zweitens", "frei"]


def test_checkboxes_become_marks():
    md = "- [x] bezahlt\n- [ ] offen\n\nJa ☒ Nein ☐\n\n| Option | |\n|---|---|\n| A | ☑ |"
    doc = parse_markdown(md)
    items = [b for b in doc.blocks if b.kind == "list_item"]
    assert [b.checked for b in items] == [True, False]
    assert [(md[m.start:m.end], m.selected) for m in doc.marks] == [
        ("[x]", True), ("[ ]", False), ("☒", True), ("☐", False), ("☑", True)
    ]
    assert norms(md) == ["bezahlt", "offen", "ja", "nein", "option", "a"]


def test_pipe_table_cells():
    md = "| Pos | Betrag |\n| --- | --: |\n| 1 | 800.00 |\n| 2 | |"
    (table,) = parse_markdown(md).blocks
    assert (table.row_count, table.col_count) == (3, 2)
    cells = {(c.row, c.col): (c.text, c.header) for c in table.cells}
    assert cells[(0, 1)] == ("Betrag", True)
    assert cells[(1, 1)] == ("800.00", False)
    assert cells[(2, 1)] == ("", False)
    assert "|" not in "".join(tokens(md))


def test_html_table_spans():
    md = (
        "<table><thead><tr><th>A</th><th colspan=\"2\">B C</th></tr></thead>"
        "<tr><td rowspan=\"2\">x</td><td>y</td><td>z</td></tr><tr><td>u</td><td>v</td></tr></table>"
    )
    (table,) = parse_markdown(md).blocks
    assert (table.row_count, table.col_count) == (3, 3)
    cells = {c.text: (c.row, c.col, c.row_span, c.col_span, c.header) for c in table.cells}
    assert cells["B C"] == (0, 1, 1, 2, True)
    assert cells["x"] == (1, 0, 2, 1, False)
    assert cells["u"] == (2, 1, 1, 1, False)  # skips the cell occupied by the rowspan
    assert tokens(md) == ["A", "B", "C", "x", "y", "z", "u", "v"]


def test_di_page_comments():
    md = 'Text\n\n<!-- PageNumber="3 / 4" -->\n\n<!-- PageBreak -->'
    doc = parse_markdown(md)
    assert [(b.kind, b.role, b.text) for b in doc.blocks] == [
        ("paragraph", None, "Text"),
        ("comment", "pageNumber", "3 / 4"),
    ]


def test_fenced_code_and_formula_blocks():
    md = "```yaml\nport: 8080\n```\n\n$$\nx^2\n$$\n\nEnde"
    doc = parse_markdown(md)
    assert [b.kind for b in doc.blocks] == ["code", "formula", "paragraph"]
    assert tokens(md) == ["port:", "8080", "Ende"]


def test_token_offsets_point_into_content():
    md = "## Titel\n\n> **Hinweis:** Text mit `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    for tok in parse_markdown(md).tokens:
        assert md[tok.start:tok.end].strip() == md[tok.start:tok.end]
        assert tok.end > tok.start
