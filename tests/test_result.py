"""Document-Intelligence-style result assembly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swissdocai.result import PAGE_BREAK, PageInput, build_analyze_result  # noqa: E402
from swissdocai.words import PageWords, Word, group_lines  # noqa: E402


def words_from_lines(lines: list[str], top: float = 100.0) -> PageWords:
    words = []
    for row, text in enumerate(lines):
        x = 50.0
        for token in text.split():
            width = 10.0 * len(token)
            words.append(Word(token, (x, top + 30 * row, x + width, top + 30 * row + 14), 0.9))
            x += width + 8
    page = PageWords(1240, 1754, words, source="ocr")
    page.lines = group_lines(words)
    return page


PAGE1_MD = "# Rechnung\n\nKunde: Müller AG\n\n| Pos | Betrag |\n| --- | --- |\n| 1 | 800.00 |\n| 2 | |"
PAGE1_WORDS = ["Rechnung", "Kunde: Müller AG", "Pos Betrag", "1 800.00", "2", "Seite 1"]
PAGE2_MD = "# Zahlung\n\nBitte innert 30 Tagen zahlen. Erfundener Satz hier."
PAGE2_WORDS = ["Zahlung", "Bitte innert 30 Tagen zahlen."]


def build():
    pages = [
        PageInput(1, PAGE1_MD, words_from_lines(PAGE1_WORDS), unit="inch", scale=1 / 150),
        PageInput(2, PAGE2_MD, words_from_lines(PAGE2_WORDS), unit="inch", scale=1 / 150),
    ]
    return build_analyze_result(pages, model_id="test")["analyzeResult"]


def test_content_and_page_spans():
    result = build()
    assert result["content"] == PAGE1_MD + PAGE_BREAK + PAGE2_MD
    assert result["pages"][1]["spans"] == [{"offset": len(PAGE1_MD) + len(PAGE_BREAK), "length": len(PAGE2_MD)}]
    assert result["pages"][0]["unit"] == "inch"
    assert result["pages"][0]["width"] == round(1240 / 150, 4)


def test_every_word_span_matches_content():
    result = build()
    content = result["content"]
    linked = 0
    for page in result["pages"]:
        for word in page["words"]:
            if "span" in word:
                s = word["span"]
                assert content[s["offset"]:s["offset"] + s["length"]] == word["content"]
                linked += 1
    assert linked == 15
    unlinked = [w["content"] for p in result["pages"] for w in p["words"] if "span" not in w]
    assert unlinked == ["Seite", "1"]  # the footer is on the page, not in the markdown


def test_paragraph_roles_and_regions():
    paragraphs = build()["paragraphs"]
    assert [p.get("role") for p in paragraphs] == ["title", None, "sectionHeading", None]
    assert paragraphs[0]["content"] == "Rechnung"
    assert paragraphs[0]["boundingRegions"][0]["pageNumber"] == 1
    assert len(paragraphs[0]["boundingRegions"][0]["polygon"]) == 8
    # 3 of the last paragraph's 8 tokens have no words on the page
    assert paragraphs[3]["grounding"] == 0.625


def test_table_cells_including_empty_cell_box():
    (table,) = build()["tables"]
    assert (table["rowCount"], table["columnCount"]) == (3, 2)
    cells = {(c["rowIndex"], c["columnIndex"]): c for c in table["cells"]}
    assert cells[(0, 0)]["kind"] == "columnHeader"
    assert cells[(1, 1)]["content"] == "800.00"
    empty = cells[(2, 1)]
    assert empty["content"] == "" and empty["spans"] == []
    # box inferred from row 2's band and column 1's band
    x0, y0 = empty["boundingRegions"][0]["polygon"][:2]
    column_x0 = min(cells[(r, 1)]["boundingRegions"][0]["polygon"][0] for r in (0, 1))
    assert x0 == column_x0
    assert y0 == cells[(2, 0)]["boundingRegions"][0]["polygon"][1]


def test_grounding_stats_flag_invented_text():
    result = build()
    stats = result["swissdocai"]["pages"][1]
    content = result["content"]
    ungrounded = [content[s["offset"]:s["offset"] + s["length"]] for s in stats["ungroundedSpans"]]
    assert ungrounded == ["Erfundener Satz hier."]
    assert stats["wordCoverage"] == 1.0
