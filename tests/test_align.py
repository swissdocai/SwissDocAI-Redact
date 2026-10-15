"""Token alignment between markdown and page words."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swissdocai.align import align_tokens, word_char_spans  # noqa: E402
from swissdocai.markdown_doc import parse_markdown  # noqa: E402
from swissdocai.text import normalize_token  # noqa: E402


def pairs(md: str, words: list[str]) -> dict[str, str]:
    """word -> markdown substring it was aligned to."""
    doc = parse_markdown(md)
    norms = [normalize_token(w) for w in words]
    matches = align_tokens([t.norm for t in doc.tokens], norms)
    spans = word_char_spans(matches, [(t.start, t.end) for t in doc.tokens], norms, md)
    return {words[j]: md[s:e] for j, (s, e) in spans.items()}


def test_normalization():
    assert normalize_token("„Zürich“,") == "zürich"
    assert normalize_token("1’234.50") == "1'234.50"
    assert normalize_token("Straße") == "strasse"  # Swiss spelling compares equal
    assert normalize_token("ﬁnal") == "final"
    assert normalize_token("______") == "_"
    assert normalize_token("–") == "-"


def test_exact_typo_and_hyphenation():
    md = "Rechnung an Müller AG in Zürich. Betrag: CHF 1'234.50"
    words = ["Rechnung", "an", "Müller", "AG", "in", "Zürlch.", "Be-", "trag:", "CHF", "1’234.50"]
    got = pairs(md, words)
    assert got["Zürlch."] == "Zürich."  # OCR typo absorbed by the gap DP
    assert got["Be-"] == "Be" and got["trag:"] == "trag:"  # hyphenated line break
    assert got["1’234.50"] == "1'234.50"


def test_hyphenated_split_ignores_token_punctuation():
    # Fedlex line breaks: the markdown joins the word, the cut must fall where the
    # first line's part ends, not shift by the trailing period or semicolon
    got = pairs("Les services au maximum. Les accords", ["Les", "services", "au", "maxi-", "mum.", "Les", "accords"])
    assert got["maxi-"] == "maxi" and got["mum."] == "mum."
    got = pairs("vor der Umsetzung des Abkommens; fordert", ["vor", "der", "Umsetzung", "des", "Abkom-", "mens;",
                                                           "fordert"])
    assert got["Abkom-"] == "Abkom" and got["mens;"] == "mens;"


def test_merged_words():
    got = pairs("Total CHF 100", ["TotalCHF", "100"])
    assert got["TotalCHF"] == "Total CHF"


def test_short_words_need_exact_match():
    got = pairs("di la casa", ["da", "la", "casa"])
    assert "da" not in got


def test_reordered_columns_recovered():
    left = "Die linke Spalte erzählt eine lange Geschichte über Basel und Bern"
    right = "Rechts steht ein Kasten mit Kontaktdaten der Firma"
    # the word layer reads the sidebar first, the markdown reads it last
    words = right.split() + left.split()
    got = pairs(left + "\n\n" + right, words)
    assert len(got) == len(words)


def test_unrelated_words_stay_unmatched():
    got = pairs("Hallo Welt", ["Seite", "1", "/", "2", "Hallo", "Welt"])
    assert set(got) == {"Hallo", "Welt"}
