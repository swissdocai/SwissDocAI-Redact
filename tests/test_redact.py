"""Redaction: rules, locating found strings on the page words, bars, PDF output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image  # noqa: E402

from swissdocai.pii import Entity, detect_with_rules  # noqa: E402
from swissdocai.pipeline import SourcePage, merge_word_layers  # noqa: E402
from swissdocai.redact import (  # noqa: E402
    PageState,
    find_all,
    locate,
    missing_text,
    redaction_bars,
    save_pdf,
    signature_images,
    text_stream,
    word_stream,
)
from swissdocai.words import PageWords, Word, group_lines  # noqa: E402


def words_on_lines(*lines: str, height: float = 20.0) -> list[Word]:
    out = []
    for row, line in enumerate(lines):
        x = 10.0
        for token in line.split():
            width = 10.0 * len(token)
            out.append(Word(token, (x, 10 + row * 30, x + width, 10 + row * 30 + height)))
            x += width + 8
    group_lines(out)
    return out


def page_state(words: list[Word], spans=None) -> PageState:
    page = PageState(SourcePage(1, Image.new("RGB", (800, 600), "white"), None, "pixel", 1.0),
                     PageWords(800, 600, words, group_lines(words)))
    page.stream = word_stream(words)
    page.word_spans = spans or [None] * len(words)
    return page


def found(text: str):
    return {(e.text, e.category) for _, _, e in detect_with_rules(text)}


def test_rules_check_checksums():
    assert ("756.9217.0769.85", "identifier") in found("AHV-Nr. 756.9217.0769.85")
    assert not any(c == "identifier" for _, c in found("AHV-Nr. 756.9217.0769.84"))  # wrong check digit
    assert ("CH93 0076 2011 6238 5295 7", "financial") in found("IBAN CH93 0076 2011 6238 5295 7")
    assert not any(c == "financial" for _, c in found("IBAN CH93 0076 2011 6238 5295 8"))


def test_rules_phone_email_birth_date():
    hits = found("Tel. 079 555 01 23, +41 72 288 12 12, praxis@example.ch, Geb.Datum 14.03.1988, *14.03.1988")
    assert ("079 555 01 23", "contact") in hits and ("+41 72 288 12 12", "contact") in hits
    assert ("praxis@example.ch", "contact") in hits
    assert ("14.03.1988", "birth_date") in hits
    assert not any(c == "birth_date" for _, c in found("Termin am 12.11.2026"))


def test_streams_ignore_spacing_and_punctuation_but_respect_word_boundaries():
    stream = word_stream(words_on_lines("Herr Bauer, Andreas", "8999Musterhausen ZH"))
    assert find_all(stream, "bauerandreas")  # comma and line layout do not matter
    assert find_all(stream, "8999musterhausen")  # glued postcode and town
    assert find_all(stream, "8999")  # digit → letter change is a token boundary
    assert not find_all(stream, "eitz")  # but not inside a word
    assert not find_all(stream, "ett")


def test_found_name_is_redacted_on_every_page():
    p1 = page_state(words_on_lines("Sehr geehrter Herr Bauer", "Andreas Bauer"))
    p2 = page_state(words_on_lines("Bauer, Andreas *14.03.1988"))
    assert locate(Entity("Bauer", "person", "llm"), [p1, p2], None) == 3
    assert set(p1.redact) == {3, 5} and set(p2.redact) == {0}


def test_reader_text_catches_ocr_misreadings():
    words = words_on_lines("Tel +417 72 88 2. 12")  # OCR misread the number
    content = "Tel +41 72 288 12 12"
    page = page_state(words, spans=[(0, 3), (4, 7), (8, 10), (11, 14), (15, 17), (18, 20)])
    assert locate(Entity("+41 72 288 12 12", "contact", "rule:phone"), [page], text_stream(content)) == 5
    assert 0 not in page.redact  # "Tel" stays


def test_bars_merge_neighbouring_words_and_pad():
    words = words_on_lines("Andreas Bauer wohnt hier")
    bars = redaction_bars(words, {0, 1, 3})
    assert len(bars) == 2
    x0, y0, x1, y1 = bars[0]
    assert x0 < words[0].bbox[0] and x1 > words[1].bbox[2] and y0 < words[0].bbox[1]


def test_missing_words_keep_columns_apart():
    left = [Word("Andreas", (10, 100, 80, 120)), Word("Bauer", (90, 100, 140, 120)),
            Word("Musterstrasse", (10, 130, 180, 150)), Word("12", (190, 130, 210, 150))]
    right = [Word("Handchirurgie", (500, 100, 620, 120)), Word("PD", (500, 130, 520, 150)),
             Word("Vogt", (530, 130, 570, 150))]
    words = [left[0], left[1], right[0], left[2], left[3], right[1], right[2]]  # OCR order mixes columns
    page = page_state(words)
    assert missing_text(page) == "Andreas Bauer\nMusterstrasse 12\n\nHandchirurgie\nPD Vogt"


def test_signature_next_to_redacted_name():
    page = page_state(words_on_lines("x"))
    page.source.images = [(10, 200, 90, 250), (10, 10, 790, 590)]  # signature, full-page scan
    assert signature_images(page, [(10, 260, 200, 280)]) == [(10, 200, 90, 250)]
    assert signature_images(page, [(10, 560, 200, 580)]) == []


def qr_code(text: str, size: int, swiss_cross: bool = False) -> Image.Image:
    """A QR code of ``size`` pixels with its quiet zone; with the Swiss cross of a QR-bill."""
    import cv2
    from PIL import ImageDraw

    code = Image.fromarray(cv2.QRCodeEncoder.create().encode(text)).convert("L").resize((size, size), Image.NEAREST)
    if swiss_cross:
        draw, c, s = ImageDraw.Draw(code), size // 2, size * 7 // 92  # 7 of 46 mm
        draw.rectangle((c - s, c - s, c + s, c + s), fill=0)
        draw.rectangle((c - s * 3 // 5, c - s // 5, c + s * 3 // 5, c + s // 5), fill=255)
        draw.rectangle((c - s // 5, c - s * 3 // 5, c + s // 5, c + s * 3 // 5), fill=255)
    return code


def test_qr_codes_are_found_whole_also_a_qr_bills_and_a_skewed_one():
    from PIL import ImageDraw

    from swissdocai.redact import find_qr_codes

    page = Image.new("L", (1654, 2339), 255)  # A4 at 200 dpi
    draw = ImageDraw.Draw(page)
    for row in range(30):
        draw.text((100, 100 + 30 * row), "Rechnung 4711 vom 12.03.2026, Andreas Bauer, CHF 1949.75 " * 2, fill=0)
    bill = "SPC\n0200\n1\nCH4431999123000889012\nS\nRobert Schneider AG\nRue du Lac\n1268\n2501\nBiel\nCH" + "\n" * 7 \
        + "1949.75\nCHF\nS\nPia-Maria Rutschmann-Schnyder\nGrosse Marktgasse\n28\n9400\nRorschach\nCH\nNON\n\n\nEPD\n"
    codes = [(qr_code(bill, 362, swiss_cross=True), (520, 1600)),  # 46 mm
             (qr_code("https://example.ch/patient?id=4711002", 118), (1300, 60)),  # 15 mm
             (qr_code("BEGIN:VCARD\nFN:Andreas Bauer\nEND:VCARD", 197).rotate(12, expand=True, fillcolor=255),
              (100, 1300))]
    for code, at in codes:
        page.paste(code, at)
    found = find_qr_codes(page.convert("RGB"))
    assert len(found) == 3
    for code, (x, y) in codes:  # the code itself lies inside its quiet zone (a tenth of the size)
        q = code.width // 10
        assert any(b[0] <= x + q and b[1] <= y + q and b[2] >= x + code.width - q and b[3] >= y + code.height - q
                   for b in found), (x, y)
    text_only = Image.new("RGB", (1654, 1000), "white")
    ImageDraw.Draw(text_only).text((100, 100), "Zahlbar bis 30.04.2026 – QR-Code siehe Rückseite", fill=0)
    assert find_qr_codes(text_only) == []


def test_text_layer_words_win_and_ocr_fills_gaps():
    pdf = [Word("Bauer", (10, 10, 60, 30))]
    ocr = [Word("Bauer", (11, 10, 61, 31)), Word("Kunze", (10, 200, 60, 220))]
    assert [w.text for w in merge_word_layers(pdf, ocr)] == ["Bauer", "Kunze"]


def test_pdf_keeps_page_sizes_and_has_no_text(tmp_path):
    import pypdfium2 as pdfium

    out = tmp_path / "out.pdf"
    save_pdf([Image.new("RGB", (1654, 2339), "white"), Image.new("RGB", (2339, 1654), "white")],
             [(595.3, 842.0), (842.0, 595.3)], out)
    pdf = pdfium.PdfDocument(str(out))
    assert [tuple(round(v, 1) for v in pdf[i].get_size()) for i in range(len(pdf))] == [(595.3, 842.0),
                                                                                       (842.0, 595.3)]
    assert pdf[0].get_textpage().get_text_range() == ""
    pdf.close()


def test_implausible_llm_items_are_dropped():
    from swissdocai.pii import plausible

    assert plausible(Entity("Daniel Frei", "person", "llm"))
    assert plausible(Entity("8999 Musterhausen ZH", "address", "llm"))
    assert plausible(Entity("https://pacs.example/show?patid=4711002", "identifier", "llm"))
    assert not plausible(Entity("Herr", "person", "llm"))
    assert not plausible(Entity("Dr. med.", "person", "llm"))
    assert not plausible(Entity("Schmerzen im Rücken seit einer Woche. Schmerzen auf beiden Seiten", "person", "llm"))
    assert not plausible(Entity("LWS-Syndrom", "identifier", "llm"))  # no number: not an identifier
    assert not plausible(Entity("keine auffälligkeit", "person", "llm"))


def test_fragments_and_non_values_are_dropped():
    from swissdocai.pii import drop_fragments, plausible

    assert not plausible(Entity("12", "address", "llm"))
    assert not plausible(Entity("2026", "birth_date", "llm"))
    assert plausible(Entity("14.03.1988", "birth_date", "llm"))
    assert plausible(Entity("14. März 1988", "birth_date", "llm"))
    assert not plausible(Entity("100.-", "financial", "llm"))
    assert not plausible(Entity("288 12", "contact", "llm"))
    entities = [Entity("8999 Musterhausen ZH", "address", "llm"), Entity("8999", "address", "llm"),
                Entity("Musterhausen", "address", "llm"), Entity("Andreas Bauer", "person", "llm"),
                Entity("Bauer", "person", "llm")]
    assert [e.text for e in drop_fragments(entities)] == ["8999 Musterhausen ZH", "Andreas Bauer", "Bauer"]


def test_one_pass_over_the_document_and_values_trimmed():
    import json as _json

    from swissdocai.pii import detect_with_llm

    asked = []

    class Client:
        def chat(self, messages, **kw):
            asked.append(messages[1]["content"])
            return _json.dumps({"entities": [
                {"text": "Daniel Frei", "category": "person"},
                {"text": "Bauer Andreas", "category": "person"},
                {"text": "Geb.Datum 14.03.1988", "category": "birth_date"},
                {"text": "Tel. 079 555 01 23", "category": "contact"}]})

    found = {(e.text, e.category) for e in detect_with_llm(Client(), ["Seite 1", "Seite 2"])}
    assert found == {("Daniel Frei", "person"), ("Bauer Andreas", "person"), ("14.03.1988", "birth_date"),
                     ("079 555 01 23", "contact"),
                     ("Daniel", "person"), ("Frei", "person"), ("Bauer", "person"), ("Andreas", "person")}
    assert len(asked) == 1 and "=== Page 1 ===" in asked[0] and "=== Page 2 ===" in asked[0]
    # a long document goes in several requests of at most max_chars
    asked.clear()
    detect_with_llm(Client(), ["a" * 300, "b" * 300], max_chars=400)
    assert len(asked) == 2


def test_an_empty_answer_is_asked_again_sampled():
    import json as _json

    from swissdocai.pii import detect_with_llm

    replies = iter(["{\"entities\": []}", "not json",
                    _json.dumps({"entities": [{"text": "Ursula Kuhn", "category": "person"}]})])
    temperatures = []

    class Client:
        def chat(self, messages, temperature=0.0, **kw):
            temperatures.append(temperature)
            return next(replies)

    found = detect_with_llm(Client(), ["Sehr geehrte Frau Kuhn"], retry_empty=2)
    assert ("Ursula Kuhn", "person") in {(e.text, e.category) for e in found}
    assert temperatures == [0.0, 0.6, 0.6]
    # an answer with values is not asked again
    replies = iter([_json.dumps({"entities": [{"text": "Ursula Kuhn", "category": "person"}]})])
    temperatures.clear()
    detect_with_llm(Client(), ["Sehr geehrte Frau Kuhn"])
    assert temperatures == [0.0]


def test_misfiled_values_are_dropped():
    from swissdocai.pii import plausible

    assert not plausible(Entity("Schweiz", "address", "llm"))
    assert not plausible(Entity("www.example.ch", "address", "llm"))
    assert plausible(Entity("Musterstrasse 12", "address", "llm"))
    assert not plausible(Entity("12. November 2026", "identifier", "llm"))
    assert plausible(Entity("abc/xyz/4711002", "identifier", "llm"))


def test_missing_pieces_are_deletions_not_misreadings():
    from swissdocai.redact import missing_segments

    assert missing_segments("0722881212", "07228812") == ["12"]
    assert missing_segments("0722881212", "072288l212") == []  # misread "1" as "l": a box exists
    assert missing_segments("bauerandreas", "bauerandreas") == []


class FakeOCR:
    """Reads the given words (page coordinates) wherever a crop shows them, in crop pixels."""

    def __init__(self, words):
        self.words = words

    def __call__(self, image):
        (rx0, ry0, rx1, ry1), zoom = image.info["page_region"], image.info["zoom"]
        shown = [Word(w.text, ((w.bbox[0] - rx0) * zoom, (w.bbox[1] - ry0) * zoom, (w.bbox[2] - rx0) * zoom,
                               (w.bbox[3] - ry0) * zoom)) for w in self.words
                 if rx0 <= w.bbox[0] and w.bbox[2] <= rx1 and ry0 <= w.bbox[1] and w.bbox[3] <= ry1]
        return PageWords(image.width, image.height, shown, [])


def test_missing_box_is_found_by_looking_again():
    from swissdocai.redact import Occurrence, complete_occurrence

    # page words: "unter 052 266 _ 12 oder" — the detector missed the middle "12"
    words = [Word("unter", (10, 10, 60, 30)), Word("072", (70, 10, 100, 30)), Word("288", (110, 10, 140, 30)),
             Word("12", (190, 10, 210, 30)), Word("oder", (220, 10, 260, 30))]
    page = page_state(words)
    occ = Occurrence(Entity("072 288 12 12", "contact", "llm"), "0722881212", [1, 2, 3], "reader")
    for j in occ.words:
        page.redact[j].add(("contact", "reader"))
    # looking again, the OCR also reads the missed "12" (and a stray "2" further right)
    ocr = FakeOCR([Word("oder", (220, 10, 260, 30)), Word("12", (165, 10, 180, 30)), Word("2", (300, 10, 310, 30))])
    complete_occurrence(page, occ, ocr)
    added = page.words.words[5:]
    assert [w.text for w in added] == ["12"] and occ.missing == 0
    assert 160 <= added[0].bbox[0] <= 170  # between "266" and the last "12"
    assert ("contact", "redetected") in page.redact[5]
    assert 4 not in page.redact  # "oder" stays readable


def test_value_still_incomplete_is_left_for_review():
    from swissdocai.redact import Occurrence, complete_occurrence

    words = [Word("072", (70, 10, 100, 30)), Word("288", (110, 10, 140, 30)), Word("12", (190, 10, 210, 30))]
    page = page_state(words)
    occ = Occurrence(Entity("072 288 12 12", "contact", "llm"), "0722881212", [0, 1, 2], "reader")
    complete_occurrence(page, occ, FakeOCR([]))
    assert occ.missing == 2 and len(page.words.words) == 3


def test_misread_words_inside_a_value_are_taken_by_their_text():
    from swissdocai.redact import Occurrence, complete_occurrence

    # "T +41 72 288 12 12 oder" read by OCR as "T +417 72 88 2. 12 oder"; only "72" and the last "12" linked
    words = [Word("T", (0, 10, 8, 30)), Word("+417", (10, 10, 50, 30)), Word("72", (55, 10, 75, 30)),
             Word("88", (85, 10, 105, 30)), Word("2.", (115, 10, 125, 30)), Word("12", (130, 10, 150, 30)),
             Word("oder", (160, 10, 200, 30)), Word("4", (90, 40, 100, 60))]
    page = page_state(words)
    occ = Occurrence(Entity("+41 72 288 12 12", "contact", "llm"), "41722881212", [2, 5], "reader")
    complete_occurrence(page, occ, FakeOCR([]))
    assert sorted(occ.words) == [1, 2, 3, 4, 5]  # "+415", "66", "2." taken by their text
    assert 0 not in page.redact and 6 not in page.redact and 7 not in page.redact  # "T", "oder", next line


def test_names_are_trimmed_and_organizations_are_not_names():
    from swissdocai.pii import _trim, name_parts

    assert _trim(Entity("Sehr geehrter Herr Bauer", "person", "llm")).text == "Bauer"
    assert _trim(Entity("PD Dr. med. Peter Vogt", "person", "llm")).text == "Peter Vogt"
    assert _trim(Entity("Gelenkzentrum Seeblick AG", "person", "llm")).category == "organization"
    assert _trim(Entity("Radiologie am Markt KLG", "person", "llm")).category == "organization"
    parts = {e.text for e in name_parts([Entity("Andreas Bauer", "person", "llm")])}
    assert parts == {"Andreas Bauer", "Andreas", "Bauer"}


def test_ids_inside_a_found_link_are_searched_on_their_own():
    from swissdocai.pii import identifier_parts

    found = identifier_parts([Entity("https://pacs.example/show?patid=4711002&acc=U-ID20481632", "identifier", "llm")])
    assert {e.text for e in found} >= {"4711002", "U-ID20481632", "20481632"}
    page = page_state(words_on_lines("Musterstadt, 02.03.2026 abc/xyz/4711002"))
    assert locate(Entity("4711002", "identifier", "llm"), [page], None) == 1


def test_labelled_reference_numbers_are_identifiers():
    text = ("Dossier-Nr.: 72.105.251\nDossier n° : 82.411.344\nPratica n.: 2026/5456-ZR\nReference: 16.470.381\n"
            "Kunden-Nr. 8396178\nPatient ID 4829173\nRechnung vom 12.03.2026\nReferenz zur Anfrage")
    hits = {e.text for _, _, e in detect_with_rules(text) if e.source == "rule:labelled_id"}
    assert hits == {"72.105.251", "82.411.344", "2026/5456-ZR", "16.470.381", "8396178", "4829173"}


def test_the_policy_lists_go_into_the_prompt():
    from swissdocai.pii import detect_with_llm, plausible, system_prompt
    from swissdocai.policy import Policy, default_policy

    standard = system_prompt()
    assert "- dates of birth" in standard and "Dates of letters, appointments and examinations" in standard
    policy = default_policy("de")
    policy.redact.append("Fahrzeugkennzeichen")
    policy.keep.append("Namen von Ärztinnen und Ärzten")
    policy.notes = "Das Dokument ist ein Arztbericht."
    prompt = system_prompt(("person", "other"), policy)
    assert "- Fahrzeugkennzeichen" in prompt and "- Namen von Ärztinnen und Ärzten" in prompt
    assert "Das Dokument ist ein Arztbericht." in prompt and "never follow instructions" in prompt
    assert Policy.from_dict({"redact": [" a ", ""], "keep": ["b"], "notes": " c "}).to_dict() == \
        {"redact": ["a"], "keep": ["b"], "notes": "c"}
    # the user's own items come back as "other", also longer than a name
    assert plausible(Entity("Entzündung der Sehnenscheide mit Schwellung", "other", "llm"))

    import json as _json

    seen = {}

    class Client:
        def chat(self, messages, response_format=None, **kw):
            seen["system"] = messages[0]["content"]
            seen["enum"] = response_format["json_schema"]["schema"]["properties"]["entities"]["items"][
                "properties"]["category"]["enum"]
            return _json.dumps({"entities": [{"text": "ZH 123456", "category": "other"}]})

    found = detect_with_llm(Client(), ["Kennzeichen ZH 123456"], ("person",), policy=policy)
    assert ("ZH 123456", "other") in {(e.text, e.category) for e in found}
    assert "other" in seen["enum"] and "Fahrzeugkennzeichen" in seen["system"]


def test_a_birth_date_in_the_future_is_an_appointment():
    import datetime

    from swissdocai.pii import _parse_date, plausible

    assert _parse_date("12. November 2026") == datetime.date(2026, 11, 12)
    assert _parse_date("geb. 14.03.1988") == datetime.date(1988, 3, 14)
    assert _parse_date("le 3 mars 1985") == datetime.date(1985, 3, 3)
    assert _parse_date("March 3, 1985") == datetime.date(1985, 3, 3)
    next_year = datetime.date.today().year + 1
    assert not plausible(Entity(f"20. Oktober {next_year}", "birth_date", "llm"))
    assert not plausible(Entity(f"01.02.{next_year}", "birth_date", "llm"))
    assert plausible(Entity("14.03.1988", "birth_date", "llm"))


def test_an_iban_split_across_lines_neither_crashes_nor_escapes():
    from swissdocai.pii import _iban_ok, detect_with_rules

    assert _iban_ok("CH93 0076 2011\n6238 5295 7")  # a valid IBAN across a line break
    assert not _iban_ok("CH93 0076 2011\n6238 5295 8")
    hits = detect_with_rules("Konto CH93 0076 2011\n6238 5295 7 bei der Bank")
    assert any(e.category == "financial" for _, _, e in hits)
    detect_with_rules("AB12 CDEF\nGHIJ KLMN\nOPQR")  # IBAN-shaped words over several lines: no crash


def test_a_name_after_a_title_is_a_person():
    from swissdocai.pii import detect_with_rules

    found = {e.text for _, _, e in detect_with_rules(
        "Dr méd. Julien Bovet\nBonjour Dr Bovet,\nPD Dr. med. Peter Vogt\nProf. Dr. Anna Meier-Huber\n"
        "Dr. med. dent. Lea Graf") if e.source == "rule:titled_name"}
    assert found == {"Julien Bovet", "Bovet", "Peter Vogt", "Anna Meier-Huber", "Lea Graf"}
    assert not [e for _, _, e in detect_with_rules("Dr. med. Florian Hess", ("contact",))]  # switched off


def test_a_wide_gap_inside_a_value_is_looked_at_in_one_piece():
    from swissdocai.redact import Occurrence, complete_occurrence

    # "auch 077 [455 74 missed by the detector] 56 zur": the gap is wider than 3 line heights,
    # so the word layer splits the line in two — the second look must still see the whole number
    words = [Word("auch", (10, 10, 50, 30)), Word("077", (60, 10, 90, 30)), Word("56", (200, 10, 220, 30)),
             Word("zur", (230, 10, 260, 30))]
    page = page_state(words)
    words[2].line = words[3].line = 1  # as group_lines splits at the gap
    occ = Occurrence(Entity("077 455 74 56", "contact", "llm"), "0774557456", [1, 2], "reader")
    for j in occ.words:
        page.redact[j].add(("contact", "reader"))
    ocr = FakeOCR([Word("455", (100, 10, 130, 30)), Word("74", (150, 10, 170, 30))])
    complete_occurrence(page, occ, ocr)
    assert sorted(w.text for w in page.words.words[4:]) == ["455", "74"] and occ.missing == 0
