"""Desktop app: the review logic the window calls, and its texts (no models, no GPU)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image  # noqa: E402

from swissdocai.pii import Entity  # noqa: E402
from swissdocai.pipeline import SourcePage  # noqa: E402
from swissdocai.redact import (  # noqa: E402
    PageState, RedactionJob, kept_readable, locate, page_boxes, render_job, set_entity_redacted, toggle_at,
    word_stream,
)
from swissdocai.words import PageWords, Word, group_lines  # noqa: E402


def make_job(tmp_path) -> RedactionJob:
    texts = ["Sehr", "geehrte", "Frau", "Kuhn", "Termin", "bei", "Ursula", "Kuhn"]
    words = [Word(t, (20 + 110 * (i % 4), 20 + 60 * (i // 4), 120 + 110 * (i % 4), 60 + 60 * (i // 4)))
             for i, t in enumerate(texts)]
    source = SourcePage(1, Image.new("RGB", (500, 200), "white"), None, "pixel", 1.0, size_pt=(360, 144))
    page = PageState(source, PageWords(500, 200, words, group_lines(words)))
    page.stream = word_stream(words)
    page.word_spans = [None] * len(words)
    entity = Entity("Ursula Kuhn", "person", "llm")
    surname = Entity("Kuhn", "person", "llm")
    for e in (entity, surname):
        locate(e, [page], None)
    return RedactionJob(tmp_path / "letter.pdf", [page], [entity, surname], [], ("person",), 72)


def center(word):
    x0, y0, x1, y1 = word.bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def test_clicks_switch_words_and_remove_boxes(tmp_path):
    page = make_job(tmp_path).pages[0]
    kuhn = 3  # "Kuhn" in the salutation: found
    assert kuhn in page.redact
    assert toggle_at(page, *center(page.words.words[kuhn])) and kuhn in page.removed  # kept readable
    assert toggle_at(page, *center(page.words.words[kuhn])) and kuhn not in page.removed  # redacted again
    termin = 4  # not found: a click blacks it out by hand
    assert toggle_at(page, *center(page.words.words[termin])) and ("manual", "manual") in page.redact[termin]
    assert not toggle_at(page, 480, 190)  # empty paper: nothing to switch
    page.manual_boxes.append((400, 150, 480, 190))
    assert any(category == "manual" for _, category in page_boxes(page))
    assert toggle_at(page, 440, 170) and not page.manual_boxes  # a click removes a drawn box


def test_unticking_a_value_keeps_it_readable_everywhere(tmp_path):
    job = make_job(tmp_path)
    surname = job.entities[1]
    set_entity_redacted(job, surname, False)
    assert {3, 7} <= job.pages[0].removed and kept_readable(job, surname)
    set_entity_redacted(job, surname, True)
    assert not job.pages[0].removed and not kept_readable(job, surname)


def test_the_reviewers_changes_reach_the_pdf(tmp_path):
    job = make_job(tmp_path)
    page = job.pages[0]
    page.removed.update({3})
    page.manual_boxes.append((400, 150, 480, 190))
    out = tmp_path / "letter.redacted.pdf"
    report = render_job(job, out, verify=False)
    assert out.exists()
    assert report["pages"][0]["keptReadable"] == 1 and report["pages"][0]["manualBoxes"] == 1
    assert all(item["box_pt"] != [round(v * 72 / 150, 2) for v in page.words.words[3].bbox]
               for item in report["pages"][0]["words"])


def test_qr_codes_are_blacked_out_whole_and_can_be_kept_readable(tmp_path):
    from swissdocai.redact import set_qr_code_redacted

    job = make_job(tmp_path)
    page = job.pages[0]
    page.qr_codes = [(400, 120, 480, 190)]
    assert ((400, 120, 480, 190), "qr_code") in page_boxes(page)
    assert toggle_at(page, 440, 160) and not any(c == "qr_code" for _, c in page_boxes(page))  # kept readable
    assert toggle_at(page, 440, 160) and ((400, 120, 480, 190), "qr_code") in page_boxes(page)  # and back
    set_qr_code_redacted(page, 0, False)
    assert render_job(job, tmp_path / "kept.pdf", verify=False)["pages"][0]["qrCodes"] == 0
    set_qr_code_redacted(page, 0, True)
    assert render_job(job, tmp_path / "out.pdf", verify=False)["pages"][0]["qrCodes"] == 1


def test_the_qr_code_switch_reaches_the_analysis(monkeypatch):
    from swissdocai.app import session

    assert session.Settings().redact_qr_codes  # unless the user switches it off
    assert session.Settings.from_dict({"language": "fr"}).redact_qr_codes  # also for settings saved before
    seen = {}
    monkeypatch.setattr(session, "analyze_document", lambda *args, **kwargs: seen.update(kwargs))
    for on in (True, False):
        engine = session.Session(session.Settings(reader="none", redact_qr_codes=on))
        monkeypatch.setattr(engine, "pii_client", lambda: None)
        monkeypatch.setattr(engine, "ocr", lambda: None)
        engine.analyze("letter.pdf", None)
        assert seen["qr_codes"] is on


def test_the_window_speaks_four_languages_completely():
    locales = Path(__file__).parent.parent / "desktop" / "src" / "locales"

    def keys(tree, prefix=""):
        out = {}
        for key, value in tree.items():
            if isinstance(value, dict):
                out.update(keys(value, f"{prefix}{key}."))
            else:
                out[f"{prefix}{key}"] = value
        return out

    texts = {lang: keys(json.loads((locales / f"{lang}.json").read_text(encoding="utf-8")))
             for lang in ("de", "fr", "it", "en")}
    assert all(set(t) == set(texts["de"]) for t in texts.values())
    placeholders = lambda s: sorted(part.split("}}")[0] for part in s.split("{{")[1:])  # noqa: E731
    for key, german in texts["de"].items():
        assert all(placeholders(t[key]) == placeholders(german) for t in texts.values()), key
        assert "ß" not in german, key  # Swiss spelling
