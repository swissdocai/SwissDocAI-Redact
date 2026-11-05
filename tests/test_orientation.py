"""Pages that lie sideways or upside down: straightened to be read, and written straight.

The first tests need no models. Those at the end read a dense test page with the real OCR on the
CPU and are skipped when the models are not in the cache."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from swissdocai.pii import Entity  # noqa: E402
from swissdocai.pipeline import SourcePage  # noqa: E402
from swissdocai.redact import (  # noqa: E402
    PageState, RedactionJob, analyze_document, burn, render_job, straighten, turn_box, verify_redaction,
)
from swissdocai.words import PageWords, Word, group_lines  # noqa: E402

ANGLES = (90, 180, -90)


def sent_image(straight: Image.Image, angle: int) -> Image.Image:
    """The page as a scanner delivers it: content turned clockwise by ``angle``."""
    return straight.rotate(-angle, expand=True)


class Orientation:
    """A fake model: always the same answer, and counts its calls."""

    def __init__(self, angle=0, confidence=1.0):
        self.answer, self.calls = (angle, confidence), 0

    def __call__(self, image):
        self.calls += 1
        return self.answer


def image_page(image, **fields) -> SourcePage:
    return SourcePage(1, image, None, "pixel", 1.0, size_pt=(image.width * 0.48, image.height * 0.48), **fields)


# --- the boxes ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", ANGLES)
def test_a_box_lands_on_the_same_pixels_of_the_page_as_it_came_in(angle):
    straight = Image.new("RGB", (300, 200), "white")
    box = (40, 30, 90, 70)
    marked = straight.copy()
    marked.paste("black", box)  # exactly the pixels of the box
    wanted = np.asarray(sent_image(marked, angle).convert("L")) == 0
    burned = np.asarray(burn(sent_image(straight, angle), [turn_box(box, angle, *straight.size)]).convert("L")) == 0
    assert burned[wanted].all()  # every pixel of the box is black
    assert burned.sum() - wanted.sum() <= (box[2] - box[0] + box[3] - box[1]) + 1  # and at most a pixel more per side


def test_the_formulas():
    assert turn_box((10, 20, 30, 40), 0, 100, 200) == (10, 20, 30, 40)
    assert turn_box((10, 20, 30, 40), -90, 100, 200) == (20, 70, 40, 90)
    assert turn_box((10, 20, 30, 40), 180, 100, 200) == (70, 160, 90, 180)
    assert turn_box((10, 20, 30, 40), 90, 100, 200) == (160, 10, 180, 30)
    assert turn_box((0.1, 0.2, 0.3, 0.4), 90, 1, 1) == pytest.approx((0.6, 0.1, 0.8, 0.3))
    with pytest.raises(ValueError):
        turn_box((0, 0, 1, 1), 45, 100, 200)


# --- which pages are turned ----------------------------------------------------------------


def test_a_page_with_a_text_layer_is_never_turned_and_the_model_is_not_asked():
    orientation = Orientation(90)
    words = [Word(t, (10 * i, 0, 10 * i + 8, 10)) for i, t in enumerate(["Sehr", "geehrte", "Frau", "Kuhn"])]
    source = image_page(Image.new("RGB", (300, 200), "white"), pdf_marks=[])
    source.pdf_words = words
    assert straighten(source, orientation) == (source, 0, False)
    assert orientation.calls == 0


def test_a_broken_text_layer_does_not_count():
    orientation = Orientation(90)
    source = image_page(Image.new("RGB", (300, 200), "white"))
    source.pdf_words = [Word("(cid:12)(cid:13)", (0, 0, 8, 10))] * 3
    straight, angle, _ = straighten(source, orientation)
    assert angle == 90 and orientation.calls == 1
    assert straight.pdf_words is None and straight.pdf_marks is None


@pytest.mark.parametrize("confidence", [0.79, 0.5, 0.0])
def test_an_unsure_orientation_leaves_the_page_as_it_is(confidence):
    source = image_page(Image.new("RGB", (300, 200), "white"))
    straight, angle, unknown = straighten(source, Orientation(90, confidence))
    assert straight is source and angle == 0 and unknown


class Model:
    """A fake orientation model: sees where a red pixel lies, and is weak at -90 degrees."""

    def __init__(self, weak=(90, 0.55), sure=0.97):
        self.weak, self.sure, self.calls = weak, sure, 0

    def __call__(self, pages):
        self.calls += 1
        page = pages[0]
        height, width = page.shape[:2]
        corners = {(0, 0): 0, (0, width - 1): 90, (height - 1, width - 1): 180, (height - 1, 0): -90}
        angle = next((a for (y, x), a in corners.items() if tuple(page[y, x]) == (255, 0, 0)), None)
        if angle == -90:
            angle, confidence = self.weak
        elif angle is None:
            angle, confidence = 0, 0.3
        else:
            confidence = self.sure
        return [0], [angle], [confidence]


def orientation_with(model):
    from swissdocai.words import PageOrientation

    orientation = PageOrientation()
    orientation._predictor = model
    return orientation


def marked_page() -> Image.Image:
    straight = Image.new("RGB", (300, 200), "white")
    straight.putpixel((0, 0), (255, 0, 0))
    return straight


def test_a_sure_answer_needs_one_look():
    model = Model()
    assert orientation_with(model)(sent_image(marked_page(), 90)) == (90, 0.97) and model.calls == 1


def test_an_unsure_page_is_shown_in_all_four_positions_and_they_vote():
    model = Model()
    angle, confidence = orientation_with(model)(sent_image(marked_page(), -90))
    assert model.calls == 4  # four small looks at the page, the OCR does not read it
    assert angle == -90 and confidence == pytest.approx(0.97)  # three agree: the weak answer loses
    source = image_page(sent_image(marked_page(), -90))
    straight, found, unknown = straighten(source, orientation_with(Model()))
    assert found == -90 and not unknown and straight.image.size == (300, 200)


def test_two_answers_that_agree_are_not_enough():
    class Split(Model):
        def __call__(self, pages):
            self.calls += 1
            return [0], [[90, 90, 180, 180][self.calls - 1]], [0.7]

    assert orientation_with(Split())(marked_page())[1] < 0.8


def test_a_page_without_text_stays_unsure():
    source = image_page(Image.new("RGB", (300, 200), "white"))
    assert straighten(source, orientation_with(Model())) == (source, 0, True)


def test_exactly_the_threshold_turns_the_page():
    assert straighten(image_page(Image.new("RGB", (300, 200), "white")), Orientation(180, 0.8))[1] == 180


def test_a_quarter_turn_swaps_width_and_height():
    source = image_page(Image.new("RGB", (300, 200), "white"))
    for angle in (90, -90):
        straight, _, _ = straighten(source, Orientation(angle))
        assert straight.image.size == (200, 300) and straight.size_pt == pytest.approx((96, 144))
    straight, _, _ = straighten(source, Orientation(180))
    assert straight.image.size == (300, 200) and straight.size_pt == pytest.approx((144, 96))


@pytest.mark.parametrize("angle", ANGLES)
def test_a_signature_image_stays_where_it_is_on_the_straight_page(angle):
    straight_image = Image.new("RGB", (300, 200), "white")
    box = (40, 30, 90, 70)
    sent = sent_image(straight_image, angle)
    on_sent = turn_box(box, angle, *straight_image.size)
    straight, _, _ = straighten(image_page(sent, images=[on_sent]), Orientation(angle))
    assert straight.images[0] == pytest.approx(box)


# --- through analysis and output -----------------------------------------------------------


class FakeOcr:
    """"Reads" one word at a fixed place of the page, whatever way up it is."""

    def __init__(self):
        self.images = []

    def __call__(self, image):
        self.images.append(image)
        words = [Word("Kuhn", (40, 30, 90, 70)), Word("Termin", (120, 30, 190, 70)), Word("heute", (200, 30, 260, 70))]
        return PageWords(image.width, image.height, words, group_lines(words), source="ocr")


class FakeClient:
    model = "fake"

    def detect(self, pages, categories, policy=None):
        return [Entity("Kuhn", "person", "llm")]


def test_words_codes_the_reader_and_the_pdf_all_get_the_straight_page(tmp_path, monkeypatch):
    from swissdocai import redact

    straight = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(straight).rectangle((0, 0, 20, 20), fill="red")
    sent = sent_image(straight, 90)
    path = tmp_path / "scan.png"
    sent.save(path)
    read, qr = [], []
    monkeypatch.setattr(redact, "find_qr_codes", lambda image: qr.append(image.size) or [])

    def reader(image):
        read.append(image)
        return "Kuhn Termin heute"

    ocr = FakeOcr()
    job = analyze_document(path, FakeClient(), reader=reader, ocr=ocr, orientation=Orientation(90))
    page = job.pages[0]
    assert page.angle == 90 and page.source.image.size == (300, 200)
    assert ocr.images[0].size == (300, 200) and read[0].size == (300, 200) and qr == [(300, 200)]
    assert np.array_equal(np.asarray(read[0]), np.asarray(straight))

    out = tmp_path / "out.pdf"
    report = render_job(job, out, verify=False)
    assert report["pages"][0]["angle"] == 90
    box = page.words.words[0].bbox
    assert report["pages"][0]["words"][0]["box_pt"] == [round(v * 72 / 150, 2) for v in box]

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(out))
    assert pdf[0].get_size() == pytest.approx((300 * 72 / 150, 200 * 72 / 150))
    rendered = np.asarray(pdf[0].render(scale=150 / 72).to_pil().convert("RGB")).astype(int)
    pdf.close()
    x0, y0, x1, y1 = (round(v) for v in box)
    assert rendered[(y0 + y1) // 2, (x0 + x1) // 2].sum() < 60  # the bar sits on the word
    assert rendered[8, 8, 0] > 180 and rendered[8, 8, 1] < 90  # and the red corner is top left again


def test_the_check_reads_the_straight_page_with_the_angle_of_the_analysis():
    ocr = FakeOcr()
    sent = Image.new("RGB", (200, 300), "white")
    report = verify_redaction([sent], [Entity("Kuhn", "person", "llm")], ocr, angles=[90])
    assert ocr.images[0].size == (300, 200) and not report["ok"]


def test_a_page_of_unknown_orientation_is_checked_in_all_four_directions():
    class ReadsOnlyUpsideDown(FakeOcr):
        def __call__(self, image):
            found = super().__call__(image)
            found.words = found.words if image.getpixel((0, 0)) == (255, 0, 0) else []
            return found

    image = Image.new("RGB", (300, 200), "white")
    image.putpixel((299, 199), (255, 0, 0))  # top left once it is turned by 180 degrees
    ocr = ReadsOnlyUpsideDown()
    assert not verify_redaction([image], [Entity("Kuhn", "person", "llm")], ocr, angles=[None])["ok"]
    assert len(ocr.images) == 4
    assert verify_redaction([image], [Entity("Kuhn", "person", "llm")], ReadsOnlyUpsideDown(), angles=[0])["ok"]


def test_the_report_names_a_page_of_unknown_orientation(tmp_path):
    source = image_page(Image.new("RGB", (300, 200), "white"))
    page = PageState(source, PageWords(300, 200, [], []), orientation_unknown=True)
    job = RedactionJob(tmp_path / "scan.png", [page], [], [], ("person",), 200)
    report = render_job(job, tmp_path / "out.pdf", verify=False)
    assert report["pages"][0]["orientationUnknown"] is True


# --- the setup and the model file ----------------------------------------------------------


def test_the_setup_downloads_the_orientation_model_with_the_others(tmp_path, monkeypatch):
    pytest.importorskip("onnxtr")
    from swissdocai import engine, words

    monkeypatch.setenv("SWISSDOCAI_CACHE", str(tmp_path))
    fetched = []
    monkeypatch.setattr(engine, "download", lambda url, target, **kwargs: fetched.append((url, target)))
    words.ensure_ocr_models()
    assert any("mobilenet_v3_small_page_orientation" in url and "8_bit" not in url
               and target.parent == tmp_path / "onnxtr" / "models" for url, target in fetched)

    from swissdocai.app.session import Session, Settings

    session = Session(Settings(engine="swissdocai", reader="none"))
    assert any("page_orientation" in name for name, _ in session.downloads_needed())


def test_a_document_never_downloads_the_model(tmp_path, monkeypatch):
    pytest.importorskip("onnxtr")
    import httpx

    from swissdocai import words

    def no_network(*args, **kwargs):
        raise AssertionError("no download while reading a document")

    monkeypatch.setenv("SWISSDOCAI_CACHE", str(tmp_path))
    monkeypatch.setattr(httpx, "stream", no_network)
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(FileNotFoundError, match="setup"):
        words.PageOrientation()(Image.new("RGB", (64, 64), "white"))


# --- the real models on a dense page -------------------------------------------------------


WORDS = ("Versicherung Kunde Vertrag Antrag Leistung Rechnung Betrag Datum Adresse Strasse Gemeinde Kanton Termin "
         "Bericht Praxis Patient Behandlung Diagnose Rezept Therapie Kontrolle Abteilung Sekretariat Unterschrift "
         "Zahlung Konto Bank Monat Woche Jahr Mitarbeiter Arbeitgeber Lohn Abzug Beitrag Rente Pension Steuer "
         "Einkommen Wohnung Miete Kaution Schaden Meldung Fall Akte Nummer Seite").split()


def dense_page(size=30, columns=(150,), width=1350, gap=46) -> tuple[Image.Image, list[tuple[str, tuple]]]:
    """A4 at 200 dpi with about 500 words under a letterhead, and where each lies."""
    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    draw.text((150, 130), "Kantonsspital Musterstadt", fill="black", font=ImageFont.load_default(size=44))
    font = ImageFont.load_default(size=size)
    placed, n = [], 0
    for left in columns:
        x, y = left, 300
        while y < 2200:
            word = f"{WORDS[n % len(WORDS)]}{n // len(WORDS) + 1}"
            box = draw.textbbox((x, y), word, font=font)
            if box[2] > left + width:
                x, y = left, y + gap
                continue
            n += 1
            draw.text((x, y), word, fill="black", font=font)
            placed.append((word, box))
            x = box[2] + 22
    return image, placed


def unsure_page():
    """A layout the orientation model is unsure about when it lies at -90 degrees."""
    return dense_page(size=26, columns=(150, 860), width=640, gap=38)


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("onnxtr")
    from swissdocai import words
    from swissdocai.engine import cache_dir

    os.environ["SWISSDOCAI_OCR_DEVICE"] = "cpu"  # never the GPU in a test
    reader = words.OnnxWordReader()
    folder = cache_dir() / "onnxtr" / "models"
    names = [url.rpartition("/")[2] for url in words._onnx_model_urls(reader.det_arch, reader.reco_arch)]
    if not all((folder / name).exists() for name in names):
        pytest.skip(f"OCR models not in {folder}")
    return reader, words.PageOrientation()


class NoPii:
    model = "none"

    def detect(self, pages, categories, policy=None):
        return []


def read_scan(tmp_path, angle, models) -> tuple[RedactionJob, list]:
    reader, orientation = models
    straight, placed = dense_page()
    path = tmp_path / f"scan{angle}.png"
    sent_image(straight, angle).save(path)
    return analyze_document(path, NoPii(), ocr=reader, orientation=orientation, qr_codes=False), placed


@pytest.mark.parametrize("angle", [0, *ANGLES])
def test_the_model_reports_the_clockwise_angle_of_the_content(angle, models):
    straight, _ = dense_page()
    found, confidence = models[1](sent_image(straight, angle))
    assert found == angle and confidence >= 0.8


@pytest.mark.parametrize("angle", ANGLES)
def test_a_turned_page_is_read_as_well_as_a_straight_one_and_the_boxes_land_on_the_words(tmp_path, angle, models):
    job, placed = read_scan(tmp_path, angle, models)
    page = job.pages[0]
    assert page.angle == angle
    found: dict[str, list] = {}
    for word in page.words.words:
        found.setdefault(word.text, []).append(word.bbox)
    assert sum(text in found for text, _ in placed) / len(placed) >= 0.98

    width = page.source.image.width
    distances = []
    for text, box in placed:
        if len(found.get(text, [])) != 1:  # only words that occur once on the page
            continue
        read = found[text][0]
        distances.append(max(abs((read[0] + read[2]) - (box[0] + box[2])) / 2,
                             abs((read[1] + read[3]) - (box[1] + box[3])) / 2))
    assert len(distances) > 0.9 * len(placed)
    assert max(distances) / width <= 0.002


def test_a_page_the_model_is_unsure_about_is_straightened_all_the_same(tmp_path, models):
    reader, orientation = models
    straight, placed = unsure_page()
    sent = sent_image(straight, -90)
    assert orientation._ask(sent)[1] < 0.8  # else this page tests nothing
    path = tmp_path / "scan.png"
    sent.save(path)
    page = analyze_document(path, NoPii(), ocr=reader, orientation=orientation, qr_codes=False).pages[0]
    assert page.angle == -90 and not page.orientation_unknown
    texts = {word.text for word in page.words.words}
    assert sum(text in texts for text, _ in placed) / len(placed) >= 0.98


def test_a_redacted_page_upside_down_passes_the_check_only_when_nothing_is_readable(tmp_path, models):
    reader, _ = models
    job, placed = read_scan(tmp_path, 180, models)
    page = job.pages[0]
    secret = placed[100][0]
    entity = Entity(secret, "person", "llm")
    other = next(j for j, word in enumerate(page.words.words) if word.text == placed[5][0])
    target = next(j for j, word in enumerate(page.words.words) if word.text == secret)

    page.redact[other].add(("person", "exact"))  # the bar sits on the wrong word: the value stays readable
    job.entities = [entity]
    report = render_job(job, tmp_path / "leak.pdf")
    assert not report["verification"]["ok"]
    upside_down = page.source.image.rotate(180)
    assert verify_redaction([upside_down], [entity], reader)["ok"]  # read as it lay, it would have passed
    assert not verify_redaction([upside_down], [entity], reader, angles=[180])["ok"]

    page.redact.clear()
    page.redact[target].add(("person", "exact"))
    assert render_job(job, tmp_path / "clean.pdf")["verification"]["ok"]
