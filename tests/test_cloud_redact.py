"""The cloud option against a fake SwissDocAI server (a real HTTP server on this machine): which
endpoint a document goes to, the job's course, its errors, and what enters the review."""

import base64
import io
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from swissdocai import saas  # noqa: E402
from swissdocai.app.api import findings, page_view  # noqa: E402
from swissdocai.app.session import Session, Settings  # noqa: E402
from swissdocai.policy import Policy  # noqa: E402
from swissdocai.redact import (  # noqa: E402
    analyze_document, analyze_in_cloud, cloud_values, page_boxes, render_job, set_entity_redacted,
    set_qr_code_redacted, toggle_at, turn_box,
)
from swissdocai.saas import (  # noqa: E402
    SaasBusy, SaasClient, SaasError, SaasGone, SaasNoCredit, SaasTooLarge, SaasToken, SaasUnreadable,
    SwissDocAIPii, SwissDocAIReader, SwissDocAIRedact,
)
from swissdocai.words import PageWords, Word, group_lines  # noqa: E402

SECRET = "Ursula Kuhn"


def result() -> dict:
    return {
        "entities": [
            {"id": "e1", "text": "ursula.kuhn@example.ch", "category": "contact", "source": "rule:email", "found": True},
            {"id": "e2", "text": SECRET, "category": "person", "source": "llm", "found": True},
            {"id": "e3", "text": "756.9217.0769.85", "category": "identifier", "source": "rule:ahv", "found": False},
        ],
        "pages": [{"page": 1, "angle": 0, "width_pt": 240.0, "height_pt": 120.0, "word_source": "ocr"}],
        "items": [
            {"id": "p1-1", "page": 1, "kind": "value", "entity": "e1", "entities": ["e1", "e2"], "how": "exact",
             "missing_chars": 0, "boxes": [[0.1, 0.1, 0.5, 0.2]]},
            {"id": "p1-2", "page": 1, "kind": "value", "entity": "e2", "entities": ["e2"], "how": "reader",
             "missing_chars": 3, "boxes": [[0.1, 0.4, 0.9, 0.5], [0.1, 0.5, 0.3, 0.6]]},
            {"id": "p1-3", "page": 1, "kind": "qr_code", "boxes": [[0.7, 0.7, 0.9, 0.9]]},
            {"id": "p1-4", "page": 1, "kind": "signature", "boxes": [[0.1, 0.7, 0.3, 0.8]]},
        ],
    }


class FakeServer:
    """Answers each route from a script of (status, json), the last one again and again."""

    def __init__(self):
        self.requests: list[tuple[str, str, dict, dict]] = []
        self.script: dict[tuple[str, str], list] = {}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self, method):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                fake.requests.append((method, self.path, dict(self.headers), body))
                steps = fake.script.get((method, self.path))
                if not steps:
                    status, data = 404, {"detail": "Job not found"}
                else:
                    status, data = steps.pop(0) if len(steps) > 1 else steps[0]
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self.answer("GET")

            def do_POST(self):
                self.answer("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self) -> list[str]:
        return [path for _, path, _, _ in self.requests]

    def posted(self, path: str) -> list[dict]:
        return [body for method, p, _, body in self.requests if method == "POST" and p == path]

    def job(self, *states, path="/v1/redact", job_id="j1"):
        self.script["POST", path] = [(200, {"job_id": job_id, "document_id": "d"})]
        self.script["GET", f"/v1/redact/{job_id}"] = list(states)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(saas, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(saas.time, "sleep", lambda seconds: None)
    fake = FakeServer()
    yield fake
    fake.server.shutdown()


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "Rechnung Ursula Kuhn.png"
    Image.new("RGB", (500, 250), "white").save(path)
    return path


def client(server) -> SaasClient:
    return SaasClient(token="sd_test", base_url=server.url, retries=1)


def done(pages=1, data=None) -> tuple[int, dict]:
    return 200, {"job_id": "j1", "status": "done", "kind": "redact", "pages": pages, "result": data or result()}


# --- the job -----------------------------------------------------------------------------


def test_the_job_from_start_to_findings(server, scan):
    server.job((200, {"status": "queued", "position": 3, "wait_seconds": 45, "stage": "", "done": 0, "total": 0}),
               (200, {"status": "queued", "position": 1, "wait_seconds": 5}),
               (200, {"status": "running", "stage": "pages", "done": 0, "total": 0}),
               (200, {"status": "running", "stage": "reading", "done": 1, "total": 2}),
               (200, {"status": "running", "stage": "personal data", "done": 0, "total": 1}),
               done())
    seen = []
    policy = Policy(redact=["Namen"], keep=["Firmen"], notes="Arztbericht")
    job = analyze_document(scan, SwissDocAIPii(client(server)), reader=SwissDocAIReader(client(server)),
                           categories=("person", "contact"), policy=policy, qr_codes=False,
                           progress=lambda stage, done_, total: seen.append((stage, done_, total)))
    assert seen[:5] == [("queued", 3, 45), ("queued", 1, 5), ("pages", 0, 0), ("reading", 1, 2), ("personal data", 0, 1)]

    assert [path for path in server.paths() if path.startswith("/v1/") and "redact" not in path] == []
    body = server.posted("/v1/redact")[0]
    assert base64.b64decode(body["image_base64"]) == scan.read_bytes()
    assert body["categories"] == ["person", "contact"] and body["qr_codes"] is False
    assert body["policy"] == {"redact": ["Namen"], "keep": ["Firmen"], "notes": "Arztbericht"}
    assert not body.get("render")
    assert body["document_id"].startswith("swissdocai-redact-")  # never the file name
    assert "Kuhn" not in json.dumps({k: v for k, v in body.items() if k != "image_base64"})
    assert server.requests[0][2]["Authorization"] == "Bearer sd_test"
    assert len(job.entities) == 3 and [e.text for e in job.not_found] == ["756.9217.0769.85"]


def test_a_second_fetch_is_a_404_and_says_so(server):
    server.job((404, {"detail": "Job not found"}))
    with pytest.raises(SaasGone) as error:
        SwissDocAIRedact(client(server)).redact(b"%PDF", "d", ("person",))
    assert error.value.code == "gone"


@pytest.mark.parametrize("status, kind, code", [(400, SaasUnreadable, "unreadable"), (401, SaasToken, "token"),
                                                (413, SaasTooLarge, "tooLarge"), (429, SaasNoCredit, "credit"),
                                                (503, SaasBusy, "busy")])
def test_errors_when_starting(server, status, kind, code):
    server.script["POST", "/v1/redact"] = [(status, {"detail": "no"})]
    with pytest.raises(kind) as error:
        SwissDocAIRedact(client(server)).redact(b"%PDF", "d", ("person",))
    assert error.value.code == code
    expected = 2 if status == 503 else 1  # only a busy service is asked again
    assert len(server.posted("/v1/redact")) == expected


def test_a_job_that_fails_answers_with_its_http_error(server):
    server.job((200, {"status": "running", "stage": "words", "done": 0, "total": 1}), (413, {"detail": "At most 50 pages"}))
    with pytest.raises(SaasTooLarge, match="At most 50 pages"):
        SwissDocAIRedact(client(server)).redact(b"%PDF", "d", ("person",))


def test_a_service_that_is_briefly_away_does_not_lose_the_result(server):
    server.job((503, {"detail": "unavailable"}), (503, {"detail": "unavailable"}), done())
    assert SwissDocAIRedact(client(server)).redact(b"%PDF", "d", ("person",))["entities"]


def test_a_job_lost_during_an_outage_is_busy_not_gone(server):
    server.job((503, {"detail": "Too many documents"}), (404, {"detail": "Job not found"}))
    with pytest.raises(SaasBusy):
        SwissDocAIRedact(client(server)).redact(b"%PDF", "d", ("person",))


def test_a_service_that_stays_away_is_reported_as_busy(server):
    server.job((503, {"detail": "unavailable"}))
    with pytest.raises(SaasBusy):
        SwissDocAIRedact(client(server)).redact(b"%PDF", "d", ("person",))


def test_more_than_50_pages_never_go_up(server, tmp_path, monkeypatch):
    from swissdocai import redact

    path = tmp_path / "long.pdf"
    path.write_bytes(b"%PDF")
    pages = [redact.SourcePage(i + 1, Image.new("RGB", (10, 10)), None, "pixel", 1.0, size_pt=(5, 5)) for i in range(51)]
    monkeypatch.setattr(redact, "load_pages", lambda *args: pages)
    with pytest.raises(SaasTooLarge):
        analyze_in_cloud(path, SwissDocAIRedact(client(server)))
    assert not server.requests


# --- which way a document goes -----------------------------------------------------------


class Ocr:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        words = [Word("Ursula", (20, 20, 90, 40)), Word("Kuhn", (100, 20, 160, 40)), Word("Termin", (200, 20, 280, 40))]
        return PageWords(image.width, image.height, words, group_lines(words), source="ocr")


def session(server, monkeypatch, tmp_path, **settings) -> Session:
    monkeypatch.setenv("SWISSDOCAI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SWISSDOCAI_API_KEY", "sd_test")
    monkeypatch.setenv("SWISSDOCAI_API_URL", server.url)
    monkeypatch.setattr("swissdocai.app.secret.load_token", lambda: "sd_test")
    app = Session(Settings(redact_qr_codes=False, **settings))
    app._ocr = Ocr()
    app._orientation = lambda image: (0, 1.0)
    return app


def test_reading_and_finding_in_the_cloud_use_one_job(server, scan, monkeypatch, tmp_path):
    server.job(done())
    app = session(server, monkeypatch, tmp_path, engine="swissdocai", reader="swissdocai")
    job = app.analyze(scan, lambda *args: None)
    assert "/v1/redact" in server.paths() and "/v1/ocr" not in server.paths() and "/v1/pii" not in server.paths()
    assert app._ocr.calls == 0  # nothing is read here
    assert job.verifier is not None


def test_only_finding_in_the_cloud_stays_on_pii_and_sends_no_image(server, scan, monkeypatch, tmp_path):
    server.script["POST", "/v1/pii"] = [(200, {"entities": [{"text": SECRET, "category": "person"}], "pages": 1})]
    app = session(server, monkeypatch, tmp_path, engine="swissdocai", reader="none")
    job = app.analyze(scan, lambda *args: None)
    assert server.paths() == ["/v1/pii"]
    body = server.posted("/v1/pii")[0]
    assert set(body) == {"pages", "categories", "policy", "document_id"}
    assert body["pages"] == ["Ursula Kuhn Termin"]  # the text of the page
    assert all(isinstance(page, str) and len(page) < 1000 for page in body["pages"])
    assert "image_base64" not in json.dumps(body)
    assert job.verifier is None  # the check stays local


def test_only_reading_in_the_cloud_stays_on_ocr(server, scan, monkeypatch, tmp_path):
    server.script["POST", "/v1/ocr"] = [(200, {"text": "Ursula Kuhn Termin"})]
    app = session(server, monkeypatch, tmp_path, engine="local", reader="swissdocai")
    monkeypatch.setattr(app, "pii_client", lambda: type("Pii", (), {"detect": lambda *a: [], "model": "x"})())
    assert app.analyze(scan, lambda *args: None).verifier is None
    assert server.paths() == ["/v1/ocr"]


def test_nothing_in_the_cloud_sends_nothing(server, scan, monkeypatch, tmp_path):
    app = session(server, monkeypatch, tmp_path, engine="local", reader="none")
    monkeypatch.setattr(app, "pii_client", lambda: type("Pii", (), {"detect": lambda *a: [], "model": "x"})())
    app.analyze(scan, lambda *args: None)
    assert server.requests == []


# --- the review ----------------------------------------------------------------------------


def cloud_job(server, scan, data=None):
    server.job(done(data=data))
    return analyze_in_cloud(scan, SwissDocAIRedact(client(server)))


def test_findings_enter_the_review_on_the_page_as_it_is_shown(server, scan):
    job = cloud_job(server, scan)
    page = job.pages[0]
    view = page_view(job, 0)
    assert (view["width"], view["height"]) == (500, 250)
    boxes = {tuple(b["box"]): b["category"] for b in view["boxes"]}
    assert boxes[(50.0, 25.0, 250.0, 50.0)] == "contact"  # fractions times the shown page: no padding on top
    assert boxes[(50.0, 100.0, 450.0, 125.0)] == "person" and boxes[(50.0, 125.0, 150.0, 150.0)] == "person"
    assert boxes[(350.0, 175.0, 450.0, 225.0)] == "qr_code" and boxes[(50.0, 175.0, 150.0, 200.0)] == "signature"
    assert len(view["boxes"]) == 5  # every place once


def test_a_place_is_listed_under_its_value_and_a_check_is_asked_for(server, scan):
    job = cloud_job(server, scan)
    listed = findings(job)
    groups = {g["category"]: g["items"] for g in listed["groups"]}
    assert groups["contact"][0]["text"] == "ursula.kuhn@example.ch" and groups["contact"][0]["places"] == 1
    assert groups["person"][0]["text"] == SECRET and groups["person"][0]["places"] == 1
    assert listed["notFound"] == ["756.9217.0769.85"]
    assert [r["text"] for r in listed["review"]] == [SECRET]  # missing_chars > 0: yellow
    assert page_view(job, 0)["review"][0]["box"] == [50.0, 100.0, 450.0, 150.0]
    assert [q["box"] for q in listed["qrCodes"]] == [[350.0, 175.0, 450.0, 225.0]]


def test_a_place_is_switched_as_one(server, scan):
    job = cloud_job(server, scan)
    page = job.pages[0]
    assert toggle_at(page, 100, 110)  # one of the two lines of the name
    assert len([1 for _, c in page_boxes(page) if c == "person"]) == 0
    assert toggle_at(page, 100, 140)  # the other line, kept readable with it: both come back
    assert len([1 for _, c in page_boxes(page) if c == "person"]) == 2
    set_entity_redacted(job, job.entities[1], False)
    assert not [1 for _, c in page_boxes(page) if c == "person"]
    assert toggle_at(page, 100, 190) and not [1 for _, c in page_boxes(page) if c == "signature"]
    set_qr_code_redacted(page, 0, False)
    assert not [1 for _, c in page_boxes(page) if c == "qr_code"]


def test_a_box_of_an_unknown_kind_is_blacked_out_all_the_same(server, scan):
    data = result()
    data["items"] = [{"id": "x", "page": 1, "kind": "stamp", "boxes": [[0.1, 0.1, 0.2, 0.2]]},
                     {"id": "y", "page": 1, "kind": "value", "entity": "e9", "boxes": [[0.3, 0.3, 0.4, 0.4]]}]
    page = cloud_job(server, scan, data).pages[0]
    assert [c for _, c in page_boxes(page)] == ["manual", "manual"]


def test_findings_on_a_page_of_another_shape_are_refused(server, scan):
    data = result()
    data["pages"][0].update(width_pt=120.0, height_pt=240.0)
    with pytest.raises(SaasError, match="shape"):
        cloud_job(server, scan, data)


def test_an_answer_without_a_page_is_refused(server, scan):
    data = result()
    data["pages"] = []
    with pytest.raises(SaasError, match="every page"):
        cloud_job(server, scan, data)


@pytest.mark.parametrize("angle", [90, 180, -90])
def test_a_page_the_service_turned_is_reviewed_and_written_straight(server, tmp_path, angle):
    import pypdfium2 as pdfium

    straight = Image.new("RGB", (500, 250), "white")
    straight.paste("red", (0, 0, 30, 30))
    path = tmp_path / "scan.png"
    sent = straight.rotate(-angle, expand=True)
    sent.save(path)
    data = result()
    data["pages"][0].update(angle=angle, width_pt=sent.width * 0.48, height_pt=sent.height * 0.48)
    data["items"] = [{"id": "a", "page": 1, "kind": "value", "entity": "e2", "entities": ["e2"], "how": "exact",
                      "missing_chars": 0, "boxes": [list(turn_box((0.1, 0.4, 0.5, 0.6), angle, 1, 1))]}]
    job = cloud_job(server, path, data)
    view = page_view(job, 0)
    assert (view["width"], view["height"]) == (500, 250)
    assert view["boxes"][0]["box"] == pytest.approx([50, 100, 250, 150])

    out = tmp_path / "out.pdf"
    report = render_job(job, out, verify=False)
    assert report["pages"][0]["angle"] == angle
    pdf = pdfium.PdfDocument(str(out))
    assert pdf[0].get_size() == pytest.approx((240, 120))
    rendered = pdf[0].render(scale=150 / 72).to_pil().convert("RGB")
    pdf.close()
    assert rendered.getpixel((10, 10))[0] > 180 and rendered.getpixel((10, 10))[1] < 90  # the red corner is top left
    assert sum(rendered.getpixel((150, 125))) < 60  # and the bar on the straight page


def switched_rules() -> dict:
    data = result()
    data["entities"] += [
        {"id": "e4", "text": "079 555 01 23", "category": "contact", "source": "rule:phone", "found": True},
        {"id": "e5", "text": "praxis@example.ch", "category": "contact", "source": "llm", "found": True},
        {"id": "e6", "text": "071 222 33 44", "category": "contact", "source": "rule:phone", "found": False},
    ]
    data["items"] += [
        {"id": "p1-5", "page": 1, "kind": "value", "entity": "e4", "entities": ["e4"], "how": "exact",
         "missing_chars": 0, "boxes": [[0.5, 0.1, 0.7, 0.2]]},
        {"id": "p1-6", "page": 1, "kind": "value", "entity": "e5", "entities": ["e5"], "how": "exact",
         "missing_chars": 0, "boxes": [[0.5, 0.3, 0.7, 0.4]]},
    ]
    return data


def test_a_safety_rule_switched_off_also_counts_in_the_cloud(server, scan):
    server.job(done(data=switched_rules()))
    job = analyze_in_cloud(scan, SwissDocAIRedact(client(server)), rule_categories=("person", "identifier"))
    texts = {e.text for e in job.entities}
    assert "079 555 01 23" not in texts and "071 222 33 44" not in texts  # only the rule found them
    assert "praxis@example.ch" in texts  # the model found it too
    assert [e.text for e in job.not_found] == ["756.9217.0769.85"]
    page = job.pages[0]
    assert (250.0, 25.0, 350.0, 50.0) not in [box for box, _ in page_boxes(page)]
    # the name sits in the e-mail address: that place stays, under the name
    place = next(occ for occ in page.occurrences if occ.words and page.words.words[occ.words[0]].bbox[1] == 25.0)
    assert place.entity.text == SECRET
    assert {v["text"] for v in cloud_values(job)} == {SECRET, "praxis@example.ch", "756.9217.0769.85"}


def test_with_the_safety_rules_on_nothing_is_left_out(server, scan):
    server.job(done(data=switched_rules()))
    job = analyze_in_cloud(scan, SwissDocAIRedact(client(server)), rule_categories=("person", "contact", "identifier"))
    assert {"079 555 01 23", "071 222 33 44"} <= {e.text for e in job.entities}


def test_the_settings_reach_the_cloud_job(server, scan, monkeypatch, tmp_path):
    server.job(done(data=switched_rules()))
    app = session(server, monkeypatch, tmp_path, engine="swissdocai", reader="swissdocai", always_redact_contacts=False)
    assert "079 555 01 23" not in {e.text for e in app.analyze(scan, lambda *args: None).entities}


def test_a_multi_page_tiff_goes_up_as_a_pdf_a_bmp_as_a_png(server, tmp_path):
    frames = [Image.new("RGB", (500, 250), "white"), Image.new("RGB", (500, 250), "gray")]
    tiff = tmp_path / "scan.tif"
    frames[0].save(tiff, save_all=True, append_images=frames[1:])
    data = result()
    data["pages"].append({**data["pages"][0], "page": 2})
    cloud_job(server, tiff, data)
    assert base64.b64decode(server.posted("/v1/redact")[-1]["image_base64"])[:4] == b"%PDF"

    bmp = tmp_path / "scan.bmp"
    frames[0].save(bmp)
    cloud_job(server, bmp)
    assert base64.b64decode(server.posted("/v1/redact")[-1]["image_base64"])[:4] == b"\x89PNG"


# --- the check -------------------------------------------------------------------------------


def verify_answer(ok=True, leaks=()) -> tuple[int, dict]:
    return 200, {"status": "done", "pages": 1, "result": {"ok": ok, "still_readable": list(leaks)}}


def test_the_service_checks_the_redacted_pages_before_the_pdf_is_written(server, scan, tmp_path):
    job = cloud_job(server, scan)
    server.script["POST", "/v1/redact/verify"] = [(200, {"job_id": "v1", "document_id": "d"})]
    server.script["GET", "/v1/redact/v1"] = [(200, {"status": "queued", "position": 1, "wait_seconds": 1}), verify_answer()]
    out = tmp_path / "out.pdf"
    report = render_job(job, out)
    assert report["verification"] == {"ok": True, "stillReadable": []} and out.exists()
    body = server.posted("/v1/redact/verify")[0]
    assert {(v["text"], v["category"]) for v in body["values"]} == {
        ("ursula.kuhn@example.ch", "contact"), (SECRET, "person"), ("756.9217.0769.85", "identifier")}
    assert body["document_id"] == job.document_id
    page = Image.open(io.BytesIO(base64.b64decode(body["pages"][0])))
    assert page.format == "JPEG" and page.size == (500, 250)
    assert page.convert("L").getpixel((100, 110)) < 30  # the bars are in


def test_a_value_kept_readable_is_not_asked_for(server, scan):
    job = cloud_job(server, scan)
    set_entity_redacted(job, job.entities[1], False)
    values = cloud_values(job)
    assert (SECRET, "person") in {(v["text"], v["category"]) for v in values}  # it also sits in the e-mail address
    set_entity_redacted(job, job.entities[0], False)
    assert {v["text"] for v in cloud_values(job)} == {"756.9217.0769.85"}  # the value without a place stays


def test_what_is_still_readable_reaches_the_report_without_the_text(server, scan, tmp_path):
    job = cloud_job(server, scan)
    server.script["POST", "/v1/redact/verify"] = [(200, {"job_id": "v1", "document_id": "d"})]
    server.script["GET", "/v1/redact/v1"] = [verify_answer(False, [{"page": 1, "category": "person", "text": SECRET}])]
    report = render_job(job, tmp_path / "out.pdf")
    assert report["verification"] == {"ok": False, "stillReadable": [{"page": 1, "category": "person"}]}


def test_no_unchecked_pdf_when_the_service_cannot_be_asked(server, scan, tmp_path):
    job = cloud_job(server, scan)
    server.script["POST", "/v1/redact/verify"] = [(503, {"detail": "busy"})]
    out = tmp_path / "out.pdf"
    with pytest.raises(SaasBusy):
        render_job(job, out)
    assert not out.exists()


def test_the_check_keeps_to_the_limits(server, monkeypatch):
    monkeypatch.setattr(saas, "VERIFY_MAX_VALUES", 2)
    monkeypatch.setattr(saas, "REDACT_MAX_PAGES", 2)
    server.script["POST", "/v1/redact/verify"] = [(200, {"job_id": "v1", "document_id": "d"})]
    server.script["GET", "/v1/redact/v1"] = [verify_answer(False, [{"page": 1, "category": "person", "text": "x"}])]
    images = [Image.new("RGB", (20, 20), "white")] * 3
    values = [{"text": f"v{i}", "category": "person"} for i in range(3)]
    report = SwissDocAIRedact(client(server)).verify(images, values, "d")
    sent = server.posted("/v1/redact/verify")
    assert [(len(b["pages"]), len(b["values"])) for b in sent] == [(2, 2), (2, 1), (1, 2), (1, 1)]
    assert sorted({leak["page"] for leak in report["stillReadable"]}) == [1, 3]  # numbered in the document


# --- logs ------------------------------------------------------------------------------------


def test_nothing_of_the_document_reaches_the_log(server, scan, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    job = cloud_job(server, scan)
    server.script["POST", "/v1/redact/verify"] = [(200, {"job_id": "v1", "document_id": "d"})]
    server.script["GET", "/v1/redact/v1"] = [verify_answer(False, [{"page": 1, "category": "person", "text": SECRET}])]
    render_job(job, tmp_path / "out.pdf")
    assert not any(word in caplog.text for word in ("Kuhn", "ursula", "756.9217", "Rechnung"))


def test_a_failed_document_logs_no_file_name(server, scan, tmp_path, monkeypatch, caplog):
    from swissdocai.app import api

    monkeypatch.setenv("SWISSDOCAI_CACHE", str(tmp_path / "cache"))
    caplog.set_level(logging.DEBUG)
    engine = api.Engine(api.Settings())
    monkeypatch.setattr(engine.session, "analyze", lambda *args: (_ for _ in ()).throw(SaasNoCredit("no credit")))
    doc = engine.add([str(scan)])[0]
    engine._analyze(doc)
    assert doc.status == "failed" and doc.summary()["errorCode"] == "credit"
    assert "Kuhn" not in caplog.text


def test_the_waiting_place_reaches_the_window(tmp_path, scan, monkeypatch):
    from swissdocai.app import api

    monkeypatch.setenv("SWISSDOCAI_CACHE", str(tmp_path / "cache"))
    engine = api.Engine(api.Settings())
    doc = engine.add([str(scan)])[0]
    engine._progress(doc)("queued", 4, 95)
    assert doc.summary()["progress"] == {"stage": "queued", "done": 4, "total": 95}
