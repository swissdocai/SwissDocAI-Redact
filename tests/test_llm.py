"""The chat client for the local llama.cpp server, and the SwissDocAI API client
(no network: httpx is mocked)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx  # noqa: E402
import pytest  # noqa: E402

from swissdocai import llm  # noqa: E402


class FakeResponse:
    def __init__(self, status: int, content: str = "ok"):
        self.status_code = status
        self.text = content
        self._content = content

    def json(self):
        return {"choices": [{"finish_reason": "stop", "message": {"content": self._content}}]}


@pytest.fixture
def calls(monkeypatch):
    sent = []
    replies = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append({"url": url, "headers": headers, "body": json})
        return replies.pop(0) if replies else FakeResponse(200)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return sent, replies


def client(model: str = "m", api_key: str = "") -> llm.ChatClient:
    return llm.ChatClient(model, base_url="http://llm.test/v1/", api_key=api_key)


def test_requests_are_non_streaming_deterministic_and_without_thinking(calls):
    sent, _ = calls
    assert client("some-model", "test-key").chat([{"role": "user", "content": "hi"}]) == "ok"
    body = sent[0]["body"]
    assert sent[0]["url"] == "http://llm.test/v1/chat/completions"
    assert sent[0]["headers"] == {"Authorization": "Bearer test-key"}
    assert body["stream"] is False and body["temperature"] == 0.0
    assert body["reasoning_effort"] == "none"


def test_reasoning_field_can_be_omitted(calls):
    sent, _ = calls
    client().chat([], reasoning_effort=None)
    assert "reasoning_effort" not in sent[0]["body"]
    assert sent[0]["headers"] == {}  # the local server needs no key


def test_rate_limits_are_retried_but_client_errors_are_not(calls):
    sent, replies = calls
    replies.extend([FakeResponse(429), FakeResponse(503), FakeResponse(200, "done")])
    assert client().chat([]) == "done"
    assert len(sent) == 3
    replies.append(FakeResponse(400, "bad request"))
    with pytest.raises(RuntimeError, match="400"):
        client().chat([])


def test_env_file_fills_missing_variables_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nSWISSDOCAI_TEST_TOKEN='abc'\nKEEP=from-file\n", encoding="utf-8")
    monkeypatch.delenv("SWISSDOCAI_TEST_TOKEN", raising=False)
    monkeypatch.setenv("KEEP", "from-env")
    llm.load_env(env)
    assert llm.os.environ["SWISSDOCAI_TEST_TOKEN"] == "abc"
    assert llm.os.environ["KEEP"] == "from-env"
    monkeypatch.delenv("SWISSDOCAI_TEST_TOKEN")


def test_swissdocai_reader_sends_the_page_and_reads_the_answer(monkeypatch):
    import base64

    import httpx
    from PIL import Image

    from swissdocai.saas import SaasClient, SaasError, SwissDocAIReader

    sent = {}

    class Response:
        status_code = 200

        def json(self):
            return {"text": "```markdown\n# Befund\nHerr Bauer\n```", "pages": 1, "processing_time_ms": 900,
                    "queue_time_ms": 0}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    reader = SwissDocAIReader(SaasClient(token="sd_test", base_url="https://saas.example/"))
    assert reader(Image.new("RGB", (40, 20), "white")) == "# Befund\nHerr Bauer"
    assert sent["url"] == "https://saas.example/v1/ocr" and sent["headers"]["Authorization"] == "Bearer sd_test"
    assert base64.b64decode(sent["body"]["image_base64"])[:3] == b"\xff\xd8\xff"  # a JPEG

    class Denied(Response):
        status_code = 401
        text = "invalid token"

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Denied())
    import pytest

    with pytest.raises(SaasError, match="token"):
        reader(Image.new("RGB", (40, 20), "white"))


def test_swissdocai_pii_sends_the_texts_and_lists_and_reads_the_findings(monkeypatch):
    from swissdocai.policy import Policy
    from swissdocai.saas import SaasClient, SaasError, SwissDocAIPii

    sent = {}

    class Response:
        status_code = 200

        def json(self):
            return {"entities": [{"text": "Andreas Bauer", "category": "person"},
                                 {"text": "14.03.1988", "category": "birth_date"},
                                 {"text": "Herr", "category": "person"},  # implausible: dropped
                                 {"text": "Seeblick AG", "category": "organization"}],  # not asked for
                    "pages": 1, "model": "m", "processing_time_ms": 3900, "queue_time_ms": 0}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    service = SwissDocAIPii(SaasClient(token="sd_test", base_url="https://saas.example/"))
    found = service.detect(["Andreas Bauer, geb. 14.03.1988"], ("person", "birth_date"),
                           Policy(["Namen"], ["Kliniken"], "Arztberichte"))
    assert {(e.text, e.category) for e in found} == {("Andreas Bauer", "person"), ("14.03.1988", "birth_date"),
                                                     ("Andreas", "person"), ("Bauer", "person")}  # as the local step
    assert sent["url"] == "https://saas.example/v1/pii" and sent["headers"]["Authorization"] == "Bearer sd_test"
    assert sent["body"]["pages"] == ["Andreas Bauer, geb. 14.03.1988"]
    assert sent["body"]["categories"] == ["person", "birth_date"]
    assert sent["body"]["policy"] == {"redact": ["Namen"], "keep": ["Kliniken"], "notes": "Arztberichte"}
    assert sent["body"]["document_id"].startswith("swissdocai-redact-")

    class Denied(Response):
        status_code = 401
        text = "invalid token"

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Denied())
    with pytest.raises(SaasError, match="token"):
        service.detect(["x"], ("person",))


def test_swissdocai_pii_keeps_to_the_limits_and_halves_on_413(monkeypatch):
    from swissdocai import saas
    from swissdocai.saas import SaasClient, SwissDocAIPii

    requests = []

    class Response:
        def __init__(self, status, pages):
            self.status_code, self.pages, self.text = status, pages, "too many findings"

        def json(self):
            return {"entities": [{"text": f"Person {p[-1]}", "category": "person"} for p in self.pages]}

    def post(url, json=None, headers=None, timeout=None):
        requests.append((len(json["pages"]), json["document_id"]))
        return Response(413 if len(json["pages"]) > 2 else 200, json["pages"])  # at most 2 pages fit an answer

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(saas, "PII_MAX_PAGES", 4)
    service = SwissDocAIPii(SaasClient(token="sd_test", base_url="https://saas.example/"))
    service.document_id = "doc-1"
    pages = [f"Text Seite {n}" for n in range(1, 7)]  # 6 pages: batches of 4 + 2, the 4 halved on 413
    found = {e.text for e in service.detect(pages, ("person",))}
    assert {f"Person {n}" for n in range(1, 7)} <= found  # every page's value (plus the parts of the names)
    assert [n for n, _ in requests] == [4, 2, 2, 2]
    assert {d for _, d in requests} == {"doc-1"}
    assert [len(b) for b in saas._batches(["x" * 300_000, "y" * 300_000, "z"])] == [1, 2]  # 500,000 characters


def test_one_document_id_for_all_calls_about_a_document():
    from swissdocai.saas import SaasClient, SwissDocAIPii, SwissDocAIReader, new_document_id

    reader = SwissDocAIReader(SaasClient(token="sd_test"))
    service = SwissDocAIPii(SaasClient(token="sd_test"))
    document_id = new_document_id()
    for s in (reader, service):  # as redact.analyze_document does
        s.document_id = document_id
    assert reader.document_id == service.document_id and document_id.startswith("swissdocai-redact-")
