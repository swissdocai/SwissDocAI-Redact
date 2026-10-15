"""The desktop engine's local API, with a fake model (no GPU, no downloads)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from swissdocai.redact import render_job  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SWISSDOCAI_CACHE", str(tmp_path / "cache"))
    from swissdocai.app import api
    from test_app import make_job

    engine = api.Engine(api.Settings())

    class FakeSession:
        settings = engine.settings

        def analyze(self, path, progress):
            progress("words", 0, 1)
            job = make_job(tmp_path)
            if self.settings.redact_qr_codes:
                job.pages[0].qr_codes = [(400.0, 120.0, 480.0, 190.0)]
            return job

        def save(self, job, out, progress):
            return render_job(job, out, verify=False, progress=progress)

        def downloads_needed(self):
            return [("gemma-4-E4B_q4_0-it.gguf", 5.15)]

        def hardware(self):
            return {"gpu": "NVIDIA GeForce RTX 4070", "vramGB": 12.0, "backend": "cuda",
                    "backendName": "NVIDIA CUDA", "chosen": False, "runsOn": "GPU", "ocr": "Dml"}

        def close(self):
            pass

    engine.session = FakeSession()
    return TestClient(api.create_app(engine, "secret")), tmp_path


def wait_for(client, doc_id, status, headers):
    for _ in range(200):
        doc = next(d for d in client.get("/api/documents", headers=headers).json() if d["id"] == doc_id)
        if doc["status"] == status:
            return doc
        time.sleep(0.02)
    raise AssertionError(f"{doc_id} never reached {status}: {doc}")


def test_the_window_flow(client):
    client, tmp_path = client
    h = {"X-Token": "secret"}
    assert client.get("/api/health").status_code == 403  # no token, no answer
    (tmp_path / "letter.pdf").write_bytes(b"%PDF-1.4")
    added = client.post("/api/documents", json={"paths": [str(tmp_path / "letter.pdf"), "C:/nope.exe"]}, headers=h)
    assert [d["name"] for d in added.json()] == ["letter.pdf"]
    doc_id = added.json()[0]["id"]
    client.post("/api/analyze", json={}, headers=h)
    assert wait_for(client, doc_id, "review", h)["pages"] == 1

    page = client.get(f"/api/documents/{doc_id}/pages/0", headers=h).json()
    assert page["words"][3]["state"] == "redacted" and page["boxes"]
    image = client.get(f"/api/documents/{doc_id}/pages/0/image", headers=h)
    assert image.headers["content-type"] == "image/jpeg"
    x0, y0, x1, y1 = page["words"][3]["box"]
    clicked = client.post(f"/api/documents/{doc_id}/pages/0/click", json={"x": (x0 + x1) / 2, "y": (y0 + y1) / 2},
                          headers=h).json()
    assert clicked["changed"] and clicked["page"]["words"][3]["state"] == "kept"
    boxed = client.post(f"/api/documents/{doc_id}/pages/0/boxes", json={"box": [480, 190, 400, 150]}, headers=h)
    assert {"box": [400.0, 150.0, 480.0, 190.0], "category": "manual"} in boxed.json()["boxes"]

    found = client.get(f"/api/documents/{doc_id}/findings", headers=h).json()
    item = next(i for g in found["groups"] for i in g["items"] if i["text"] == "Ursula Kuhn")
    after = client.post(f"/api/documents/{doc_id}/entities/{item['index']}", json={"redacted": False},
                        headers=h).json()
    assert not next(i for g in after["groups"] for i in g["items"] if i["text"] == "Ursula Kuhn")["redacted"]
    assert after["manual"] == 1

    out = tmp_path / "letter.redacted.pdf"
    client.post(f"/api/documents/{doc_id}/save", json={"path": str(out)}, headers=h)
    assert wait_for(client, doc_id, "saved", h)["savedTo"] == str(out) and out.exists()


def test_qr_codes_are_listed_and_can_be_kept_readable(client):
    client, tmp_path = client
    h = {"X-Token": "secret"}
    assert client.get("/api/settings", headers=h).json()["redact_qr_codes"]  # on unless switched off
    (tmp_path / "bill.pdf").write_bytes(b"%PDF-1.4")
    doc_id = client.post("/api/documents", json={"paths": [str(tmp_path / "bill.pdf")]}, headers=h).json()[0]["id"]
    client.post("/api/analyze", json={}, headers=h)
    wait_for(client, doc_id, "review", h)

    code = {"page": 0, "index": 0, "box": [400.0, 120.0, 480.0, 190.0], "redacted": True}
    assert client.get(f"/api/documents/{doc_id}/findings", headers=h).json()["qrCodes"] == [code]
    page = client.get(f"/api/documents/{doc_id}/pages/0", headers=h).json()
    assert {"box": code["box"], "category": "qr_code"} in page["boxes"] and page["hidden"] == []
    kept = client.post(f"/api/documents/{doc_id}/pages/0/qr-codes/0", json={"redacted": False}, headers=h).json()
    assert kept["qrCodes"] == [dict(code, redacted=False)]
    page = client.get(f"/api/documents/{doc_id}/pages/0", headers=h).json()
    assert page["hidden"] == [{"box": code["box"], "category": "qr_code"}]
    assert all(b["category"] != "qr_code" for b in page["boxes"])
    assert client.post(f"/api/documents/{doc_id}/pages/0/qr-codes/1", json={"redacted": False},
                       headers=h).status_code == 404


def test_the_first_start_shows_the_downloads_the_hardware_and_the_free_space(client):
    client, _ = client
    setup = client.get("/api/setup", headers={"X-Token": "secret"}).json()
    assert setup["needed"] == [{"name": "gemma-4-E4B_q4_0-it.gguf", "gb": 5.15}] and not setup["running"]
    assert setup["hardware"]["backend"] == "cuda" and setup["freeGB"] > 0


def test_settings_keep_the_users_lists(client):
    client, _ = client
    h = {"X-Token": "secret"}
    settings = client.get("/api/settings", headers=h).json()
    assert settings["language"] == "de" and settings["policyIsDefault"]
    assert "Geburtsdaten" in settings["policy"]["redact"]
    policy = dict(settings["policy"], redact=settings["policy"]["redact"] + ["Fahrzeugkennzeichen"])
    saved = client.put("/api/settings", json={"policy": policy, "language": "fr"}, headers=h).json()
    assert "Fahrzeugkennzeichen" in saved["policy"]["redact"] and not saved["policyIsDefault"]
    assert saved["language"] == "fr"
    reset = client.put("/api/settings", json={"policy": None}, headers=h).json()
    assert reset["policyIsDefault"] and "Dates de naissance" in reset["policy"]["redact"]  # French now
    assert client.get("/api/policy/default?language=it", headers=h).json()["redact"][1] == "Date di nascita"


def test_the_swissdocai_token_is_stored_apart_and_never_returned(client, monkeypatch):
    client, tmp_path = client
    h = {"X-Token": "secret"}
    import swissdocai.saas as saas

    monkeypatch.delenv("SWISSDOCAI_API_KEY", raising=False)
    monkeypatch.setattr(saas, "api_key", lambda: None)  # no token from .env in this test
    assert client.get("/api/settings", headers=h).json()["swissdocaiToken"] is None
    saved = client.put("/api/settings", json={"reader": "swissdocai", "swissdocai_token": "sd_secret_9876"},
                       headers=h).json()
    assert saved["reader"] == "swissdocai" and saved["swissdocaiToken"] == "…9876"
    stored = (tmp_path / "cache" / "settings.json").read_text(encoding="utf-8")
    assert "sd_secret" not in stored and "sd_secret" not in str(saved)
    from swissdocai.app.secret import load_token

    assert load_token() == "sd_secret_9876"
    assert client.put("/api/settings", json={"swissdocai_token": ""}, headers=h).json()["swissdocaiToken"] is None
