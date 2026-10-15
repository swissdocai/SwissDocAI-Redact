"""The desktop app's engine: a local HTTP API the Tauri window talks to.

    python -m swissdocai.app.api --port 0 --token <secret> [--parent-pid <app>]

Listens on 127.0.0.1 only and answers only requests carrying the token (the
window gets both from the Tauri shell that started this process). Prints
"SWISSDOCAI_ENGINE_READY <port>" once it accepts requests. With --parent-pid it
ends together with the app, also when the app is killed.

Documents are analyzed one after another by one worker thread (one GPU); the
window polls their status and edits the redaction through small calls (click,
box, value on/off) — the same review logic as redact.toggle_at.
"""

import argparse
import io
import json
import logging
import queue
import secrets
import socket
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..engine import cache_dir
from ..policy import Policy, default_policy
from ..redact import (
    RedactionJob, area_boxes, kept_readable, page_boxes, set_entity_redacted, set_qr_code_redacted, toggle_at,
)
from .session import Session, Settings

logger = logging.getLogger(__name__)

FILE_TYPES = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
SETTINGS_FILE = "settings.json"


@dataclass
class Document:
    id: str
    path: Path
    status: str = "waiting"  # waiting | queued | analyzing | review | saving | saved | failed
    error: str = ""
    stage: str = ""
    done: int = 0
    total: int = 0
    job: RedactionJob | None = None
    report: dict | None = None
    saved_to: str = ""

    def summary(self) -> dict:
        return {"id": self.id, "name": self.path.name, "path": str(self.path), "status": self.status,
                "error": self.error, "progress": {"stage": self.stage, "done": self.done, "total": self.total},
                "pages": len(self.job.pages) if self.job else 0, "savedTo": self.saved_to,
                "verified": (self.report or {}).get("verification", {}).get("ok")}


@dataclass
class Setup:
    running: bool = False
    what: str = ""
    done: float = 0
    total: float = 0
    error: str = ""


@dataclass
class Engine:
    """Everything the API serves: settings, documents, the worker."""

    settings: Settings
    documents: dict[str, Document] = field(default_factory=dict)
    setup: Setup = field(default_factory=Setup)

    def __post_init__(self):
        self.session = Session(self.settings)
        self.lock = threading.RLock()
        self.work: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    # --- settings ------------------------------------------------------------------------

    @staticmethod
    def load_settings() -> Settings:
        path = cache_dir() / SETTINGS_FILE
        try:
            return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return Settings()

    def save_settings(self, data: dict) -> Settings:
        if "swissdocai_token" in data:  # kept encrypted, never in the settings file
            from .secret import save_token

            save_token(str(data.pop("swissdocai_token") or "").strip())
        with self.lock:
            settings = Settings.from_dict({**self.settings.to_dict(), **data})
            if "policy" in data and data["policy"] is not None:
                settings.policy = Policy.from_dict(data["policy"]).to_dict()
            path = cache_dir() / SETTINGS_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(settings.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            self.settings = settings
            self.session.settings = settings  # takes effect with the next document
            return settings

    def settings_dict(self) -> dict:
        data = self.settings.to_dict()
        data["policy"] = self.settings.effective_policy().to_dict()
        data["policyIsDefault"] = self.settings.policy is None
        from .secret import from_environment, token_hint

        data["swissdocaiToken"] = token_hint()
        data["swissdocaiTokenFromEnvironment"] = from_environment()
        return data

    # --- work ----------------------------------------------------------------------------

    def _worker(self):
        while True:
            task = self.work.get()
            try:
                task()
            except Exception:  # noqa: BLE001 — each task reports its own failure
                logger.exception("task failed")

    def add(self, paths: list[str]) -> list[Document]:
        added = []
        with self.lock:
            known = {d.path for d in self.documents.values()}
            for p in paths:
                path = Path(p)
                if path.suffix.lower() in FILE_TYPES and path.exists() and path not in known:
                    doc = Document(uuid.uuid4().hex[:12], path)
                    self.documents[doc.id] = doc
                    added.append(doc)
        return added

    def analyze(self, ids: list[str] | None = None) -> None:
        with self.lock:
            docs = [d for d in self.documents.values() if (ids is None or d.id in ids)
                    and d.status in ("waiting", "failed")]
            for doc in docs:
                doc.status, doc.error = "queued", ""
                self.work.put(lambda doc=doc: self._analyze(doc))

    def _progress(self, doc: Document):
        def report(stage, done, total):
            doc.stage, doc.done, doc.total = stage, int(done), int(total)
        return report

    def _analyze(self, doc: Document) -> None:
        doc.status = "analyzing"
        try:
            doc.job = self.session.analyze(doc.path, self._progress(doc))
            doc.status, doc.stage = "review", ""
        except Exception as error:  # noqa: BLE001 — shown in the window
            logger.exception("analysis of %s failed", doc.path)
            doc.status, doc.error = "failed", f"{type(error).__name__}: {error}"

    def save(self, doc: Document, out: str) -> None:
        doc.status = "saving"

        def run():
            try:
                doc.report = self.session.save(doc.job, out, self._progress(doc))
                doc.status, doc.saved_to, doc.stage = "saved", out, ""
            except Exception as error:  # noqa: BLE001
                logger.exception("saving %s failed", doc.path)
                doc.status, doc.error = "review", f"{type(error).__name__}: {error}"
        self.work.put(run)

    def start_setup(self) -> None:
        if self.setup.running:
            return
        self.setup = Setup(running=True)

        def progress(what, done, total):
            self.setup.what, self.setup.done, self.setup.total = what, done, total

        def run():
            try:
                self.session.download_all(progress)
            except Exception as error:  # noqa: BLE001
                logger.exception("download failed")
                self.setup.error = f"{type(error).__name__}: {error}"
            finally:
                self.setup.running = False
        threading.Thread(target=run, daemon=True).start()

    def close(self) -> None:
        self.session.close()


# --- what the window sees -------------------------------------------------------------------


def _box(words, indices):
    boxes = [words[j].bbox for j in indices]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def page_view(job: RedactionJob, n: int) -> dict:
    page = job.pages[n]
    words = page.words.words
    state = {}
    for j, marks in page.redact.items():
        state[j] = "kept" if j in page.removed else "manual" if ("manual", "manual") in marks else "redacted"
    return {
        "width": page.source.image.width, "height": page.source.image.height,
        "words": [{"box": list(w.bbox), "text": w.text, "state": state.get(j)} for j, w in enumerate(words)],
        "boxes": [{"box": list(box), "category": category} for box, category in page_boxes(page)],
        "hidden": [{"box": list(box), "category": category} for box, category in area_boxes(page)
                   if tuple(box) in page.hidden_boxes],  # signatures and QR codes switched off
        "review": [{"box": _box(words, occ.words), "text": occ.entity.text} for occ in page.occurrences
                   if occ.missing and not set(occ.words) <= page.removed],
    }


def findings(job: RedactionJob) -> dict:
    groups: dict[str, list] = {}
    for index, entity in sorted(enumerate(job.entities), key=lambda x: (x[1].category, x[1].text.casefold())):
        places = [(i, occ) for i, page in enumerate(job.pages) for occ in page.occurrences if occ.entity is entity]
        if not places:
            continue
        first_page, first = places[0]
        groups.setdefault(entity.category, []).append({
            "index": index, "text": entity.text, "places": len(places), "source": entity.source,
            "redacted": not kept_readable(job, entity),
            "page": first_page, "box": _box(job.pages[first_page].words.words, first.words)})
    review = [{"text": occ.entity.text, "page": i, "box": _box(page.words.words, occ.words)}
              for i, page in enumerate(job.pages) for occ in page.occurrences
              if occ.missing and not set(occ.words) <= page.removed]
    manual = sum(1 for page in job.pages for marks in page.redact.values() if ("manual", "manual") in marks) \
        + sum(len(page.manual_boxes) for page in job.pages)
    qr_codes = [{"page": i, "index": k, "box": list(box), "redacted": tuple(box) not in page.hidden_boxes}
                for i, page in enumerate(job.pages) for k, box in enumerate(page.qr_codes)]
    return {"groups": [{"category": c, "items": items} for c, items in groups.items()], "review": review,
            "notFound": [e.text for e in job.not_found], "manual": manual, "qrCodes": qr_codes}


def create_app(engine: Engine, token: str, origins: list[str] | None = None):
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="swissdocai desktop engine", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.method != "OPTIONS" and not secrets.compare_digest(
                request.headers.get("x-token", "") or request.query_params.get("token", ""), token):
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=origins or ["tauri://localhost", "http://tauri.localhost",
                                                                 "https://tauri.localhost", "http://localhost:1420"],
                       allow_methods=["*"], allow_headers=["*"])

    def doc_or_404(doc_id: str) -> Document:
        doc = engine.documents.get(doc_id)
        if doc is None:
            raise HTTPException(404, "no such document")
        return doc

    def job_or_409(doc: Document) -> RedactionJob:
        if doc.job is None:
            raise HTTPException(409, "not analyzed yet")
        return doc.job

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/settings")
    def get_settings():
        return engine.settings_dict()

    @app.put("/api/settings")
    def put_settings(data: dict = Body(...)):
        engine.save_settings(data)
        return engine.settings_dict()

    @app.get("/api/policy/default")
    def get_default_policy(language: str = "de"):
        return default_policy(language).to_dict()

    @app.get("/api/setup")
    def get_setup():
        s = engine.setup
        return {"needed": [{"name": n, "gb": gb} for n, gb in engine.session.downloads_needed()],
                "running": s.running, "what": s.what, "done": s.done, "total": s.total, "error": s.error,
                "hardware": engine.session.hardware(), "freeGB": round(_free_gb(cache_dir()), 1)}

    @app.post("/api/setup")
    def post_setup():
        engine.start_setup()
        return get_setup()

    @app.get("/api/documents")
    def list_documents():
        return [d.summary() for d in engine.documents.values()]

    @app.post("/api/documents")
    def add_documents(data: dict = Body(...)):
        return [d.summary() for d in engine.add(list(data.get("paths", [])))]

    @app.delete("/api/documents/{doc_id}")
    def remove_document(doc_id: str):
        doc = doc_or_404(doc_id)
        if doc.status in ("queued", "analyzing", "saving"):
            raise HTTPException(409, "busy")
        engine.documents.pop(doc_id, None)
        return {"ok": True}

    @app.post("/api/analyze")
    def analyze(data: dict = Body(default={})):
        engine.analyze(data.get("ids"))
        return list_documents()

    @app.get("/api/documents/{doc_id}/findings")
    def get_findings(doc_id: str):
        return findings(job_or_409(doc_or_404(doc_id)))

    @app.get("/api/documents/{doc_id}/pages/{n}")
    def get_page(doc_id: str, n: int):
        job = job_or_409(doc_or_404(doc_id))
        if not 0 <= n < len(job.pages):
            raise HTTPException(404, "no such page")
        return page_view(job, n)

    @app.get("/api/documents/{doc_id}/pages/{n}/image")
    def get_page_image(doc_id: str, n: int):
        job = job_or_409(doc_or_404(doc_id))
        if not 0 <= n < len(job.pages):
            raise HTTPException(404, "no such page")
        buffer = io.BytesIO()
        job.pages[n].source.image.convert("RGB").save(buffer, "JPEG", quality=88)
        return Response(buffer.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/documents/{doc_id}/pages/{n}/click")
    def click(doc_id: str, n: int, data: dict = Body(...)):
        job = job_or_409(doc_or_404(doc_id))
        changed = toggle_at(job.pages[n], float(data["x"]), float(data["y"]))
        return {"changed": changed, "page": page_view(job, n)}

    @app.post("/api/documents/{doc_id}/pages/{n}/boxes")
    def add_box(doc_id: str, n: int, data: dict = Body(...)):
        job = job_or_409(doc_or_404(doc_id))
        x0, y0, x1, y1 = (float(v) for v in data["box"])
        job.pages[n].manual_boxes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        return page_view(job, n)

    @app.post("/api/documents/{doc_id}/entities/{index}")
    def set_entity(doc_id: str, index: int, data: dict = Body(...)):
        job = job_or_409(doc_or_404(doc_id))
        if not 0 <= index < len(job.entities):
            raise HTTPException(404, "no such value")
        set_entity_redacted(job, job.entities[index], bool(data["redacted"]))
        return findings(job)

    @app.post("/api/documents/{doc_id}/pages/{n}/qr-codes/{index}")
    def set_qr_code(doc_id: str, n: int, index: int, data: dict = Body(...)):
        job = job_or_409(doc_or_404(doc_id))
        if not 0 <= n < len(job.pages) or not 0 <= index < len(job.pages[n].qr_codes):
            raise HTTPException(404, "no such QR code")
        set_qr_code_redacted(job.pages[n], index, bool(data["redacted"]))
        return findings(job)

    @app.post("/api/documents/{doc_id}/save")
    def save(doc_id: str, data: dict = Body(...)):
        doc = doc_or_404(doc_id)
        job_or_409(doc)
        engine.save(doc, str(data["path"]))
        return doc.summary()

    @app.get("/api/documents/{doc_id}/report")
    def report(doc_id: str):
        return doc_or_404(doc_id).report or {}

    @app.post("/api/shutdown")
    def shutdown():
        """The window is closing: stop the model servers, then this process."""
        import os

        engine.close()
        threading.Timer(0.2, lambda: os._exit(0)).start()
        return {"ok": True}

    return app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _free_gb(path: Path) -> float:
    """Free disk space where the models go (the folder or its nearest existing parent)."""
    import shutil

    while not path.exists() and path.parent != path:
        path = path.parent
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


def _exit_with(engine: Engine, parent: int) -> None:
    """End with the app that started this engine, also when the app is killed or crashes: stop the
    model servers, then exit. (Not by watching a stdin pipe: on Windows a pending read on it
    stalls every DLL that loads meanwhile, NumPy's included.)"""
    import os
    import sys
    import time

    def wait():
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            handle = kernel32.OpenProcess(0x00100000, False, parent)  # SYNCHRONIZE
            if not handle:
                logger.warning("cannot watch the app (process %d): %s", parent, ctypes.get_last_error())
                return
            kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)
        else:
            while os.getppid() == parent:  # an orphan is adopted by another process
                time.sleep(1)
        logger.info("the app has gone: stopping")
        try:
            engine.close()
        finally:
            os._exit(0)
    threading.Thread(target=wait, daemon=True).start()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="swissdocai desktop engine (local API for the app window)")
    parser.add_argument("--port", type=int, default=0, help="0: any free port")
    parser.add_argument("--token", default=None, help="Required in the X-Token header (default: SWISSDOCAI_TOKEN)")
    parser.add_argument("--origin", action="append", help="Allowed browser origin (CORS), repeatable")
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="The app's process: stop (with the model servers) when it ends")
    args = parser.parse_args()
    import os

    token = args.token or os.environ.get("SWISSDOCAI_TOKEN") or secrets.token_urlsafe(24)
    log_dir = cache_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(log_dir / "engine.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    engine = Engine(Engine.load_settings())
    if args.parent_pid:
        _exit_with(engine, args.parent_pid)
    port = args.port or _free_port()
    config = uvicorn.Config(create_app(engine, token, args.origin), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    def announce():
        import time

        while not server.started:
            time.sleep(0.05)
        print(f"SWISSDOCAI_ENGINE_READY {port}", flush=True)
    threading.Thread(target=announce, daemon=True).start()
    try:
        server.run()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
