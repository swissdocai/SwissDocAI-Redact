"""SwissDocAI cloud (ocr.swissdocai.ch): the SaaS's OCR as the page reader, and its PII
step in place of the local model.

    POST {base}/v1/ocr   Authorization: Bearer sd_…
    {"image_base64": "…", "document_id": "…"}  ->  {"text", "pages", "processing_time_ms", "queue_time_ms"}

    POST {base}/v1/pii   Authorization: Bearer sd_…
    {"pages": ["…", …], "categories": […], "policy": {"redact", "keep", "notes"}, "document_id": "…"}
        ->  {"entities": [{"text", "category"}, …], "pages", "model", "processing_time_ms", "queue_time_ms"}
    at most 200 pages and 500,000 characters per request; 413 when the findings do not
    fit one answer: then fewer pages per request.

Every page sent costs a page of credit, at each endpoint: with both in the cloud a
document page costs two. All calls for one document carry the same document_id.

The OCR service takes PDF, PNG or JPEG (up to 50 MB) and transcribes every
page to Markdown; several pages come back joined with "--- Seite N ---". The PII
service takes what the local PII step would read (pii.detect_with_llm: one text
per page, the categories, the user's lists) and returns what it finds. Both run
in Switzerland and are billed to the owner of the token: every call costs money.

    reader = SwissDocAIReader()          # token from SWISSDOCAI_API_KEY (.env)
    analyze_document(path, SwissDocAIPii(), reader=reader, ...)
"""

import base64
import io
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)

SAAS_URL = "https://ocr.swissdocai.ch"


class SaasError(RuntimeError):
    pass


class SaasTooLarge(SaasError):
    """413: the request (or the findings for it) is too large: send fewer pages."""


# the PII endpoint's limits per request
PII_MAX_PAGES = 200
PII_MAX_CHARS = 500_000


def api_key() -> str | None:
    from .llm import load_env

    load_env()
    return os.environ.get("SWISSDOCAI_API_KEY") or None


class SaasClient:
    def __init__(self, token: str | None = None, base_url: str | None = None, timeout: float = 300, retries: int = 2):
        self.token = token or api_key()
        if not self.token:
            raise SaasError("No SwissDocAI API token: set SWISSDOCAI_API_KEY or enter it in the settings")
        self.base_url = (base_url or os.environ.get("SWISSDOCAI_API_URL") or SAAS_URL).rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def ocr(self, data: bytes, document_id: str | None = None) -> dict:
        """The service's answer for one document (PDF, PNG or JPEG bytes)."""
        return self._post("/v1/ocr", {"image_base64": base64.b64encode(data).decode("ascii"),
                                      "document_id": document_id or _document_id()}, "OCR")

    def pii(self, pages: list[str], categories, policy=None, document_id: str | None = None) -> dict:
        """The personal data the service finds in ``pages`` (one text per page), guided by the
        user's lists (policy.Policy)."""
        return self._post("/v1/pii", {"pages": list(pages), "categories": list(categories),
                                      "policy": policy.to_dict() if policy else None,
                                      "document_id": document_id or _document_id()}, "PII")

    def _post(self, path: str, body: dict, what: str) -> dict:
        """POST with the token; retries when the service is busy or failing (429, 5xx)."""
        import httpx

        headers = {"Authorization": f"Bearer {self.token}"}
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(f"{self.base_url}{path}", json=body, headers=headers, timeout=self.timeout)
            except httpx.TransportError as error:
                if attempt == self.retries:
                    raise SaasError(f"SwissDocAI not reachable: {error}") from error
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (401, 403):
                    raise SaasError("SwissDocAI rejected the API token")
                if response.status_code == 413:
                    raise SaasTooLarge(f"SwissDocAI {what}: too large ({response.text[:200]})")
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise SaasError(f"SwissDocAI {what} failed: {response.status_code} {response.text[:200]}")
            wait = 2 ** attempt * 3
            logger.warning("SwissDocAI %s busy or failing, retrying in %ds", what, wait)
            time.sleep(wait)
        raise SaasError(f"SwissDocAI {what} failed")


def _document_id() -> str:
    return f"swissdocai-redact-{uuid.uuid4().hex[:12]}"


def new_document_id() -> str:
    """One id for all the calls about one document (OCR of every page, PII)."""
    return _document_id()


class SwissDocAIReader:
    """Page image -> Markdown through the SaaS OCR (a drop-in for the GLM-OCR reader)."""

    model_id = "swissdocai.ch/v1/ocr"

    def __init__(self, client: SaasClient | None = None, quality: int = 90):
        self.client = client or SaasClient()
        self.quality = quality
        self.document_id: str | None = None  # set per document by redact.analyze_document

    def __call__(self, image) -> str:
        from .vlm import clean_output

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=self.quality)
        answer = self.client.ocr(buffer.getvalue(), self.document_id)
        logger.info("SwissDocAI OCR: %d chars in %d ms (queued %d ms)", len(answer.get("text", "")),
                    answer.get("processing_time_ms", 0), answer.get("queue_time_ms", 0))
        return clean_output(answer.get("text", ""))


class SwissDocAIPii:
    """The PII step through the SaaS, in place of the local model (redact.analyze_document):
    the page texts and the user's lists go up, the values found come back. The rules (AHV,
    IBAN … with checksums), locating, review and the black bars stay on this machine."""

    model = "swissdocai.ch/v1/pii"

    def __init__(self, client: SaasClient | None = None):
        self.client = client or SaasClient()
        self.document_id: str | None = None  # set per document by redact.analyze_document

    def detect(self, pages: list[str], categories, policy=None) -> list:
        """The values the service finds in ``pages``, in requests within its limits, completed
        as the local PII step does: the parts of every name ("Herr Bauer" further down) and
        the identifiers inside links. Whatever the service already did, the result is the same."""
        from .pii import Entity, drop_fragments, identifier_parts, name_parts, plausible

        allowed = set(categories) | ({"other"} if policy is not None else set())
        found: dict[tuple[str, str], Entity] = {}
        for batch in _batches(pages):
            for item in self._ask(batch, categories, policy):
                entity = Entity(str(item.get("text", "")).strip(), str(item.get("category", "")).strip(), "llm")
                if entity.text and entity.category in allowed and plausible(entity):  # whatever the service sends
                    found.setdefault((entity.text.casefold(), entity.category), entity)
        return identifier_parts(name_parts(drop_fragments(list(found.values()))))

    def _ask(self, pages: list[str], categories, policy) -> list[dict]:
        """The service's findings for ``pages``; on 413 (more findings than one answer holds)
        each half is asked on its own."""
        try:
            answer = self.client.pii(pages, categories, policy, self.document_id)
        except SaasTooLarge:
            if len(pages) == 1:
                raise SaasError("SwissDocAI PII: a single page has more personal data than one answer holds") from None
            half = len(pages) // 2
            logger.info("SwissDocAI PII: too much for one answer, asking for %d and %d pages", half, len(pages) - half)
            return self._ask(pages[:half], categories, policy) + self._ask(pages[half:], categories, policy)
        logger.info("SwissDocAI PII: %d pages, %d values (%s) in %d ms (queued %d ms)", len(pages),
                    len(answer.get("entities", [])), answer.get("model", "?"), answer.get("processing_time_ms", 0),
                    answer.get("queue_time_ms", 0))
        return answer.get("entities", [])


def _batches(pages: list[str]) -> list[list[str]]:
    """Consecutive pages in requests of at most PII_MAX_PAGES pages and PII_MAX_CHARS characters."""
    batches, current, chars = [], [], 0
    for page in pages:
        if current and (len(current) == PII_MAX_PAGES or chars + len(page) > PII_MAX_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(page)
        chars += len(page)
    if current:
        batches.append(current)
    return batches
