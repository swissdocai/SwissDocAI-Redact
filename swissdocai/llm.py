"""OpenAI-compatible chat API: how the app talks to its local llama.cpp server
(engine.py) — the PII model and the page reader.

``load_env`` reads a ``.env`` file in the working directory (the SwissDocAI
token, saas.py) without overriding the environment.
"""

import base64
import io
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Page → markdown for a general vision model. Spells out the conventions the
# rest of the pipeline relies on: every word copied verbatim (the word layer is
# aligned to it and PII is searched in it), Document-Intelligence markdown.
PAGE_TO_MARKDOWN = """Transcribe this document page into markdown.

Rules:
- Copy every piece of text exactly as printed or handwritten: same spelling, numbers, punctuation and language. Do not correct, translate, summarize or leave anything out, including small print, form labels, stamps and handwritten entries.
- Transcribe the whole page, from the very first line at the top to the very last line at the bottom. Horizontal lines, section breaks and blank areas are not the end of the page: keep going until nothing is left.
- Follow the reading order; for multi-column pages read each column top to bottom.
- Running headers, footers and page numbers become HTML comments on their own lines: <!-- PageHeader="..." -->, <!-- PageFooter="..." -->, <!-- PageNumber="..." -->.
- Headings use #, ##, ###. Lists use - or 1. Bold/italic only where the page shows it.
- Tables are HTML: <table>, one <tr> per line, <th>/<td> cells, rowspan/colspan where cells span.
- Checkboxes and radio buttons: ☒ when checked or filled, ☐ when empty, followed by their label.
- Signatures, photos and logos without text: leave them out.
- Output only the markdown, without code fences or comments of your own."""


def load_env(path: str | Path = ".env") -> None:
    """Set variables from a KEY=VALUE file without overriding the environment."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def image_part(image, max_side: int = 2000) -> dict:
    """A PIL page image as an OpenAI ``image_url`` content part (PNG data URL)."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


class ChatClient:
    """Minimal OpenAI-compatible chat completions client (non-streaming)."""

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: float = 300.0, retries: int = 4):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat(self, messages: list[dict], response_format: dict | None = None, max_tokens: int = 8192,
             temperature: float = 0.0, reasoning_effort: str | None = "none", **extra) -> str:
        """Content of the first choice. Retries rate limits and server errors.

        Thinking is off by default (``reasoning_effort="none"``): transcription
        and extraction gain nothing from it but latency and tokens. ``None``
        omits the field.
        """
        import httpx

        body = {"model": self.model, "messages": messages, "stream": False, "temperature": temperature,
                "max_completion_tokens": max_tokens, **extra}
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if response_format:
            body["response_format"] = response_format
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body,
                                      timeout=self.timeout)
            except httpx.TransportError as error:
                if attempt == self.retries:
                    raise
                logger.warning("LLM request failed (%s), retrying", error)
            else:
                if response.status_code == 200:
                    choice = response.json()["choices"][0]
                    if choice.get("finish_reason") == "length":
                        logger.warning("LLM output hit max_tokens=%d — truncated", max_tokens)
                    return choice["message"].get("content") or ""
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise RuntimeError(f"LLM API {response.status_code}: {response.text[:300]}")
                logger.warning("LLM API %d, retrying", response.status_code)
            time.sleep(min(60, 2 ** attempt * 2))
        raise AssertionError("unreachable")


class ApiMarkdownModel:
    """Page image → markdown with a vision model behind a chat API.

    Drop-in for ``vlm.MarkdownModel`` (``DocumentAnalyzer(markdown_model=...)``).
    """

    def __init__(self, client: ChatClient, prompt: str = PAGE_TO_MARKDOWN, max_tokens: int = 8192):
        self.client = client
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.model_id = self.client.model

    def __call__(self, image) -> str:
        from .vlm import clean_output

        messages = [{"role": "user", "content": [image_part(image), {"type": "text", "text": self.prompt}]}]
        return clean_output(self.client.chat(messages, max_tokens=self.max_tokens))
