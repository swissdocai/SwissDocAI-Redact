"""swissdocai — automatic redaction of personal data in Swiss documents.

Every page is read twice: a word layer with exact boxes (the PDF text layer
or OCR) and a page reader's transcription. A language model finds the
personal data in the transcription, and each finding is blacked out on the
boxes of its words. The desktop app (app/, desktop/) and ``swissdocai
redact`` (cli.py) run it on this machine; the SwissDocAI services (OCR, PII)
are the cloud option.

Modules:
- ``redact``          find, locate, review and black out personal data (the product)
- ``pii``             the PII step: prompt, rules (AHV, IBAN, UID, ...), plausibility
- ``policy``          the editable lists of what to black out and what to keep
- ``engine``          local models: llama.cpp, model downloads, hardware
- ``llm``             chat client for the local llama.cpp server, API page reader
- ``saas``            the SwissDocAI API: OCR as page reader, PII in place of the local model
- ``words``           word layer (PDF text layer, OnnxTR or docTR OCR), pixel boxes
- ``selection_marks`` checkboxes (PDF form fields/drawings, image detection)
- ``markdown_doc``    markdown → blocks / table cells / word tokens with offsets
- ``align``           markdown tokens ↔ page words
- ``pipeline``        pages of a PDF or image; DocumentAnalyzer: file → analyze result
- ``result``          analyze result JSON (content, pages, paragraphs, tables)
- ``vlm``             page image → markdown with a transformers model (PyTorch, optional)
- ``app``             the desktop app's engine: API, session, token storage
"""

__version__ = "0.4.0"

from .pipeline import DocumentAnalyzer  # noqa: E402


def __getattr__(name):
    # MarkdownModel pulls in torch/transformers: import it only when used.
    if name == "MarkdownModel":
        from .vlm import MarkdownModel

        return MarkdownModel
    raise AttributeError(name)


__all__ = ["DocumentAnalyzer", "MarkdownModel", "__version__"]
