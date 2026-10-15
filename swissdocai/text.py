"""Token normalization shared by markdown parsing and word alignment.

Both sides of an alignment (markdown tokens and detected page words) pass
through ``normalize_token`` so that typographic variants compare equal:
curly vs. straight quotes, en/em dashes, ligatures, superscripts, case,
and the German sharp s (``casefold`` maps ß → ss, matching Swiss spelling).
"""

import html
import re
import unicodedata

_TRANSLATE = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‹": '"', "›": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
        "−": "-",
        "­": None,  # soft hyphen
        "​": None,  # zero-width space
        "﻿": None,
    }
)

_REPEATED_CHAR = re.compile(r"(.)\1+")


def _is_punct_or_symbol(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def normalize_token(text: str) -> str:
    """Canonical form of one word for comparison.

    Words with letters or digits lose leading/trailing punctuation
    ("(Zürich)," → "zürich"); pure-punctuation tokens such as underscore
    form fields keep their characters with repeats collapsed, so a
    48-underscore field still matches a 34-underscore one.
    """
    text = unicodedata.normalize("NFKC", html.unescape(text)).translate(_TRANSLATE).casefold()
    text = text.strip()
    if not text:
        return ""
    if any(ch.isalnum() for ch in text):
        start, end = 0, len(text)
        while start < end and _is_punct_or_symbol(text[start]):
            start += 1
        while end > start and _is_punct_or_symbol(text[end - 1]):
            end -= 1
        return text[start:end]
    return _REPEATED_CHAR.sub(r"\1", text)

