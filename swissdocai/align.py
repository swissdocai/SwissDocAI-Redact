"""Align markdown word tokens to page words (OCR output or PDF text layer).

The markdown (from a page reader or a ground-truth file) and the word layer are
two independent readings of the same page. Aligning them gives every page
word a character span in the markdown — the link Document Intelligence
calls ``span`` — and tells us which markdown text has no visual support.

Three passes, strict to lenient:

1. **Anchors** — difflib's longest matching blocks over normalized tokens
   (monotone, robust to insertions and deletions on either side).
2. **Gap filling** — between consecutive anchors a small DP aligns what is
   left with fuzzy similarity and 1:2 / 2:1 merges. This absorbs OCR
   misreads and hyphenated line breaks ("Doku-" "ment" ↔ "Dokument").
3. **Reorder recovery** — runs of >= 3, then >= 2 tokens still unmatched
   on both sides are matched out of order, then single tokens unique among
   the leftovers. This catches sidebars, columns, headers and page numbers
   that the word layer and the markdown read in a different order.
"""

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from rapidfuzz.distance import Levenshtein

# Largest gap (md tokens x words) the DP handles; larger gaps are left to
# reorder recovery, which is what they usually need anyway.
_MAX_GAP_CELLS = 40_000
_MIN_RECOVERY_RUN = 3
_MERGE_THRESHOLD = 0.85


@dataclass(frozen=True)
class Match:
    """md tokens [md_start, md_end) correspond to words [word_start, word_end).

    At most one side spans two items (2:1 or 1:2 merges); exact anchors
    are always 1:1.
    """

    md_start: int
    md_end: int
    word_start: int
    word_end: int
    score: float


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    longest = max(len(a), len(b))
    if longest <= 3:
        return 0.0  # short words must match exactly ("di" vs "da")
    sim = Levenshtein.normalized_similarity(a, b)
    threshold = 0.75 if longest <= 5 else 0.6
    return sim if sim >= threshold else 0.0


def _align_gap(md: list[str], words: list[str], i1: int, i2: int, j1: int, j2: int) -> list[Match]:
    """Monotone fuzzy alignment of md[i1:i2] against words[j1:j2]."""
    n, m = i2 - i1, j2 - j1
    if n == 0 or m == 0 or n * m > _MAX_GAP_CELLS:
        return []

    # best[i][j]: best score aligning md[i1:i1+i] with words[j1:j1+j]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0:
                candidates.append((best[i - 1][j], (i - 1, j, 0.0)))
            if j > 0:
                candidates.append((best[i][j - 1], (i, j - 1, 0.0)))
            if i > 0 and j > 0:
                s = _similarity(md[i1 + i - 1], words[j1 + j - 1])
                if s:
                    candidates.append((best[i - 1][j - 1] + s, (i - 1, j - 1, s)))
            if i > 0 and j > 1:  # one md token = two words (hyphenation, split)
                s = Levenshtein.normalized_similarity(
                    md[i1 + i - 1], words[j1 + j - 2] + words[j1 + j - 1]
                )
                if s >= _MERGE_THRESHOLD:
                    candidates.append((best[i - 1][j - 2] + 1.5 * s, (i - 1, j - 2, s)))
            if i > 1 and j > 0:  # two md tokens = one word (merged by OCR)
                s = Levenshtein.normalized_similarity(
                    md[i1 + i - 2] + md[i1 + i - 1], words[j1 + j - 1]
                )
                if s >= _MERGE_THRESHOLD:
                    candidates.append((best[i - 2][j - 1] + 1.5 * s, (i - 2, j - 1, s)))
            score, pointer = max(candidates, key=lambda c: c[0])
            best[i][j] = score
            back[i][j] = pointer

    matches = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj, s = back[i][j]
        if s > 0:
            matches.append(Match(i1 + pi, i1 + i, j1 + pj, j1 + j, s))
        i, j = pi, pj
    matches.reverse()
    return matches


def align_tokens(md: list[str], words: list[str]) -> list[Match]:
    """Align two sequences of normalized tokens; empty strings never match.

    Returns matches sorted by markdown position. Every md token and every
    word appears in at most one match.
    """
    # Work on compacted sequences without empty tokens, map back at the end.
    md_idx = [i for i, t in enumerate(md) if t]
    w_idx = [j for j, t in enumerate(words) if t]
    a = [md[i] for i in md_idx]
    b = [words[j] for j in w_idx]

    matches: list[Match] = []
    prev_i = prev_j = 0
    for i, j, size in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        if i > prev_i and j > prev_j:
            matches.extend(_align_gap(a, b, prev_i, i, prev_j, j))
        matches.extend(Match(i + k, i + k + 1, j + k, j + k + 1, 1.0) for k in range(size))
        prev_i, prev_j = i + size, j + size

    matches.extend(_recover_reordered(a, b, matches))

    # Map compacted indices back to the caller's token indices.
    result = [
        Match(
            md_idx[m.md_start], md_idx[m.md_end - 1] + 1,
            w_idx[m.word_start], w_idx[m.word_end - 1] + 1,
            m.score,
        )
        for m in matches
    ]
    result.sort(key=lambda m: (m.md_start, m.word_start))
    return result


def _recover_reordered(a: list[str], b: list[str], matches: list[Match]) -> list[Match]:
    """Match leftovers out of order: runs of 3+, then 2+ tokens, then
    single tokens that occur exactly once among the leftovers on both sides
    (a one-word page header the PDF stream holds at its very end)."""
    used_a = {k for m in matches for k in range(m.md_start, m.md_end)}
    used_b = {k for m in matches for k in range(m.word_start, m.word_end)}
    recovered: list[Match] = []

    def take(ia: int, jb: int) -> None:
        recovered.append(Match(ia, ia + 1, jb, jb + 1, 1.0))
        used_a.add(ia)
        used_b.add(jb)

    for min_run in (_MIN_RECOVERY_RUN, 2):
        for _ in range(8):  # each round can unlock runs that crossed a previous one
            free_a = [k for k in range(len(a)) if k not in used_a]
            free_b = [k for k in range(len(b)) if k not in used_b]
            if not free_a or not free_b:
                break
            found = False
            sm = SequenceMatcher(None, [a[k] for k in free_a], [b[k] for k in free_b], autojunk=False)
            for i, j, size in sm.get_matching_blocks():
                if size < min_run:
                    continue
                found = True
                for k in range(size):
                    take(free_a[i + k], free_b[j + k])
            if not found:
                break

    free_a = [k for k in range(len(a)) if k not in used_a]
    free_b = [k for k in range(len(b)) if k not in used_b]
    count_a = Counter(a[k] for k in free_a)
    count_b = Counter(b[k] for k in free_b)
    where_b = {b[k]: k for k in free_b}
    for ia in free_a:
        token = a[ia]
        if count_a[token] == 1 and count_b.get(token) == 1 and (len(token) >= 3 or token.isdigit()):
            take(ia, where_b[token])
    return recovered


def word_char_spans(
    matches: list[Match],
    token_spans: list[tuple[int, int]],
    word_norms: list[str],
    text: str,
) -> dict[int, tuple[int, int]]:
    """Character span in ``text`` (the markdown) for every matched word.

    1:1 and 2:1 matches map the word to the covered token range. In 1:2
    matches (one md token read as two words, e.g. a hyphenated line break)
    the token's letters and digits are divided between the words in
    proportion to theirs ("maximum." → "maxi" + "mum."), so punctuation at
    the token's edges does not shift the cut; punctuation at a cut goes to
    the left piece ("l’" + "elenco"). Every word gets a non-empty span.
    """
    spans: dict[int, tuple[int, int]] = {}
    for m in matches:
        start = token_spans[m.md_start][0]
        end = token_spans[m.md_end - 1][1]
        n_words = m.word_end - m.word_start
        if n_words == 1:
            spans[m.word_start] = (start, end)
            continue
        lengths = [max(1, sum(ch.isalnum() for ch in word_norms[j])) for j in range(m.word_start, m.word_end)]
        total = sum(lengths)
        core = [k for k in range(start, end) if text[k].isalnum()] or list(range(start, end))
        cursor, done = start, 0
        for offset, length in enumerate(lengths):
            j = m.word_start + offset
            done += length
            if offset == n_words - 1:
                piece_end = end
            else:
                piece_end = core[max(1, min(len(core), round(len(core) * done / total))) - 1] + 1
                while piece_end < end - 1 and not text[piece_end].isalnum():
                    piece_end += 1
                piece_end = min(piece_end, end - 1)
            spans[j] = (cursor, max(cursor + 1, piece_end))
            cursor = spans[j][1]
    return spans
