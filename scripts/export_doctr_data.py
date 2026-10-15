"""
Export word crops for training the recognizer (scripts/train_doctr.py).

Reads datasets with a ``words`` column (output/dataset_paged,
output/fedlex_dataset, a unified dataset) and writes docTR's recognition
format:

    <out>/reco/{train,val}/images/<id>.jpg + labels.json   {"<id>.jpg": "Wort", ...}

Crops keep a small margin around the word and are stored at a fixed height;
words with characters outside the Swiss vocabulary (math glyphs, symbols) are
skipped.

``--real-reco`` adds word crops from real pages as <out>/reco/real/{train,val}/:
the app's word models (stock detector, Swiss recognizer) box and read the
page, and the page's reference markdown, aligned to those words as in the
pipeline (swissdocai/align.py), gives each box its text. Only 1:1 matches
with a close reading are kept, and only where a difference is more likely
the recognizer's than the reference text's (plausible_label); the crop is cut
from the detected box, as the recognizer sees it in the app. ``readings.json``
holds what the recognizer read, to see where it was wrong.

Usage:
    uv run --no-sync python scripts/export_doctr_data.py \
        --datasets output/dataset_paged output/fedlex_dataset --out output/doctr_data
    uv run --no-sync python scripts/export_doctr_data.py --out output/doctr_data \
        --real-reco output/unified_dataset_v4 --real-reco-pages 2000
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from swissdocai.vocab import SWISS_VOCAB  # noqa: E402

logger = logging.getLogger(__name__)
_VOCAB = set(SWISS_VOCAB)
# crops from labelled boxes get this margin around the word, in units of the box height
CROP_MARGIN = 0.15
# the sources of the unified dataset without word boxes that are real documents
REAL_SOURCES = ("olmocr", "doclingmatix")


def usable_word(word: dict, width: int, height: int) -> bool:
    text = word["text"]
    x0, y0, x1, y1 = word["bbox"]
    return (
        0 < len(text) <= 30
        and all(ch in _VOCAB for ch in text)
        and x1 - x0 >= 3 and y1 - y0 >= 5
        and x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height
    )


_NOT_TEXT = set("\\$<>{}|")  # markdown, LaTeX or HTML left in a token
_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_QUOTE_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"'})


def plausible_label(reading: str, label: str) -> bool:
    """Whether the reference text's word is a label to trust for a crop the recognizer read as
    ``reading``. Kept where they agree, and where the recognizer misread letters or their case
    ("COUrse" → "course"); dropped where the difference is more likely the reference text's
    or the box's: one inside the other (a cut box, punctuation the markdown moved), quote
    style, all capitals on one side (headings the markdown re-cased), single letters, HTML
    entities."""
    if reading == label:
        return True
    if _ENTITY.search(label) or len(label) < 2 or len(reading) < 2:
        return False
    if reading in label or label in reading:
        return False
    if reading.translate(_QUOTE_FOLD) == label.translate(_QUOTE_FOLD):
        return False
    return not any(text.isupper() and sum(ch.isalpha() for ch in text) >= 2 for text in (reading, label))


def export_crops(datasets: list[str], out: Path, words_per_page: int, crop_height: int, max_pages: int,
                 rng: random.Random, counts: Counter) -> None:
    """Sampled word crops from the labelled boxes of ``datasets``."""
    from datasets import load_from_disk

    for split_in, split_out in (("train", "train"), ("validation", "val")):
        folder = out / "reco" / split_out
        (folder / "images").mkdir(parents=True, exist_ok=True)
        labels = {}
        for dataset_path in datasets:
            ds = load_from_disk(dataset_path)[split_in]
            tag = Path(dataset_path).name
            n = len(ds) if not max_pages else min(max_pages, len(ds))
            for idx in range(n):
                row = ds[idx]
                words = row.get("words") or []
                if not words:
                    continue
                image = row["image"].convert("RGB")
                width, height = image.size
                candidates = [w for w in words if usable_word(w, width, height)]
                for k, w in enumerate(rng.sample(candidates, min(words_per_page, len(candidates)))):
                    x0, y0, x1, y1 = w["bbox"]
                    pad = CROP_MARGIN * (y1 - y0)
                    crop = image.crop((max(0, x0 - pad), max(0, y0 - pad),
                                       min(width, x1 + pad), min(height, y1 + pad)))
                    ratio = crop_height / crop.height
                    crop = crop.resize((max(1, round(crop.width * ratio)), crop_height))
                    name = f"{tag}_{idx:06d}_{k:03d}.jpg"
                    crop.save(folder / "images" / name, quality=92)
                    labels[name] = w["text"]
                    counts[f"reco_{split_out}"] += 1
                if idx % 500 == 0:
                    logger.info("%s %s: %d/%d pages", tag, split_in, idx, n)
        (folder / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")


def export_real_crops(dataset_path: str, out: Path, pages: int, words_per_page: int, crop_height: int,
                      recognizer: str, rng: random.Random, counts: Counter) -> None:
    """Word crops from real pages labelled with their reference markdown, ``pages`` per split
    (validation: 1/20), half of them from each source."""
    from datasets import load_from_disk

    from swissdocai.align import align_tokens
    from swissdocai.markdown_doc import parse_markdown
    from swissdocai.text import normalize_token
    from swissdocai.words import DoctrWordReader

    # docTR on the GPU (twice as fast as OnnxTR on DirectML); the same stock detector as the app
    reader = DoctrWordReader(reco_arch=recognizer, batch_size=256)
    ds = load_from_disk(dataset_path)
    for split_in, split_out, n in (("train", "train", pages), ("validation", "val", max(50, pages // 20))):
        folder = out / "reco" / "real" / split_out
        (folder / "images").mkdir(parents=True, exist_ok=True)
        labels, readings = {}, {}
        sources = ds[split_in].select_columns(["source"])["source"]
        for prefix in REAL_SOURCES:
            indices = [i for i, src in enumerate(sources) if src.startswith(prefix)]
            for k, idx in enumerate(sorted(rng.sample(indices, min(n // len(REAL_SOURCES), len(indices))))):
                row = ds[split_in][idx]
                image = row["image"].convert("RGB")
                words = reader(image).words
                doc = parse_markdown(row["markdown"] or "")
                matches = align_tokens([t.norm for t in doc.tokens], [normalize_token(w.text) for w in words])
                pairs = []
                for m in matches:
                    if m.md_end - m.md_start != 1 or m.word_end - m.word_start != 1 or m.score < 0.75:
                        continue
                    token, word = doc.tokens[m.md_start], words[m.word_start]
                    text = doc.content[token.start:token.end].replace("­", "").replace("​", "")
                    x0, y0, x1, y1 = word.bbox
                    if (0 < len(text) <= 30 and all(ch in _VOCAB for ch in text) and not set(text) & _NOT_TEXT
                            and x1 - x0 >= 3 and y1 - y0 >= 5):
                        pairs.append((text, word))
                counts[f"real_reco_{split_out}_matched"] += len(pairs)
                pairs = [(text, word) for text, word in pairs if plausible_label(word.text, text)]
                for j, (text, word) in enumerate(rng.sample(pairs, min(words_per_page, len(pairs)))):
                    crop = image.crop(tuple(round(v) for v in word.bbox))
                    ratio = crop_height / crop.height
                    crop = crop.resize((max(1, round(crop.width * ratio)), crop_height))
                    name = f"{prefix}_{idx:06d}_{j:03d}.jpg"
                    crop.save(folder / "images" / name, quality=92)
                    labels[name] = text
                    readings[name] = word.text
                    counts[f"real_reco_{split_out}"] += 1
                    counts[f"real_reco_{split_out}_misread"] += word.text != text
                if k % 250 == 0:
                    logger.info("real crops %s %s: %d pages, %s", prefix, split_out, k, dict(counts))
        (folder / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        (folder / "readings.json").write_text(json.dumps(readings, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export word crops for the recognizer")
    parser.add_argument("--datasets", nargs="*", default=[], help="Datasets with word boxes")
    parser.add_argument("--out", default="output/doctr_data")
    parser.add_argument("--words-per-page", type=int, default=60, help="Crops sampled per page")
    parser.add_argument("--crop-height", type=int, default=64)
    parser.add_argument("--max-pages", type=int, default=0, help="Per dataset split (0 = all)")
    parser.add_argument("--real-reco", help="Also export word crops from real pages of this unified dataset")
    parser.add_argument("--real-reco-pages", type=int, default=3000, help="Real pages for crops (validation: 1/20)")
    parser.add_argument("--real-reco-recognizer", default="output/doctr_models/parseq_swiss.pt",
                        help="Recognizer whose readings are stored next to the labels")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    rng = random.Random(args.seed)
    out = Path(args.out)
    counts: Counter = Counter()
    if args.datasets:
        export_crops(args.datasets, out, args.words_per_page, args.crop_height, args.max_pages, rng, counts)
    if args.real_reco:
        export_real_crops(args.real_reco, out, args.real_reco_pages, args.words_per_page, args.crop_height,
                          args.real_reco_recognizer, rng, counts)
    (out / "vocab.txt").write_text(SWISS_VOCAB, encoding="utf-8")
    logger.info("Exported: %s", dict(counts))


if __name__ == "__main__":
    main()
