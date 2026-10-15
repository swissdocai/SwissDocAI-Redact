"""
Evaluate the OCR word layer against exact word boxes.

For pages with ground-truth words (synthetic, Fedlex, olmOCR text layers)
runs docTR (any det/reco: names, hub repos or fine-tuned .pt files) and
reports, overall and per language / theme:

- detection precision / recall / F1 (predicted vs. true word boxes, IoU >= 0.5)
- word accuracy: true words found with exactly the right text
- CER: character error rate over matched words
- detection recall on words with a digit (phone numbers, IBANs, dates): docTR's
  pretrained detector is unsure about digit groups
- box size relative to the true box (docTR's own boxes carry a margin)
- word accuracy on words with an umlaut or ß
- word accuracy with quotes folded (’ ‘ ‚ → ', “ ” „ → "): on scans the
  typographic and straight forms often look the same, and benchmarks differ
  in which one they write

``--degrade`` turns the clean renders into scan-like images first (fixed
seed), which is closer to what OCR sees in practice.

Besides our datasets, ``--dataset`` takes the public form benchmarks with
word boxes (real scans, never trained on; non-commercial licenses, so
evaluation only): an XFUND ``<lang>.val.json`` with its images extracted next
to it, or a FUNSD split directory (``annotations/`` + ``images/``), and the
dpi benchmark's typeset scans (``output/dpi_benchmark/truth.json`` from
scripts/eval_dpi.py) at the app's 200 dpi.

Usage:
    uv run --no-sync python scripts/eval_word_layer.py --dataset output/dataset_paged --max-pages 150
    uv run --no-sync python scripts/eval_word_layer.py --dataset output/fedlex_dataset \
        --ocr-reco output/doctr_models/parseq_swiss.pt --degrade
    uv run --no-sync python scripts/eval_word_layer.py --dataset output/benchmarks/xfund/de.val.json \
        output/benchmarks/funsd/dataset/testing_data
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from swissdocai.selection_marks import _iou  # noqa: E402

logger = logging.getLogger(__name__)


def match_words(truth: list[dict], predicted) -> list[tuple[int, int]]:
    """Greedy 1:1 matching of true and predicted boxes by IoU >= 0.5."""
    pairs = []
    for i, t in enumerate(truth):
        for j, p in enumerate(predicted):
            iou = _iou(tuple(t["bbox"]), p.bbox)
            if iou >= 0.5:
                pairs.append((iou, i, j))
    used_t, used_p, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


DPI_BENCHMARK_DPI = 200  # the app's default page resolution


def _height(box) -> float:
    return box[3] - box[1]


def _width(box) -> float:
    return box[2] - box[0]


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


_UMLAUTS = set("äöüÄÖÜß")


_QUOTES = str.maketrans({"’": "'", "‘": "'", "‚": "'", "ʼ": "'", "“": '"', "”": '"', "„": '"'})


def _fold_quotes(text: str) -> str:
    return text.translate(_QUOTES)


def _benchmark_words(words: list[dict]) -> list[dict]:
    return [{"text": w["text"].strip(), "bbox": [float(v) for v in w["box"]]} for w in words if w["text"].strip()]


def iter_pages(path: str, split: str, seed: int, max_pages: int):
    """(source, image, true words, language, theme) per page of a saved dataset,
    an XFUND json or a FUNSD split directory."""
    from PIL import Image

    p = Path(path)
    if p.name == "truth.json":  # the dpi benchmark's typeset scans (scripts/eval_dpi.py), boxes at 300 dpi
        from swissdocai.pipeline import load_pages

        scale = DPI_BENCHMARK_DPI / 300
        for doc in json.loads(p.read_text(encoding="utf-8"))[:max_pages]:
            pages = load_pages(doc["pdf"], DPI_BENCHMARK_DPI)
            for n, page in enumerate(pages):
                words = [{"text": w["text"].strip("*"), "bbox": [v * scale for v in b[1:]]}
                         for w in doc["words"] for b in w["boxes"] if b[0] == n]
                yield "dpi-benchmark", page.image, words, doc["language"], None
        return
    if p.suffix == ".json":  # XFUND: <lang>.<split>.json, images next to it
        data = json.loads(p.read_text(encoding="utf-8"))
        for doc in data["documents"][:max_pages]:
            words = _benchmark_words([w for entity in doc["document"] for w in entity["words"]])
            yield f"xfund-{data['lang']}", Image.open(p.parent / doc["img"]["fname"]), words, data["lang"], None
        return
    if (p / "annotations").is_dir():  # FUNSD; its training split only checks detector training (never trained on)
        source = "funsd-train" if p.name == "training_data" else "funsd"
        for ann in sorted((p / "annotations").glob("*.json"))[:max_pages]:
            form = json.loads(ann.read_text(encoding="utf-8"))["form"]
            words = _benchmark_words([w for entity in form for w in entity["words"]])
            yield source, Image.open(p / "images" / f"{ann.stem}.png"), words, "en", None
        return
    from datasets import load_from_disk

    ds = load_from_disk(path)[split]
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    done = 0
    for i in indices:
        if done >= max_pages:
            break
        row = ds[i]
        if not row.get("words"):
            continue
        done += 1
        yield p.name, row["image"], row["words"], row.get("language", "?"), row.get("theme")


def main():
    parser = argparse.ArgumentParser(description="Word layer evaluation")
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-pages", type=int, default=150, help="Per dataset")
    parser.add_argument("--ocr-det", default="db_resnet50")
    parser.add_argument("--ocr-reco", default="parseq")
    parser.add_argument("--degrade", action="store_true", help="Scan-like degradation (fixed seed)")
    parser.add_argument("--out", help="Write the scores as JSON here")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    from scan_augment import scan_like
    from swissdocai.words import DoctrWordReader

    reader = DoctrWordReader(det_arch=args.ocr_det, reco_arch=args.ocr_reco)
    totals: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    rng = random.Random(args.seed)
    for path in args.dataset:
        done = 0
        for source, image, truth, language, theme in iter_pages(path, args.split, args.seed, args.max_pages):
            image = image.convert("RGB")
            if args.degrade:
                image = scan_like(image, rng, strength=1.0, rotate=False)
            predicted = reader(image).words
            pairs = match_words(truth, predicted)
            correct = sum(1 for t, p in pairs if truth[t]["text"] == predicted[p].text)
            correct_q = sum(1 for t, p in pairs if _fold_quotes(truth[t]["text"]) == _fold_quotes(predicted[p].text))
            cer = sum(Levenshtein.normalized_distance(truth[t]["text"], predicted[p].text) for t, p in pairs)
            umlaut_words = sum(bool(set(w["text"]) & _UMLAUTS) for w in truth)
            umlaut_correct = sum(1 for t, p in pairs if set(truth[t]["text"]) & _UMLAUTS
                                 and truth[t]["text"] == predicted[p].text)
            digit_words = sum(has_digit(w["text"]) for w in truth)
            digit_matched = sum(has_digit(truth[t]["text"]) for t, _ in pairs)
            # box size against the true box: the conventions differ (docTR's boxes carry a margin)
            scale_h = sum(_height(predicted[p].bbox) / max(1e-6, _height(truth[t]["bbox"])) for t, p in pairs)
            scale_w = sum(_width(predicted[p].bbox) / max(1e-6, _width(truth[t]["bbox"])) for t, p in pairs)
            groups = ["all", f"source:{source}", f"language:{language}"]
            if theme:
                groups.append(f"theme:{theme}")
            for g in groups:
                s = totals[g]
                s["pages"] += 1
                s["true"] += len(truth)
                s["predicted"] += len(predicted)
                s["matched"] += len(pairs)
                s["correct"] += correct
                s["correct_q"] += correct_q
                s["cer_sum"] += cer
                s["true_umlauts"] += umlaut_words
                s["correct_umlauts"] += umlaut_correct
                s["true_digits"] += digit_words
                s["matched_digits"] += digit_matched
                s["scale_h"] += scale_h
                s["scale_w"] += scale_w
            done += 1
            if done % 25 == 0:
                logger.info("%s: %d pages", source, done)

    report = {}
    for group, s in sorted(totals.items()):
        precision = s["matched"] / max(1, s["predicted"])
        recall = s["matched"] / max(1, s["true"])
        report[group] = {
            "pages": int(s["pages"]),
            "det_precision": round(precision, 4),
            "det_recall": round(recall, 4),
            "det_f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
            "word_accuracy": round(s["correct"] / max(1, s["true"]), 4),
            "word_accuracy_quotes_folded": round(s["correct_q"] / max(1, s["true"]), 4),
            "cer_matched": round(s["cer_sum"] / max(1, s["matched"]), 4),
            "word_accuracy_umlauts": round(s["correct_umlauts"] / max(1, s["true_umlauts"]), 4),
            "umlaut_words": int(s["true_umlauts"]),
            "det_recall_digits": round(s["matched_digits"] / max(1, s["true_digits"]), 4),
            "digit_words": int(s["true_digits"]),
            "box_scale_h": round(s["scale_h"] / max(1, s["matched"]), 3),
            "box_scale_w": round(s["scale_w"] / max(1, s["matched"]), 3),
        }
    width = max(len(g) for g in report)
    print(f"\n{'group':{width}}  pages  det_P   det_R   det_F1  word_acc  acc_q  acc_ä  CER    R_digit  box_h  box_w")
    for group, r in report.items():
        print(f"{group:{width}}  {r['pages']:5d}  {r['det_precision']:.3f}   {r['det_recall']:.3f}   "
              f"{r['det_f1']:.3f}   {r['word_accuracy']:.3f}     {r['word_accuracy_quotes_folded']:.3f}  "
              f"{r['word_accuracy_umlauts']:.3f}  "
              f"{r['cer_matched']:.3f}  {r['det_recall_digits']:.3f}    {r['box_scale_h']:.2f}   "
              f"{r['box_scale_w']:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "scores": report}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
