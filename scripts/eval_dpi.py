"""
What the page resolution (dpi) costs and buys, end to end.

The PII benchmark texts (output/pii_benchmark/benchmark.jsonl) are typeset as scanned-looking
PDFs (A4, 300 dpi, slight blur, noise and JPEG artefacts) — so the exact pixel
box of every personal value, decoy and word is known. Each PDF then goes
through the app's local pipeline (OnnxTR words, GLM-OCR reader, Gemma PII
step) at every --dpi, and the result is scored on the page itself:

- a personal value counts as redacted when the black boxes cover >= 90% of
  every one of its boxes; recall, documents fully redacted;
- decoys and other words: blacked out when >= 50% covered;
- OCR word accuracy: rendered words read exactly (box IoU > 0.5);
- seconds per page for words, reading, personal data, and in total.

Usage:
    uv run --no-sync python scripts/eval_dpi.py --dpi 150 --dpi 200 --dpi 300
    uv run --no-sync python scripts/eval_dpi.py --render-only --limit 2   # just the PDFs (no models)
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

logger = logging.getLogger(__name__)

RENDER_DPI = 300
PAGE = (2480, 3508)  # A4 at 300 dpi
MARGIN = 220
FONT_SIZE = 44  # ~10.5 pt at 300 dpi
LINE = int(FONT_SIZE * 1.45)


def _font():
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    raise SystemExit("no TrueType font found (Arial, DejaVu Sans or Liberation Sans)")


def typeset(text: str, font) -> tuple[list[Image.Image], list[tuple | None]]:
    """The text on A4 pages; per character of ``text`` its (page, x0, y0, x1, y1), None if not drawn.

    Markdown emphasis (*) is not drawn; everything else is, line by line, wrapped at the margin."""
    pages = [Image.new("L", PAGE, 255)]
    draw = ImageDraw.Draw(pages[0])
    where: list[tuple | None] = [None] * len(text)
    y = MARGIN
    width = PAGE[0] - 2 * MARGIN
    offset = 0
    for paragraph in text.split("\n"):
        chars = [(i, ch) for i, ch in enumerate(paragraph, offset) if ch != "*"]
        offset += len(paragraph) + 1
        if not chars:
            y += LINE // 2
            continue
        # greedy wrap on spaces
        lines, current = [], []
        for item in chars:
            current.append(item)
            line_text = "".join(ch for _, ch in current)
            if font.getlength(line_text) > width and " " in line_text.strip():
                cut = max(k for k, (_, ch) in enumerate(current) if ch == " ")
                lines.append(current[:cut])
                current = current[cut + 1:]
        lines.append(current)
        for line in lines:
            if y + LINE > PAGE[1] - MARGIN:
                pages.append(Image.new("L", PAGE, 255))
                draw = ImageDraw.Draw(pages[-1])
                y = MARGIN
            line_text = "".join(ch for _, ch in line)
            draw.text((MARGIN, y), line_text, font=font, fill=20)
            for k, (i, ch) in enumerate(line):
                if ch.isspace():
                    continue
                x0 = MARGIN + font.getlength(line_text[:k])
                where[i] = (len(pages) - 1, x0, y, MARGIN + font.getlength(line_text[:k + 1]), y + FONT_SIZE * 1.15)
            y += LINE
    return pages, where


def boxes(where: list, start: int, end: int) -> list[tuple]:
    """Boxes (page, x0, y0, x1, y1) covering text[start:end], one per page line."""
    runs: dict[tuple, list] = {}
    for item in where[start:end]:
        if item is not None:
            runs.setdefault((item[0], item[2]), []).append(item)
    return [(p, min(i[1] for i in run), y, max(i[3] for i in run), run[0][4]) for (p, y), run in runs.items()]


def scan_look(page: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    blurred = page.filter(ImageFilter.GaussianBlur(0.8))
    noisy = np.clip(np.asarray(blurred, dtype=np.float32) + rng.normal(0, 7, (PAGE[1], PAGE[0])), 0, 255)
    return Image.fromarray(noisy.astype(np.uint8))


def render(doc: dict, out_dir: Path, font) -> dict:
    """The document as a scanned-looking PDF; returns its ground truth."""
    text = doc["text"]
    pages, where = typeset(text, font)
    pdf = out_dir / f"{doc['id']}.pdf"
    scans = [scan_look(p, hash(doc["id"]) % 2**32 + i).convert("RGB") for i, p in enumerate(pages)]
    # JPEG artefacts as in a real scan, then an image-only PDF at 300 dpi
    jpegs = []
    for scan in scans:
        import io

        buffer = io.BytesIO()
        scan.save(buffer, "JPEG", quality=70)
        jpegs.append(Image.open(io.BytesIO(buffer.getvalue())))
    jpegs[0].save(pdf, "PDF", resolution=RENDER_DPI, save_all=True, append_images=jpegs[1:])
    words = [(m.group(0), boxes(where, m.start(), m.end())) for m in re.finditer(r"[^\s*]+", text)]
    return {
        "id": doc["id"], "language": doc["language"], "pdf": str(pdf), "pages": len(pages),
        "pii": [{"value": p["value"], "category": p["category"], "boxes": boxes(where, *p["span"])} for p in doc["pii"]],
        "decoys": [{"value": d["value"], "boxes": boxes(where, *d["span"])} for d in doc["decoys"]],
        "words": [{"text": w, "boxes": b} for w, b in words if b],
    }


# --- scoring on the page ----------------------------------------------------------------


def coverage(mask: np.ndarray, box: tuple, scale: float) -> float:
    _, x0, y0, x1, y1 = box
    x0, y0, x1, y1 = (int(round(v * scale)) for v in (x0, y0, x1, y1))
    area = mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return float(area.mean()) if area.size else 0.0


def black_masks(job) -> list[np.ndarray]:
    from swissdocai.redact import page_boxes

    masks = []
    for page in job.pages:
        mask = np.zeros((page.source.image.height, page.source.image.width), dtype=bool)
        for (x0, y0, x1, y1), _ in page_boxes(page):
            mask[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
        masks.append(mask)
    return masks


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _norm(word: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", word).casefold()


def score(truth: dict, job, dpi: int) -> dict:
    scale = dpi / RENDER_DPI
    masks = black_masks(job)

    def covered(box, needed):
        page = box[0]
        return page < len(masks) and coverage(masks[page], box, scale) >= needed

    pii = [(p["category"], all(covered(b, 0.9) for b in p["boxes"]), p["value"]) for p in truth["pii"] if p["boxes"]]
    personal = {p["value"] for p in truth["pii"]}
    decoys = [any(covered(b, 0.5) for b in d["boxes"]) for d in truth["decoys"]
              if d["boxes"] and d["value"] not in personal]
    special = [b for item in truth["pii"] + truth["decoys"] for b in item["boxes"]]

    def inside_special(box) -> bool:  # a word of a personal value or decoy (their boxes span several words)
        page, x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return any(p == page and sx0 - 2 <= cx <= sx1 + 2 and sy0 - 2 <= cy <= sy1 + 2
                   for p, sx0, sy0, sx1, sy1 in special)

    others = [w for w in truth["words"] if not any(inside_special(b) for b in w["boxes"])]
    collateral = sum(any(covered(b, 0.5) for b in w["boxes"]) for w in others)
    # OCR: is each rendered word read exactly where it is?
    read = 0
    for w in truth["words"]:
        page, x0, y0, x1, y1 = w["boxes"][0]
        if page >= len(job.pages):
            continue
        target = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
        best = max(job.pages[page].words.words, key=lambda o: _iou(o.bbox, target), default=None)
        if best is not None and _iou(best.bbox, target) > 0.5 and _norm(best.text) == _norm(w["text"]):
            read += 1
    return {"pii": pii, "decoys": decoys, "collateral": collateral, "others": len(others),
            "words_read": read, "words": len(truth["words"])}


def main():
    parser = argparse.ArgumentParser(description="Page resolution: speed and quality, end to end")
    parser.add_argument("--benchmark", default="output/pii_benchmark/benchmark.jsonl")
    parser.add_argument("--work", default="output/dpi_benchmark")
    parser.add_argument("--dpi", type=int, action="append", help="Resolutions to compare (default 150, 200, 300)")
    parser.add_argument("--reader", choices=("local", "none"), default="local")
    parser.add_argument("--policy", default="de", help="Standard lists in this language (as the app)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "swissdocai", "onnxtr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    docs = [json.loads(line) for line in Path(args.benchmark).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        docs = docs[:args.limit]
    work = Path(args.work)
    (work / "pdf").mkdir(parents=True, exist_ok=True)
    font = _font()
    truths = [render(doc, work / "pdf", font) for doc in docs]
    (work / "truth.json").write_text(json.dumps(truths, ensure_ascii=False), encoding="utf-8")
    logger.info("%d documents typeset as scans (%d pages)", len(truths), sum(t["pages"] for t in truths))
    if args.render_only:
        return

    from swissdocai.engine import DEFAULT_MODEL, READER_MODEL, LocalEngine
    from swissdocai.pii import DEFAULT_CATEGORIES
    from swissdocai.policy import default_policy
    from swissdocai.redact import analyze_document
    from swissdocai.words import OnnxWordReader

    ocr = OnnxWordReader()
    pii_engine = LocalEngine(DEFAULT_MODEL)
    reader_engine = LocalEngine(READER_MODEL) if args.reader == "local" else None
    results_path = work / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    try:
        for dpi in args.dpi or [150, 200, 300]:
            per_doc, stage_seconds = [], defaultdict(float)
            pages_total = 0
            for truth in truths:
                marks = []

                def progress(stage, done, total, marks=marks):
                    marks.append((stage, time.time()))

                t = time.time()
                reader = reader_engine.reader() if reader_engine else None
                if reader:
                    reader.close = lambda: None  # keep the reader's server across documents
                job = analyze_document(truth["pdf"], pii_engine.client(), reader=reader, ocr=ocr,
                                       categories=DEFAULT_CATEGORIES, dpi=dpi,
                                       progress=progress, policy=default_policy(args.policy))
                end = time.time()
                marks.append(("end", end))
                stages = [m[0] for m in marks]
                for i, (stage, at) in enumerate(marks[:-1]):
                    nxt = next((m[1] for m in marks[i + 1:] if m[0] != stage), end)
                    if i == 0 or stages[i - 1] != stage:
                        stage_seconds[stage] += nxt - at
                pages_total += len(job.pages)
                s = score(truth, job, dpi)
                s["seconds"] = end - t
                per_doc.append(s)
                missed = [v for _, ok, v in s["pii"] if not ok]
                logger.info("%d dpi %s: %d/%d values, OCR %d/%d words, %.1fs%s", dpi, truth["id"],
                            sum(ok for _, ok, _ in s["pii"]), len(s["pii"]), s["words_read"], s["words"],
                            s["seconds"], f" — missed {missed}" if missed else "")
            values = [ok for s in per_doc for _, ok, _ in s["pii"]]
            decoys = [d for s in per_doc for d in s["decoys"]]
            summary = {
                "dpi": dpi, "reader": args.reader, "documents": len(per_doc), "pages": pages_total,
                "recall": round(sum(values) / max(1, len(values)), 4),
                "documents_fully_redacted": round(sum(all(ok for _, ok, _ in s["pii"]) for s in per_doc)
                                                  / max(1, len(per_doc)), 4),
                "decoys_redacted": round(sum(decoys) / max(1, len(decoys)), 4),
                "collateral": round(sum(s["collateral"] for s in per_doc) / max(1, sum(s["others"] for s in per_doc)), 4),
                "ocr_word_accuracy": round(sum(s["words_read"] for s in per_doc) / max(1, sum(s["words"] for s in per_doc)), 4),
                "seconds_per_page": {stage: round(sec / max(1, pages_total), 2) for stage, sec in stage_seconds.items()},
                "total_seconds_per_page": round(sum(s["seconds"] for s in per_doc) / max(1, pages_total), 2),
                "missed": [{"id": t["id"], "value": v} for t, s in zip(truths, per_doc) for _, ok, v in s["pii"] if not ok],
            }
            results[f"{dpi}dpi-{args.reader}"] = summary
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            logger.info("%d dpi: recall %.3f, all values in %.0f%% of documents, OCR words %.1f%%, %.1fs/page", dpi,
                        summary["recall"], 100 * summary["documents_fully_redacted"],
                        100 * summary["ocr_word_accuracy"], summary["total_seconds_per_page"])
    finally:
        pii_engine.stop()
        if reader_engine:
            reader_engine.stop()

    print(f"\n{'setup':16s} {'recall':>7s} {'docs ok':>8s} {'decoys':>7s} {'collat.':>8s} {'OCR':>6s} {'s/page':>7s}  stages")
    for name, s in results.items():
        stages = " ".join(f"{k}={v}" for k, v in s["seconds_per_page"].items())
        print(f"{name:16s} {s['recall']:7.3f} {s['documents_fully_redacted']:8.2f} {s['decoys_redacted']:7.2f} "
              f"{s['collateral']:8.3f} {s['ocr_word_accuracy']:6.3f} {s['total_seconds_per_page']:7.2f}  {stages}")


if __name__ == "__main__":
    main()
