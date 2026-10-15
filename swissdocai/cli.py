"""Command line: ``swissdocai redact``, ``swissdocai analyze`` and ``swissdocai doctor``.

    swissdocai analyze invoice.pdf --model <model> -o invoice.json --visualize vis/
    swissdocai analyze doc.pdf --markdown-from doc.md          # no model needed
    swissdocai redact letter.pdf -o letter.redacted.pdf --preview review/   # all on this machine, as the app
    swissdocai redact letter.pdf --llm swissdocai              # PII step at the SwissDocAI service
    swissdocai doctor                                          # hardware and what would run on it
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("swissdocai")


def parse_page_list(spec: str | None) -> set[int] | None:
    """'1,3-5' → {1, 3, 4, 5}."""
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        elif part:
            pages.add(int(part))
    return pages


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Markdown model (checkpoint dir or hub id, e.g. SeerRay-Lab/Xiaomi-OCR-0)")
    parser.add_argument("--prompt", help="Page instruction for --model (default: its own for known OCR models)")
    parser.add_argument("--processor-from", help="Load the processor from here instead of --model")
    parser.add_argument("--words", choices=["auto", "pdf", "ocr", "hybrid"], default="auto",
                        help="Word layer: PDF text layer when usable (auto), only PDF, always OCR, or both (hybrid)")
    parser.add_argument("--ocr-det", default="db_resnet50",
                        help="docTR detection architecture or local/hub model")
    parser.add_argument("--ocr-reco", default="parseq",
                        help="docTR recognition architecture or local/hub model")
    parser.add_argument("--dpi", type=int, default=150, help="PDF render resolution")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-vision-tokens", type=int, default=None,
                        help="Override the checkpoint's image token budget")
    parser.add_argument("--verbose", action="store_true")


def build_analyzer(args, need_model: bool = True):
    """DocumentAnalyzer from parsed CLI arguments."""
    from .pipeline import DocumentAnalyzer

    markdown_model = None
    if need_model:
        from .vlm import MarkdownModel

        markdown_model = MarkdownModel(
            args.model,
            processor_from=args.processor_from,
            max_vision_tokens=args.max_vision_tokens,
            max_new_tokens=args.max_new_tokens,
            prompt=args.prompt,
        )
    ocr = None
    if args.words != "pdf":
        from .words import DoctrWordReader

        ocr = DoctrWordReader(det_arch=args.ocr_det, reco_arch=args.ocr_reco)
    return DocumentAnalyzer(
        markdown_model=markdown_model,
        word_mode=args.words,
        ocr=ocr,
        dpi=args.dpi,
        model_id=args.model or "markdown-from-file",
    )


def cmd_analyze(args) -> None:
    from .pipeline import load_pages, split_markdown_pages

    if not args.model and not args.markdown_from:
        sys.exit("give --model, or --markdown-from to skip the model")
    markdown_pages = None
    if args.markdown_from:
        markdown_pages = split_markdown_pages(Path(args.markdown_from).read_text(encoding="utf-8"))

    analyzer = build_analyzer(args, need_model=markdown_pages is None)
    page_numbers = parse_page_list(args.pages)
    started = time.time()
    result = analyzer.analyze(args.input, page_numbers=page_numbers, markdown_pages=markdown_pages)
    elapsed = time.time() - started

    output = Path(args.output) if args.output else Path(args.input).with_suffix(".analyze.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    analyze = result["analyzeResult"]
    for stats in analyze["swissdocai"]["pages"]:
        logger.info(
            "page %d: %d words (%s), word coverage %s, markdown grounded %s",
            stats["pageNumber"], stats["wordCount"], stats["wordSource"],
            stats["wordCoverage"], stats["markdownGrounded"],
        )
    logger.info(
        "%d pages, %d paragraphs, %d tables, %d selection marks in %.1fs → %s",
        len(analyze["pages"]), len(analyze["paragraphs"]), len(analyze["tables"]),
        sum(len(p.get("selectionMarks", [])) for p in analyze["pages"]), elapsed, output,
    )

    if args.markdown_out:
        Path(args.markdown_out).write_text(analyze["content"], encoding="utf-8")

    if args.visualize:
        from .visualize import draw_result

        vis_dir = Path(args.visualize)
        vis_dir.mkdir(parents=True, exist_ok=True)
        images = [p.image for p in load_pages(args.input, args.dpi, page_numbers)]
        for page, image in zip(analyze["pages"], draw_result(result, images)):
            image.save(vis_dir / f"{Path(args.input).stem}_page{page['pageNumber']:03d}.png")
        logger.info("Annotated pages written to %s", vis_dir)


def cmd_doctor(args) -> None:
    from .engine import describe

    print(json.dumps(describe(), indent=1, ensure_ascii=False))


def _pii_client(args) -> tuple[object, object]:
    """What finds the personal data (the local model's chat client, or the SwissDocAI PII
    service), and the local engine to stop afterwards (or None)."""
    if args.llm == "swissdocai":
        from .saas import SwissDocAIPii

        return SwissDocAIPii(), None
    if args.llm == "local" or args.llm.startswith("local:"):
        from .engine import DEFAULT_MODEL, LocalEngine

        engine = LocalEngine(args.llm.split(":", 1)[1] if ":" in args.llm else DEFAULT_MODEL)
        return engine.client(), engine
    sys.exit(f"--llm {args.llm}: local, local:<model> or swissdocai")


def _policy(path: str | None):
    """The policy file (JSON: redact, keep, notes), a language code for the standard policy, or
    None for "none" (the categories only, no lists)."""
    from .policy import Policy, default_policy

    if not path or path == "none":
        return None
    if path in ("de", "fr", "it", "en"):
        return default_policy(path)
    return Policy.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_redact(args) -> None:
    from .pii import CATEGORIES, DEFAULT_CATEGORIES
    from .redact import redact_document
    from .words import DoctrWordReader, OnnxWordReader

    categories = tuple(c.strip() for c in args.categories.split(",")) if args.categories else DEFAULT_CATEGORIES
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        sys.exit(f"unknown categories {unknown}; choose from {sorted(CATEGORIES)}")
    reader = None
    if args.reader.lower() == "local":
        from .engine import READER_MODEL, LocalEngine

        reader = LocalEngine(READER_MODEL).reader()
    elif args.reader.lower() == "swissdocai":
        from .saas import SwissDocAIReader

        reader = SwissDocAIReader()
    elif args.reader.lower() != "none":
        from .vlm import MarkdownModel

        reader = MarkdownModel(args.reader, max_new_tokens=args.max_new_tokens)
    output = Path(args.output) if args.output else Path(args.input).with_name(Path(args.input).stem + ".redacted.pdf")
    client, engine = _pii_client(args)
    try:
        report = redact_document(
            args.input, output, client=client, reader=reader,
            ocr=OnnxWordReader(det_arch=args.ocr_det, reco_arch=args.ocr_reco) if args.ocr == "onnx"
            else DoctrWordReader(det_arch=args.ocr_det, reco_arch=args.ocr_reco or "parseq"),
            categories=categories,
            dpi=args.dpi, verify=not args.no_verify, report_text=args.report_text, preview_dir=args.preview,
            policy=_policy(args.policy), qr_codes=args.qr_codes,
        )
    finally:
        if engine:
            engine.stop()
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for page in report["pages"]:
        logger.info("page %d (%s): %d words redacted in %d bars, %d signature(s), %d QR code(s)", page["page"],
                    page["wordSource"], page["redactedWords"], page["bars"], page["signatures"], page["qrCodes"])
    review = sum(len(page["needsReview"]) for page in report["pages"])
    if review:
        logger.warning("%d found value(s) only partly boxed even after looking again — check 'needsReview' in %s",
                       review, report_path)
    if report["notFound"]:
        logger.warning("%d detected strings could not be located on the pages (see the report)",
                       len(report["notFound"]))
    verification = report.get("verification")
    if verification and not verification["ok"]:
        logger.warning("still readable after redaction: %d — check %s", len(verification["stillReadable"]),
                       report_path)
    logger.info("%s → %s (+ %s) in %.1fs", args.input, output, report_path.name, report["seconds"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="swissdocai", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one document into a result JSON")
    analyze.add_argument("input", help="PDF or image file (PNG/JPEG/TIFF, multi-page TIFF ok)")
    analyze.add_argument("-o", "--output", help="Result JSON (default: <input>.analyze.json)")
    analyze.add_argument("--markdown-from",
                         help="Use this markdown (pages split by <!-- PageBreak -->) instead of the model")
    analyze.add_argument("--pages", help="Page selection, e.g. '1,3-5' (default: all)")
    analyze.add_argument("--markdown-out", help="Also write the markdown content here")
    analyze.add_argument("--visualize", help="Write annotated page images into this directory")
    _add_engine_args(analyze)
    analyze.set_defaults(func=cmd_analyze)

    redact = sub.add_parser("redact", help="Black out personal data and write an image-only PDF")
    redact.add_argument("input", help="PDF or image file")
    redact.add_argument("-o", "--output", help="Redacted PDF (default: <input>.redacted.pdf)")
    redact.add_argument("--report", help="Redaction report JSON (default: <output>.report.json)")
    redact.add_argument("--preview", help="Write review images (red boxes instead of black) into this directory")
    redact.add_argument("--ocr", choices=("onnx", "doctr"), default="onnx",
                        help="Word layer through ONNX Runtime, no PyTorch (default, as the app), or docTR with "
                             "PyTorch")
    redact.add_argument("--reader", default="local",
                        help="Model that transcribes each page for the PII step: 'local' (default, as the app: "
                             "GLM-OCR through llama.cpp), 'swissdocai' (the SwissDocAI OCR service), a transformers "
                             "model id (e.g. zai-org/GLM-OCR, needs PyTorch) or 'none' (page words only)")
    redact.add_argument("--llm", default="local",
                        help="local (default, as the app: the PII step on this machine with llama.cpp, see "
                             "swissdocai doctor), local:<model>, or swissdocai (the SwissDocAI PII service: "
                             "SWISSDOCAI_API_KEY, SWISSDOCAI_API_URL)")
    redact.add_argument("--policy", default="de",
                        help="What to black out and keep: a JSON file {redact: [...], keep: [...], notes: ''}, "
                             "de/fr/it/en for the standard lists in that language (default: de, as the app; the "
                             "German lists did best on documents in all four languages), or none for the "
                             "categories only")
    redact.add_argument("--categories", help="Comma-separated, default: person,birth_date,address,contact,"
                                             "identifier,financial (also: date, organization)")
    redact.add_argument("--no-qr-codes", dest="qr_codes", action="store_false",
                        help="Leave QR codes readable (by default every one is blacked out, as in the app: a "
                             "QR-bill's holds names, addresses and the IBAN)")
    redact.add_argument("--dpi", type=int, default=200, help="Render resolution of the output pages")
    redact.add_argument("--ocr-det", default="db_resnet50")
    redact.add_argument("--ocr-reco", default=None,
                        help="Recognizer (default: the Swiss one with --ocr onnx; docTR's parseq, which reads no "
                             "umlauts, with --ocr doctr), e.g. output/doctr_models/parseq_swiss.pt")
    redact.add_argument("--max-new-tokens", type=int, default=4096)
    redact.add_argument("--no-verify", action="store_true", help="Skip OCR-ing the result for leftovers")
    redact.add_argument("--report-text", action="store_true",
                        help="Put the redacted strings into the report (it then contains personal data)")
    redact.add_argument("--verbose", action="store_true")
    redact.set_defaults(func=cmd_redact)

    doctor = sub.add_parser("doctor", help="Show the detected hardware and the local engine setup")
    doctor.add_argument("--verbose", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("PIL", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)  # FontBBox noise on KaTeX fonts
    args.func(args)


if __name__ == "__main__":
    main()
