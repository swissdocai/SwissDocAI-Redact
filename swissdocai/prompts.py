"""What a page reader (the local one in engine.py, vlm.py) is asked per page, and
how large the page image may be.
"""

INSTRUCTION = "Convert this document image to markdown format. Output only the markdown."

# Off-the-shelf OCR models trained on their own prompts. Xiaomi-OCR-0's
# document prompt says "ignoring headers and footers"; for redaction we ask
# for everything (it still drops running headers/footers — the word layer
# covers them). Tables come back as OTSL (converted in vlm.clean_output).
MODEL_PROMPTS = {
    # GLM-OCR reads plain text in reading order (no tables), headers/footers included
    "glm-ocr": "Text Recognition:",
    "xiaomi-ocr": (
        "Extract all information from the document image, including headers, footers, letterheads, addresses "
        "and signatures, and represent it in markdown format. Tables should be expressed in OTSL format, "
        "formulas in the document should be represented using LATEX format, and the parsing should be "
        "organized according to the reading order."
    ),
}


def prompt_for(model_path: str) -> str:
    """The page instruction for ``model_path``: its own for known OCR models, else ours."""
    name = model_path.lower()
    return next((prompt for key, prompt in MODEL_PROMPTS.items() if key in name), INSTRUCTION)


def configure_image_budget(processor, max_vision_tokens: int) -> None:
    """Cap page resolution at ``max_vision_tokens`` visual tokens.

    One visual token covers (patch_size * merge_size)^2 pixels: 28x28 for
    Qwen2-VL/2.5-VL, 32x32 for Qwen3-VL (16px patches, 2x2 merge).
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not getattr(image_processor, "size", None):
        return
    token_px = getattr(image_processor, "patch_size", 14) * getattr(image_processor, "merge_size", 2)
    max_pixels = max_vision_tokens * token_px * token_px
    image_processor.size["longest_edge"] = max_pixels
    image_processor.size["shortest_edge"] = min(
        image_processor.size.get("shortest_edge") or 65536, max_pixels
    )
