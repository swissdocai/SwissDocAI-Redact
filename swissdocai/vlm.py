"""Page image → markdown with a transformers vision-language model: an OCR
model such as GLM-OCR or Xiaomi-OCR-0, or a checkpoint of our own. Optional:
needs PyTorch and transformers (``pip install swissdocai[vlm]``); the app reads
pages through llama.cpp instead (engine.py).
"""

import logging
import re

logger = logging.getLogger(__name__)

# The model must never emit our page separator; strip it if it does.
_PAGE_BREAK = re.compile(r"\s*<!--\s*PageBreak\s*-->\s*")
# Chat models (e.g. the base Qwen3-VL) like to wrap the whole answer in a
# ```markdown fence, which would turn the page into one code block.
_OUTER_FENCE = re.compile(r"^```(?:markdown|md)[ \t]*\n(.*?)(?:\n```)?\s*$", re.S | re.I)


def clean_output(text: str) -> str:
    """Generated page markdown without an outer ```markdown fence or page separators,
    with OTSL tables (Xiaomi-OCR-0, Docling) as HTML."""
    from .otsl import convert_otsl_tables

    text = text.strip()
    fenced = _OUTER_FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    return convert_otsl_tables(_PAGE_BREAK.sub("\n\n", text).strip())


class MarkdownModel:
    """Greedy page-to-markdown generation with a transformers VLM.

    Args:
        model_path: Hub id or local checkpoint (a full model; merge LoRA first).
        processor_from: Load the processor from here instead (checkpoints
            saved by older transformers ship a stale processor config).
        max_vision_tokens: Override the checkpoint's image budget. By default
            the saved processor's budget is kept.
        max_new_tokens: Generation cap per page.
        device: "cuda" / "cpu"; defaults to cuda when available.
        prompt: Instruction sent with each page; defaults to the model's own
            for known OCR models (prompts.MODEL_PROMPTS), else
            prompts.INSTRUCTION.
    """

    def __init__(
        self,
        model_path: str,
        processor_from: str | None = None,
        max_vision_tokens: int | None = None,
        max_new_tokens: int = 4096,
        device: str | None = None,
        prompt: str | None = None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from .prompts import configure_image_budget, prompt_for

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt or prompt_for(model_path)
        # Qwen-VL families are built into transformers: never run code shipped with a checkpoint
        self.processor = AutoProcessor.from_pretrained(processor_from or model_path)
        if max_vision_tokens:
            configure_image_budget(self.processor, max_vision_tokens)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype).to(self.device)
        self.model.eval()
        self.model_id = model_path
        logger.info("Markdown model %s loaded on %s", model_path, self.device)

    def __call__(self, image) -> str:
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        if len(generated) >= self.max_new_tokens:
            logger.warning("Generation hit max_new_tokens=%d — page output truncated", self.max_new_tokens)
        return clean_output(self.processor.decode(generated, skip_special_tokens=True))
