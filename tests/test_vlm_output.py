"""Cleaning of generated page markdown."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swissdocai.vlm import clean_output  # noqa: E402


def test_outer_markdown_fence_is_removed():
    assert clean_output("```markdown\n# Titel\n\n- [x] Ja\n```") == "# Titel\n\n- [x] Ja"


def test_truncated_fence_without_close_is_removed():
    assert clean_output("```md\n# Titel\n\nText bricht ab") == "# Titel\n\nText bricht ab"


def test_real_code_blocks_are_kept():
    page = "```python\nprint(1)\n```"
    assert clean_output(page) == page
    page = "Text\n\n```markdown\n# Beispiel\n```"
    assert clean_output(page) == page


def test_page_separator_is_removed():
    assert clean_output("A\n\n<!-- PageBreak -->\n\nB") == "A\n\nB"


def test_known_ocr_models_get_their_own_prompt():
    from swissdocai.prompts import INSTRUCTION, prompt_for

    assert "OTSL" in prompt_for("SeerRay-Lab/Xiaomi-OCR-0")
    assert "headers, footers" in prompt_for("C:/models/xiaomi-ocr-0")
    assert prompt_for("zai-org/GLM-OCR") == "Text Recognition:"
    assert prompt_for("output/runs/qwen3vl-2b-run1/final") == INSTRUCTION
