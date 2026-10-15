"""Checkbox detection in images and linking to the markdown."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from swissdocai.result import PageInput, build_analyze_result  # noqa: E402
from swissdocai.selection_marks import SelectionMark, detect_selection_marks, remove_mark_words  # noqa: E402
from swissdocai.words import PageWords, Word, group_lines  # noqa: E402

DPI = 150
BOX = 20  # px ≈ 3.4 mm at 150 dpi


def page_with_boxes() -> Image.Image:
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    # an empty box, a ticked box, a box with an X touching its border
    for k, tick in enumerate(["", "x", "X"]):
        x, y = 100, 100 + 60 * k
        d.rectangle((x, y, x + BOX, y + BOX), outline="black", width=2)
        if tick == "x":
            d.line((x + 5, y + 10, x + 9, y + 15), fill="black", width=2)
            d.line((x + 9, y + 15, x + 16, y + 4), fill="black", width=2)
        elif tick == "X":
            d.line((x, y, x + BOX, y + BOX), fill="black", width=2)
            d.line((x, y + BOX, x + BOX, y), fill="black", width=2)
    d.ellipse((300, 100, 320, 120), outline="black", width=3)  # round letter "O"
    d.rectangle((400, 100, 420, 120), fill="black")  # solid bullet
    for k in range(4):  # comb field: touching boxes
        d.rectangle((500 + 20 * k, 100, 520 + 20 * k, 120), outline="black", width=2)
    d.rectangle((700, 100, 760, 160), outline="black", width=2)  # too large
    return img


def test_detects_checkboxes_and_states_only():
    marks = sorted(detect_selection_marks(page_with_boxes(), DPI), key=lambda m: m.bbox[1])
    assert [m.selected for m in marks] == [False, True, True]
    for mark, top in zip(marks, (100, 160, 220)):
        assert abs(mark.bbox[0] - 100) <= 2 and abs(mark.bbox[1] - top) <= 2


def test_ocr_words_on_boxes_are_removed():
    marks = [SelectionMark((100, 100, 120, 120), True)]
    words = [Word("X", (101, 101, 119, 119)), Word("Ja", (130, 102, 150, 118))]
    assert [w.text for w in remove_mark_words(words, marks)] == ["Ja"]


def test_marks_link_to_markdown_glyphs():
    words = [Word("bezahlt", (130, 100, 200, 120)), Word("offen", (130, 160, 190, 180))]
    page = PageWords(1240, 1754, words, source="ocr")
    page.lines = group_lines(words)
    page.marks = [
        SelectionMark((100, 160, 120, 180), selected=False, confidence=0.9),
        SelectionMark((100, 100, 120, 120), selected=True, confidence=0.9),
    ]
    md = "- ☒ bezahlt\n- ☒ offen"  # the model claims both are ticked
    result = build_analyze_result([PageInput(1, md, page)], "test")["analyzeResult"]
    content = result["content"]
    marks = result["pages"][0]["selectionMarks"]
    linked = {(m["state"], content[m["span"]["offset"]], m["confidence"]) for m in marks}
    # the visual state wins; the disagreement halves the confidence
    assert linked == {("selected", "☒", 0.9), ("unselected", "☒", 0.45)}
