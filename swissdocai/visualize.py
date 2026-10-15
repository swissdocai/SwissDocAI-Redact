"""Draw an analyze result over its page images (quality checks).

Colors: green = word aligned to the markdown, red = word without a span
(not in the markdown), blue = paragraph, orange = table cell, purple =
selection mark (filled when selected, dashed-thin when not linked).
"""

from PIL import Image, ImageDraw


def _to_pixels(polygon: list[float], unit: str, dpi: int) -> list[tuple[float, float]]:
    factor = dpi if unit == "inch" else 1.0
    return [(polygon[k] * factor, polygon[k + 1] * factor) for k in range(0, len(polygon), 2)]


def draw_result(result: dict, images: list[Image.Image]) -> list[Image.Image]:
    """One annotated copy of each page image, in result page order."""
    analyze = result["analyzeResult"]
    dpi = analyze.get("swissdocai", {}).get("renderDpi", 150)
    out = []
    for page, image in zip(analyze["pages"], images):
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas, "RGBA")
        unit = page["unit"]
        for word in page["words"]:
            color = (0, 160, 0, 255) if "span" in word else (220, 0, 0, 255)
            draw.polygon(_to_pixels(word["polygon"], unit, dpi), outline=color)
        for para in analyze["paragraphs"]:
            for region in para.get("boundingRegions", []):
                if region["pageNumber"] == page["pageNumber"]:
                    draw.polygon(
                        _to_pixels(region["polygon"], unit, dpi),
                        outline=(30, 90, 230, 255), fill=(30, 90, 230, 25),
                    )
        for table in analyze["tables"]:
            for cell in table["cells"]:
                for region in cell.get("boundingRegions", []):
                    if region["pageNumber"] == page["pageNumber"]:
                        draw.polygon(
                            _to_pixels(region["polygon"], unit, dpi),
                            outline=(240, 140, 0, 255), fill=(240, 140, 0, 20),
                        )
        for mark in page.get("selectionMarks", []):
            fill = (150, 40, 200, 110) if mark["state"] == "selected" else None
            draw.polygon(
                _to_pixels(mark["polygon"], unit, dpi),
                outline=(150, 40, 200, 255), fill=fill, width=3 if "span" in mark else 1,
            )
        out.append(canvas)
    return out
