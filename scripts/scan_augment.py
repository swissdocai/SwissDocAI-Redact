"""Scan-like degradation of clean renders, for training the recognizer (train_doctr.py)
and scoring the word layer on degraded pages (eval_word_layer.py --degrade)."""

import io
import random

import numpy as np
from PIL import Image, ImageFilter


def scan_like(image: Image.Image, rng: random.Random, strength: float = 1.0,
              rotate: bool = True) -> Image.Image:
    """Clean render → plausible scan: blur, rotation, contrast, noise, JPEG.

    ``rotate=False`` keeps the geometry (word boxes must stay exact).
    """
    if rng.random() < 0.5 * strength:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
    if rotate and rng.random() < 0.3 * strength:
        image = image.rotate(rng.uniform(-3, 3), expand=True, fillcolor=(255, 255, 255))
    if rng.random() < 0.4 * strength:
        arr = np.asarray(image, dtype=np.float32)
        lo, hi = rng.uniform(0, 60), rng.uniform(190, 255)
        arr = lo + arr / 255.0 * (hi - lo)
        arr += np.random.normal(0, rng.uniform(2, 12), arr.shape)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5 * strength:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=rng.randint(35, 85))
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return image
