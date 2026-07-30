"""
Text recognition — the OCR wrapper shared by the OCR, chat and plate agents.

RapidOCR (PaddleOCR models exported to ONNX) on **CPU**, deliberately. Text
extraction is the one heavy capability that does not need the GPU, and on a 6 GB
card every megabyte matters — InsightFace, YOLO and CLIP all want space. Running
OCR on CPU keeps the GPU entirely for vision.

Text regions come back with their positions, not just their content. That is
what makes the chat agent possible: knowing *where* on a screenshot a string
appeared is how you tell a contact name from a message body.

One quirk worth knowing: RapidOCR sometimes drops spaces, returning
"inthatwhitecarKL07AB1234". Entity extraction is written to survive that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Below this the recognition is usually garbage.
MIN_TEXT_SCORE = 0.45

_OCR: Any | None = None


def get_ocr() -> Any:
    """Load RapidOCR once. Models are bundled, so there is no download."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
        logger.info("RapidOCR ready (CPU)")
    return _OCR


@dataclass
class TextRegion:
    """One recognised string and where it was found."""

    text: str
    score: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def describe(self) -> str:
        return f"{self.text!r} @{self.bbox} ({self.score:.2f})"


def read_text(
    image: "str | Path | Any", *, min_score: float = MIN_TEXT_SCORE
) -> list[TextRegion]:
    """Recognise text in an image, returning regions ordered top-to-bottom."""
    ocr = get_ocr()
    source = str(image) if isinstance(image, (str, Path)) else image

    raw, _elapsed = ocr(source)
    if not raw:
        return []

    regions: list[TextRegion] = []
    for entry in raw:
        try:
            box, text, score = entry
        except (ValueError, TypeError):
            continue
        if not text or float(score) < min_score:
            continue

        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        regions.append(
            TextRegion(
                text=str(text).strip(),
                score=float(score),
                bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            )
        )

    # Reading order: top to bottom, then left to right.
    return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))


def full_text(regions: list[TextRegion]) -> str:
    """Join regions into one string for entity extraction and display."""
    return "\n".join(r.text for r in regions)
