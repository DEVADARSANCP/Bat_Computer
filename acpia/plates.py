"""
Number plate reading.

There is no dedicated plate-detection model here — plates are read from the
vehicle regions YOLO already found, by OCR. That is a deliberate trade: a
plate detector would be more accurate but is another model competing for 6 GB of
VRAM, and RapidOCR is already loaded and running on CPU.

Two techniques make up most of the difference:

**Upscaling.** A plate that occupies 40 pixels of width is unreadable at native
resolution and often fine at 3x. OCR models are trained on text at reasonable
sizes, so enlarging a small crop genuinely helps rather than just interpolating
noise.

**Consensus across preprocessing variants.** The same crop is read several ways —
raw, contrast-equalised, and cropped to the lower portion where plates usually
sit. If two independent variants produce the same string, that agreement is far
stronger evidence than one high-confidence read. If they disagree, that is
recorded too: a plate that reads three different ways is a plate we cannot claim.

**Honest limits.** Angled, dirty, motion-blurred or distant plates will fail, and
OCR confuses O/0, I/1, S/5 and B/8 on the characters that matter. When the
variants disagree the agent reports low confidence rather than picking a winner.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .extract import extract_plates
from .ocr import read_text

logger = logging.getLogger(__name__)

#: Plates below this width in the source image are unlikely to survive OCR even
#: upscaled. Reported as "too small to read" rather than guessed at.
MIN_VEHICLE_WIDTH = 90

#: Target width to upscale a vehicle crop to before reading.
TARGET_CROP_WIDTH = 900

#: Plates sit low on a vehicle. This fraction of the box height, measured from
#: the bottom, is tried as an additional focused variant.
PLATE_BAND = 0.45


@dataclass
class PlateRead:
    """A plate recovered from one vehicle region."""

    plate: str
    confidence: float
    votes: int
    variants_tried: int
    candidates: dict[str, int] = field(default_factory=dict)
    raw_text: str = ""

    @property
    def unanimous(self) -> bool:
        return self.votes == self.variants_tried and self.variants_tried > 1

    @property
    def contested(self) -> bool:
        """More than one distinct plate string was read from the same vehicle."""
        return len(self.candidates) > 1

    def describe(self) -> str:
        basis = f"{self.votes}/{self.variants_tried} reads agreed"
        if self.contested:
            others = ", ".join(f"{p}({n})" for p, n in self.candidates.items() if p != self.plate)
            basis += f"; also read as {others}"
        return f"{self.plate} ({basis})"


def _upscale(crop: np.ndarray, target_width: int = TARGET_CROP_WIDTH) -> np.ndarray:
    import cv2

    height, width = crop.shape[:2]
    if width >= target_width:
        return crop
    scale = min(target_width / max(width, 1), 4.0)
    return cv2.resize(
        crop, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC
    )


def _variants(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Preprocessing variants to read independently."""
    import cv2

    out: list[tuple[str, np.ndarray]] = []

    upscaled = _upscale(crop)
    out.append(("raw", upscaled))

    # Contrast-equalised: plates are high-contrast text that CLAHE tends to
    # recover well from shadow or glare.
    grey = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(grey)
    out.append(("contrast", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)))

    # Lower band only — removes windscreen text, roof racks and background.
    height = crop.shape[0]
    band = crop[int(height * (1 - PLATE_BAND)) :, :]
    if band.size and band.shape[0] > 8:
        out.append(("lower_band", _upscale(band)))

    return out


def read_plate(crop: np.ndarray) -> PlateRead | None:
    """Read a registration plate from a vehicle crop, by consensus.

    Returns None when no variant produced anything plate-shaped.
    """
    votes: Counter[str] = Counter()
    scores: dict[str, list[float]] = {}
    all_text: list[str] = []

    variants = _variants(crop)
    for _name, image in variants:
        try:
            regions = read_text(image, min_score=0.3)
        except Exception as exc:  # noqa: BLE001 - one bad variant must not fail the read
            logger.debug("plate variant failed: %s", exc)
            continue

        text = " ".join(r.text for r in regions)
        all_text.append(text)
        mean_score = (sum(r.score for r in regions) / len(regions)) if regions else 0.0

        for extraction in extract_plates(text):
            plate = extraction.entity.value
            votes[plate] += 1
            scores.setdefault(plate, []).append(mean_score)

    if not votes:
        return None

    plate, count = votes.most_common(1)[0]
    ocr_confidence = sum(scores[plate]) / len(scores[plate])

    # Confidence is agreement-weighted: a single read is treated as provisional
    # even when OCR was confident about it, because OCR is confidently wrong about
    # plate characters more often than it is uncertain.
    agreement = count / max(len(variants), 1)
    confidence = min(0.35 + 0.45 * agreement + 0.2 * ocr_confidence, 0.97)

    # Disagreement between variants is a real warning sign, not noise.
    if len(votes) > 1:
        confidence *= 0.7

    return PlateRead(
        plate=plate,
        confidence=round(float(confidence), 3),
        votes=count,
        variants_tried=len(variants),
        candidates=dict(votes),
        raw_text=" | ".join(all_text)[:400],
    )
