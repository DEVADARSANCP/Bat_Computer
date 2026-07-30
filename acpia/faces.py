"""
Face detection, embedding, and identity matching.

Wraps InsightFace (buffalo_l) and owns every decision about when two faces are
the same person. Both the face agent and the person register go through here, so
there is exactly one definition of "match" in the system.

Three things worth knowing about the design:

**Low-quality faces are kept, not filtered.** The instinct is to discard small,
blurry, background faces. That would be wrong here — the single most valuable
image in a case can be a victim's selfie with a stranger out of focus behind
them. Nobody noticed that face; the system is supposed to. So small faces are
detected, embedded, and reported with reduced confidence rather than dropped.

**Thresholds are constants in one place, and they are calibrated, not guessed.**
`acpia calibrate-faces` measures the actual score distribution on real data.
The defaults below are a starting point, not a claim.

**The model is loaded once.** buffalo_l is ~275 MB and takes seconds to
initialise; reloading it per image would make video analysis unusable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
#
# Cosine similarity between L2-normalised ArcFace embeddings, so the range is
# [-1, 1] and same-person pairs sit far above different-person pairs.
#
# MEASURED, not guessed. Sampled over 15 faces across 9 images:
#
#     different people   n=21   p50 0.033   p95 0.165   max 0.188
#     same face          n=14   p50 0.759   p95 0.977   max 0.980
#
# Different-person scores never exceeded 0.19, so both thresholds below sit in a
# wide empty band. Caveat worth remembering: the same-person figures come from
# rescaled copies of one photograph, which is optimistic — the same person shot
# on different days, in different light, at an angle will score materially
# lower. That is why the weak threshold is set well below the observed
# same-person range rather than just above the different-person one.
#
# Re-measure on real footage with `python -m acpia.cli calibrate-faces <dir>`.

#: Same person, confident enough to MERGE two sightings into one identity.
#: A false merge invents a link between two people, so this stays strict.
FACE_MATCH_STRONG = 0.50

#: Possibly the same person — reported as a hypothesis, never merged. Set low
#: because a missed match costs a real lead, while a spurious "possible" costs
#: only a sentence the planner must corroborate before acting on.
FACE_MATCH_WEAK = 0.30

#: Minimum detector confidence to accept a face at all. Low on purpose: the
#: background-face case matters more than a few false positives, which downstream
#: matching will discard anyway.
MIN_DET_SCORE = 0.35

#: Faces smaller than this (max side, pixels) are embedded but flagged as
#: low quality — ArcFace degrades on tiny crops.
SMALL_FACE_PX = 60

#: Detector input size. 640 is the InsightFace default and a reasonable
#: speed/recall trade-off on a 6 GB GPU.
DET_SIZE = (640, 640)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

_ANALYSER: Any | None = None


def get_face_analyser(*, force_cpu: bool = False) -> Any:
    """Load buffalo_l once and reuse it.

    Falls back to CPU if CUDA is unavailable. On first run this downloads
    ~275 MB into ~/.insightface.
    """
    global _ANALYSER
    if _ANALYSER is not None:
        return _ANALYSER

    # InsightFace and onnxruntime are noisy on stdout/stderr at import time.
    os.environ.setdefault("GLOG_minloglevel", "3")
    from insightface.app import FaceAnalysis

    providers = (
        ["CPUExecutionProvider"]
        if force_cpu
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    analyser = FaceAnalysis(name="buffalo_l", providers=providers)
    analyser.prepare(ctx_id=-1 if force_cpu else 0, det_size=DET_SIZE)
    _ANALYSER = analyser
    logger.info("InsightFace buffalo_l ready (providers=%s)", providers)
    return analyser


def release_face_analyser() -> None:
    """Drop the model so its VRAM can be reclaimed.

    Matters on a 6 GB card where YOLO and CLIP also need room.
    """
    global _ANALYSER
    _ANALYSER = None


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@dataclass
class DetectedFace:
    """One face found in one image or frame."""

    bbox: tuple[int, int, int, int]
    embedding: np.ndarray  # L2-normalised, 512-d
    det_score: float
    locator: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def is_small(self) -> bool:
        return max(self.width, self.height) < SMALL_FACE_PX

    @property
    def quality(self) -> float:
        """Rough 0-1 usability score, combining detector confidence and size.

        Used to temper claim confidence — a face this small is still reported,
        but the system should not stake an identity on it alone.
        """
        size_factor = min(max(self.width, self.height) / 112.0, 1.0)
        return round(float(self.det_score) * (0.5 + 0.5 * size_factor), 3)

    def describe(self) -> str:
        note = " (small/background)" if self.is_small else ""
        return f"{self.width}x{self.height}px det={self.det_score:.2f}{note}"


def detect_faces(image: "np.ndarray | str | Path", *, locator_prefix: str = "") -> list[DetectedFace]:
    """Find and embed every face in an image.

    Accepts a path or an already-loaded BGR array (so video frames can be passed
    without a round trip through disk).
    """
    import cv2

    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        img = image

    analyser = get_face_analyser()
    found = []

    for index, face in enumerate(analyser.get(img)):
        if float(face.det_score) < MIN_DET_SCORE:
            continue

        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        x1, y1, x2, y2 = (int(v) for v in face.bbox)

        detected = DetectedFace(
            bbox=(x1, y1, x2, y2),
            embedding=embedding,
            det_score=float(face.det_score),
            locator=f"{locator_prefix}face:{index}@{x1},{y1},{x2},{y2}",
        )

        # Demographic estimates are recorded as weak context only. They are not
        # reliable enough to drive any decision and must never gate a match.
        if getattr(face, "age", None) is not None:
            detected.attributes["estimated_age"] = int(face.age)
        if getattr(face, "sex", None) is not None:
            detected.attributes["estimated_sex"] = str(face.sex)

        found.append(detected)

    return found


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings.

    InsightFace returns L2-normalised vectors, so this is a dot product — but
    the normalisation is done defensively in case a caller passes raw output.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class FaceMatch:
    """Result of comparing a face against a set of known embeddings."""

    identity: str | None
    similarity: float
    strong: bool
    label: str = ""

    @property
    def matched(self) -> bool:
        return self.identity is not None

    def describe(self) -> str:
        if not self.matched:
            return f"no match (best {self.similarity:.3f})"
        strength = "strong" if self.strong else "possible"
        return f"{self.label or self.identity} ({strength}, {self.similarity:.3f})"


def match_embedding(
    embedding: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
    *,
    weak_threshold: float = FACE_MATCH_WEAK,
    strong_threshold: float = FACE_MATCH_STRONG,
    labels: dict[str, str] | None = None,
) -> FaceMatch:
    """Best match for `embedding` among `candidates` [(identity, embedding), ...].

    Returns the best candidate above `weak_threshold`, flagging whether it also
    cleared `strong_threshold`. A weak match is a hypothesis for the planner to
    corroborate, not an assertion.
    """
    best_identity: str | None = None
    best_score = -1.0

    for identity, candidate in candidates:
        score = cosine_similarity(embedding, candidate)
        if score > best_score:
            best_score, best_identity = score, identity

    if best_identity is None or best_score < weak_threshold:
        return FaceMatch(identity=None, similarity=max(best_score, 0.0), strong=False)

    return FaceMatch(
        identity=best_identity,
        similarity=best_score,
        strong=best_score >= strong_threshold,
        label=(labels or {}).get(best_identity, ""),
    )


# --------------------------------------------------------------------------
# Serialisation — embeddings live in SQLite as raw float32 bytes
# --------------------------------------------------------------------------


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def embedding_from_bytes(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)
