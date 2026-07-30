"""
Object detection — the YOLO wrapper shared by the vision and video agents.

Kept separate from the agents for two reasons: the video agent needs the same
detector frame by frame, and on a 6 GB card the model has to be a single shared
instance that can be released when CLIP or InsightFace needs the room.

The class list is deliberately narrow. COCO has 80 classes and most of them
(broccoli, toaster, giraffe) are noise in an investigation. Detecting everything
would bury the two categories that actually matter — **vehicles**, because they
carry plates, and **carried items**, because an object that moves between people
is evidence of contact.

======================================================================
WHAT THIS CAN DETECT — 13 classes
======================================================================

  PEOPLE
    person

  VEHICLES                      (feed the plate agent)
    car            motorcycle          bus
    truck          bicycle

  CARRIED ITEMS                 (feed the appearance agent)
    backpack       handbag             suitcase
    umbrella       tie

  DEVICES
    cell phone     laptop

======================================================================

Anything not on that list is ignored, even if YOLO sees it.

**Adding a class is one line** — put its COCO id in the right group below.
Available and occasionally useful: `bench` (13), `bird` (14), `dog` (16),
`cat` (15), `bottle` (39), `chair` (56), `laptop` (63), `book` (73),
`clock` (74). The full COCO 80 are all detectable by the same model; the
filter is a policy choice, not a capability limit.

**What it does NOT do:** distinguish *instances*. It reports "a backpack",
never "the same backpack as the one in the other photograph". Telling one
specific object from another of the same class is `appearance_agent`'s job,
and it needs an embedding, not a class label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import EntityType

logger = logging.getLogger(__name__)

#: Small model on purpose — plenty for demo footage and kind to 6 GB of VRAM.
#: `yolo11x` exists in the older project folder but the extra accuracy is not
#: worth the memory next to InsightFace and CLIP.
DEFAULT_MODEL = "yolo11s.pt"

#: Below this, detections are noise.
MIN_CONFIDENCE = 0.35

#: COCO ids we care about, mapped to how the investigation should treat them.
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CARRIED_CLASSES = {24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase"}
DEVICE_CLASSES = {63: "laptop", 67: "cell phone"}
PERSON_CLASS = 0

WANTED_CLASSES: dict[int, str] = {
    PERSON_CLASS: "person",
    **VEHICLE_CLASSES,
    **CARRIED_CLASSES,
    **DEVICE_CLASSES,
}

#: A carried item is rarely inside the person's box — a bag hangs below and to
#: the side, a hat sits above. The person box is expanded before testing overlap.
#: Values are fractions of the person's height/width. Lifted from the umbrella
#: association trick in the earlier SEIVR++ pipeline, generalised to any item.
PERSON_BOX_PAD_X = 0.25
PERSON_BOX_PAD_Y = 0.20


_MODEL: Any | None = None


def get_detector(model_name: str = DEFAULT_MODEL) -> Any:
    """Load YOLO once and reuse it. Downloads on first call (~19 MB)."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO

        _MODEL = YOLO(model_name)
        logger.info("YOLO %s ready", model_name)
    return _MODEL


def release_detector() -> None:
    """Drop the model so its VRAM can be reclaimed."""
    global _MODEL
    _MODEL = None


@dataclass
class Detection:
    """One detected object."""

    label: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    locator: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return max(self.width, 0) * max(self.height, 0)

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASSES

    @property
    def is_carried(self) -> bool:
        return self.class_id in CARRIED_CLASSES

    @property
    def entity_type(self) -> EntityType:
        if self.is_person:
            return EntityType.PERSON
        if self.is_vehicle:
            return EntityType.VEHICLE
        if self.class_id in DEVICE_CLASSES:
            return EntityType.DEVICE
        return EntityType.OBJECT

    def describe(self) -> str:
        return f"{self.label} ({self.width}x{self.height}px, conf {self.confidence:.2f})"


def detect_objects(
    image: "Any",
    *,
    min_confidence: float = MIN_CONFIDENCE,
    locator_prefix: str = "",
    model_name: str = DEFAULT_MODEL,
) -> list[Detection]:
    """Detect the classes we care about. Accepts a path or a BGR array."""
    model = get_detector(model_name)
    source = str(image) if isinstance(image, (str, Path)) else image

    results = model.predict(source, verbose=False, conf=min_confidence)
    detections: list[Detection] = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0])
            if class_id not in WANTED_CLASSES:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=WANTED_CLASSES[class_id],
                    class_id=class_id,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    locator=f"{locator_prefix}obj:{index}@{x1},{y1},{x2},{y2}",
                )
            )

    # Largest first — the salient thing in a photograph is usually the big one.
    return sorted(detections, key=lambda d: d.area, reverse=True)


def associate_carried_items(
    detections: list[Detection],
) -> list[tuple[Detection, Detection]]:
    """Pair each carried item with the person most likely holding it.

    Geometry only — overlap between an item and a person's padded bounding box.
    That is a weaker claim than it sounds (two people standing close, a bag on
    the ground beside someone), so the resulting relationship is asserted at
    modest confidence and the planner is expected to corroborate it.

    Returns (person, item) pairs.
    """
    people = [d for d in detections if d.is_person]
    items = [d for d in detections if d.is_carried]
    if not people or not items:
        return []

    pairs: list[tuple[Detection, Detection]] = []

    for item in items:
        best_person, best_overlap = None, 0.0
        ix, iy = item.centre

        for person in people:
            px1, py1, px2, py2 = person.bbox
            pad_x = person.width * PERSON_BOX_PAD_X
            pad_y = person.height * PERSON_BOX_PAD_Y
            ex1, ey1 = px1 - pad_x, py1 - pad_y
            ex2, ey2 = px2 + pad_x, py2 + pad_y

            if not (ex1 <= ix <= ex2 and ey1 <= iy <= ey2):
                continue

            # Prefer the person whose padded box contains the item most centrally,
            # so a bag between two people goes to the nearer one.
            span_x = max(ex2 - ex1, 1)
            span_y = max(ey2 - ey1, 1)
            closeness = 1.0 - (
                abs(ix - (ex1 + ex2) / 2) / span_x + abs(iy - (ey1 + ey2) / 2) / span_y
            )
            if closeness > best_overlap:
                best_person, best_overlap = person, closeness

        if best_person is not None:
            pairs.append((best_person, item))

    return pairs
