"""
Vision agent — what is physically present in an image.

======================================================================
DETECTABLE OBJECTS — 13 classes
======================================================================

  person

  car        motorcycle    bus       truck      bicycle     <- vehicles
  backpack   handbag       suitcase  umbrella   tie         <- carried items
  cell phone laptop                                         <- devices

======================================================================

Everything else YOLO sees is discarded. See `acpia/detection.py` to add a
class — it is one line, and the full COCO 80 are available.

Detects people, vehicles, carried items and devices, and works out which person
is carrying which item.

Two of these matter more than the rest:

**Vehicles** carry number plates, and a plate is one of the few genuinely unique
identifiers available from a photograph. Every vehicle detected here is a
candidate for the plate agent.

**Carried items** are how you prove contact between two people. An object that
belongs to one person and later appears in another's possession is not an
inference about identity — it is evidence they met. This agent finds and locates
the items; matching a *specific* item across photographs is the appearance
agent's job, because "a backpack" is a class and "*that* backpack" is an
embedding.

The person↔item association is geometry only: does the item sit inside the
person's padded bounding box. That is genuinely weak — a bag on the ground beside
someone looks identical to a bag over their shoulder — so the relationship is
asserted at modest confidence and the reasoning says so.
"""

from __future__ import annotations

from typing import Iterator

from ..detection import Detection, associate_carried_items, detect_objects
from ..schema import (
    Claim,
    ClaimKind,
    Entity,
    EntityType,
    Evidence,
    EvidenceKind,
    EvidenceRef,
    RelationType,
    Relationship,
    RiskLevel,
)
from .base import Agent, AgentContext, registry


class VisionAgent(Agent):
    name = "vision_agent"
    description = (
        "Detect the physical contents of an image: people, vehicles (car, motorcycle, bus, "
        "truck, bicycle), carried items (backpack, handbag, suitcase, umbrella) and devices "
        "(phone, laptop). Reports where each object is in the frame and works out which "
        "person appears to be carrying which item. "
        "Run this on any image before the plate agent (it finds the vehicles) or the "
        "appearance agent (it finds the carried items). It identifies object CLASSES, not "
        "specific instances — it can tell you there is a backpack, not whether it is the same "
        "backpack as one in another photograph. "
        "Keywords: objects, vehicle, car, bag, backpack, carrying, what is in this image, scene."
    )
    handles = {EvidenceKind.IMAGE}
    cost = 3

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        detections = detect_objects(evidence.path)
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        if not detections:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.85,
                reason=f"No people, vehicles or notable objects detected in {evidence.filename}.",
                source=ref,
                payload={"object_count": 0},
            )
            return

        # --- vehicles: the plate-bearing objects ----------------------------
        for detection in (d for d in detections if d.is_vehicle):
            yield self._vehicle_claim(evidence, ctx, detection)

        # --- carried items --------------------------------------------------
        carried = [d for d in detections if d.is_carried]
        for detection in carried:
            yield self._object_claim(evidence, ctx, detection)

        # --- who is carrying what -------------------------------------------
        for person, item in associate_carried_items(detections):
            yield self._carrying_claim(evidence, ctx, person, item)

        # --- scene summary ---------------------------------------------------
        yield self._scene_claim(evidence, ctx, ref, detections)

    # ---------------------------------------------------------------- claims

    def _vehicle_claim(
        self, evidence: Evidence, ctx: AgentContext, detection: Detection
    ) -> Claim:
        # Identity is per-sighting: without a plate we cannot say two cars in two
        # photographs are the same car. The plate agent supplies real identity.
        entity = Entity(
            type=EntityType.VEHICLE,
            value=f"{detection.label} in {evidence.filename}",
            attributes={
                "vehicle_type": detection.label,
                "bbox": list(detection.bbox),
                "detection_confidence": round(detection.confidence, 3),
                "plate_read": False,
            },
        )
        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.ENTITY,
            confidence=round(detection.confidence, 3),
            reason=(
                f"A {detection.label} is visible in {evidence.filename} "
                f"({detection.width}x{detection.height}px). Worth running plate recognition on — "
                f"a registration number would identify this vehicle uniquely, which the shape "
                f"and colour alone cannot."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=detection.locator,
                excerpt=evidence.filename,
            ),
            entities=[entity],
            payload={"detection": detection.describe(), "bbox": list(detection.bbox)},
        )

    def _object_claim(
        self, evidence: Evidence, ctx: AgentContext, detection: Detection
    ) -> Claim:
        entity = Entity(
            type=EntityType.OBJECT,
            value=f"{detection.label} in {evidence.filename}",
            attributes={
                "object_class": detection.label,
                "carryable": True,
                "bbox": list(detection.bbox),
                "detection_confidence": round(detection.confidence, 3),
                "instance_matched": False,
            },
        )
        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.ENTITY,
            confidence=round(detection.confidence, 3),
            reason=(
                f"A {detection.label} is visible in {evidence.filename}. Carried items are "
                f"worth tracking: an object belonging to one person that later appears in "
                f"another's possession is evidence the two were in contact."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=detection.locator,
                excerpt=evidence.filename,
            ),
            entities=[entity],
            payload={"detection": detection.describe(), "bbox": list(detection.bbox)},
        )

    def _carrying_claim(
        self, evidence: Evidence, ctx: AgentContext, person: Detection, item: Detection
    ) -> Claim:
        person_entity = Entity(
            type=EntityType.PERSON,
            value=f"person at {person.bbox[0]},{person.bbox[1]} in {evidence.filename}",
            attributes={"bbox": list(person.bbox), "source": "vision_agent"},
        )
        item_entity = Entity(
            type=EntityType.OBJECT,
            value=f"{item.label} in {evidence.filename}",
            attributes={"object_class": item.label, "bbox": list(item.bbox)},
        )

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.RELATIONSHIP,
            # Deliberately modest. Geometric association cannot distinguish a bag
            # over someone's shoulder from a bag on the ground beside them.
            confidence=0.6,
            reason=(
                f"A person in {evidence.filename} appears to be carrying a {item.label}. "
                f"This is inferred from position alone — the item overlaps the person's "
                f"outline — so it should be confirmed visually before being relied on."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=f"{person.locator}+{item.locator}",
                excerpt=evidence.filename,
            ),
            entities=[person_entity, item_entity],
            relationships=[
                Relationship(
                    subject_id=person_entity.id,
                    predicate=RelationType.CARRYING,
                    object_id=item_entity.id,
                    attributes={"basis": "bounding_box_overlap", "item_class": item.label},
                )
            ],
            payload={"person_bbox": list(person.bbox), "item_bbox": list(item.bbox)},
        )

    def _scene_claim(
        self,
        evidence: Evidence,
        ctx: AgentContext,
        ref: EvidenceRef,
        detections: list[Detection],
    ) -> Claim:
        counts: dict[str, int] = {}
        for detection in detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
        inventory = ", ".join(
            f"{n} {label}{'s' if n > 1 else ''}" for label, n in sorted(counts.items())
        )

        people = sum(1 for d in detections if d.is_person)
        risk = RiskLevel.NONE
        note = ""
        if people > 1 and any(d.is_vehicle for d in detections):
            # Several people around a vehicle is the shape of a meeting or a
            # pickup. Not suspicious in itself, but worth a second look.
            risk = RiskLevel.LOW
            note = (
                " Multiple people are present alongside a vehicle, which is worth reviewing "
                "in the context of the surrounding evidence."
            )

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.OBSERVATION,
            confidence=0.85,
            risk=risk,
            reason=f"{evidence.filename} contains {inventory}.{note}",
            source=ref,
            payload={
                "object_count": len(detections),
                "counts": counts,
                "objects": [d.describe() for d in detections[:20]],
            },
        )


registry.register(VisionAgent())
