"""
ANPR agent — reads registration plates and gives vehicles a real identity.

This matters more than it sounds. Without a plate, a vehicle is only ever "a
white car in this photograph" — there is no way to say two photographs show the
*same* car, because white cars are interchangeable. A plate changes that
completely: it is close to a unique identifier, it is stable across photographs,
cases and years, and it resolves to a registered owner.

So this agent is what turns a vehicle sighting into a **vehicle**. The entity it
creates is keyed on the normalised plate, which means:

* the same car photographed on three different days converges on one node,
* a plate read in this case matches a plate recorded in an old one,
* and "who owns this" becomes a question a registry lookup can answer.

Reading is by consensus across several preprocessing variants of the vehicle
crop. Where the variants disagree, the agent says so and lowers its confidence
rather than picking a winner — a plate is the single value in this system most
likely to be acted on, and a wrong character sends an investigation to the wrong
person.
"""

from __future__ import annotations

from typing import Iterator

from ..detection import detect_objects
from ..plates import MIN_VEHICLE_WIDTH, read_plate
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


class AnprAgent(Agent):
    name = "anpr_agent"
    description = (
        "Read vehicle registration plates from an image and identify the vehicles carrying "
        "them. Finds vehicles, extracts the plate from each, and creates a vehicle identity "
        "keyed on the registration number — so the same car photographed on different days, "
        "or in a different case, resolves to the same entity. "
        "A plate is the strongest identifier available from a photograph: unlike a face it is "
        "designed to be unique, and unlike a description ('white car') it cannot be confused "
        "with another vehicle. Run this on any image containing a vehicle. "
        "Reports honestly when a plate is too small, too angled or too blurred to read. "
        "Keywords: plate, number plate, registration, licence, anpr, vehicle id, car number."
    )
    handles = {EvidenceKind.IMAGE}
    cost = 5

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        import cv2

        image = cv2.imread(evidence.path)
        if image is None:
            raise ValueError(f"Could not read image: {evidence.path}")

        vehicles = [d for d in detect_objects(evidence.path) if d.is_vehicle]
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        if not vehicles:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.85,
                reason=f"No vehicles found in {evidence.filename}, so no plates to read.",
                source=ref,
                payload={"vehicles": 0},
            )
            return

        read_any = False

        for vehicle in vehicles:
            if vehicle.width < MIN_VEHICLE_WIDTH:
                yield self._too_small_claim(evidence, ctx, vehicle)
                continue

            x1, y1, x2, y2 = vehicle.bbox
            crop = image[max(0, y1) : y2, max(0, x1) : x2]
            if crop.size == 0:
                continue

            result = read_plate(crop)
            if result is None:
                yield self._unreadable_claim(evidence, ctx, vehicle)
                continue

            read_any = True
            yield self._plate_claim(evidence, ctx, vehicle, result)

        if not read_any and vehicles:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.8,
                reason=(
                    f"{len(vehicles)} vehicle(s) found in {evidence.filename} but no plate could "
                    f"be read. A closer or straighter photograph of the same vehicle would be "
                    f"worth looking for in the rest of the evidence."
                ),
                source=ref,
                payload={"vehicles": len(vehicles), "plates_read": 0},
            )

    # ---------------------------------------------------------------- claims

    def _plate_claim(self, evidence: Evidence, ctx: AgentContext, vehicle, result) -> Claim:
        plate = Entity(
            type=EntityType.PLATE,
            value=result.plate,
            attributes={
                "read_confidence": result.confidence,
                "votes": result.votes,
                "variants_tried": result.variants_tried,
                "contested": result.contested,
                "alternatives": result.candidates,
            },
        )
        # Keyed on the plate, so every sighting of this registration — in any
        # photograph, in any case — is the same vehicle node.
        vehicle_entity = Entity(
            type=EntityType.VEHICLE,
            value=result.plate,
            attributes={
                "vehicle_type": vehicle.label,
                "identified_by": "registration_plate",
                "bbox": list(vehicle.bbox),
            },
        )

        if result.contested:
            reliability = (
                f"Reads disagreed across preprocessing variants ({result.describe()}), so this "
                f"registration should be confirmed visually before it is acted on."
            )
            risk = RiskLevel.LOW
        elif result.unanimous:
            reliability = (
                f"Every preprocessing variant read the same registration "
                f"({result.votes}/{result.variants_tried}), which is a strong read."
            )
            risk = RiskLevel.MEDIUM
        else:
            reliability = (
                f"{result.votes} of {result.variants_tried} reads produced this registration."
            )
            risk = RiskLevel.LOW

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.ENTITY,
            confidence=result.confidence,
            risk=risk,
            reason=(
                f"Registration '{result.plate}' read from a {vehicle.label} in "
                f"{evidence.filename}. {reliability} This identifies the vehicle uniquely — it "
                f"can be checked against a registry and against vehicles in previous cases."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=vehicle.locator,
                excerpt=evidence.filename,
            ),
            entities=[plate, vehicle_entity],
            relationships=[
                Relationship(
                    subject_id=vehicle_entity.id,
                    predicate=RelationType.MATCHES,
                    object_id=plate.id,
                    attributes={"basis": "plate_recognition"},
                )
            ],
            payload={
                "plate": result.plate,
                "vehicle_type": vehicle.label,
                "consensus": result.describe(),
                "raw_ocr": result.raw_text,
            },
        )

    def _too_small_claim(self, evidence: Evidence, ctx: AgentContext, vehicle) -> Claim:
        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.OBSERVATION,
            confidence=0.75,
            reason=(
                f"A {vehicle.label} in {evidence.filename} is too small to read a plate from "
                f"({vehicle.width}x{vehicle.height}px). Its presence is still recorded — the "
                f"same vehicle may be readable in another photograph."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=vehicle.locator,
                excerpt=evidence.filename,
            ),
            payload={"reason": "vehicle_too_small", "bbox": list(vehicle.bbox)},
        )

    def _unreadable_claim(self, evidence: Evidence, ctx: AgentContext, vehicle) -> Claim:
        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.OBSERVATION,
            confidence=0.7,
            reason=(
                f"A {vehicle.label} in {evidence.filename} is large enough to attempt a plate "
                f"read, but no registration could be recovered — likely angled, obscured, "
                f"blurred, or the plate is not in frame."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=vehicle.locator,
                excerpt=evidence.filename,
            ),
            payload={"reason": "plate_unreadable", "bbox": list(vehicle.bbox)},
        )


registry.register(AnprAgent())
