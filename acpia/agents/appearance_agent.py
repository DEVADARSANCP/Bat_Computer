"""
Appearance agent — recognises *specific* objects across photographs.

Where the vision agent says "there is a backpack", this one says "that is the
same backpack as the one in file 14". The difference is the whole point: an
object that belongs to one person and later appears in another's possession is
not an inference about identity, it is **evidence the two were in contact**.

======================================================================
WHAT IT MATCHES — carried items only
======================================================================

  backpack    handbag    suitcase    umbrella    tie

======================================================================

Vehicles are excluded on purpose: two silver hatchbacks are visually
interchangeable, and a plate is a far better vehicle identifier than appearance
could ever be. People are excluded because faces do that job properly.

**How matching works, and its honest limits.**

Each sighting gets two signatures — a CLIP embedding (shape, type, texture) and
an HSV colour histogram — and both must agree. CLIP alone is not sufficient: it
encodes *semantics*, so two different black backpacks score high, and a
confident wrong match here would fabricate a link between two people. Colour is
where CLIP is blind and where distinct objects differ most.

The consequence, stated plainly: **a plain, common object cannot be matched
reliably by any method here.** Two black backpacks genuinely are
indistinguishable. When the evidence does not support an instance match, this
agent says so rather than guessing.
"""

from __future__ import annotations

from typing import Iterator

from ..appearance import (
    match_signature,
    signature_for,
    signature_from_bytes,
    signature_to_bytes,
)
from ..detection import CARRIED_CLASSES, associate_carried_items, detect_objects
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

MATCHABLE_LABELS = set(CARRIED_CLASSES.values())


class AppearanceAgent(Agent):
    name = "appearance_agent"
    description = (
        "Identify SPECIFIC carried objects (backpack, handbag, suitcase, umbrella, tie) and "
        "recognise the same physical object across different photographs. This is instance "
        "matching, not classification: it answers 'is this the same bag as the one in that "
        "other photo', which the vision agent cannot. "
        "Its most valuable finding is an object changing hands — an item seen with one person "
        "and later with another is evidence the two were in contact, independent of any face "
        "match. Combines shape/type similarity with colour, and reports both, so a match can "
        "be justified rather than asserted. Plain, common objects cannot be matched reliably "
        "and it will say so. "
        "Keywords: same bag, same object, belongs to, changed hands, instance, appearance, item."
    )
    handles = {EvidenceKind.IMAGE}
    cost = 6

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        import cv2

        image = cv2.imread(evidence.path)
        if image is None:
            raise ValueError(f"Could not read image: {evidence.path}")

        detections = detect_objects(evidence.path)
        items = [d for d in detections if d.label in MATCHABLE_LABELS]
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        if not items:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.85,
                reason=f"No carried items found in {evidence.filename} to match.",
                source=ref,
                payload={"items": 0},
            )
            return

        # Who is holding what, so a match can be reported as a transfer.
        holders = {id(item): person for person, item in associate_carried_items(detections)}

        for item in items:
            signature = signature_for(image, item.bbox)
            if signature is None or signature.is_small:
                yield Claim(
                    case_id=ctx.case_id,
                    agent=self.name,
                    kind=ClaimKind.OBSERVATION,
                    confidence=0.5,
                    reason=(
                        f"A {item.label} in {evidence.filename} is too small "
                        f"({item.width}x{item.height}px) for reliable instance matching."
                    ),
                    source=EvidenceRef(
                        evidence_id=evidence.evidence_id, locator=item.locator,
                        excerpt=evidence.filename,
                    ),
                    payload={"label": item.label, "bbox": list(item.bbox)},
                )
                continue

            gallery = [
                (row["object_id"], signature_from_bytes(row["signature"], row["width"], row["height"]))
                for row in ctx.store.list_object_instances(ctx.case_id, label=item.label)
            ]
            match = match_signature(signature, gallery)

            object_id = (
                match.identity
                if (match.matched and match.strong)
                else Entity.derive_id(
                    EntityType.OBJECT, f"object:{evidence.evidence_id}:{item.locator}"
                )
            )

            holder = holders.get(id(item))
            ctx.store.add_object_instance(
                case_id=ctx.case_id,
                object_id=object_id,
                label=item.label,
                evidence_id=evidence.evidence_id,
                locator=item.locator,
                signature=signature_to_bytes(signature),
                width=signature.width,
                height=signature.height,
                holder_id=f"{holder.bbox}" if holder else None,
            )

            yield self._sighting_claim(evidence, ctx, item, object_id, match, holder)

            if match.matched:
                transfer = self._transfer_claim(evidence, ctx, item, object_id, match)
                if transfer:
                    yield transfer

    # ---------------------------------------------------------------- claims

    def _sighting_claim(self, evidence, ctx, item, object_id, match, holder) -> Claim:
        entity = Entity(
            id=object_id,
            type=EntityType.OBJECT,
            value=f"{item.label} ({object_id[-6:]})",
            attributes={
                "object_class": item.label,
                "instance_matched": match.matched and match.strong,
                "bbox": list(item.bbox),
                "clip_similarity": round(match.clip_similarity, 3),
                "colour_similarity": round(match.colour_similarity, 3),
            },
        )

        entities = [entity]
        relationships = []
        if holder is not None:
            person = Entity(
                type=EntityType.PERSON,
                value=f"person at {holder.bbox[0]},{holder.bbox[1]} in {evidence.filename}",
                attributes={"bbox": list(holder.bbox), "source": "appearance_agent"},
            )
            entities.append(person)
            relationships.append(
                Relationship(
                    subject_id=person.id,
                    predicate=RelationType.CARRYING,
                    object_id=entity.id,
                    attributes={"item_class": item.label},
                )
            )

        if match.matched and match.strong:
            reason = (
                f"The {item.label} in {evidence.filename} is the SAME physical object as one "
                f"already seen in this case — {match.describe()}."
            )
            confidence = min(match.combined, 0.95)
        elif match.matched:
            reason = (
                f"The {item.label} in {evidence.filename} resembles one seen elsewhere in this "
                f"case but not closely enough to call it the same object — {match.describe()}. "
                f"Treat as a possibility requiring corroboration, not a link."
            )
            confidence = round(match.combined * 0.7, 3)
        else:
            reason = (
                f"A {item.label} in {evidence.filename} does not match any object seen so far. "
                f"Recorded so later sightings can be compared against it."
            )
            confidence = 0.7

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.RELATIONSHIP if relationships else ClaimKind.ENTITY,
            confidence=float(confidence),
            reason=reason,
            source=EvidenceRef(
                evidence_id=evidence.evidence_id, locator=item.locator, excerpt=evidence.filename
            ),
            entities=entities,
            relationships=relationships,
            payload={
                "object_id": object_id,
                "label": item.label,
                "match": match.describe(),
                "clip_similarity": round(match.clip_similarity, 3),
                "colour_similarity": round(match.colour_similarity, 3),
            },
        )

    def _transfer_claim(self, evidence, ctx, item, object_id, match) -> Claim | None:
        """Flag an object that has been seen in more than one place.

        Where and with whom the object appeared is left to the planner and the
        graph — this agent knows the object recurred, not who owned it. Saying
        "changed hands" outright would overclaim; saying nothing would waste the
        most important finding the agent can produce.
        """
        if not (match.matched and match.strong):
            return None

        sightings = ctx.store.object_id_sightings(ctx.case_id, object_id)
        distinct_evidence = {s["evidence_id"] for s in sightings}
        if len(distinct_evidence) < 2:
            return None

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.PATTERN,
            confidence=round(min(match.combined, 0.9), 3),
            risk=RiskLevel.MEDIUM,
            reason=(
                f"The same {item.label} now appears in {len(distinct_evidence)} separate pieces "
                f"of evidence. If it is associated with different people in different files, "
                f"that is evidence of contact between them — check who is carrying it in each. "
                f"Match quality: {match.describe()}."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id, locator=item.locator, excerpt=evidence.filename
            ),
            payload={
                "object_id": object_id,
                "label": item.label,
                "sighting_count": len(sightings),
                "evidence_ids": sorted(distinct_evidence),
            },
        )


registry.register(AppearanceAgent())
