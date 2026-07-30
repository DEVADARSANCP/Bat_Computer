"""
Face agent — finds people, works out who they are, and tracks identities across
a case.

Three jobs, in order:

1. **Detect and embed** every face in an image, including small ones in the
   background. This is deliberate: the highest-value image in a case is often a
   victim's own selfie with a stranger out of focus behind them. Nobody looking
   at that photograph notices. The system is supposed to.

2. **Assign a stable identity.** Each new face is matched against every face
   already seen in the case. A match inherits that `face_id`; otherwise a new one
   is created. This is what makes "the same person appears in five photographs"
   a fact the system can state, and it is what feeds the recurrence trigger.

3. **Check the register.** If the face matches someone the investigator enrolled,
   the claim says so — victim, known associate, or person of interest. That is
   what stops the watchlist filling up with family.

The confidence on every claim is tempered by face quality. A 40-pixel background
face produces a real claim with honest low confidence, not a confident assertion
or a silent discard.
"""

from __future__ import annotations

from typing import Iterator

from ..faces import (
    FACE_MATCH_STRONG,
    FACE_MATCH_WEAK,
    DetectedFace,
    detect_faces,
    embedding_from_bytes,
    embedding_to_bytes,
    match_embedding,
)
from ..persons import PersonRegister, PersonStatus
from ..schema import (
    Claim,
    ClaimKind,
    Entity,
    EntityType,
    Evidence,
    EvidenceKind,
    EvidenceRef,
    RiskLevel,
)
from .base import Agent, AgentContext, registry


class FaceAgent(Agent):
    name = "face_agent"
    description = (
        "Detect every human face in an image, generate a face embedding for each, and "
        "determine WHO it is. Assigns each face a stable identity so the same person can be "
        "recognised across different photographs, and checks every face against the people "
        "the investigator has enrolled (victim, known associates, persons of interest). "
        "Detects small and out-of-focus background faces as well as clear foreground ones — "
        "a bystander nobody noticed is often the most important thing in a photograph. "
        "Reports an honest confidence: a tiny background face produces a low-confidence claim, "
        "not a guess. "
        "Keywords: face, person, who, identity, recognise, victim, suspect, people, bystander."
    )
    handles = {EvidenceKind.IMAGE}
    cost = 4

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        faces = detect_faces(evidence.path)
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        if not faces:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.9,
                reason=f"No human faces detected in {evidence.filename}.",
                source=ref,
                payload={"face_count": 0},
            )
            return

        register = PersonRegister(ctx.store, ctx.case_id)
        gallery = self._load_gallery(ctx)

        for face in faces:
            face_id, gallery_match = self._resolve_identity(face, gallery, ctx, evidence)
            identification = register.identify(face.embedding)

            # Keep the new instance in the in-memory gallery so two faces in the
            # SAME image are compared against each other, not just against
            # earlier files.
            gallery.append((face_id, face.embedding))

            yield self._build_claim(
                evidence=evidence,
                ctx=ctx,
                ref=ref,
                face=face,
                face_id=face_id,
                identification=identification,
                gallery_similarity=gallery_match,
            )

        if len(faces) > 1:
            yield self._co_presence_claim(evidence, ctx, ref, faces)

    # ------------------------------------------------------------- identity

    def _load_gallery(self, ctx: AgentContext) -> list[tuple[str, "object"]]:
        return [
            (row["face_id"], embedding_from_bytes(row["embedding"]))
            for row in ctx.store.list_face_instances(ctx.case_id)
        ]

    def _resolve_identity(
        self,
        face: DetectedFace,
        gallery: list[tuple[str, "object"]],
        ctx: AgentContext,
        evidence: Evidence,
    ) -> tuple[str, float]:
        """Give this face a stable id — an existing one if it matches, else a new one."""
        match = match_embedding(
            face.embedding,
            gallery,
            weak_threshold=FACE_MATCH_WEAK,
            strong_threshold=FACE_MATCH_STRONG,
        )

        if match.matched and match.strong:
            face_id = match.identity
        elif match.matched:
            # Weak match: same person is plausible but not established. Creating a
            # separate identity is the safer error — two ids for one person can be
            # merged later, whereas silently merging two people is a false link
            # that could send an investigation the wrong way.
            face_id = Entity.derive_id(
                EntityType.PERSON, f"face:{evidence.evidence_id}:{face.locator}"
            )
        else:
            face_id = Entity.derive_id(
                EntityType.PERSON, f"face:{evidence.evidence_id}:{face.locator}"
            )

        ctx.store.add_face_instance(
            case_id=ctx.case_id,
            face_id=face_id,
            evidence_id=evidence.evidence_id,
            locator=face.locator,
            embedding=embedding_to_bytes(face.embedding),
            det_score=face.det_score,
            quality=face.quality,
            is_small=face.is_small,
        )
        return face_id, match.similarity

    # --------------------------------------------------------------- claims

    def _build_claim(
        self,
        *,
        evidence: Evidence,
        ctx: AgentContext,
        ref: EvidenceRef,
        face: DetectedFace,
        face_id: str,
        identification,
        gallery_similarity: float,
    ) -> Claim:
        display = identification.name or f"unidentified person {face_id[-6:]}"

        entity = Entity(
            id=face_id,
            type=EntityType.PERSON,
            value=display,
            attributes={
                "register_status": identification.status.value,
                "register_name": identification.name,
                "relationship": identification.relationship,
                "match_confidence": round(identification.similarity, 3),
                "bbox": list(face.bbox),
                "det_score": round(face.det_score, 3),
                "quality": face.quality,
                "background_face": face.is_small,
                **face.attributes,
            },
        )

        # Confidence reflects what we actually know. An identified face inherits
        # the register match; an unidentified one is only as good as the detection.
        confidence = (
            min(identification.similarity + 0.25, 0.97)
            if identification.identified
            else face.quality
        )

        risk = RiskLevel.NONE
        if identification.status is PersonStatus.PERSON_OF_INTEREST:
            risk = RiskLevel.HIGH
        elif identification.status is PersonStatus.UNKNOWN and face.is_small:
            # An unnoticed background face is not itself alarming, but it is the
            # kind of thing that gets missed, so it is worth surfacing.
            risk = RiskLevel.LOW

        reason = self._reason(evidence.filename, face, identification, gallery_similarity)

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.ENTITY,
            confidence=round(float(confidence), 3),
            risk=risk,
            reason=reason,
            source=EvidenceRef(
                evidence_id=ref.evidence_id, locator=face.locator, excerpt=evidence.filename
            ),
            entities=[entity],
            payload={
                "face_id": face_id,
                "identification": identification.as_dict(),
                "gallery_similarity": round(gallery_similarity, 3),
                "geometry": face.describe(),
            },
        )

    @staticmethod
    def _reason(
        filename: str, face: DetectedFace, identification, gallery_similarity: float
    ) -> str:
        where = "in the background of" if face.is_small else "in"

        if identification.identified:
            body = f"{identification.describe()} appears {where} {filename}"
        else:
            body = f"An unidentified person appears {where} {filename}"

        if face.is_small:
            body += (
                ". This face is small and easily overlooked by a human reviewer, so the "
                "identification is tentative"
            )
        if gallery_similarity >= FACE_MATCH_STRONG:
            body += (
                f". Matches a person already seen elsewhere in this case "
                f"(similarity {gallery_similarity:.2f})"
            )
        elif gallery_similarity >= FACE_MATCH_WEAK:
            body += (
                f". Possibly the same person as one seen elsewhere in this case "
                f"(similarity {gallery_similarity:.2f}) — too weak to assert without "
                f"corroboration"
            )
        return body + "."

    @staticmethod
    def _co_presence_claim(
        evidence: Evidence, ctx: AgentContext, ref: EvidenceRef, faces: list[DetectedFace]
    ) -> Claim:
        """Record that several people were physically together in one frame.

        Co-presence is the raw material of the "victim and unidentified subject
        were metres apart" finding, so it is asserted explicitly rather than left
        for the planner to infer from two separate claims.
        """
        return Claim(
            case_id=ctx.case_id,
            agent="face_agent",
            kind=ClaimKind.OBSERVATION,
            confidence=0.85,
            risk=RiskLevel.LOW,
            reason=(
                f"{len(faces)} people appear together in {evidence.filename}. Everyone in this "
                f"frame was in the same place at the same moment."
            ),
            source=ref,
            payload={
                "face_count": len(faces),
                "faces": [
                    {"locator": f.locator, "geometry": f.describe(), "quality": f.quality}
                    for f in faces
                ],
            },
        )


registry.register(FaceAgent())
