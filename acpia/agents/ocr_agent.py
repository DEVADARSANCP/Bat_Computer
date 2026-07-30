"""
OCR agent — reads any text visible in an image and pulls entities out of it.

Text in photographs is the bridge between what a camera saw and what the rest of
the case knows. A street sign names a location; a plate on a van identifies a
vehicle; a phone number scrawled on a note connects to a chat account. Once the
text is entities rather than pixels, correlation with everything else is free —
a phone number found here derives the same entity id as the same number found in
a document, so they converge on one node with no matching step.

Runs on CPU (RapidOCR/ONNX) so the GPU stays free for the vision models.

Scope note: this agent reads text from *any* image. Chat screenshots get their
own agent, because understanding a screenshot means understanding its layout —
which region is the contact name, which circle is the profile picture, which
bubbles are messages — and that is a different problem from reading strings.
"""

from __future__ import annotations

from typing import Iterator

from ..extract import extract_all
from ..ocr import full_text, read_text
from ..schema import (
    Claim,
    ClaimKind,
    EntityType,
    Evidence,
    EvidenceKind,
    EvidenceRef,
    RiskLevel,
)
from .base import Agent, AgentContext, registry

#: Longer than this and the "image" is really a document; the excerpt is trimmed
#: so claim summaries stay readable in planner prompts.
EXCERPT_LIMIT = 400


class OcrAgent(Agent):
    name = "ocr_agent"
    description = (
        "Read any text visible in an image and extract structured entities from it: "
        "vehicle registration plates, phone numbers, email addresses, social handles and "
        "URLs. Works on street signs, documents, notes, posters, screenshots and text "
        "written on vehicles or buildings. "
        "Use this on any image that might contain writing. Entities it finds link "
        "automatically to the same entities found elsewhere in the case, so a phone number "
        "read from a photograph will connect to the same number in a chat or a document. "
        "Keywords: text, read, ocr, writing, sign, number, plate, phone, email, document."
    )
    handles = {EvidenceKind.IMAGE, EvidenceKind.DOCUMENT}
    cost = 2

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        regions = read_text(evidence.path)
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        if not regions:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.8,
                reason=f"No readable text found in {evidence.filename}.",
                source=ref,
                payload={"text_regions": 0},
            )
            return

        text = full_text(regions)
        mean_score = sum(r.score for r in regions) / len(regions)

        # --- the text itself ------------------------------------------------
        yield Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.OBSERVATION,
            confidence=round(mean_score, 3),
            reason=(
                f"Recovered {len(regions)} lines of text from {evidence.filename}: "
                f"\"{text[:EXCERPT_LIMIT].replace(chr(10), ' / ')}\""
                + ("…" if len(text) > EXCERPT_LIMIT else "")
            ),
            source=ref,
            payload={
                "text": text,
                "text_regions": len(regions),
                "mean_confidence": round(mean_score, 3),
                "regions": [r.describe() for r in regions[:30]],
            },
        )

        # --- entities found in it -------------------------------------------
        for extraction in extract_all(text):
            yield self._entity_claim(evidence, ctx, extraction, mean_score)

    # ---------------------------------------------------------------- claims

    def _entity_claim(
        self, evidence: Evidence, ctx: AgentContext, extraction, ocr_confidence: float
    ) -> Claim:
        entity = extraction.entity

        # A plate is close to a unique identifier for a vehicle, so it is worth
        # surfacing on sight — it is the value most likely to resolve to a real
        # person through a registry or a previous case.
        risk = RiskLevel.LOW if entity.type is EntityType.PLATE else RiskLevel.NONE

        why = {
            EntityType.PLATE: (
                "A registration number identifies a vehicle uniquely and can be checked "
                "against a registry or previous cases."
            ),
            EntityType.PHONE: (
                "A phone number may connect to a messaging account or contact record "
                "elsewhere in the evidence."
            ),
            EntityType.EMAIL: "An email address may link to an account used elsewhere.",
            EntityType.HANDLE: "An online identifier that may appear in other sources.",
        }.get(entity.type, "")

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.ENTITY,
            # Bounded by how well the text was read: a confidently-shaped plate
            # recovered from unreadable pixels is not a confident plate.
            confidence=round(min(ocr_confidence, 0.95), 3),
            risk=risk,
            reason=(
                f"{entity.type.value.title()} '{entity.value}' read from {evidence.filename} "
                f"(as written: '{extraction.matched_text}'). {why}"
            ).strip(),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=f"text:{extraction.matched_text}",
                excerpt=extraction.context,
            ),
            entities=[entity],
            payload={
                "matched_text": extraction.matched_text,
                "context": extraction.context,
                "ocr_confidence": round(ocr_confidence, 3),
            },
        )


registry.register(OcrAgent())
