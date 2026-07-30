"""
Chat agent — reads messaging screenshots as conversations, not as pictures.

Four things come out of a chat screenshot, and each is a different kind of
evidence:

**The profile picture.** An identity signal that survives a name change. The
same picture under two different contact names is one person presenting as two
identities. The same picture in an older case, filed under a phone number rather
than a name, links the two investigations.

**The contact name.** How the victim labelled that person — often a nickname,
which is exactly what a face or a plate will never tell you.

**What was said.** The victim writing *"that white creta was following me"* or
*"he was waiting outside again"* is frequently the single most important thing
in a case, and nothing in the vision stack will ever find it. Mentions of
people, vehicles, places and concerning behaviour are extracted so those
screenshots surface for a human instead of sitting unread.

**Structured identifiers.** Phone numbers, plates and handles typed into the
conversation, which link to the same entities found anywhere else in the case.

Concerning statements are flagged for **human attention**, never treated as
conclusions. "Following me" in a message is a reason for an investigator to read
that screenshot, not a finding the system is entitled to assert on its own.
"""

from __future__ import annotations

from typing import Iterator

from ..appearance import clip_embedding, crop_region, signature_for, signature_to_bytes
from ..chat import analyse_layout, extract_mentions
from ..extract import extract_all
from ..faces import detect_faces
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

#: Profile pictures are compared as objects, under a reserved label so they are
#: never matched against bags or luggage.
DP_LABEL = "profile_picture"


class ChatAgent(Agent):
    name = "chat_agent"
    description = (
        "Read a messaging app screenshot (WhatsApp, Telegram, Instagram) as a conversation. "
        "Extracts the CONTACT NAME, the PROFILE PICTURE, and the MESSAGE TEXT, then finds who "
        "and what the messages refer to: names and nicknames, vehicle descriptions such as "
        "'white creta', places, and statements suggesting someone was following, waiting for "
        "or pressuring the sender. "
        "Its strongest finding is the SAME PROFILE PICTURE appearing under DIFFERENT contact "
        "names or phone numbers — one person operating as two identities. "
        "Use this on any screenshot of a chat. For plain text in a photograph use ocr_agent "
        "instead. "
        "Keywords: chat, screenshot, whatsapp, message, conversation, contact, profile picture, "
        "dp, mentioned, nickname, said, texting."
    )
    handles = {EvidenceKind.IMAGE}
    cost = 5

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        import cv2

        image = cv2.imread(evidence.path)
        if image is None:
            raise ValueError(f"Could not read image: {evidence.path}")

        layout = analyse_layout(evidence.path, image.shape)
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        if not layout.looks_like_chat:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.6,
                reason=(
                    f"{evidence.filename} does not appear to be a chat screenshot "
                    f"(no contact header found). If it contains text, ocr_agent will read it."
                ),
                source=ref,
                payload={"looks_like_chat": False},
            )
            return

        contact = self._contact_entity(layout.contact_name)

        # --- the conversation itself -----------------------------------------
        yield Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.ENTITY,
            confidence=round(layout.contact_name_confidence, 3),
            reason=(
                f"{evidence.filename} is a conversation with a contact saved as "
                f"'{layout.contact_name}'. {len(layout.body_regions)} message lines recovered."
            ),
            source=ref,
            entities=[contact],
            payload={
                "contact_name": layout.contact_name,
                "message_lines": len(layout.body_regions),
                "message_text": layout.message_text[:2000],
            },
        )

        # --- profile picture --------------------------------------------------
        if layout.dp_bbox:
            yield from self._profile_picture_claims(evidence, ctx, image, layout, contact)

        # --- what the messages refer to ---------------------------------------
        yield from self._mention_claims(evidence, ctx, ref, layout, contact)

        # --- identifiers typed into the conversation ---------------------------
        for extraction in extract_all(layout.message_text):
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.ENTITY,
                confidence=0.85,
                risk=RiskLevel.LOW if extraction.entity.type is EntityType.PLATE else RiskLevel.NONE,
                reason=(
                    f"{extraction.entity.type.value.title()} '{extraction.entity.value}' appears "
                    f"in the conversation with '{layout.contact_name}' — "
                    f"context: {extraction.context}"
                ),
                source=EvidenceRef(
                    evidence_id=evidence.evidence_id,
                    locator=f"message:{extraction.matched_text}",
                    excerpt=extraction.context,
                ),
                entities=[extraction.entity],
                payload={"context": extraction.context},
            )

    # ------------------------------------------------------------- internals

    @staticmethod
    def _contact_entity(name: str) -> Entity:
        return Entity(
            type=EntityType.HANDLE,
            value=name,
            attributes={"kind": "chat_contact", "as_saved": name},
        )

    def _profile_picture_claims(
        self, evidence: Evidence, ctx: AgentContext, image, layout, contact: Entity
    ) -> Iterator[Claim]:
        """Embed the avatar and compare it with every profile picture seen so far."""
        from ..appearance import compare, signature_from_bytes

        signature = signature_for(image, layout.dp_bbox)
        if signature is None:
            return

        gallery = [
            (
                row["object_id"],
                signature_from_bytes(row["signature"], row["width"], row["height"]),
                row["evidence_id"],
            )
            for row in ctx.store.list_object_instances(ctx.case_id, label=DP_LABEL)
        ]

        best_id, best_clip, best_colour = None, 0.0, 0.0
        for object_id, candidate, _ev in gallery:
            _combined, clip_similarity, colour_similarity = compare(signature, candidate)
            if clip_similarity > best_clip:
                best_id, best_clip, best_colour = object_id, clip_similarity, colour_similarity

        # Profile pictures are usually the *identical file* reused, so a genuine
        # match sits very high. The bar is set accordingly — a loose threshold
        # here would link unrelated accounts, which is a serious false lead.
        identical = best_clip >= 0.97 and best_colour >= 0.60

        dp_id = (
            best_id
            if identical
            else Entity.derive_id(EntityType.OBJECT, f"dp:{evidence.evidence_id}")
        )

        ctx.store.add_object_instance(
            case_id=ctx.case_id,
            object_id=dp_id,
            label=DP_LABEL,
            evidence_id=evidence.evidence_id,
            locator=f"dp@{layout.dp_bbox}",
            signature=signature_to_bytes(signature),
            width=signature.width,
            height=signature.height,
            holder_id=contact.value,
        )

        dp_entity = Entity(
            id=dp_id,
            type=EntityType.OBJECT,
            value=f"profile picture {dp_id[-6:]}",
            attributes={"kind": "profile_picture", "used_by": contact.value},
        )

        if identical:
            others = [
                row for row in ctx.store.list_object_instances(ctx.case_id, label=DP_LABEL)
                if row["object_id"] == dp_id and row["evidence_id"] != evidence.evidence_id
            ]
            other_names = sorted({row["holder_id"] for row in others if row["holder_id"]})
            different_name = [n for n in other_names if n != layout.contact_name]

            if different_name:
                yield Claim(
                    case_id=ctx.case_id,
                    agent=self.name,
                    kind=ClaimKind.RELATIONSHIP,
                    confidence=round(min(best_clip, 0.95), 3),
                    risk=RiskLevel.HIGH,
                    reason=(
                        f"The contact '{layout.contact_name}' in {evidence.filename} uses the "
                        f"SAME profile picture as {', '.join(repr(n) for n in different_name)}. "
                        f"One person is presenting under more than one identity, or one account "
                        f"is being operated under different names. "
                        f"(image similarity {best_clip:.3f})"
                    ),
                    source=EvidenceRef(
                        evidence_id=evidence.evidence_id,
                        locator=f"dp@{layout.dp_bbox}",
                        excerpt=evidence.filename,
                    ),
                    entities=[contact, dp_entity],
                    relationships=[
                        Relationship(
                            subject_id=contact.id,
                            predicate=RelationType.MATCHES,
                            object_id=dp_entity.id,
                            attributes={
                                "basis": "shared_profile_picture",
                                "also_used_by": different_name,
                            },
                        )
                    ],
                    payload={
                        "also_used_by": different_name,
                        "clip_similarity": round(best_clip, 3),
                        "colour_similarity": round(best_colour, 3),
                    },
                )
                return

        # A face in the avatar is worth its own entity — it can match a person
        # photographed elsewhere in the case.
        crop = crop_region(image, layout.dp_bbox)
        if crop is not None:
            faces = detect_faces(crop)
            if faces:
                yield Claim(
                    case_id=ctx.case_id,
                    agent=self.name,
                    kind=ClaimKind.ENTITY,
                    confidence=round(float(faces[0].quality), 3),
                    risk=RiskLevel.LOW,
                    reason=(
                        f"The profile picture for '{layout.contact_name}' contains a face. "
                        f"It can be compared against people photographed elsewhere in this case."
                    ),
                    source=EvidenceRef(
                        evidence_id=evidence.evidence_id,
                        locator=f"dp_face@{layout.dp_bbox}",
                        excerpt=evidence.filename,
                    ),
                    entities=[dp_entity],
                    payload={"has_face": True, "det_score": round(faces[0].det_score, 3)},
                )
                return

        yield Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.ENTITY,
            confidence=0.7,
            reason=(
                f"Profile picture recorded for '{layout.contact_name}'. Later screenshots "
                f"will be compared against it."
            ),
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=f"dp@{layout.dp_bbox}",
                excerpt=evidence.filename,
            ),
            entities=[dp_entity],
            payload={"first_sighting": True},
        )

    def _mention_claims(
        self, evidence: Evidence, ctx: AgentContext, ref: EvidenceRef, layout, contact: Entity
    ) -> Iterator[Claim]:
        """Who and what the conversation refers to."""
        mentions = extract_mentions(layout.message_text)
        if mentions.is_empty:
            return

        for name in mentions.people:
            person = Entity(
                type=EntityType.PERSON,
                value=name,
                attributes={"source": "chat_mention", "mentioned_by_contact": layout.contact_name},
            )
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.RELATIONSHIP,
                confidence=0.7,
                reason=(
                    f"'{name}' is referred to in the conversation with '{layout.contact_name}' "
                    f"in {evidence.filename}. A name used in a message may be a nickname and "
                    f"may correspond to a person identified elsewhere by face or vehicle."
                ),
                source=ref,
                entities=[contact, person],
                relationships=[
                    Relationship(
                        subject_id=contact.id,
                        predicate=RelationType.MENTIONS,
                        object_id=person.id,
                        attributes={"extraction_method": mentions.method},
                    )
                ],
                payload={"mentioned_name": name, "method": mentions.method},
            )

        for description in mentions.vehicles:
            vehicle = Entity(
                type=EntityType.VEHICLE,
                value=description,
                attributes={"source": "chat_mention", "described_as": description},
            )
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.ENTITY,
                confidence=0.75,
                risk=RiskLevel.MEDIUM,
                reason=(
                    f"A vehicle is described as '{description}' in the conversation with "
                    f"'{layout.contact_name}'. Compare with vehicles detected in photographs — "
                    f"a described vehicle that matches a photographed one connects what the "
                    f"victim wrote to what the images show."
                ),
                source=ref,
                entities=[vehicle],
                payload={"description": description, "method": mentions.method},
            )

        for place in mentions.locations:
            location = Entity(
                type=EntityType.LOCATION,
                value=place,
                attributes={"source": "chat_mention"},
            )
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.ENTITY,
                confidence=0.65,
                reason=(
                    f"'{place}' is referred to in the conversation with '{layout.contact_name}'. "
                    f"Compare with GPS locations recovered from photograph metadata."
                ),
                source=ref,
                entities=[location],
                payload={"place": place, "method": mentions.method},
            )

        if mentions.concerns:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.7,
                risk=RiskLevel.HIGH,
                reason=(
                    f"The conversation with '{layout.contact_name}' in {evidence.filename} "
                    f"contains statements a human should read: "
                    + " / ".join(f'"{c}"' for c in mentions.concerns[:4])
                    + ". This is flagged for investigator attention — the quoted words are "
                    "reported as written and have not been interpreted."
                ),
                source=ref,
                entities=[contact],
                payload={
                    "concerns": mentions.concerns,
                    "method": mentions.method,
                    "requires_human_review": True,
                },
            )


registry.register(ChatAgent())
