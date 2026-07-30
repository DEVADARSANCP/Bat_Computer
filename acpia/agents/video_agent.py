"""
Video agent — who and what appears in footage, and what happens.

Runs the whole perception stack over sampled frames: detect and track objects,
then identify the people by face and the vehicles by plate, and finally turn
trajectories into events.

The important design choice is that **tracker ids are not identities**. ByteTrack
gives an id that survives within one clip and breaks on occlusion; it means
nothing across two clips. So a track here is only a grouping of detections, and
who it actually is comes from running face recognition on the track's best frame
against the same case-wide gallery the image agents use.

That is what makes the centrepiece capability work: a person appearing in two
separate videos, or in a video and a photograph, resolves to one identity — not
because a tracker followed them, but because their face matched.

Events are stated at the confidence they deserve. "A person reaches a car and is
not seen again" is reported as *consistent with entering the vehicle*, because a
person walking behind a van produces an identical signature and an investigator
should be told which one this is.
"""

from __future__ import annotations

from typing import Iterator

from ..faces import detect_faces, embedding_from_bytes, embedding_to_bytes, match_embedding
from ..faces import FACE_MATCH_STRONG, FACE_MATCH_WEAK
from ..persons import PersonRegister, PersonStatus
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
from ..video import Track, collect_tracks, detect_events
from .base import Agent, AgentContext, registry


class VideoAgent(Agent):
    name = "video_agent"
    description = (
        "Analyse a video: detect and follow people and vehicles across frames, identify who "
        "the people are by FACE (not by tracker id, so the same person is recognised in other "
        "videos and photographs too), read plates from the vehicles, and report what happened "
        "and when. "
        "Produces a timeline of events with timestamps — when someone appears, when they are "
        "at a vehicle, and when someone reaches a vehicle and is not seen again. "
        "Use this on any video file. It samples frames rather than processing every one, so a "
        "short clip takes seconds rather than minutes. "
        "Keywords: video, footage, cctv, clip, track, follow, movement, entered car, timeline, "
        "when did, who is in the video."
    )
    handles = {EvidenceKind.VIDEO}
    cost = 9

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        tracks, info, frames = collect_tracks(evidence.path)
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        yield Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.OBSERVATION,
            confidence=0.95,
            reason=(
                f"{evidence.filename} is {info.duration:.1f}s of video at {info.fps:.0f}fps "
                f"({info.width}x{info.height}). Sampled {info.sampled} frames and followed "
                f"{len(tracks)} distinct objects."
            ),
            source=ref,
            payload={
                "duration_s": round(info.duration, 2),
                "fps": round(info.fps, 2),
                "frames_sampled": info.sampled,
                "tracks": len(tracks),
            },
        )

        if not tracks:
            return

        register = PersonRegister(ctx.store, ctx.case_id)
        identities: dict[int, str] = {}

        for track in tracks.values():
            frame = frames.get(track.track_id)
            if frame is None:
                continue

            if track.label == "person":
                claim, label = self._identify_person(evidence, ctx, track, frame, register)
                if claim:
                    yield claim
                if label:
                    identities[track.track_id] = label

            elif track.label in {"car", "motorcycle", "bus", "truck", "bicycle"}:
                claim, label = self._read_vehicle(evidence, ctx, track, frame)
                if claim:
                    yield claim
                if label:
                    identities[track.track_id] = label

        for event in detect_events(tracks, frame_size=(info.width, info.height)):
            yield self._event_claim(evidence, ctx, event, identities)

    # ------------------------------------------------------------- identity

    def _identify_person(
        self, evidence: Evidence, ctx: AgentContext, track: Track, frame, register
    ) -> tuple[Claim | None, str | None]:
        """Put a name (or a stable id) to a tracked person using their best frame."""
        best = track.best_detection()
        x1, y1, x2, y2 = best.bbox
        crop = frame[max(0, y1) : y2, max(0, x1) : x2]
        if crop.size == 0:
            return None, None

        faces = detect_faces(crop)
        locator = f"t={best.timestamp:.1f}s track={track.track_id}"

        if not faces:
            return (
                Claim(
                    case_id=ctx.case_id,
                    agent=self.name,
                    kind=ClaimKind.OBSERVATION,
                    confidence=0.6,
                    risk=RiskLevel.LOW,
                    reason=(
                        f"A person appears in {evidence.filename} ({track.timespan()}) but no "
                        f"face is visible, so they cannot be identified from this footage. "
                        f"Other signals — clothing, a carried item, an associated vehicle, or "
                        f"the time and place — may still establish who this is."
                    ),
                    source=EvidenceRef(
                        evidence_id=evidence.evidence_id, locator=locator,
                        excerpt=evidence.filename,
                    ),
                    payload={
                        "track_id": track.track_id,
                        "timespan": track.timespan(),
                        "face_visible": False,
                    },
                ),
                None,
            )

        face = max(faces, key=lambda f: f.det_score)

        gallery = [
            (row["face_id"], embedding_from_bytes(row["embedding"]))
            for row in ctx.store.list_face_instances(ctx.case_id)
        ]
        gallery_match = match_embedding(
            face.embedding, gallery,
            weak_threshold=FACE_MATCH_WEAK, strong_threshold=FACE_MATCH_STRONG,
        )
        identification = register.identify(face.embedding)

        face_id = (
            gallery_match.identity
            if (gallery_match.matched and gallery_match.strong)
            else Entity.derive_id(
                EntityType.PERSON, f"face:{evidence.evidence_id}:{locator}"
            )
        )

        ctx.store.add_face_instance(
            case_id=ctx.case_id,
            face_id=face_id,
            evidence_id=evidence.evidence_id,
            locator=locator,
            embedding=embedding_to_bytes(face.embedding),
            det_score=face.det_score,
            quality=face.quality,
            is_small=face.is_small,
        )

        display = identification.name or f"unidentified person {face_id[-6:]}"
        entity = Entity(
            id=face_id,
            type=EntityType.PERSON,
            value=display,
            attributes={
                "register_status": identification.status.value,
                "register_name": identification.name,
                "seen_in_video": evidence.filename,
                "timespan": track.timespan(),
            },
        )

        risk = RiskLevel.NONE
        if identification.status is PersonStatus.PERSON_OF_INTEREST:
            risk = RiskLevel.HIGH

        detail = ""
        if gallery_match.matched and gallery_match.strong:
            detail = (
                f" This is the same person seen elsewhere in the case "
                f"(similarity {gallery_match.similarity:.2f})."
            )
        elif gallery_match.matched:
            detail = (
                f" Possibly the same person as one seen elsewhere "
                f"(similarity {gallery_match.similarity:.2f}) — too weak to assert alone."
            )

        return (
            Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.ENTITY,
                confidence=round(
                    min(identification.similarity + 0.25, 0.95)
                    if identification.identified
                    else float(face.quality),
                    3,
                ),
                risk=risk,
                reason=(
                    f"{identification.describe() if identification.identified else 'An unidentified person'} "
                    f"appears in {evidence.filename} from {track.timespan()}.{detail}"
                ),
                source=EvidenceRef(
                    evidence_id=evidence.evidence_id, locator=locator, excerpt=evidence.filename
                ),
                entities=[entity],
                payload={
                    "track_id": track.track_id,
                    "timespan": track.timespan(),
                    "identification": identification.as_dict(),
                    "gallery_similarity": round(gallery_match.similarity, 3),
                },
            ),
            display,
        )

    def _read_vehicle(
        self, evidence: Evidence, ctx: AgentContext, track: Track, frame
    ) -> tuple[Claim | None, str | None]:
        """Try to read a plate from a tracked vehicle's clearest frame."""
        best = track.best_detection()
        x1, y1, x2, y2 = best.bbox
        crop = frame[max(0, y1) : y2, max(0, x1) : x2]
        locator = f"t={best.timestamp:.1f}s track={track.track_id}"

        if crop.size == 0 or (x2 - x1) < MIN_VEHICLE_WIDTH:
            return None, None

        result = read_plate(crop)
        if result is None:
            return (
                Claim(
                    case_id=ctx.case_id,
                    agent=self.name,
                    kind=ClaimKind.OBSERVATION,
                    confidence=0.65,
                    reason=(
                        f"A {track.label} appears in {evidence.filename} ({track.timespan()}) "
                        f"but no registration could be read from it."
                    ),
                    source=EvidenceRef(
                        evidence_id=evidence.evidence_id, locator=locator,
                        excerpt=evidence.filename,
                    ),
                    payload={"track_id": track.track_id, "plate_read": False},
                ),
                None,
            )

        plate = Entity(type=EntityType.PLATE, value=result.plate)
        vehicle = Entity(
            type=EntityType.VEHICLE,
            value=result.plate,
            attributes={"vehicle_type": track.label, "identified_by": "registration_plate"},
        )

        return (
            Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.ENTITY,
                confidence=result.confidence,
                risk=RiskLevel.MEDIUM,
                reason=(
                    f"Registration '{result.plate}' read from a {track.label} in "
                    f"{evidence.filename} at {track.timespan()} ({result.describe()}). "
                    f"This identifies the vehicle across cases and photographs."
                ),
                source=EvidenceRef(
                    evidence_id=evidence.evidence_id, locator=locator, excerpt=evidence.filename
                ),
                entities=[plate, vehicle],
                relationships=[
                    Relationship(
                        subject_id=vehicle.id,
                        predicate=RelationType.MATCHES,
                        object_id=plate.id,
                        attributes={"basis": "plate_recognition", "source": "video"},
                    )
                ],
                payload={
                    "track_id": track.track_id,
                    "plate": result.plate,
                    "consensus": result.describe(),
                },
            ),
            result.plate,
        )

    # --------------------------------------------------------------- events

    def _event_claim(self, evidence: Evidence, ctx: AgentContext, event, identities) -> Claim:
        who = identities.get(event.person_track, "an unidentified person")
        what = identities.get(event.vehicle_track, "a vehicle")

        if event.kind == "person_entered_vehicle":
            risk = RiskLevel.HIGH
            reason = (
                f"At {event.clock} in {evidence.filename}, {who} reaches {what} and is not seen "
                f"again in the footage. This is consistent with entering the vehicle, but a "
                f"person passing behind it would look the same — confirm visually before "
                f"relying on it."
            )
        elif event.kind == "person_at_vehicle":
            risk = RiskLevel.MEDIUM
            reason = f"At {event.clock} in {evidence.filename}, {who} is at {what}."
        else:
            risk = RiskLevel.NONE
            reason = f"In {evidence.filename}: {event.description} ({who})."

        return Claim(
            case_id=ctx.case_id,
            agent=self.name,
            kind=ClaimKind.EVENT,
            confidence=event.confidence,
            risk=risk,
            reason=reason,
            source=EvidenceRef(
                evidence_id=evidence.evidence_id,
                locator=f"t={event.timestamp:.1f}s",
                excerpt=evidence.filename,
            ),
            payload={
                "event": event.kind,
                "timestamp_s": round(event.timestamp, 2),
                "clock": event.clock,
                "person": who,
                "vehicle": what,
            },
        )


registry.register(VideoAgent())
