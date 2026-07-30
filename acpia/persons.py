"""
The person register — who the investigator already knows about.

Solves a real flaw in frequency-based triage: the faces that recur most in a
victim's photographs are family, friends and classmates. Promoting people by how
often they appear is biased toward the innocent, and buries the person who
matters among a dozen relatives.

Enrolment fixes that by letting a human say, up front, who is already accounted
for.

**Known associates are labelled, never deleted.** In child protection the
offender is frequently someone known to the victim. A system that silently
filtered out "family" would hide exactly the person most likely to matter. So
enrolment changes how loudly someone is surfaced and what it takes to escalate
them — never whether they are analysed. Every sighting is still recorded, still
in the timeline, still on the graph.

Enrolment is human-only. There is no tool for it, for the same reason the planner
cannot change its own autonomy mode: a system able to declare someone a known
associate is a system able to suppress them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .faces import (
    FACE_MATCH_STRONG,
    FACE_MATCH_WEAK,
    FaceMatch,
    detect_faces,
    embedding_from_bytes,
    embedding_to_bytes,
    match_embedding,
)
from .store import Store


class PersonStatus(str, Enum):
    """What the investigator has told us about a person."""

    VICTIM = "victim"
    PERSON_OF_INTEREST = "person_of_interest"
    KNOWN_ASSOCIATE = "known_associate"
    UNKNOWN = "unknown"  # never enrolled; the default for a detected face

    @property
    def label(self) -> str:
        return {
            "victim": "the victim",
            "person_of_interest": "a person of interest from a previous investigation",
            "known_associate": "a known associate (accounted for by the investigator)",
            "unknown": "an unidentified person",
        }[self.value]

    @property
    def suppresses_recurrence(self) -> bool:
        """Whether appearing often is, by itself, uninteresting for this person.

        True for known associates and the victim — both are *expected* to appear
        throughout the evidence. It says nothing about whether they can be
        escalated for an actual reason.
        """
        return self in (PersonStatus.KNOWN_ASSOCIATE, PersonStatus.VICTIM)


@dataclass
class Identification:
    """Outcome of matching a detected face against the register."""

    status: PersonStatus
    person_id: str | None = None
    name: str | None = None
    relationship: str | None = None
    similarity: float = 0.0
    strong: bool = False

    @property
    def identified(self) -> bool:
        return self.person_id is not None

    def describe(self) -> str:
        if not self.identified:
            return f"unidentified person (best register match {self.similarity:.2f})"
        confidence = "confirmed" if self.strong else "probable"
        detail = f" ({self.relationship})" if self.relationship else ""
        return f"{self.name}{detail} — {self.status.label}, {confidence} {self.similarity:.2f}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "name": self.name,
            "relationship": self.relationship,
            "similarity": round(self.similarity, 3),
            "strong": self.strong,
        }


class PersonRegister:
    """Enrolment and identification against known people."""

    def __init__(self, store: Store, case_id: str | None = None) -> None:
        self.store = store
        self.case_id = case_id
        self._cache: list[tuple[str, np.ndarray]] | None = None
        self._meta: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ enrolment

    def enroll(
        self,
        name: str,
        status: PersonStatus,
        photo_paths: list[str | Path],
        *,
        relationship: str | None = None,
        global_entry: bool = False,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Enrol a person from one or more reference photographs.

        `global_entry=True` makes the entry visible to every case — used for the
        prior-suspect gallery, which is exactly what should span investigations.
        Victims and known associates stay scoped to their own case.
        """
        if not photo_paths:
            raise ValueError("At least one reference photograph is required.")

        person_id = self.store.enroll_person(
            name=name,
            status=status.value,
            case_id=None if global_entry else self.case_id,
            relationship=relationship,
            notes=notes,
        )

        accepted, rejected = 0, []
        for path in photo_paths:
            faces = detect_faces(path)
            if not faces:
                rejected.append(f"{Path(path).name}: no face detected")
                continue
            if len(faces) > 1:
                # Reference photos must be unambiguous. Guessing which face is
                # the subject is exactly the kind of silent error that produces
                # a wrong "known associate" label later.
                rejected.append(
                    f"{Path(path).name}: {len(faces)} faces found — use a photo with one person"
                )
                continue

            self.store.add_reference_face(
                person_id, embedding_to_bytes(faces[0].embedding), str(path)
            )
            accepted += 1

        self._cache = None
        self.store.audit(
            self.case_id,
            actor="human",
            action="person_enrolled",
            detail={
                "person_id": person_id,
                "name": name,
                "status": status.value,
                "relationship": relationship,
                "global": global_entry,
                "references_accepted": accepted,
                "references_rejected": rejected,
            },
        )

        return {
            "ok": accepted > 0,
            "person_id": person_id,
            "name": name,
            "status": status.value,
            "references_accepted": accepted,
            "rejected": rejected,
        }

    # ------------------------------------------------------- identification

    def _references(self) -> list[tuple[str, np.ndarray]]:
        if self._cache is None:
            self._cache = []
            self._meta = {}
            for row in self.store.list_reference_faces(self.case_id):
                self._cache.append((row["person_id"], embedding_from_bytes(row["embedding"])))
                self._meta[row["person_id"]] = {
                    "name": row["name"],
                    "status": row["status"],
                    "relationship": row["relationship"],
                }
        return self._cache

    def identify(self, embedding: np.ndarray) -> Identification:
        """Match one face embedding against every enrolled reference."""
        references = self._references()
        if not references:
            return Identification(status=PersonStatus.UNKNOWN)

        match: FaceMatch = match_embedding(
            embedding,
            references,
            weak_threshold=FACE_MATCH_WEAK,
            strong_threshold=FACE_MATCH_STRONG,
        )
        if not match.matched:
            return Identification(status=PersonStatus.UNKNOWN, similarity=match.similarity)

        meta = self._meta.get(match.identity, {})
        return Identification(
            status=PersonStatus(meta.get("status", "unknown")),
            person_id=match.identity,
            name=meta.get("name"),
            relationship=meta.get("relationship"),
            similarity=match.similarity,
            strong=match.strong,
        )

    # -------------------------------------------------------------- queries

    def summary(self) -> dict[str, Any]:
        people = self.store.list_register(self.case_id)
        by_status: dict[str, list[str]] = {}
        for person in people:
            by_status.setdefault(person["status"], []).append(person["name"])
        return {
            "enrolled_total": len(people),
            "by_status": by_status,
            "people": [
                {
                    "person_id": p["person_id"],
                    "name": p["name"],
                    "status": p["status"],
                    "relationship": p["relationship"],
                    "scope": "global" if p["case_id"] is None else "case",
                }
                for p in people
            ],
        }

    def victim(self) -> dict[str, Any] | None:
        for person in self.store.list_register(self.case_id):
            if person["status"] == PersonStatus.VICTIM.value:
                return person
        return None
