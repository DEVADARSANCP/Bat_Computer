"""
ACPIA claim schema — the contract every agent returns.

This module is FROZEN. Agents come and go; this schema does not. Every agent,
no matter what it analyses, emits a list of Claims. That uniformity is what
makes the orchestrator able to reason across images, video, documents and logs
without knowing anything about how each was produced.

Three rules that make the rest of the system work:

1. Every claim carries a `source` pointing back at real evidence. Nothing is
   ever asserted without a trace back to the file (and offset) it came from.
2. Every claim carries a `reason` in plain language. That is what the
   investigator reads when asking "why does the system think this?".
3. Entity IDs are deterministic hashes of (type, normalised value). Two agents
   that independently see the same phone number produce the same entity id, so
   basic correlation falls out for free and the graph stitches itself together.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class EntityType(str, Enum):
    """Things an investigation cares about."""

    PERSON = "person"
    VEHICLE = "vehicle"
    PLATE = "plate"
    LOCATION = "location"
    DEVICE = "device"
    PHONE = "phone"
    EMAIL = "email"
    HANDLE = "handle"  # username / social handle
    ORGANISATION = "organisation"
    DOCUMENT = "document"
    OBJECT = "object"  # generic detected object
    TIMESTAMP = "timestamp"


class RelationType(str, Enum):
    """Edges in the case knowledge graph."""

    SEEN_AT = "seen_at"  # person/vehicle -> location
    SEEN_WITH = "seen_with"  # person -> person
    OWNS = "owns"  # person -> vehicle/device/phone
    CAPTURED_BY = "captured_by"  # evidence -> device
    CONTACTED = "contacted"  # person -> person
    MENTIONS = "mentions"  # document -> entity
    DEPICTS = "depicts"  # evidence -> person/object
    TRAVELLED_TO = "travelled_to"  # person/vehicle -> location
    MATCHES = "matches"  # entity -> entity (cross-source identity link)


class ClaimKind(str, Enum):
    """What sort of assertion this claim is making."""

    ENTITY = "entity"  # "this file contains a person"
    RELATIONSHIP = "relationship"  # "this person was at this location"
    EVENT = "event"  # "at 14:32 a vehicle entered the frame"
    OBSERVATION = "observation"  # free-form finding with no clean structure
    PATTERN = "pattern"  # "this recurs weekly"


class RiskLevel(str, Enum):
    """Triage priority. Drives what the investigator sees first."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def score(self) -> int:
        return {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]


# --------------------------------------------------------------------------
# Core models
# --------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class EvidenceRef(BaseModel):
    """Points at the exact place a claim came from.

    `locator` is deliberately a free string so each agent can use whatever
    addressing makes sense for its medium: "frame:1024", "page:3",
    "bytes:0-512", "line:88", "t:00:01:32.500".
    """

    evidence_id: str
    locator: str | None = None
    excerpt: str | None = Field(
        default=None,
        description="Short human-readable snippet shown next to the finding in the UI.",
    )

    def display(self) -> str:
        return f"{self.evidence_id}@{self.locator}" if self.locator else self.evidence_id


class Entity(BaseModel):
    """A thing. Identity is derived from (type, normalised value)."""

    id: str = ""
    type: EntityType
    value: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def model_post_init(self, _context: Any) -> None:
        if not self.id:
            self.id = self.derive_id(self.type, self.value)

    @staticmethod
    def derive_id(entity_type: EntityType, value: str) -> str:
        """Deterministic id so independent agents converge on the same entity.

        Normalisation is intentionally aggressive (casefold, strip non-alphanumerics)
        because the whole point is that "KL-07 AB 1234" and "KL07AB1234" are the
        same vehicle. Fuzzier matching is the entity-resolution agent's job; this
        just handles the free wins.
        """
        normalised = "".join(ch for ch in value.casefold() if ch.isalnum())
        digest = hashlib.sha1(f"{entity_type.value}:{normalised}".encode()).hexdigest()
        return f"ent_{digest[:16]}"


class Relationship(BaseModel):
    """A directed edge between two entities."""

    subject_id: str
    predicate: RelationType
    object_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.subject_id}|{self.predicate.value}|{self.object_id}"


class Claim(BaseModel):
    """One atomic assertion produced by one agent.

    A claim may introduce entities, relationships, or both — "person P was at
    location L at 14:32" is a single claim carrying two entities and one edge.
    """

    claim_id: str = Field(default_factory=lambda: _new_id("clm"))
    case_id: str
    agent: str
    kind: ClaimKind

    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Plain-language justification, shown to the investigator.")

    source: EvidenceRef
    observed_at: datetime | None = Field(
        default=None,
        description="When the asserted thing happened — NOT when we discovered it.",
    )
    created_at: datetime = Field(default_factory=_now)

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    risk: RiskLevel = RiskLevel.NONE
    payload: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """One-line form, used in planner prompts and log output."""
        bits = [f"[{self.agent}]", self.reason]
        if self.entities:
            bits.append("(" + ", ".join(f"{e.type.value}:{e.value}" for e in self.entities[:4]) + ")")
        if self.risk is not RiskLevel.NONE:
            bits.append(f"risk={self.risk.value}")
        bits.append(f"conf={self.confidence:.2f}")
        return " ".join(bits)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


class EvidenceKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    STRUCTURED = "structured"  # chat export, call log, CSV
    UNKNOWN = "unknown"


class Evidence(BaseModel):
    """A single ingested file."""

    evidence_id: str = Field(default_factory=lambda: _new_id("ev"))
    case_id: str
    path: str
    filename: str
    kind: EvidenceKind
    sha256: str
    size_bytes: int
    ingested_at: datetime = Field(default_factory=_now)
    analysed_by: list[str] = Field(
        default_factory=list,
        description="Agents that have already run on this evidence — prevents duplicate work.",
    )
    attributes: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Agent results
# --------------------------------------------------------------------------


class AgentResult(BaseModel):
    """What every agent hands back. Uniform across all of them."""

    agent: str
    evidence_id: str
    ok: bool = True
    claims: list[Claim] = Field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0
    notes: str | None = Field(
        default=None,
        description="Anything the agent wants the planner to know that isn't a claim.",
    )

    def summary(self) -> str:
        if not self.ok:
            return f"{self.agent}: FAILED — {self.error}"
        return f"{self.agent}: {len(self.claims)} claim(s) in {self.duration_ms}ms"
