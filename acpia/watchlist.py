"""
The watchlist — entities the investigation is actively tracking, and the
autonomy control that governs how they get there.

This is the mechanism that turns ACPIA from a tagger into an investigator. A
tagger labels each file independently. An investigator builds up a set of
subjects as it goes and then actively looks for them in everything it sees
afterwards. The watchlist is that set.

Four things can promote an entity onto it:

* **CO_OCCURRENCE** — it appeared alongside something already being tracked.
* **MEMORY_HIT** — it matched something recorded in an earlier case.
* **RECURRENCE** — it turned up in several separate pieces of evidence. A face
  in one photo is a detail; the same face in five is a subject.
* **HUMAN** — the investigator flagged it.

Autonomy is a per-case setting, not a global one, because the right answer
differs by case and by investigator:

* **SUPERVISED** (default) — the system proposes, a human approves. Nothing
  enters the watchlist unreviewed.
* **AUTONOMOUS** — the system acts immediately and logs everything it did.

In both modes the human can add, remove, approve and reject at any time. The
autonomy setting changes *when* a person is consulted, never *whether* they are
in control.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .schema import RiskLevel
from .store import Store


class PromotionReason(str, Enum):
    """Why an entity was put forward for the watchlist."""

    CO_OCCURRENCE = "co_occurrence"
    MEMORY_HIT = "memory_hit"
    RECURRENCE = "recurrence"
    HUMAN = "human"

    @property
    def label(self) -> str:
        return {
            "co_occurrence": "appeared alongside a tracked entity",
            "memory_hit": "matched an entity from an earlier case",
            "recurrence": "appeared across multiple pieces of evidence",
            "human": "flagged by the investigator",
        }[self.value]

    @property
    def base_risk(self) -> RiskLevel:
        """Starting risk for an entry promoted this way.

        A cross-case match is the strongest automatic signal available — it means
        this entity was significant enough to be recorded in a previous
        investigation — so it starts high.
        """
        return {
            "memory_hit": RiskLevel.HIGH,
            "co_occurrence": RiskLevel.MEDIUM,
            "recurrence": RiskLevel.MEDIUM,
            "human": RiskLevel.HIGH,
        }[self.value]


class AutonomyMode(str, Enum):
    """How much the system may do without asking."""

    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"


class EntryStatus(str, Enum):
    PENDING = "pending"  # proposed, awaiting human decision
    ACTIVE = "active"  # being tracked
    REJECTED = "rejected"  # human said no
    REMOVED = "removed"  # human removed after it was active


#: Distinct evidence items an entity must appear in before RECURRENCE fires.
#: Two is a coincidence; three is a pattern worth a human's attention.
DEFAULT_RECURRENCE_THRESHOLD = 3


@dataclass
class PromotionOutcome:
    """Result of proposing an entity. Tells the planner what actually happened."""

    entry_id: str | None
    status: EntryStatus
    reason: PromotionReason
    message: str
    requires_approval: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "status": self.status.value,
            "reason": self.reason.value,
            "message": self.message,
            "requires_approval": self.requires_approval,
        }


class Watchlist:
    """Case-scoped watchlist with autonomy-gated promotion."""

    def __init__(
        self,
        store: Store,
        case_id: str,
        *,
        recurrence_threshold: int = DEFAULT_RECURRENCE_THRESHOLD,
    ) -> None:
        self.store = store
        self.case_id = case_id
        self.recurrence_threshold = recurrence_threshold

    # ------------------------------------------------------------- autonomy

    @property
    def mode(self) -> AutonomyMode:
        raw = self.store.get_case_setting(self.case_id, "autonomy", AutonomyMode.SUPERVISED.value)
        try:
            return AutonomyMode(raw)
        except ValueError:
            return AutonomyMode.SUPERVISED

    def set_mode(self, mode: AutonomyMode, *, actor: str = "human") -> None:
        previous = self.mode
        self.store.set_case_setting(self.case_id, "autonomy", mode.value)
        self.store.audit(
            self.case_id,
            actor=actor,
            action="autonomy_changed",
            detail={"from": previous.value, "to": mode.value},
        )

    # ------------------------------------------------------------ promotion

    def propose(
        self,
        entity_id: str,
        entity_type: str,
        entity_value: str,
        reason: PromotionReason,
        detail: str,
        *,
        confidence: float = 0.7,
        proposed_by: str = "planner",
    ) -> PromotionOutcome:
        """Put an entity forward for the watchlist.

        In AUTONOMOUS mode it is added and tracked immediately. In SUPERVISED
        mode it is queued as PENDING and surfaced to the investigator. A human
        proposal is always honoured directly — the mode governs the machine, not
        the person.
        """
        existing = self.store.get_watchlist_entry(self.case_id, entity_id)
        if existing and existing["status"] in (EntryStatus.ACTIVE.value, EntryStatus.PENDING.value):
            return PromotionOutcome(
                entry_id=existing["entry_id"],
                status=EntryStatus(existing["status"]),
                reason=reason,
                message=f"'{entity_value}' is already on the watchlist ({existing['status']}).",
                requires_approval=existing["status"] == EntryStatus.PENDING.value,
            )

        human_origin = reason is PromotionReason.HUMAN or proposed_by == "human"
        auto = human_origin or self.mode is AutonomyMode.AUTONOMOUS
        status = EntryStatus.ACTIVE if auto else EntryStatus.PENDING

        entry_id = self.store.add_watchlist_entry(
            case_id=self.case_id,
            entity_id=entity_id,
            entity_type=entity_type,
            entity_value=entity_value,
            status=status.value,
            reason=reason.value,
            detail=detail,
            confidence=confidence,
            risk=reason.base_risk.value,
            added_by=proposed_by,
        )

        self.store.audit(
            self.case_id,
            actor=proposed_by,
            action="watchlist_proposed" if not auto else "watchlist_added",
            detail={
                "entry_id": entry_id,
                "entity": entity_value,
                "entity_type": entity_type,
                "reason": reason.value,
                "detail": detail,
                "status": status.value,
            },
        )

        if auto:
            message = (
                f"'{entity_value}' added to the watchlist — {reason.label}. {detail}"
                + ("" if human_origin else " (autonomous mode: added without approval)")
            )
        else:
            message = (
                f"'{entity_value}' proposed for the watchlist — {reason.label}. {detail} "
                f"AWAITING INVESTIGATOR APPROVAL — it is not being tracked yet."
            )

        return PromotionOutcome(
            entry_id=entry_id,
            status=status,
            reason=reason,
            message=message,
            requires_approval=not auto,
        )

    # ---------------------------------------------------- automatic triggers

    def check_recurrence(self, entity_id: str) -> PromotionOutcome | None:
        """Promote an entity that has turned up in enough separate evidence items."""
        entity = self.store.get_entity(self.case_id, entity_id)
        if not entity:
            return None

        sources = self.store.count_entity_evidence(self.case_id, entity_id)
        if sources < self.recurrence_threshold:
            return None
        if self.store.get_watchlist_entry(self.case_id, entity_id):
            return None

        return self.propose(
            entity_id=entity_id,
            entity_type=entity["type"],
            entity_value=entity["value"],
            reason=PromotionReason.RECURRENCE,
            detail=f"Seen in {sources} separate pieces of evidence.",
            confidence=min(0.5 + 0.1 * sources, 0.95),
        )

    def check_co_occurrence(self, entity_id: str) -> PromotionOutcome | None:
        """Promote an entity that shares evidence with something already tracked."""
        entity = self.store.get_entity(self.case_id, entity_id)
        if not entity:
            return None
        if self.store.get_watchlist_entry(self.case_id, entity_id):
            return None

        tracked = {e["entity_id"] for e in self.active()}
        if not tracked:
            return None

        neighbours = self.store.entities_sharing_evidence(self.case_id, entity_id)
        overlap = [n for n in neighbours if n["entity_id"] in tracked]
        if not overlap:
            return None

        names = ", ".join(f"'{n['value']}'" for n in overlap[:3])
        return self.propose(
            entity_id=entity_id,
            entity_type=entity["type"],
            entity_value=entity["value"],
            reason=PromotionReason.CO_OCCURRENCE,
            detail=f"Appears in the same evidence as tracked {names}.",
            confidence=0.75,
        )

    def evaluate(self, entity_id: str) -> list[PromotionOutcome]:
        """Run every automatic trigger against one entity.

        Called after an agent produces entities. MEMORY_HIT is not here — it is
        raised by the memory layer, which owns cross-case knowledge.
        """
        outcomes = []
        for check in (self.check_recurrence, self.check_co_occurrence):
            outcome = check(entity_id)
            if outcome:
                outcomes.append(outcome)
                break  # one promotion per entity; the first reason is the truthful one
        return outcomes

    # -------------------------------------------------------- human control

    def approve(self, entry_id: str, *, actor: str = "human", note: str = "") -> dict[str, Any]:
        return self._resolve(entry_id, EntryStatus.ACTIVE, actor=actor, note=note)

    def reject(self, entry_id: str, *, actor: str = "human", note: str = "") -> dict[str, Any]:
        return self._resolve(entry_id, EntryStatus.REJECTED, actor=actor, note=note)

    def remove(self, entry_id: str, *, actor: str = "human", note: str = "") -> dict[str, Any]:
        return self._resolve(entry_id, EntryStatus.REMOVED, actor=actor, note=note)

    def _resolve(
        self, entry_id: str, status: EntryStatus, *, actor: str, note: str
    ) -> dict[str, Any]:
        entry = self.store.get_watchlist_entry_by_id(entry_id)
        if not entry:
            return {"ok": False, "error": f"No watchlist entry '{entry_id}'."}

        self.store.set_watchlist_status(entry_id, status.value, resolved_by=actor, note=note)
        self.store.audit(
            self.case_id,
            actor=actor,
            action=f"watchlist_{status.value}",
            detail={"entry_id": entry_id, "entity": entry["entity_value"], "note": note},
        )
        return {
            "ok": True,
            "entry_id": entry_id,
            "entity": entry["entity_value"],
            "status": status.value,
        }

    # -------------------------------------------------------------- queries

    def active(self) -> list[dict[str, Any]]:
        return self.store.list_watchlist(self.case_id, status=EntryStatus.ACTIVE.value)

    def pending(self) -> list[dict[str, Any]]:
        return self.store.list_watchlist(self.case_id, status=EntryStatus.PENDING.value)

    def all_entries(self) -> list[dict[str, Any]]:
        return self.store.list_watchlist(self.case_id)

    def is_tracked(self, entity_id: str) -> bool:
        entry = self.store.get_watchlist_entry(self.case_id, entity_id)
        return bool(entry and entry["status"] == EntryStatus.ACTIVE.value)

    def summary(self) -> dict[str, Any]:
        active = self.active()
        pending = self.pending()
        return {
            "autonomy_mode": self.mode.value,
            "active_count": len(active),
            "pending_approval_count": len(pending),
            "active": [
                {
                    "entry_id": e["entry_id"],
                    "type": e["entity_type"],
                    "value": e["entity_value"],
                    "risk": e["risk"],
                    "reason": e["reason"],
                    "detail": e["detail"],
                }
                for e in active
            ],
            "awaiting_approval": [
                {
                    "entry_id": e["entry_id"],
                    "type": e["entity_type"],
                    "value": e["entity_value"],
                    "reason": e["reason"],
                    "detail": e["detail"],
                }
                for e in pending
            ],
        }
