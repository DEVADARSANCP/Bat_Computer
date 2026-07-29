"""
Persistence for ACPIA — cases, evidence, claims, and the audit log.

SQLite on purpose. It is a single file, needs no server, survives a laptop
reboot mid-demo, and is trivially inspectable when a judge asks "where is this
actually stored?". Entities and relationships are denormalised out of claims
into their own tables so the graph builder can query them directly.

The audit log is append-only and is the backbone of the human-oversight story:
every agent invocation, tool call, planner decision and human approval lands
here with a timestamp.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    Claim,
    Entity,
    EntityType,
    Evidence,
    EvidenceKind,
    RelationType,
    Relationship,
    RiskLevel,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    attributes  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    path        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    analysed_by TEXT NOT NULL DEFAULT '[]',
    attributes  TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence(case_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_hash ON evidence(case_id, sha256);

CREATE TABLE IF NOT EXISTS claims (
    claim_id    TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    agent       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    confidence  REAL NOT NULL,
    reason      TEXT NOT NULL,
    risk        TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    locator     TEXT,
    excerpt     TEXT,
    observed_at TEXT,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_claims_case ON claims(case_id);
CREATE INDEX IF NOT EXISTS idx_claims_evidence ON claims(evidence_id);
CREATE INDEX IF NOT EXISTS idx_claims_risk ON claims(case_id, risk);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    type        TEXT NOT NULL,
    value       TEXT NOT NULL,
    attributes  TEXT NOT NULL DEFAULT '{}',
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    mentions    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (case_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(case_id, type);

CREATE TABLE IF NOT EXISTS claim_entities (
    claim_id  TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, entity_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    rel_key     TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    subject_id  TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object_id   TEXT NOT NULL,
    claim_id    TEXT NOT NULL,
    attributes  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (case_id, rel_key, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(case_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_rel_object ON relationships(case_id, object_id);

CREATE TABLE IF NOT EXISTS audit (
    audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit(case_id, audit_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Store:
    """Thin, explicit SQLite wrapper. No ORM — the queries are the documentation."""

    def __init__(self, db_path: str | Path = "acpia.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------------------------------------------------------------- cases

    def create_case(self, case_id: str, name: str, **attributes: Any) -> str:
        self.conn.execute(
            "INSERT OR IGNORE INTO cases (case_id, name, created_at, attributes) VALUES (?,?,?,?)",
            (case_id, name, _now_iso(), json.dumps(attributes)),
        )
        self.conn.commit()
        self.audit(case_id, actor="system", action="case_created", detail={"name": name})
        return case_id

    def list_cases(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM cases ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def case_exists(self, case_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return row is not None

    # ------------------------------------------------------------- evidence

    def add_evidence(self, evidence: Evidence) -> Evidence:
        """Insert evidence. Deduplicates on (case_id, sha256).

        Returning the existing row on a hash collision is deliberate: ingesting
        the same file twice must not create two parallel investigations of it.
        """
        existing = self.conn.execute(
            "SELECT * FROM evidence WHERE case_id=? AND sha256=?",
            (evidence.case_id, evidence.sha256),
        ).fetchone()
        if existing:
            return self._row_to_evidence(existing)

        self.conn.execute(
            """INSERT INTO evidence
               (evidence_id, case_id, path, filename, kind, sha256, size_bytes,
                ingested_at, analysed_by, attributes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                evidence.evidence_id,
                evidence.case_id,
                evidence.path,
                evidence.filename,
                evidence.kind.value,
                evidence.sha256,
                evidence.size_bytes,
                evidence.ingested_at.isoformat(),
                json.dumps(evidence.analysed_by),
                json.dumps(evidence.attributes),
            ),
        )
        self.conn.commit()
        self.audit(
            evidence.case_id,
            actor="system",
            action="evidence_ingested",
            detail={
                "evidence_id": evidence.evidence_id,
                "filename": evidence.filename,
                "kind": evidence.kind.value,
                "sha256": evidence.sha256,
            },
        )
        return evidence

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        row = self.conn.execute(
            "SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)
        ).fetchone()
        return self._row_to_evidence(row) if row else None

    def list_evidence(self, case_id: str, kind: EvidenceKind | None = None) -> list[Evidence]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM evidence WHERE case_id=? AND kind=? ORDER BY ingested_at",
                (case_id, kind.value),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM evidence WHERE case_id=? ORDER BY ingested_at", (case_id,)
            ).fetchall()
        return [self._row_to_evidence(r) for r in rows]

    def mark_analysed(self, evidence_id: str, agent: str) -> None:
        """Record that `agent` has run on this evidence, so the planner doesn't repeat it."""
        row = self.conn.execute(
            "SELECT analysed_by FROM evidence WHERE evidence_id=?", (evidence_id,)
        ).fetchone()
        if not row:
            return
        analysed: list[str] = json.loads(row["analysed_by"])
        if agent not in analysed:
            analysed.append(agent)
            self.conn.execute(
                "UPDATE evidence SET analysed_by=? WHERE evidence_id=?",
                (json.dumps(analysed), evidence_id),
            )
            self.conn.commit()

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> Evidence:
        return Evidence(
            evidence_id=row["evidence_id"],
            case_id=row["case_id"],
            path=row["path"],
            filename=row["filename"],
            kind=EvidenceKind(row["kind"]),
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            ingested_at=_dt(row["ingested_at"]),
            analysed_by=json.loads(row["analysed_by"]),
            attributes=json.loads(row["attributes"]),
        )

    # ---------------------------------------------------------------- claims

    def record_claims(self, claims: Iterable[Claim]) -> int:
        """Persist claims and fan their entities/relationships out into their own tables."""
        count = 0
        for claim in claims:
            self.conn.execute(
                """INSERT OR REPLACE INTO claims
                   (claim_id, case_id, agent, kind, confidence, reason, risk,
                    evidence_id, locator, excerpt, observed_at, created_at, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    claim.claim_id,
                    claim.case_id,
                    claim.agent,
                    claim.kind.value,
                    claim.confidence,
                    claim.reason,
                    claim.risk.value,
                    claim.source.evidence_id,
                    claim.source.locator,
                    claim.source.excerpt,
                    claim.observed_at.isoformat() if claim.observed_at else None,
                    claim.created_at.isoformat(),
                    json.dumps(claim.payload, default=str),
                ),
            )

            seen_at = (claim.observed_at or claim.created_at).isoformat()
            for entity in claim.entities:
                self._upsert_entity(claim.case_id, entity, seen_at)
                self.conn.execute(
                    "INSERT OR IGNORE INTO claim_entities (claim_id, entity_id) VALUES (?,?)",
                    (claim.claim_id, entity.id),
                )

            for rel in claim.relationships:
                self.conn.execute(
                    """INSERT OR REPLACE INTO relationships
                       (rel_key, case_id, subject_id, predicate, object_id,
                        claim_id, attributes, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        rel.key,
                        claim.case_id,
                        rel.subject_id,
                        rel.predicate.value,
                        rel.object_id,
                        claim.claim_id,
                        json.dumps(rel.attributes, default=str),
                        seen_at,
                    ),
                )
            count += 1

        self.conn.commit()
        return count

    def _upsert_entity(self, case_id: str, entity: Entity, seen_at: str) -> None:
        """Insert the entity, or widen its first/last-seen window and bump its mention count."""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE case_id=? AND entity_id=?", (case_id, entity.id)
        ).fetchone()

        if row is None:
            self.conn.execute(
                """INSERT INTO entities
                   (entity_id, case_id, type, value, attributes, first_seen, last_seen, mentions)
                   VALUES (?,?,?,?,?,?,?,1)""",
                (
                    entity.id,
                    case_id,
                    entity.type.value,
                    entity.value,
                    json.dumps(entity.attributes, default=str),
                    seen_at,
                    seen_at,
                ),
            )
            return

        merged = {**json.loads(row["attributes"]), **entity.attributes}
        self.conn.execute(
            """UPDATE entities
               SET mentions = mentions + 1,
                   first_seen = MIN(first_seen, ?),
                   last_seen  = MAX(last_seen, ?),
                   attributes = ?
               WHERE case_id=? AND entity_id=?""",
            (seen_at, seen_at, json.dumps(merged, default=str), case_id, entity.id),
        )

    def get_claims(
        self,
        case_id: str,
        *,
        agent: str | None = None,
        evidence_id: str | None = None,
        min_risk: RiskLevel | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM claims WHERE case_id=?"
        params: list[Any] = [case_id]
        if agent:
            sql += " AND agent=?"
            params.append(agent)
        if evidence_id:
            sql += " AND evidence_id=?"
            params.append(evidence_id)
        if min_risk:
            allowed = [r.value for r in RiskLevel if r.score >= min_risk.score]
            sql += f" AND risk IN ({','.join('?' * len(allowed))})"
            params.extend(allowed)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def count_claims(self, case_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM claims WHERE case_id=?", (case_id,)
        ).fetchone()
        return row["n"]

    # -------------------------------------------------------------- entities

    def list_entities(
        self, case_id: str, entity_type: EntityType | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        if entity_type:
            rows = self.conn.execute(
                """SELECT * FROM entities WHERE case_id=? AND type=?
                   ORDER BY mentions DESC LIMIT ?""",
                (case_id, entity_type.value, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE case_id=? ORDER BY mentions DESC LIMIT ?",
                (case_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["attributes"] = json.loads(d["attributes"])
            out.append(d)
        return out

    def list_relationships(self, case_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM relationships WHERE case_id=? ORDER BY created_at LIMIT ?",
            (case_id, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["attributes"] = json.loads(d["attributes"])
            out.append(d)
        return out

    # ----------------------------------------------------------------- audit

    def audit(
        self, case_id: str | None, *, actor: str, action: str, detail: dict[str, Any] | None = None
    ) -> None:
        """Append-only. Never updated, never deleted — that is the point."""
        self.conn.execute(
            "INSERT INTO audit (case_id, at, actor, action, detail) VALUES (?,?,?,?,?)",
            (case_id, _now_iso(), actor, action, json.dumps(detail or {}, default=str)),
        )
        self.conn.commit()

    def get_audit(self, case_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if case_id:
            rows = self.conn.execute(
                "SELECT * FROM audit WHERE case_id=? ORDER BY audit_id DESC LIMIT ?",
                (case_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM audit ORDER BY audit_id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d["detail"])
            out.append(d)
        return out
