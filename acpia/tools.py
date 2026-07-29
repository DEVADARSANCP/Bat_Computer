"""
The tool layer — everything the orchestrator is allowed to do.

This is the single dispatch point for all capabilities, and it exists in one
place so that two very different consumers get identical behaviour:

* the in-process planner loop (fast path, used during an investigation), and
* `acpia.mcp_server`, which exposes the same tools over MCP stdio to any
  external client.

Tools come in two families:

* **Agent tools** — one per registered agent, generated automatically. Adding
  an agent adds a tool with no changes here.
* **Case tools** — reading case state, querying the graph, flagging findings,
  and declaring the investigation complete.

Every call is audited. `finish_investigation` is what lets the planner stop
deliberately rather than looping until it runs out of budget.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .agents.base import AgentContext, AgentRegistry
from .schema import EntityType, EvidenceKind, RiskLevel
from .store import Store
from .watchlist import PromotionReason, Watchlist


class ToolError(Exception):
    """Raised for bad tool input. Returned to the planner as text, never fatal."""


class ToolLayer:
    """Dispatches named tool calls against a single case."""

    def __init__(self, store: Store, registry: AgentRegistry, case_id: str) -> None:
        self.store = store
        self.registry = registry
        self.case_id = case_id
        self.ctx = AgentContext(store=store, case_id=case_id)
        self.watchlist = Watchlist(store, case_id)
        self.finished: dict[str, Any] | None = None

    # NOTE: there is deliberately no tool for changing the autonomy mode. The
    # planner must never be able to grant itself permission to act unsupervised.
    # That setting is reachable only by a human, through the CLI or the UI.

    # ------------------------------------------------------------ discovery

    def list_tools(self) -> list[dict[str, Any]]:
        """Tool schemas, agents first. This is what the planner LLM sees."""
        return self.registry.tool_schemas() + list(_CASE_TOOL_SCHEMAS)

    # ------------------------------------------------------------- dispatch

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a tool by name. Never raises — errors come back as data.

        Returning errors rather than raising is deliberate: a planner that gets
        "that evidence id does not exist" as a result can correct itself on the
        next step, whereas an exception would end the investigation.
        """
        args = arguments or {}
        self.store.audit(
            self.case_id, actor="planner", action="tool_called",
            detail={"tool": name, "arguments": args},
        )

        try:
            agent = self.registry.get(name)
            if agent is not None:
                result = self._run_agent(name, args)
            else:
                handler = _CASE_TOOLS.get(name)
                if handler is None:
                    raise ToolError(
                        f"Unknown tool '{name}'. Available: {', '.join(self._tool_names())}"
                    )
                result = handler(self, args)
            result.setdefault("ok", True)

        except ToolError as exc:
            result = {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - tool failure must not kill the case
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self.store.audit(
            self.case_id, actor="planner", action="tool_returned",
            detail={"tool": name, "ok": result.get("ok"), "error": result.get("error")},
        )
        return result

    def _tool_names(self) -> list[str]:
        return self.registry.names() + list(_CASE_TOOLS)

    # -------------------------------------------------------------- helpers

    def _require_evidence(self, args: dict[str, Any]):
        evidence_id = args.get("evidence_id")
        if not evidence_id:
            raise ToolError("Missing required argument 'evidence_id'.")
        evidence = self.store.get_evidence(evidence_id)
        if evidence is None:
            raise ToolError(f"No evidence with id '{evidence_id}' in this case.")
        if evidence.case_id != self.case_id:
            raise ToolError(f"Evidence '{evidence_id}' belongs to a different case.")
        return evidence

    def _run_agent(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        agent = self.registry.get(name)
        evidence = self._require_evidence(args)

        if not agent.can_handle(evidence):
            return {
                "ok": False,
                "error": (
                    f"{agent.name} cannot process evidence of kind '{evidence.kind.value}'. "
                    f"It handles: {', '.join(sorted(k.value for k in agent.handles))}."
                ),
            }

        result = agent.analyse(evidence, self.ctx)
        payload = {
            "ok": result.ok,
            "agent": result.agent,
            "evidence": evidence.filename,
            "claims_found": len(result.claims),
            "duration_ms": result.duration_ms,
            "error": result.error,
            "findings": [c.summary() for c in result.claims],
        }
        promotions = self._evaluate_watchlist(result.claims)
        if promotions:
            payload["watchlist"] = promotions
        return payload

    def _evaluate_watchlist(self, claims) -> list[str]:
        """Run the automatic watchlist triggers over entities an agent just produced.

        This is what makes the system accumulate subjects instead of tagging each
        file in isolation. Surfaced in the tool result so the planner sees a
        promotion the moment it happens and can act on it in the same breath.
        """
        seen: set[str] = set()
        messages: list[str] = []
        for claim in claims:
            for entity in claim.entities:
                if entity.id in seen:
                    continue
                seen.add(entity.id)
                for outcome in self.watchlist.evaluate(entity.id):
                    messages.append(outcome.message)
        return messages


# --------------------------------------------------------------------------
# Case tools
# --------------------------------------------------------------------------


def _case_overview(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    """The planner's view of the world. Kept compact — it goes in every prompt."""
    evidence = tools.store.list_evidence(tools.case_id)
    pending: list[dict[str, Any]] = []

    for item in evidence:
        remaining = tools.registry.applicable(item)
        if remaining:
            pending.append(
                {
                    "evidence_id": item.evidence_id,
                    "filename": item.filename,
                    "kind": item.kind.value,
                    "agents_not_yet_run": [a.name for a in remaining],
                }
            )

    high_risk = tools.store.get_claims(tools.case_id, min_risk=RiskLevel.MEDIUM, limit=20)
    watchlist = tools.watchlist.summary()

    return {
        "case_id": tools.case_id,
        "autonomy_mode": watchlist["autonomy_mode"],
        "evidence_total": len(evidence),
        "evidence_by_kind": _count_by(e.kind.value for e in evidence),
        "claims_total": tools.store.count_claims(tools.case_id),
        "entities_total": len(tools.store.list_entities(tools.case_id, limit=10_000)),
        "pending_analysis": pending[:25],
        "pending_count": len(pending),
        "watchlist_active": watchlist["active"],
        "watchlist_awaiting_approval": watchlist["awaiting_approval"],
        "elevated_risk_findings": [
            {"risk": c["risk"], "agent": c["agent"], "reason": c["reason"]} for c in high_risk
        ],
    }


def _list_evidence(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    kind_arg = args.get("kind")
    kind = None
    if kind_arg:
        try:
            kind = EvidenceKind(kind_arg)
        except ValueError as exc:
            raise ToolError(
                f"Invalid kind '{kind_arg}'. Valid: {', '.join(k.value for k in EvidenceKind)}"
            ) from exc

    items = tools.store.list_evidence(tools.case_id, kind)
    return {
        "count": len(items),
        "evidence": [
            {
                "evidence_id": e.evidence_id,
                "filename": e.filename,
                "kind": e.kind.value,
                "size_bytes": e.size_bytes,
                "analysed_by": e.analysed_by,
            }
            for e in items
        ],
    }


def _list_claims(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    min_risk = None
    if args.get("min_risk"):
        try:
            min_risk = RiskLevel(args["min_risk"])
        except ValueError as exc:
            raise ToolError(f"Invalid min_risk '{args['min_risk']}'.") from exc

    claims = tools.store.get_claims(
        tools.case_id,
        agent=args.get("agent"),
        evidence_id=args.get("evidence_id"),
        min_risk=min_risk,
        limit=int(args.get("limit", 50)),
    )
    return {
        "count": len(claims),
        "claims": [
            {
                "claim_id": c["claim_id"],
                "agent": c["agent"],
                "kind": c["kind"],
                "reason": c["reason"],
                "risk": c["risk"],
                "confidence": c["confidence"],
                "source": c["evidence_id"],
                "observed_at": c["observed_at"],
            }
            for c in claims
        ],
    }


def _list_entities(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    entity_type = None
    if args.get("type"):
        try:
            entity_type = EntityType(args["type"])
        except ValueError as exc:
            raise ToolError(
                f"Invalid type '{args['type']}'. Valid: {', '.join(t.value for t in EntityType)}"
            ) from exc

    entities = tools.store.list_entities(
        tools.case_id, entity_type, limit=int(args.get("limit", 50))
    )
    return {
        "count": len(entities),
        "entities": [
            {
                "entity_id": e["entity_id"],
                "type": e["type"],
                "value": e["value"],
                "mentions": e["mentions"],
                "first_seen": e["first_seen"],
                "last_seen": e["last_seen"],
            }
            for e in entities
        ],
    }


def _entity_connections(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    """Everything directly linked to one entity — the correlation primitive."""
    entity_id = args.get("entity_id")
    if not entity_id:
        raise ToolError("Missing required argument 'entity_id'.")

    lookup = {e["entity_id"]: e for e in tools.store.list_entities(tools.case_id, limit=10_000)}
    if entity_id not in lookup:
        raise ToolError(f"No entity '{entity_id}' in this case.")

    edges = []
    for rel in tools.store.list_relationships(tools.case_id):
        if rel["subject_id"] == entity_id:
            other = lookup.get(rel["object_id"])
            edges.append(
                {
                    "direction": "outgoing",
                    "predicate": rel["predicate"],
                    "other_entity": other["value"] if other else rel["object_id"],
                    "other_type": other["type"] if other else "unknown",
                    "attributes": rel["attributes"],
                }
            )
        elif rel["object_id"] == entity_id:
            other = lookup.get(rel["subject_id"])
            edges.append(
                {
                    "direction": "incoming",
                    "predicate": rel["predicate"],
                    "other_entity": other["value"] if other else rel["subject_id"],
                    "other_type": other["type"] if other else "unknown",
                    "attributes": rel["attributes"],
                }
            )

    subject = lookup[entity_id]
    return {
        "entity": {"id": entity_id, "type": subject["type"], "value": subject["value"]},
        "connection_count": len(edges),
        "connections": edges,
    }


def _analyse_evidence(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    """Run every applicable agent on one item. The planner's bulk-work shortcut."""
    evidence = tools._require_evidence(args)
    agents = tools.registry.applicable(evidence)
    if not agents:
        return {
            "evidence": evidence.filename,
            "agents_run": 0,
            "note": "Every applicable agent has already processed this evidence.",
        }

    outcomes = []
    all_claims = []
    for agent in agents:
        result = agent.analyse(evidence, tools.ctx)
        all_claims.extend(result.claims)
        outcomes.append(
            {
                "agent": agent.name,
                "ok": result.ok,
                "claims_found": len(result.claims),
                "error": result.error,
                "findings": [c.summary() for c in result.claims],
            }
        )

    payload = {
        "evidence": evidence.filename,
        "kind": evidence.kind.value,
        "agents_run": len(outcomes),
        "results": outcomes,
    }
    promotions = tools._evaluate_watchlist(all_claims)
    if promotions:
        payload["watchlist"] = promotions
    return payload


def _watchlist_status(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    return tools.watchlist.summary()


def _flag_entity(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    """Planner-initiated promotion, for judgement calls the automatic rules miss."""
    entity_id = args.get("entity_id")
    justification = args.get("justification")
    if not entity_id:
        raise ToolError("Missing required argument 'entity_id'.")
    if not justification:
        raise ToolError(
            "Missing required argument 'justification'. Flagging an entity for "
            "investigator attention always requires a stated reason."
        )

    entity = tools.store.get_entity(tools.case_id, entity_id)
    if not entity:
        raise ToolError(f"No entity '{entity_id}' in this case.")

    outcome = tools.watchlist.propose(
        entity_id=entity_id,
        entity_type=entity["type"],
        entity_value=entity["value"],
        reason=PromotionReason.CO_OCCURRENCE,
        detail=justification,
        confidence=float(args.get("confidence", 0.7)),
        proposed_by="planner",
    )
    return outcome.as_dict()


def _finish_investigation(tools: ToolLayer, args: dict[str, Any]) -> dict[str, Any]:
    """Planner declares it is done. This is how the loop terminates on purpose."""
    summary = args.get("summary")
    if not summary:
        raise ToolError("Missing required argument 'summary'.")

    tools.finished = {
        "summary": summary,
        "recommended_actions": args.get("recommended_actions", []),
        "confidence": float(args.get("confidence", 0.7)),
    }
    tools.store.audit(
        tools.case_id, actor="planner", action="investigation_finished", detail=tools.finished
    )
    return {"acknowledged": True, **tools.finished}


def _count_by(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


_CASE_TOOLS: dict[str, Callable[[ToolLayer, dict[str, Any]], dict[str, Any]]] = {
    "case_overview": _case_overview,
    "list_evidence": _list_evidence,
    "list_claims": _list_claims,
    "list_entities": _list_entities,
    "entity_connections": _entity_connections,
    "analyse_evidence": _analyse_evidence,
    "watchlist_status": _watchlist_status,
    "flag_entity": _flag_entity,
    "finish_investigation": _finish_investigation,
}


#: Written for the planner LLM: explicit triggers, explicit limits.
_CASE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "case_overview",
        "description": (
            "Get the current state of the investigation: how much evidence exists, which "
            "items still have agents that have not run on them, how many claims and entities "
            "have accumulated, and any elevated-risk findings so far. CALL THIS FIRST at the "
            "start of an investigation and whenever you need to re-orient. "
            "Keywords: status, state, overview, what is left, progress."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_evidence",
        "description": (
            "List evidence items in the case, optionally filtered by kind. Use this to find "
            "the evidence_id you need before calling an analysis agent. "
            "Keywords: files, evidence, what do I have, images, videos, documents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [k.value for k in EvidenceKind],
                    "description": "Optional filter by evidence kind.",
                }
            },
        },
    },
    {
        "name": "analyse_evidence",
        "description": (
            "Run EVERY applicable agent on one evidence item in one call. This is the "
            "efficient way to process a file you have not examined yet — prefer it over "
            "calling individual agents unless you specifically want one capability. "
            "Already-completed agents are skipped automatically. "
            "Keywords: analyse, process, examine, investigate this file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "evidence_id": {"type": "string", "description": "Evidence item to analyse."}
            },
            "required": ["evidence_id"],
        },
    },
    {
        "name": "list_claims",
        "description": (
            "List findings produced so far, optionally filtered by agent, evidence item, or "
            "minimum risk level. Use this to review what is already known before deciding "
            "what to do next. Keywords: findings, results, what was found, risks, flags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": "Only claims from this agent."},
                "evidence_id": {"type": "string", "description": "Only claims from this file."},
                "min_risk": {
                    "type": "string",
                    "enum": [r.value for r in RiskLevel],
                    "description": "Only claims at or above this risk level.",
                },
                "limit": {"type": "integer", "description": "Max claims to return (default 50)."},
            },
        },
    },
    {
        "name": "list_entities",
        "description": (
            "List the people, vehicles, locations, devices, phones and other entities "
            "discovered across the case, most-mentioned first. An entity appearing in many "
            "places is usually where the investigation should focus. "
            "Keywords: who, what, entities, people, vehicles, places, devices."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [t.value for t in EntityType],
                    "description": "Optional filter by entity type.",
                },
                "limit": {"type": "integer", "description": "Max entities (default 50)."},
            },
        },
    },
    {
        "name": "entity_connections",
        "description": (
            "Show everything linked to one entity in the case knowledge graph — the places a "
            "person was seen, the devices that captured them, who they were seen with. This "
            "is the core correlation tool: use it when an entity looks significant and you "
            "want to know what it connects to. "
            "Keywords: connections, links, related, correlate, graph, network, associated with."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity to expand. Get ids from list_entities.",
                }
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "watchlist_status",
        "description": (
            "Show the case watchlist — the entities this investigation is actively tracking "
            "(people, vehicles, plates, handles), plus anything awaiting investigator approval "
            "and the current autonomy mode. Entities reach the watchlist automatically when "
            "they recur across evidence, co-occur with something already tracked, or match an "
            "earlier case. Check this to know WHO AND WHAT YOU ARE LOOKING FOR before "
            "examining new evidence. "
            "Keywords: watchlist, tracking, suspects, subjects, shortlist, who am I looking for."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "flag_entity",
        "description": (
            "Put an entity on the watchlist because YOU judge it significant, when the "
            "automatic rules have not already done so. Use sparingly and only with a real "
            "reason — for example an entity that links two otherwise unconnected pieces of "
            "evidence. In supervised mode this is a PROPOSAL the investigator must approve; "
            "in autonomous mode it takes effect immediately. Either way it is logged. "
            "Keywords: flag, mark, track this, add to watchlist, this looks important."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity to flag."},
                "justification": {
                    "type": "string",
                    "description": "Why this entity warrants investigator attention.",
                },
                "confidence": {"type": "number", "description": "0-1 confidence."},
            },
            "required": ["entity_id", "justification"],
        },
    },
    {
        "name": "finish_investigation",
        "description": (
            "Declare the investigation complete and hand back to the human investigator. "
            "Call this ONLY when there is no remaining analysis worth doing — every item has "
            "been processed and you have followed up the significant correlations. You must "
            "provide a summary of what was found and what the investigator should do next. "
            "Keywords: done, finished, complete, conclude, hand over."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What the investigation established, in plain language.",
                },
                "recommended_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete next steps for the human investigator.",
                },
                "confidence": {
                    "type": "number",
                    "description": "0-1 confidence in the overall picture.",
                },
            },
            "required": ["summary"],
        },
    },
]
