"""
Agent base class and registry.

Every forensic capability in ACPIA is an Agent, and every Agent is exposed to
the orchestrator as an MCP tool. Adding a capability means writing one subclass
and registering it — no changes to the planner, the store, or the graph.

Two things the base class buys every agent for free:

* Uniform failure handling. An agent that raises does not crash the
  investigation; it returns `ok=False` with the error, and the planner decides
  what to do about it.
* Uniform timing and audit. Every invocation is logged before and after.

`description` is written for a language model, not a human. The planner reads
these strings to decide what to call, so they carry explicit trigger keywords
and stated limits — the same discipline that made the Minecraft MCP tools
route reliably.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

from ..schema import AgentResult, Claim, Evidence, EvidenceKind
from ..store import Store


@dataclass
class AgentContext:
    """Everything an agent is allowed to touch."""

    store: Store
    case_id: str
    config: dict[str, Any] = field(default_factory=dict)

    def option(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


class Agent(ABC):
    """Base class for every forensic capability."""

    #: Stable tool name. Becomes the MCP tool name, so keep it snake_case.
    name: ClassVar[str] = "unnamed_agent"

    #: Written for the planner LLM. State what it does, when to use it, and what
    #: it cannot do. Include trigger keywords.
    description: ClassVar[str] = "No description provided."

    #: Which evidence kinds this agent is willing to look at.
    handles: ClassVar[set[EvidenceKind]] = set()

    #: Rough cost hint so the planner can prefer cheap agents first.
    #: 1 = milliseconds, 5 = seconds, 10 = minutes.
    cost: ClassVar[int] = 5

    # ------------------------------------------------------------------ api

    def can_handle(self, evidence: Evidence) -> bool:
        """Whether this agent is applicable to a given piece of evidence."""
        return evidence.kind in self.handles

    def analyse(self, evidence: Evidence, ctx: AgentContext) -> AgentResult:
        """Run the agent, with timing, audit and error containment.

        Subclasses implement `run()`; this wrapper is what the orchestrator calls.
        """
        ctx.store.audit(
            ctx.case_id,
            actor=self.name,
            action="agent_started",
            detail={"evidence_id": evidence.evidence_id, "filename": evidence.filename},
        )
        started = time.perf_counter()

        try:
            claims = list(self.run(evidence, ctx))
            elapsed = int((time.perf_counter() - started) * 1000)
            result = AgentResult(
                agent=self.name,
                evidence_id=evidence.evidence_id,
                ok=True,
                claims=claims,
                duration_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001 - an agent failure must never abort the case
            elapsed = int((time.perf_counter() - started) * 1000)
            result = AgentResult(
                agent=self.name,
                evidence_id=evidence.evidence_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
                notes=traceback.format_exc(limit=3),
            )

        if result.ok:
            ctx.store.record_claims(result.claims)
            ctx.store.mark_analysed(evidence.evidence_id, self.name)

        ctx.store.audit(
            ctx.case_id,
            actor=self.name,
            action="agent_finished" if result.ok else "agent_failed",
            detail={
                "evidence_id": evidence.evidence_id,
                "claims": len(result.claims),
                "duration_ms": result.duration_ms,
                "error": result.error,
            },
        )
        return result

    @abstractmethod
    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim] | list[Claim]:
        """Do the actual analysis and yield claims. Raising here is safe."""

    # -------------------------------------------------------------- helpers

    def tool_schema(self) -> dict[str, Any]:
        """MCP tool definition for this agent."""
        return {
            "name": self.name,
            "description": (
                f"{self.description}\n\n"
                f"Accepts evidence of kind: {', '.join(sorted(k.value for k in self.handles))}. "
                f"Cost hint: {self.cost}/10."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "description": "ID of the evidence item to analyse.",
                    }
                },
                "required": ["evidence_id"],
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<Agent {self.name}>"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class AgentRegistry:
    """Holds the available agents and answers "what can run on this?"."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> Agent:
        if agent.name in self._agents:
            raise ValueError(f"Agent name already registered: {agent.name}")
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents)

    def applicable(self, evidence: Evidence, *, skip_completed: bool = True) -> list[Agent]:
        """Agents that can run on this evidence, cheapest first.

        `skip_completed` is what stops the planner burning cycles re-running an
        agent that has already processed a file.
        """
        candidates = [a for a in self._agents.values() if a.can_handle(evidence)]
        if skip_completed:
            candidates = [a for a in candidates if a.name not in evidence.analysed_by]
        return sorted(candidates, key=lambda a: a.cost)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [a.tool_schema() for a in self._agents.values()]


#: Process-wide registry. Agents self-register on import.
registry = AgentRegistry()
