"""
The Investigation Planner — ACPIA's reasoning engine.

This is the part that makes the system agentic rather than a pipeline. It is
given a case and no instructions beyond a goal, and it decides for itself what
to look at, which agent to invoke, what the results mean, what to follow up,
and when the investigation is finished.

The loop is deliberately a *real* multi-step agent loop with native tool
calling, not a one-shot "pick a tool" router. The difference matters: a router
can only answer "what tool handles this file", whereas this loop can notice
that a plate turned up, decide to query a registry, see the returned owner,
recognise the name from an earlier finding, and escalate — a chain no single
step could produce.

Design notes:

* **Observation is fed back in.** Every tool result re-enters the context, so
  step N+1 is informed by step N. That feedback is what separates planning
  from batching.
* **Failures are data.** A tool that errors returns its error to the planner,
  which is expected to adapt rather than stop.
* **Stopping is a decision.** The planner ends by calling
  `finish_investigation` with a summary and recommended actions. Hitting the
  step limit is treated as an abnormal outcome, not a normal exit.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from ..tools import ToolLayer

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_STEPS = 30

SYSTEM_PROMPT = """You are the Investigation Planner inside ACPIA, an evidence-analysis \
platform used by authorised child-protection investigators.

Your role is to work through a case of digital evidence autonomously and surface what a \
human investigator needs to see. You are a triage and correlation engine: you find and \
connect things, and you hand conclusions to a human. You never decide an outcome yourself.

HOW TO WORK

1. Call `case_overview` first to see what evidence exists and what has not been examined.
2. Work through unexamined evidence. `analyse_evidence` runs every applicable agent on one \
item in a single call — prefer it over invoking agents individually.
3. After analysis, LOOK AT WHAT YOU FOUND. Call `list_entities` and `list_claims` to see \
what has accumulated across the whole case, not just the last file.
4. FOLLOW UP CORRELATIONS. This is the most valuable thing you do. If an entity appears in \
several places, or two files share a device, location, or person, call `entity_connections` \
to expand it. A fact that appears once is a detail; a fact that appears repeatedly is a lead.
5. When there is nothing left worth examining, call `finish_investigation` with a summary and \
concrete recommended actions.

HOW TO THINK

- Be genuinely curious. You are not filling in a form — you are looking for the thing that \
matters in a pile of material a human does not have time to read.
- Prioritise by risk to a child, not by confidence or convenience.
- Every conclusion you state must rest on claims that agents actually produced. If evidence \
does not support something, say so plainly. Never invent a finding, a name, a place or a time.
- Absence is informative. Stripped metadata, a gap in a timeline, or a file nothing can read \
are all worth reporting.
- If a tool returns an error, read it and adapt. Errors are information, not dead ends.
- Do not re-run an agent that has already processed an item; the overview tells you what is \
outstanding.

BOUNDARIES

- You produce leads and recommendations. Investigative decisions belong to the human.
- State uncertainty honestly. An investigator acting on a weak lead you oversold is worse \
than no lead at all.
"""


@dataclass
class PlannerStep:
    """One observable step. The UI renders these live so the human can watch it reason."""

    index: int
    kind: Literal["thinking", "tool_call", "tool_result", "finished", "aborted", "error"]
    text: str = ""
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0

    def render(self) -> str:
        if self.kind == "thinking":
            return f"[{self.index:02d}] reasoning: {self.text}"
        if self.kind == "tool_call":
            args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
            return f"[{self.index:02d}] -> {self.tool}({args})"
        if self.kind == "tool_result":
            status = "ok" if self.result.get("ok", True) else "FAILED"
            detail = self.result.get("error") or _brief(self.result)
            return f"[{self.index:02d}] <- {self.tool} [{status}] {detail}"
        if self.kind == "finished":
            return f"[{self.index:02d}] investigation complete: {self.text}"
        if self.kind == "aborted":
            return f"[{self.index:02d}] ABORTED: {self.text}"
        return f"[{self.index:02d}] ERROR: {self.text}"


@dataclass
class InvestigationReport:
    """Outcome of a planner run."""

    case_id: str
    completed: bool
    summary: str
    recommended_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    steps_taken: int = 0
    tools_called: list[str] = field(default_factory=list)
    duration_ms: int = 0


class Planner:
    """LLM-driven investigation loop over the ACPIA tool layer."""

    def __init__(
        self,
        tools: ToolLayer,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        from groq import Groq  # lazy import so the package works without a key

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it in a .env file at the project root "
                "or export it before running an investigation."
            )

        self.tools = tools
        self.model = model
        self.temperature = temperature
        self.client = Groq(api_key=key)

    # ------------------------------------------------------------------ run

    def investigate(
        self, goal: str | None = None, *, max_steps: int = DEFAULT_MAX_STEPS
    ) -> Iterator[PlannerStep]:
        """Run the investigation, yielding each step as it happens.

        Yields steps so a caller can render live progress. The terminal step is
        `finished` or `aborted`, and carries the report in `result`.
        """
        started = time.perf_counter()
        opening = goal or (
            "Investigate this case. Examine all unexamined evidence, correlate what you "
            "find across sources, and report anything an investigator should act on."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": opening},
        ]
        tool_specs = [_to_openai_tool(t) for t in self.tools.list_tools()]
        called: list[str] = []
        index = 0

        self.tools.store.audit(
            self.tools.case_id,
            actor="planner",
            action="investigation_started",
            detail={"goal": opening, "model": self.model, "max_steps": max_steps},
        )

        while index < max_steps:
            index += 1
            step_started = time.perf_counter()

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_specs,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=1500,
                )
            except Exception as exc:  # noqa: BLE001 - surface API failures as a step
                yield PlannerStep(
                    index=index, kind="error", text=f"{type(exc).__name__}: {exc}"
                )
                yield self._abort(
                    index, called, started, f"Planner LLM call failed: {exc}"
                )
                return

            message = response.choices[0].message
            elapsed = int((time.perf_counter() - step_started) * 1000)

            # The model narrated its reasoning alongside (or instead of) a tool call.
            if message.content and message.content.strip():
                yield PlannerStep(
                    index=index,
                    kind="thinking",
                    text=message.content.strip(),
                    elapsed_ms=elapsed,
                )

            if not message.tool_calls:
                # No tool and no intent to continue — treat the prose as the conclusion.
                yield self._finish_from_prose(index, called, started, message.content or "")
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                arguments = _parse_arguments(tool_call.function.arguments)
                called.append(name)

                yield PlannerStep(
                    index=index, kind="tool_call", tool=name, arguments=arguments
                )

                call_started = time.perf_counter()
                result = self.tools.call(name, arguments)
                call_elapsed = int((time.perf_counter() - call_started) * 1000)

                yield PlannerStep(
                    index=index,
                    kind="tool_result",
                    tool=name,
                    result=result,
                    elapsed_ms=call_elapsed,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)[:8000],
                    }
                )

                if self.tools.finished is not None:
                    yield self._finish(index, called, started)
                    return

        yield self._abort(
            index,
            called,
            started,
            f"Reached the {max_steps}-step limit without concluding. "
            f"The case is partially analysed; review findings and re-run to continue.",
        )

    # -------------------------------------------------------------- outcomes

    def _finish(self, index: int, called: list[str], started: float) -> PlannerStep:
        finished = self.tools.finished or {}
        report = InvestigationReport(
            case_id=self.tools.case_id,
            completed=True,
            summary=finished.get("summary", ""),
            recommended_actions=finished.get("recommended_actions", []),
            confidence=float(finished.get("confidence", 0.7)),
            steps_taken=index,
            tools_called=called,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return PlannerStep(
            index=index, kind="finished", text=report.summary, result={"report": report}
        )

    def _finish_from_prose(
        self, index: int, called: list[str], started: float, prose: str
    ) -> PlannerStep:
        """The model stopped calling tools without formally finishing.

        Treated as a soft completion rather than an error — the prose is still
        useful — but `completed` is False so the caller knows it did not close
        the loop properly.
        """
        report = InvestigationReport(
            case_id=self.tools.case_id,
            completed=False,
            summary=prose.strip() or "Planner stopped without calling finish_investigation.",
            steps_taken=index,
            tools_called=called,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        self.tools.store.audit(
            self.tools.case_id,
            actor="planner",
            action="investigation_stopped_without_finish",
            detail={"steps": index},
        )
        return PlannerStep(
            index=index, kind="finished", text=report.summary, result={"report": report}
        )

    def _abort(
        self, index: int, called: list[str], started: float, why: str
    ) -> PlannerStep:
        report = InvestigationReport(
            case_id=self.tools.case_id,
            completed=False,
            summary=why,
            steps_taken=index,
            tools_called=called,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        self.tools.store.audit(
            self.tools.case_id, actor="planner", action="investigation_aborted",
            detail={"reason": why, "steps": index},
        )
        return PlannerStep(index=index, kind="aborted", text=why, result={"report": report})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """MCP tool schema → OpenAI/Groq function-calling schema."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema.get("inputSchema") or {"type": "object", "properties": {}},
        },
    }


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tolerate the model emitting empty or slightly malformed argument JSON."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        if parsed is None:
            # Models routinely send "null" for no-argument tools.
            return {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}


def _brief(result: dict[str, Any], limit: int = 160) -> str:
    """Compact one-line summary of a tool result for step rendering."""
    for key in ("findings", "summary", "connection_count", "count", "claims_found",
                "evidence_total", "agents_run"):
        if key in result:
            value = result[key]
            if isinstance(value, list):
                value = "; ".join(str(v) for v in value[:3]) or "none"
            text = f"{key}={value}"
            return text[:limit] + ("…" if len(text) > limit else "")
    text = json.dumps(result, default=str)
    return text[:limit] + ("…" if len(text) > limit else "")
