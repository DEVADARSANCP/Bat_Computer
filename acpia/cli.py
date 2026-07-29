"""
ACPIA command line.

    python -m acpia.cli case new CASE-001 "Operation Example"
    python -m acpia.cli ingest CASE-001 ./demo_data
    python -m acpia.cli investigate CASE-001
    python -m acpia.cli status CASE-001
    python -m acpia.cli audit CASE-001

`investigate` streams the planner's steps as they happen — the tool calls, the
results, and the reasoning between them. That live trace is the demo: a judge
watches the system decide what to do, not just see what it produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agents import registry
from .ingest import ingest_directory, ingest_file
from .schema import RiskLevel
from .store import Store
from .tools import ToolLayer

load_dotenv()

# Windows terminals default to cp1252 and will crash on box-drawing output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _rule(title: str = "") -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}" if title else "=" * 78


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_case_new(args: argparse.Namespace) -> int:
    store = Store(args.db)
    store.create_case(args.case_id, args.name)
    print(f"Created case {args.case_id}: {args.name}")
    return 0


def cmd_case_list(args: argparse.Namespace) -> int:
    store = Store(args.db)
    cases = store.list_cases()
    if not cases:
        print("No cases yet.")
        return 0
    for case in cases:
        count = store.count_claims(case["case_id"])
        print(f"{case['case_id']:<16} {case['name']:<32} {case['status']:<8} {count} claims")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    store = Store(args.db)
    if not store.case_exists(args.case_id):
        store.create_case(args.case_id, name=f"Case {args.case_id}")
        print(f"(created case {args.case_id})")

    path = Path(args.path)
    if path.is_dir():
        items = ingest_directory(store, args.case_id, path)
    elif path.is_file():
        items = [ingest_file(store, args.case_id, path)]
    else:
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    print(f"Ingested {len(items)} item(s) into {args.case_id}:")
    by_kind: dict[str, int] = {}
    for item in items:
        by_kind[item.kind.value] = by_kind.get(item.kind.value, 0) + 1
        print(f"  {item.evidence_id}  {item.kind.value:<10} {item.filename}")
    print("\nBy kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    return 0


def cmd_investigate(args: argparse.Namespace) -> int:
    from .orchestrator import Planner  # imported here so other commands work without a key

    store = Store(args.db)
    if not store.case_exists(args.case_id):
        print(f"No such case: {args.case_id}", file=sys.stderr)
        return 1

    tools = ToolLayer(store, registry, args.case_id)

    try:
        planner = Planner(tools, model=args.model)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(_rule(f"ACPIA — investigating {args.case_id}"))
    print(f"model: {args.model}   agents: {len(registry.names())}   "
          f"tools: {len(tools.list_tools())}   max steps: {args.max_steps}\n")

    report = None
    for step in planner.investigate(args.goal, max_steps=args.max_steps):
        print(step.render(), flush=True)
        if step.kind in ("finished", "aborted"):
            report = step.result.get("report")

    if report is None:
        return 1

    print(_rule("RESULT"))
    print(f"completed        : {report.completed}")
    print(f"steps taken      : {report.steps_taken}")
    print(f"tools called     : {len(report.tools_called)} "
          f"({', '.join(dict.fromkeys(report.tools_called))})")
    print(f"duration         : {report.duration_ms / 1000:.1f}s")
    print(f"confidence       : {report.confidence:.2f}")
    print(f"\nsummary:\n{report.summary}")
    if report.recommended_actions:
        print("\nrecommended actions:")
        for i, action in enumerate(report.recommended_actions, 1):
            print(f"  {i}. {action}")
    return 0 if report.completed else 2


def cmd_status(args: argparse.Namespace) -> int:
    store = Store(args.db)
    if not store.case_exists(args.case_id):
        print(f"No such case: {args.case_id}", file=sys.stderr)
        return 1

    tools = ToolLayer(store, registry, args.case_id)
    overview = tools.call("case_overview")

    print(_rule(f"CASE {args.case_id}"))
    print(f"evidence   : {overview['evidence_total']} "
          f"({', '.join(f'{k}={v}' for k, v in overview['evidence_by_kind'].items()) or 'none'})")
    print(f"claims     : {overview['claims_total']}")
    print(f"entities   : {overview['entities_total']}")
    print(f"unanalysed : {overview['pending_count']}")

    entities = store.list_entities(args.case_id, limit=15)
    if entities:
        print("\ntop entities:")
        for e in entities:
            print(f"  {e['type']:<12} {e['value']:<44} x{e['mentions']}")

    flagged = store.get_claims(args.case_id, min_risk=RiskLevel.MEDIUM, limit=10)
    if flagged:
        print("\nflagged findings:")
        for c in flagged:
            print(f"  [{c['risk']:<8}] {c['reason']}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    store = Store(args.db)
    entries = store.get_audit(args.case_id, limit=args.limit)
    if not entries:
        print("No audit entries.")
        return 0
    print(_rule(f"AUDIT TRAIL — {args.case_id or 'all cases'}"))
    for entry in reversed(entries):
        detail = ", ".join(f"{k}={v}" for k, v in entry["detail"].items() if v is not None)
        print(f"{entry['at'][:19]}  {entry['actor']:<16} {entry['action']:<34} {detail[:70]}")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    print(_rule("REGISTERED AGENTS"))
    for agent in registry.all():
        kinds = ", ".join(sorted(k.value for k in agent.handles))
        print(f"\n{agent.name}  (cost {agent.cost}/10, handles: {kinds})")
        print(f"  {agent.description[:200]}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acpia", description="ACPIA investigation assistant")
    parser.add_argument("--db", default="acpia.db", help="Database path (default: acpia.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    case = sub.add_parser("case", help="Manage cases").add_subparsers(dest="sub", required=True)
    new = case.add_parser("new", help="Create a case")
    new.add_argument("case_id")
    new.add_argument("name")
    new.set_defaults(func=cmd_case_new)
    case.add_parser("list", help="List cases").set_defaults(func=cmd_case_list)

    ingest = sub.add_parser("ingest", help="Ingest a file or directory of evidence")
    ingest.add_argument("case_id")
    ingest.add_argument("path")
    ingest.set_defaults(func=cmd_ingest)

    investigate = sub.add_parser("investigate", help="Run the autonomous planner on a case")
    investigate.add_argument("case_id")
    investigate.add_argument("--goal", default=None, help="Optional focus for the investigation")
    investigate.add_argument("--model", default="llama-3.3-70b-versatile")
    investigate.add_argument("--max-steps", type=int, default=30, dest="max_steps")
    investigate.set_defaults(func=cmd_investigate)

    status = sub.add_parser("status", help="Show case state")
    status.add_argument("case_id")
    status.set_defaults(func=cmd_status)

    audit = sub.add_parser("audit", help="Show the audit trail")
    audit.add_argument("case_id", nargs="?", default=None)
    audit.add_argument("--limit", type=int, default=60)
    audit.set_defaults(func=cmd_audit)

    sub.add_parser("agents", help="List registered agents").set_defaults(func=cmd_agents)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
