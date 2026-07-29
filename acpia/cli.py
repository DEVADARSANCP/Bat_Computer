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
from .watchlist import AutonomyMode, Watchlist

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


def cmd_watchlist(args: argparse.Namespace) -> int:
    store = Store(args.db)
    if not store.case_exists(args.case_id):
        print(f"No such case: {args.case_id}", file=sys.stderr)
        return 1

    wl = Watchlist(store, args.case_id)
    summary = wl.summary()

    print(_rule(f"WATCHLIST — {args.case_id}"))
    print(f"autonomy mode : {summary['autonomy_mode']}")
    print(f"tracking      : {summary['active_count']}")
    print(f"awaiting you  : {summary['pending_approval_count']}")

    if summary["active"]:
        print("\nACTIVE:")
        for e in summary["active"]:
            print(f"  {e['entry_id']}  [{e['risk']:<8}] {e['type']:<10} {e['value']}")
            print(f"                  {e['detail']}")

    if summary["awaiting_approval"]:
        print("\nAWAITING APPROVAL:")
        for e in summary["awaiting_approval"]:
            print(f"  {e['entry_id']}  {e['type']:<10} {e['value']}")
            print(f"                  {e['detail']}")
        print("\n  approve: python -m acpia.cli approve <case> <entry_id>")
        print("  reject : python -m acpia.cli reject  <case> <entry_id>")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    store = Store(args.db)
    result = Watchlist(store, args.case_id).approve(args.entry_id, note=args.note or "")
    if not result.get("ok"):
        print(result.get("error"), file=sys.stderr)
        return 1
    print(f"Approved — now tracking '{result['entity']}'.")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    store = Store(args.db)
    result = Watchlist(store, args.case_id).reject(args.entry_id, note=args.note or "")
    if not result.get("ok"):
        print(result.get("error"), file=sys.stderr)
        return 1
    print(f"Rejected — '{result['entity']}' will not be tracked.")
    return 0


def cmd_autonomy(args: argparse.Namespace) -> int:
    """Change how much the system may do unsupervised.

    Human-only by design: there is no tool for this, so the planner cannot grant
    itself permission to act without review.
    """
    store = Store(args.db)
    if not store.case_exists(args.case_id):
        print(f"No such case: {args.case_id}", file=sys.stderr)
        return 1

    wl = Watchlist(store, args.case_id)
    if args.mode is None:
        print(f"{args.case_id} autonomy mode: {wl.mode.value}")
        return 0

    wl.set_mode(AutonomyMode(args.mode))
    print(f"{args.case_id} autonomy mode set to: {args.mode}")
    if args.mode == AutonomyMode.AUTONOMOUS.value:
        print("  The system will now add entities to the watchlist without asking.")
        print("  Everything it does is still logged and reversible.")
    else:
        print("  The system will propose watchlist additions for your approval.")
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

    watchlist = sub.add_parser("watchlist", help="Show the case watchlist")
    watchlist.add_argument("case_id")
    watchlist.set_defaults(func=cmd_watchlist)

    approve = sub.add_parser("approve", help="Approve a pending watchlist entry")
    approve.add_argument("case_id")
    approve.add_argument("entry_id")
    approve.add_argument("--note", default=None)
    approve.set_defaults(func=cmd_approve)

    reject = sub.add_parser("reject", help="Reject a pending watchlist entry")
    reject.add_argument("case_id")
    reject.add_argument("entry_id")
    reject.add_argument("--note", default=None)
    reject.set_defaults(func=cmd_reject)

    autonomy = sub.add_parser("autonomy", help="Show or set the autonomy mode (human only)")
    autonomy.add_argument("case_id")
    autonomy.add_argument(
        "mode", nargs="?", choices=[m.value for m in AutonomyMode],
        help="Omit to show the current mode.",
    )
    autonomy.set_defaults(func=cmd_autonomy)

    sub.add_parser("agents", help="List registered agents").set_defaults(func=cmd_agents)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
