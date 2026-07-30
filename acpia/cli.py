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


def cmd_enroll(args: argparse.Namespace) -> int:
    """Enrol a known person. Human-only — there is deliberately no tool for this."""
    from .persons import PersonRegister, PersonStatus

    store = Store(args.db)
    if not store.case_exists(args.case_id):
        store.create_case(args.case_id, name=f"Case {args.case_id}")

    register = PersonRegister(store, args.case_id)
    try:
        status = PersonStatus(args.status)
    except ValueError:
        print(f"Invalid status: {args.status}", file=sys.stderr)
        return 1

    result = register.enroll(
        args.name,
        status,
        args.photo,
        relationship=args.relationship,
        global_entry=args.global_entry,
        notes=args.note,
    )

    if result["references_accepted"]:
        scope = "globally (visible to all cases)" if args.global_entry else f"in {args.case_id}"
        print(f"Enrolled '{args.name}' as {status.value} {scope}")
        print(f"  reference faces accepted: {result['references_accepted']}")
    else:
        print(f"Enrolment FAILED for '{args.name}' — no usable reference face.", file=sys.stderr)

    for rejected in result["rejected"]:
        print(f"  rejected: {rejected}", file=sys.stderr)
    return 0 if result["references_accepted"] else 1


def cmd_register(args: argparse.Namespace) -> int:
    from .persons import PersonRegister

    store = Store(args.db)
    summary = PersonRegister(store, args.case_id).summary()

    print(_rule(f"PERSON REGISTER — {args.case_id}"))
    if not summary["people"]:
        print("Nobody enrolled yet.")
        print("\n  python -m acpia.cli enroll CASE-001 --name \"Anna\" --status victim \\")
        print("         --photo .\\data\\reference\\victim.jpg")
        return 0

    for person in summary["people"]:
        rel = f" ({person['relationship']})" if person["relationship"] else ""
        print(f"  {person['status']:<20} {person['name']}{rel}   [{person['scope']}]")
    return 0


def cmd_calibrate_faces(args: argparse.Namespace) -> int:
    """Measure the actual face-similarity separation on real data.

    Thresholds should be derived from evidence, not inherited from a tutorial.
    Point this at a folder of real footage and it reports where the boundary
    between "same person" and "different person" actually falls.
    """
    import itertools
    from pathlib import Path as _Path

    import numpy as np

    from .faces import FACE_MATCH_STRONG, FACE_MATCH_WEAK, cosine_similarity, detect_faces

    folder = _Path(args.folder)
    images = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        print(f"No images found in {folder}", file=sys.stderr)
        return 1

    print(f"Scanning {len(images)} images...")
    per_image: dict[str, list] = {}
    for path in images:
        try:
            faces = detect_faces(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
            continue
        if faces:
            per_image[path.name] = faces

    total = sum(len(v) for v in per_image.values())
    print(f"{total} faces across {len(per_image)} images\n")

    # Two faces in one photograph are necessarily different people, which gives a
    # free sample of the different-person distribution with no labelling needed.
    different = [
        cosine_similarity(a.embedding, b.embedding)
        for faces in per_image.values()
        for a, b in itertools.combinations(faces, 2)
    ]

    if not different:
        print("No multi-face images — cannot estimate the different-person distribution.")
        print("Add some group photos and re-run.")
        return 0

    arr = np.array(different)
    print(_rule("DIFFERENT-PERSON SIMILARITY (same photo, so definitely different people)"))
    print(f"  samples : {len(arr)}")
    print(f"  median  : {np.percentile(arr, 50):.3f}")
    print(f"  p95     : {np.percentile(arr, 95):.3f}")
    print(f"  max     : {arr.max():.3f}   <-- the false-positive ceiling")
    print()
    print(f"  current weak threshold   : {FACE_MATCH_WEAK:.2f}")
    print(f"  current strong threshold : {FACE_MATCH_STRONG:.2f}")
    print()

    margin = FACE_MATCH_WEAK - arr.max()
    if margin > 0.10:
        print(f"  OK — {margin:.2f} of headroom above the worst different-person pair.")
    elif margin > 0:
        print(f"  TIGHT — only {margin:.2f} of headroom. Consider raising the weak threshold.")
    else:
        print("  UNSAFE — different people score above the weak threshold. Raise it.")
    return 0


def cmd_calibrate_objects(args: argparse.Namespace) -> int:
    """Measure instance-matching separation on real object crops.

    Point at a folder where the SAME distinctive object appears in several
    photographs. Reports whether the current thresholds actually separate it
    from other objects of the same class.
    """
    import itertools
    from pathlib import Path as _Path

    import cv2
    import numpy as np

    from .appearance import (
        CLIP_STRONG,
        CLIP_WEAK,
        COLOUR_STRONG,
        COLOUR_WEAK,
        compare,
        signature_for,
    )
    from .detection import CARRIED_CLASSES, detect_objects

    folder = _Path(args.folder)
    images = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        print(f"No images found in {folder}", file=sys.stderr)
        return 1

    wanted = set(CARRIED_CLASSES.values())
    crops: list[tuple[str, str, object]] = []

    print(f"Scanning {len(images)} images for carried items...")
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        for detection in detect_objects(str(path)):
            if detection.label not in wanted:
                continue
            signature = signature_for(image, detection.bbox)
            if signature and not signature.is_small:
                crops.append((path.name, detection.label, signature))

    if len(crops) < 2:
        print(f"Only {len(crops)} usable object crops found — need at least 2.")
        print("This calibration needs photos containing backpacks, handbags or suitcases.")
        return 0

    print(f"{len(crops)} object crops\n")
    print(_rule("PAIRWISE SIMILARITY (same class only)"))
    print(f"{'file A':<22} {'file B':<22} {'clip':>6} {'colour':>7}  verdict")
    print("-" * 74)

    rows = []
    for (na, la, sa), (nb, lb, sb) in itertools.combinations(crops, 2):
        if la != lb:
            continue
        _, clip_similarity, colour_similarity = compare(sa, sb)
        if clip_similarity >= CLIP_STRONG and colour_similarity >= COLOUR_STRONG:
            verdict = "SAME OBJECT"
        elif clip_similarity >= CLIP_WEAK and colour_similarity >= COLOUR_WEAK:
            verdict = "possible"
        else:
            verdict = "different"
        rows.append((clip_similarity, colour_similarity, verdict))
        print(
            f"{na[:21]:<22} {nb[:21]:<22} {clip_similarity:6.3f} "
            f"{colour_similarity:7.3f}  {verdict}"
        )

    if rows:
        clips = np.array([r[0] for r in rows])
        colours = np.array([r[1] for r in rows])
        print()
        print(f"clip   range {clips.min():.3f} – {clips.max():.3f}")
        print(f"colour range {colours.min():.3f} – {colours.max():.3f}")
        print()
        print(f"current rule: SAME requires clip>={CLIP_STRONG} AND colour>={COLOUR_STRONG}")
        print(f"              possible     clip>={CLIP_WEAK} AND colour>={COLOUR_WEAK}")
        print()
        print("Check the verdicts against what you know. If pairs you know are the")
        print("same object read 'different', lower the thresholds in acpia/appearance.py.")
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

    enroll = sub.add_parser("enroll", help="Enrol a known person (human only)")
    enroll.add_argument("case_id")
    enroll.add_argument("--name", required=True)
    enroll.add_argument(
        "--status", required=True,
        choices=["victim", "person_of_interest", "known_associate"],
    )
    enroll.add_argument(
        "--photo", required=True, nargs="+",
        help="One or more reference photos. Each must contain exactly one face.",
    )
    enroll.add_argument("--relationship", default=None, help="e.g. father, sister, teacher")
    enroll.add_argument(
        "--global", dest="global_entry", action="store_true",
        help="Visible to all cases — use for the prior-suspect gallery.",
    )
    enroll.add_argument("--note", default=None)
    enroll.set_defaults(func=cmd_enroll)

    register = sub.add_parser("register", help="Show enrolled people")
    register.add_argument("case_id")
    register.set_defaults(func=cmd_register)

    calibrate = sub.add_parser(
        "calibrate-faces", help="Measure real face-similarity separation on a folder"
    )
    calibrate.add_argument("folder")
    calibrate.set_defaults(func=cmd_calibrate_faces)

    calibrate_obj = sub.add_parser(
        "calibrate-objects", help="Measure object instance-matching separation on a folder"
    )
    calibrate_obj.add_argument("folder")
    calibrate_obj.set_defaults(func=cmd_calibrate_objects)

    sub.add_parser("agents", help="List registered agents").set_defaults(func=cmd_agents)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
