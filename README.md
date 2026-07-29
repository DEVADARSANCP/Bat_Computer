# ACPIA — Agentic Child Protection Investigation Assistant

An agentic investigation platform for authorised child-protection investigators.

An orchestrating LLM is given a case of digital evidence and no instructions. It decides
what to examine, invokes specialised forensic agents through an MCP tool layer, accumulates
structured claims into a knowledge graph, follows up the correlations it finds, and reports
what a human should act on — with every assertion traceable back to the file it came from.

> **All development and demonstration uses synthetic or self-generated benign data.**
> The engine is domain-agnostic: the same capabilities that track a person across
> self-shot footage would, in an authorised deployment, be pointed at case evidence.

---

## Requirements

- **Python 3.12** (system install — see the environment note below)
- A free Groq API key: <https://console.groq.com/keys>

## Setup

```powershell
cd C:\001WorkSpace\code\hackathons\Bat_Computer

pip install -r requirements.txt

# Copy the template and paste your Groq key into GROQ_API_KEY.
# .env is gitignored and must never be committed.
copy .env.example .env
```

> **Environment note.** Do **not** activate the `batcomputer` conda env — it contains no
> Python interpreter and only shadows the system install. Every dependency lives in
> system Python 3.12. If you want an isolated env, build a real one deliberately;
> otherwise run everything with the system `python`.

## Run an investigation

```powershell
# 1. Ingest evidence (creates the case automatically)
python -m acpia.cli ingest CASE-001 .\demo_data

# 2. Turn the planner loose on it
python -m acpia.cli investigate CASE-001

# 3. Inspect what it found
python -m acpia.cli status CASE-001
python -m acpia.cli audit  CASE-001
```

`investigate` streams the planner's reasoning live — every tool call, every result, and the
decisions between them. That trace *is* the demo: it shows the system choosing what to do,
not just producing output.

### All commands

| Command | Purpose |
|---|---|
| `case new <id> <name>` | Create a case explicitly (optional — `ingest` does it for you) |
| `case list` | List all cases |
| `ingest <case> <path>` | Ingest a file or a whole directory |
| `investigate <case>` | Run the autonomous planner |
| `status <case>` | Evidence, claims, entities, flagged findings |
| `audit <case>` | Append-only trail of every action taken |
| `agents` | List registered agents and what they do |

Useful flags on `investigate`:

```powershell
--goal "Focus on anything involving vehicles"   # steer the investigation
--model llama-3.1-8b-instant                    # smaller/faster planner
--max-steps 50                                  # longer investigations
```

Every command takes `--db <path>` if you want a separate database.

## Run as an MCP server

The same tools the internal planner uses are published over MCP stdio, so any MCP client
can drive an investigation:

```powershell
python -m acpia.mcp_server --case CASE-001 --db acpia.db
```

---

## Architecture

```
evidence files
      │  sha256 + classify
      ▼
   Evidence ──────────────────────────────────┐
                                              │
  ┌───────────────────────────────────────────┴────────────┐
  │                    TOOL LAYER                          │
  │  agent tools (auto-generated, one per agent)           │
  │  case tools: overview · analyse · claims · entities    │
  │              connections · finish_investigation        │
  └──────────┬──────────────────────────────┬──────────────┘
             │                              │
   Investigation Planner            MCP stdio server
   (LLM, multi-step,                (external clients:
    observes and re-plans)           Claude Desktop, IDEs)
             │
             ▼
   Claims ──► SQLite ──► entities · graph edges · audit log
```

### Why it is built this way

**One claim schema, frozen.** Every agent — image, video, document, whatever comes next —
returns a list of `Claim`. That uniformity is what lets the planner reason across media
types without knowing how any of them were produced.

**Deterministic entity IDs.** An entity's id is `sha1(type + normalised_value)`. Two agents
that independently encounter the same phone number produce the same id, so exact-match
correlation is free and the graph stitches itself together.

**One tool layer, two consumers.** The internal planner and the MCP server execute identical
code. They cannot drift apart.

**Tool errors are data, not exceptions.** A planner told *"no evidence with that id"*
corrects itself on the next step. One handed a traceback stops investigating.

**Adding a capability is one file.** Subclass `Agent`, add one import. No changes to the
planner, the store, the graph, or the MCP server.

### Human oversight

Required by the brief, and implemented rather than asserted:

- Every claim carries a `source` pointing at the exact file and offset it came from.
- Every claim carries a plain-language `reason` — what the investigator reads when asking
  *why does the system think this?*
- Every agent invocation, tool call and planner decision is written to an append-only
  audit log.
- The planner produces **leads and recommendations**. Investigative decisions are the
  human's.

---

## Project layout

```
acpia/
  schema.py            frozen claim/entity/relationship contract
  store.py             SQLite: cases, evidence, claims, entities, graph, audit
  ingest.py            files → Evidence (hash, classify, dedup)
  tools.py             the tool layer — single dispatch point
  mcp_server.py        MCP stdio wrapper over the tool layer
  cli.py               command line
  agents/
    base.py            Agent ABC + registry
    metadata_agent.py  EXIF / GPS / device / capture time
  orchestrator/
    planner.py         the reasoning loop

Check_list/            requirements coverage, build log, parked ideas
Problem_Overview/      the hackathon brief
Proposal/              round-one proposal
```

## Credits

Built on capabilities developed across earlier projects, including two team projects the
author contributed to (SEIVR++ — semantic video event extraction; Bharathi — multi-agent
orchestration and notification) and two solo projects (a Minecraft MCP agent server;
an EXIF/GPS metadata extraction tool).
