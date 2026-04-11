---
name: hierarchical-implicit-intent-memory
description: >-
  Designs long-term user-centric memory for any agent using hierarchical implicit
  intent alignment (preference and routine), streaming updates, and dual filters
  (execution-like vs state-like). Use when building personalized agents, memory
  layers, user modeling, or adapting PersonalAlign/HIM-style ideas beyond GUI.
---

# Hierarchical Implicit Intent Memory (General Agent)

This skill transfers the **PersonalAlign / HIM-Agent** idea from GUI agents to **arbitrary tool-using agents** (coding assistants, research agents, chatbots with tools, RPA, etc.). GUI-specific notions map to **abstract observations** and **behavior traces**.

## Core idea (domain-agnostic)

1. **Implicit intents** are not declared goals; they are inferred from repeated patterns in how the user steers the agent over time.
2. **Two-level memory** separates what should steer personalization:
   - **Preference intent**: stable, user-specific choices (tone, depth, risk, format, tools, libraries).
   - **Routine intent**: recurring situational patterns (time-of-day workflows, recurring task types, standing habits).
3. **Hierarchical alignment**: at decision time, retrieve **preferences** for “how to behave,” and **routines** for “what situation this resembles,” then compose them in prompts or policies without overwriting base agent behavior.
4. **Streaming updates**: memory evolves as new interactions arrive; **one-off** episodes should not pollute long-term stores.

## Mapping from GUI to general agent

| GUI / AndroidIntent | General agent analogue |
|---------------------|-------------------------|
| Screen + widget trajectory | Ordered trace of **tool calls**, **edits**, **retrieved docs**, **subgoals** |
| Click/type/scroll | `tool_name(args)`, file paths, commands, API routes |
| Spatial/action similarity | **Call-sequence similarity** (skills/tools) + **evaluation-tool** similarity (lint/test/typecheck ids) |
| Time + scenario fields | **Session metadata**: hour, weekday, project/repo, channel, device |
| User profile CSV | **Profile store**: user_id, tags, optional structured prefs |

## Dual filters (replace GUI-specialized code with concepts)

### A. Execution-style filter (behavioral consistency)

**Purpose:** Decide whether a new episode reflects a **stable preference** (repeatable *way of doing*) vs noise.

**General signals (OpenClaw / tool-agent default in this skill):**

- **Call similarity:** multiset Jaccard + **LCS ratio** on `call_sequence` (skill/tool names). No DTW—lighter and stable for variable-length traces.
- **Evaluation-tool similarity:** Jaccard on `evaluation_tools` (e.g. `ruff`, `pytest`, `mypy`) to capture “how you verify” without GUI coordinates.
- **Text:** embedding cosine + word Jaccard on episode `text` (intent summary).
- **Reject** or down-weight: single-occurrence flukes, failed/aborted runs, exploratory one-offs.

**Implementation hints:** Maintain clusters over **embedded step summaries** + parallel **call/eval** lists; cap cluster size; decay optionally (paper-style).

### B. State-style filter (situational / routine)

**Purpose:** Mine **routines**: when/where/under which context the user tends to need what.

**General signals:**

- **Temporal**: hour buckets, weekday vs weekend, timezone.
- **Contextual**: active project, repo, product area, conversation topic cluster.
- **Regularity**: entropy of context distribution; routines appear as **tight multimodal context** (e.g., “every Monday morning, summarization requests”).

**Implementation hints:** After preference clusters exist, score candidate routine clusters by **consistency of context features** and **support** (frequency), analogous to the paper’s state filter feeding **Routine Intent Memory**.

## Memory objects (minimal schema)

Use prototypes or summaries, not raw logs, in long-term stores.

**Preference memory entry (example fields)**

- `summary`: short natural-language intent (what the user consistently wants).
- `evidence_ids`: pointers to source episodes (optional).
- `embedding`: for retrieval.
- `strength` / `last_seen`: for decay and pruning.

**Routine memory entry (example fields)**

- `context_signature`: compact description of when it applies (schedule + domain).
- `linked_preference_ids`: optional links to prefs that often co-occur.
- `embedding`: for “current situation” matching.

## OpenClaw-style layout (USER.md / MEMORY.md extension)

Keep authoritative **`USER.md`** / **`MEMORY.md`** for what the user **declares**. Add **inferred** stores beside them so the runtime can **read on demand**:

| Artifact | Role |
|----------|------|
| `preference_memory.json` / `preference_memory.md` | Standing preferences (prototypes: summary, typical `call_sequence`, typical `evaluation_tools`) |
| `routine_memory.json` / `routine_memory.md` | Recurrent **context** (time/scenario density, linked preference summary) for proactive hints |

**Convention:** agent loads these files **only when** personalization or proactive policy needs them (e.g. session start, before planning, or a dedicated “memory” skill). **Explicit USER/Memory always overrides** inferred bullets.

Generate/update with `export_openclaw_memory(mem, directory)` in [scripts/him_memory.py](scripts/him_memory.py) after batch `observe()` + `build_routine_memory()`.

Example stub to append to `USER.md`:

```markdown
<!-- Inferred memory (optional; regenerated by pipeline) -->
See `preference_memory.md` for inferred standing preferences. Declared rules above take precedence.
```

## Integration pattern (plug-in for an agent)

1. **Observe** each episode: `text`, `call_sequence`, `evaluation_tools`, `time`, `context` (e.g. channel / project).
2. **Update (async):** execution filter → **preference** prototypes; state filter → **routine** list; export to `preference_memory*` / `routine_memory*`.
3. **Retrieve (sync):** load JSON or MD snippets by need; or `retrieve_snippets(query)` for embedding-only top-k.
4. **Condition** the policy: inject a short block; inferred memory stays **subordinate** to explicit user instructions.
5. **Safety:** no secrets in memory files; redact PII; retention limits.

## Prompt block template (for the acting model)

Use a fixed, short structure so personalization is predictable:

```markdown
## User-centric memory (inferred; may be incomplete)
- Standing preferences: [bullets from preference memory]
- Recurrent contexts: [bullets from routine memory]
Apply only when consistent with the user's current explicit request.
```

## Anti-patterns

- Dumping full chat history instead of **prototype-level** memory.
- Flattening prefs and routines into one blob (hurts alignment and interpretability).
- Updating memory from every turn without **one-off** rejection (pollutes with noise).
- Letting inferred memory **override** explicit user instructions in the current message.

## End-to-end pipeline

```
  Conversations (OpenClaw / Claude Code)
       │
       ▼
  ingest_conversations.py  ──►  episodes.jsonl
       │
       ▼
  daily_update.py          ──►  him_checkpoint.json
       │                        preference_memory.{json,md}
       │                        routine_memory.{json,md}
       ▼
  setup_proactive.py       ──►  HEARTBEAT.md  (OpenClaw heartbeat patrol)
                                cron commands  (openclaw cron add ...)
```

### Step 1 — Ingest conversations

```bash
python scripts/ingest_conversations.py \
    --source openclaw \
    --input  /path/to/conversations.jsonl \
    --output episodes.jsonl --user-id u1
```

Supports `--source openclaw` and `--source claude`. Produces Episode JSONL.

### Step 2 — Daily update (batch observe + export)

```bash
python scripts/daily_update.py \
    --episodes   episodes.jsonl \
    --checkpoint him_checkpoint.json \
    --export-dir ./memory
```

Loads checkpoint → `observe()` each episode → saves checkpoint → writes `preference_memory.{json,md}` + `routine_memory.{json,md}`.

Schedule with OpenClaw cron (daily at 02:00, isolated session):

```bash
openclaw cron add --name him-daily-update \
    --cron "0 2 * * *" --session isolated \
    --cmd "python /path/to/scripts/daily_update.py --episodes /path/to/episodes.jsonl"
```

### Step 3 — Proactive triggers from routines

```bash
python scripts/setup_proactive.py \
    --routine-memory ./memory/routine_memory.json \
    --heartbeat-out  ./HEARTBEAT.md
```

Generates:

- **`HEARTBEAT.md`** — patrol checklist for OpenClaw's heartbeat loop (runs every 30 min by default; agent replies `HEARTBEAT_OK` when nothing applies).
- **Cron commands** — printed to stdout; one `openclaw cron add` per routine peak hour for time-precise triggers.

### Step 4 — Agent reads memory on demand

Agent loads `preference_memory.md` / `routine_memory.md` at session start or via a dedicated skill. Append to `USER.md`:

```markdown
<!-- Inferred memory (optional; regenerated by pipeline) -->
See `preference_memory.md` for inferred standing preferences. Declared rules above take precedence.
```

## Layout

```
hierarchical-implicit-intent-memory/
├── SKILL.md
├── requirements.txt
├── scripts/
│   ├── him_memory.py              # core: HIMMemory + export_openclaw_memory
│   ├── daily_update.py            # pipeline: load → observe → save → export
│   ├── setup_proactive.py         # generate HEARTBEAT.md + cron suggestions
│   ├── ingest_conversations.py    # OpenClaw / Claude logs → Episode JSONL
│   └── demo.py                    # standalone toy example
└── reference/
    └── reference.md
```

## Reference implementation (Python, standalone)

Copy this folder anywhere; it does **not** import the PersonalAlign codebase. Only needs `numpy`.

```bash
cd hierarchical-implicit-intent-memory
pip install -r requirements.txt
python scripts/demo.py
```

**Embedding is optional (two modes):**

| Mode | Instantiation | Embedding channel | Auto-weights |
|------|--------------|-------------------|--------------|
| **No model** (default) | `HIMMemory()` | Feature-hash BoW (deterministic, no deps) | emb 0.30 / jaccard 0.30 / call 0.25 / eval 0.15 |
| **With model** | `HIMMemory(embedding_fn=make_sentence_transformer_fn())` | Dense sentence vector | emb 0.65 / jaccard 0.10 / call 0.15 / eval 0.10 |

Weights auto-set unless you pass a custom `HIMConfig`.

## Optional deep dive

Paper-aligned terminology and checklist: [reference/reference.md](reference/reference.md).
