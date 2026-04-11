# Hierarchical Implicit Intent Memory — Skill Extension

This directory (`skills/hierarchical-implicit-intent-memory/`) packages the core ideas from the **PersonalAlign** paper as a **portable, standalone skill** that any tool-using agent (OpenClaw, Claude Code, etc.) can adopt — without depending on the original GUI benchmark or codebase.

## Motivation

PersonalAlign introduces a two-level memory — **Preference Intent Memory** and **Routine Intent Memory** — to align a GUI agent with a user's *implicit* intents inferred from long-term interaction records. The approach is powerful, but the reference implementation is tightly coupled to Android GUI trajectories (screen coordinates, click/scroll actions, app-specific parsing).

Modern agent ecosystems (OpenClaw, Claude Code, coding assistants, RPA) operate in a fundamentally different action space — tool calls, file edits, shell commands, lint/test invocations — yet they face the **same personalization gap**: users rarely spell out their preferences and routines; they just act, and the agent is expected to learn.

This skill bridges that gap.

## What changed from the paper

| Paper (GUI / AndroidIntent) | Skill (general agent) | Why |
|-----|-----|-----|
| Click / scroll / type actions | `call_sequence` (tool/skill names) | GUI coordinates have no analogue in tool-agent traces |
| DTW on action coordinates | **Jaccard + LCS ratio** on call sequences | Lighter, order-aware, works on variable-length symbolic traces |
| Spatial action distance | **Evaluation-tool similarity** (Jaccard on `evaluation_tools` like `ruff`, `pytest`, `mypy`) | Captures "how you verify" — a key preference signal with no GUI equivalent |
| Sentence embedding (required) | **Optional**: model-based (sentence-transformers) or **model-free** (feature-hash BoW, zero deps) | Removes the hard dependency on a GPU / embedding model |
| Static user profile CSV | Extends **`USER.md` / `MEMORY.md`** (OpenClaw convention) with inferred `preference_memory` and `routine_memory` files | Fits the file-based memory pattern already used in production agents |
| Offline-only evaluation | **Daily update pipeline** + **proactive triggers** via OpenClaw Heartbeat / Cron | Closes the loop from "research artifact" to "running agent memory" |

## Design decisions

### 1. Two output formats (JSON + Markdown)

Each memory type is exported as both `.json` (machine-readable, for programmatic retrieval) and `.md` (human-readable, injectible into system prompts or `USER.md` stubs). This follows the convention that agents like OpenClaw already load `.md` files as context.

### 2. Explicit > Inferred

Anything declared in `USER.md` / `MEMORY.md` by the user always overrides inferred memory. The generated `preference_memory.md` header makes this explicit. Inferred memory is a soft default, never a hard constraint.

### 3. Proactive = Routine + Heartbeat

The paper evaluates "proactive suggestion" as a separate task. In a live agent, the equivalent is: routines (mined from temporal/contextual density) are materialized into an OpenClaw `HEARTBEAT.md` patrol checklist and/or precise cron jobs. The heartbeat mechanism provides a natural "should I act now?" check every 30 minutes at near-zero cost (the agent replies `HEARTBEAT_OK` when nothing applies).

### 4. No model required (but benefits from one)

When no `embedding_fn` is provided, the system uses a deterministic feature-hash bag-of-words vector and **automatically re-weights** the similarity channels: text Jaccard and call/eval similarity carry more load (emb 0.30 / jaccard 0.30 / call 0.25 / eval 0.15). With a sentence-transformer model, the embedding channel dominates (0.65). Users can override weights via `HIMConfig`.

## Pipeline overview

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
  setup_proactive.py       ──►  HEARTBEAT.md
                                cron commands (stdout)
```

| Script | Role |
|--------|------|
| `ingest_conversations.py` | Parse OpenClaw or Claude Code conversation logs into Episode JSONL |
| `daily_update.py` | Load checkpoint → observe new episodes → save → export memory files |
| `setup_proactive.py` | Generate `HEARTBEAT.md` + `openclaw cron add` commands from routines |
| `him_memory.py` | Core library: `HIMMemory`, `Episode`, similarity functions, export |
| `demo.py` | Standalone toy example |


## Skill Overall
```
hierarchical-implicit-intent-memory/
├── SKILL.md
├── requirements.txt
├── scripts/
│   ├── him_memory.py
│   ├── daily_update.py
│   ├── setup_proactive.py
│   ├── ingest_conversations.py
│   ├── demo.py
│   └── .gitignore
└── reference/
    └── reference.md
```

## Relation to the paper

This skill is a **derivative work** that generalizes PersonalAlign's contribution — hierarchical implicit intent alignment via streaming preference/routine clustering — to non-GUI agent settings. The core algorithmic ideas (execution-style filter for preferences, state-style filter for routines, streaming cluster updates with decay, core/edge promotion) are preserved; the domain-specific components (GUI action parsing, coordinate DTW, app-specific features) are replaced with tool-agent equivalents.

For the original paper, benchmark, and citation:

```bibtex
@article{lyu2026personalalign,
  title={PersonalAlign: Hierarchical Implicit Intent Alignment for Personalized GUI Agent with Long-Term User-Centric Records},
  author={Lyu, Yibo and Chen, Gongwei and Shao, Rui and Guan, Weili and Nie, Liqiang},
  journal={arXiv preprint arXiv:2601.09636},
  year={2026}
}
```
