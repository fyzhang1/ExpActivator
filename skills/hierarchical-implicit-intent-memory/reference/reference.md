# Reference: PersonalAlign / HIM-Agent (source concepts)

## Paper framing

- **PersonalAlign**: hierarchical implicit intent alignment so a personalized agent matches **implicit** user intents from **long-term** interaction records.
- **AndroidIntent**: GUI benchmark with annotated preference vs routine-style evaluation (trajectories, actions, intents).

## HIM-Agent (memory module)

- **Streaming update**: memory is not static; new episodes continuously refine prototypes.
- **Preference Intent Memory (PIM)**: captures stable user-specific **ways** of interacting (in GUI: action patterns tied to preference intents).
- **Routine Intent Memory (RIM)**: captures **recurring situational** structure (in GUI: time/scene regularity after preference structure exists).
- **Execution-based filter**: uses **behavioral traces**; in this skill’s code: **call-sequence** + **evaluation-tool** similarity (not GUI DTW).
- **State-based filter**: uses **state/context regularity** (e.g., temporal and scene statistics) to surface routines.

## Porting checklist

When implementing outside GUI:

1. Define **call_sequence** (skill/tool names) and **evaluation_tools** (lint/test ids) per episode.
2. Define **situation features** for routines (schedule, project, topic, modality).
3. Choose **embedding model** for text summaries of steps/episodes.
4. Choose **clustering / prototype** strategy (cap size, decay, merge rules).
5. Define **retrieval query**: current situation vector + optional recent task embedding.
6. Define **evaluation**: offline logs with held-out episodes; measure consistency with user satisfaction proxies if no labels.

This file is optional context; the agent should prefer [SKILL.md](../SKILL.md) for integration steps.

## Standalone code

A portable reference implementation lives in [scripts/him_memory.py](../scripts/him_memory.py) and [scripts/demo.py](../scripts/demo.py). It exports **`preference_memory` / `routine_memory`** as `.json` and `.md` for OpenClaw-style runtimes. Copy the whole `hierarchical-implicit-intent-memory` directory and run without the rest of the PersonalAlign repository.
