#!/usr/bin/env python3
"""
Convert OpenClaw / Claude Code conversation logs into Episode JSONL
that daily_update.py can consume.

Supported sources:
  --source openclaw   Reads OpenClaw JSONL conversation export.
  --source claude     Reads Claude Code JSONL (projects/.../conversations/).

Output: one Episode JSON object per line → feed to daily_update.py --episodes.

Usage:
  python scripts/ingest_conversations.py \
      --source openclaw \
      --input  /path/to/conversations.jsonl \
      --output episodes.jsonl \
      --user-id u1
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# OpenClaw conversations
# ---------------------------------------------------------------------------

def _parse_openclaw_entry(entry: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
    """
    OpenClaw logs are typically JSONL with fields like:
      { "role": "user"|"assistant", "content": "...", "timestamp": "...",
        "channel": "...", "skills_used": [...] }

    We group user+assistant pairs into one episode.
    """
    content = entry.get("content", "")
    if not content or entry.get("role") not in ("user", "assistant"):
        return None
    ts = entry.get("timestamp", "")
    time_str = _normalize_timestamp(ts)
    skills = entry.get("skills_used") or entry.get("tools_used") or []
    channel = entry.get("channel", "unknown")

    eid = hashlib.sha1(f"{user_id}:{ts}:{content[:64]}".encode()).hexdigest()[:12]
    return {
        "episode_id": eid,
        "user_id": user_id,
        "text": content[:512],
        "time": time_str,
        "call_sequence": [str(s) for s in skills],
        "evaluation_tools": [],
        "context": {"scenario": channel},
    }


def ingest_openclaw(lines: List[str], user_id: str) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    pending_user: Optional[Dict[str, Any]] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        role = entry.get("role", "")

        if role == "user":
            pending_user = entry
        elif role == "assistant" and pending_user is not None:
            user_text = pending_user.get("content", "")[:256]
            asst_text = entry.get("content", "")[:256]
            combined_text = f"[user] {user_text} [assistant] {asst_text}"
            ts = pending_user.get("timestamp", entry.get("timestamp", ""))
            skills = list(set(
                (pending_user.get("skills_used") or []) + (entry.get("skills_used") or [])
                + (entry.get("tools_used") or [])
            ))
            channel = pending_user.get("channel", entry.get("channel", "unknown"))
            eid = hashlib.sha1(f"{user_id}:{ts}:{user_text[:64]}".encode()).hexdigest()[:12]
            episodes.append({
                "episode_id": eid,
                "user_id": user_id,
                "text": combined_text,
                "time": _normalize_timestamp(ts),
                "call_sequence": skills,
                "evaluation_tools": list(set(entry.get("eval_tools") or [])),
                "context": {"scenario": channel},
            })
            pending_user = None

    return episodes


# ---------------------------------------------------------------------------
# Claude Code conversations
# ---------------------------------------------------------------------------

def ingest_claude(lines: List[str], user_id: str) -> List[Dict[str, Any]]:
    """
    Claude Code agent transcripts are JSONL, each line:
      { "type": "human"|"assistant"|"tool_use"|"tool_result", "content": "...",
        "timestamp": "...", ... }

    We merge a sequence of (human + tool_use* + assistant) into one episode.
    """
    episodes: List[Dict[str, Any]] = []
    buf_text = ""
    buf_tools: List[str] = []
    buf_eval: List[str] = []
    buf_ts = ""

    EVAL_PATTERNS = re.compile(r"(pytest|ruff|mypy|eslint|tsc|jest|cargo test|go test)", re.I)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        t = entry.get("type", entry.get("role", ""))

        if t in ("human", "user"):
            buf_text = entry.get("content", "")[:256]
            buf_ts = entry.get("timestamp", "")
            buf_tools = []
            buf_eval = []

        elif t == "tool_use":
            name = entry.get("name", entry.get("tool", ""))
            if name:
                buf_tools.append(name)

        elif t == "tool_result":
            output = entry.get("content", "")
            for m in EVAL_PATTERNS.finditer(output):
                buf_eval.append(m.group(1).lower())

        elif t in ("assistant",):
            asst = entry.get("content", "")[:256]
            combined = f"[user] {buf_text} [assistant] {asst}" if buf_text else asst
            eid = hashlib.sha1(f"{user_id}:{buf_ts}:{buf_text[:64]}".encode()).hexdigest()[:12]
            episodes.append({
                "episode_id": eid,
                "user_id": user_id,
                "text": combined,
                "time": _normalize_timestamp(buf_ts),
                "call_sequence": buf_tools,
                "evaluation_tools": list(set(buf_eval)),
                "context": {"scenario": "claude_code"},
            })
            buf_text = ""
            buf_tools = []
            buf_eval = []

    return episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_timestamp(ts: str) -> str:
    """Try to normalise to YYYYMMDD_HHMMSS; pass through if unparseable."""
    if not ts:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            dt = datetime.strptime(ts[:19], fmt)
            return dt.strftime("%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return ts[:15]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest conversations into Episode JSONL")
    parser.add_argument("--source", choices=["openclaw", "claude"], required=True)
    parser.add_argument("--input", required=True, help="Input JSONL conversation file")
    parser.add_argument("--output", default="episodes.jsonl", help="Output Episode JSONL")
    parser.add_argument("--user-id", default="default", help="User ID tag")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    if args.source == "openclaw":
        episodes = ingest_openclaw(raw_lines, args.user_id)
    else:
        episodes = ingest_claude(raw_lines, args.user_id)

    with open(args.output, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"Wrote {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
