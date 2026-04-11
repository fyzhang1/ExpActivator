#!/usr/bin/env python3
"""
Generate proactive trigger artifacts from routine_memory.json:

  1. HEARTBEAT.md  — patrol checklist consumed by OpenClaw's heartbeat loop.
  2. Cron commands  — printed to stdout; paste into `openclaw cron add` or crontab.

Usage:
  python scripts/setup_proactive.py \
      --routine-memory ./memory/routine_memory.json \
      --heartbeat-out  ./HEARTBEAT.md

OpenClaw heartbeat docs:
  The Gateway injects HEARTBEAT.md every N minutes (default 30).
  If the agent has nothing to report it replies HEARTBEAT_OK (suppressed).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _hour_label(h: int) -> str:
    return f"{h:02d}:00"


def build_heartbeat_md(routines: List[Dict[str, Any]]) -> str:
    if not routines:
        return "# Heartbeat Patrol\n\nNo routines detected yet. Reply HEARTBEAT_OK.\n"

    lines = [
        "# Heartbeat Patrol (auto-generated from routine_memory)",
        "",
        "Check the items below. If nothing applies right now, reply **HEARTBEAT_OK**.",
        "",
    ]
    for r in routines:
        cid = r.get("cluster_id", "?")
        cc = r.get("cluster_content") or {}
        summary = cc.get("center_text", "(no summary)")
        times = r.get("collect_time", [])
        scenes = r.get("collect_scene", [])
        time_str = ", ".join(_hour_label(h) for h, _ in times) if times else "any"
        scene_str = ", ".join(s for s, _ in scenes) if scenes else "any"

        lines.append(f"## Routine {cid}")
        lines.append(f"- **What**: {summary}")
        lines.append(f"- **When**: around {time_str}")
        lines.append(f"- **Where**: {scene_str}")
        lines.append(f"- [ ] Is it time for this? If yes, propose or execute.")
        lines.append("")

    lines.append("---")
    lines.append("If none of the above apply, reply **HEARTBEAT_OK**.")
    return "\n".join(lines)


def build_cron_suggestions(routines: List[Dict[str, Any]]) -> List[str]:
    """Generate `openclaw cron add` commands (one per routine peak hour)."""
    cmds: List[str] = []
    for r in routines:
        cid = r.get("cluster_id", 0)
        times = r.get("collect_time", [])
        cc = r.get("cluster_content") or {}
        summary = cc.get("center_text", "routine check")[:60]
        if not times:
            continue
        peak_hour = times[0][0]
        cron_expr = f"0 {peak_hour} * * *"
        cmds.append(
            f'openclaw cron add --name "him-routine-{cid}" '
            f'--cron "{cron_expr}" --session isolated '
            f'--cmd "echo Routine {cid}: {summary}"'
        )
    return cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate proactive triggers from routine memory")
    parser.add_argument("--routine-memory", required=True, help="Path to routine_memory.json")
    parser.add_argument("--heartbeat-out", default="./HEARTBEAT.md", help="Path to write HEARTBEAT.md")
    parser.add_argument("--print-cron", action="store_true", default=True, help="Print cron suggestions to stdout")
    args = parser.parse_args()

    with open(args.routine_memory, "r", encoding="utf-8") as f:
        data = json.load(f)
    routines = data.get("routines", [])

    hb = build_heartbeat_md(routines)
    Path(args.heartbeat_out).write_text(hb, encoding="utf-8")
    print(f"Wrote {args.heartbeat_out} ({len(routines)} routines)")

    if args.print_cron:
        cmds = build_cron_suggestions(routines)
        if cmds:
            print("\n# Suggested cron commands (paste into terminal):")
            for c in cmds:
                print(c)
        else:
            print("No cron suggestions (no routine has a clear peak hour).")


if __name__ == "__main__":
    main()
