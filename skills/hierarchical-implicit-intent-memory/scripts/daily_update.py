#!/usr/bin/env python3
"""
Daily update pipeline:
  1. Load existing HIM checkpoint (or start fresh).
  2. Read today's new episodes from a JSONL file.
  3. observe() each episode → clusters update.
  4. Save checkpoint + export OpenClaw memory files.

Usage:
  python scripts/daily_update.py \
      --episodes   episodes.jsonl \
      --checkpoint  him_checkpoint.json \
      --export-dir  ./memory

Schedule via OpenClaw cron (isolated session, daily at 02:00):
  openclaw cron add --name him-daily-update \
      --cron "0 2 * * *" --session isolated \
      --cmd "python /path/to/scripts/daily_update.py --episodes /path/to/episodes.jsonl"

Or via system crontab:
  0 2 * * * cd /path/to/skill && python scripts/daily_update.py
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from him_memory import (
    HIMMemory,
    episodes_from_simple_rows,
    export_openclaw_memory,
)


def load_episodes_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="HIM daily update")
    parser.add_argument("--episodes", required=True, help="Path to episodes JSONL (one JSON object per line)")
    parser.add_argument("--checkpoint", default="him_checkpoint.json", help="HIM checkpoint file (read & write)")
    parser.add_argument("--export-dir", default="./memory", help="Directory for preference_memory / routine_memory exports")
    parser.add_argument("--scenario-key", default="scenario", help="Context key used for routine mining")
    args = parser.parse_args()

    mem = HIMMemory()

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        mem.load(str(ckpt))
        print(f"Loaded checkpoint: {ckpt} ({len(mem.clusters)} clusters)")
    else:
        print("No existing checkpoint; starting fresh.")

    raw_rows = load_episodes_jsonl(args.episodes)
    if not raw_rows:
        print("No new episodes. Exiting.")
        return
    episodes = episodes_from_simple_rows(raw_rows)
    print(f"Observing {len(episodes)} new episodes ...")

    for ep in episodes:
        mem.observe(ep)

    mem.save(str(ckpt))
    print(f"Saved checkpoint: {ckpt} ({len(mem.clusters)} clusters)")

    paths = export_openclaw_memory(
        mem,
        args.export_dir,
        scenario_key=args.scenario_key,
    )
    for name, p in paths.items():
        print(f"  {name} -> {p}")
    print("Daily update done.")


if __name__ == "__main__":
    main()
