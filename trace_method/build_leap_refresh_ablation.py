"""Reindex sealed fixed-pool evidence to isolate reference refresh timing."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieval_position(position: int, variant: str) -> int:
    if position < 0:
        raise ValueError("Negative position")
    if position == 0 or variant == "current":
        return position
    if variant == "delay1":
        return max(1, position - 1)
    if variant == "refresh4":
        return 1 + 4 * ((position - 1) // 4)
    raise ValueError(variant)


def write_once(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing to replace changed sealed input: {path}")
        return
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="trace_method/results/leap_ablations/refresh")
    args = parser.parse_args()
    out = REPO / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    source = REPO / "trace_method/results/grounded/qwen3_vl_8b_contexts.json"
    test_path = REPO / "data/personal_execution/personal_execution.json"
    original = json.loads(source.read_text())
    # Only prediction-available input fields leave the input-loading boundary.
    queries = [{k: row[k] for k in ("id", "user_id", "ambiguous_instruction", "img_path")}
               for row in json.loads(test_path.read_text())]
    assert len(queries) == 7915
    assert original["protocol"]["trajectory_top_k"] == 8
    assert original["protocol"]["test_sha256"] == sha(test_path)
    expected = {r["id"] for r in queries}
    assert expected == set(original["contexts"]) == set(original["sources"])
    episodes = {}
    for row in queries:
        episode, position = row["id"].rsplit("_", 1)
        episodes.setdefault(episode, []).append(row)
    assert len(episodes) == 775
    for episode, rows in episodes.items():
        assert len({r["ambiguous_instruction"] for r in rows}) == 1
        assert len({r["user_id"] for r in rows}) == 1
        assert sorted(int(r["id"].rsplit("_", 1)[1]) for r in rows) == list(range(len(rows)))

    metadata = []
    contexts = {v: {} for v in ("current", "delay1", "refresh4")}
    sources = {v: {} for v in contexts}
    for row in queries:
        episode, suffix = row["id"].rsplit("_", 1)
        position = int(suffix)
        item = {"step_id": row["id"], "episode_id": episode,
                "user_id": int(row["user_id"]), "position": position,
                "episode_steps": len(episodes[episode]), "retrieval_step_ids": {}}
        for variant in contexts:
            source_id = f"{episode}_{retrieval_position(position, variant)}"
            assert source_id in expected
            contexts[variant][row["id"]] = copy.deepcopy(original["contexts"][source_id])
            sources[variant][row["id"]] = copy.deepcopy(original["sources"][source_id])
            item["retrieval_step_ids"][variant] = source_id
            if position:
                src = sources[variant][row["id"]]
                assert int(src["user_id"]) == int(row["user_id"])
                assert f"{src['user_id']}_{src['time']}" != episode
        metadata.append(item)

    for variant in contexts:
        protocol = {**original["protocol"], "experiment": "leap_reference_refresh",
                    "variant": variant, "source_contexts_sha256": sha(source),
                    "target_answers_used_for_selection": False,
                    "target_original_instruction_used_for_selection": False,
                    "policy_screen": "current", "trajectory_pool_fixed_per_episode": True}
        write_once(out / f"{variant}_contexts.json",
                   {"protocol": protocol, "contexts": contexts[variant], "sources": sources[variant]})
    write_once(out / "step_metadata.json", {"rows": metadata})
    inputs = [source, test_path, REPO / "trace_method/LEAP_ABLATION_PROTOCOL.md",
              Path(__file__), REPO / "src/eval_execution/evaluation_new_backbones_vllm.py",
              REPO / "src/eval_execution/evaluation_new_backbones.py",
              REPO / "src/eval_execution/prompt.py", REPO / "src/eval_execution/utils.py"]
    inputs += [out / f"{v}_contexts.json" for v in contexts] + [out / "step_metadata.json"]
    write_once(out / "source_hashes.json", {str(p.relative_to(REPO)): sha(p) for p in inputs})
    print(json.dumps({"steps": len(queries), "episodes": len(episodes),
                      "same_context_as_current": {v: sum(contexts[v][r['id']] == contexts['current'][r['id']]
                                                             for r in queries if not r['id'].endswith('_0'))
                                                    for v in contexts}}, indent=2))


if __name__ == "__main__":
    main()
