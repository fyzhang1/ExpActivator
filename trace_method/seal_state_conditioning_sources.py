"""Record inference sources before a new run; never replace an existing seal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCES = ["src/eval_execution/evaluation_new_backbones_vllm.py",
           "src/eval_execution/evaluation_new_backbones.py", "src/eval_execution/prompt.py",
           "src/eval_execution/utils.py", "trace_method/build_state_conditioning_experiment.py",
           "trace_method/build_state_conditioning_full_context.py", "trace_method/annotate_state_conditioning_tasks.py",
           "trace_method/STATE_CONDITIONING_PROTOCOL.md"]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="trace_method/results/state_conditioning/qwen3_vl_8b")
    args = parser.parse_args()
    directory = REPO / args.directory
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "inference_source_hashes.json"
    if target.exists():
        previous = json.loads(target.read_text())
        for name, expected in previous.items():
            contents = (REPO / name).read_bytes()
            if digest(contents) != expected and name.endswith("STATE_CONDITIONING_PROTOCOL.md"):
                contents = ((REPO / name).read_text().split("## Additional exploratory checks")[0].rstrip() + "\n").encode()
            assert digest(contents) == expected, f"Sealed source changed: {name}; use an isolated directory for a new study."
        print(f"Existing source seal verified and preserved: {target}")
        return
    assert not list(directory.glob("*_policy_gpu0.json")), "Seal new sources before policy inference."
    value = {name: digest((REPO / name).read_bytes()) for name in SOURCES}
    target.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Sealed {len(value)} source files: {target}")


if __name__ == "__main__":
    main()
