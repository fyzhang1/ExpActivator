"""Audit of every execution comparison the manuscript states, on all backbones.

For each backbone this recomputes, from the sealed per-step runs, the paired
difference between LEAF and every other condition on SSR, Type, and CSR with
episode- and user-cluster bootstrap intervals, and the opening/within-trajectory
decomposition.  Nothing is re-run; the script only reads existing outputs.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.eval_execution.utils import make_weights

R = "expactivator/results"
BACKBONES = {
    "Qwen3-VL-8B": ("qwen3_vl_8b", f"{R}/paper_baselines/qwen3_vl_8b_base_gpu0.json"),
    "MAI-UI-8B": ("mai_ui_8b", f"{R}/backbones/mai_ui_8b_gpu0.json"),
    "Qwen3.5-9B": ("qwen35_9b", f"{R}/backbones/qwen35_9b_gpu0.json"),
    "GUI-Owl-1.5-8B": ("gui_owl_1_5_8b", f"{R}/backbones/gui_owl_1_5_8b_gpu0.json"),
}
METHODS = ["Base", "Recent", "Retrieve", "LLM-UM", "HIM-Agent", "LEAF"]
FILES = {"Recent": "recent", "Retrieve": "retrieve", "LLM-UM": "llm_um",
         "HIM-Agent": "him_agent"}
DRAWS = 20000
SEED = 20260923
OUT = f"{R}/leaf_result_audit/audit.json"


def load(path: str) -> list[dict]:
    payload = json.loads((REPO / path).read_text())
    rows = payload["outputs"] if isinstance(payload, dict) else payload
    return [{"id": r["episode_id"], "sr": int(r["sr"]),
             "type": int(r["type_acc"])} for r in rows]


def runs_for(slug: str, base: str) -> dict[str, list[dict]]:
    runs = {"Base": load(base)}
    for name, key in FILES.items():
        runs[name] = load(f"{R}/paper_baselines/{slug}_{key}_gpu0.json")
    runs["LEAF"] = load(f"{R}/grounded/{slug}_grounded_policy_gpu0.json")
    order = [r["id"] for r in runs["LEAF"]]
    for name, rows in runs.items():
        if [r["id"] for r in rows] != order:
            raise ValueError(f"{slug}/{name}: step order differs")
    return runs


def episode_table(rows: list[dict]):
    """Per-episode sums needed to rebuild SSR, Type, and CSR on any resample."""
    grouped = defaultdict(list)
    for r in rows:
        episode, pos = r["id"].rsplit("_", 1)
        grouped[episode].append((int(pos), r["sr"], r["type"]))
    episodes = sorted(grouped)
    table = []
    for e in episodes:
        steps = sorted(grouped[e])
        w = make_weights(len(steps))
        sr = np.array([s for _, s, _ in steps], float)
        ty = np.array([t for _, _, t in steps], float)
        table.append([len(steps), sr.sum(), ty.sum(), float(w @ sr), float(w.sum())])
    return episodes, np.array(table)


def metrics_from(table: np.ndarray, weights: np.ndarray) -> np.ndarray:
    s = weights @ table
    if s.ndim == 1:
        s = s[None, :]
    return np.stack([s[:, 1] / s[:, 0], s[:, 2] / s[:, 0], s[:, 3] / s[:, 4]], 1)


def compare(ref_rows, met_rows, cluster: str, rng) -> dict:
    episodes, ref = episode_table(ref_rows)
    _, met = episode_table(met_rows)
    if cluster == "user":
        users = [e.split("_")[0] for e in episodes]
        names = sorted(set(users))
        index = {u: i for i, u in enumerate(names)}
        ref = np.stack([ref[[index[u] == k for u in users]].sum(0) for k in range(len(names))])
        met = np.stack([met[[index[u] == k for u in users]].sum(0) for k in range(len(names))])
    n = len(ref)
    point = metrics_from(met, np.ones(n))[0] - metrics_from(ref, np.ones(n))[0]
    w = rng.multinomial(n, np.full(n, 1 / n), size=DRAWS).astype(float)
    boot = metrics_from(met, w) - metrics_from(ref, w)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return {m: {"delta": float(point[i] * 100),
                "ci95": [float(lo[i] * 100), float(hi[i] * 100)],
                "excludes_zero": bool(lo[i] > 0 or hi[i] < 0)}
            for i, m in enumerate(["ssr", "type", "csr"])}


def main() -> None:
    rng = np.random.default_rng(SEED)
    report = {"protocol": {"draws": DRAWS, "seed": SEED,
                           "clusters": ["episode", "user"]}, "backbones": {}}
    for label, (slug, base) in BACKBONES.items():
        runs = runs_for(slug, base)
        entry = {"metrics": {}, "leaf_vs": {}, "decomposition": {}}
        for name, rows in runs.items():
            _, t = episode_table(rows)
            m = metrics_from(t, np.ones(len(t)))[0] * 100
            entry["metrics"][name] = {"ssr": m[0], "type": m[1], "csr": m[2]}
        for name in METHODS[:-1]:
            entry["leaf_vs"][name] = {
                c: compare(runs[name], runs["LEAF"], c, rng)
                for c in ("episode", "user")}
        for phase, keep in (("opening", lambda r: r["id"].endswith("_0")),
                            ("within", lambda r: not r["id"].endswith("_0"))):
            sub = {n: [r for r in rows if keep(r)] for n, rows in runs.items()}
            users = np.array([int(r["id"].split("_")[0]) for r in sub["LEAF"]])
            names = np.unique(users)
            idx = np.searchsorted(names, users)
            count = np.bincount(idx, minlength=len(names)).astype(float)
            w = rng.multinomial(len(names), np.full(len(names), 1 / len(names)),
                                size=DRAWS).astype(float)
            ph = {"n_steps": len(sub["LEAF"]), "ssr": {}, "vs_base": {}, "leaf_vs": {}}
            sums = {n: np.bincount(idx, weights=np.array([r["sr"] for r in rows], float),
                                   minlength=len(names)) for n, rows in sub.items()}
            for n in METHODS:
                ph["ssr"][n] = float(sums[n].sum() / count.sum() * 100)
            def diff(a, b):
                point = (sums[b].sum() - sums[a].sum()) / count.sum() * 100
                boot = (w @ (sums[b] - sums[a])) / (w @ count) * 100
                lo, hi = np.quantile(boot, [0.025, 0.975])
                return {"delta": float(point), "ci95": [float(lo), float(hi)],
                        "excludes_zero": bool(lo > 0 or hi < 0)}
            for n in METHODS[1:]:
                ph["vs_base"][n] = diff("Base", n)
            for n in METHODS[:-1]:
                ph["leaf_vs"][n] = diff(n, "LEAF")
            entry["decomposition"][phase] = ph
        report["backbones"][label] = entry

        print(f"\n######## {label}")
        m = entry["metrics"]
        print("  SSR  " + "  ".join(f"{n}={m[n]['ssr']:.2f}" for n in METHODS))
        for name in METHODS[:-1]:
            e = entry["leaf_vs"][name]["episode"]; u = entry["leaf_vs"][name]["user"]
            flag = lambda d: "" if d["excludes_zero"] else "  <-- CI CROSSES 0"
            print(f"  LEAF-{name:9s} SSR {e['ssr']['delta']:+6.2f} ep[{e['ssr']['ci95'][0]:+.2f},{e['ssr']['ci95'][1]:+.2f}]{flag(e['ssr'])}"
                  f" usr[{u['ssr']['ci95'][0]:+.2f},{u['ssr']['ci95'][1]:+.2f}]{flag(u['ssr'])}")
            print(f"  {'':14s} Type {e['type']['delta']:+6.2f} ep[{e['type']['ci95'][0]:+.2f},{e['type']['ci95'][1]:+.2f}]{flag(e['type'])}"
                  f"   CSR {e['csr']['delta']:+6.2f} ep[{e['csr']['ci95'][0]:+.2f},{e['csr']['ci95'][1]:+.2f}]{flag(e['csr'])}")
        w = entry["decomposition"]["within"]
        print("  WITHIN-TRAJECTORY SSR: " + "  ".join(f"{n}={w['ssr'][n]:.2f}" for n in METHODS))
        for n in METHODS[1:]:
            d = w["vs_base"][n]
            print(f"    {n:9s} - Base {d['delta']:+6.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]"
                  + ("  significant" if d["excludes_zero"] else ""))
    dest = REPO / OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
