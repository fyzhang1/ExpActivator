"""
HIM-style hierarchical memory (reference implementation).

Maps PersonalAlign / HIM-Agent ideas to tool-using agents (e.g. OpenClaw):
  - Preference memory: embedding + text Jaccard + call-sequence similarity + eval-tool similarity.
  - Routine memory: state-style filter (time/context entropy -> dense routines).

Sequence alignment uses call / evaluation-tool similarity (not GUI DTW).
No dependency on the PersonalAlign repo. Requires numpy only.

Embedding is **optional**: if no embedding_fn is provided, a deterministic
model-free bag-of-words feature-hash vector is used and weights auto-adjust so
the text Jaccard / call / eval channels carry more load.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


WEIGHTS_WITH_MODEL = {"weight_emb": 0.65, "weight_jaccard": 0.10, "weight_call": 0.15, "weight_eval": 0.10}
WEIGHTS_NO_MODEL = {"weight_emb": 0.30, "weight_jaccard": 0.30, "weight_call": 0.25, "weight_eval": 0.15}


@dataclass
class HIMConfig:
    """Hyperparameters. Weights are auto-set if you use HIMMemory(embedding_fn=None)."""

    core_cluster_radius: float = 0.6
    max_pool_size: int = 50
    max_samples_per_cluster: int = 30
    decay_lambda: float = 0.99
    weight_emb: float = 0.65
    weight_jaccard: float = 0.10
    weight_call: float = 0.15
    weight_eval: float = 0.10
    core_weight_threshold: float = 3.0
    routine_density_threshold: float = 0.6
    embedding_dim: int = 256


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """One interaction episode (turn, task, or session chunk)."""

    episode_id: str
    user_id: str
    text: str
    time: str = ""
    call_sequence: List[str] = field(default_factory=list)
    evaluation_tools: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    action_sequence: List[str] = field(default_factory=list)

    def calls(self) -> List[str]:
        """Skill/tool invocation names (OpenClaw calls); legacy: action_sequence."""
        if self.call_sequence:
            return list(self.call_sequence)
        return list(self.action_sequence)

    def hour_bucket(self) -> int:
        if not self.time:
            return 12
        m = re.search(r"_(\d{2})\d{4}$|(\d{2}):", str(self.time))
        if m:
            return int(m.group(1) or m.group(2))
        return 12


# ---------------------------------------------------------------------------
# Embeddings (pluggable)
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


# --- model-free fallback: feature-hash bag-of-words -------------------------

def _stable_word_hash(word: str, dim: int) -> Tuple[int, int]:
    """Deterministic bucket + sign from SHA-1 (stable across runs)."""
    h = hashlib.sha1(word.encode("utf-8")).digest()
    bucket = struct.unpack_from("<I", h, 0)[0] % dim
    sign = 1 if (h[4] & 1) else -1
    return bucket, sign


def bow_embedding(text: str, dim: int = 256) -> np.ndarray:
    """
    Model-free deterministic embedding via feature hashing.
    Captures lexical overlap: texts sharing words produce similar cosine.
    """
    tokens = re.findall(r"\w+", text.lower())
    v = np.zeros(dim, dtype=np.float64)
    for tok in tokens:
        bucket, sign = _stable_word_hash(tok, dim)
        v[bucket] += sign
    return _normalize(v)


# --- optional model-based embedding ------------------------------------------

def make_sentence_transformer_fn(model_name: str = "all-MiniLM-L6-v2") -> Callable[[str], np.ndarray]:
    """pip install sentence-transformers; returns a (text -> vector) function."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def _fn(text: str) -> np.ndarray:
        e = model.encode(text, normalize_embeddings=True)
        return np.asarray(e, dtype=np.float64)

    return _fn


# ---------------------------------------------------------------------------
# Similarities: call + evaluation tools (replaces DTW)
# ---------------------------------------------------------------------------


def jaccard_words(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def jaccard_sets(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lcs_ratio(seq1: Sequence[str], seq2: Sequence[str]) -> float:
    """Order similarity in [0,1] from longest common subsequence."""
    n, m = len(seq1), len(seq2)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    lcs = int(dp[n, m])
    return 2.0 * lcs / (n + m)


def call_sequence_similarity(seq1: Sequence[str], seq2: Sequence[str]) -> float:
    """
    Blend multiset overlap (tool/skill names) with LCS order similarity.
    Fits OpenClaw-style call traces without DTW.
    """
    jac = jaccard_sets(seq1, seq2)
    lcs = lcs_ratio(seq1, seq2)
    return 0.5 * jac + 0.5 * lcs


def evaluation_tool_similarity(e1: Sequence[str], e2: Sequence[str]) -> float:
    """Similarity of lint/test/typecheck tool ids (order-free)."""
    return jaccard_sets(e1, e2)


def cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def combined_similarity(
    emb: np.ndarray,
    text: str,
    calls: Sequence[str],
    eval_tools: Sequence[str],
    cluster_center_emb: np.ndarray,
    cluster_text: str,
    cluster_calls: Sequence[str],
    cluster_eval: Sequence[str],
    cfg: HIMConfig,
) -> float:
    ce = cosine_dense(emb, cluster_center_emb)
    jac = jaccard_words(text, cluster_text)
    csim = call_sequence_similarity(calls, cluster_calls)
    esim = evaluation_tool_similarity(eval_tools, cluster_eval)
    return (
        cfg.weight_emb * ce
        + cfg.weight_jaccard * jac
        + cfg.weight_call * csim
        + cfg.weight_eval * esim
    )


# ---------------------------------------------------------------------------
# Preference cluster
# ---------------------------------------------------------------------------


class _PreferenceCluster:
    def __init__(
        self,
        center_emb: np.ndarray,
        record: Episode,
        embedding_fn: Callable[[str], np.ndarray],
        cfg: HIMConfig,
        is_core: bool = False,
    ):
        self.center = np.asarray(center_emb, dtype=np.float64)
        self.weight = 1.0
        self.samples: List[Episode] = [record]
        self.embed_list: List[np.ndarray] = [self.center.copy()]
        self.call_list: List[List[str]] = [list(record.calls())]
        self.eval_list: List[List[str]] = [list(record.evaluation_tools)]
        self.is_core = is_core
        self.center_text = record.text
        self.center_calls: List[str] = list(record.calls())
        self.center_eval: List[str] = list(record.evaluation_tools)
        self._embedding_fn = embedding_fn
        self._cfg = cfg
        self.last_update = record.time[:8] if len(record.time) >= 8 else record.time

    def _update_centers(self) -> None:
        best_i = 0
        best_d = float("inf")
        for i, e in enumerate(self.embed_list):
            d = float(np.linalg.norm(e - self.center))
            if d < best_d:
                best_d = d
                best_i = i
        self.center_text = self.samples[best_i].text
        self.center = self.embed_list[best_i].copy()

        n = len(self.call_list)
        if n == 0:
            self.center_calls = []
            self.center_eval = []
            return

        best_calls: List[str] = list(self.call_list[-1])
        best_cs = -1.0
        for seq_i in self.call_list:
            avg = sum(call_sequence_similarity(seq_i, seq_j) for seq_j in self.call_list) / float(n)
            if avg > best_cs:
                best_cs = avg
                best_calls = list(seq_i)
        self.center_calls = best_calls if best_cs >= 0.25 else list(self.call_list[-1])

        ne = len(self.eval_list)
        if ne == 0:
            self.center_eval = []
        else:
            best_ev: List[str] = list(self.eval_list[-1])
            best_es = -1.0
            for ev_i in self.eval_list:
                avg_e = sum(evaluation_tool_similarity(ev_i, ev_j) for ev_j in self.eval_list) / float(
                    ne
                )
                if avg_e > best_es:
                    best_es = avg_e
                    best_ev = list(ev_i)
            self.center_eval = best_ev if best_es >= 0.2 else list(self.eval_list[-1])

    def decay(self) -> None:
        self.weight *= self._cfg.decay_lambda

    def update(self, emb: np.ndarray, record: Episode, sim: float) -> None:
        w = float(sim)
        self.center = (self.center * self.weight + emb * w) / (self.weight + w)
        self.weight += w
        self.samples.append(record)
        e = np.asarray(emb, dtype=np.float64)
        self.embed_list.append(e)
        self.call_list.append(list(record.calls()))
        self.eval_list.append(list(record.evaluation_tools))
        m = self._cfg.max_samples_per_cluster
        if len(self.samples) > m:
            self.samples = self.samples[-m:]
            self.embed_list = self.embed_list[-m:]
            self.call_list = self.call_list[-m:]
            self.eval_list = self.eval_list[-m:]
        self.last_update = record.time[:8] if len(record.time) >= 8 else record.time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_text": self.center_text,
            "center_calls": self.center_calls,
            "center_eval": self.center_eval,
            "cluster_len": len(self.samples),
            "weight": self.weight,
            "is_core": self.is_core,
            "example_texts": [s.text for s in self.samples],
            "last_update": self.last_update,
        }


# ---------------------------------------------------------------------------
# State filter -> routines
# ---------------------------------------------------------------------------


def _entropy_norm(counts: np.ndarray, max_bins: int) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts[counts > 0] / s
    h = float(-np.sum(p * np.log2(p)))
    max_h = math.log2(max_bins) if max_bins > 1 else 1.0
    return h / max_h if max_h > 1e-12 else 0.0


def _topk_from_bincount(bc: np.ndarray, k: int) -> List[Tuple[int, int]]:
    pairs = sorted(zip(range(len(bc)), bc.tolist()), key=lambda x: -x[1])[:k]
    picked = [(int(i), int(v)) for i, v in pairs if v > 0][:k]
    return picked or [(int(np.argmax(bc)), int(bc.max()))]


def score_preference_clusters_for_routines(
    clusters: List[_PreferenceCluster],
    scenario_key: str = "scenario",
) -> List[Dict[str, Any]]:
    max_len = max((len(c.samples) for c in clusters), default=1)
    raw: List[Dict[str, Any]] = []
    for c in clusters:
        L = len(c.samples)
        hours = np.array([s.hour_bucket() for s in c.samples], dtype=np.int64)
        bc = np.bincount(hours, minlength=24)
        time_entropy = _entropy_norm(bc, 24)

        scenes = [str(s.context.get(scenario_key, "unknown")) for s in c.samples]
        uniq, cnt = np.unique(scenes, return_counts=True)
        if len(uniq) > 1:
            scene_entropy = _entropy_norm(cnt.astype(np.float64), len(uniq))
        else:
            scene_entropy = 0.0

        time_density = 1.0 - time_entropy
        scene_density = 1.0 - scene_entropy
        density = (time_density + scene_density) / 2.0
        length = L / max_len
        consist = c.weight / max(L, 1)
        total_density = (length + density + consist) / 3.0

        sc = Counter(scenes)

        raw.append(
            {
                "time_cluster_density": time_entropy,
                "scene_cluster_density": scene_entropy,
                "center_hours": _topk_from_bincount(bc, 2),
                "center_scenes": sc.most_common(2),
                "total_cluster_density": total_density,
                "cluster_consistency": consist,
                "cluster_len": L,
            }
        )
    return raw


def mine_routines(
    clusters: List[_PreferenceCluster],
    preference_dicts: List[Dict[str, Any]],
    heat: float,
    scenario_key: str = "scenario",
) -> List[Dict[str, Any]]:
    scores = score_preference_clusters_for_routines(clusters, scenario_key=scenario_key)
    out: List[Dict[str, Any]] = []
    for idx, s in enumerate(scores):
        if s["total_cluster_density"] > heat:
            content = dict(preference_dicts[idx])
            content.pop("center_text_emb", None)
            out.append(
                {
                    "cluster_id": idx,
                    "total_cluster_density": s["total_cluster_density"],
                    "collect_time": s["center_hours"],
                    "collect_scene": s["center_scenes"],
                    "cluster_content": content,
                }
            )
    return out


# ---------------------------------------------------------------------------
# OpenClaw export: preference_memory + routine_memory as json/md
# ---------------------------------------------------------------------------


def export_openclaw_memory(
    mem: "HIMMemory",
    directory: str | Path,
    *,
    scenario_key: str = "scenario",
    routine_heat: Optional[float] = None,
) -> Dict[str, str]:
    """
    Write preference_memory.{json,md} and routine_memory.{json,md} for agent read-in.
    Extend USER.md / MEMORY.md with @include or links to these paths in your setup.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    prefs = mem.preference_prototypes()
    routines = mem.build_routine_memory(heat=routine_heat, scenario_key=scenario_key)

    pref_json = d / "preference_memory.json"
    rout_json = d / "routine_memory.json"
    pref_md = d / "preference_memory.md"
    rout_md = d / "routine_memory.md"

    with open(pref_json, "w", encoding="utf-8") as f:
        json.dump(
            {"version": 1, "source": "hierarchical-implicit-intent-memory", "clusters": prefs},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(rout_json, "w", encoding="utf-8") as f:
        json.dump(
            {"version": 1, "source": "hierarchical-implicit-intent-memory", "routines": routines},
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines_pref = [
        "# preference_memory (inferred)",
        "",
        "Inject **only when** consistent with the user's current message. Explicit USER.md overrides.",
        "",
        "## Standing preferences (prototypes)",
        "",
    ]
    for p in prefs:
        cid = p.get("cluster_id", 0)
        lines_pref.append(f"### cluster {cid}")
        lines_pref.append(f"- **summary**: {p.get('center_text', '')}")
        lines_pref.append(f"- **typical_calls**: `{p.get('center_calls', [])}`")
        if p.get("center_eval"):
            lines_pref.append(f"- **typical_eval_tools**: `{p.get('center_eval', [])}`")
        lines_pref.append(f"- **support**: n={p.get('cluster_len')}, core={p.get('is_core')}")
        lines_pref.append("")

    lines_rout = [
        "# routine_memory (inferred)",
        "",
        "Use for **proactive** timing and context hints. Do not override explicit user refusal.",
        "",
        "## Recurrent contexts",
        "",
    ]
    for r in routines:
        lines_rout.append(f"### routine {r.get('cluster_id')}")
        lines_rout.append(f"- **density**: {round(float(r.get('total_cluster_density', 0)), 4)}")
        lines_rout.append(f"- **typical_times (hour,count)**: {r.get('collect_time')}")
        lines_rout.append(f"- **typical_scenarios**: {r.get('collect_scene')}")
        cc = r.get("cluster_content") or {}
        if cc.get("center_text"):
            lines_rout.append(f"- **linked_summary**: {cc.get('center_text')}")
        lines_rout.append("")

    pref_md.write_text("\n".join(lines_pref), encoding="utf-8")
    rout_md.write_text("\n".join(lines_rout), encoding="utf-8")

    return {
        "preference_memory.json": str(pref_json),
        "preference_memory.md": str(pref_md),
        "routine_memory.json": str(rout_json),
        "routine_memory.md": str(rout_md),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class HIMMemory:
    """
    Streaming preference clusters + periodic routine mining.

    If ``embedding_fn`` is **not** provided:
      - Uses deterministic feature-hash BoW (no model needed).
      - Weights auto-adjust: emb channel is weaker, text/call/eval channels carry more.
    If ``embedding_fn`` is provided (e.g. ``make_sentence_transformer_fn()``):
      - Uses the model's dense vectors for the emb channel.
      - Weights stay at model-optimised defaults (emb=0.65).
    You can always override weights via ``config``.
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        config: Optional[HIMConfig] = None,
    ):
        self._has_model = embedding_fn is not None
        self._cfg = config or HIMConfig()

        if embedding_fn is not None:
            self._embedding_fn = embedding_fn
        else:
            dim = self._cfg.embedding_dim
            self._embedding_fn = lambda t: bow_embedding(t, dim)

        _user_set_weights = (
            config is not None
            and any(
                getattr(config, k) != getattr(HIMConfig(), k)
                for k in ("weight_emb", "weight_jaccard", "weight_call", "weight_eval")
            )
        )
        if not _user_set_weights:
            defaults = WEIGHTS_WITH_MODEL if self._has_model else WEIGHTS_NO_MODEL
            for k, v in defaults.items():
                setattr(self._cfg, k, v)

        self._pool: List[_PreferenceCluster] = []

    @property
    def clusters(self) -> List[_PreferenceCluster]:
        return self._pool

    def observe(self, episode: Episode) -> None:
        emb = np.asarray(self._embedding_fn(episode.text), dtype=np.float64)
        emb = _normalize(emb)
        cfg = self._cfg
        calls = episode.calls()
        ev = list(episode.evaluation_tools)

        for c in self._pool:
            c.decay()
            c._update_centers()

        assigned = False
        core = [c for c in self._pool if c.is_core]
        for c in core:
            sim = combined_similarity(
                emb,
                episode.text,
                calls,
                ev,
                c.center,
                c.center_text,
                c.center_calls,
                c.center_eval,
                cfg,
            )
            if sim >= cfg.core_cluster_radius:
                c.update(emb, episode, sim)
                assigned = True
                break

        if not assigned:
            edge = [c for c in self._pool if not c.is_core]
            for c in edge:
                sim = combined_similarity(
                    emb,
                    episode.text,
                    calls,
                    ev,
                    c.center,
                    c.center_text,
                    c.center_calls,
                    c.center_eval,
                    cfg,
                )
                if sim >= cfg.core_cluster_radius:
                    c.update(emb, episode, sim)
                    if c.weight > cfg.core_weight_threshold:
                        c.is_core = True
                    assigned = True
                    break

        if not assigned:
            self._pool.append(
                _PreferenceCluster(emb, episode, self._embedding_fn, cfg, is_core=False)
            )

        self._pool.sort(key=lambda x: x.weight, reverse=True)
        self._pool[:] = self._pool[: cfg.max_pool_size]

    def preference_prototypes(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, c in enumerate(self._pool):
            d = c.to_dict()
            d["cluster_id"] = i
            d["center_text_emb"] = c.center.tolist()
            out.append(d)
        return out

    def build_routine_memory(
        self,
        heat: Optional[float] = None,
        scenario_key: str = "scenario",
    ) -> List[Dict[str, Any]]:
        h = heat if heat is not None else self._cfg.routine_density_threshold
        prefs = self.preference_prototypes()
        return mine_routines(self._pool, prefs, h, scenario_key=scenario_key)

    def retrieve_snippets(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 3,
    ) -> Dict[str, List[str]]:
        q = query_embedding
        if q is None:
            q = np.asarray(self._embedding_fn(query_text), dtype=np.float64)
        q = _normalize(q)
        scored: List[Tuple[float, str]] = []
        for c in self._pool:
            s = cosine_dense(q, _normalize(c.center))
            scored.append((s, c.center_text))
        scored.sort(key=lambda x: -x[0])
        return {"preferences": [t for _, t in scored[:top_k]]}

    def save(self, path: str) -> None:
        payload = {
            "config": asdict(self._cfg),
            "embedding_mode": "model" if self._has_model else "bow",
            "clusters": [],
        }
        for c in self._pool:
            payload["clusters"].append(
                {
                    "center": c.center.tolist(),
                    "weight": c.weight,
                    "is_core": c.is_core,
                    "center_text": c.center_text,
                    "center_calls": c.center_calls,
                    "center_eval": c.center_eval,
                    "samples": [asdict(s) for s in c.samples],
                    "last_update": c.last_update,
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        Load clusters from JSON. If the checkpoint was saved with a different
        embedding mode (model vs bow), embeddings are **recomputed** from text
        using the current embedding_fn. Center vectors are kept from the file.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        _merge_config_from_json(self._cfg, payload.get("config", {}))
        self._pool = []
        for block in payload.get("clusters", []):
            samples = _samples_from_json(block.get("samples", []))
            first = samples[0]
            center = _normalize(np.asarray(block["center"], dtype=np.float64))
            pc = _PreferenceCluster(
                center,
                first,
                self._embedding_fn,
                self._cfg,
                is_core=block.get("is_core", False),
            )
            pc.weight = float(block["weight"])
            pc.samples = samples
            pc.embed_list = [
                _normalize(np.asarray(self._embedding_fn(s.text), dtype=np.float64)) for s in samples
            ]
            pc.call_list = [list(s.calls()) for s in samples]
            pc.eval_list = [list(s.evaluation_tools) for s in samples]
            pc.center = center.copy()
            pc.center_text = block["center_text"]
            pc.center_calls = list(block.get("center_calls", block.get("center_action", [])))
            pc.center_eval = list(block.get("center_eval", []))
            pc.last_update = block.get("last_update", "")
            self._pool.append(pc)


def _episode_from_dict(s: Dict[str, Any]) -> Episode:
    d = dict(s)
    if "call_sequence" not in d and "action_sequence" in d:
        d["call_sequence"] = list(d.get("action_sequence", []))
    d.setdefault("call_sequence", [])
    d.setdefault("evaluation_tools", [])
    d.setdefault("action_sequence", [])
    d.setdefault("context", {})
    d.setdefault("time", "")
    fields = set(Episode.__dataclass_fields__)
    return Episode(**{k: v for k, v in d.items() if k in fields})


def _samples_from_json(raw: List[Dict[str, Any]]) -> List[Episode]:
    return [_episode_from_dict(s) for s in raw]


def episodes_from_simple_rows(rows: List[Dict[str, Any]]) -> List[Episode]:
    out: List[Episode] = []
    for r in rows:
        calls = list(r.get("call_sequence", r.get("action_sequence", [])))
        legacy_action = list(r.get("action_sequence", []))
        out.append(
            Episode(
                episode_id=str(r["episode_id"]),
                user_id=str(r["user_id"]),
                text=str(r["text"]),
                time=str(r.get("time", "")),
                call_sequence=calls,
                evaluation_tools=list(r.get("evaluation_tools", [])),
                context=dict(r.get("context", {})),
                action_sequence=legacy_action,
            )
        )
    return out


def _merge_config_from_json(cfg: HIMConfig, data: Dict[str, Any]) -> None:
    """Apply saved config; map legacy weight_dtw -> weight_call."""
    if not data:
        return
    d = dict(data)
    if "weight_dtw" in d and "weight_call" not in d:
        d["weight_call"] = float(d.pop("weight_dtw", 0.15))
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
