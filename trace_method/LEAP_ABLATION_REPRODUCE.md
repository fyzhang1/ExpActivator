# Reproduce the LEAP ablations

Use a separate checkout or copy of the study directories for a new replication.
The builders verify existing sealed inputs rather than replacing them; the
evaluators resume only matching checkpoints. All commands below run from the
repository root. They reuse the existing grounded history index, calibrated
visual contexts, local model checkpoint and released benchmark screenshots.

## Environments

- Analysis/building: `/storage/zhangfuyao/miniforge3/envs/hymem/bin/python`.
- Frozen policy inference: `/storage/zhangfuyao/miniforge3/envs/hymem-serve/bin/python`.
- Official proactive Semantic scoring:
  `/storage/zhangfuyao/miniforge3/envs/agent/bin/python`, with SentenceTransformers,
  fuzzywuzzy, scikit-learn and the local Qwen3-Embedding-0.6B model. This study
  uses CPU scoring to keep GPU 0 available for the policy runs.
- LaTeX: `/storage/zhangfuyao/miniforge3/envs/latex/bin/tectonic`.

Policy inference uses physical GPU 0 only. Do not run the two policy scripts
concurrently. Their local Qwen3-VL checkpoint and inference settings are
specified in the scripts and recorded in every output JSON. No model is trained.

## Reference refresh

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 USE_TF=0
ANALYSIS_PY=/storage/zhangfuyao/miniforge3/envs/hymem/bin/python
"$ANALYSIS_PY" trace_method/build_leap_refresh_ablation.py
OUTPUT_ROOT=trace_method/results/leap_ablations/refresh \
  VARIANTS='current delay1 refresh4' \
  bash scripts/execution/eval_state_conditioning.sh
"$ANALYSIS_PY" trace_method/analyze_leap_refresh_ablation.py
```

The builder uses the production current-state contexts. Since each task's
Top-8 neighborhood is fixed and the visual selector is stateless, reindexing
the selected evidence by the retrieval-screen position reproduces selection
on that screen. The policy's current screenshot and action prefix do not
change. All 7,915 steps run in each condition; the primary contrast excludes
the 775 application-entry steps.

## Proactive context factors

```bash
"$ANALYSIS_PY" trace_method/run_leap_proactive_ablation.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 \
  /storage/zhangfuyao/miniforge3/envs/agent/bin/python \
  trace_method/analyze_leap_proactive_ablation.py
"$ANALYSIS_PY" trace_method/plot_leap_proactive_ablation.py
```

This experiment uses the exact production MiniLM historical intent index.
Qwen3-Embedding-0.6B is the Semantic scoring encoder. The full selector is
required to reproduce all production source IDs and suggestion strings before
any result is scored. The score curves include every threshold; no working
threshold is selected from test outcomes. Raw-mass ranking preserves the exact
trigger set. The full 315-query scored outputs and scoring embeddings are saved.

## User-associated application evidence

```bash
"$ANALYSIS_PY" trace_method/build_leap_user_history_ablation.py
"$ANALYSIS_PY" trace_method/build_leap_partial_user_history.py
bash scripts/execution/eval_leap_user_history.sh
"$ANALYSIS_PY" trace_method/analyze_leap_user_history.py
```

Complete ten-record matching covers only 25 queries. The separately recorded
coverage extension changes only admissibly matched records and retains all
775 queries; it does not relax either matching caliper. The strict comparison
reuses exactly matching contexts in the complete entry-only runs. All three
policy conditions use the same 775-entry batch schedule. This is an application
choice diagnostic, not a complete removal of personalization or an evaluation
of later content preference recovery.

## Export

```bash
"$ANALYSIS_PY" trace_method/export_leap_ablation_results.py
```

Then compile `leap_ablation_report.tex` from
`trace_method/results/leap_ablations` using Tectonic. The exporter requires all
three analyses to be complete and writes all declared conditions. The result
paragraphs, three tables, methods and activation plot are separate paper-ready
assets; their presence does not certify the full manuscript for submission.

The study reports paired user-cluster bootstrap intervals with 20,000 draws,
episode resampling for execution sensitivity, and query resampling for the
normalization sensitivity. Confidence intervals are exploratory and unadjusted
for multiplicity. They do not estimate policy runtime variability. All input,
code and output hashes are preserved with the corresponding analyses.

The original fixed-history study and its negative equal-length subgroup remain
under `trace_method/results/state_conditioning`; these ablations do not erase
or supersede that result.
