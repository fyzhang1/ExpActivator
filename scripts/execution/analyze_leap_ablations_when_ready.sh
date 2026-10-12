#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_PY=/storage/zhangfuyao/miniforge3/envs/hymem/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 USE_TF=0
if [[ -n "${REFRESH_PID:-}" ]]; then
    while kill -0 "$REFRESH_PID" 2>/dev/null; do sleep 10; done
fi
"$ANALYSIS_PY" expactivator/analyze_leap_refresh_ablation.py \
    > expactivator/results/leap_ablations/refresh/analysis.log 2>&1
echo "Refresh analysis completed"
if [[ -n "${ENTRY_PID:-}" ]]; then
    while kill -0 "$ENTRY_PID" 2>/dev/null; do sleep 10; done
fi
"$ANALYSIS_PY" expactivator/analyze_leap_user_history.py \
    > expactivator/results/leap_ablations/user_history_partial/analysis.log 2>&1
echo "User-history analysis completed"
"$ANALYSIS_PY" expactivator/export_leap_ablation_results.py
(
    cd expactivator/results/leap_ablations
    /storage/zhangfuyao/miniforge3/envs/latex/bin/tectonic leap_ablation_report.tex --keep-logs -c minimal
)
echo "Complete ablation report exported and compiled"
