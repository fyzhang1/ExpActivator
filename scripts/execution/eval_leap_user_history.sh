#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${WAIT_PID:-}" ]]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
fi
RUNNER=/storage/zhangfuyao/miniforge3/envs/hymem-serve/bin/python
MODEL_PATH=/storage/zhangfuyao/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
STUDY_DIR=expactivator/results/leap_ablations/user_history_partial
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for variant in own partial_other base; do
    context_args=()
    if [[ "$variant" != base ]]; then
        context_args=(--baseline-name expactivator_grounded --memory-contexts "$STUDY_DIR/${variant}_contexts.json")
    fi
    echo "Running entry-only user history condition: $variant"
    "$RUNNER" src/eval_execution/evaluation_new_backbones_vllm.py \
        --model-name Qwen3-VL-8B --model-path "$MODEL_PATH" --prompt-style qwen \
        --test-path "$STUDY_DIR/entry_test.json" \
        --history-root /storage/zhangfuyao/datasets/fingertip20k \
        --device cuda:0 --request-batch-size 256 --max-num-seqs 64 \
        --max-model-len 8192 --max-new-tokens 256 \
        "${context_args[@]}" --save-file "$STUDY_DIR/${variant}_policy_gpu0.json" \
        --resume --print-first 0 > "$STUDY_DIR/${variant}_policy_gpu0.log" 2>&1
    echo "Completed entry-only user history condition: $variant"
done
