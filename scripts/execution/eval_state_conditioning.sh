#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. All outputs are isolated from production runs.
RUNNER="${RUNNER:-/storage/zhangfuyao/miniforge3/envs/hymem-serve/bin/python}"
MODEL_PATH="${MODEL_PATH:-/storage/zhangfuyao/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B}"
MODEL_KEY="${MODEL_KEY:-qwen3_vl_8b}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen}"
HISTORY_ROOT="${HISTORY_ROOT:-/storage/zhangfuyao/datasets/fingertip20k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-trace_method/results/state_conditioning/$MODEL_KEY}"
VARIANTS="${VARIANTS:-top1_ordinal top1_state top8_previous top8_state}"

export CUDA_VISIBLE_DEVICES=0
export USE_TF=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$OUTPUT_ROOT"
for variant in $VARIANTS; do
    contexts="$OUTPUT_ROOT/${variant}_contexts.json"
    if [[ "$variant" == top8_state ]]; then
        contexts="trace_method/results/grounded/${MODEL_KEY}_contexts.json"
    fi
    echo "Running $MODEL_KEY / $variant"
    "$RUNNER" src/eval_execution/evaluation_new_backbones_vllm.py \
        --model-name "$MODEL_NAME" --model-path "$MODEL_PATH" \
        --prompt-style "$PROMPT_STYLE" --history-root "$HISTORY_ROOT" \
        --device cuda:0 --request-batch-size 256 --max-num-seqs 64 \
        --max-model-len 8192 --max-new-tokens 256 --baseline-name trace_grounded \
        --memory-contexts "$contexts" --save-file "$OUTPUT_ROOT/${variant}_policy_gpu0.json" \
        --resume --print-first 0 > "$OUTPUT_ROOT/${variant}_policy_gpu0.log" 2>&1
    echo "Completed $MODEL_KEY / $variant"
done
