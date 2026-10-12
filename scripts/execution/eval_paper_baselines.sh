#!/usr/bin/env bash
set -euo pipefail

: "${MAI_MODEL:?Set MAI_MODEL to the local MAI-UI-8B snapshot}"
: "${QWEN35_MODEL:?Set QWEN35_MODEL to the local Qwen3.5-9B snapshot}"
: "${OWL_MODEL:?Set OWL_MODEL to the local GUI-Owl-1.5-8B snapshot}"
: "${HISTORY_ROOT:?Set HISTORY_ROOT to the Fingertip history root}"

PYTHON_BIN="${VLLM_PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-expactivator/results/paper_baselines}"
unset VLLM_PYTHON

export CUDA_VISIBLE_DEVICES=0
export USE_TF=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0

run_model() {
  local key="$1"
  local name="$2"
  local model="$3"
  local style="$4"
  local method

  for method in recent retrieve llm_um him_agent; do
    "$PYTHON_BIN" src/eval_execution/evaluation_new_backbones_vllm.py \
      --model-name "$name" \
      --model-path "$model" \
      --prompt-style "$style" \
      --history-root "$HISTORY_ROOT" \
      --device cuda:0 \
      --request-batch-size 256 \
      --max-num-seqs 64 \
      --max-model-len 8192 \
      --max-new-tokens 256 \
      --baseline-name "$method" \
      --memory-contexts "$OUTPUT_ROOT/${method}_contexts.json" \
      --save-file "$OUTPUT_ROOT/${key}_${method}_gpu0.json" \
      --resume
  done
}

run_model mai_ui_8b MAI-UI-8B "$MAI_MODEL" qwen
run_model qwen35_9b Qwen3.5-9B "$QWEN35_MODEL" qwen
run_model gui_owl_1_5_8b GUI-Owl-1.5-8B-Instruct "$OWL_MODEL" owl
