#!/usr/bin/env bash
set -euo pipefail

: "${QWEN3_VL_MODEL:?Set QWEN3_VL_MODEL to the local Qwen3-VL-8B snapshot}"
: "${MAI_MODEL:?Set MAI_MODEL to the local MAI-UI-8B snapshot}"
: "${QWEN35_MODEL:?Set QWEN35_MODEL to the local Qwen3.5-9B snapshot}"
: "${OWL_MODEL:?Set OWL_MODEL to the local GUI-Owl-1.5-8B snapshot}"
: "${TEXT_MODEL:?Set TEXT_MODEL to the local multilingual MiniLM snapshot}"
: "${HISTORY_ROOT:?Set HISTORY_ROOT to the Fingertip history root}"

MODEL_PYTHON="${MODEL_PYTHON:-python}"
VISUAL_PYTHON="${VISUAL_PYTHON:-$MODEL_PYTHON}"
VLLM_RUNNER="${VLLM_PYTHON:-$MODEL_PYTHON}"
VISUAL_BATCH_SIZE="${VISUAL_BATCH_SIZE:-64}"
BACKBONES="${BACKBONES:-qwen3_vl_8b mai_ui_8b qwen35_9b gui_owl_1_5_8b}"
# VLLM_PYTHON configures this runner rather than vLLM itself. Avoid forwarding
# it as an unknown VLLM_* environment variable to the engine.
unset VLLM_PYTHON
OUTPUT_ROOT="${OUTPUT_ROOT:-trace_method/results/grounded}"
TEXT_CACHE="$OUTPUT_ROOT/execution_history_index_gpu0.pt"
ACTION_INDEX="$OUTPUT_ROOT/execution_app_predictions_gpu0.json"
APP_LABELS="$OUTPUT_ROOT/execution_app_labels.json"
HISTORY_CALIBRATION="$OUTPUT_ROOT/execution_history_calibration.json"
TRAJECTORY_TOP_K=8

export CUDA_VISIBLE_DEVICES=0
export USE_TF=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0

if [[ ! -f "$APP_LABELS" ]]; then
  "$MODEL_PYTHON" trace_method/compile_history_app_labels.py \
    --model-path "$QWEN3_VL_MODEL" \
    --reuse trace_method/results/execution_app_labels.json \
    --output "$APP_LABELS" --batch-size 256
fi

if [[ ! -f "$TEXT_CACHE" || ! -f "$ACTION_INDEX" ]]; then
  "$MODEL_PYTHON" trace_method/build_execution_index.py \
    --embedding-model "$TEXT_MODEL" --device cuda:0 \
    --history-labels "$APP_LABELS" --cache "$TEXT_CACHE" \
    --output "$ACTION_INDEX"
fi

"$MODEL_PYTHON" trace_method/calibrate_execution.py \
  --cache "$TEXT_CACHE" --history-labels "$APP_LABELS" \
  --history-root "$HISTORY_ROOT" --output "$HISTORY_CALIBRATION"

run_backbone() {
  local key="$1"
  local name="$2"
  local model="$3"
  local style="$4"
  local cache_root="$OUTPUT_ROOT/cache/$key"
  local calibration="$OUTPUT_ROOT/${key}_visual_calibration.json"
  local contexts="$OUTPUT_ROOT/${key}_contexts.json"
  local predictions="$OUTPUT_ROOT/${key}_grounded_policy_gpu0.json"

  "$VISUAL_PYTHON" trace_method/build_visual_index.py \
    --model-path "$model" --model-name "$name" --device cuda:0 \
    --scope calibration --history-root "$HISTORY_ROOT" \
    --batch-size "$VISUAL_BATCH_SIZE" --pixels 200704 \
    --trajectory-top-k "$TRAJECTORY_TOP_K" --text-cache "$TEXT_CACHE" \
    --output-dir "$cache_root/calibration"

  "$MODEL_PYTHON" trace_method/calibrate_visual.py \
    --history-root "$HISTORY_ROOT" \
    --trajectory-top-k "$TRAJECTORY_TOP_K" --text-cache "$TEXT_CACHE" \
    --visual-dir "$cache_root/calibration" --output "$calibration"

  "$VISUAL_PYTHON" trace_method/build_visual_index.py \
    --model-path "$model" --model-name "$name" --device cuda:0 \
    --scope full --history-root "$HISTORY_ROOT" \
    --batch-size "$VISUAL_BATCH_SIZE" --pixels 200704 \
    --trajectory-top-k "$TRAJECTORY_TOP_K" --index "$ACTION_INDEX" \
    --output-dir "$cache_root/full"

  "$MODEL_PYTHON" trace_method/build_execution_grounded_evidence.py \
    --history-root "$HISTORY_ROOT" --index "$ACTION_INDEX" \
    --history-calibration "$HISTORY_CALIBRATION" \
    --calibration "$calibration" --visual-dir "$cache_root/full" \
    --output "$contexts"

  "$VLLM_RUNNER" src/eval_execution/evaluation_new_backbones_vllm.py \
    --model-name "$name" --model-path "$model" --prompt-style "$style" \
    --history-root "$HISTORY_ROOT" --device cuda:0 \
    --request-batch-size 256 --max-num-seqs 64 --max-model-len 8192 \
    --max-new-tokens 256 --baseline-name trace_grounded \
    --memory-contexts "$contexts" --save-file "$predictions" --resume
}

should_run() {
  [[ " $BACKBONES " == *" $1 "* ]]
}

should_run qwen3_vl_8b && run_backbone qwen3_vl_8b Qwen3-VL-8B "$QWEN3_VL_MODEL" qwen
should_run mai_ui_8b && run_backbone mai_ui_8b MAI-UI-8B "$MAI_MODEL" qwen
should_run qwen35_9b && run_backbone qwen35_9b Qwen3.5-9B "$QWEN35_MODEL" qwen
should_run gui_owl_1_5_8b && run_backbone gui_owl_1_5_8b GUI-Owl-1.5-8B-Instruct "$OWL_MODEL" owl

"$MODEL_PYTHON" trace_method/analyze_grounded_execution.py \
  --history-root "$HISTORY_ROOT"
