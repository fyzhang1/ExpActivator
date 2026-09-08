# TRACE: training-free long-horizon personalization

This directory contains only the final TRACE method, its controlled ablations,
and the commands needed to reproduce the reported results.  The upstream
HIM-Agent baseline remains unchanged in `../HIM_agent/`, `../scripts/HIM_agent/`,
and `../skills/hierarchical-implicit-intent-memory/`.

TRACE combines:

1. omission-aware residual-goal voting for the personalized first app action;
2. goal-restricted same-user trajectory memory; and
3. current-screen synchronization with a frozen Qwen3-VL layer-8 visual state.

It fits no model or prediction parameter.  The complete execution result covers
775 episodes and 7,915 actions; proactive evaluation covers 215 positive and
100 negative states.  See [`results/REPORT.md`](results/REPORT.md) for metrics,
confidence intervals, leakage audits, and limitations.

## Layout

```text
trace_method/
├── execution_core.py              shared trajectory/action utilities
├── evaluation_stats.py            paired bootstrap and McNemar statistics
├── compile_history_app_labels.py  frozen training-side app compilation
├── build_execution_index.py       omission-aware history index and app vote
├── calibrate_execution.py         history-only k calibration
├── build_visual_index.py          frozen Qwen visual feature compiler
├── calibrate_visual.py            history-only visual representation selection
├── evaluate_rgb_ablation.py       RGB and controlled execution ablations
├── evaluate_execution.py          final Qwen-state execution evaluation
├── analyze_execution.py           complete execution analysis and audits
├── analyze_backbones.py           complete-backbone and TRACE controls
├── build_proactive_index.py       official chronological proactive memory
├── evaluate_proactive.py          training-free proactive prediction
├── analyze_proactive.py           official metrics and uncertainty
└── results/                       report plus ignored generated artifacts
```

Generated JSON, CSV, index, and feature-cache files under `results/` are kept
out of Git.  `results/REPORT.md` is retained as lightweight documentation.

## Environment

The commands below assume the repository root is the working directory.

```bash
PYTHON_BIN=python
QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct
TEXT_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
HISTORY_ROOT=/path/to/fingertip20k
```

Historical screenshots are not committed.  Set `HISTORY_ROOT` to the extracted
Fingertip history directory and pass `--history-root "$HISTORY_ROOT"` to the
commands that need screenshots.

## Personalized execution

Compile the factual history and select both neighborhood sizes without opening
execution-test answers:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" trace_method/compile_history_app_labels.py \
  --model-path "$QWEN_MODEL"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" trace_method/build_execution_index.py \
  --embedding-model "$TEXT_MODEL" --device cuda:0
"$PYTHON_BIN" trace_method/calibrate_execution.py \
  --history-root "$HISTORY_ROOT"
```

Select the frozen visual representation on historical pseudo-held-out
trajectories, then compile the full feature cache:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" trace_method/build_visual_index.py \
  --model-path "$QWEN_MODEL" --device cuda:0 --scope calibration \
  --history-root "$HISTORY_ROOT" --batch-size 64 --pixels 200704
"$PYTHON_BIN" trace_method/calibrate_visual.py \
  --history-root "$HISTORY_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" trace_method/build_visual_index.py \
  --model-path "$QWEN_MODEL" --device cuda:0 --scope full \
  --history-root "$HISTORY_ROOT" --batch-size 64 --pixels 200704
```

Evaluate the controlled RGB ablation once, followed by final TRACE and its
paired analysis:

```bash
"$PYTHON_BIN" trace_method/evaluate_rgb_ablation.py \
  --history-root "$HISTORY_ROOT"
"$PYTHON_BIN" trace_method/evaluate_execution.py \
  --history-root "$HISTORY_ROOT"
"$PYTHON_BIN" trace_method/analyze_execution.py
```

The retained paired baseline is `results/execution_baseline_gpu0.json`.  Its
predictions come from the upstream evaluator
`../src/eval_execution/evaluation_qwen3vl.py`; HIM-Agent itself can be rerun
with `../scripts/HIM_agent/eval_execution.sh`.

### Complete additional backbones

The continuous-batching evaluator uses the same ambiguous instructions,
teacher-forced history, image budget, action parser, and 0–999 coordinate
protocol for all three additional frozen models.  It refuses to run unless
physical GPU 0 is the only visible CUDA device.  Set the model variables to
local snapshot directories:

```bash
VLLM_PYTHON=python
MAI_MODEL=/path/to/MAI-UI-8B/snapshot
QWEN35_MODEL=/path/to/Qwen3.5-9B/snapshot
OWL_MODEL=/path/to/GUI-Owl-1.5-8B-Instruct/snapshot

CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM_PYTHON" \
  src/eval_execution/evaluation_new_backbones_vllm.py \
  --model-name MAI-UI-8B --model-path "$MAI_MODEL" --prompt-style qwen \
  --history-root "$HISTORY_ROOT" --device cuda:0 --request-batch-size 256 \
  --max-num-seqs 64 --max-new-tokens 256 \
  --save-file trace_method/results/backbones/mai_ui_8b_gpu0.json

CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM_PYTHON" \
  src/eval_execution/evaluation_new_backbones_vllm.py \
  --model-name Qwen3.5-9B --model-path "$QWEN35_MODEL" --prompt-style qwen \
  --history-root "$HISTORY_ROOT" --device cuda:0 --request-batch-size 256 \
  --max-num-seqs 64 --max-new-tokens 256 \
  --save-file trace_method/results/backbones/qwen35_9b_gpu0.json

CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM_PYTHON" \
  src/eval_execution/evaluation_new_backbones_vllm.py \
  --model-name GUI-Owl-1.5-8B-Instruct --model-path "$OWL_MODEL" \
  --prompt-style owl --history-root "$HISTORY_ROOT" --device cuda:0 \
  --request-batch-size 256 --max-num-seqs 64 --max-new-tokens 256 \
  --save-file trace_method/results/backbones/gui_owl_1_5_8b_gpu0.json

"$PYTHON_BIN" trace_method/analyze_backbones.py
```

Use `--resume` to continue any matching checkpoint.  The Transformers-only
fallback is `../src/eval_execution/evaluation_new_backbones.py`.

## Proactive suggestion

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" trace_method/build_proactive_index.py \
  --embedding-model "$TEXT_MODEL" --device cuda:0
"$PYTHON_BIN" trace_method/evaluate_proactive.py

# Direct frequency-normalization ablation used by analyze_proactive.py.
"$PYTHON_BIN" trace_method/evaluate_proactive.py \
  --selection-power 0 \
  --output trace_method/results/proactive_rawmass_predictions.json \
  --positive-csv trace_method/results/proactive_rawmass_positive.csv \
  --negative-csv trace_method/results/proactive_rawmass_negative.csv

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" trace_method/analyze_proactive.py \
  --embed-model "$EMBED_MODEL" --device cuda:0
```

HIM-Agent's proactive baseline remains available at
`../scripts/HIM_agent/eval_proactive.sh`.

## Generated outputs

- `results/execution_predictions_gpu0.json`: final 7,915-step predictions.
- `results/execution_analysis.json`: metrics, intervals, ablations, and audits.
- `results/execution_rgb_ablation_gpu0.json`: controlled execution ablations.
- `results/execution_visual_calibration.json`: history-only Qwen layer choice.
- `results/backbones/*.json`: complete additional-backbone predictions.
- `results/backbone_analysis.json`: paired backbone and TRACE controls.
- `results/proactive_predictions.json`: all 315 proactive predictions.
- `results/proactive_analysis.json`: locally reproducible proactive metrics.

These generated files stay local and are ignored by Git; only the lightweight
method report is versioned.

Only the final method, its controlled ablations, and reproducibility code are
retained here; exploratory predecessors are not dependencies of TRACE.
