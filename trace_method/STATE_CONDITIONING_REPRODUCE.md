# Reproducing the state-conditioning study

Run commands from the Personal-GUI repository root. This study reuses the current
TRACE history split, semantic ranking, calibrated visual descriptors, and native
grounded policy evaluator. It does not train a model or change production TRACE.

## Prerequisites

- The released execution test (`data/personal_execution/personal_execution.json`)
  and episode descriptions (`personal_samples.json`): 775 episodes, 7,915 steps.
- Fingertip screenshots and action logs at
  `/storage/zhangfuyao/datasets/fingertip20k`.
- Existing TRACE artifacts under `trace_method/results/grounded`: the semantic
  index, app labels, four calibrated visual caches, and four grounded contexts.
  Their preparation belongs to the existing TRACE pipeline; these scripts do
  not rebuild the underlying model features from a clean checkout.
- CPU analysis environment: `/storage/zhangfuyao/miniforge3/envs/hymem/bin/python`
  (NumPy, Pillow, pandas, PyTorch, matplotlib).
- GPU inference environment: `/storage/zhangfuyao/miniforge3/envs/hymem-serve/bin/python`
  (vLLM 0.28.0, Transformers 5.16.1, PyTorch 2.13.0+cu130;
  exact installed versions are in each prediction's `protocol`).
- Local frozen model snapshots: Qwen3-VL-8B-Instruct revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and, for the second task annotator,
  Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

The evaluator uses physical GPU 0 exclusively. Do not run the annotators and
execution evaluator concurrently on that GPU. Paths below reflect the study
machine; adapt model and dataset paths explicitly on another machine.

## Build contexts and annotate task relevance

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_TF=0
ANALYSIS_PY=/storage/zhangfuyao/miniforge3/envs/hymem/bin/python
INFERENCE_PY=/storage/zhangfuyao/miniforge3/envs/hymem-serve/bin/python
QWEN3_PATH=/storage/zhangfuyao/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
QWEN35_PATH=/storage/zhangfuyao/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
STUDY_DIR=trace_method/results/state_conditioning/qwen3_vl_8b

for model_key in qwen3_vl_8b qwen35_9b mai_ui_8b gui_owl_1_5_8b; do
  "$ANALYSIS_PY" trace_method/build_state_conditioning_experiment.py --model-key "$model_key"
done
"$ANALYSIS_PY" trace_method/build_state_conditioning_full_context.py

export CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
"$INFERENCE_PY" trace_method/annotate_state_conditioning_tasks.py \
  --input "$STUDY_DIR/task_relevance_inputs.json" \
  --output "$STUDY_DIR/task_relevance_qwen3.json" --model-path "$QWEN3_PATH"
"$INFERENCE_PY" trace_method/annotate_state_conditioning_tasks.py \
  --input "$STUDY_DIR/task_relevance_inputs.json" \
  --output "$STUDY_DIR/task_relevance_qwen35.json" --model-path "$QWEN35_PATH"
"$ANALYSIS_PY" trace_method/seal_state_conditioning_sources.py
```

For a new replication, use a separate checkout/output directory so existing
predictions, annotations, and seals remain intact. The source sealer preserves
and verifies an existing seal; it does not replace one after predictions exist.
The current study's primary protocol text is preserved verbatim in
`results/state_conditioning/protocol_at_inference_start.md`, checked against the
hash recorded before execution inference. Later exploratory checks are listed
separately in `STATE_CONDITIONING_PROTOCOL.md`.

## Run the five policy conditions

```bash
RUNNER="$INFERENCE_PY" MODEL_PATH="$QWEN3_PATH" \
  VARIANTS='top1_ordinal top1_state top8_previous top8_state top1_trajectory' \
  bash scripts/execution/eval_state_conditioning.sh
```

Each condition runs all 7,915 ordered steps using the same frozen model,
200,704-pixel image budget, 256-token output limit, 8,192-token context limit,
256-request batches, at most 64 sequences, and greedy decoding with seed 0.
The full-record prompt audit found at most 2,623 text tokens; no source record
was truncated to fit the context limit. Raw outputs are retained, including
parsing failures if any. `--resume` checks the completed prefix and settings.

The current execution schedule ran the first four conditions sequentially, then
the full-record condition. Other backbone runs in this study score historical
evidence only; they are not new complete policy ablations.

## Analyze and export

```bash
"$ANALYSIS_PY" trace_method/analyze_state_conditioning_experiment.py --require-policy
"$ANALYSIS_PY" trace_method/export_state_conditioning_paper.py
"$ANALYSIS_PY" trace_method/make_state_conditioning_cases.py
"$ANALYSIS_PY" -m unittest discover -s tests -p test_state_conditioning_analysis.py -v
```

The analyzer verifies source identity, episode isolation, unchanged inference
sources, matching test/context hashes, prediction ordering, entry-context
equality, complete primary runs, and shared inference settings. Its main paired
bootstrap resamples 82 users as whole clusters (20,000 draws), with episode
clustering as sensitivity analysis. Entry steps are excluded from the primary
estimand. Confidence intervals describe resampling of benchmark data; they do
not separately estimate variation across inference runs.

All results are written under `trace_method/results/state_conditioning`.
`RESULTS_zh.md` is the readable report, `analysis.json` and `paired_results.csv`
contain the full statistics, and the `.tex`/`.pdf`/`.png` files provide tables,
prose, and figures. Case figures use outcome-selected examples whose current
screens and reference screens were visually checked by the assistant. They are
not independent human annotations or a random sample of the benchmark.

The study uses teacher-forced prefixes and the original benchmark scorer. It
does not establish closed-loop success, preference recovery, or a benefit that
increases with action position. Inspect the strict task/app subset, the equal
trajectory-length subset, and the evidence-to-policy gap before choosing the
paper's claim scope.
