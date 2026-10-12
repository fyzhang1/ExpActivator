# ExpActivator: training-free long-horizon personalization

This directory contains ExpActivator, its controlled ablations, and the commands
needed to reproduce the results. Grounded-policy execution results are
currently being regenerated; older reports that call the retrieved action the
final prediction describe a superseded direct-transfer experiment. The upstream
HIM-Agent baseline remains unchanged in `../HIM_agent/`, `../scripts/HIM_agent/`,
and `../skills/hierarchical-implicit-intent-memory/`.

ExpActivator combines:

1. omission-aware residual-goal voting for a personalized first-app candidate;
2. goal-restricted same-user trajectory memory; and
3. current-screen synchronization with a history-selected visual state from
   the frozen GUI backbone under evaluation; and
4. a frozen GUI policy that receives the retrieved action as grounded evidence
   and may adopt, adapt, or reject it.

It fits no model or prediction parameter. The execution protocol covers 775
episodes and 7,915 actions; proactive evaluation covers 215 positive and 100
negative states.

## Layout

```text
expactivator/
├── execution_core.py              shared trajectory/action utilities
├── evaluation_stats.py            paired bootstrap and McNemar statistics
├── compile_history_app_labels.py  frozen training-side app compilation
├── build_execution_index.py       omission-aware history index and app vote
├── calibrate_execution.py         history-only k calibration
├── build_visual_index.py          frozen backbone visual feature compiler
├── calibrate_visual.py            history-only visual representation selection
├── build_execution_grounded_evidence.py  per-step retrieved evidence builder
├── analyze_grounded_execution.py  grounded-policy metrics and integrity audit
├── evaluate_rgb_ablation.py       RGB and controlled execution ablations
├── evaluate_execution.py          selected-state execution evaluation
├── analyze_execution.py           complete execution analysis and audits
├── analyze_backbones.py           complete-backbone and ExpActivator controls
├── build_proactive_index.py       official chronological proactive memory
├── evaluate_proactive.py          training-free proactive prediction
├── analyze_proactive.py           official metrics and uncertainty
├── analyze_proactive_paper_baselines.py  full model × memory matrix
├── shared_memory.py               shared record schema and task adapters
├── build_shared_memory.py         byte-exact task-view migration
├── verify_shared_memory.py        end-to-end architectural equivalence audit
```

## Environment

The commands below assume the repository root is the working directory.

```bash
PYTHON_BIN=python
VLLM_PYTHON=python
QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct
TEXT_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
HISTORY_ROOT=/path/to/fingertip20k
VISUAL_CACHE=/path/to/the/sealed/full-visual-cache
```

Historical screenshots are not committed.  Set `HISTORY_ROOT` to the extracted
Fingertip history directory and pass `--history-root "$HISTORY_ROOT"` to the
commands that need screenshots.

## History memory

Every execution-branch script builds its memory through
`execution_core.execution_memory_frame`, which is the official chronological
first-80%-per-user split of `samples/total.csv` (15,966 rows).  This is
byte-identical to the history the four paper baselines read, and it contains
none of the 775 execution-test episodes.

`samples/train_set.csv` must not be used as memory.  It is not that
chronological split: it retains 490 of the test episodes, 489 of them with the
explicit original instruction that the ambiguous query paraphrases, so
indexing it leaks first-action answers and hands the method 2,566 records the
baselines never see.  `audit_memory_split` recomputes the intersection from
the data and raises on any non-empty overlap; `tests/test_history_split.py`
asserts both that the split matches the baselines and that the audit rejects
`train_set.csv`.

## Personalized execution

Compile the factual history and select both neighborhood sizes without opening
execution-test answers.  Label compilation joins on the `(user_id, time)`
episode identity, so `--reuse` carries over every already-compiled label and
only the records a previous split did not cover reach the GPU:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" expactivator/compile_history_app_labels.py \
  --model-path "$QWEN_MODEL" \
  --reuse expactivator/results/execution_app_labels.json
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" expactivator/build_execution_index.py \
  --embedding-model "$TEXT_MODEL" --device cuda:0
"$PYTHON_BIN" expactivator/calibrate_execution.py \
  --history-root "$HISTORY_ROOT"
```

Select the frozen visual representation on historical pseudo-held-out
trajectories, then compile the full feature cache:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" expactivator/build_visual_index.py \
  --model-path "$QWEN_MODEL" --device cuda:0 --scope calibration \
  --history-root "$HISTORY_ROOT" --batch-size 64 --pixels 200704
"$PYTHON_BIN" expactivator/calibrate_visual.py \
  --history-root "$HISTORY_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" expactivator/build_visual_index.py \
  --model-path "$QWEN_MODEL" --device cuda:0 --scope full \
  --history-root "$HISTORY_ROOT" --batch-size 64 --pixels 200704
```

The legacy RGB and selected-state evaluators below measure retrieval as a
direct-action ablation. They are not the final grounded-policy method:

```bash
"$PYTHON_BIN" expactivator/evaluate_rgb_ablation.py \
  --history-root "$HISTORY_ROOT"
"$PYTHON_BIN" expactivator/evaluate_execution.py \
  --history-root "$HISTORY_ROOT"
"$PYTHON_BIN" expactivator/analyze_execution.py
```

For the final method, build a complete step-level evidence file and pass it to
the frozen policy evaluator with `--baseline-name expactivator_grounded`. The evidence
builder never emits a final action and records the source of every candidate.
The all-backbone runner performs history-only calibration, cache construction,
evidence construction, and policy inference:

```bash
QWEN3_VL_MODEL=/path/to/Qwen3-VL-8B/snapshot \
MAI_MODEL=/path/to/MAI-UI-8B/snapshot \
QWEN35_MODEL=/path/to/Qwen3.5-9B/snapshot \
OWL_MODEL=/path/to/GUI-Owl-1.5-8B/snapshot \
TEXT_MODEL=/path/to/paraphrase-multilingual-MiniLM-L12-v2 \
HISTORY_ROOT="$HISTORY_ROOT" MODEL_PYTHON="$PYTHON_BIN" \
VLLM_PYTHON="$VLLM_PYTHON" scripts/execution/eval_expactivator_grounded.sh
```

The legacy Qwen3-VL baseline is retained at
`results/execution_baseline_gpu0.json`; its predictions come from the upstream
evaluator `../src/eval_execution/evaluation_qwen3vl.py`. The controlled vLLM
base used for current comparisons is
`results/paper_baselines/qwen3_vl_8b_base_gpu0.json`; HIM-Agent itself can be rerun
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
  --save-file expactivator/results/backbones/mai_ui_8b_gpu0.json

CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM_PYTHON" \
  src/eval_execution/evaluation_new_backbones_vllm.py \
  --model-name Qwen3.5-9B --model-path "$QWEN35_MODEL" --prompt-style qwen \
  --history-root "$HISTORY_ROOT" --device cuda:0 --request-batch-size 256 \
  --max-num-seqs 64 --max-new-tokens 256 \
  --save-file expactivator/results/backbones/qwen35_9b_gpu0.json

CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM_PYTHON" \
  src/eval_execution/evaluation_new_backbones_vllm.py \
  --model-name GUI-Owl-1.5-8B-Instruct --model-path "$OWL_MODEL" \
  --prompt-style owl --history-root "$HISTORY_ROOT" --device cuda:0 \
  --request-batch-size 256 --max-num-seqs 64 --max-new-tokens 256 \
  --save-file expactivator/results/backbones/gui_owl_1_5_8b_gpu0.json
```

Use `--resume` to continue any matching checkpoint.  The Transformers-only
fallback is `../src/eval_execution/evaluation_new_backbones.py`.

### Reproduce the paper memory baselines

The four memory baselines use the same 15,966-row, first-80%-per-user
chronological history split and seal one context for each of the 775 test
episodes before execution inference. `Recent` supplies the latest three full
intent/app/action records. `Retrieve` supplies the three history records whose
intents have the highest Qwen3-Embedding-0.6B cosine similarity to the
ambiguous query. `HIM-Agent` runs the released history filter unchanged, keeps
clusters with at least three members, and retrieves the top three cluster
representatives. None of these builders reads test answers.

The paper reports GPT-5.1 (`gpt-5.1-2025-11-13`) as the LLM-UM summarizer, but
neither those generated profiles nor a runnable LLM-UM builder are included in
the release. This offline reproduction therefore uses the fixed local
Qwen3.5-9B snapshot with greedy decoding. The generated file retains the raw
output, parsed summaries, token counts, model revision, and the paper-model
identifier so this substitution remains explicit.

```bash
EMBED_MODEL=/path/to/Qwen3-Embedding-0.6B

CUDA_VISIBLE_DEVICES=0 USE_TF=0 "$PYTHON_BIN" \
  HIM_agent/HIM_execution_filter.py \
  --embed-model "$EMBED_MODEL" --data-path "$HISTORY_ROOT" \
  --save-base expactivator/results/paper_baselines/him_preference_memory

CUDA_VISIBLE_DEVICES=0 USE_TF=0 "$VLLM_PYTHON" \
  src/eval_execution/build_llm_um_profiles_vllm.py \
  --model-name Qwen3.5-9B --model-path "$QWEN35_MODEL" \
  --history-root "$HISTORY_ROOT" \
  --max-summaries 10 --max-actions-per-summary 3 \
  --output expactivator/results/paper_baselines/llm_um_profiles.json

# Retry only any profiles whose first response was incomplete at the output cap.
CUDA_VISIBLE_DEVICES=0 USE_TF=0 "$VLLM_PYTHON" \
  src/eval_execution/build_llm_um_profiles_vllm.py \
  --model-name Qwen3.5-9B --model-path "$QWEN35_MODEL" \
  --history-root "$HISTORY_ROOT" --resume-existing \
  --max-summaries 6 --max-actions-per-summary 2 \
  --output expactivator/results/paper_baselines/llm_um_profiles.json

for METHOD in recent retrieve llm_um him_agent; do
  CUDA_VISIBLE_DEVICES=0 USE_TF=0 "$PYTHON_BIN" \
    src/eval_execution/build_paper_baseline_contexts.py \
    --method "$METHOD" --embedding-model "$EMBED_MODEL" \
    --history-root "$HISTORY_ROOT" \
    --him-dir expactivator/results/paper_baselines/him_preference_memory \
    --output "expactivator/results/paper_baselines/${METHOD}_contexts.json"
done

MAI_MODEL=/path/to/MAI-UI-8B/snapshot \
QWEN3_VL_MODEL=/path/to/Qwen3-VL-8B-Instruct/snapshot \
QWEN35_MODEL=/path/to/Qwen3.5-9B/snapshot \
OWL_MODEL=/path/to/GUI-Owl-1.5-8B-Instruct/snapshot \
HISTORY_ROOT="$HISTORY_ROOT" VLLM_PYTHON="$VLLM_PYTHON" \
  scripts/execution/eval_paper_baselines.sh

"$PYTHON_BIN" expactivator/analyze_paper_baselines.py
```

See [`PAPER_BASELINE_RESULTS.md`](PAPER_BASELINE_RESULTS.md) for the memory
baseline result table, paired uncertainty, integrity audit, and the LLM-UM
reconstruction caveat. Its older ExpActivator execution row is superseded and must
not be used until the grounded-policy rerun is complete.

The matrix runner fixes the image budget at 200,704 pixels and the model
context ceiling at 8,192 tokens. The latter was selected after auditing all
16 memory-baseline prompt sets; 4,096 would reject long histories, while every audited prompt
plus its image and 256-token response allowance fits within 8,192. Runs are
checkpointed, resumable, and rejected if the baseline label, context hash,
test hash, model revision, or GPU assignment changes.

The raw runs above remain controls. Final ExpActivator runs must use the grounded
runner shown in the personalized-execution section, not
`evaluate_execution.py`, whose direct-emission behavior is retained only as an
ablation. Each backbone receives its own history-selected visual evidence and
then makes the final decision with its frozen policy model.

Existing chunks are resumed only when their manifest matches the requested
model, revision, scope, image list, pixel budget, and representation readout.
Qwen3-VL-family backbones use their native merged deep-stack states; Qwen3.5
uses hooked native unmerged visual-token states because it has no deep-stack
mergers.  Calibration and full-cache manifests must match before evaluation.

## Proactive suggestion

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" expactivator/build_proactive_index.py \
  --embedding-model "$TEXT_MODEL" --device cuda:0
"$PYTHON_BIN" expactivator/evaluate_proactive.py

# Direct frequency-normalization ablation used by analyze_proactive.py.
"$PYTHON_BIN" expactivator/evaluate_proactive.py \
  --selection-power 0 \
  --output expactivator/results/proactive_rawmass_predictions.json \
  --positive-csv expactivator/results/proactive_rawmass_positive.csv \
  --negative-csv expactivator/results/proactive_rawmass_negative.csv

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" expactivator/analyze_proactive.py \
  --embed-model "$EMBED_MODEL" --device cuda:0
```

HIM-Agent's proactive baseline remains available at
`../scripts/HIM_agent/eval_proactive.sh`.

The proactive matrix evaluates Recent, Retrieve, LLM-UM, HIM-Agent, and ExpActivator
with Qwen3-VL-8B, MAI-UI-8B, Qwen3.5-9B, and GUI-Owl-1.5-8B on all 215
positive and 100 negative queries.  The runner rebuilds the audited contexts,
runs all 20 backbone-method combinations on physical GPU 0, recomputes native
model-independent ExpActivator-direct, and then regenerates the combined report:

```bash
bash scripts/proactive/eval_paper_baseline_matrix.sh
```

The versioned result table and the exact reconstruction boundary are in
[`PROACTIVE_BASELINE_RESULTS.md`](PROACTIVE_BASELINE_RESULTS.md).  In
particular, proactive Retrieve and LLM-UM were not released, so their local
reconstructions are labeled rather than presented as exact reproductions.
Each backbone ExpActivator row decodes the target-free state selector's routine
context.  ExpActivator-direct retains the native protocol that emits the synchronized
historical medoid without a backbone decoder; both protocols remain explicit.

## Shared multi-view record equivalence

The execution and proactive branches can share one identity-stable record
store without sharing one latent.  `build_shared_memory.py` joins records by
the unique `(user_id, time)` key, preserves the task-specific history masks,
and migrates the sealed latent views byte-for-byte.  Both branches now read
the same chronological first-80% split, so the two masks cover the same
15,966 records while keeping separate latents:

- execution reads residual goal, app preference, trajectory, and visual-cache
  references;
- proactive suggestion reads full intent, time, and scenario.

This is a schema-migration test, not a new scoring method.  Re-encoding is
deliberately avoided so GPU numerical variation cannot be mistaken for an
architectural effect.

```bash
"$PYTHON_BIN" expactivator/build_shared_memory.py \
  --history-root "$HISTORY_ROOT"
"$PYTHON_BIN" expactivator/evaluate_proactive.py \
  --shared-memory expactivator/results/shared_multiview_memory_gpu0.pt \
  --output expactivator/results/shared_proactive_predictions.json \
  --positive-csv expactivator/results/shared_proactive_positive.csv \
  --negative-csv expactivator/results/shared_proactive_negative.csv
"$PYTHON_BIN" expactivator/evaluate_execution.py \
  --shared-memory expactivator/results/shared_multiview_memory_gpu0.pt \
  --index expactivator/results/shared_execution_app_predictions_gpu0.json \
  --history-root "$HISTORY_ROOT" --visual-dir "$VISUAL_CACHE" \
  --output expactivator/results/shared_execution_predictions_gpu0.json
"$PYTHON_BIN" expactivator/verify_shared_memory.py
```

The audit checks that all 7,915 execution outputs and all 315 proactive
predictions are exactly equal to their non-shared counterparts.  The store has
20,000 unique records, and both task masks select the same 15,966 history
records.  Re-run it after regenerating the execution artifacts; the counts it
prints supersede any figure quoted here.

## Generated outputs

- `results/execution_predictions_gpu0.json`: final 7,915-step predictions.
- `results/execution_analysis.json`: metrics, intervals, ablations, and audits.
- `results/execution_rgb_ablation_gpu0.json`: controlled execution ablations.
- `results/execution_visual_calibration.json`: history-only Qwen layer choice.
- `results/backbones/*_gpu0.json`: complete raw-backbone predictions.
- `results/backbones/*_expactivator_calibration.json`: history-only representation
  selection for each frozen backbone.
- `results/backbones/*_expactivator_gpu0.json`: complete backbone-specific ExpActivator
  predictions.
- `results/backbone_analysis.json`: paired backbone and ExpActivator controls.
- `results/proactive_predictions.json`: all 315 proactive predictions.
- `results/proactive_analysis.json`: locally reproducible proactive metrics.
- `results/proactive_paper_baselines/`: complete raw model × memory runs,
  context/source audit, hashes, bootstrap intervals, and scored analysis.
- `results/shared_multiview_memory_gpu0.pt`: shared records, task masks, and
  separately addressable sealed views.
- `results/shared_memory_equivalence.json`: exact end-to-end equivalence audit.

These generated files stay local and are ignored by Git; only the lightweight
method report is versioned.

Only the final method, its controlled ablations, and reproducibility code are
retained here; exploratory predecessors are not dependencies of ExpActivator.
