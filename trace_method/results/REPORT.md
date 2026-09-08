# TRACE: training-free long-horizon personalization for GUI agents

## Bottom line

TRACE (**T**raining-free **R**ecord **A**lignment for **C**onsistent
**E**xecution) now covers both parts of the unchanged AndroidIntent benchmark:
all 775 personalized-execution episodes (7,915 actions), all 215 proactive
positive states, and all 100 proactive negative states.  It keeps every
encoder and GUI model frozen, fits no prediction parameters, and uses neither
gradient descent, policy training, reinforcement learning, counterfactual
generation, learned routing, nor learned gating.

| Benchmark part | Released evaluation scope | TRACE coverage |
|---|---:|---:|
| Personalized execution | 775 episodes / 7,915 actions | 775 / 7,915 |
| Proactive positives | 215 states | 215 |
| Proactive negatives | 100 states | 100 |

The released local scorer computes every execution metric and five of the six
proactive metrics.  The remaining proactive LLM-Judgment number requires the
paper's external DeepSeek-V3 judge, but the release provides neither a judge
implementation nor a configured endpoint.  It is therefore marked unrun
rather than replaced with a non-comparable number.

### Personalized execution

| Method | Type | SSR | CSR | First-step SR | Non-first SR |
|---|---:|---:|---:|---:|---:|
| Paired Qwen3-VL baseline | 52.748 | 20.354 | 24.632 | 26.452 | 19.692 |
| Omission-aware app-only ablation | 54.769 | 25.509 | 49.516 | 79.097 | 19.692 |
| RGB-TRACE ablation | 56.235 | 37.701 | 52.647 | 79.097 | 33.207 |
| **TRACE (frozen Qwen visual state)** | **58.825** | **39.116** | **53.615** | **79.097** | **34.776** |
| TRACE − app-only ablation | **+4.056** | **+13.607** | **+4.099** | 0.000 | **+15.084** |

Paired episode bootstrap 95% confidence intervals for TRACE over its app-only
ablation are `[+11.691,+15.629]` percentage points for SSR,
`[+2.203,+6.025]` points for CSR, and `[+2.288,+5.870]` points for Type.

### Complete backbone-specific TRACE stress test

Three additional frozen local backbones were evaluated on the same 775
episodes and 7,915 teacher-forced actions with ambiguous instructions:
MAI-UI-8B, Qwen3.5-9B, and GUI-Owl-1.5-8B-Instruct.  TRACE was then integrated
with **each model separately**.  This is not only an app selector: each full
run uses that backbone's own frozen visual latent to synchronize every one of
the 7,140 non-first test screens to goal-restricted historical states.

`+ TRACE app` is the narrow ablation that replaces only the 775 episode-start
actions and leaves the raw model's later predictions unchanged.  `+ full
TRACE` includes both that app decision and backbone-specific visual state
synchronization.

| Backbone / method | Type | SSR | CSR | First-step SR | Non-first SR |
|---|---:|---:|---:|---:|---:|
| MAI-UI-8B raw | 51.762 | 17.890 | 22.260 | 22.323 | 17.409 |
| MAI-UI-8B + TRACE app only | 54.340 | 23.449 | 47.963 | 79.097 | 17.409 |
| **MAI-UI-8B + full TRACE (own latent)** | **58.939** | **39.330** | **53.812** | **79.097** | **35.014** |
| Qwen3.5-9B raw | 50.120 | 17.183 | 21.427 | 22.839 | 16.569 |
| Qwen3.5-9B + TRACE app only | 52.697 | 22.691 | 47.307 | 79.097 | 16.569 |
| **Qwen3.5-9B + full TRACE (own latent)** | **59.495** | **39.949** | **54.870** | **79.097** | **35.700** |
| GUI-Owl-1.5-8B raw | 58.181 | 22.742 | 21.515 | 26.839 | 22.297 |
| GUI-Owl-1.5-8B + TRACE app only | 58.800 | 27.858 | 45.982 | 79.097 | 22.297 |
| **GUI-Owl-1.5-8B + full TRACE (own latent)** | **60.430** | **40.076** | **54.875** | **79.097** | **35.840** |
| Fixed Qwen3-VL TRACE reference | 58.825 | 39.116 | 53.615 | 79.097 | 34.776 |

With first-step outputs identical, adding the later state synchronizer over
the app-only ablation raises SSR by `+15.881` points for MAI-UI (95% CI
`[+14.040,+17.831]`), `+17.258` for Qwen3.5
(`[+15.388,+19.194]`), and `+12.217` for GUI-Owl
(`[+10.273,+14.168]`).  CSR rises by `+5.849`, `+7.562`, and `+8.893`
points; non-first SR rises by `+17.605`, `+19.132`, and `+13.543` points.
Thus the improvement is not attributable only to choosing the correct app.

Relative to each raw backbone, the corresponding full-TRACE SSR gains are
`+21.440`, `+22.767`, and `+17.334` points.  The model-specific full systems
also slightly exceed the fixed Qwen3-VL TRACE reference: MAI-UI by `+0.215`,
Qwen3.5 by `+0.834`, and GUI-Owl by `+0.960` SSR points.  The MAI-UI and
Qwen3.5 SSR intervals against that reference cross zero; GUI-Owl's interval is
`[+0.091,+1.824]`.  Qwen3.5 and GUI-Owl themselves are indistinguishable on
SSR (`+0.126`, 95% CI `[-0.843,+1.097]`) and CSR (`+0.005`,
`[-0.891,+0.925]`).

Representation choice was sealed before test scoring.  The same 756
non-first actions from final-readable historical trajectories were held out,
with only earlier same-user histories available as memory.  Against RGB-MSE,
history-only calibration selected MAI-UI `deep8_raw` at 33.995%
(`+2.778`, exact McNemar `p=0.0987`), Qwen3.5 `spatial_centered` at 32.540%
(`+1.323`, `p=0.4534`), and GUI-Owl `pyramid_raw` at 34.656%
(`+3.439`, `p=0.0450`).  Qwen3-VL-family checkpoints expose native merged
deep-stack features; Qwen3.5 instead uses its native unmerged visual-token
states.  No model weights or prediction parameters are trained.

Each full cache contains the same 34,319 image paths in 68 chunks.  Manifests
bind calibration and evaluation to model path, revision, architecture,
readout, pixel budget, and scope.  All feature extraction and raw-model runs
used physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`; GPU 1 was never exposed.
All full TRACE files cover 7,915 ordered unique step IDs, match the official
test SHA-256, record zero exact-episode-exclusion violations, and declare no
test-answer access during prediction.

The three raw runs share one vLLM 0.28.0 / Transformers 5.16.1 inference
stack, deterministic BF16 decoding, a 200,704-pixel image budget, and a
256-token output cap.  Unrecoverable raw action parses were 66/7,915 (0.834%)
for MAI-UI, 16/7,915 (0.202%) for Qwen3.5, and 1/7,915 (0.013%) for GUI-Owl;
every raw response is retained for audit.  The older paired Qwen3-VL file came
from the upstream Transformers evaluator, so raw-backbone-versus-Qwen3-VL
deltas can include an inference-backend effect and are descriptive.

### Proactive suggestion

The proactive results use the official chronological first-80%-per-user
memory split, not `train_set.csv`.  The semantic score is reproduced with the
official `Qwen3-Embedding-0.6B` cosine-plus-edit scorer.  HIM-Agent numbers are
the published Table 6 reference, not a locally controlled rerun.

| Method | Semantic | Precision | Recall | FA (lower) | F1 |
|---|---:|---:|---:|---:|---:|
| Published HIM-Agent | 53.50 | 78.10 | 81.40 | 49.00 | 79.70 |
| **TRACE** | **55.35** | **82.71** | **82.33** | **37.00** | **82.52** |
| TRACE − published HIM | +1.85 | +4.61 | +0.93 | **−12.00** | +2.82 |

TRACE predicts a suggestion for 177/215 positive states and 37/100 negative
states (TP=177, FP=37, FN=38, TN=63).  Stratified bootstrap 95% intervals are
`[51.21,59.58]` for Semantic, `[78.97,86.60]` for Precision,
`[76.74,87.44]` for Recall, `[28.00,46.00]` for FA, and `[78.76,85.98]` for
F1.  These intervals quantify TRACE uncertainty; they do not turn the
published-number comparison into a paired significance test.

## Research problems and method

### Supporting component: structured action decoding

TRACE avoids unconstrained text generation at the personalized app decision.
It selects a canonical app name drawn solely from factual training histories
and emits a valid `open_app` action.  This addresses action-name and format
instability, but by itself covers only the first action of each trajectory.

### Innovation 1: omission-aware personal-goal alignment

**Problem.** A factual history contains the preference that the test query
deliberately omits.  For example, the history says “open NetEase Cloud Music
and play my favorites”, whereas the query says only “play my favorites”.  If
the full history sentence is used as a retrieval key, the answer-bearing app
token is present on only one side of the comparison.  This mismatch becomes
more harmful as a long personal history accumulates records from many apps.

**Solution.** TRACE deterministically separates each historical request into
the residual GUI goal and its recorded action label.  A frozen multilingual
encoder represents only residual goals.  For query `q`, same-user histories
are ranked by cosine similarity and their canonical training-side actions are
aggregated by a fixed reciprocal-rank vote:

```text
score(a | q,u) = sum_{i in top-k(u), action_i=a} max(cos(E(q),E(goal_i)),0) / rank_i
prediction      = argmax_a score(a | q,u)
```

There is no fitted classifier.  The app neighborhood size `k=10` was selected
on a chronological pseudo-held-out subset of training histories only (3,231
records); the execution test was not accessed during calibration.

**Measured contribution.** Compared with matching the query against complete
historical instructions, goal factorization improves first-step SR from
72.903% to 79.097% (`+6.194` points; 76 paired wins, 28 losses,
`p=2.74e-6`).  It raises full SSR by `+0.606` points and CSR by `+2.616`
points while leaving every later prediction fixed.

### Innovation 2: goal-constrained visual trajectory synchronization

**Problem.** Long GUI trajectories are not aligned by step number.  The fifth
action in a historical task need not represent the same progress state as the
fifth action in the current task: loading delays, pop-ups, alternative paths,
and different content can insert or remove steps.  HIM-Agent compares action
trajectories while constructing memory, but it does not synchronize the
current screenshot to a particular historical pre-action state at execution
time.

**Solution.** TRACE first restricts action memory to the five most relevant
same-user residual-goal trajectories.  It then compares the current screenshot
with every historical pre-action screenshot in this small pool.  Its final
state representation is the normalized mean of the frozen Qwen3-VL visual
tower's layer-8 patch tokens.  Cosine similarity identifies the historical
state synchronized with the current screen.  The attached structured action
is transferred; click coordinates are normalized to the current screen
resolution.  This supports click, long-click, scroll, wait, type, back/home,
termination, and app actions without training a policy.

The trajectory neighborhood size `k=5` was selected using only the final
readable historical episode of each user as pseudo-held-out data: 839 actions
from 83 users, with only earlier histories available as memory.

The visual representation was also selected without opening the execution
test.  On the same historical pseudo-held-out protocol, excluding the first
open-app state, 16 frozen Qwen variants (eight layer/spatial summaries, with
and without history centering) were compared with RGB-MSE on 756 actions.  The
fixed layer-8 global representation was best: 34.127% versus 31.217% for RGB
(`+2.910` points; 82 wins, 60 losses, `p=0.0777`).  This calibration advantage
is directional rather than statistically conclusive, so exactly one selected
Qwen representation was carried into the complete formal evaluation.

**Measured contribution.** With the app prediction held fixed, adding visual
trajectory synchronization improves full SSR from 25.509% to 39.116%
(`+13.607` points) and CSR from 49.516% to 53.615% (`+4.099` points).  On the
7,140 non-first actions, success rises from 19.692% to 34.776% (`+15.084`
points; 1,827 wins, 750 losses, `p=9.19e-103`).

A direct “same task, same step number” transfer obtains only 19.916% on these
actions.  Visual synchronization reaches 34.776% (`+14.860` points; 1,751
wins, 690 losses, `p=1.62e-105`), isolating the benefit of resolving trajectory
phase mismatch rather than merely replaying a similar task.

Replacing RGB-MSE alone with the frozen Qwen representation raises Type from
56.235% to 58.825% (`+2.590` points; 95% CI `[+1.321,+3.848]`), SSR from
37.701% to 39.116% (`+1.415`; CI `[+0.289,+2.510]`), non-first SR from 33.207%
to 34.776% (`+1.569`; 772 wins, 660 losses, `p=0.00334`), and the CSR point
estimate from 52.647% to 53.615% (`+0.969`).  The CSR interval
`[-0.089,+2.053]` crosses zero, so that last incremental gain is not claimed
as statistically established.  Because both formal SSR and CSR point
estimates improved under the predeclared adoption rule, Qwen visual matching
replaces RGB-MSE in final TRACE.

For instruction-free proactive assistance, the same principle is applied to
the available user state rather than a screenshot.  TRACE first requires a
near-duplicate intent family to have at least 12 historical occurrences and
four occurrences in the current scenario.  A circular one-hour kernel binds
the current time and scenario to each stable family.  Crucially, selection
uses the conditional state likelihood

```text
p(state | routine) ∝ time-and-scenario kernel mass / routine support
```

rather than raw frequency.  This prevents a user's most common routine from
overriding a less frequent routine that better matches the current state.  The
output is a deterministic same-scenario medoid from factual history.  The
predictor receives only user ID, time, and scenario—never the target
instruction, app, or intent class.

With proactive trigger decisions held exactly fixed, removing frequency
normalization reduces official Semantic from 55.35% to 53.24% (`−2.11`
points).  The paired bootstrap interval for this delta is
`[−0.30,+4.65]`, so the direction is useful but not statistically conclusive
on 177 emitted positive suggestions.

## Action-level effects

| Gold non-first action | Count | Baseline SR | TRACE SR | Delta |
|---|---:|---:|---:|---:|
| click | 3,252 | 24.662 | 26.876 | +2.214 |
| scroll | 1,416 | 4.590 | 38.277 | +33.686 |
| wait | 1,075 | 20.465 | 41.209 | +20.744 |
| finished | 775 | 15.484 | 43.097 | +27.613 |
| type | 459 | 37.691 | 58.824 | +21.133 |
| press_back | 70 | 24.286 | 14.286 | -10.000 |
| press_home | 44 | 9.091 | 13.636 | +4.545 |
| long_click | 24 | 20.833 | 16.667 | -4.167 |

The method is not uniformly better.  Its gain is concentrated in repeated GUI
states such as scrolling, waiting, termination, text entry, and ordinary
clicks; `press_back` and the rare long-click class remain limitations.  Against
RGB-TRACE specifically, Qwen also reduces `wait` by 4.279 points while
improving the larger overall action set.

## Duplicate and non-overlap audit

The released training and execution files share `(user_id,time)` for 490 of
775 test episodes.  TRACE explicitly removes the identical episode from every
trajectory-memory candidate pool, including for those 490 episodes.  Thus no
downstream action can be copied from the same recorded trajectory.

On the 285 episodes absent from the training table entirely:

| Method | Type | SSR | CSR | First-step SR | Non-first SR |
|---|---:|---:|---:|---:|---:|
| Paired Qwen3-VL baseline | 52.986 | 20.677 | 24.894 | 27.719 | 19.908 |
| RGB-TRACE ablation | 56.369 | 38.523 | 48.579 | 74.035 | 34.648 |
| **TRACE (Qwen visual)** | **58.992** | **39.524** | **49.208** | **74.035** | **35.758** |
| TRACE − paired baseline | **+6.006** | **+18.847** | **+24.313** | **+46.316** | **+15.850** |

The non-overlap bootstrap interval is `[+15.194,+22.764]` points for SSR and
`[+19.011,+29.604]` for CSR.  This rules out exact episode reuse as the main
explanation for the improvement.

## Relation to HIM-Agent

HIM-Agent incrementally clusters records into preference/routine prototypes
using semantic similarity, lexical overlap, action-trajectory DTW, and fixed
thresholds.  It then supplies a prototype center to the GUI agent.  TRACE
targets two gaps not handled by that design:

1. it removes the answer-bearing preference before matching a vague query to
   explicit histories; and
2. it synchronizes the current visual state to an individual historical
   action opportunity instead of treating an entire trajectory prototype as
   execution context.

For proactive assistance, HIM-Agent supplies a prototype's center intent and
most frequent states to an LLM.  TRACE instead performs state-conditioned
competition between stable routine families and emits the selected historical
medoid directly.  This removes the online long-history prompt and avoids raw
frequency domination.

PersonalAlign Table 5 reports Qwen3-VL + HIM-Agent at 52.0 Type, 24.0 SSR, and
42.3 CSR.  TRACE is numerically higher at 58.825 / 39.116 / 53.615.  This is a
published-number comparison, not a controlled reproduction: the paper's
Qwen3-VL baseline (46.6 / 20.6 / 26.6) differs from the paired local baseline
(52.748 / 20.354 / 24.632).  The defensible primary claim is the paired gain
over the local baseline and the method's controlled ablations.

PersonalAlign Table 6 reports HIM-Agent at 53.5 Semantic, 78.1 Precision,
81.4 Recall, 49.0 FA, and 79.7 F1.  TRACE is numerically better on all five
locally reproducible proactive metrics.  This remains a published-number
comparison because the release does not include the authors' predictions or
configured GPT-5.1/DeepSeek-V3 endpoints.

## What “training-free” means here

- Frozen text encoder and frozen visual tower from the GUI backbone being
  evaluated; no weight updates.
- No classifier or policy parameters are fitted.
- No gradient descent, backpropagation, fine-tuning, RL, counterfactual
  generation, learned router, or learned gate.
- App and action candidates come only from factual training histories.
- `k=10` and `k=5` are fixed hyperparameters chosen on history-only temporal
  pseudo-validation, analogous to choosing HIM-Agent's fixed thresholds.

TRACE is **parameter-training-free, not data-free**: it requires historical
examples and an offline memory index.

For proactive inference, TRACE uses zero online LLM prompt tokens; it performs
only lookup, matrix similarity, a fixed state kernel, and medoid selection.
The published HIM-Agent row reports 1,605 tokens per proactive query.  This is
an architectural token saving, although frozen embedding compilation and the
12 MB per-benchmark index are not free.  Online latency and energy have not
yet been measured under a controlled serving setup.

## Scope and limitations

1. Evaluation remains PersonalAlign's offline, teacher-forced screenshot
   protocol.  An online rollout may drift into unseen screens after an error.
2. Frozen visual summaries are stronger than RGB-MSE here, but can still
   confuse visually similar states requiring different actions.  TRACE adds
   one visual-tower forward pass for each new screen; historical screen
   features can be compiled offline.
3. Exact trajectory duplicates are excluded, but the app component follows the
   official training protocol; the fully non-overlapping episode audit is
   therefore reported separately.
4. The current benchmark contains omitted app preferences most prominently.
   Contacts, payment methods, addresses, and settings require broader datasets.
5. `press_back` and long-click actions decline relative to the baseline, and
   `wait` declines relative to RGB-TRACE; these should be addressed before
   claiming universal per-action improvement.
6. Proactive thresholds are a fixed development configuration.  Because the
   public benchmark has no separate validation split and its labels were
   available during development, proactive results should be described as
   exploratory rather than sealed-test generalization until reproduced on a
   new user cohort.
7. The external DeepSeek-V3 LLM-Judgment metric is unavailable locally.  The
   official Semantic and identification metrics are complete; Judgment is not.
8. The Qwen layer/summary was selected on histories only, but the final
   RGB-versus-Qwen adoption decision used the released execution test point
   estimates under a fixed rule.  The incremental replacement gain should be
   confirmed on a new sealed cohort before being presented as final
   generalization evidence.

## Reproducibility artifacts

Execution code:

- `execution_core.py` and `evaluation_stats.py`: shared evaluation primitives.
- `compile_history_app_labels.py`: frozen training-side label compilation.
- `build_execution_index.py`: frozen omission-aware history compilation.
- `calibrate_execution.py`: history-only neighborhood calibration.
- `build_visual_index.py`: frozen Qwen3-VL-family and Qwen3.5 visual-state
  compilation with strict model/cache manifests.
- `calibrate_visual.py`: history-only per-backbone layer/summary selection.
- `evaluate_rgb_ablation.py`: RGB-MSE and controlled execution ablations.
- `evaluate_execution.py`: complete selected-state evaluation.
- `analyze_execution.py`: paired statistics, overlap, and leakage audits.
- `../src/eval_execution/evaluation_new_backbones.py`: batched Transformers
  evaluator and shared tolerant action parser for recent VLMs.
- `../src/eval_execution/evaluation_new_backbones_vllm.py`: GPU-0-pinned
  continuous-batching evaluator used for the complete stress test.
- `analyze_backbones.py`: paired raw, app-only, full model-specific TRACE, and
  cross-backbone controls.

Proactive code:

- `build_proactive_index.py`: official chronological 80% history index.
- `evaluate_proactive.py`: all 315 states and exact-state exclusion.
- `analyze_proactive.py`: official metrics and bootstrap intervals.

Generated results (ignored by Git in the code-only repository):

- `execution_baseline_gpu0.json`: one paired local baseline.
- `execution_app_predictions_gpu0.json`: app predictions and ranked evidence.
- `execution_history_calibration.json`: history-only neighborhood selection.
- `execution_rgb_ablation_gpu0.json`: RGB and controlled ablation outputs.
- `execution_visual_calibration.json`: test-free representation selection.
- `execution_predictions_gpu0.json`: adopted complete 7,915-step outputs.
- `execution_analysis.json`: all execution intervals and audits.
- `backbones/*_gpu0.json`: complete raw MAI-UI, Qwen3.5, and GUI-Owl outputs.
- `backbones/*_trace_calibration.json`: sealed history-only representation
  selection for each backbone.
- `backbones/*_trace_gpu0.json`: complete model-specific TRACE outputs.
- `backbone_analysis.json`: their paired comparisons and parser audit.
- `proactive_predictions.json` and `proactive_analysis.json`: all proactive
  predictions and locally reproducible metrics.
- `proactive_rawmass_analysis.json`: frequency-normalization ablation.

Prediction files, indexes, and visual feature chunks are generated under
`results/`.  They stay out of the code-only Git repository and can be
regenerated with the commands in `trace_method/README.md`.

All frozen-encoder compilation and full-backbone runs, including three
separate encodings of the 34,319-image full manifest, 23,745 raw-backbone
predictions, and 23,745 model-specific TRACE decisions, were executed using
physical GPU 0 for every GPU operation.  Cosine matching and action transfer
are deterministic CPU computation.
