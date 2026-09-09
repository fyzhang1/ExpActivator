# PersonalAlign paper-baseline reconstruction

## Result

The local baseline reconstruction covers 775 episodes and 7,915 ordered
actions for each of four memory baselines on MAI-UI-8B, Qwen3.5-9B, and
GUI-Owl-1.5-8B-Instruct. The consolidated table additionally includes the
published Qwen3-VL baselines and our complete Qwen3-VL TRACE result. All
numbers are percentages; higher is better except for parse error. The raw
models without memory context are included as references.

### Consolidated four-backbone table

Each cell reports `Type / SSR / CSR` in percent. TRACE is deliberately placed
in the last row.

| Method | Qwen3-VL-8B | MAI-UI-8B | Qwen3.5-9B | GUI-Owl-1.5-8B |
|---|---:|---:|---:|---:|
| Base | 46.600 / 20.600 / 26.600 | 51.762 / 17.890 / 22.260 | 50.120 / 17.183 / 21.427 | 58.181 / 22.742 / 21.515 |
| Recent† | 49.400 / 21.100 / 33.200 | 47.593 / 18.143 / 21.158 | 49.387 / 17.915 / 25.471 | 59.078 / 23.917 / 27.244 |
| Retrieve | 51.000 / 22.400 / 35.400 | 49.690 / 21.554 / 32.707 | 49.210 / 19.949 / 32.535 | 60.253 / 28.187 / 41.641 |
| LLM-UM‡ | 51.200 / 22.300 / 35.200 | 48.086 / 19.015 / 28.323 | 52.445 / 20.708 / 39.831 | 57.941 / 24.839 / 30.762 |
| HIM-Agent | 52.000 / 24.000 / 42.300 | 48.162 / 19.912 / 28.673 | 51.358 / 20.960 / 37.833 | 58.724 / 27.189 / 35.583 |
| **TRACE (ours)** | **58.825 / 39.116 / 53.615** | **58.939 / 39.330 / 53.812** | **59.495 / 39.949 / 54.870** | **60.430 / 40.076 / 54.875** |

The Qwen3-VL baseline cells are the published values from PersonalAlign
Table 5; the other three baseline columns are the full local reconstructions
reported below. Qwen3-VL TRACE is our complete local result, so comparisons
between that cell and the published Qwen3-VL cells are descriptive rather
than paired. The ready-to-paste version is
[`COMPLETE_EXPERIMENT_TABLE.tex`](COMPLETE_EXPERIMENT_TABLE.tex).

In the Qwen3-VL column, LLM-UM‡ is the paper's original GPT-5.1 result. In the
three locally reconstructed columns it uses the offline Qwen3.5-9B substitute
described below.

## Detailed local results

### MAI-UI-8B

| Method | Type | SSR | CSR | First-step SR | Non-first SR | Parse error |
|---|---:|---:|---:|---:|---:|---:|
| Raw base | 51.762 | 17.890 | 22.260 | 22.323 | 17.409 | 0.834 |
| Recent† | 47.593 | 18.143 | 21.158 | 17.806 | 18.179 | 1.541 |
| **Retrieve** | **49.690** | **21.554** | **32.707** | **38.194** | **19.748** | 1.162 |
| LLM-UM* | 48.086 | 19.015 | 28.323 | 30.065 | 17.815 | 1.327 |
| HIM-Agent | 48.162 | 19.912 | 28.673 | 33.935 | 18.389 | 1.402 |

### Qwen3.5-9B

| Method | Type | SSR | CSR | First-step SR | Non-first SR | Parse error |
|---|---:|---:|---:|---:|---:|---:|
| Raw base | 50.120 | 17.183 | 21.427 | 22.839 | 16.569 | 0.202 |
| Recent† | 49.387 | 17.915 | 25.471 | 32.129 | 16.373 | 0.139 |
| Retrieve | 49.210 | 19.949 | 32.535 | 42.968 | **17.451** | 0.076 |
| LLM-UM* | **52.445** | 20.708 | **39.831** | **59.742** | 16.471 | 0.051 |
| HIM-Agent | 51.358 | **20.960** | 37.833 | 56.258 | 17.129 | 0.164 |

### GUI-Owl-1.5-8B-Instruct

| Method | Type | SSR | CSR | First-step SR | Non-first SR | Parse error |
|---|---:|---:|---:|---:|---:|---:|
| Raw base | 58.181 | 22.742 | 21.515 | 26.839 | 22.297 | 0.013 |
| Recent† | 59.078 | 23.917 | 27.244 | 34.452 | 22.773 | 0.000 |
| **Retrieve** | **60.253** | **28.187** | **41.641** | **66.581** | 24.020 | 0.000 |
| LLM-UM* | 57.941 | 24.839 | 30.762 | 42.452 | 22.927 | 0.000 |
| HIM-Agent | 58.724 | 27.189 | 35.583 | 54.581 | **24.216** | 0.000 |

Bold values identify the best memory baseline for each metric, not the raw
reference. `*` marks the LLM-UM substitution described below.

## Paired findings

The paired bootstrap resamples whole episodes (20,000 samples, seed 20260908).
Intervals below are 95% confidence intervals in percentage points.

- On MAI-UI-8B, Retrieve is the best memory baseline on all five accuracy
  measures. Relative to the raw model, it improves SSR by +3.664
  `[+2.770,+4.567]` and CSR by +10.447 `[+7.852,+13.027]`. Relative to
  HIM-Agent, it improves SSR by +1.642 `[+0.785,+2.503]` and CSR by +4.033
  `[+0.983,+7.048]`.
- On Qwen3.5-9B, HIM-Agent has the highest SSR (20.960), while LLM-UM has the
  highest Type (52.445), CSR (39.831), and first-step SR (59.742). Their SSR
  difference is only +0.253 for HIM-Agent `[-0.624,+1.148]`. Against the raw
  model, LLM-UM gains +3.525 SSR `[+2.618,+4.439]` and +18.404 CSR
  `[+15.597,+21.049]`; HIM-Agent gains +3.778 SSR `[+2.851,+4.727]` and
  +16.406 CSR `[+13.708,+19.052]`.
- On GUI-Owl, Retrieve is best on Type, SSR, CSR, and first-step SR;
  HIM-Agent is ahead by 0.196 points on non-first SR. Retrieve exceeds
  HIM-Agent by +0.998 SSR `[+0.131,+1.887]` and +6.057 CSR
  `[+3.638,+8.424]`.

The ranking is therefore backbone-dependent: Retrieve leads MAI-UI and
GUI-Owl, whereas Qwen3.5 makes the compressed LLM-UM/HIM contexts more
competitive. Most of each method's gain comes from the personalized first
action; the non-first gains are much smaller under teacher-forced action
history.

## Reconstruction protocol and limits

The released repository does not include runnable end-to-end implementations
of Recent, Retrieve, or LLM-UM, and its HIM-Agent execution path does not
directly expose this three-backbone matrix. This experiment reconstructs the
four contexts without using execution-test answers:

- Recent†: latest three full intent/app/action records;
- Retrieve: top-three same-user intent records by normalized
  Qwen3-Embedding-0.6B cosine similarity;
- HIM-Agent: the released history filter unchanged, followed by its released
  `cluster_len >= 3`, top-three execution retrieval behavior and record
  representation;
- LLM-UM*: deterministic per-user summaries from the fixed local Qwen3.5-9B
  snapshot. Its input is capped at the most recent 150 split-history records
  and 30,000 tokens per user. The first pass requested at most 10 summaries
  and three actions each; six incomplete profiles were regenerated with a
  six-summary/two-action cap while preserving the other 77.

The paper specifies `gpt-5.1-2025-11-13` for LLM-UM, but neither the generated
profiles nor a runnable builder are released and that model was unavailable
offline. Consequently LLM-UM* is an explicit local substitute, not an exact
reproduction of the paper's LLM-UM numbers. These results also extend the
paper baselines to three different GUI backbones, so they should not be read
as a replication of the paper's published Qwen3-VL table.

All contexts use the chronological first 80% of each user's factual history
(15,966 rows) and were sealed for all 775 episodes before inference. Each run
uses the same ambiguous instruction, screenshot, teacher-forced prior actions,
native backbone prompt, BF16 greedy decoding, 200,704-pixel image budget,
256-token response cap, and 8,192-token model context. Every audited prompt,
including its image and response allowance, fits that ceiling.

The analysis validated exact ordered coverage, unique step IDs, test and
context hashes, model revisions, baseline labels, prompt protocol, and
physical GPU 0 isolation for all 12 runs. The model snapshots were
`e00a0097abb9cc621cac5172d8c4809f0839c94e` (MAI-UI),
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Qwen3.5), and
`06d5faecff74840bab2be2425e9c42667a5d04fc` (GUI-Owl). Inference used vLLM
0.28.0, Transformers 5.16.1, and PyTorch 2.13.0+cu130.

The ignored machine-readable artifact is
`results/paper_baseline_analysis.json`; the reproducer and integrity analysis
are [`../scripts/execution/eval_paper_baselines.sh`](../scripts/execution/eval_paper_baselines.sh)
and [`analyze_paper_baselines.py`](analyze_paper_baselines.py).
