# LEAP decision-conditioned experience ablations

Protocol recorded on 2026-09-13 before generating the new predictions.
This is an exploratory study on an already inspected benchmark. All valid
conditions and raw outputs are retained, regardless of effect direction.
Historical experiments and the active manuscript are preserved.

## Execution: reference refresh

The primary contrast is current-state selection minus one-step-delayed
selection. A secondary contrast compares current-state selection with a
reference refreshed every four non-entry steps (positions 1, 5, 9, ...).
All three conditions use the same fixed per-episode Top-8 trajectory pool,
frozen Qwen3-VL-8B policy, current screenshot, ambiguous instruction,
teacher-forced action prefix, evidence schema, and inference settings.
The first non-entry step and application entry are identical across conditions.
For delayed selection, the retrieval screen is s[max(1,t-1)]. For periodic
refresh it is s[1+4*floor((t-1)/4)]. The policy always receives s[t].

Because the semantic neighborhood is fixed across a task and the selector is
stateless, the sealed current-state evidence at the retrieval-screen position
is exactly the evidence obtained by rerunning selection on that screen.
The builder reindexes those existing contexts and source records; it does not
consult target actions, outcome scores, or complete target instructions.
Current-state inference is rerun with the same batch settings as the controls.

Primary outcome: benchmark policy SSR on all 7,140 non-entry steps from 775
episodes. Secondary outcomes: reference-action agreement under the same scorer,
Type accuracy, interactive-action SSR, positions after the first non-entry step,
and identical-context diagnostics. The reference score is not policy success.
Report paired 95% bootstrap intervals over users (20,000 draws), with episode
clustering as sensitivity. The current-vs-delayed contrast is primary; the
refresh-every-four contrast is secondary and not selected according to results.
No closed-loop recovery or increasing-with-horizon claim follows from this
teacher-forced experiment. Intervals describe data sampling, not runtime noise.

## Additional studies

Proactive context-factor ablations and a matched user-history replacement
control will be specified separately before their corresponding new outputs.
They must not be substituted for an unfavorable reference-refresh result.
No threshold or subgroup is chosen to maximize test improvement.

## Reporting

Keep complete raw outputs, input/source hashes, all declared conditions,
failures, uncertainty, and inference settings. Main-text placement depends on
scientific relevance and validity. A contradicting result changes the claim
scope; it is not removed from the experiment record. Invalid implementations
are corrected with the reason documented, not silently replaced.
