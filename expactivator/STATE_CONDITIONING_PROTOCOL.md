# State-conditioned use of personal experience: analysis protocol

Recorded on 2026-09-12 before constructing new ablation predictions. This is
an exploratory mechanism study on an already-used public benchmark, not a new
sealed-test claim. Existing ExpActivator outputs were available during design.

## Question and primary estimand

Holding the retrieved historical trajectory fixed for an episode, does
current-screen state matching improve non-entry action evidence and the frozen
GUI policy's non-entry step success rate relative to ordinal alignment?

The primary comparison uses Qwen3-VL-8B and all 775 episodes / 7,915 steps.
First-step evidence, current screenshots, ambiguous instructions, teacher-forced
prefixes, policy prompts, decoding, and history split are identical. The
primary outcome excludes the 775 entry steps. No prediction parameter is fit.

## Fixed controls

1. `top1_ordinal`: select the highest-ranked readable same-user historical
   trajectory using the existing residual-goal ranking. For target position
   t > 0, use historical index min(t - 1, T - 1). The entry action is synthetic
   in the benchmark and is absent from the recorded screenshot trajectory.
2. `top1_state`: use exactly the same historical trajectory for every step of
   the episode; select its historical state by the already history-calibrated
   visual descriptor and current screenshot. No access to target actions.
3. `top8_previous`: start from the existing full ExpActivator's selected historical
   trajectory/state, but supply the preceding historical action (clamped at
   the trajectory beginning). This is a deliberately perturbed phase control,
   not a competitive baseline; it isolates local evidence mismatch conditional
   on ExpActivator's trajectory selection.
4. Existing `top8_state` ExpActivator: the complete current grounded-policy result.

Implementation note before policy inference: also rerun `top8_state` under the
current evaluator to avoid relying on an unavailable historical prompt-source
hash. Keep the prior output as a reproducibility reference, not the only
control for the new phase perturbation.

All new contexts have the same generic retrieval-basis string, evidence schema,
and advisory instruction. The labels above are never disclosed to the policy.
For a matched comparison with existing ExpActivator, preserve its retrieval-basis
string in all conditions. Source metadata is outside the policy context.

Direct evidence scores are diagnostics, not ExpActivator policy outputs. Report both
and do not splice base outputs into an otherwise grounded ablation.

## Task relevance and subgroup analysis

Report all episodes without selecting on outcome. Independently annotate the
fixed top-one source's task relevance using only the ambiguous query and
historical task description; no target action, target original instruction,
screenshots, or experiment output is available to the annotator. Relevance
means matching the requested operation and substantive object/constraints,
allowing the entry app to be omitted. Distinguish matching, partial, and
unrelated. Automated annotations are not human gold and require an audit.

Predefined descriptive strata: absolute non-entry position (1-3, 4-7, 8+),
episode length (<=7, 8-12, >=13 total steps), source/target step-count equality,
exact screenshot duplicate versus nonduplicate, action type, and task relevance.
Step-count mismatch is not an annotation of semantic phase. Screenshot identity
is tested from decoded RGB pixels, not embedding similarity alone.

## Statistics and claim limits

Use paired resampling of users as the main 95% uncertainty interval, with
episode-cluster resampling as a sensitivity analysis. Keep all steps of a
sampled cluster together. Report effect sizes, discordant wins/losses, and
sample sizes; subgroup comparisons are exploratory. Do not infer a
between-subgroup difference from one interval excluding zero and another not.

This offline protocol cannot establish closed-loop recovery. Non-entry action
accuracy does not by itself establish recovery of personal preference. Such a
claim requires independently identified preference-bearing decisions and a
control separating user-specific information from generic GUI competence.
If that evidence is unavailable, state the narrower conclusion explicitly.

## Deliverables

New reproducible builders/analysis scripts, complete raw predictions, manifests
and source hashes, a report, paper-ready LaTeX table/paragraph, and standalone
figures. Preserve existing production method and previous manuscript edits.

## Additional exploratory checks (recorded after evidence diagnostics)

The primary pair remains unchanged. Add `top1_trajectory`, which supplies the
same fixed source's complete action record through the native paper memory
prompt, to check whether step selection helps compared with full-record
prompting. Its evidence format, token budget actually consumed, and prompt
template differ from the action-evidence pair; report those differences rather
than interpreting it as a pure state-only intervention. Entry evidence stays
identical. This comparator was added after evidence-only scores were available
and before either its policy output or the state-matched policy output existed.

Also report a conservative subgroup requiring both goal-match consensus and
exact canonical source/target app-name agreement. Target app labels are used
only to define this diagnostic subgroup, never to choose evidence or predict
actions. This is an additional exploratory check, not a prespecified endpoint.

Before the state-matched policy run, add the intersection of literal residual-goal
equality and exact app equality, an interactive-action subset (click, long click,
scroll, type), and sensitivity analyses with click-distance thresholds 0.07 and
0.05 instead of the benchmark's 0.14. These are exploratory robustness checks
motivated by inspection of evidence and screenshots. Other action scores remain
unchanged, including the benchmark's treatment of one nonstandard target label.

During the state-policy run, an additional audit scores typed values by exact
whitespace-normalized, case-insensitive equality. It reports all parseable type
targets and a descriptive subset whose value has at least two characters and
does not occur literally in the ambiguous query. Literal absence does not
establish missing semantics or personal preference: paraphrases, partial input,
and information visible in the screen or action prefix remain possible. This
check was added after evidence outputs and the ordinal policy were available;
all results, including zero recoveries, must be reported without selecting values.

After the complete primary policy pair, decompose policy differences into the
four exhaustive groups defined by ordinal/state reference correctness: neither,
state only, ordinal only, and both correct. This is an outcome-conditioned error
description, not evidence that reference correctness causally mediates the policy
effect. Keep all four groups and do not replace the primary aggregate with one.

The opposite directions in the length strata are followed by a direct paired
interaction check (unequal-length gain minus equal-length gain), using the same
resampled users in both strata. This additional test is exploratory and was
implemented after the complete primary pair; report its interval rather than
inferring an interaction only from the two separate within-stratum intervals.
