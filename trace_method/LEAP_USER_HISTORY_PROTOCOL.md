# LEAP user-specific application evidence control

Recorded 2026-09-13 before constructing new donor assignments or predictions.
This bounded study addresses the omitted application preference at task entry;
it does not identify user-specific content recovery at later interaction steps.

For each of the 775 execution queries, take the production top-ten same-user
residual-goal neighbors and their sealed vote weights. Match each neighbor to a
distinct historical record belonging to another user. Admissible donor edges
require cosine >= 0.90 between the two historical residual goals and absolute
difference <= 0.02 between their query similarities. Prefer exact normalized
residual-goal matches, then higher residual-goal cosine and closer query scores.
Use minimum-cost bipartite assignment with no donor reuse within a query. No
application labels, test app, target action, prediction, or outcome enters the
assignment cost or eligibility. All 775 queries enter the matching audit.

Evaluate all queries admitting a complete ten-record match; report exclusions
and balance for the full requested population. Do not relax calipers based on
outcomes or retain only cases where the application prediction changes. Donors
may belong to multiple other users, so this is a matched cross-user record
control, not a coherent single-user profile swap.

Own-user evidence is the exact production application vote. In the matched
control, replace the ten app labels by the donor labels while preserving the
original ranks and positive-cosine weights. This isolates user-associated app
choices conditional on closely matched historical intent evidence. It is a
diagnostic intervention, not a separately optimized retrieval baseline.
The frozen policy sees identical current inputs and the same evidence schema;
only the app reference changes. Include a no-history frozen-policy baseline.

Primary outcome: policy application-entry SSR, own minus matched cross-user,
with paired 20,000-draw user-cluster bootstrap. Secondary: reference app
agreement, own minus no-history, matched coverage, semantic and query-score
balance, and number of unchanged reference apps. Report the full matched set.
Data were previously inspected; this is exploratory, not confirmatory. Limited
coverage or residual confounding must constrain personalization claims.
