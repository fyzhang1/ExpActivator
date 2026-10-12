# LEAP proactive context-factor ablations

Recorded 2026-09-13 before new ablation outputs. Exploratory benchmark analysis;
retain every condition and do not tune on the test set.

Use the exact frozen historical intent index and all 315 released query states
(215 positive, 100 negative). Retrieval receives only user ID, time, scenario;
exclude records with the query timestamp. The existing index manifest identifies
paraphrase-multilingual-MiniLM-L12-v2 as the retrieval encoder. Qwen3-Embedding-0.6B
is the official evaluation encoder, not the routine retrieval encoder.

Declared conditions: full, no_time, no_scenario, raw_mass. Full must reproduce
the sealed production source IDs and suggestion strings. No_time replaces the
time kernel by one. No_scenario removes the scenario indicator, the scenario
support gate, and the same-scenario restriction on representative selection.
Raw_mass keeps the full eligible set and switches only selection power from
one to zero. All other parameters stay at their published configuration.

For full/no_time/no_scenario, primary threshold-independent diagnostic is AUROC
of max contextual mass across support-eligible routines; use -1 when none is
support-eligible. Report complete recall/false-alarm curves and the existing
delta=0.25 operating point as an unretuned removal diagnostic. Do not interpret
that single fixed-delta comparison as controlling for changed score scales.
No operating threshold is selected from the test curve. Use paired user-cluster
bootstrap intervals (20,000 draws) for AUROC and balanced-accuracy differences.

For raw_mass, require bit-identical trigger decisions and compare official
Semantic on the same emitted positive queries. Report both conditional Semantic
and all-positive coverage-weighted Semantic (abstentions assigned zero), clearly
labeling the latter as an additional diagnostic rather than an official metric.
For no_time/no_scenario, conditional scores can change because coverage changes;
also report the all-positive diagnostic. Semantic follows the official per-row
rounding of the average of Qwen3-Embedding cosine and fuzzy edit similarity.

These are model-independent selector ablations. They do not replace the
backbone-decoded suggestion results and will be labeled separately. Generated
suggestions, selected source records, scores, threshold curves and hashes are
saved before inspecting outcomes. A negative finding constrains the claims.
