# Partial matched-history replacement: coverage extension

Recorded 2026-09-13 after inspecting matching coverage (25/775 complete matches)
but before any user-history ablation policy predictions or target scoring.
The strict ten-record matching protocol remains intact; its 25-task result
will also be reported. No similarity calipers are relaxed.

For all 775 task-entry queries, reuse exactly the same admissible edges and
assignment cost. Replace every admissibly assigned source record by its
cross-user donor; leave unmatched source records unchanged. Preserve all ten
original ranks and vote weights. This creates a partial replacement diagnostic,
not a complete foreign-user memory. It can understate the consequence of
removing user-specific evidence because some own-user records remain.

Run own, partially replaced, and no-history policy conditions on the same
775 entry steps. Primary extension outcome is own minus partially replaced
policy SSR over all 775 queries. Secondary strata are queries with at least
one admissible replacement and the original 25 complete matches. These strata
depend only on matching inputs, not target answers or policy correctness.
Record the number of replaced records, changed reference apps, balance and
coverage. Reuse the full-run outputs for the original 25-task strict comparison
because both their own and donor contexts are identical to the strict study.
The all-775 entry-only batches differ from earlier full-trajectory batches;
all new entry conditions use the same entry-only batch schedule.

Use paired user-cluster 95% bootstrap intervals with 20,000 draws, plus episode
resampling sensitivity. Retain all conditions and disclose the coverage-driven
extension; it must not be presented as an independent confirmation.
