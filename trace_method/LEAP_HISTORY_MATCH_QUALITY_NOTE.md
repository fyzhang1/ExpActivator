# Historical-goal matching quality sensitivity

Recorded 2026-09-13 before any own/foreign/no-history entry policy runs.
After constructing the fixed matching inputs, a seed-20260913 random sample
of 20 replaced record pairs was reviewed with app labels and target answers
hidden. This assistant review found that embedding similarity does not ensure
complete task equivalence: for example, playing a saved video and saving a
video can have similar embeddings. No pair or query was removed or rematched.

Add one prespecified-for-entry-inference sensitivity stratum to the unchanged
775-query intervention: at least one record is replaced, and every replaced
record's historical residual goal matches its donor exactly after the already
defined alphanumeric/lowercase normalization. Keep queries with unchanged
application references in this stratum. This uses only matching inputs, not
policy outcomes. Report it alongside all queries, any replacement, and complete
ten-record matching. The small assistant review is not a human-gold semantic
validation, and the primary cosine-caliper experiment remains approximate.

Dot products use the stored half-precision normalized text vectors converted
back to float32. Small norm deviations can produce values just above one;
these are approximate cached cosine scores. No caliper or assignment is
changed after inspecting this numerical property.
