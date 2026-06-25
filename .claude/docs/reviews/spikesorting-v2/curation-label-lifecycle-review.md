# Spike Sorting V2 Curation Label / Unit Identity Lifecycle Review

Date: 2026-06-25

Scope: `CurationV2` labels, merge previews vs applied merges, parent/child curation semantics, analyzer-driven curation materialization, UnitMatch matchable-unit identity, TrackedUnit identity derivation, exported Units NWB label semantics, and downstream label filtering.

Method: main-agent source/test review plus two independent read-only agents: one focused on source-level lifecycle semantics and one focused on tests/docs coverage.

## Findings

### 1. High: AnalyzerCuration is curation-keyed, but computes over the raw sort namespace

Evidence:

- `AnalyzerCurationSelection` identity includes `sorting_id` and `curation_id` and FKs to `CurationV2`: [src/spyglass/spikesorting/v2/metric_curation.py:657](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:657), [src/spyglass/spikesorting/v2/metric_curation.py:718](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:718).
- The compute path ignores the selected curation's unit set and loads analyzers by `sorting_id` only: [src/spyglass/spikesorting/v2/metric_curation.py:916](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:916), [src/spyglass/spikesorting/v2/metric_curation.py:933](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:933), [src/spyglass/spikesorting/v2/metric_curation.py:952](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:952).
- Labels and merge suggestions are computed from those sort-level analyzers and raw-sort unit ids: [src/spyglass/spikesorting/v2/metric_curation.py:957](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:957), [src/spyglass/spikesorting/v2/metric_curation.py:966](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:966), [src/spyglass/spikesorting/v2/metric_curation.py:967](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:967).
- `materialize_curation` creates a child of the selected curation, but passes labels/merge groups that were computed in raw-sort space: [src/spyglass/spikesorting/v2/metric_curation.py:1399](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:1399), [src/spyglass/spikesorting/v2/metric_curation.py:1407](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/metric_curation.py:1407).
- The user docs explicitly recommend re-running AnalyzerCuration on a merged curation and say the second pass computes over post-merge templates: [docs/src/Features/SpikeSortingV2.md:544](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:544), [docs/src/Features/SpikeSortingV2.md:556](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:556).

Impact: users following the documented auto -> manual merge -> auto loop can believe the second pass recomputed metrics and labels over merged units, while the current compute path still uses the original sort analyzer. For an applied-merge parent, that means metrics, labels, and suggested merges can be in the wrong unit namespace and then materialized under a child `curation_id`.

Fix direction:

- Short-term: reject `AnalyzerCurationSelection.insert_selection` on non-pass-through curations, especially `merges_applied=True` or curations with non-self merge groups, until curated analyzers exist.
- Long-term: build analyzer-curation inputs from `CurationV2.get_sorting` / curated-unit metadata, not from `Sorting.get_analyzer({"sorting_id": ...})`.
- Add an end-to-end test that creates an applied-merge parent, inserts an AnalyzerCurationSelection on that parent, and asserts either a clear rejection or metrics indexed by the merged unit ids.

### 2. High: child curations do not compose with parent curation state

Evidence:

- `parent_curation_id` is validation-only in the class docstring and is checked only for existence within the same sort: [src/spyglass/spikesorting/v2/curation.py:68](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:68), [src/spyglass/spikesorting/v2/curation.py:571](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:571).
- Every new curation fetches upstream `Sorting.Unit` rows, not parent `CurationV2.Unit` rows: [src/spyglass/spikesorting/v2/curation.py:312](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:312), [src/spyglass/spikesorting/v2/curation.py:320](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:320).
- The pure curation plan builds from the original sort unit map: [src/spyglass/spikesorting/v2/_curation_transforms.py:121](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_curation_transforms.py:121), [src/spyglass/spikesorting/v2/_curation_transforms.py:178](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_curation_transforms.py:178).
- Labels written into the child are only the new `labels` argument; parent labels are not inherited: [src/spyglass/spikesorting/v2/curation.py:765](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:765), [src/spyglass/spikesorting/v2/curation.py:770](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:770).
- Tests validate that a child can reference a parent, but they do not pin inheritance/composition semantics: [tests/spikesorting/v2/single_session/test_curation_insert.py:132](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:132), [tests/spikesorting/v2/test_curation_wrappers.py:258](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_curation_wrappers.py:258).

Impact: a child curation created from a labeled parent with `labels=None` drops `reject` / `noise` / `artifact` labels and can make previously excluded units matchable again. A child of an applied-merge parent can also revert to raw contributor unit ids rather than the parent's curated unit identity. This is a dangerous mismatch with the intuitive meaning of "child curation" unless the system explicitly defines parentage as provenance-only.

Fix direction:

- Decide and document the contract. If `parent_curation_id` is provenance-only, say child curations do not inherit labels or unit sets and require callers to pass the complete desired curation state.
- If parent-relative behavior is intended, seed child planning from parent `CurationV2.Unit`, `UnitLabel`, and `MergeGroup` rows, then apply deltas.
- Add tests for a labeled parent followed by a child with `labels=None`, and an applied-merge parent followed by a child. Assert the chosen unit ids, labels, and `get_matchable_unit_ids` behavior.

### 3. High: DB labels can diverge from the already-written Units NWB labels

Evidence:

- `CurationV2.UnitLabel` explicitly supports direct validated inserts: [src/spyglass/spikesorting/v2/curation.py:108](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:108), [src/spyglass/spikesorting/v2/curation.py:134](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:134).
- Tests exercise direct post-creation `UnitLabel.insert1`, including custom labels: [tests/spikesorting/v2/single_session/test_curation_insert.py:688](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:688), [tests/spikesorting/v2/single_session/test_curation_insert.py:734](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:734).
- The curated Units NWB `curation_label` column is written only while staging the curation artifact: [src/spyglass/spikesorting/v2/_units_nwb.py:890](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_units_nwb.py:890), [src/spyglass/spikesorting/v2/_units_nwb.py:911](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_units_nwb.py:911).
- DB-facing label consumers read current `UnitLabel` rows: [src/spyglass/spikesorting/v2/curation.py:1069](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:1069), [src/spyglass/spikesorting/v2/curation.py:1931](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:1931).
- Downstream grouped-spike reads filter labels from the NWB table when the label column is present: [src/spyglass/spikesorting/analysis/v1/group.py:218](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/analysis/v1/group.py:218), [src/spyglass/spikesorting/analysis/v1/group.py:236](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/analysis/v1/group.py:236).

Impact: after a direct `UnitLabel` insert/delete, `CurationV2.get_matchable_unit_ids`, `summarize_curation`, and brain-region filters can see one label state while `SpikeSortingOutput.fetch_nwb`, `get_spike_times`, and `SortedSpikesGroup.fetch_spike_data` see the older label state embedded in the NWB. The same `merge_id` can therefore mean different included units depending on which API consumes it.

Fix direction:

- Treat `CurationV2` part rows as immutable after `insert_curation`, or require a single label-update API that creates a new curation and new NWB artifact.
- If direct part inserts remain supported for admin/debug reasons, document that they are not semantically valid curation edits.
- Add a regression test: insert a curation, mutate `UnitLabel` directly, then assert either the mutation is rejected or DB/NWB label consumers remain synchronized.

### 4. Medium-high: downstream include-label filters are skipped when v2 omits the label column for all-unlabeled curations

Evidence:

- The Units NWB writer intentionally omits `curation_label` when all units are unlabeled to avoid PyNWB dtype inference failures: [src/spyglass/spikesorting/v2/_units_nwb.py:904](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_units_nwb.py:904), [tests/spikesorting/v2/single_session/test_curation_insert.py:1138](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:1138).
- `SortedSpikesGroup.fetch_spike_data` only calls `filter_units` when it finds a `label` or `curation_label` column: [src/spyglass/spikesorting/analysis/v1/group.py:218](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/analysis/v1/group.py:218), [src/spyglass/spikesorting/analysis/v1/group.py:236](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/analysis/v1/group.py:236).
- The pure filter semantics are correct: include-only filters should keep only units carrying the include label, so unlabeled units should be excluded: [tests/spikesorting/v2/test_downstream_consumers.py:465](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_downstream_consumers.py:465).
- Existing tests cover the pure helper, not the missing-column integration path; the integration call is intentionally bypassed under `test_mode`: [tests/spikesorting/v2/test_downstream_consumers.py:420](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_downstream_consumers.py:420), [src/spyglass/spikesorting/analysis/v1/group.py:228](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/analysis/v1/group.py:228).

Impact: for an all-unlabeled v2 curation, a downstream group configured with `include_labels=["accept"]` or another include-only filter can include every unit instead of none, because the missing column is treated as "no label filtering" rather than "empty label list for each unit."

Fix direction:

- In `SortedSpikesGroup.fetch_spike_data`, when no label column exists, synthesize `[[] for unit in units]` and still run `filter_units` whenever include/exclude parameters are non-empty.
- Add an integration test with an all-unlabeled v2 curation and an include-only `UnitSelectionParams` row.
- Document that omitted `curation_label` means every unit is unlabeled, not that label state is unknown.

### 5. Medium: UnitMatch can persist FK-valid but non-matchable units returned by a backend

Evidence:

- `UnitMatch.make_fetch` freezes `matchable_unit_ids` from the selected curation labels: [src/spyglass/spikesorting/v2/unit_matching.py:580](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:580), [src/spyglass/spikesorting/v2/unit_matching.py:592](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:592).
- `make_compute` extracts only those matchable units into the matcher bundle: [src/spyglass/spikesorting/v2/unit_matching.py:787](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:787), [src/spyglass/spikesorting/v2/unit_matching.py:792](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:792).
- `canonicalize_match_pairs` validates that pair sides belong to pinned curations, but it does not validate unit ids against the frozen matchable set: [src/spyglass/spikesorting/v2/_matcher_graph.py:99](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_matcher_graph.py:99), [src/spyglass/spikesorting/v2/_matcher_graph.py:142](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_matcher_graph.py:142).
- `UnitMatch.Pair` FKs only require membership in `CurationV2.Unit`, not membership in the matchable subset: [src/spyglass/spikesorting/v2/unit_matching.py:459](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:459), [src/spyglass/spikesorting/v2/unit_matching.py:471](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:471).
- `TrackedUnit` later re-derives the matchable universe and will raise if an edge endpoint is absent: [src/spyglass/spikesorting/v2/unit_matching.py:864](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:864), [src/spyglass/spikesorting/v2/_matcher_graph.py:289](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_matcher_graph.py:289).

Impact: the built-in UnitMatch backend should only echo units it was fed, so this is mainly a custom-backend or backend-bug boundary. But if it happens, invalid pairs can be written and only fail later at `TrackedUnit.populate`, turning a matcher-output validation error into a downstream lifecycle failure.

Fix direction:

- Pass the frozen allowed unit sets into `canonicalize_match_pairs` and reject pair sides outside `(sorting_id, curation_id, matchable_unit_ids)`.
- Revalidate pairs read back from the staged NWB before `UnitMatch.Pair.insert`.
- Add a fake-backend test that returns an excluded but FK-valid unit id and assert `UnitMatch.populate` fails before writing pairs.

### 6. Medium: TrackedUnit docs say "curated units" but the identity universe is matchable curated units

Evidence:

- The docs say tracked units form a strict partition where "each curated unit belongs to exactly one tracked unit": [docs/src/Features/SpikeSortingV2.md:960](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:960).
- `TrackedUnit.make` seeds nodes from `CurationV2.get_matchable_unit_ids`, which excludes `reject`, `noise`, and `artifact` by default: [src/spyglass/spikesorting/v2/unit_matching.py:893](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:893), [src/spyglass/spikesorting/v2/curation.py:1903](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:1903).
- The tracked-unit singleton test describes "curated units" but asserts against the matchable set: [tests/spikesorting/v2/test_unitmatch.py:1118](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_unitmatch.py:1118).

Impact: users may expect rejected/noise/artifact units to appear as singleton tracked identities. In reality, they are outside the matching/tracking universe, and a curation with zero matchable units cannot run UnitMatch at all.

Fix direction:

- Update docs and test language to say "matchable curated units."
- List the default excluded labels in the UnitMatch/TrackedUnit section.
- Add a small test or docs example for an all-excluded member curation.

### 7. Medium-low: curation_id allocation is race-prone and stages work before detecting conflicts

Evidence:

- New curation ids are assigned as `max(existing_ids) + 1`: [src/spyglass/spikesorting/v2/curation.py:663](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:663).
- The curated Units NWB is staged before the transaction that inserts the master row: [src/spyglass/spikesorting/v2/curation.py:376](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:376), [src/spyglass/spikesorting/v2/curation.py:391](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:391).

Impact: two concurrent curation inserts for the same sorting can choose the same `curation_id`; one loses after doing filesystem work. Cleanup should remove the staged file, but users still see a spurious failure and must retry.

Fix direction:

- Allocate `curation_id` under a per-sorting lock or use retry-on-duplicate to refetch/replan with the next id.
- Add a two-worker insert/materialize race test if concurrent curation creation is expected.

### 8. Low: export and split semantics need clearer user docs

Evidence:

- Tests establish that curated Units NWB labels are ragged list columns, omitted when all units are unlabeled, and present when at least one unit is labeled: [tests/spikesorting/v2/single_session/test_curation_insert.py:569](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:569), [tests/spikesorting/v2/single_session/test_curation_insert.py:1138](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:1138), [tests/spikesorting/v2/test_metric_curation_nwb.py:72](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_metric_curation_nwb.py:72).
- The user docs mention export but do not define this label-column contract: [docs/src/Features/SpikeSortingV2.md:1017](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:1017).
- The docs describe labels and merges but do not explicitly state whether split curation is unsupported or modeled elsewhere: [docs/src/Features/SpikeSortingV2.md:79](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:79), [docs/src/Features/SpikeSortingV2.md:455](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:455).

Impact: external readers may expect scalar labels or an always-present `curation_label` column. Users may also assume split curation has merge-like lineage support.

Fix direction:

- Add an export subsection documenting `curation_label` as a ragged per-unit list, omitted for all-unlabeled/zero-unit NWBs.
- Add an explicit note that v2 curation currently models labels and merges; splits require a new sorting/curation lineage or are not yet represented.

## Positives

- Label value validation is strong and covers direct part inserts, custom-label opt-in, scalar-string mistakes, and ordered-row bypass attempts: [src/spyglass/spikesorting/v2/curation.py:134](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:134), [tests/spikesorting/v2/single_session/test_curation_insert.py:688](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:688), [tests/spikesorting/v2/single_session/test_curation_insert.py:746](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_insert.py:746).
- Merge behavior is carefully specified and tested: preview vs applied merges, absorbed-label handling, canonical merged ids, invalid groups, overlapping groups, and lazy-vs-applied frame equality all have coverage: [tests/spikesorting/v2/single_session/test_curation_merges.py:96](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_merges.py:96), [tests/spikesorting/v2/single_session/test_curation_merges.py:200](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/single_session/test_curation_merges.py:200).
- `MergeGroup` self-entries make merge provenance queryable without dropping unmerged or preview contributor units: [src/spyglass/spikesorting/v2/curation.py:795](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/curation.py:795), [src/spyglass/spikesorting/v2/_curation_transforms.py:285](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_curation_transforms.py:285).
- UnitMatch avoids "latest curation" drift by pinning one explicit curation per member and hashing the curation choices: [src/spyglass/spikesorting/v2/unit_matching.py:197](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:197), [src/spyglass/spikesorting/v2/_matcher_graph.py:37](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_matcher_graph.py:37).
- The consumer-boundary guard rejects v2 preview curations before decoding groups consume them: [src/spyglass/spikesorting/spikesorting_merge.py:346](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/spikesorting_merge.py:346), [tests/spikesorting/v2/test_preview_merge_warning.py:184](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_preview_merge_warning.py:184).

## Suggested triage

- Treat finding 1 as the urgent correctness fix because it directly contradicts the documented post-merge analyzer-curation loop.
- Treat findings 2 and 3 as lifecycle contract decisions: either make curations immutable snapshots or provide explicit parent-relative/update semantics.
- Finding 4 is a downstream behavior bug with a small local fix once the omitted-label-column contract is accepted.
- Findings 5 and 6 are matcher/tracker boundary hardening and docs alignment.
