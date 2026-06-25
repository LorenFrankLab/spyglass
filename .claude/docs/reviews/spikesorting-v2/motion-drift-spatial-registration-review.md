# Spike Sorting V2 Motion / Drift / Spatial Registration Review

Date: 2026-06-25

Scope: same-day concatenation, motion correction, drift QC, UnitMatch spatial inputs, probe/channel geometry contracts, session-order semantics, unit-location coordinate frames, and tests/docs that protect these contracts.

Method: main-agent source review plus two independent read-only agents: one focused on source-level motion/spatial semantics and one focused on tests/docs coverage.

## Findings

### 1. High: concat compatibility can equate different physical electrode spaces

Evidence:

- `assert_concat_compatible` checks channel-id order and numeric `get_channel_locations()` equality, but it does not compare electrode identity, electrode group, probe/shank identity, brain region, implant identity, or per-channel electrode-table semantics: [src/spyglass/spikesorting/v2/_concat_recording.py:217](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:217), [src/spyglass/spikesorting/v2/_concat_recording.py:248](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:248), [src/spyglass/spikesorting/v2/_concat_recording.py:279](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:279).
- The materialized concat artifact anchors its AnalysisNWB parent to the first member only: [src/spyglass/spikesorting/v2/session_group.py:872](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:872).
- For concat sorts, `Sorting.Unit` electrode/region metadata is explicitly anchored to the first member's electrode table: [src/spyglass/spikesorting/v2/sorting.py:1153](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/sorting.py:1153).
- Unit peak-channel attribution later resolves an `Electrode` row by anchor `nwb_file_name` and `sort_group_id`: [src/spyglass/spikesorting/v2/sorting.py:2623](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/sorting.py:2623).

Impact: two members can share channel ids and relative probe coordinates while referring to different physical electrodes, shanks, electrode groups, or brain regions. They would pass concat compatibility, then later sessions' unit metadata would be interpreted in the first member's electrode frame. Motion correction makes this easier to miss because the corrected trace cache looks spatially registered even though the metadata frame is only an anchor-member frame.

Fix direction:

- Add a channel-ordered spatial semantics signature before concat. At minimum compare electrode ids, electrode group/probe/shank identity when available, relative x/y/z coordinates, and brain-region labels or a deliberate region-ambiguity policy.
- If strict equality is too limiting for chronic workflows, persist explicit per-member electrode provenance instead of relying on only the first member's `Electrode` FK for all units.
- Add tests where channel ids and numeric locations match but electrode group or brain region differs, and assert concat selection/materialization fails or requires an explicit override.

### 2. High/medium: UnitMatch can receive 3D channel positions despite a 2D matcher contract

Evidence:

- `SessionMatcherInput.channel_positions_path` documents a `.npy` file of shape `(n_channels, 2)`: [src/spyglass/spikesorting/v2/matcher_protocol.py:45](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/matcher_protocol.py:45).
- `extract_unitmatch_bundle` saves `recording.get_channel_locations()` directly, with no 3D-to-2D projection or shape guard: [src/spyglass/spikesorting/v2/_unitmatch_backend.py:107](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_unitmatch_backend.py:107), [src/spyglass/spikesorting/v2/_unitmatch_backend.py:184](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_unitmatch_backend.py:184).
- `UnitMatchBackend.match` checks shape equality across sessions, but not `ndim == 2` or `shape[1] == 2`, then passes the array into UnitMatch geometry setup: [src/spyglass/spikesorting/v2/_unitmatch_backend.py:244](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_unitmatch_backend.py:244), [src/spyglass/spikesorting/v2/_unitmatch_backend.py:264](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_unitmatch_backend.py:264).
- Analyzer construction already recognizes the 3D/2D issue and projects probes before computing 2D-dependent extensions: [src/spyglass/spikesorting/v2/_sorting_analyzer.py:544](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_sorting_analyzer.py:544), [src/spyglass/spikesorting/v2/_sorting_analyzer.py:574](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_sorting_analyzer.py:574).

Impact: a valid Spyglass recording with x/y/z probe coordinates can produce `(n_channels, 3)` UnitMatch inputs. If all sessions share the same 3D shape, the backend geometry check passes, and UnitMatch may fail deep or compute spatial/drift features in a coordinate frame the wrapper did not promise.

Fix direction:

- Share the analyzer's projection helper or introduce a UnitMatch-specific geometry normalization helper.
- Require exact 2D positions at the matcher boundary, and fail clearly for non-planar 3D geometry unless an explicit projection policy is configured.
- Add tests that planar 3D recordings produce 2D UnitMatch bundles, non-planar 3D geometry fails or records an explicit policy, and a prebuilt `(n, 3)` `channel_positions.npy` raises before UnitMatch internals run.

### 3. Medium-high: concat split/back-mapping reads live SessionGroup members after materialization

Evidence:

- `ConcatenatedRecording.make_fetch` resolves the member plan at populate time and carries only `member_index`, `nwb_file_name`, and `recording_id` through compute: [src/spyglass/spikesorting/v2/session_group.py:723](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:723), [src/spyglass/spikesorting/v2/session_group.py:756](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:756).
- `ConcatenatedRecording.MemberBoundary` stores only `member_index` and `end_sample`: [src/spyglass/spikesorting/v2/session_group.py:592](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:592).
- `split_sorting_by_session` later fetches current `SessionGroup.Member` rows and aligns persisted boundaries by `member_index`: [src/spyglass/spikesorting/v2/session_group.py:1030](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:1030), [src/spyglass/spikesorting/v2/session_group.py:1038](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:1038).

Impact: if group membership or member order is edited after the concat artifact is materialized, a concat-frame sorting can be split and labeled using a different member set than the one used to build the recording. The sample math may still look valid while the returned per-session sorting keys point to the wrong session/electrode frame.

Fix direction:

- Persist a frozen member snapshot on the concat artifact, not just `member_index`: `nwb_file_name`, `sort_group_id`, `interval_list_name`, `team_name`, `recording_id`, and pre-motion sample count.
- Make `split_sorting_by_session` use the frozen snapshot, or fail if live `SessionGroup.Member` rows no longer match the stored snapshot/hash.
- Add a test that materializes a concat artifact, mutates/forges `SessionGroup.Member`, and asserts split back-mapping either uses the original frozen members or raises a targeted integrity error.

### 4. Medium: applied concat motion correction discards the motion field and coordinate-frame context

Evidence:

- Concat motion correction deliberately pins `output_motion=False` and `output_motion_info=False`: [src/spyglass/spikesorting/v2/_concat_recording.py:310](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:310), [src/spyglass/spikesorting/v2/_concat_recording.py:376](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:376).
- The persisted concat row stores the corrected `ElectricalSeries`, a preset label in the filtering description, and a `content_hash`, but no displacement trajectory: [src/spyglass/spikesorting/v2/session_group.py:875](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:875), [src/spyglass/spikesorting/v2/session_group.py:876](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:876).
- `DriftEstimate` does store a compact `Motion` blob, but it is QC-only and never applied: [src/spyglass/spikesorting/v2/recording.py:2061](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2061), [src/spyglass/spikesorting/v2/recording.py:2141](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2141).

Impact: corrected traces, template peak channels, and unit locations are interpreted against static electrode geometry, but the displacement field that produced those corrected signals is unavailable. Users cannot audit boundary behavior, per-session displacement, or whether a reported location should be read as a raw-electrode-frame coordinate or a motion-corrected/registered-frame coordinate.

Fix direction:

- Persist compact motion metadata for concat artifacts: displacement, temporal bins, spatial bins, direction, interpolation method, and member-boundary alignment.
- Add an explicit coordinate-frame/provenance flag to spatial exports and unit-location displays.
- Add a hermetic concat test where fake `correct_motion` returns same-shape altered traces and motion metadata, then assert both the corrected traces and motion/frame metadata round trip.

### 5. Medium: UnitMatch chronological drift order is not part of the selection identity

Evidence:

- `UnitMatchSelection` identity stores session group, matcher params, and `curation_set_hash`: [src/spyglass/spikesorting/v2/unit_matching.py:197](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:197), [src/spyglass/spikesorting/v2/unit_matching.py:209](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:209).
- `curation_set_hash` folds only `(member_index, sorting_id, curation_id)`: [src/spyglass/spikesorting/v2/_matcher_graph.py:37](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_matcher_graph.py:37), [src/spyglass/spikesorting/v2/_matcher_graph.py:64](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_matcher_graph.py:64).
- `UnitMatch.make_fetch` derives `recording_date` from live `Session.session_start_time`, then `_extract_and_match` feeds UnitMatch in chronological order because UnitMatch aligns each session to the previous one: [src/spyglass/spikesorting/v2/unit_matching.py:606](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:606), [src/spyglass/spikesorting/v2/unit_matching.py:775](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:775), [src/spyglass/spikesorting/v2/_matcher_graph.py:84](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_matcher_graph.py:84).

Impact: the code correctly freezes `recording_date` between fetch and compute for a single populate attempt, but an edit to `Session.session_start_time` before population can change UnitMatch feed order under the same `unitmatch_id`. Because drift correction is sequential, that can change the scientific output without changing selection identity.

Fix direction:

- Include frozen `recording_date` / chronological order in `UnitMatchSelection` identity or in a persisted member-plan snapshot.
- Alternatively, make `Session.session_start_time` edits invalidate or reject existing UnitMatch selections for that group.
- Add a test that changing session dates after `UnitMatchSelection.insert_selection` either mints a different identity or raises before matching.

### 6. Medium: DriftEstimate is keyed only by Recording despite storing an algorithm-dependent motion estimate

Evidence:

- `DriftEstimate` primary key is only `Recording`; `motion_preset` is a secondary attribute: [src/spyglass/spikesorting/v2/recording.py:2184](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2184).
- The preset comes from a module default, not a parameter lookup: [src/spyglass/spikesorting/v2/recording.py:2200](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2200), [src/spyglass/spikesorting/v2/recording.py:2202](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2202).
- Tests assert there is exactly one drift row per recording and that the stored preset equals `_DEFAULT_PRESET`: [tests/spikesorting/v2/test_drift_estimate.py:72](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_drift_estimate.py:72), [tests/spikesorting/v2/test_drift_estimate.py:87](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_drift_estimate.py:87).

Impact: changing the default preset, comparing estimators, or adding estimator kwargs cannot produce separate rows for the same recording. Existing and new rows can silently mix drift metrics computed under different algorithms, which changes the meaning of `max_abs_displacement_um`.

Fix direction:

- Introduce `DriftEstimateParameters` or make preset/params part of the primary key.
- At minimum, fail clearly when an existing row's `motion_preset` differs from the requested/default preset.
- Add tests that two presets for the same recording either produce distinct rows or a deterministic error.

### 7. Medium-low: spatial semantics are under-documented in UnitMatch and unit-location APIs

Evidence:

- UnitMatch docs state the validated polymer-probe path but do not state the hard same-probe/same-channel-frame requirement or that cross-probe spatial registration is out of scope: [docs/src/Features/SpikeSortingV2.md:896](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:896), [docs/src/Features/SpikeSortingV2.md:946](/Users/edeno/Documents/GitHub/spyglass/docs/src/Features/SpikeSortingV2.md:946).
- The backend does reject mismatched same-shape channel positions, but the test only covers different channel counts: [src/spyglass/spikesorting/v2/_unitmatch_backend.py:247](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_unitmatch_backend.py:247), [tests/spikesorting/v2/test_unitmatch_backend.py:198](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_unitmatch_backend.py:198), [tests/spikesorting/v2/test_unitmatch_backend.py:211](/Users/edeno/Documents/GitHub/spyglass/tests/spikesorting/v2/test_unitmatch_backend.py:211).
- `MatchPair.drift_estimate_um` defaults to `0.0` because UnitMatch has no per-pair drift source, which is documented in code but easy to read as measured "no drift" in outputs: [src/spyglass/spikesorting/v2/matcher_protocol.py:64](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/matcher_protocol.py:64), [src/spyglass/spikesorting/v2/unit_matching.py:459](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:459).

Impact: users may compare `unit_locations` across sessions as biological positions, assume UnitMatch performs cross-probe registration, or interpret pair-level `drift_estimate_um=0.0` as a measured zero. These are mostly documentation/test-contract gaps rather than immediate runtime bugs.

Fix direction:

- Document `unit_locations` as per-sort analyzer coordinates in the current recording/probe frame, not cross-session registered biological coordinates.
- Add a docs caveat: UnitMatch requires same probe, same channel order, and same channel positions; it does not solve cross-probe registration.
- Prefer `NULL` for unavailable pair drift if schema compatibility allows it, or document the `0.0` placeholder prominently in the user-facing UnitMatch section.
- Add a same-shape shifted-position backend test.

## Positives

- Same-day `auto` motion correction is explicitly resolved to `rigid_fast`, while multi-day `auto` is rejected instead of silently choosing a cross-day preset: [src/spyglass/spikesorting/v2/_concat_recording.py:93](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:93), [src/spyglass/spikesorting/v2/_concat_recording.py:136](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:136).
- The concat materializer guards the sample-count invariant before inserting `MemberBoundary`, which protects split back-mapping from a future motion-correction resampling change: [src/spyglass/spikesorting/v2/session_group.py:851](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/session_group.py:851).
- Baseline concat compatibility now has clear channel-id and numeric geometry checks before SI concatenation: [src/spyglass/spikesorting/v2/_concat_recording.py:264](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:264), [src/spyglass/spikesorting/v2/_concat_recording.py:279](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/_concat_recording.py:279).
- UnitMatch pins explicit curations, re-validates ownership at populate time, and feeds sessions in chronological order for drift correction: [src/spyglass/spikesorting/v2/unit_matching.py:278](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:278), [src/spyglass/spikesorting/v2/unit_matching.py:540](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:540), [src/spyglass/spikesorting/v2/unit_matching.py:775](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/unit_matching.py:775).
- `DriftEstimate` is correctly read-only with respect to upstream `Recording`, stores enough information to rehydrate a SpikeInterface `Motion`, and refuses non-finite summaries: [src/spyglass/spikesorting/v2/recording.py:2061](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2061), [src/spyglass/spikesorting/v2/recording.py:2222](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2222), [src/spyglass/spikesorting/v2/recording.py:2225](/Users/edeno/Documents/GitHub/spyglass/src/spyglass/spikesorting/v2/recording.py:2225).

## Suggested triage

- Treat findings 1 and 2 as correctness fixes before broader chronic/UnitMatch rollout.
- Findings 3 and 5 are provenance/identity hardening; they are especially important if `SessionGroup` rows can be edited outside helper APIs.
- Finding 4 is an MVP boundary decision: acceptable only if docs clearly say applied concat motion is not auditable beyond the corrected trace content hash.
- Finding 6 is low-cost schema hygiene if drift QC is expected to support more than one estimator.
