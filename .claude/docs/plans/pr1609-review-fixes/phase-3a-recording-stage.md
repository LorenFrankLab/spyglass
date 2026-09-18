# Phase 3a — Recording stage: filter before restriction; normalize and persist geometry

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#filter-before-restriction)

**Inputs to read first:**

- `src/spyglass/spikesorting/v2/recording.py:2101-2137` — the make_compute pipeline to reorder.
- `src/spyglass/spikesorting/v2/_recording_restriction.py:413-491` (`restrict_recording_times`, unchanged) and `:494-624` (`restrict_recording`, loses its channel block).
- `src/spyglass/spikesorting/v2/_recording_preprocessing.py:31-250` — the function to split; `:252` `filtering_description`.
- `src/spyglass/spikesorting/v2/_recording_geometry.py:166-252` — the tetrode repair (unchanged; now followed by an assertion).
- `src/spyglass/spikesorting/v2/_recording_nwb.py:184-360` — `write_nwb_artifact` (series write 320-347, content hash 350-358).
- `src/spyglass/spikesorting/v2/_sorting_analyzer.py:844-863` — projection warning; keep the warning, add the uniqueness assertion.
- `src/spyglass/spikesorting/v2/_pipeline_preflight.py` — grep `rel_x` for the existing electrode-geometry check.
- SpikeInterface 0.104.3 `core/baserecordingsnippets.py:240-320` — `get_probe` → `create_dummy_probe_from_locations` collapses 3D locations to x-y (the failure to pre-empt); `set_channel_locations` / `get_channel_locations(axes=...)` semantics.
- `tests/spikesorting/v2/single_session/test_recording.py:136-220` — the tetrode-geometry test that cannot currently detect the dropped repair (fixture already carries rel_x/rel_y = ±6.25).
- `tests/_data/raw/minirec20230622.nwb` — real Frank-lab file with x-z geometry (rel_x = ±6.25, rel_y = 0, rel_z = ±6.25); downloaded by the CI legacy job (`test-conda.yml:585-600`) and present locally.

**Designs referenced:** [designs.md#filter-before-restriction](designs.md#filter-before-restriction), [designs.md#geometry-normalization](designs.md#geometry-normalization).

## Tasks

- **Baseline capture before any edit** (do not skip): on the `mearec_polymer_smoke` fixture and on a synthetic recording with three disjoint intervals separated by gaps, record `timestamps_override` (as an array), `n_selected_intervals`, `recording.get_channel_ids()`, and the finished preprocessed traces to `.npz` files in the scratchpad. Also record the *continuously filtered reference*: the channel-sliced full recording passed through the same bandpass, sampled at the same frame ranges. The reference, not the old-order output, is the correctness target for traces.
- **Extract `select_sort_group_channels`** from the end of `restrict_recording` per the design; slim `restrict_recording` to interval intersection + `restrict_recording_times`; drop its now-unused channel parameters and docstring paragraphs.
- **`select_distinct_plane` / `normalize_channel_locations` / `assert_unique_contact_positions`** in `_recording_geometry.py` per the design. Normalization runs on the channel-sliced recording (so removed/excluded channels do not participate) and sets 2D channel locations without constructing a probe.
- **Split `apply_pre_motion_preprocessing`** into `apply_temporal_preprocessing` and `apply_spatial_preprocessing` per the design; delete the original (single caller at `recording.py:2124`). Update the module docstring's step list (lines 1-27) and `filtering_description`'s consumer contract if it names the old function.
- **Reorder `Recording.make_compute`** (`recording.py:2101-2137`) to read → select channels → normalize locations → temporal → restrict → spatial → tetrode repair → `assert_unique_contact_positions`, merging `applied_steps`. Update the docstring above it (2080-2090 mention the rebuild path sharing this code; confirm the rebuild path is the same function and needs no separate change).
- **Persist normalized geometry** in `write_nwb_artifact`: write `recording.get_channel_locations()` (2D) into `rel_x`/`rel_y` and `rel_z = 0` for the series' electrode rows, after `io.write(nwbfile)` and before `recording_content_fingerprint`; assert lengths match, read back, assert equality. No `get_probe()` call anywhere in this path.
- **Uniqueness assertion at analyzer build** in `_sorting_analyzer.py:844-863`: keep the non-planar warning; after projection assert every contact position is unique (same helper), raising a message that names the recording and points at `Probe.Electrode`.
- **Preflight effective-geometry check**: in the existing `Probe.Electrode` geometry check in `_pipeline_preflight.py`, compute the effective 2D geometry (`select_distinct_plane` on the group's rows, then the tetrode-repair predicate) and report an error when any two contacts coincide, naming file, sort group, and positions.
- **Comparison after the edits**: re-run the baseline script; assert `timestamps_override`, `n_selected_intervals`, and channel ids are exactly equal; traces agree with the continuously filtered reference within the design's tolerance (`max_abs_diff <= 1e-3 * rms(reference)`, RMS within 0.1%), with the reference requested per interval at the same frame boundaries. SpikeInterface's 5 ms filter margin makes exact equality impossible (observed ~1.5e-4 between sliced and whole-recording requests).
- **CHANGELOG** (`[Unreleased]` → Spike sorting v2): recordings are bandpass-filtered on the continuous source before interval restriction (existing v2 recording artifacts must be recreated per the preproduction upgrade sequence); electrode geometry is normalized to a distinct 2D plane and persisted in the recording artifact; Recording.make and preflight reject coincident contacts. Update `docs/src/Features/SpikeSortingV2.md` where it describes the preprocessing order (grep "bandpass" / "restrict").

## Deliberately not in this phase

- Whitening, noise levels, artifact masking, observed intervals — phase 3b.
- Raising `min_segment_length` in the shipped recipes (not a fix; see design).
- Concatenated recordings: `ConcatenatedRecording` builds from member artifacts that are already filtered and normalized; no change.
- Interpolated bad channels entering the global median (appendix suggestion).
- Specific-reference subtraction with unequal channel offsets and no bandpass (owner's probe: `[-20, -10]` µV where physical-unit subtraction gives `[-220, -110]` µV, then `set_channel_offsets(0.0)` at `_recording_preprocessing.py:218` hides the heterogeneity from the writer's guard). Tracked in the overview follow-ups; the strict-xfail oracle test in the validation slice keeps the gap visible and is computed from raw counts so it cannot share the defect.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/v2/test_recording_services.py::test_restriction_output_unchanged_by_reorder` | `timestamps_override`, `n_selected_intervals`, channel ids equal the captured baseline for the 3-interval synthetic and the smoke fixture |
| `...::test_restricted_traces_match_continuously_filtered_reference` | for the 3-interval synthetic, restricted samples agree with the continuously filtered recording requested at the same interval boundaries: `max_abs_diff <= 1e-3 * rms`, RMS within 0.1%, including the first/last samples of each interval; the old restrict-then-filter order fails this by orders of magnitude |
| `...::test_sliver_after_highpass_matches_continuous_filter` | for a 1.5 ms interval inside a longer recording with 0.3 Hz / 200 µV drift and a 600 Hz high-pass: RMS of the restricted sliver within 1% of the continuously filtered samples; max abs error < 1 µV (currently ~96 µV) |
| `tests/spikesorting/v2/test_recording_geometry.py::test_select_distinct_plane_prefers_xy_then_xz` | x-z tetrode → `"xz"` with 4 unique positions; planar x-y → `"xy"`; all-zero → `None` |
| `...::test_normalize_on_channel_subset` | a 3-channel subset of an x-z tetrode normalizes on those three channels only |
| `...::test_assert_unique_contact_positions_requires_all_distinct` | 4 contacts with 3 distinct positions raise (2 distinct is not enough) |
| `tests/spikesorting/v2/single_session/test_recording.py::test_tetrode_geometry_persists_across_reload` | on a fixture whose raw electrodes have rel_x/rel_y all zero (write one from `chronic_minirec_a.nwb` with zeroed rel_* columns), `Recording.get_recording(...).get_channel_locations()` is the 12.5 µm square (absolute positions) and `get_probe()` succeeds |
| `...::test_xz_geometry_reloads_with_four_positions` (`pytest.mark.slow`, uses `minirec20230622.nwb`) | after `Recording.populate` on one tetrode group of the real file, `get_recording(...).get_probe()` succeeds with 4 unique 2D positions and `create_sorting_analyzer` does not raise |
| `tests/spikesorting/v2/test_sorting_analyzer.py::test_degenerate_geometry_raises_at_build` | a recording whose persisted locations coincide raises `ValueError` mentioning `Probe.Electrode` before any extension is computed |
| `tests/spikesorting/v2/test_preflight.py::test_preflight_rejects_coincident_contacts_after_repair` | preflight error names the sort group when the effective geometry (after the tetrode-repair predicate) has coincident contacts; the real x-z tetrode passes |
| `tests/spikesorting/v2/test_recording_nwb.py::test_persist_geometry_writes_series_rows_only` | only the rows referenced by the series' electrode region change; other rows keep raw values; content hash differs when geometry differs |
| `...::test_recording_semantic_round_trip` (contract test) | on a fixture built to discriminate (x-z geometry, non-contiguous and reordered electrode ids, a UNIFORM non-unit gain such as 0.195 µV/count with non-zero uniform offset, a two-interval selection): after `Recording.populate` + `get_recording` AND after `_rebuild_nwb_artifact`, timestamps, channel-id order, 2D channel locations, and the observed/selection spans are EXACTLY equal, and traces in µV (`return_in_uV=True`) agree with a reference read from the pre-write lazy recording at the writer's own chunk boundaries (`SpikeInterfaceRecordingDataChunkIterator` with the same `buffer_gb`, i.e. the same `get_traces(start, end)` requests the writer issued) to `1e-6` µV; a whole-recording single request is NOT a valid reference because the lazy filter's edge values legitimately depend on request boundaries (probe: 0.00239 µV with no serialization error). The fixture is long enough for at least three write chunks so the boundary distinction is exercised — "loads successfully" is not the assertion |
| `...::test_heterogeneous_gains_rejected_by_writer` | a fixture with unequal per-channel gains makes `write_nwb_artifact` raise via `resolve_conversion_and_offset` (the writer supports only uniform gain/offset; bandpass and reference preserve the inequality) with a message naming the channels |
| `...::test_specific_reference_physical_units_oracle` (`xfail(strict=True)`, tracked follow-up) | with unequal channel OFFSETS, bandpass OFF, and `reference_mode="specific"`: reloaded µV traces equal an independent oracle computed from RAW counts as `(raw_i·gain_i + off_i) − (raw_ref·gain_ref + off_ref)` — not the already-preprocessed in-memory recording, which shares the bug (the probe found `[-20, -10]` µV where the oracle gives `[-220, -110]` µV) |

## Fixtures

- Synthetic: `si.generate_recording` (4 ch, 30 kHz, 120 s) with an added 0.3 Hz sinusoid (200 µV) and interval lists built in-test; 3D location arrays for the plane-selection tests.
- Real: `tests/_data/raw/minirec20230622.nwb` (fetched by CI's legacy job; add the same download step to the `pytest-v2` job for the one slow test, or gate it with `SPYGLASS_V2_REQUIRE_FIXTURES` on the `schedule` tier and document that it runs nightly, not per-PR).
- Derived: a copy of `chronic_minirec_a.nwb` with zeroed `rel_x/rel_y/rel_z` written by an h5py helper in `tests/spikesorting/v2/_ingest_helpers.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (`apply_pre_motion_preprocessing`; the channel block inside `restrict_recording`; any `get_probe()`-gated persistence).
- User-facing documentation listed as tasks is updated, not deferred.
