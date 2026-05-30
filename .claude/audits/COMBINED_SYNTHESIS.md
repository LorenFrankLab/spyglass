# Spike Sorting v1 → v2 Parity Audit — Combined Synthesis

**Generated 2026-05-30 from two automated multi-agent audits.**


- Main audit (6 modules + 4 cross-cutting): 130 confirmed, 4 refuted, 70 untested branches.
- Sorting-only follow-up audit (sorting.py died in main): 53 confirmed, 1 refuted.
- **Total confirmed: 183 findings.**

## Rollups

- **By severity:** critical 0 | high 25 | medium 71 | low 87
- **By classification:** intentional+justified 96 | intentional+unjustified 23 | unintentional drift 7 | untested branch 20 | new in v2 29 | missing in v2 5 | uncertain 3

## v2 stub gaps (v1 features not yet ported)

- MetricCuration (v1/metric_curation.py 710 lines: WaveformParameters, MetricParameters, MetricCurationParameters, MetricCurationSelection, MetricCuration with waveform cache + quality-metric compute + auto-label/merge) -- v2/metric_curation.py is a 3-line stub at line 1-3.
- FigURLCuration (v1/figurl_curation.py 322 lines: kachery-cloud / sortingview-based interactive curation URL generation) -- v2/figpack_curation.py is a 3-line stub (renamed Fig -> figpack).
- BurstPair / BurstPairParams / BurstPairSelection (v1/burst_curation.py 397 lines: burst-parent vs burst-child detection via cross-correlogram asymmetry, ISI violations, waveform similarity, with diagnostic plots) -- no v2 module and no stub; entire feature absent.
- RecordingRecompute / RecordingRecomputeVersions / RecordingRecomputeSelection (v1/recompute.py 981 lines: tracks pynwb / spikeinterface dependency versions, replays preprocessing, hash-diffs against original AnalysisNwbfile) -- no v2 equivalent; v2 keeps only the inline NwbfileHasher digest column.
- Cross-session unit matching (v2/unit_matching.py + v2/matcher_protocol.py 3-line stubs) -- v1 also has no equivalent, but v2 reserves this as a planned feature with stubs.
- ConcatenatedRecording materializer (v2/session_group.py: ConcatenatedRecording.make() and ConcatenatedRecordingSelection.insert_selection raise NotImplementedError) -- schema is final-shape but consumer logic is not implemented. SortingSelection's ConcatenatedRecordingSource path is correspondingly gated with NotImplementedError in Sorting.make.
- SessionGroup helpers (SessionGroup.create_group, SessionGroup.is_multi_day) raise NotImplementedError in v2/session_group.py:69-96.
- Mountainsort4-on-modern-SpikeInterface guard: v1 ArtifactDetection and v1 MetricCuration call _require_legacy_si_environment (v1/artifact.py:152, v1/metric_curation.py:17) -- v2 has no equivalent legacy-SI gate because it targets the modern SortingAnalyzer API.

## sorting (53 findings)

### sorting#1  [HIGH | INT-JUST] Clusterless schema `noise_levels` default changed from `np.asarray([1.0])` to `None`; default ROW still passes `[1.0]` for v1 parity but the implicit default has flipped
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:177`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:159; runtime at src/spyglass/spikesorting/v2/sorting.py:1146-1176`
- v1 behavior: v1's `default_clusterless` row hard-codes `'noise_levels': np.asarray([1.0])` (numpy array). v1 docstring at sorting.py:176 notes: 'noise levels needs to be 1.0 so the units are in uV and not MAD'. v1 has no other clusterless row, and any new user row defaulted to whatever the dict literal carried.
- v2 behavior: v2 `ClusterlessThresholderSchema.noise_levels: list[float] | None = Field(default=None)`. So a user calling `ClusterlessThresholderSchema()` with no args gets `noise_levels=None`, which the runtime at sorting.py:1146-1150 interprets as 'omit, fall through to SI per-channel MAD'. The v2 production default row at sorting.py:175 explicitly passes `{'noise_levels': [1.0]}` to preserve v1's raw-microvolt semantics for that specific row, but the schema-level default has flipped from raw-uV semantics to MAD-multiplier semantics. Type changed from numpy array to Python list.
- documented rationale: Schema default was flipped to None to support smoke / synthetic-fixture rows that want MAD-multiplier semantics; the shipped 'default' row preserves v1 behavior by explicitly opting back into [1.0]. This is the correct fix for the 1400x divergence: the dangerous behavior is silent injection of a per-row noise_levels value that doesn't match the recording's channel count. Schema_version=3 encodes the change.
- verifier reasoning: Verified directly against the source:

1. v1 sorting.py:177 — `"noise_levels": np.asarray([1.0])` is hard-coded in default_clusterless, with the comment at line 176: "noise levels needs to be 1.0 so the units are in uV and not MAD". Confirmed.

2. v2 _params/sorter.py:159 — `noise_levels: list[float] | None = Field(default=None)` confirmed. Type flipped from numpy array to Python list-or-None. The class docstring (lines 134-148) extensively documents the rationale: schema default is None (SI per-channel MAD), callers wanting raw-uV semantics must opt in with `[1.0]`. schema_version=3 confirmed

### sorting#2  [HIGH | UNTESTED] Mode A cleanup (analyzer_folder removed when _write_units_nwb fails) is implemented but untested
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:461-496`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:607-626`
- v1 behavior: v1 has no analyzer concept and no Mode A. _save_sorting_results errors propagate up; the only artifact is the units NWB itself which is unlinked by AnalysisNwbfile's own cleanup if make_insert fails.
- v2 behavior: make_compute wraps _write_units_nwb in try/except (sorting.py:607-626) and rmtrees analyzer_folder on failure. This branch is critical because the analyzer folder is 5-50 GB; a leak per failed populate is catastrophic. The Mode B equivalent (failure during make_insert) IS tested by test_sorting_make_rollback_cleans_units_nwb at line 3537+, but Mode A is not.
- documented rationale: Same exact disk-leak risk as Mode B which IS tested; the asymmetry suggests Mode A was added later (audit-trail consistent with the file's structure) without backfilling its test.
- verifier reasoning: CONFIRMED. Verified by reading code and tests directly.

Evidence:
1. v2 sorting.py:607-626 contains the Mode A try/except: `_build_analyzer` runs first (line 599-605), then `_write_units_nwb` is wrapped in try/except (607-614); on Exception the analyzer_folder is rmtree'd (618-620) with a nested defensive logger (621-625). The comment block at 615-617 explicitly names this "Mode A cleanup" and says it prevents "a half-built scratch dir" from leaking.

2. v1 sorting.py:461-496 (`_save_sorting_results`) has no analyzer concept and no equivalent cleanup. v2 Mode A is therefore a v2-only safety b

### sorting#3  [HIGH | DRIFT] _apply_artifact_mask: empty valid_times silently masks the entire recording instead of failing
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:359, src/spyglass/spikesorting/v1/recording.py:747`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1014-1027`
- v1 behavior: v1 calls _consolidate_intervals(artifact_removed_intervals, timestamps) at sorting.py:359 which would crash on empty input at recording.py:747 (start_indices[0] on empty array raises IndexError). v1 fails loudly.
- v2 behavior: _apply_artifact_mask at sorting.py:993-1055 iterates 'for vs, ve in valid_times'; an empty valid_times skips the loop. cursor=timestamps[0] then triggers the trailing 'if cursor < timestamps[-1]' branch (sorting.py:1023), generating a frame_range covering [0, n_samples). The entire recording is then masked to zeros and the sort runs on all-zero data -- silently producing a degenerate (zero-unit or near-zero-unit) sort.
- documented rationale: v2's intent (per docstring) is to compute the complement of valid_times. Empty valid_times means 'no valid windows' which logically IS the whole recording as complement -- so the code's behavior is internally consistent. But the scientific intent of 'entire recording is artifact' is 'do not sort; emit zero units with a clear signal' -- not 'silently sort on zeros'. Recommend an early-return guard: if valid_times.size == 0, raise a clear error like NoValidArtifactWindowError. v2 has no such guard.
- verifier reasoning: Verified by reading the cited code directly.

V2 behavior (sorting.py:993-1055) traced with `valid_times.shape == (0, 2)`:
- Line 1015: `cursor = timestamps[0]`
- Line 1016: `for vs, ve in valid_times:` -- empty iterable, loop body never executes
- Line 1023: `if cursor < timestamps[-1]:` -- TRUE
- Lines 1024-1027: `start = searchsorted(timestamps, timestamps[0]) = 0`, `end = len(timestamps)`, appends `(0, len(timestamps))` to `frame_ranges`
- Lines 1029-1055: builds `artifact_frames = np.arange(0, n_samples)` and passes to `sip.remove_artifacts` which zeros every sample.

V1 behavior (v1/reco

### sorting#4  [HIGH | UNTESTED] _rebuild_analyzer_folder has no test coverage including the artifact_id sub-branch
- **v1**: `n/a (no v1 equivalent)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:855-922`
- v1 behavior: v1 has no concept of a rebuildable analyzer folder; on missing artifact files the user re-populates the table.
- v2 behavior: get_analyzer (sorting.py:821-853) calls _rebuild_analyzer_folder when the folder is missing. _rebuild_analyzer_folder (sorting.py:855-922) re-reads the canonical sorting from the units NWB, re-applies the artifact mask if artifact_id is set, and rebuilds the analyzer. The rebuild path has its own DB I/O (sel_row fetch1, resolve_source, IntervalList fetch1 in the artifact branch) and its own cleanup exception handler.
- documented rationale: The rebuild path is invoked silently inside get_analyzer; users in production who run a re-populate after manually deleting the scratch folder will hit it. Untested means the behavior is unverified -- including whether get_sorting(key) on a zero-unit sort would actually take this path (it would: _rebuild_analyzer_folder calls get_sorting which returns empty NumpySorting, then _build_analyzer short-circuits, then load_sorting_analyzer at sorting.py:853 would fail on the still-missing folder). The interaction with zero-unit sorts is the most concerning untested case.
- verifier reasoning: Verified the citations directly. sorting.py:821-853 shows get_analyzer raises ZeroUnitAnalyzerError early for n_units==0 (so the zero-unit-takes-rebuild concern in the rationale is partly off), then checks `folder.exists()` and calls `_rebuild_analyzer_folder` if missing. sorting.py:855-922 implements the rebuild: fetch sel_row, resolve_source (raises NotImplementedError for concat at line 871-874), Recording().get_recording(), then if sel_row['artifact_id'] is not None it does extra DB I/O (RecordingSelection.fetch1, IntervalList.fetch1) and applies the artifact mask, then calls get_sorting +

### sorting#5  [HIGH | UNTESTED] noise_levels=None branch in _run_clusterless_thresholder pins random_seed but is not directly tested
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:177`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1146-1167`
- v1 behavior: v1's clusterless_thresholder default ALWAYS shipped noise_levels=[1.0] (sorting.py:177); there was no None branch.
- v2 behavior: v2's _run_clusterless_thresholder at sorting.py:1146-1167 has a noise_levels-is-None branch that (a) pops noise_levels, (b) sets random_slices_kwargs={'seed': random_seed} so SI's get_noise_levels MAD estimate is deterministic across runs. This is the v2-introduced 'auto-MAD' path. Per the inline comment, the path was added to undo the v1 1400x clusterless bug -- it MUST be deterministic, but no direct unit test exercises this branch.
- documented rationale: Given the prior 1400x divergence bug in the noise_levels area, this branch is high-risk and should be tested. Direct unit test: call _run_clusterless_thresholder twice with noise_levels=None + same random_seed, assert identical sorted spike samples. The integration MS4 determinism test exists but does not isolate this branch.
- verifier reasoning: Verified directly against source: v1 sorting.py:177 always ships noise_levels=np.asarray([1.0]); v2 sorting.py:1146-1167 introduces a noise_levels-is-None branch that pops the key and sets random_slices_kwargs={'seed': _random_seed} to pin SI's get_noise_levels MAD seed, with an extensive inline comment citing SI PR #3359 (seed=0 -> seed=None default change) and naming Spyglass as the explicit seeder. The only direct unit test that calls Sorting._run_clusterless_thresholder is test_single_session_pipeline.py:4042-4079 (test_clusterless_detect_peaks_strips_random_seed), which passes noise_level

### sorting#6  [HIGH | INT-JUST] v2 pins SI random seeds for whitening and noise-level estimation; v1 does not (determinism improvement)
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:428-430`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1260-1284, 1146-1167`
- v1 behavior: v1 calls sip.whiten(recording, dtype=np.float64) without seed (line 429). clusterless_thresholder receives noise_levels=[1.0] as a hard parameter, so noise-level estimation seed is moot for v1's clusterless. For other sorters that internally estimate covariance, v1 inherits SI's nondeterministic default.
- v2 behavior: _run_si_sorter at 1281-1284 calls sip.whiten(..., seed=_random_seed) with _random_seed defaulting to 0; _run_clusterless_thresholder at 1163-1167 pins detect_peaks' random_slices_kwargs={'seed': _random_seed} when noise_levels is None. The seed is exposed via SorterParameters.job_kwargs['random_seed']; _resolved_job_kwargs merges from dj.config + per-row override.
- documented rationale: Explicitly addresses the prior-bug context (noise_levels=[1.0] causing 1400x divergence): the seed pin makes the noise-level estimation deterministic too when noise_levels is None, removing another source of run-to-run drift.
- verifier reasoning: Verified by direct read. v1/sorting.py:429 calls sip.whiten(recording, dtype=np.float64) with no seed. v2/sorting.py:1281-1284 calls sip.whiten(recording, dtype=_np.float64, seed=_random_seed) with _random_seed sourced from job_kwargs.get('random_seed', 0). v2/sorting.py:1146-1167 (_run_clusterless_thresholder) pins random_slices_kwargs={'seed': _random_seed} on the noise_levels=None path so SI's get_noise_levels MAD-from-random-chunks call is deterministic. Inline comments at v2:1263-1284 cite SI PR #3359 (default flipped from seed=0 to seed=None, 2024-10-25), state the 'seed must be explicit

### sorting#7  [HIGH | INT-JUST] v2 uniqueness invariant: `(recording_id, sorter, sorter_params_name, artifact_id-or-NULL)` enforced via DuplicateSelectionError. v1 dedupe was permissive on the full key shape with no error
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:223-229`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:285-319`
- v1 behavior: `cls & key` restricts on whatever columns are in `key`. If the caller omits e.g. `interval_list_name`, v1 happily aliases multiple distinct sorts together and returns them all as a list. No error raised on multiplicity. Silent dedup, never fail.
- v2 behavior: v2 explicitly restricts on `(sorter, sorter_params_name)` + `recording_id` + (`artifact_id` value OR `IS NULL`) — see v2/sorting.py:285-305. If more than one master matches that logical identity, `DuplicateSelectionError` is raised (v2/sorting.py:314-319). The NULL handling for artifact_id is explicit: `None` is treated as a distinct identity value via `"artifact_id IS NULL"`, NOT a wildcard.
- documented rationale: v1's permissive dedup was a latent integrity hole that could quietly alias requests; v2 makes uniqueness explicit and audited. The artifact_id=None test fixes a prior bug where None was treated as wildcard.
- verifier reasoning: Verified all four claim components against source:

(1) v1 dedup at src/spyglass/spikesorting/v1/sorting.py:223-229 is exactly `query = cls & key; if query: logger.info(...); return query.fetch(as_dict=True)`. It restricts on whatever the caller passes in `key` and silently returns the matching rows as a list — no uniqueness enforcement, no error on multiplicity, no NULL-vs-value distinction.

(2) v2 at src/spyglass/spikesorting/v2/sorting.py:285-319 builds an explicit master_restriction = {sorter, sorter_params_name}, joins the `RecordingSource` part-table on `recording_id`, and either restri

### sorting#8  [MEDIUM | INT-UNJ] Franklab MS4 preset row names renamed (uppercase 'KHz' to lowercase 'kHz' + '_ms4' suffix); breaks v1 lookups by name
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:158, 163`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:109, 125`
- v1 behavior: v1 ships `'franklab_tetrode_hippocampus_30KHz'` and `'franklab_probe_ctx_30KHz'` (capital K) and these are the canonical Frank-lab MS4 preset names referenced from notebooks, scripts and pipelines.
- v2 behavior: v2 ships `'franklab_tetrode_hippocampus_30kHz_ms4'` and `'franklab_probe_ctx_30kHz_ms4'` (lowercase k, plus a `_ms4` suffix to disambiguate from MS5).
- documented rationale: The `_ms4`/`_ms5` suffix disambiguates between sorter versions for the same preset; the lowercase-k normalization is stylistic. Both are intentional but rename without a back-compat alias means hand-maintained v1 references silently miss.
- verifier reasoning: Verified directly. v1/sorting.py:158 and :163 ship `franklab_tetrode_hippocampus_30KHz` and `franklab_probe_ctx_30KHz` (capital K, no sorter suffix). v2/sorting.py:109 and :125 ship `franklab_tetrode_hippocampus_30kHz_ms4` and `franklab_probe_ctx_30kHz_ms4` (lowercase k, with `_ms4` suffix). A grep of the codebase confirms no alias/back-compat row exists for the v1 capital-K names in v2; any v1 caller doing a name lookup against v2 (e.g., `(SorterParameters & {"sorter_params_name": "franklab_tetrode_hippocampus_30KHz"})`) gets an empty restriction. The `_ms4` suffix is justified by the coexist

### sorting#9  [MEDIUM | INT-UNJ] MS4 schema `adjacency_radius` constraint `ge=0.0` rejects SI's documented sentinel value of -1 ('use all channels')
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:147`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:57`
- v1 behavior: v1 stores `adjacency_radius: 100` (int) in `mountain_default`. SI's `get_default_sorter_params('mountainsort4')` returns `adjacency_radius: -1` (sentinel: use all channels in graph). v1 stored either value verbatim without validation; a user could insert any int.
- v2 behavior: v2 schema declares `adjacency_radius: float = Field(default=100.0, ge=0.0)`. A user trying to insert SI's `-1` sentinel ('use all channels') hits `ValidationError: Input should be greater than or equal to 0`. Verified by direct call: `MountainSort4Schema(adjacency_radius=-1)` raises `greater_than_equal` error.
- documented rationale: Validation tightening likely intended to catch nonsensical negative radii, but in doing so the schema implicitly bans SI's documented 'use all channels' encoding. A v1 user with `adjacency_radius=-1` in their params dict cannot port to v2.
- verifier reasoning: All four claim components verified directly:

1. v1 file:line confirmed (src/spyglass/spikesorting/v1/sorting.py:147): `mountain_default` stores `"adjacency_radius": 100` verbatim in a `blob` column with no Pydantic validation — any int (including -1) could be inserted.

2. v2 file:line confirmed (src/spyglass/spikesorting/v2/_params/sorter.py:57): `adjacency_radius: float = Field(default=100.0, ge=0.0)`.

3. Pydantic rejection reproduced. Constructing `MountainSort4Schema(adjacency_radius=-1)` with the exact schema definition from the file raises `ValidationError: Input should be greater than

### sorting#10  [MEDIUM | INT-JUST] MS5 v2 schema strips ~13 fields that v1's `sis.get_default_sorter_params('mountainsort5')` row carries
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:184-188 (auto-inserts default row from SI)`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:68-88`
- v1 behavior: v1's MS5 default row is `sis.get_default_sorter_params('mountainsort5')`, which returns 20 fields: `detect_sign, detect_threshold, detect_time_radius_msec, filter, freq_max, freq_min, npca_per_channel, npca_per_subdivision, scheme, scheme1_detect_channel_radius, scheme2_detect_channel_radius, scheme2_max_num_snippets_per_training_batch, scheme2_phase1_detect_channel_radius, scheme2_training_duration_sec, scheme2_training_recording_sampling_mode, scheme3_block_duration_sec, snippet_T1, snippet_T2, snippet_mask_radius, whiten`. A v1 user can edit any of these via the params blob.
- v2 behavior: v2's MS5 schema has `extra='forbid'` and only 8 fields: `schema_version, scheme, detect_sign, detect_threshold, detect_time_radius_msec, snippet_T1, snippet_T2, scheme2_phase1_detect_channel_radius, scheme2_detect_channel_radius`. So 12 SI MS5 fields (filter, freq_min, freq_max, npca_per_channel, npca_per_subdivision, scheme1_detect_channel_radius, scheme2_max_num_snippets_per_training_batch, scheme2_training_duration_sec, scheme2_training_recording_sampling_mode, scheme3_block_duration_sec, snippet_mask_radius, whiten) are NOT customizable via the schema -- a user trying to insert them hits `extra_forbidden`. The final SI call falls back to SI's runtime defaults for stripped fields.
- documented rationale: MS5 upstream pipeline does bandpass + whiten in the recording stage, so MS5's internal filter/whiten flags are redundant. The 8 stripped scheme/npca/snippet_mask fields fall through to SI's defaults at sort time. A power-user wanting to tune `snippet_mask_radius` from 250 must hack `extra='forbid'`.
- verifier reasoning: All elements of the claim verified directly against code:

1. **v1 location** (`src/spyglass/spikesorting/v1/sorting.py:184-188`): Confirmed v1 auto-inserts a "default" row per available sorter via `sis.get_default_sorter_params(sorter)`. For MS5 this yields 20 fields.

2. **SI MS5 default field list**: Empirically confirmed by running `sis.get_default_sorter_params('mountainsort5')` — returns exactly 20 fields matching the claim's list verbatim: `detect_sign, detect_threshold, detect_time_radius_msec, filter, freq_max, freq_min, npca_per_channel, npca_per_subdivision, scheme, scheme1_detect_c

### sorting#11  [MEDIUM | INT-JUST] Sorting master adds `analyzer_folder` and `n_units` columns; varchar(40) bumped to varchar(72) for object_id; time_of_sort type changes from int (Unix) to datetime
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:233-240`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:422-430`
- v1 behavior: Sorting master columns: `analysis_file_name, object_id: varchar(40), time_of_sort: int`. No analyzer_folder. No n_units.
- v2 behavior: Sorting master columns: `analysis_file_name, object_id: varchar(72), analyzer_folder: varchar(255), n_units: int, time_of_sort: datetime`. The analyzer_folder is the binary SortingAnalyzer folder on disk (sparse=True); n_units is the cached unit count for fast existence checks.
- documented rationale: SortingAnalyzer is the v2 architectural pivot; the analyzer folder is a persistent on-disk artifact. n_units cached so `get_sorting` can short-circuit to an empty NumpySorting without opening the units NWB (which has no spike_times column for zero-unit sorts). varchar(72) accommodates longer NWB object IDs; datetime is more readable than Unix epoch.
- verifier reasoning: All claims verified directly. v1/sorting.py:233-240 shows the v1 master table with `object_id: varchar(40), time_of_sort: int` (Unix). v2/sorting.py:422-430 shows the v2 master with `object_id: varchar(72), analyzer_folder: varchar(255), n_units: int, time_of_sort: datetime`. The class docstring at v2/sorting.py:411-419 explicitly documents the SortingAnalyzer(format="binary_folder", sparse=True) architecture, justifying the analyzer_folder column. The n_units column is used as a zero-unit guard at v2/sorting.py:749-765 in get_sorting, with a comment explaining that NwbSortingExtractor cannot 

### sorting#12  [MEDIUM | INT-JUST] Zero-unit handling: v1 silently writes empty Units NWB and master row; v2 distinguishes graceful (n_units=0 row OK) vs hard-error (ZeroUnitSortError) via require_units flag
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:578-581`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1396-1403, 1534-1538, 1571-1572, 842-848, exceptions.py:72-89`
- v1 behavior: v1 _write_sorting_to_nwb at 578-581: 'if sorting.get_num_units() == 0: nwbf.units = pynwb.misc.Units(name="units", description="Empty units table.")'. master row inserts normally with object_id of the empty Units table. No way for caller to opt into a hard error.
- v2 behavior: v2 _write_units_nwb at 1534-1538 has the equivalent empty-Units fallback (cited as 'v1 has the same guard at v1/sorting.py:578'). _build_analyzer short-circuits at 1396-1403 for zero-unit sortings (SI's estimate_sparsity crashes on np.concatenate([])). _populate_unit_part at 1571-1572 returns early. Sorting master row commits with n_units=0. get_analyzer raises ZeroUnitAnalyzerError at 842-848 (no analyzer folder ever written). pipeline.py:229 raises ZeroUnitSortError when require_units=True is set.
- documented rationale: Explicit improvement: v1 'silently writes empty NWB' is preserved as the default (zero units is a legitimate quiet-shank result), but the new ZeroUnitSortError/ZeroUnitAnalyzerError exceptions let downstream callers distinguish 'expected empty result' from 'pipeline failure'.
- verifier reasoning: Verified all cited locations directly. v1/sorting.py:578-581 writes an empty Units NWB and proceeds to commit the master row with that empty Units object_id (line 600-602) — no opt-in for a hard error. v2/sorting.py:1396-1403 short-circuits _build_analyzer for zero-unit sortings with a docstring explaining the SI estimate_sparsity/np.concatenate([]) crash; 1530-1538 keeps the v1-parity empty-Units fallback (explicitly citing v1/sorting.py:578 in the comment); 1571-1572 returns from _populate_unit_part on zero units; 841-848 raises ZeroUnitAnalyzerError when get_analyzer is called on an n_units

### sorting#13  [MEDIUM | UNTESTED] _run_si_sorter global job_kwargs restore-on-raise path is untested
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:452-455`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1305-1335`
- v1 behavior: v1 does not use set_global_job_kwargs; it splatted sorter_params into sis.run_sorter directly at sorting.py:452-455.
- v2 behavior: _run_si_sorter installs job_kwargs via si.set_global_job_kwargs (sorting.py:1305-1306) inside a try/finally; the finally at 1324-1335 restores the previous global. The success path is tested by test_run_si_sorter_restores_global_job_kwargs at test_single_session_pipeline.py:4001 (mocks run_sorter to return 'DUMMY'). The raise path -- run_sorter throws and the finally still must restore -- is untested.
- documented rationale: Standard try/finally semantics make this 'probably correct', but the audit-task asked specifically about job_kwargs threading; this is a real untested branch. A regression that moved the restore outside the finally would leak global state.
- verifier reasoning: Verified all four cited locations directly. v1 (sorting.py:452-455) does not touch global job kwargs; it splats sorter_params straight into sis.run_sorter. v2 (sorting.py:1304-1335) captures previous_global, installs sj_kwargs gated by `if sj_kwargs`, wraps run_sorter in try at 1322-1323, and restores via reset_global_job_kwargs() + set_global_job_kwargs(**previous_global) inside a finally gated on `if sj_kwargs`. The non-trivial reset-then-reapply ordering is documented at 1326-1333: since SI's set_global_job_kwargs UPDATES rather than REPLACES, keys absent from the default global (chunk_size

### sorting#14  [MEDIUM | INT-JUST] v1 auto-ships a 'default' row for every sis.available_sorters(); v2 only ships 6 curated sorters
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:184-189`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:106-183 (`_DEFAULT_CONTENTS`); src/spyglass/spikesorting/v2/_params/sorter.py:199-230 (`_SORTER_SCHEMAS`/`_get_sorter_schema`)`
- v1 behavior: v1 extends contents with `[sorter, 'default', sis.get_default_sorter_params(sorter)] for sorter in sis.available_sorters()` so the Lookup table contains a 'default' row for every SI-registered sorter (~18 sorters: combinato, hdsort, herdingspikes, ironclust, kilosort, kilosort2, kilosort2_5, kilosort3, klusta, mountainsort4, mountainsort5, pykilosort, spykingcircus, spykingcircus2, tridesclous, tridesclous2, waveclus, waveclus_snippets, yass), populated with SI's canonical default kwargs dict.
- v2 behavior: v2 ships only 7 explicit rows: 2 MS4 presets (tetrode/cortex), 1 MS5 row, 1 KS4 row, 1 SC2 row, 1 TDC2 row, 1 clusterless_thresholder row. There is no fallback insertion loop. A v2 user wanting a kilosort2_5/ironclust/tridesclous default must insert it themselves; `_get_sorter_schema` will return `GenericSorterParamsSchema` (extra='allow') for them.
- documented rationale: v2 deliberately replaced silent ~18-row dynamic-insert with explicit per-sorter Pydantic dispatch + user-inserts-as-needed. The justification is shipping less; cost is breakage for v1 code that used a sorter via the auto-default name like ('kilosort2_5','default').
- verifier reasoning: All citations verified directly against the source. v1/sorting.py:184-189 contains the dynamic `contents.extend([[sorter, "default", sis.get_default_sorter_params(sorter)] for sorter in sis.available_sorters()])` loop, so v1 auto-ships a 'default' row for every SI-registered sorter. v2/sorting.py:106-183 ships exactly 7 explicit `_DEFAULT_CONTENTS` rows across 6 curated sorters (MS4 tetrode + MS4 cortex, MS5, KS4, SC2, TDC2, clusterless_thresholder) with no fallback insertion loop. The title's "6 curated sorters" matches unique sorter names; the body's "7 explicit rows" matches row count (MS4 

### sorting#15  [MEDIUM | DRIFT] v1 crashes on empty valid_times; v2 silently masks the entire recording
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:359-361 + v1/recording.py:741-747`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1014-1027`
- v1 behavior: _consolidate_intervals at line 747 dereferences start_indices[0] / stop_indices[0]; on an empty intervals array this raises IndexError before the masking branch is even evaluated. v1 effectively requires at least one valid interval.
- v2 behavior: _apply_artifact_mask with empty valid_times produces frame_ranges = [], skips the for-loop, and then the trailing branch (cursor=timestamps[0] < timestamps[-1]) appends the entire recording to artifact_frames. The sort runs on an all-zeros recording and almost certainly returns zero units.
- documented rationale: Likely uncovered edge case. A defensive 'if len(valid_times) == 0: raise ValueError' would match v1's noisy-failure semantics and avoid masking an entire recording.
- verifier reasoning: Claim verified by direct code reads.

v1 (src/spyglass/spikesorting/v1/sorting.py:359-361 → src/spyglass/spikesorting/v1/recording.py:741-747): `_consolidate_intervals` calls `np.searchsorted` on the empty intervals array and then immediately accesses `start_indices[0], stop_indices[0]` at line 747 → IndexError on empty input. Loud failure.

v2 (src/spyglass/spikesorting/v2/sorting.py:1014-1027): With `valid_times` of shape (0, 2): `cursor = timestamps[0]`; `for vs, ve in valid_times` iterates 0 times; trailing branch `if cursor < timestamps[-1]` is True → appends `(0, len(timestamps))` to `fr

### sorting#16  [MEDIUM | INT-JUST] v1 stored `noise_levels` as `np.asarray([1.0])` (numpy array); v2 stores as Python list `[1.0]` and broadcasts to `n_channels` at runtime
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:177; runtime at :411`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:159 (storage); src/spyglass/spikesorting/v2/sorting.py:1168-1176 (broadcast)`
- v1 behavior: v1's `default_clusterless` row stores `noise_levels: np.asarray([1.0])` and passes it directly to `detect_peaks` via `**sorter_params`. SI's `locally_exclusive` then indexes `noise_levels[chan] * detect_threshold` per channel -- a singleton array gets indexed only at [0] for chan=0 and would IndexError or silently misbehave on multi-channel recordings.
- v2 behavior: v2 schema stores `noise_levels: list[float]` (Python list). Runtime at sorting.py:1168-1176 takes `noise_levels=[1.0]` (singleton) and explicitly broadcasts via `_np.full(recording.get_num_channels(), float(nl[0]))` to a per-channel array before passing to `detect_peaks`. So v2 correctly handles the singleton.
- documented rationale: v2 fixes a latent v1 bug: v1's literal `np.asarray([1.0])` on a multi-channel recording would silently misread or crash inside SI's locally_exclusive. v2's explicit broadcast is the right fix and is well-documented.
- verifier reasoning: Verified all citations directly. v1/sorting.py:177 stores `"noise_levels": np.asarray([1.0])` (numpy singleton) and v1/sorting.py:411 passes it directly to `detect_peaks(recording, **sorter_params)` with no broadcast. v2/_params/sorter.py:159 declares `noise_levels: list[float] | None = Field(default=None)`; the v2 default row at v2/sorting.py:175 ships `{"noise_levels": [1.0]}` (singleton Python list). The runtime at v2/sorting.py:1168-1176 explicitly broadcasts a singleton to `n_channels` via `_np.full(recording.get_num_channels(), float(nl[0]), dtype=_np.float64)` before forwarding to `dete

### sorting#17  [MEDIUM | NEW-V2] v2 SorterParameters adds `params_schema_version` and `job_kwargs` columns beyond v1's `params` blob
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:138-144`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:88-104`
- v1 behavior: Schema is `(sorter, sorter_param_name) -> params: blob`. No versioning, no job-kwargs storage.
- v2 behavior: Schema is `(sorter, sorter_params_name) -> params: blob, params_schema_version=1: int, job_kwargs=null: blob`. The `insert1` override (v2/sorting.py:97-104) validates `params` against the per-sorter Pydantic schema and asserts `params_schema_version` matches. `job_kwargs` flows into `_resolved_job_kwargs` at compute time (v2/sorting.py:581) so per-row n_jobs/chunk_duration overrides are addressable.
- documented rationale: Schema-versioning enables forward evolution of pydantic schemas without losing the ability to detect stale rows. Per-row job_kwargs allows operators to override system-level n_jobs without editing each sorter's params blob. Critical foundation for catching the noise_levels=[1.0] class of injection bug — pydantic validation runs at insert time, not at populate time.
- verifier reasoning: Verified all claims against code at cited line numbers.

v1 schema (v1/sorting.py:138-144) is exactly `(sorter, sorter_param_name) -> sorter_params: blob` — no versioning, no job-kwargs column. (Note: v1 calls the secondary attribute `sorter_params` not `params` and the PK component `sorter_param_name` not `sorter_params_name`, but this is naming convention, not a substantive deviation from the finding's claim.)

v2 schema (v2/sorting.py:88-95) confirms exactly: `sorter varchar(64) / sorter_params_name varchar(128) / params: blob / params_schema_version=1: int / job_kwargs=null: blob`. Two gen

### sorting#18  [MEDIUM | DRIFT] v2 _apply_artifact_mask does NOT sort valid_times before walking; v1 _consolidate_intervals does
- **v1**: `src/spyglass/spikesorting/v1/recording.py:733-736`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1014-1027`
- v1 behavior: _consolidate_intervals at v1/recording.py:734-735 checks if intervals are monotone and calls np.sort(intervals, axis=0) when not. Unsorted valid_times still produces a correct mask.
- v2 behavior: _apply_artifact_mask iterates valid_times in caller order, using cursor = max(cursor, ve). Unsorted valid_times can leave gaps unmasked because the cursor advances past intermediate valid windows or produces wrong start frames when vs <= cursor. No sort, no monotonicity assertion.
- documented rationale: Probably an assumption that upstream IntervalList is always sorted. If that assumption holds it is harmless, but v2 silently produces wrong masks where v1 silently produced correct masks. Worth either documenting the assumption with an assert, or sorting defensively.
- verifier reasoning: Verified directly against the cited code.

v1 evidence (src/spyglass/spikesorting/v1/recording.py:733-735): `_consolidate_intervals` explicitly checks `if not np.all(intervals[:-1] <= intervals[1:]): intervals = np.sort(intervals, axis=0)`. v1 is defensively sorted.

v2 evidence (src/spyglass/spikesorting/v2/sorting.py:1014-1027): the loop `for vs, ve in valid_times: ... cursor = max(cursor, ve)` walks intervals in caller order with no sort and no monotonicity assertion. The docstring (lines 995-1002) makes no mention of an ordering precondition.

Concrete failure mode on unsorted input: with 

### sorting#19  [MEDIUM | NEW-V2] v2 `Sorting.Unit` part-table introduces per-unit Electrode FK (and peak_amplitude_uv / n_spikes) that v1 had no DataJoint-side equivalent of
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:232-322 (no Part class anywhere in v1 SpikeSorting)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:432-448, 1549-1648`
- v1 behavior: No Unit part table. Per-unit metadata (peak amplitude, n_spikes, brain region) lived only inside the analysis NWB file's `units` table. Not queryable via DataJoint join.
- v2 behavior: `Sorting.Unit` part with `-> Electrode` composite FK (nwb_file_name, electrode_group_name, electrode_id), plus `peak_amplitude_uv`, `n_spikes`. Populated by `_populate_unit_part` (v2/sorting.py:1549-1648) inside the make transaction. Brain region reachable via `Sorting.Unit * Electrode * BrainRegion`.
- documented rationale: New functionality, not a parity break: enables per-unit brain-region queries directly through DataJoint (shared-contracts 'Unit-Level Brain Region Tracing'). v1 callers using `nwb.units.to_dataframe()` are still supported via `Sorting.get_sorting(as_dataframe=True)` (v2/sorting.py:770-796).
- verifier reasoning: Verified directly against both files. (1) v1 sorting.py contains only three classes (SpikeSorterParameters L83, SpikeSortingSelection L198, SpikeSorting L233); no Part class anywhere. v1's SpikeSorting master at L233-240 stores only one object_id; per-unit metadata is reachable only by opening the analysis NWB. (2) v2 sorting.py L432-448 defines `Sorting.Unit` as a SpyglassMixinPart with composite FK `-> Electrode` plus `peak_amplitude_uv: float` and `n_spikes: int`, primary key `unit_id: int`. The docstring at L433-439 explicitly cites the shared-contracts entry "Unit-Level Brain Region Traci

### sorting#20  [MEDIUM | INT-JUST] v2 adds robust scratch-directory and AnalysisNwbfile cleanup on failure; v1 leaks both
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:305-322 (no cleanup), 419 (implicit __del__ cleanup)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1247-1342, 614-626, 686-705, 924-958`
- v1 behavior: v1's _run_spike_sorter uses 'sorter_temp_dir = tempfile.TemporaryDirectory(dir=temp_dir)' as a local (line 419); cleanup happens at object destruction via TemporaryDirectory.__del__. If sis.run_sorter raises and the GC holds a reference, the scratch dir leaks. v1's make_compute then make_insert: if _save_sorting_results writes the AnalysisNwbfile to disk and make_insert's AnalysisNwbfile().add or insert1 raises, the file is on disk but UNREGISTERED. No cleanup.
- v2 behavior: _run_si_sorter wraps the SI call in try/finally at 1247-1342 with explicit sorter_temp_dir.cleanup(). make_compute at 614-626 catches any exception in _write_units_nwb and explicitly shutil.rmtree's analyzer_folder before re-raising. make_insert at 686-705 catches any registration failure and unlinks the staged AnalysisNwbfile AND rmtree's analyzer_folder before re-raising. Sorting.delete override at 924-958 collects analyzer_folder paths BEFORE cascade delete and rmtree's each, closing the row-deletion lifecycle.
- documented rationale: Explicit failure-mode A (between _build_analyzer and end of make_compute) and failure-mode B (registration failure in make_insert) documented at sorting.py:546-549, 656-659. Analyzer folders are documented as '5-50 GB scratch' so leak is a real disk-cost concern.
- verifier reasoning: All cited locations verified by direct read.

v1 sorting.py:305-322: make_insert does AnalysisNwbfile().add() then insert1() with NO try/except. A failure between or after add() leaves the analysis NWB on disk but unregistered (orphan). Confirmed.

v1 sorting.py:419: sorter_temp_dir = tempfile.TemporaryDirectory(dir=temp_dir) is a bare local. Cleanup only via __del__ at GC time; if SI raises and the frame holds the reference, cleanup is non-deterministic. Confirmed.

v2 sorting.py:1243-1342 (_run_si_sorter): try/finally with explicit sorter_temp_dir.cleanup() at line 1342. Confirmed — and the 

### sorting#21  [MEDIUM | INT-JUST] v2 fixes v1 trailing-edge artifact mask off-by-one (last valid sample zeroed, last frame n-1 left unmasked)
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:382-388`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1023-1027, 1029-1055; test src/../tests/spikesorting/v2/test_single_session_pipeline.py:2905-3020`
- v1 behavior: Back-gap trigger range is np.arange(artifact_removed_intervals_ind[-1][1], len(timestamps) - 1). Because _consolidate_intervals returns stop_index INCLUSIVE (uses searchsorted(side='right') - 1), starting the trigger at stop_ind[-1] (rather than stop_ind[-1] + 1) ZEROS the last valid sample of the trailing valid interval. Ending at len(timestamps) - 1 LEAVES the very last recording sample (frame n-1) UNMASKED even when it falls in an artifact region. The two bugs partially cancel for the very-end case but mis-account boundary samples in general.
- v2 behavior: _apply_artifact_mask walks valid_times left-to-right with cursor = max(cursor, ve); trailing branch builds artifact range [searchsorted(timestamps, cursor), len(timestamps)). Includes frame n-1; first frame after last valid window is the first artifact frame. Off-by-one explicitly pinned by test_apply_artifact_mask_zeroes_artifact_frames at test_single_session_pipeline.py:2920-3020 (asserts last_artifact frame is zero and first valid frame is unchanged).
- documented rationale: v2 docstring at sorting.py:996-1055 documents the rewrite for boundary correctness and SI 0.104 ms_before/ms_after semantics. Test enforces both: last_artifact frame zero AND first-valid frame unchanged.
- verifier reasoning: Verified all citations directly. v1/sorting.py:382-388 confirms the trailing trigger range is np.arange(artifact_removed_intervals_ind[-1][1], len(timestamps) - 1) — start is the INCLUSIVE stop index of the last valid interval (per _consolidate_intervals at v1/recording.py:741-744 which returns searchsorted(side='right') - 1), and the end is exclusive at len(timestamps)-1, so frame n-1 is never zeroed. The inner gap loop at v1:378 correctly uses +1 for the start, making the trailing branch internally inconsistent. v2/sorting.py:1014-1027 walks valid_times with cursor = max(cursor, ve), and the

### sorting#22  [MEDIUM | NEW-V2] v2 introduces SortingAnalyzer with hardcoded analyzer-extension parameters (max_spikes_per_unit=500, waveforms ms_before=1.0/ms_after=2.0); v1 had none
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:324-459 (sorter run only)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1419-1448, 1437-1447`
- v1 behavior: v1 captures only the SortingExtractor; no SortingAnalyzer, no random_spikes / noise_levels / templates / waveforms computation. NWB write contains only spike_times, id, obs_intervals, curation_label.
- v2 behavior: make_compute calls _build_analyzer which constructs si.create_sorting_analyzer(sparse=True, format='binary_folder', return_in_uV=True, overwrite=True) and computes ['random_spikes', 'noise_levels', 'templates', 'waveforms'] with extension_params {'random_spikes': {'max_spikes_per_unit': 500, 'method': 'uniform'}, 'waveforms': {'ms_before': 1.0, 'ms_after': 2.0}}. These values are HARDCODED in source; not exposed via SorterParameters, job_kwargs, or dj.config.
- documented rationale: Documented in class docstring at 411-419 as the v2 architectural contract (analyzer-backed templates). Hardcoded extension params is an undocumented design decision; sufficient for franklab fixture but not user-tunable. Worth surfacing via SorterParameters.job_kwargs or a dedicated analyzer-params blob.
- verifier reasoning: Verified by direct read. v1 sorting.py:324-459 (_run_spike_sorter) returns only (sorting, timestamps) — no SortingAnalyzer construction, no extensions computed; v1 NWB write at lines 461+ contains only spike_times/id/obs_intervals/curation_label. v2 sorting.py:1419-1448 (_build_analyzer) constructs si.create_sorting_analyzer(sparse=True, format='binary_folder', return_in_uV=True, overwrite=True) and computes ['random_spikes','noise_levels','templates','waveforms'] with extension_params containing literally {'random_spikes': {'max_spikes_per_unit': 500, 'method': 'uniform'}, 'waveforms': {'ms_b

### sorting#23  [MEDIUM | INT-JUST] v2 introduces source-part polymorphism (Recording xor ConcatenatedRecording) replacing v1's flat single FK to SpikeSortingRecording
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:199-206`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:215-234,261-276,369-406`
- v1 behavior: Flat secondary key: `-> SpikeSortingRecording`. Exactly one source shape (single-session recording). Cannot represent concatenated sources.
- v2 behavior: Master has no Recording FK directly; instead, a part-table polymorphism: `RecordingSource` part FKs to `Recording`, `ConcatenatedRecordingSource` part FKs to `ConcatenatedRecording`. `insert_selection` enforces exactly-one via `has_recording == has_concat` check, and `resolve_source` re-checks at fetch time.
- documented rationale: Schema-first / zero-migration design: declare ConcatenatedRecordingSource final shape now, gate the runtime, so adding concat sorting later requires no DDL change. The 'NotImplementedError today' is explicitly the intended state.
- verifier reasoning: Verified all cited code directly. v1/sorting.py:199-206 confirms the flat `-> SpikeSortingRecording` FK. v2/sorting.py:215-234 confirms the master has no Recording FK, with two part tables RecordingSource (-> Recording) and ConcatenatedRecordingSource (-> ConcatenatedRecording). v2/sorting.py:261-276 confirms `has_recording == has_concat` exactly-one enforcement and the NotImplementedError gate on the concat path. v2/sorting.py:369-406 confirms resolve_source re-checks (total != 1 raises SchemaBypassError) and returns a SourceResolution discriminating on kind. The intentional justification is 

### sorting#24  [MEDIUM | INT-JUST] v2 replaces mandatory IntervalList FK with nullable ArtifactDetection FK; non-artifact custom interval lists no longer addressable
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:205, 259-265`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:219, 499-514`
- v1 behavior: Mandatory `-> IntervalList` FK. Caller could supply ANY interval (artifact-removed, raw-data-valid-times, or arbitrary user-defined intervals). v1 `make_compute` then masked the recording with whatever valid_times those rows held. The 'artifact-ness' was convention, not enforcement.
- v2 behavior: Nullable `-> [nullable] ArtifactDetection` FK. The only addressable interval source is an ArtifactDetection-populated IntervalList row named `artifact_{artifact_id}` (looked up at v2/sorting.py:504-512). `artifact_id=None` means no masking (sort the full recording). Cannot point at an arbitrary user-curated interval.
- documented rationale: v2 tightens semantics: artifact masking flows through a typed schema path. Loss of ability to point at arbitrary IntervalList rows is the cost. `artifact_id=None` covers the 'sort everything' case; ArtifactDetection covers the artifact-mask case; there is no v1-style 'use this arbitrary interval' case.
- verifier reasoning: All cited file:line references verified directly. v1/sorting.py:205 confirms mandatory `-> IntervalList` FK on SpikeSortingSelection; v1/sorting.py:259-265 confirms make_fetch reads valid_times from whatever IntervalList row the FK points at (no enforcement of artifact-ness). v2/sorting.py:215-220 confirms the FK was replaced with `-> [nullable] ArtifactDetection`, and v2/sorting.py:499-514 confirms the only interval-list lookup path is the canonical `artifact_{artifact_id}` name via `artifact_interval_list_name()` (verified at v2/utils.py:509-519). The justification comment at v2/sorting.py:2

### sorting#25  [MEDIUM | INT-UNJ] v2 ships KS4 / MS4 default rows regardless of whether the sorter is installed; only an MS4-specific test gates this
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:184-189`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:106-160; gating test at tests/spikesorting/v2/test_v1_parity.py:226-278`
- v1 behavior: v1's auto-extension at contents.extend iterates `sis.available_sorters()` (NOT `installed_sorters()`), so v1 also ships rows for not-installed sorters. The failure mode is the same: a `populate()` on an uninstalled-sorter row hits an SI error at sort time.
- v2 behavior: v2 ships KS4, MS4, MS5 unconditionally as part of `_DEFAULT_CONTENTS`. There is no install-time guard. The test test_ms4_default_row_only_shipped_when_ms4_installed at test_v1_parity.py:226-278 pins the contract for MS4 only: it FAILS on platforms without MS4 installed. The current env reports only ['spykingcircus2','tridesclous2'] as installed, so MS4 and KS4 default rows are present-but-broken if the user populates them.
- documented rationale: v1 had the same behavior, so this is parity-preserving in a literal sense, but the v2 test documents the brittleness as a known issue and the fix has not landed.
- verifier reasoning: Confirmed the finding directly against code and env:

(1) v1/sorting.py:184-189 iterates `sis.available_sorters()`, shipping rows for all SI-registered sorters regardless of install — exposure is real in v1, as the claim states.

(2) v2/sorting.py:106-160 hardcodes MS4 (two rows: tetrode and ctx variants), MS5, KS4, SC2, TDC2, clusterless_thresholder in `_DEFAULT_CONTENTS`. v2/sorting.py:185-198 `insert_default` calls `cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)` with no install gating. The docstring at line 191-192 acknowledges 'MS4 + KS4 are not deterministic' but still ships bot

### sorting#26  [LOW | NEW-V2] ArtifactDetection FK is nullable in v2; artifact_id=None skips masking entirely (new graceful path)
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:198-206 (FK), 364-397 (mask path)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:218-220 (FK), 558-562 (compute guard), 1480-1486 (obs fallback)`
- v1 behavior: v1's SpikeSortingSelection requires an IntervalList FK ('-> IntervalList'). The artifact-removed intervals come from a populated IntervalList row, and v1 always evaluates the masking branch (lines 364-397) even when the only valid interval covers the full recording.
- v2 behavior: SortingSelection FK is '-> [nullable] ArtifactDetection' (sorting.py:219). make_compute at 558 wraps the mask call in 'if sel_row.get("artifact_id") is not None and obs_intervals is not None'. When artifact_id is unset, the recording is sorted as-is. _write_units_nwb falls back to the full recording timestamp envelope as obs_intervals (1480-1486).
- documented rationale: Intentional: shared-contracts pattern that lets users skip the artifact-detection pipeline. The fallback to full-window obs_intervals matches v1 semantics for an all-valid IntervalList.
- verifier reasoning: All cited code matches. v1 SpikeSortingSelection at src/spyglass/spikesorting/v1/sorting.py:198-206 has '-> IntervalList' as a non-nullable FK; the mask path at 364-397 always evaluates. v2 SortingSelection at src/spyglass/spikesorting/v2/sorting.py:218-220 has '-> [nullable] ArtifactDetection'; make_compute at 558-562 guards 'if sel_row.get("artifact_id") is not None and obs_intervals is not None'; _write_units_nwb at 1480-1486 falls back to '[timestamps[0], timestamps[-1]]' when obs_intervals is None. The v2 docstring at 208-213 documents this as an intentional shared-contracts pattern ("mat

### sorting#27  [LOW | UNTESTED] Clusterless default row's `params_schema_version=3` is set explicitly in _DEFAULT_CONTENTS but the DB column default is 1; user-inserts without specifying the column would silently mismatch
- **v1**: `N/A`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:93; src/spyglass/spikesorting/v2/utils.py:256-267`
- v1 behavior: v1 has no params_schema_version concept; there is no schema-version validation.
- v2 behavior: v2 `SorterParameters` schema declares `params_schema_version=1: int` as the DB-level default (sorting.py:93). The clusterless schema is at `schema_version=3`. The shipped row explicitly passes 3 in the contents tuple, so it lands correctly. But `_assert_schema_version_matches` (utils.py:256-258) returns early when `params_schema_version` is absent from the supplied row; in that case DataJoint applies the DB default of 1, and the params blob (schema_version=3) ends up paired with an outer-column 1. The drift-detector cannot then catch the divergence at insert time because it already returned early.
- documented rationale: Likely an oversight: the assertion correctly catches mismatches when both columns are present, but it ignores the implicit DB-default path. A defensive fix would compute `outer = row.get('params_schema_version', model_cls.model_fields['schema_version'].default)` and validate unconditionally.
- verifier reasoning: Directly verified by reading the cited code.

(1) `src/spyglass/spikesorting/v2/sorting.py:93` — DB column default literally is `params_schema_version=1: int`.
(2) `src/spyglass/spikesorting/v2/sorting.py:177-180` — clusterless default row ships `3` explicitly and an inline comment confirms `ClusterlessThresholderSchema` is at `schema_version=3`.
(3) `src/spyglass/spikesorting/v2/utils.py:257-259` — `_assert_schema_version_matches` reads `inner` then literally executes `if "params_schema_version" not in row: return`. Confirmed early-return path.
(4) The helper's own docstring (utils.py:246-247

### sorting#28  [LOW | INT-JUST] Clusterless schema `local_radius_um` retyped from int to float; runtime renames to `radius_um`
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:128 (docstring), :175 (default), :405-408 (rename)`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:158; src/spyglass/spikesorting/v2/sorting.py:1134-1137`
- v1 behavior: v1 docstring at sorting.py:128 declares `local_radius_um: int`. Default row stores `100` (int). Runtime at v1/sorting.py:405-408 renames key to `radius_um` for SI>=0.99.1.
- v2 behavior: v2 schema `local_radius_um: float = Field(default=100.0, gt=0.0)`. Default is `100.0` (float). Runtime at sorting.py:1136-1137 same rename to `radius_um`. Same renaming behavior, but stored value is float rather than int.
- documented rationale: Aligned with SI's actual float-typed surface; doesn't materially change behavior.
- verifier reasoning: Verified all cited locations directly. v1/sorting.py:128 docstring says `local_radius_um: int`, v1/sorting.py:175 default contents row stores int `100`, and v1/sorting.py:405-408 renames `local_radius_um` -> `radius_um` for SI>=0.99.1. v2/_params/sorter.py:158 defines `local_radius_um: float = Field(default=100.0, gt=0.0)` and v2/sorting.py:1136-1137 performs the identical rename to `radius_um`. The v2 file has a docstring comment at lines 1134-1135 documenting the rename rationale ("v1-era kwarg rename: SI 0.99 ``local_radius_um`` became ``radius_um`` in 0.101+"). Tests at tests/spikesorting/

### sorting#29  [LOW | INT-JUST] Clusterless schema drops `outputs` and `random_chunk_kwargs` fields; v1 stored both
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:178-181`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:115-159, especially 125-133 (docstring explains both drops)`
- v1 behavior: v1's `default_clusterless` row stores `outputs: 'sorting'` (Spyglass routing hint) and `random_chunk_kwargs: {}`. v1 runtime at v1/sorting.py:404 strips `outputs` before calling `detect_peaks`. `random_chunk_kwargs` is passed through but the SI 0.99 -> 0.104 rename means it has no effect.
- v2 behavior: v2 schema `extra='forbid'` rejects both fields outright. A v1 row literally inserted into v2 would fail validation. The v2 runtime at sorting.py:1143-1144 strips both kwargs defensively (`for stale in ("outputs", "random_chunk_kwargs"): params.pop(stale, None)`), but at insert time they cannot land. Verified: `ClusterlessThresholderSchema(outputs='sorting')` and `ClusterlessThresholderSchema(random_chunk_kwargs={})` both raise `extra_forbidden`.
- documented rationale: v1's `outputs='sorting'` is a Spyglass routing hint never consumed by SI; SI 0.104 also renamed `random_chunk_kwargs` to `random_slices_kwargs`. Schema correctly drops both rather than carry dead fields.
- verifier reasoning: All citations verified by direct read. v1/sorting.py:177-181 contains the `default_clusterless` row with `random_chunk_kwargs: {}` and `outputs: "sorting"` exactly as claimed. v1/sorting.py:404 confirms `sorter_params.pop("outputs", None)` runtime strip. v2 schema at _params/sorter.py:115-159 uses `model_config = ConfigDict(extra="forbid")` (line 152) and does not declare either field, so per pydantic semantics a v1 row containing them would raise `extra_forbidden` at validation time. The defensive runtime strip is present at sorting.py:1143-1144 exactly as cited (`for stale in ("outputs", "ra

### sorting#30  [LOW | INT-JUST] Clusterless schema's `peak_sign` expanded to `['neg','pos','both']`; v1 docstring documented only `('neg','pos')`
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:124`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:156`
- v1 behavior: v1 docstring at sorting.py:124 declares `peak_sign: enum('neg', 'pos')`. The default row stores `peak_sign='neg'`. v1 stores arbitrary strings without validation.
- v2 behavior: v2 schema `peak_sign: Literal['neg', 'pos', 'both'] = 'neg'`. The 'both' option is a v2 addition matching SI's actual surface (SI `detect_peaks` accepts 'both').
- documented rationale: v1 docstring undercounted SI's actual API; v2 schema matches SI's full surface.
- verifier reasoning: Verified directly. V1 sorting.py:123-124 documents `peak_sign: enum ("neg", "pos")` — only two values. V2 _params/sorter.py:156 declares `peak_sign: Literal["neg", "pos", "both"] = "neg"` — adds 'both'. Default unchanged ('neg' on both sides). SI's `detect_peaks` source (peak_detection.py:366, 448) documents `peak_sign: "neg" | "pos" | "both"` and asserts `peak_sign in ("both", "neg", "pos")` at lines 392/482, confirming SI's actual surface includes 'both'. V1 stored arbitrary strings without pydantic validation (definition is just `sorter_params: blob`), so v1 would have silently accepted 'bo

### sorting#31  [LOW | INT-JUST] ConcatenatedRecordingSource has no key_source filter on `Sorting`; populate over a manually-inserted concat row would attempt make and raise NotImplementedError mid-iteration
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:232-322 (no concat path)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:457-474 (no key_source); src/spyglass/spikesorting/v2/sorting.py:271-276 (helper gate)`
- v1 behavior: No concat concept; no equivalent code path.
- v2 behavior: `Sorting` class has no `key_source` override (verified by grep). `Sorting.populate(...)` iterates over ALL SortingSelection master rows including any backed by `ConcatenatedRecordingSource`. The first such row dispatches into `make_fetch`, which calls `resolve_source` (v2/sorting.py:469) and raises `NotImplementedError` (v2/sorting.py:471-474). `insert_selection` rejects concat at line 271-276 so concat rows cannot be created via the helper — but a raw `cls.insert1` would bypass this. There is no silent-fallthrough: the NotImplementedError fires before any compute. However, mid-populate failure is louder than necessary.
- documented rationale: Guard at make_fetch is defense-in-depth — by construction (insert_selection-only path), no concat row should ever exist. A bypass via raw insert is the only path to reach the NotImplementedError at populate time. Could be improved by adding `key_source = SortingSelection - SortingSelection.ConcatenatedRecordingSource.proj()` to skip concat rows in populate, but the current behavior is correct, just loud.
- verifier reasoning: All claims verified by direct code reading:

1. No `key_source` override in v2/sorting.py — grep returned zero matches; the `Sorting` class at v2/sorting.py:410-455 jumps straight from definition/`Unit` part/`_parallel_make = True` to `make_fetch` without a `key_source` line.

2. `make_fetch` at v2/sorting.py:457-474 calls `SortingSelection.resolve_source(key)` and raises `NotImplementedError` at lines 471-474 if `source.kind != "recording"`, before any compute. The docstring at lines 460-465 explicitly states this is a "Layer-2 source re-check" that "fires here; concat raises NotImplementedEr

### sorting#32  [LOW | INT-JUST] Job-kwargs global mutation in v2; v1 had none — small concern about parallel-populate worker isolation
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:419-455 (no global mutation)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1297-1335`
- v1 behavior: v1 had no si.set_global_job_kwargs / get_global_job_kwargs calls. Job-related controls were passed via sorter_params and tempdir directly.
- v2 behavior: _run_si_sorter at 1304-1335 mutates SI global state via si.set_global_job_kwargs(**sj_kwargs), captures previous_global, then restores in finally via reset + re-apply (1325-1335). Within a worker process the restore is exact. _parallel_make=True (455) means workers are separate processes so cross-worker contamination is structurally impossible. Within a single worker, recursive Sorting.make calls would see the prior global re-applied correctly.
- documented rationale: Necessary for SI 0.104 API surface. Cleanup-on-exception not wrapped (the finally restores only after the try block returns), so an unhandled raise inside the sort would leak the mutated global. Inspecting: the try at 1322-1323 only wraps the sis.run_sorter call; if it raises, the finally at 1324-1335 still runs because Python finally is unconditional. So the leak concern is moot — the restore always fires.
- verifier reasoning: All citations verified directly. v1 sorting.py:419-455 contains no si.set_global_job_kwargs / get_global_job_kwargs / reset_global_job_kwargs calls (confirmed by grep). v2 sorting.py:1297-1335 implements exactly the pattern described: captures previous_global at 1304, conditionally sets sj_kwargs at 1306, calls sis.run_sorter inside try at 1322-1323, and restores via reset+re-apply in finally at 1324-1335. The long inline comment at 1287-1303 explains why job_kwargs must NOT be splatted into run_sorter (SI 0.104 sorter validators reject unknown kwargs like n_jobs / chunk_duration / progress_ba

### sorting#33  [LOW | INT-JUST] KS4 schema selects only 5 KS4 fields (Th_universal/Th_learned/nblocks/max_cluster_subset/do_CAR) as typed defaults; rest pass through via `extra='allow'`
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:184-188`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:91-112`
- v1 behavior: v1's KS4 default row would be `sis.get_default_sorter_params('kilosort4')`, which (in environments where KS4 is installed) returns ~30+ fields covering batch_size, nearest_chans, n_pcs, dminx, x_centers, etc. Every field is part of the stored params dict.
- v2 behavior: v2 schema only types 5 KS4 fields as named defaults and uses `extra='allow'` for the rest. The default row stores `{schema_version: 1, Th_universal: 9.0, Th_learned: 8.0, nblocks: 1, max_cluster_subset: 25000, do_CAR: True}` (6 keys). All other KS4 fields fall back to SI's per-version defaults at sort time, which could shift across SI upgrades without a Spyglass schema bump.
- documented rationale: Author deliberately mirrors v1's 'pass any SI kwarg through' escape hatch via `extra='allow'`. Trade-off: KS4 SI default changes silently flow into v2 unless the stored params override them.
- verifier reasoning: Verified all citations directly. v1/sorting.py:184-189 confirmed - extends contents with sis.get_default_sorter_params(sorter) for every available sorter, which in a KS4-installed environment produces ~30+ fields. v2/_params/sorter.py:91-112 confirmed - Kilosort4Schema types exactly 5 KS4 fields (Th_universal=9.0, Th_learned=8.0, nblocks=1, max_cluster_subset=25_000, do_CAR=True) plus schema_version, with model_config = ConfigDict(extra="allow"). The docstring at lines 96-100 explicitly cites v1/sorting.py:184-189 as the matching escape hatch, matching the finding's rationale verbatim. test_pa

### sorting#34  [LOW | INT-UNJ] MS4 schema `freq_min`/`freq_max` ranged with `le=15000.0` rejects high-sampling-rate use cases
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:159, 164`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:58-59`
- v1 behavior: v1 stores `freq_min`/`freq_max` as integers with no range validation. A user wanting `freq_max=20000` for a 96-kHz recording could insert it.
- v2 behavior: v2 schema constrains `freq_min: float = Field(gt=0.0, le=15000.0)` and `freq_max: float = Field(gt=0.0, le=15000.0)`. A user trying to insert `freq_max=20000` for a high-sample-rate recording gets `ValidationError`. Verified: `MountainSort4Schema(freq_min=20000)` raises `less_than_equal` error.
- documented rationale: Likely defense-in-depth for the lab's standard 30 kHz Intan/Trodes recordings (Nyquist=15 kHz). For unusual sampling rates this is a silent breakage.
- verifier reasoning: Verified divergence directly. v1/sorting.py:159,164 stores freq_min/freq_max as integers inside a blob `sorter_params` dict with no range validation (DataJoint blob column, accepts any value). v2 _params/sorter.py:58-59 defines `freq_min: float = Field(default=600.0, gt=0.0, le=15000.0)` and `freq_max: float = Field(default=6000.0, gt=0.0, le=15000.0)`. A user passing freq_max=20000 (valid in v1) would get a pydantic `less_than_equal` ValidationError in v2.

Docstring at sorter.py:38-51 documents the defaults and `extra="forbid"` policy but does NOT justify the 15 kHz ceiling. There is no ment

### sorting#35  [LOW | INT-JUST] MS4 schema's `detect_threshold` typed as `float`/`gt=0.0`; v1 stored it as an int
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:152`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:64`
- v1 behavior: v1 `mountain_default` stores `detect_threshold: 3` (int). v1 has no validation; a user could insert `detect_threshold=0` or a negative value.
- v2 behavior: v2 schema `detect_threshold: float = Field(default=3.0, gt=0.0)`. The default normalizes to `3.0` (float), and `0` is rejected. Verified: `MountainSort4Schema(detect_threshold=0)` raises `greater_than` error.
- documented rationale: Type tightening to documented float; gt=0.0 rules out nonsensical zero/negative thresholds. Not documented in the schema.
- verifier reasoning: Verified both citations directly. v1 sorting.py:152 stores `"detect_threshold": 3` (int literal) inside the raw `mountain_default` dict with no validation. v2 _params/sorter.py:64 declares `detect_threshold: float = Field(default=3.0, gt=0.0)`, normalizing to float and rejecting non-positive values. The pydantic `gt=0.0` constraint is a real behavioral divergence: v1 would accept `detect_threshold=0` or negative values without complaint, v2 raises a ValidationError. Test coverage exists for the default match (test_params_validation.py:208-218 `test_ms4_default_matches_v1_mountain_default` asse

### sorting#36  [LOW | INT-JUST] MS4 schema's `freq_min`/`freq_max` defaults silently choose v1's tetrode preset values (600/6000) over SI's MS4 defaults (300/6000)
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:145-153 (`mountain_default`), :159 (tetrode), :164 (cortex)`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:58-59`
- v1 behavior: v1's `mountain_default` block does NOT include `freq_min` / `freq_max`; those keys come from per-row presets: `freq_min=600/freq_max=6000` for the tetrode preset and `freq_min=300/freq_max=6000` for the cortex preset. v1's auto-shipped `mountainsort4`/`default` row uses SI defaults `freq_min=300, freq_max=6000`.
- v2 behavior: v2 schema bakes `freq_min: float = Field(default=600.0)` and `freq_max: float = Field(default=6000.0)` directly into `MountainSort4Schema`. So a user calling `MountainSort4Schema()` with no args gets the *tetrode* preset values, not SI's MS4 wrapper defaults. The cortex preset row still overrides `freq_min=300` explicitly, so the cortex preset itself is correct.
- documented rationale: Author chose tetrode-preset values (Frank-lab production default) over SI's wrapper defaults. Justified for the lab; documented in the schema docstring.
- verifier reasoning: Verified all cited file:line references directly.

v1 confirmation (src/spyglass/spikesorting/v1/sorting.py):
- Lines 145-153: `mountain_default` dict contains detect_sign, adjacency_radius, filter, whiten, num_workers, clip_size, detect_threshold, detect_interval — NO `freq_min`/`freq_max` keys.
- Line 159: tetrode preset row `{**mountain_default, "freq_min": 600, "freq_max": 6000}`.
- Line 164: cortex preset row `{**mountain_default, "freq_min": 300, "freq_max": 6000}`.

v2 confirmation (src/spyglass/spikesorting/v2/_params/sorter.py):
- Line 58: `freq_min: float = Field(default=600.0, gt=0.

### sorting#37  [LOW | UNTESTED] MS4 tetrode preset row (`franklab_tetrode_hippocampus_30kHz_ms4`) has no parameter-pinning test for `freq_min=600`; cortex preset is exercised end-to-end but tetrode preset is not
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:159`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:109-116; coverage gap at tests/spikesorting/v2/test_single_session_pipeline.py:7693-7705`
- v1 behavior: v1 tetrode preset stores `freq_min=600, freq_max=6000` plus mountain_default. No specific v1 test pinned this; it was exercised through manual notebook use.
- v2 behavior: v2 tetrode preset is in `_DEFAULT_CONTENTS` at lines 109-116 with `{'freq_min': 600.0, 'freq_max': 6000.0}`. The pinning test `test_ms4_default_matches_v1_mountain_default` at test_params_validation.py:208-218 covers the MS4 schema's mountain_default-derived fields but NOT `freq_min=600`. The `_MS4_GT_CASES` integration test list at test_single_session_pipeline.py:7703-7705 only uses `franklab_probe_ctx_30kHz_ms4` (cortex/300 Hz). The tetrode row's filter band is not asserted anywhere.
- documented rationale: Tetrode-MS4 ground-truth recovery on synthetic MEArec fixtures was found impossible (4-channel MS4 limitation), so the case was dropped. A pure parameter-pinning test (assert the stored params dict has `freq_min=600`) would be cheap and would catch silent drift.
- verifier reasoning: All claims verified by direct inspection.

(1) v1 row at src/spyglass/spikesorting/v1/sorting.py:155-160 stores `franklab_tetrode_hippocampus_30KHz` with `{**mountain_default, "freq_min": 600, "freq_max": 6000}`.

(2) v2 row at src/spyglass/spikesorting/v2/sorting.py:106-116 stores `franklab_tetrode_hippocampus_30kHz_ms4` with `{"freq_min": 600.0, "freq_max": 6000.0}` (passed through `_validate_params` so MS4 mountain_default fields are filled by the schema).

(3) The pinning test `test_ms4_default_matches_v1_mountain_default` at tests/spikesorting/v2/test_params_validation.py:208-218 asserts 

### sorting#38  [LOW | DRIFT] No enforcement that artifact_id's source-recording matches the sort's recording_id — relies on implicit IntervalList nwb_file_name overlap
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:205, 259-265`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:215-220, 499-514`
- v1 behavior: IntervalList FK shares `nwb_file_name` column with SpikeSortingRecording's FK chain. DataJoint enforces this column-name overlap as a constraint: the IntervalList row must belong to the same nwb_file_name as the recording. Result: the FK chain itself prevents cross-session artifact references.
- v2 behavior: ArtifactDetection FK uses `artifact_id` (a surrogate UUID); RecordingSource part FK uses `recording_id` (also a surrogate UUID). DataJoint has no overlap to constrain. A user could pass `artifact_id` for ArtifactDetection on Recording-B and `recording_id`=Recording-A. The implicit cross-check happens only at make_fetch (v2/sorting.py:504-512) where `IntervalList & {nwb_file_name, artifact_interval_list_name(artifact_id)}` would fail to find a row IF the artifact's session differs. For `SharedArtifactGroup` artifacts spanning multiple sessions, the lookup succeeds whenever Recording's session is in the group — which may or may not be the intended semantic.
- documented rationale: SharedArtifactGroup is an explicit cross-session feature — so allowing artifact_id to span sessions is by design. For the single-RecordingSource artifact case, no insert-time check ensures the sort's Recording == artifact's Recording. Worth a runtime invariant assertion in `insert_selection` (look up the artifact's RecordingSource part and confirm recording_id match) for the single-source case.
- verifier reasoning: CORE claim is confirmed but its rationale needs correction.

CONFIRMED parts:
- v1 FK chain: SpikeSortingSelection (v1/sorting.py:198-206) FK chains both SpikeSortingRecording (which propagates nwb_file_name via -> Raw, -> SortGroup, -> IntervalList in SpikeSortingRecordingSelection at v1/recording.py:147-157) AND IntervalList. DataJoint merges same-named columns, so the IntervalList row used as artifact source MUST share nwb_file_name with the recording. Cross-session artifact references are structurally impossible in v1.
- v2 surrogate-UUID FKs: SortingSelection master FK is `-> [nullable] A

### sorting#39  [LOW | UNTESTED] NonIntegerUnitIDError has zero test coverage even though v2 explicitly added the guard
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:594-598`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1610-1618`
- v1 behavior: v1 passes sorting.get_unit_ids() through directly to nwbf.add_unit(id=unit_id, ...) at sorting.py:594-598 without any int-cast or validation. Non-integer IDs propagate to pynwb which silently converts or raises a pynwb-shaped error -- no clear failure mode.
- v2 behavior: v2 _populate_unit_part wraps int(unit_id) in try/except at sorting.py:1610-1618 and raises NonIntegerUnitIDError with a clear message and a remap suggestion. The check is real production code; the guard exists in defense of the rationale that 'kilosort4 may return numpy.int64' (audit-task context).
- documented rationale: The guard reads as defensive programming for a case nobody has hit. Either build a test that constructs a synthetic NumpySorting with object-dtype unit_ids (e.g., strings) and asserts the raise, or document why the guard is unreachable in practice and remove it.
- verifier reasoning: Verified directly: v1/sorting.py:592-599 passes sorter unit_ids to nwbf.add_unit(id=...) with no validation; v2/sorting.py:1609-1618 wraps int(unit_id) in try/except and raises NonIntegerUnitIDError defined at v2/exceptions.py:33-36 with a remap-suggestion message. grep -rn 'NonIntegerUnit' across tests/ returns zero matches — no test exercises the raise path. The only int(unit_id) usages in tests (baseline_capture.py:279, test_single_session_pipeline.py:2687) are successful conversions, not error-path tests. The audit rationale (kilosort4 numpy.int64) is unconvincing because int(np.int64(x)) 

### sorting#40  [LOW | INT-JUST] Return type of `insert_selection` changed from inconsistent (dict OR list of dicts) to consistent (single PK dict)
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:223-229`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:312-313, 346`
- v1 behavior: `insert_selection` returns `key` (a dict, augmented with `sorting_id`) on new insert, but `query.fetch(as_dict=True)` (a LIST of dicts) when a similar row already exists. Caller code must handle both shapes.
- v2 behavior: Always returns a single PK-only dict: `{sorting_id: <uuid>}`. The existing-row path returns `dict(next(iter(unique)))`, the new-row path returns `{k: new_master_key[k] for k in cls.primary_key}`.
- documented rationale: v1's inconsistent return was a latent bug; v2 fixes it. Note: a similar issue lurks in v1/recording.py:176-182 for RecordingSelection. Test `test_sorting_selection_artifact_id_none_is_distinct_identity` (test_single_session_pipeline.py:5107-5174) implicitly pins the v2 dict-return shape.
- verifier reasoning: Verified directly. v1/sorting.py:223-229 confirms: line 226 returns `query.fetch(as_dict=True)` (list of dicts) on the existing-row path, line 229 returns `key` (a single dict mutated with `sorting_id`) on the new-row path -- inconsistent return shape. v2/sorting.py:312-313 returns `dict(next(iter(unique)))` (single PK dict reconstructed from the unique tuple set) on the existing path, and v2/sorting.py:346 returns `{k: new_master_key[k] for k in cls.primary_key}` (single PK-only dict) on the new path. Both v2 paths return a single dict with only primary-key fields. The test at tests/spikesort

### sorting#41  [LOW | INT-UNJ] Zero-unit sort persists a phantom analyzer_folder path that points to a never-created directory
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:578-602`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1396-1403, 674, 957`
- v1 behavior: v1 has no analyzer concept. On zero units, _write_sorting_to_nwb writes an empty Units NWB and the Sorting row carries (analysis_file_name, object_id, time_of_sort) only -- every persisted reference is to a real artifact on disk.
- v2 behavior: On zero units, _build_analyzer returns the would-be folder path WITHOUT calling create_sorting_analyzer (sorting.py:1396-1403). make_insert then commits the master row with analyzer_folder=str(folder) (sorting.py:674) and n_units=0 -- the path column points to a directory that was NEVER created. Callers must read n_units first; get_analyzer guards (sorting.py:841-848), delete() guards via folder.exists() (sorting.py:957), but external code or external folder-discovery scripts that scan analyzer_folder paths would dereference a missing directory.
- documented rationale: Intentional per the inline comment, but the rationale ('get_analyzer guards it') only covers Spyglass callers. Storing a path to a non-existent folder is a design smell -- NULL or '' would be safer. Suggest changing the schema/insert to store NULL for zero-unit rows; the existing exists() guards in delete()/get_analyzer would still work.
- verifier reasoning: Verified directly. sorting.py:1396-1403 returns the would-be folder path without calling create_sorting_analyzer when sorting.get_num_units() == 0. make_insert at sorting.py:674 unconditionally commits analyzer_folder=str(analyzer_folder). The inline comment at 1387-1395 explicitly documents this as intentional ("Return the (not yet created) folder path for the row; Sorting.get_analyzer raises ZeroUnitAnalyzerError"). The test at test_single_session_pipeline.py:3509-3512 asserts only isinstance(str) and truthy, with a comment at lines 3506-3508 explicitly acknowledging the design choice ("the 

### sorting#42  [LOW | INT-UNJ] _run_si_sorter MS4 path mutates numpy globally (np.Inf = np.inf) with no teardown
- **v1**: `n/a (v1 used SI 0.99, no MS4-on-numpy2 compat issue)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1257-1258`
- v1 behavior: v1 does not mutate numpy. The spikeextractors compatibility issue did not exist at v1's pinned SI version (0.99) which preceded numpy 2.0's removal of np.Inf.
- v2 behavior: Inside _run_si_sorter at sorting.py:1257-1258, the code does `_np.Inf = _np.inf` if sorter is mountainsort4 and `_np.Inf` is missing. This is a GLOBAL module mutation that persists for the lifetime of the process. There is no try/finally to remove the alias after the sort. Subsequent test runs or unrelated code see numpy with an undocumented Inf attribute they did not have before.
- documented rationale: Intentional shim for a transitive dep (spikeextractors 0.9.11). The behavior is benign-but-permanent. The right fix is either upstream (pin spikeextractors to a fixed version) or use a try/finally with del _np.Inf for tighter scoping. Currently untested either way.
- verifier reasoning: Verified directly. v2 sorting.py:1257-1258 inside `_run_si_sorter` executes `_np.Inf = _np.inf` when sorter is mountainsort4 and the attr is missing. The surrounding try/finally blocks (outer at 1247/1336 for tempdir, inner at 1322-1335 for global job kwargs) do not remove the alias; there is no `del _np.Inf` anywhere in the file. The comment at lines 1254-1256 explicitly states the mutation "survives subsequent calls", confirming the permanent process-wide effect is intentional. v1 sorting.py contains no `np.Inf` reference (confirmed via grep), so v1 does not mutate numpy — consistent with it

### sorting#43  [LOW | DRIFT] _run_si_sorter: sorter_temp_dir.cleanup() failure can mask the upstream sort exception
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:419-420`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1336-1342`
- v1 behavior: v1 creates sorter_temp_dir = tempfile.TemporaryDirectory(dir=temp_dir) without an explicit cleanup() call (sorting.py:419-420); GC handles teardown. If the sort fails, the original exception propagates; the tempdir cleanup happens later and any failure during it is swallowed by GC.
- v2 behavior: v2 wraps the sorter call in a try/finally and calls sorter_temp_dir.cleanup() in the outer finally (sorting.py:1336-1342). Python's exception machinery means that if cleanup() raises (permission error, subprocess holding a file), the original sort exception is REPLACED by the cleanup exception. The actual failure cause is lost.
- documented rationale: The explicit cleanup is correct for the parallel-populate pool. Recommend wrapping the cleanup in its own try/except that logs but does not re-raise -- so an actual sort failure surfaces with its original traceback. Or use ExitStack to preserve the primary exception.
- verifier reasoning: Verified directly against both files. v1/sorting.py:419-420 creates the TemporaryDirectory with no explicit cleanup() call anywhere in the file (grep confirms only three references: creation, naming, and use as output_folder). v2/sorting.py:1247 opens a try: block that ends at 1336-1342 with a finally: sorter_temp_dir.cleanup(). Python try/finally semantics: if the finally raises while another exception is in flight, the new exception replaces the original in the propagated traceback (the original is retained as __context__ but the visible failure surface is the cleanup error). The docstring a

### sorting#44  [LOW | INT-JUST] v1 SpikeSortingSelection silently drops keys not in the table; v2 SortingSelection.insert_selection validates required keys explicitly
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:209-229 (no explicit key validation)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:261-283, 325-333`
- v1 behavior: Passing a key with missing required FKs (e.g., no `interval_list_name`) silently aliases via `cls & key` which still matches; or fails at `cls.insert1` with a DataJointError that doesn't say what was missing.
- v2 behavior: Required keys (`sorter`, `sorter_params_name`) raise an explicit `ValueError` at v2/sorting.py:278-283 before any DB operation. `recording_id` xor `concat_recording_id` is required via a clear ValueError at 263-270. Missing SorterParameters row is translated by `_ensure_lookup_row_exists` (v2/sorting.py:325-333) into a friendly 'call SorterParameters.insert_default()' message.
- documented rationale: Quality-of-error improvement; matches the defense-in-depth pattern. Maps obscure DataJoint errors into user-actionable messages.
- verifier reasoning: Verified by reading the cited lines directly. v1/sorting.py:209-229: insert_selection has zero key validation — it runs `cls & key` (line 223) then `cls.insert1(key, skip_duplicates=True)` (line 228); any missing FK surfaces as an opaque DataJoint error. v2/sorting.py:261-283: explicit ValueError for `recording_id` xor `concat_recording_id` (263-270), NotImplementedError for concat (271-276), and explicit ValueError 'requires {required!r} in key' for missing 'sorter'/'sorter_params_name' (278-283). v2/sorting.py:325-333: `_ensure_lookup_row_exists` translates missing SorterParameters into a fr

### sorting#45  [LOW | INT-JUST] v1 schema column was `sorter_param_name` (no s); v2 renamed to `sorter_params_name` (with s) — breaks any code referencing the v1 column name
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:141 (schema), src/spyglass/spikesorting/v1/sorting.py:90 (docstring uses different name)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:90, 278-282`
- v1 behavior: Column literally named `sorter_param_name` (no trailing 's') in `SpikeSorterParameters.definition` at v1/sorting.py:141. Note the v1 docstring at v1/sorting.py:90 documents the field as `sorter_params_name` — v1 docstring and schema disagreed.
- v2 behavior: Column named `sorter_params_name` (with 's'), consistent throughout v2 (v2/sorting.py:90, in SortingSelection.insert_selection at 278-282).
- documented rationale: Resolves a v1 self-inconsistency (docstring vs schema). v2 picks the plural form. Breaks any external code/notebook that queried by the v1 column name; given v2 already has a fresh-schema break, this fits.
- verifier reasoning: Verified directly. v1/sorting.py:141 schema column reads `sorter_param_name: varchar(200)` (no trailing 's'), while v1/sorting.py:90 docstring documents the same field as `sorter_params_name` (with 's') — v1 was self-inconsistent. v2/sorting.py:90 reads `sorter_params_name: varchar(128)` (with 's'), matching the v1 docstring spelling. v2 uses `sorter_params_name` consistently throughout: insert_selection requirement check (278-282), restriction dicts (287, 329, 481), error messages (1412), and the SortingSelection definition. Test fixtures confirm the rename: tests/spikesorting/v2/_smoke_const

### sorting#46  [LOW | NEW-V2] v2 _populate_unit_part raises RuntimeError on sort-group / channel mismatch; v1 had no equivalent check
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:546-602 (no electrode association)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1610-1626`
- v1 behavior: v1 never associates units with electrodes at the DataJoint layer; only spike_times survive into NWB. Any sort-group / recording channel-id mismatch is silent at the v1 sorting stage and surfaces (or not) later in metric computation.
- v2 behavior: _populate_unit_part at 1620-1626 raises 'RuntimeError(f"Sorting.make: peak channel {peak_chan} for unit {int_unit_id} is not in sort group {sort_group_id} for {nwb_file_name!r}. Sort group / recording channel-id mismatch.")' when the peak channel resolved from the SortingAnalyzer is not present in SortGroupV2.SortGroupElectrode rows.
- documented rationale: Reasonable to upgrade to a named exception (e.g. ChannelGroupMismatchError) for consistency with the rest of the v2 exception surface.
- verifier reasoning: Verified directly: v2 sorting.py:1620-1626 raises bare RuntimeError when peak_chan is not in electrode_by_id (built from SortGroupV2.SortGroupElectrode rows at 1597-1606). v1 sorting.py:546-602 (_write_sorting_to_nwb) only writes spike_times to NWB; grep for electrode_by_id/peak_chan/peak_channel in v1 sorting.py returned no matches, confirming v1 has no DataJoint-layer unit-to-electrode association and therefore no equivalent check. v2/exceptions.py defines 10 named exception classes (NonIntegerUnitIDError, ZeroUnitSortError, RecordingTruncatedError, etc.) but no ChannelGroupMismatchError — t

### sorting#47  [LOW | NEW-V2] v2 adds `prune_orphaned_selections` for source-part orphans; no v1 equivalent (and impossible in v1's flat schema)
- **v1**: `n/a — no equivalent code in v1/sorting.py`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:348-366`
- v1 behavior: Source orphans cannot exist (no source-part split). The flat SpikeSortingSelection has all FKs inline — DataJoint cascade-deletes the master when its FK targets vanish.
- v2 behavior: Source orphan = master row whose source-part rows are all gone (e.g., upstream Recording was cascade-deleted but the SortingSelection master rolled into a separate transaction). `prune_orphaned_selections` (v2/sorting.py:348-366) finds and optionally deletes these via `cautious_delete`. Uses shared `find_orphaned_masters` helper at v2/utils.py:69-91.
- documented rationale: Necessary maintenance hook introduced by the source-part design. Tested by `test_prune_orphaned_selections_finds_master_without_part` (test_single_session_pipeline.py:1723-1760).
- verifier reasoning: Verified directly against code:

1. v1 has no `prune_orphaned` method — grep confirms zero matches in `src/spyglass/spikesorting/v1/sorting.py`. v1's flat schema (single SpikeSortingSelection with all FKs inline) makes source-orphans structurally impossible, since DataJoint cascade-deletes the master when its FK targets vanish. The claim is correct on this point.

2. v2 code at `src/spyglass/spikesorting/v2/sorting.py:348-366` matches the claim verbatim: classmethod `prune_orphaned_selections(dry_run=True)` delegates to `find_orphaned_masters(cls, [cls.RecordingSource, cls.ConcatenatedRecordin

### sorting#48  [LOW | INT-JUST] v2 clusterless_thresholder strips fewer params than v1 (no whiten pop), relies on Pydantic extra='forbid' to catch the dead field
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:400-408`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1133-1167, _params/sorter.py:115-159`
- v1 behavior: v1 _run_spike_sorter at 401-404 pops 'tempdir', 'whiten', 'outputs' from sorter_params before detect_peaks. v1's accepted ANY dict and silently dropped these three keys.
- v2 behavior: v2 _run_clusterless_thresholder pops only 'outputs' and 'random_chunk_kwargs' (lines 1143-1144). The 'whiten' key is rejected at insert1 time because ClusterlessThresholderSchema (sorter.py:115-159) uses model_config = ConfigDict(extra='forbid'). v2 ALSO does not pop 'noise_levels' default; instead handles the None case explicitly at 1146-1167.
- documented rationale: Pydantic guards eliminate the need for runtime pop. Defense-in-depth: invalid keys fail at SorterParameters.insert1 instead of at sort time.
- verifier reasoning: Verified directly against source. v1 sorting.py:400-408 pops `tempdir`, `whiten`, `outputs` (and renames `local_radius_um`->`radius_um`) before calling `detect_peaks`. v2 sorting.py:1133-1167 pops only `outputs` and `random_chunk_kwargs` (also renames `local_radius_um`). The missing `whiten`/`tempdir` pops are safe because `ClusterlessThresholderSchema` at _params/sorter.py:115-159 uses `model_config = ConfigDict(extra="forbid")` (line 152) and does not declare `whiten` or `tempdir` as fields — so those keys cannot enter the params blob via `SorterParameters.insert1`. The schema docstring (sor

### sorting#49  [LOW | INT-JUST] v2 hardcodes ('kilosort2_5', 'kilosort3', 'ironclust') as MATLAB sorters needing singularity_image=True; v1 did the same with the same tuple
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:439-450`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1062-1067, 1313-1319`
- v1 behavior: v1 at line 439: 'if sorter.lower() in ["kilosort2_5", "kilosort3", "ironclust"]' sets singularity_image=True and strips ('tempdir', 'mp_context', 'max_threads_per_process') from sorter_params.
- v2 behavior: v2 at 1062-1067 defines _MATLAB_SORTERS = ('kilosort2_5', 'kilosort3', 'ironclust') and _MATLAB_SORTER_STRIP_KWARGS = ('tempdir', 'mp_context', 'max_threads_per_process'). _run_si_sorter at 1313-1319 mirrors the v1 carve-out. Class-level constants are subclass-extensible; the v1 inline list was not.
- documented rationale: Pure refactor for extensibility; behavior preserved exactly.
- verifier reasoning: Verified by direct reading of cited lines. v1 sorting.py:439-450 contains the inline check `if sorter.lower() in ["kilosort2_5", "kilosort3", "ironclust"]` which sets `singularity_image=True` and strips `["tempdir", "mp_context", "max_threads_per_process"]` from sorter_params. v2 sorting.py:1062-1067 defines class-level constants `_MATLAB_SORTERS = ("kilosort2_5", "kilosort3", "ironclust")` and `_MATLAB_SORTER_STRIP_KWARGS = ("tempdir", "mp_context", "max_threads_per_process")` with identical contents. v2 sorting.py:1313-1319 mirrors the v1 logic: `if sorter.lower() in Sorting._MATLAB_SORTERS:

### sorting#50  [LOW | INT-JUST] v2 hardcodes float64 whitening AFTER artifact mask; v1 same order — preserved
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:391-397, 428-430`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:558-562, 1260-1285`
- v1 behavior: v1 _run_spike_sorter applies sip.remove_artifacts at 391-397, then sip.whiten(recording, dtype=np.float64) at 429. Order: mask -> whiten.
- v2 behavior: v2 make_compute calls _apply_artifact_mask at 558-562, then _run_sorter -> _run_si_sorter applies sip.whiten(recording, dtype=np.float64, seed=_random_seed) at 1282-1284. Order: mask -> whiten. Documented at 1226-1228 'Runs AFTER the upstream artifact mask was applied in Sorting.make_compute -- artifact-masked frames should not bias whitening's covariance estimate.'
- documented rationale: v2 docstring at sorting.py:1222-1228 acknowledges v1 parity with file:line.
- verifier reasoning: Verified directly. v1/sorting.py:391-397 applies sip.remove_artifacts; v1/sorting.py:428-430 then conditionally applies sip.whiten(recording, dtype=np.float64) when sorter_params['whiten'] is truthy, and disables sorter-internal whitening. v2/sorting.py:558-562 calls _apply_artifact_mask when artifact_id is set, and v2/sorting.py:1260-1285 then conditionally calls sip.whiten(recording, dtype=_np.float64, seed=_random_seed) with the same whiten=False handoff. Order mask -> whiten is preserved in both versions. The v2 docstring at sorting.py:1222-1228 explicitly cites the v1 line range and docum

### sorting#51  [LOW | NEW-V2] v2 introduces Sorting.Unit part-table with peak_amplitude_uv + Electrode FK; v1 has no equivalent
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:546-602 (writes only spike_times/id/obs_intervals/curation_label)`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:432-448 (schema), 1548-1648 (populate)`
- v1 behavior: v1 has no Unit part-table. The Units table inside the AnalysisNwbfile is the only per-unit storage; columns are spike_times, id (writer-assigned), obs_intervals, curation_label.
- v2 behavior: Sorting.Unit at 432-448 stores one row per unit with -> master FK, unit_id, -> Electrode (full per-unit channel FK), peak_amplitude_uv (float, peak template amplitude in microvolts), n_spikes. Populated by _populate_unit_part at 1548-1648 using template_tools.get_template_extremum_channel / get_template_extremum_amplitude on the SortingAnalyzer.
- documented rationale: Intentional new feature: per-unit brain region tracing and peak amplitude indexing. Consistent with v2's shared-contracts 'Unit-Level Brain Region Tracing' pattern.
- verifier reasoning: Verified all claims directly. v1 sorting.py contains only SpikeSorterParameters (83), SpikeSortingSelection (198), and SpikeSorting (233) classes with no Part tables. v1's _write_sorting_to_nwb at 546-602 writes only spike_times/id/obs_intervals/curation_label into the NWB Units table inside the AnalysisNwbfile (no per-unit DataJoint storage). v2 Sorting.Unit at 432-448 has the exact schema as claimed (-> master, unit_id, -> Electrode, peak_amplitude_uv float, n_spikes int). The docstring at 433-439 explicitly documents the brain region tracing pattern via Sorting.Unit * Electrode * BrainRegio

### sorting#52  [LOW | INT-JUST] v2 raises NonIntegerUnitIDError on non-int unit ids; v1 silently accepted whatever id pynwb gave back
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:594-597`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1610-1618, exceptions.py:33-36, 1526`
- v1 behavior: v1 _write_sorting_to_nwb passes 'id=unit_id' to nwbf.add_unit (line 596) without conversion; pynwb tolerates string/int/float id columns. spike_times_to_valid_samples at v1/sorting.py:33+ accepts arbitrary unit_id (only used in warning string).
- v2 behavior: v2 _write_units_nwb casts id=int(unit_id) (line 1526). _populate_unit_part at 1610-1618 wraps the int(unit_id) cast in try/except and raises NonIntegerUnitIDError with explicit message naming the offending id when the conversion fails. Sorting.Unit.unit_id column is typed 'int' so a non-int would fail there anyway, but the explicit raise gives a clean error class to catch.
- documented rationale: Defense-in-depth: v2's Sorting.Unit schema requires int unit_id, so an explicit raise gives a clean error before the framework-level IntegrityError.
- verifier reasoning: Verified all cited file:line locations directly. (1) v1/sorting.py:594-597 passes `id=unit_id` raw without int conversion; v1's spike_times_to_valid_samples (line 33+) only uses unit_id for the warning string at line 73 — confirmed v1 silently accepts whatever pynwb tolerates. (2) v2/sorting.py:1526 casts `id=int(unit_id)` in _write_units_nwb. (3) v2/sorting.py:1610-1618 wraps `int_unit_id = int(unit_id)` in try/except and raises NonIntegerUnitIDError with explicit message naming the offending id (uses f"...{unit_id!r}..."). (4) exceptions.py:33-36 documents NonIntegerUnitIDError as a ValueErr

### sorting#53  [LOW | INT-JUST] v2 stores time_of_sort as datetime; v1 stores as int Unix-time (schema-shape divergence)
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:234-240, 295`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:422-430, 676`
- v1 behavior: v1 SpikeSorting definition declares 'time_of_sort: int # in Unix time, to the nearest second' (line 239). make_compute writes 'time_of_sort = int(time.time())' (line 295).
- v2 behavior: v2 Sorting definition declares 'time_of_sort: datetime' (line 429). make_insert writes '_dt.datetime.now()' (line 676).
- documented rationale: Schema-shape decision documented at file-level docstring as 'zero-migration policy'. Compatible with downstream filtering tooling.
- verifier reasoning: Verified all four citations directly. v1 sorting.py:239 declares `time_of_sort: int               # in Unix time, to the nearest second` exactly as claimed. v1 sorting.py:295 writes `time_of_sort = int(time.time())`. v2 sorting.py:429 declares `time_of_sort: datetime`. v2 sorting.py:676 writes `"time_of_sort": _dt.datetime.now()`. The v2 file-level docstring (lines 1-16) explicitly invokes the "zero-migration policy" with all tables "final-shape" — providing the rationale that this is an intentional schema-shape change rather than drift. The baseline file tests/spikesorting/v2/baselines/legacy

## recording (21 findings)

### recording#1  [HIGH | INT-JUST] v1 `set_group_by_shank` silently deletes/re-inserts on rerun; v2 refuses by default and requires explicit opt-in
- **v1**: `src/spyglass/spikesorting/v1/recording.py:81-95`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:127-228`
- v1 behavior: If existing rows for nwb_file_name, `(SortGroup & ...).delete()` and re-insert without warning (lines 81-95).
- v2 behavior: `_handle_existing` (lines 150-228) refuses by default. Caller must either pass explicit non-overlapping sort_group_ids (additive) or `delete_existing_entries=True, confirm=True` after viewing `preview_existing_entries`.
- documented rationale: Documented in SortGroupV2 class docstring lines 99-112
- verifier reasoning: Verified all three claims against the actual code.

v1 (src/spyglass/spikesorting/v1/recording.py:81-95) confirms silent delete-and-reinsert:
```
existing_entries = SortGroup & {"nwb_file_name": nwb_file_name}
if existing_entries and test_mode:
    return
elif existing_entries:
    # delete any current groups
    (SortGroup & {"nwb_file_name": nwb_file_name}).delete()
```
No warning, no preview, no confirmation. The only escape is `test_mode=True`, which silently no-ops.

v2 (src/spyglass/spikesorting/v2/recording.py:150-228) confirms the inspect-before-destroy contract in `_handle_existing`:


### recording#2  [HIGH | INT-UNJ] v1 `set_group_by_shank` supports per-electrode-group references; v2 only allows a single reference for all sort groups
- **v1**: `src/spyglass/spikesorting/v1/recording.py:50-95; src/spyglass/spikesorting/utils.py:76-93`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:233-273`
- v1 behavior: Accepts `references: dict` keyed by electrode-group name (lines 50-95). Falls back to per-shank `electrodes['original_reference_electrode']` from the Electrode config (utils.py:84-93). Each shank's sort_reference_electrode_id can therefore differ.
- v2 behavior: Single int kwarg `sort_reference_electrode_id` applied to every sort group it creates (lines 233-273). Explicitly documents 'Per-group references are not supported in v2; if you need them, build the rows manually'. Never consults `original_reference_electrode`.
- documented rationale: no documented rationale beyond 'not supported'
- verifier reasoning: Verified the cited code in both versions.

v1 (src/spyglass/spikesorting/v1/recording.py:50-95): `set_group_by_shank(cls, nwb_file_name, references: dict = None, omit_ref_electrode_group=False, omit_unitrode=True)` — accepts a `references: dict` with the docstring saying "Keys: electrode groups. Values: reference electrode." It delegates to `get_group_by_shank` in utils.py.

v1 fallback (src/spyglass/spikesorting/utils.py:76-93): when `references` is None, the loop iterates per (electrode_group, shank) and reads `electrodes['original_reference_electrode'][match_names_bool]`, asserting the valu

### recording#3  [MEDIUM | MISSING] v1 `_validate_file` and `recompute`/`save_to` recompute-prior-to-deletion path missing in v2
- **v1**: `src/spyglass/spikesorting/v1/recording.py:269-404, 673-681`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1138-1180`
- v1 behavior: `_make_file(save_to=...)` writes a copy to a temp dir for hash-diff before delete, and `_validate_file` walks the H5 hierarchy to verify ProcessedElectricalSeries/electrodes are present (lines 269-404). Used by RecordingRecomputeSelection.
- v2 behavior: `_rebuild_nwb_artifact` rebuilds in the existing slot only (lines 1138-1180). No `save_to` parameter, no upstream-NWB-edit detection beyond a hash-mismatch warning. No `_validate_file`.
- documented rationale: Implicit per known v2 gaps; not individually documented in recording.py
- verifier reasoning: Verified the reviewer's claim by reading both files directly.

v1 src/spyglass/spikesorting/v1/recording.py:
- Line 269-367: `_make_file` signature includes `save_to: Union[str, Path] = None` (line 274). The `elif save_to:` branch at line 330-333 calls `cls._validate_file(file_path)` to verify the H5 hierarchy before allowing a recompute-prior-to-delete.
- Line 369-404: `_validate_file` walks `acquisition/ProcessedElectricalSeries/electrodes` H5 path and returns the electrodes object_id, raising FileNotFoundError or KeyError if any segment is missing.
- Line 673-681: `recompute` method itself 

### recording#4  [MEDIUM | INT-JUST] v1 `insert_selection` returns full row list on duplicates; v2 returns single PK-only dict consistently
- **v1**: `src/spyglass/spikesorting/v1/recording.py:162-182`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:672-725`
- v1 behavior: `if query: return query.fetch(as_dict=True)` — returns a list of full row dicts when an existing row matches (line 176-179). On insert, returns the modified input dict (with `recording_id` added).
- v2 behavior: Always returns single PK-only dict `{recording_id: ...}` (lines 698-725). Raises DuplicateSelectionError on multi-match (integrity bug).
- documented rationale: Documented under shared-contracts
- verifier reasoning: Verified all claims by reading the cited code directly.

v1 (src/spyglass/spikesorting/v1/recording.py:176-182):
```
query = cls & key
if query:
    logger.warning("Similar row(s) already inserted.")
    return query.fetch(as_dict=True)  # list[dict] of FULL rows
key["recording_id"] = uuid.uuid4()
cls.insert1(key, skip_duplicates=True)
return key  # input dict mutated with recording_id
```
Confirmed: returns list-of-full-row-dicts on duplicate, mutated input key on fresh insert. Inconsistent return type.

v2 (src/spyglass/spikesorting/v2/recording.py:672-725):
```
existing = (cls & keys_minus_

### recording#5  [MEDIUM | INT-UNJ] v1 `omit_ref_electrode_group` uses per-shank `original_reference_electrode`; v2 uses single passed-in `sort_reference_electrode_id`
- **v1**: `src/spyglass/spikesorting/utils.py:84-116`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:313-335`
- v1 behavior: For each shank, looks up the shank's actual reference electrode (`shank_elect_ref = electrodes['original_reference_electrode'][match_names_bool]`) and identifies which electrode group it belongs to; if that group equals the current group, the shank is omitted (utils.py:84-116).
- v2 behavior: Checks the passed-in `sort_reference_electrode_id` against the electrode table. If the param is -1 (default), `ref_match.any()` returns False and `omit_ref_electrode_group` is silently a no-op. Only the group containing the passed-in id is omitted (lines 313-335).
- documented rationale: no documented rationale; flag preserved as a kwarg but its semantics changed
- verifier reasoning: Verified at file:line. v1 utils.py:83-94 sets `sort_ref_id` from per-shank `original_reference_electrode` table column when no `references` dict is passed: `shank_elect_ref = electrodes["original_reference_electrode"][match_names_bool]`. Then v1 utils.py:97-116 computes `ref_elec_group` from that per-shank reference and skips the shank if `str(e_group) == str(ref_elec_group)`. v2 recording.py:313-335 takes a completely different path: it checks the SINGLE passed-in `sort_reference_electrode_id` kwarg (default -1 per v2 recording.py:238) against the electrode table via `ref_match = electrodes["

### recording#6  [MEDIUM | INT-UNJ] v1 inserts an IntervalList row for the intersected sort interval; v2 inserts no IntervalList row from Recording.make
- **v1**: `src/spyglass/spikesorting/v1/recording.py:185-260`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:990-1100 (make_insert)`
- v1 behavior: make_fetch/make_insert insert the sort_interval_valid_times into common.IntervalList under interval_list_name=recording_id (recording.py:212-216, 255-257).
- v2 behavior: Recording.make has no IntervalList insert; the saved range is recorded only on the Recording row (saved_start, saved_end, duration_s). No `recording_id`-keyed IntervalList row exists for downstream queries.
- documented rationale: no documented rationale (recording.py module docstring at v2/recording.py:1-15 does not mention this drop)
- verifier reasoning: Verified the divergence holds in code.

V1 confirmed at src/spyglass/spikesorting/v1/recording.py:
- Lines 211-216: `sort_interval_valid_times.set_key(nwb_file_name=..., interval_list_name=key["recording_id"], pipeline="spikesorting_recording_v1")`
- Lines 255-257: `IntervalList.insert1(sort_interval_valid_times.as_dict, skip_duplicates=True)` inside `make_insert`.

V2 confirmed at src/spyglass/spikesorting/v2/recording.py:
- `make_insert` (lines 990-1100) only calls `AnalysisNwbfile().add(nwb_file_name, analysis_file_name)` and `self.insert1(...)` on the Recording row itself. No `IntervalList

### recording#7  [MEDIUM | INT-JUST] v1 silently selects gains[0] for heterogeneous channel gains; v2 raises ValueError
- **v1**: `src/spyglass/spikesorting/v1/recording.py:858`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1862-1870`
- v1 behavior: `conversion=np.unique(recording.get_channel_gains())[0] * 1e-6` (line 858) — silently picks the first unique gain even when multiple exist, mis-scaling channels.
- v2 behavior: Raises ValueError 'heterogeneous channel gains' when more than one unique gain is detected (lines 1862-1869).
- documented rationale: Documented as a deliberate correctness fix
- verifier reasoning: Verified all four pillars of the claim by reading the cited code directly:

1. v1/recording.py:858 contains exactly: `conversion=np.unique(recording.get_channel_gains())[0] * 1e-6`. This silently picks index [0] of the unique gains array with no length check — confirmed latent bug if a sort group ever contains heterogeneous gains.

2. v2/recording.py:1862-1870 contains:
```
gains = _np.unique(recording.get_channel_gains())
if len(gains) != 1:
    raise ValueError(
        "Recording.make: recording has heterogeneous channel "
        f"gains {gains.tolist()}; v2 ElectricalSeries write "
      

### recording#8  [MEDIUM | NEW-V2] v2 RecordingTruncatedError fires when saved duration < requested chunk total; v1 silently writes whatever survives
- **v1**: `src/spyglass/spikesorting/v1/recording.py:241-261`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:990-1068`
- v1 behavior: No coverage check between requested `valid_times` and saved data. A frame_slice that excludes data due to interval misalignment writes a short ElectricalSeries with no error.
- v2 behavior: make_insert computes `requested_total = sum(end - start for start, end in sort_valid_times)` vs `saved_total = saved_end - saved_start`. If `missing > 1.5/fs`, raises RecordingTruncatedError after unlinking the file (lines 1036-1068).
- documented rationale: Documented as #1585 fix
- verifier reasoning: Verified directly:

v1 src/spyglass/spikesorting/v1/recording.py:241-261 — `make_insert` does only `IntervalList.insert1(...)`, `AnalysisNwbfile().add(...)`, `self.insert1(...)`, `self._record_environment(...)`. No coverage comparison between requested `sort_interval_valid_times` and the saved data. Truncation/short-write would proceed silently.

v2 src/spyglass/spikesorting/v2/recording.py:1042-1068 — explicitly computes:
```
requested_total = float(sum(end - start for start, end in sort_valid_times))
saved_total = float(saved_end - saved_start)
tolerance = 1.5 / sampling_frequency
missing = 

### recording#9  [MEDIUM | NEW-V2] v2 always-on monotonicity repair of source timestamps; v1 has no equivalent
- **v1**: `src/spyglass/spikesorting/v1/recording.py:554`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1546-1636`
- v1 behavior: Calls `recording.get_times()` directly (line 554). No detection or repair of non-monotonic timestamps. Downstream `np.searchsorted` in `_consolidate_intervals` returns wrong indices on stitched epochs with floating-point artifacts.
- v2 behavior: `_repaired_timestamps` (lines 1546-1636) detects via chunked monotonicity check, repairs via shifted cummax, logs warning, and persists provenance (`timestamps_adjusted`, `n_adjusted_samples`).
- documented rationale: Documented as defensive port
- verifier reasoning: Verified all claims by reading the cited code.

v1 at src/spyglass/spikesorting/v1/recording.py:551-554:
```
recording = se.read_nwb_recording(nwb_file_abs_path, load_time_vector=True)
all_timestamps = recording.get_times()
```
No monotonicity check, no repair. A grep for `monoton|cummax|maximum.accumulate|adjusted` against v1/recording.py returns zero hits.

v2 at src/spyglass/spikesorting/v2/recording.py:1546-1636 implements `_repaired_timestamps` exactly as claimed: chunked monotonicity check (`_count_non_monotonic_chunked`), early-return fast path when `n_issues == 0`, shifted cummax repai

### recording#10  [MEDIUM | INT-JUST] v2 drops `electrodes_id` column and adds new metadata columns
- **v1**: `src/spyglass/spikesorting/v1/recording.py:185-195`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:794-806`
- v1 behavior: SpikeSortingRecording has `electrodes_id=null: varchar(40)` (separate object_id for the ProcessedElectricalSeries/electrodes sub-object) and `hash=null: varchar(32)` (line 193-194).
- v2 behavior: Recording schema (lines 794-806) has no `electrodes_id`; carries instead `electrical_series_path`, `n_channels`, `sampling_frequency`, `duration_s`, `cache_hash char(64)`, `timestamps_adjusted`, `n_adjusted_samples`. `object_id` is widened to varchar(72).
- documented rationale: Documented at v2/recording.py:771-792 (cache_hash matches shared-contracts.md; electrical_series_path commentary at line 1118-1123)
- verifier reasoning: Verified directly against source:

v1 (src/spyglass/spikesorting/v1/recording.py:185-195):
```
class SpikeSortingRecording(SpyglassMixin, dj.Computed):
    definition = """
    # Processed recording.
    -> SpikeSortingRecordingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the processed recording in NWB file
    electrodes_id=null: varchar(40) # Object ID for the processed electrodes
    hash=null: varchar(32) # Hash of the NWB file
    """
```

v2 (src/spyglass/spikesorting/v2/recording.py:794-806):
```
definition = """
-> RecordingSelection
---
-> Analys

### recording#11  [MEDIUM | UNTESTED] v2's `omit_ref_electrode_group=True` in set_group_by_shank has no test coverage
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:313-335`
- v1 behavior: Behavior tested only indirectly via downstream pipeline tests in v1.
- v2 behavior: Code path at lines 313-335 with non-trivial mismatch from v1 (see related finding).
- verifier reasoning: Verified the claim:

1. v2 code at `src/spyglass/spikesorting/v2/recording.py:313-335` exists exactly as the reviewer describes — a branch guarded by `if omit_ref_electrode_group:` that uses `sort_reference_electrode_id` (function-level kwarg, default -1) to identify which electrode group to skip:
```
if omit_ref_electrode_group:
    ref_match = electrodes["electrode_id"] == (
        sort_reference_electrode_id
    )
    if ref_match.any():
        ref_group = electrodes["electrode_group_name"][ref_match][0]
        if str(ref_group) == str(e_group):
            ... skipped.append({..., "reas

### recording#12  [MEDIUM | UNTESTED] v2's empty-intersection ValueError after min_segment_length filter has no test coverage
- **v1**: `src/spyglass/spikesorting/v1/recording.py:715-765`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1424-1436`
- v1 behavior: `_consolidate_intervals` crashes with IndexError on `intervals[:, 0]` if `valid_times` ends up empty (recording.py:715-765). No graceful handling.
- v2 behavior: Explicit ValueError if `len(valid_times) == 0` after intersection + min_segment_length filter (lines 1424-1436) directing user to lower the threshold or fix IntervalList.
- documented rationale: n/a — defensive guard
- verifier reasoning: Verified all three claims:

1. **v1 behavior (recording.py:715-765)**: `_consolidate_intervals` does no empty-check on `intervals`. Lines 727-731 check `shape[1] != 2`, but if `valid_times` is `[]`, `np.array([])` has shape `(0,)`, which fails the shape check via IndexError (no `shape[1]`); even after `.reshape(-1, 2)` giving shape `(0, 2)`, line 747 `start, stop = start_indices[0], stop_indices[0]` raises IndexError on the empty array. v1/recording.py:557-560 calls `_consolidate_intervals(valid_sort_times, all_timestamps)` directly without an empty-check. So v1 does crash ungracefully.

2. **

### recording#13  [LOW | INT-JUST] v1 `get_group_by_shank` crashes on non-numeric `electrode_group_name`; v2 normalizes
- **v1**: `src/spyglass/spikesorting/utils.py:50`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:75-87`
- v1 behavior: `e_groups.sort(key=int)` (utils.py:50) raises ValueError if any group name is non-decimal (e.g., 'probeA').
- v2 behavior: Uses `_electrode_group_sort_key` (lines 75-87) which routes non-decimal names to a secondary lexical sort.
- documented rationale: Documented in function docstring
- verifier reasoning: Verified all four claim elements directly against source:

1. v1 src/spyglass/spikesorting/utils.py:50 reads exactly `e_groups.sort(key=int)  # sort electrode groups numerically`. `int("probeA")` raises ValueError, so non-decimal `electrode_group_name` values crash `get_group_by_shank`.

2. v2 src/spyglass/spikesorting/v2/recording.py:75-87 defines `_electrode_group_sort_key(name)` with body `text = str(name); return (0, int(text)) if text.isdecimal() else (1, text)`. The docstring (lines 76-85) explicitly explains: numeric names sort numerically and ahead of non-numeric names; non-numeric nam

### recording#14  [LOW | NEW-V2] v1 default preprocessing preset has one row; v2 ships three (default_franklab, default_neuropixels, no_filter)
- **v1**: `src/spyglass/spikesorting/v1/recording.py:127-143`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:592-649`
- v1 behavior: `contents = [['default', {freq_min: 300, freq_max: 6000, margin_ms: 5, seed: 0, min_segment_length: 1}]]` — single row inserted by `insert_default` (lines 127-143).
- v2 behavior: `_DEFAULT_CONTENTS` contains three Pydantic-validated rows: `default_franklab` (whiten=None, defaults), `default_neuropixels` (300-6000, whiten=None), `no_filter` (1.0-14999.0, whiten=None). Each with `params_schema_version=2`.
- documented rationale: Documented for schema_version=2 / whiten=None alignment with sorter-side whitening
- verifier reasoning: Verified directly in code.

v1 at src/spyglass/spikesorting/v1/recording.py:127-138:
```
contents = [
    [
        "default",
        {
            "frequency_min": 300,
            "frequency_max": 6000,
            "margin_ms": 5,
            "seed": 0,
            "min_segment_length": 1,
        },
    ]
]
```
(Reviewer's summary said freq_min/freq_max; actual v1 uses frequency_min/frequency_max — minor inaccuracy but the structural claim of a single 'default' row is correct.)

v2 at src/spyglass/spikesorting/v2/recording.py:592-634 ships three rows in `_DEFAULT_CONTENTS`: 'default_frankl

### recording#15  [LOW | INT-JUST] v1 preprocessing param schema includes `margin_ms` and `seed`; v2 drops both fields entirely with `extra='forbid'`
- **v1**: `src/spyglass/spikesorting/v1/recording.py:127-138`
- **v2**: `src/spyglass/spikesorting/v2/_params/preprocessing.py:74-110`
- v1 behavior: Default preset has `margin_ms: 5` and `seed: 0` (lines 127-138) — declared in schema but never referenced in code (grep confirms no usage).
- v2 behavior: `PreprocessingParamsSchema` has no `seed`, no `margin_ms`. `ConfigDict(extra='forbid')` rejects any blob containing them (preprocessing.py:21,55,70,95).
- documented rationale: Implicit cleanup of dead v1 fields; not individually documented but consistent with v2 schema-version=2 cleanup
- verifier reasoning: Verified the divergence is real and the classification holds.

V1 (src/spyglass/spikesorting/v1/recording.py:127-138):
```
contents = [["default", {"frequency_min": 300, "frequency_max": 6000,
    "margin_ms": 5, "seed": 0, "min_segment_length": 1}]]
```
Confirmed `margin_ms` and `seed` are dead: grep finds them only in this schema row and the docstring (line 112). The v1 consumer `_get_preprocessed_recording` (line 540-626) only reads `filter_params["frequency_min"]` and `filter_params["frequency_max"]` — `margin_ms` and `seed` are never indexed.

V2 (src/spyglass/spikesorting/v2/_params/prep

### recording#16  [LOW | INT-JUST] v1 single existing-row path uses delete-then-recreate in test mode; v2 doesn't have a test-mode early exit
- **v1**: `src/spyglass/spikesorting/v1/recording.py:81-86`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:150-228`
- v1 behavior: `if existing_entries and test_mode: return` early-exits in test mode (line 82-83). Otherwise destroys and recreates.
- v2 behavior: No test_mode handling. Always enforces inspect-before-destroy contract regardless of test_mode.
- documented rationale: Implicit in correctness-first goal
- verifier reasoning: Verified both file locations cited by reviewer. v1 src/spyglass/spikesorting/v1/recording.py:81-86 reads: `existing_entries = SortGroup & {"nwb_file_name": nwb_file_name}; if existing_entries and test_mode: return; elif existing_entries: ... .delete()`. v1 also imports test_mode at line 22. v2 src/spyglass/spikesorting/v2/recording.py:150-228 defines `_handle_existing` with NO reference to test_mode anywhere in the file (grep returned empty). v2 instead enforces a uniform contract requiring `delete_existing_entries=True, confirm=True` after reviewing `preview_existing_entries`, or an explicit 

### recording#17  [LOW | UNCERTAIN] v1 supports `rounding`/`precision_lookup` knob on hash; v2 hardcodes default precision (4 decimals) with no user control
- **v1**: `src/spyglass/spikesorting/v1/recording.py:275-300, 352-357`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:571-590`
- v1 behavior: `_make_file` exposes `rounding: int = 4` parameter passed through to `AnalysisNwbfile().get_hash(..., precision_lookup=dict(ProcessedElectricalSeries=rounding))` (lines 275-300, 352-357).
- v2 behavior: `_hash_nwb_recording` calls `AnalysisNwbfile().get_hash(analysis_file_name)` with no precision_lookup, relying on the module default `PRECISION_LOOKUP = dict(ProcessedElectricalSeries=4)`.
- documented rationale: no documented rationale; v2 just simplifies the API
- verifier reasoning: Verified at the cited lines.

v1 (src/spyglass/spikesorting/v1/recording.py:270-300, 352-357): `_make_file` declares `rounding: int = 4` (line 275) with docstring at lines 297-300 explicitly documenting it as "Decimal places to round to when hashing. Default 4, which is typical for microvolt precision. Only used for hash computation, does not affect data written to NWB file." It is threaded into the hash call at line 355: `precision_lookup=dict(ProcessedElectricalSeries=rounding)`.

v2 (src/spyglass/spikesorting/v2/utils.py:571-590): `_hash_nwb_recording(analysis_file_name)` calls `AnalysisNwb

### recording#18  [LOW | INT-UNJ] v2 ElectricalSeries 'filtering' / 'description' strings differ from v1
- **v1**: `src/spyglass/spikesorting/v1/recording.py:840-857`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1895-1907`
- v1 behavior: `filtering='Bandpass filtered for spike band'`, `description='Referenced and filtered recording from {nwb_file_name} for spike sorting'`, table_region `description='Sort group'` (lines 840, 855-857).
- v2 behavior: `filtering='Bandpass filter + common reference'`, `description='Pre-motion preprocessed recording from {nwb_file_name} for spike sorting'`, table_region `description='Sort group electrodes'` (lines 1897, 1904-1907).
- documented rationale: no documented rationale; appears to reflect the new bandpass+CMR stage semantics
- verifier reasoning: Verified all three string divergences at the cited line numbers.

v1/recording.py:840-857 exactly contains:
- `description="Sort group"` (line 842)
- `filtering="Bandpass filtered for spike band"` (line 855)
- `description="Referenced and filtered recording from " + f"{nwb_file_name} for spike sorting"` (lines 856-857)

v2/recording.py:1895-1907 exactly contains:
- `description="Sort group electrodes"` (line 1897)
- `filtering="Bandpass filter + common reference"` (line 1904)
- `description=(f"Pre-motion preprocessed recording from " f"{nwb_file_name} for spike sorting")` (lines 1905-1907)

Te

### recording#19  [LOW | INT-JUST] v2 fixes a v1 off-by-one bug in `_consolidate_intervals` that silently dropped the last sample of every disjoint interval
- **v1**: `src/spyglass/spikesorting/v1/recording.py:741-744, used with frame_slice at recording.py:567-583`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:435-506`
- v1 behavior: `_consolidate_intervals` uses `stop_indices = np.searchsorted(timestamps, intervals[:,1], side='right') - 1`, returning an inclusive stop index, then passes it as `frame_slice(end_frame=stop)` which is EXCLUSIVE — dropping the last sample (~33 us at 30 kHz) of each disjoint chunk.
- v2 behavior: Uses `side='right'` directly as exclusive end (utils.py:480-488). Adjusts join condition to `next_start <= stop` (was `>= next_start - 1` in v1). The v2 docstring explicitly documents the v1 bug.
- documented rationale: Documented at utils.py:440-450
- verifier reasoning: Verified the claim by reading the cited code directly.

V1 (src/spyglass/spikesorting/v1/recording.py:741-744):
```
start_indices = np.searchsorted(timestamps, intervals[:, 0], side="left")
stop_indices = (
    np.searchsorted(timestamps, intervals[:, 1], side="right") - 1
)
```
V1 join condition at line 753: `if stop >= next_start - 1:` (consistent with inclusive stop).

V1 then uses those indices as `end_frame=` at recording.py:567-579 inside `frame_slice(...)` / array slicing `all_timestamps[start:stop]`. Python slicing and SpikeInterface's `frame_slice` both treat `end_frame` as EXCLUSIVE 

### recording#20  [LOW | INT-JUST] v2 reference-channel removal uses `remove_channels` AFTER common_reference; v1 uses `channel_slice` AFTER common_reference
- **v1**: `src/spyglass/spikesorting/v1/recording.py:597-606`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1745-1757`
- v1 behavior: After `common_reference(reference='single', ref_channel_ids=ref_channel_id)`, does `recording.channel_slice(channel_ids=recording_channel_ids)` (lines 597-606).
- v2 behavior: After `common_reference`, conditionally calls `recording.remove_channels([int(ref_channel_id)])` guarded by `if int(ref_channel_id) in [int(c) for c in recording.get_channel_ids()]` (lines 1745-1757).
- documented rationale: Documented SI API change
- verifier reasoning: Verified both cited code locations exactly match the reviewer's claims.

v1 src/spyglass/spikesorting/v1/recording.py:597-606:
```
if ref_channel_id >= 0:
    recording = si.preprocessing.common_reference(
        recording, reference="single", ref_channel_ids=ref_channel_id, dtype=np.float64,
    )
    recording = recording.channel_slice(channel_ids=recording_channel_ids)
```

v2 src/spyglass/spikesorting/v2/recording.py:1745-1757:
```
if ref_channel_id >= 0:
    recording = sip.common_reference(
        recording, reference="single", ref_channel_ids=[int(ref_channel_id)], dtype=_np.float64,


### recording#21  [LOW | UNTESTED] v2's invalid `sort_reference_electrode_id` error path lacks test coverage
- **v1**: `src/spyglass/spikesorting/v1/recording.py:614-619`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1765-1770`
- v1 behavior: Raises ValueError on `ref_channel_id` not in {-1, -2, >=0} (lines 614-619). No v1 test invokes this path either.
- v2 behavior: Raises ValueError at `_apply_pre_motion_preprocessing` lines 1765-1770 on the same condition.
- documented rationale: n/a — error path port
- verifier reasoning: Verified both cited code locations directly.

v1 src/spyglass/spikesorting/v1/recording.py:614-619:
```
elif ref_channel_id != -1:
    raise ValueError(
        "Invalid reference channel ID. Use -1 to skip referencing. Use "
        + "-2 to reference via global median. Use positive integer to "
        + "reference to a specific channel."
    )
```

v2 src/spyglass/spikesorting/v2/recording.py:1765-1770 (inside `_apply_pre_motion_preprocessing`):
```
elif ref_channel_id != -1:
    raise ValueError(
        "Recording.make: invalid sort_reference_electrode_id "
        f"{ref_channel_id}. Use

## artifact (16 findings)

### artifact#1  [HIGH | INT-UNJ] v2 algorithm rewritten in-memory; drops v1's chunked parallel execution via ChunkRecordingExecutor
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:278-308; src/spyglass/spikesorting/utils.py:141-205`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:870-925; v2/artifact.py:866-869 (in-memory comment)`
- v1 behavior: v1 uses spikeinterface.core.job_tools.ChunkRecordingExecutor with init_func=_init_artifact_worker and func=_compute_artifact_chunk for parallel chunked detection over disk-backed segments; honors n_jobs / chunk_duration / progress_bar from artifact_params (v1/artifact.py:278-308; utils.py:141-205).
- v2 behavior: v2 _detect_artifacts calls recording.get_traces(return_in_uV=False) ONCE into an in-memory float32 array, then scales by gains; the comment at v2/artifact.py:866-869 admits 'In-memory scan: acceptable for recordings up to a few minutes; ... chunked iteration is follow-up work alongside the recompute pipeline.' Job_kwargs schema column is wired in but only used by _resolved_job_kwargs() side effect (v2/artifact.py:712-721).
- documented rationale: Comment cites 'follow-up work alongside the recompute pipeline' but no design doc or commit explains why the chunked path was removed before the recompute work landed.
- verifier reasoning: Verified all claims against the actual code.

**v1 chunked execution confirmed**: v1/artifact.py:278-308 uses `ChunkRecordingExecutor(recording=recording, func=_compute_artifact_chunk, init_func=_init_artifact_worker, init_args=init_args, ..., **job_kwargs)` with `executor.run()` followed by `np.concatenate`. Worker funcs at utils.py:141-205 use `recording.get_traces(segment_index=..., start_frame=..., end_frame=...)` for chunked reads, and honor `n_jobs` per-row via `ensure_n_jobs(recording, n_jobs=job_kwargs.get("n_jobs", 1))` at v1/artifact.py:278.

**v2 in-memory rewrite confirmed**: v2/ar

### artifact#2  [HIGH | INT-JUST] v2 changes IntervalList.interval_list_name format: 'artifact_<uuid>' vs v1's bare '<uuid>'
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:200`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:807, 1033; src/spyglass/spikesorting/v2/utils.py:509-533`
- v1 behavior: Writes IntervalList row with interval_list_name = str(key['artifact_id']) (v1/artifact.py:200). Bare UUID string.
- v2 behavior: Writes IntervalList row(s) with interval_list_name = f'artifact_{artifact_id}' via artifact_interval_list_name helper (v2/artifact.py:807; v2/utils.py:512-519).
- documented rationale: Documented in module docstring; the prefix is also the marker used by the v2 merge-table artifact dispatcher to distinguish artifact intervals from session intervals.
- verifier reasoning: Verified all cited lines exactly as the reviewer described.

v1/artifact.py:200 — `interval_list_name=str(key["artifact_id"])` writes a bare UUID string to IntervalList (pipeline="spikesorting_artifact_v1").

v2/artifact.py:807 — `interval_list_name = artifact_interval_list_name(key["artifact_id"])`; followed at line 825 by `IntervalList.insert(interval_rows)` with `pipeline="spikesorting_artifact_v2"`. v2/artifact.py:1033 — same helper is used on the read path in `get_artifact_removed_intervals`, confirming the prefixed name is the canonical v2 form, not a one-off.

v2/utils.py:509-533 — defi

### artifact#3  [HIGH | INT-JUST] v2 default amplitude threshold dropped from 3000 (v1) to 500 uV
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:65-77 (contents 'default' = 3000); src/spyglass/spikesorting/utils.py:179 (np.abs(traces) > amplitude_thresh without gain scaling); src/spyglass/spikesorting/v1/recording.py:845 (return_scaled=False)`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:54 (Field(default=500.0)); src/spyglass/spikesorting/v2/artifact.py:874 (traces_uv = traces.astype(np.float32) * gains[None,:])`
- v1 behavior: ArtifactDetectionParameters 'default' row ships amplitude_thresh_uV=3000 (compared against RAW count traces in v1's _compute_artifact_chunk, since v1's NWB writer uses return_scaled=False so get_traces() returns ints, not uV).
- v2 behavior: ArtifactDetectionParamsSchema default amplitude_thresh_uV=500.0 (compared against gain-scaled traces_uv = traces.astype(float32)*gains, i.e. actual microvolts).
- documented rationale: Documented in src/spyglass/spikesorting/v2/_params/artifact_detection.py:44-52 and CHANGELOG; users with custom v1 thresholds must translate v1_value * probe_gain_uV_per_count to v2-equivalent uV.
- verifier reasoning: Verified all four citations directly:

1. v1 default = 3000: `src/spyglass/spikesorting/v1/artifact.py:70` contains `"amplitude_thresh_uV": 3000` in the 'default' contents row.

2. v1 raw-count comparison: `src/spyglass/spikesorting/utils.py:179` `above_a = np.abs(traces) > amplitude_thresh` — `traces` is the raw output of `recording.get_traces(...)` (line 171-175) with no gain scaling.

3. v1 writes raw ints: `src/spyglass/spikesorting/v1/recording.py:845` `recording=recording, return_scaled=False, buffer_gb=5` — confirms NWB writer stores int counts, so v1's 3000 threshold compares against c

### artifact#4  [MEDIUM | INT-JUST] v2 ArtifactDetection.delete cleans up IntervalList rows; v1 does not
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:132-207 (no delete override)`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:1076-1171`
- v1 behavior: v1 has no delete override; DataJoint's default delete will leave the IntervalList row (keyed by interval_list_name=str(artifact_id)) orphaned because IntervalList is not a child of ArtifactDetection via FK.
- v2 behavior: v2 ArtifactDetection.delete (v2/artifact.py:1076-1171) explicitly resolves source per row, collects (nwb_file_name, interval_list_name) pairs, calls super().delete first, then drops the matching IntervalList rows via cautious_delete (single-recording: one row; shared-group: one row per distinct member nwb).
- documented rationale: Documented inline; fixes a v1 hygiene gap (orphaned IntervalList rows after artifact delete).
- verifier reasoning: Verified directly. v1/artifact.py has NO `def delete` anywhere (grep across the file returns nothing); the ArtifactDetection class at lines 132-207 only defines `definition` and `make`. v2/artifact.py:1076 defines `def delete(self, *args, safemode=None, **kwargs)` with the docstring quoted in the claim: "DataJoint does not cascade through ``interval_list_name``-keyed dependencies, so the cleanup is explicit." The implementation matches the claim: it fetches rows first, resolves source kind ("recording" -> single nwb_file_name from RecordingSelection; "shared_artifact_group" -> distinct member 

### artifact#5  [MEDIUM | INT-JUST] v2 ArtifactSelection has source-part XOR pattern; v1 had a single FK to SpikeSortingRecording
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:96-129`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:363-578`
- v1 behavior: ArtifactDetectionSelection is a flat manual table with FKs to SpikeSortingRecording and ArtifactDetectionParameters (v1/artifact.py:96-104). insert_selection auto-generates artifact_id (uuid) and warns on duplicates.
- v2 behavior: ArtifactSelection is the master with `-> ArtifactDetectionParameters` and TWO mutually-exclusive source parts: RecordingSource and SharedArtifactGroupSource. insert_selection enforces XOR via a ValueError, finds existing master+source matches and raises DuplicateSelectionError on multi-match. Also includes resolve_source (raises SchemaBypassError on zero/multi source) and prune_orphaned_selections.
- documented rationale: Documented in module docstring and the source-part design notes.
- verifier reasoning: All claims verified against actual code.

V1 at src/spyglass/spikesorting/v1/artifact.py:96-129 matches the claim exactly: ArtifactDetectionSelection is dj.Manual with `artifact_id: uuid` PK and FKs to SpikeSortingRecording + ArtifactDetectionParameters. insert_selection warns "Similar row(s) already inserted." on duplicates and returns existing rows.

V2 at src/spyglass/spikesorting/v2/artifact.py:363-578 matches:
- Master `ArtifactSelection` has only `-> ArtifactDetectionParameters` in dependencies (lines 375-379)
- Two part tables: `RecordingSource` (FK to Recording, lines 381-386) and `Sha

### artifact#6  [MEDIUM | NEW-V2] v2 SharedArtifactGroup.insert_group loads every member recording for n_samples / dtype check at insert time
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:301-344`
- v1 behavior: n/a (no SharedArtifactGroup in v1)
- v2 behavior: insert_group calls Recording().get_recording({'recording_id': rid}) per member to compare exact get_num_samples()/get_dtype() before accepting the group (v2/artifact.py:311-344).
- documented rationale: Documented inline; defense-in-depth to surface aggregate_channels invariants at insert time rather than populate time.
- verifier reasoning: Verified the claim holds up exactly as described.

1. Code at v2/artifact.py:301-344 matches the claim:
   - Line 311-326: `for rid in member_recording_ids: rec_obj = Recording().get_recording({"recording_id": rid}) ... per_member_sizes[str(rid)] = (int(rec_obj.get_num_samples()), str(rec_obj.get_dtype()))`
   - Lines 327-337: raises ValueError on "differing exact n_samples"
   - Lines 338-344: raises ValueError on "differing dtypes"

2. Inline rationale at v2/artifact.py:302-310 is documented as claimed: "The earlier duration_s check (sampling_frequency * duration_s, with one-sample tolerance

### artifact#7  [MEDIUM | INT-JUST] v2 _detect_artifacts uses HALF-OPEN end boundary for artifact intervals; v1 uses CLOSED end via add_removal_window
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:322-329; src/spyglass/common/common_interval.py:925-930`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:966-976`
- v1 behavior: v1 builds artifact intervals via Interval(from_inds=True) (frame indices) then add_removal_window which adds half_win seconds to each interval edge and union_consolidates (common_interval.py:918-937). The end timestamp is `timestamps[end] + half_win` -- inclusive at the original last artifact sample. Then subtract from sort_interval_valid_times produces complement intervals.
- v2 behavior: v2 manually iterates the spans then sets `end_time = timestamps[min(end_f + 1, len(timestamps) - 1)]` for the artifact interval END (v2/artifact.py:973). The next valid interval starts at the FIRST POST-artifact sample, not at the last artifact sample.
- documented rationale: Documented inline; the half-open fix is part of the v2 correctness improvements (an explicit bug fix vs v1).
- verifier reasoning: Verified all cited locations directly.

v1 (src/spyglass/spikesorting/v1/artifact.py:322-329) constructs artifact intervals via `Interval(artifact_frames, from_inds=True).add_removal_window(removal_window_ms, valid_timestamps)`. The add_removal_window implementation (src/spyglass/common/common_interval.py:925-930) sets the END as `np.minimum(timestamps[end] + half_win, timestamps[-1])`, where `end` is the inclusive last artifact frame. With half_win=0 this is `timestamps[end_f]` — the artifact sample itself. After subtract (common_interval.py:899-908), the next valid interval starts at `curren

### artifact#8  [MEDIUM | NEW-V2] v2 introduces SharedArtifactGroup for cross-recording artifact detection (no v1 equivalent)
- **v1**: `n/a (feature absent in v1)`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:151-361 (SharedArtifactGroup + insert_group); :388-393, :449-454 (SharedArtifactGroupSource part); :639-679 (shared-group fetch path); :742-780 (aggregate_channels + scan)`
- v1 behavior: v1 ArtifactDetection takes ONE SpikeSortingRecording per detection row; no mechanism to union channels across sort groups of the same session.
- v2 behavior: v2 adds SharedArtifactGroup + SharedArtifactGroup.Member tables and an ArtifactSelection.SharedArtifactGroupSource part. SharedArtifactGroup.insert_group enforces: non-empty members, all populated Recordings, single session, identical sampling_frequency, identical n_samples, identical dtype. make_compute uses si.aggregate_channels to column-stack the per-member recordings and runs ONE detection pass; make_insert writes one IntervalList row per distinct member nwb_file_name (today always 1 because of single-session enforcement).
- documented rationale: Documented in module docstring referencing Spyglass issue #928.
- verifier reasoning: All cited code matches the reviewer's description.

(1) v2/artifact.py:151-176 defines `SharedArtifactGroup` (Manual) + `Member` part with FK to `Recording`, with docstring citing "Spyglass issue #928 (behavioral artifacts visible on every probe -- chewing, licking, head-bumps)".

(2) `insert_group` (v2/artifact.py:178-360) enforces:
  - Non-empty members (line 217-221: "members list is empty")
  - All recording_ids reference populated Recording rows (line 232-243: "missing" check)
  - Single session (line 275-283: `len(sessions) != 1` raises)
  - Identical sampling_frequency (line 290-300)
  

### artifact#9  [MEDIUM | INT-JUST] v2 introduces explicit detect=bool field; v1 inferred from both-thresholds-None
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:79-87 ('none' contents, no detect key); src/spyglass/spikesorting/v1/artifact.py:256-264`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:53; src/spyglass/spikesorting/v2/artifact.py:846-851`
- v1 behavior: No 'detect' field; _get_artifact_times treats `amplitude_thresh_uV is zscore_thresh is None` as the skip condition and returns the full window (v1/artifact.py:256-264).
- v2 behavior: Explicit 'detect: bool = True' field. ArtifactDetection._detect_artifacts short-circuits on `if not validated.detect`. The 'none' preset is detect=False with amplitude_thresh_uV=None.
- documented rationale: v2 promotes the inferred semantics to an explicit boolean for schema typing; documented in v2 schema docstring.
- verifier reasoning: Verified all four cited lines.

v1/artifact.py:79-87 — 'none' preset contents confirmed: `{"zscore_thresh": None, "amplitude_thresh_uV": None, "chunk_duration": "10s", "n_jobs": 4, "progress_bar": "True"}`. No 'detect' key.

v1/artifact.py:256-264 — skip logic confirmed: `if amplitude_thresh_uV is zscore_thresh is None: ... return np.asarray([valid_timestamps[0], valid_timestamps[-1]]), np.asarray([])`.

v2/_params/artifact_detection.py:53 — `detect: bool = True` field present. Lines 18-21 docstring states: "``detect`` (so the ``"none"`` preset is a typed ``detect=False`` rather than an absent

### artifact#10  [LOW | INT-UNJ] v2 adds join_window_ms (default 1.0); v1 has no equivalent
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:321-324; src/spyglass/common/common_interval.py:508-526, 918-937`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:70; src/spyglass/spikesorting/v2/artifact.py:944-960`
- v1 behavior: v1 uses Interval(from_inds=True) which merges only frames adjacent by exactly 1 (v1/common_interval.py:508-526). Then add_removal_window expands each interval by removal_window_ms/2 on each side and union_consolidates the overlapping results. Effectively the merge window equals removal_window_ms (default 1ms = ~30 frames at 30kHz).
- v2 behavior: v2 introduces an explicit join_window_ms=1.0 field; two artifact frames within join_window_frames are folded into one span BEFORE the removal-window expansion (v2/artifact.py:944-960).
- documented rationale: no documented rationale beyond the 'merged into one' field description
- verifier reasoning: Verified the claim against code:

v2 (src/spyglass/spikesorting/v2/artifact.py:940-960):
```
fs = recording.get_sampling_frequency()
half_window_frames = int(_np.ceil(validated.removal_window_ms * 1e-3 * fs / 2))
join_window_frames = int(_np.ceil(validated.join_window_ms * 1e-3 * fs))
...
for f in frames_above[1:]:
    if f - cur_end <= join_window_frames:
        cur_end = f
    else:
        spans.append((cur_start, cur_end))
        ...
```
Then spans are expanded by half_window_frames (lines 963-965). So v2 joins in FRAMES first, then expands.

v1 (src/spyglass/spikesorting/v1/artifact.py:

### artifact#11  [LOW | INT-JUST] v2 adds min_length_s=1.0 to the schema but v1 hardcodes min_length=1 in subtract()
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:327-328`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:71; src/spyglass/spikesorting/v2/artifact.py:998-1003`
- v1 behavior: v1 calls `sort_interval_valid_times.subtract(artifact_intervals_s, min_length=1)` with a hardcoded literal (v1/artifact.py:327-328). Slivers shorter than 1 second are dropped.
- v2 behavior: min_length_s is a tunable Pydantic field defaulting to 1.0 (matches v1) and filters valid_times in _detect_artifacts (v2/artifact.py:998-1003).
- documented rationale: Promotes the v1 magic literal to a tunable parameter while preserving the default; documented in v2 schema and comments.
- verifier reasoning: Verified all three cited locations directly. v1/artifact.py:327-328 contains `sort_interval_valid_times.subtract(artifact_intervals_s, min_length=1).union_consolidate()` — hardcoded literal as claimed. v2/_params/artifact_detection.py:71 defines `min_length_s: float = Field(default=1.0, gt=0.0)`. v2/artifact.py:990-1003 contains the filter `[start, end] for start, end in kept if (end - start) >= validated.min_length_s` with an inline comment that explicitly cites `src/spyglass/spikesorting/v1/artifact.py:327-328` and explains the rationale (avoid millisecond slivers, prevent SI sorter crashes)

### artifact#12  [LOW | INT-JUST] v2 drops _check_artifact_thresholds clipping; relies on Pydantic field validators instead
- **v1**: `src/spyglass/spikesorting/utils.py:208-257`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:54-71`
- v1 behavior: _check_artifact_thresholds (utils.py:208-257) RAISES on negative thresholds but only WARNS on proportion_above_thresh > 1 or < 0, silently clipping it to 0.01 or 1. So a v1 caller passing proportion=2.0 gets a warning + clipped value, not an error.
- v2 behavior: Pydantic field `proportion_above_thresh: float = Field(default=1.0, gt=0.0, le=1.0)` (v2/_params/artifact_detection.py:68) RAISES ValidationError for out-of-range values at schema construction. amplitude_thresh_uV / zscore_thresh / removal_window_ms / min_length_s have ge=0 or gt=0 constraints.
- documented rationale: Part of v2's schema-first design; cleaner failure mode than v1's silent clipping.
- verifier reasoning: Verified all cited code directly.

V1 (src/spyglass/spikesorting/utils.py:208-257) implements _check_artifact_thresholds exactly as described: it RAISES ValueError on negative amplitude/zscore thresholds (lines 236-240: `if t < 0: raise ValueError("Amplitude and Z-Score thresholds must be >= 0, or None")`) but only WARNS and silently clips proportion_above_thresh to 0.01 (lines 243-249) or 1 (lines 250-256). So v1 callers passing proportion=2.0 get a warning + clipped value, not an exception.

V2 (src/spyglass/spikesorting/v2/_params/artifact_detection.py:54-71) replaces this with Pydantic Fie

### artifact#13  [LOW | INT-JUST] v2 drops verbose / logging knobs from the schema; reverts to fixed logger.info
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:217, 257-261`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:847-864`
- v1 behavior: v1 _get_artifact_times has `verbose: bool = False` parameter that gates the 'skipping artifact detection' info log (v1/artifact.py:217, 257-261). progress_bar='True' (note: string) in the default contents is passed through to ChunkRecordingExecutor.
- v2 behavior: v2 _detect_artifacts always logs at info level when entering the detect=False short-circuit (v2/artifact.py:847-850) and always logs the threshold configuration before scanning (v2/artifact.py:856-864). No verbose flag.
- documented rationale: Documented in v2 schema docstring; promotes scientific-vs-UI separation.
- verifier reasoning: Verified all cited lines against actual code.

v1/artifact.py:217: `verbose: bool = False` parameter exists in `_get_artifact_times`.
v1/artifact.py:257-261: The "skipping artifact detection" log is gated by `if verbose:` block:
```
if amplitude_thresh_uV is zscore_thresh is None:
    if verbose:
        logger.info(
            "Amplitude and zscore thresholds are both None, "
            + "skipping artifact detection"
        )
```
v1/artifact.py:75, 85: Default contents include `"progress_bar": "True"` (string, confirmed bug).

v2/artifact.py:846-851: `_detect_artifacts` unconditionally lo

### artifact#14  [LOW | INT-JUST] v2 pipeline tag changes from 'spikesorting_artifact_v1' to 'spikesorting_artifact_v2'
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:202`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:818`
- v1 behavior: Writes IntervalList row with pipeline='spikesorting_artifact_v1' (v1/artifact.py:202).
- v2 behavior: Writes pipeline='spikesorting_artifact_v2' (v2/artifact.py:818).
- documented rationale: no documented rationale (trivially obvious)
- verifier reasoning: Verified both citations directly. v1/artifact.py:202 writes `pipeline="spikesorting_artifact_v1"` inside `IntervalList.insert1(...)`. v2/artifact.py:818 writes `"pipeline": "spikesorting_artifact_v2"` inside the `interval_rows` list comprehension that is then bulk-inserted into `IntervalList`. Divergence is real and exact as claimed. Grep confirms `"spikesorting_artifact_v2"` appears only in v2/artifact.py and in tests/spikesorting/v2/test_single_session_pipeline.py — including an explicit assertion at line 6361: `assert rows[0]["pipeline"] == "spikesorting_artifact_v2"`. So the new tag is tes

### artifact#15  [LOW | INT-JUST] v2 returns empty (0, 2) ndarray when all valid intervals are filtered out; v1 returns the full recording window
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:313-319`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:926-938 (empty-frames path) vs 1004 (filtered-empty path)`
- v1 behavior: When artifacts span the entire recording, v1's subtract(min_length=1) drops the empty result and returns an empty np.ndarray; but if min_length filters drop ALL slivers, v1 returns an empty `times` array (common_interval.py:911-914).
- v2 behavior: v2 explicitly returns `_np.empty((0, 2))` when min_length_s drops every kept interval (v2/artifact.py:1004). However, v2 also has an EARLIER path when frames_above is empty (v2/artifact.py:926-938) where it returns the full window. So 'detected nothing' vs 'detected so much there is no valid time left' produce DIFFERENT shapes (1x2 vs 0x2).
- documented rationale: no documented rationale; the asymmetry between 'detected nothing' (returns full window) and 'all-filtered' (returns empty) may be intentional but is unexplained
- verifier reasoning: The cited code lines are accurate. v1/artifact.py:313-319 returns `[[valid_timestamps[0], valid_timestamps[-1]]]` when `len(artifact_frames) == 0`. v2/artifact.py:926-938 returns `_np.asarray([[timestamps[0], timestamps[-1]]])` when `len(frames_above) == 0`. v2/artifact.py:1004 returns `_np.empty((0, 2))` when `kept` is empty (either naturally because artifacts span entire window, or because min_length_s filter dropped all slivers).

However, the reviewer's characterization of v1 behavior is INACCURATE in one important respect. The reviewer claims "v1 returns an empty `times` array" when min_l

### artifact#16  [LOW | INT-UNJ] v2 z-score adds +1e-12 to std; v1 uses scipy.stats.zscore directly
- **v1**: `src/spyglass/spikesorting/utils.py:185, 193`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:901-903`
- v1 behavior: v1 calls `np.abs(stats.zscore(traces, axis=1))` per chunk (utils.py:185, 193). scipy.stats.zscore returns NaN where the per-frame std is zero (silent channels).
- v2 behavior: v2 computes zscores manually with epsilon: `ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12; zscores = np.abs((traces_uv - ch_mean) / ch_std)` (v2/artifact.py:901-903).
- documented rationale: no documented rationale (likely defensive numerical stability)
- verifier reasoning: Verified at exact cited lines.

v1 src/spyglass/spikesorting/utils.py:185 and 193:
  `dataz = np.abs(stats.zscore(traces, axis=1))`
Uses scipy.stats.zscore directly. For rows where the per-frame across-channel std is exactly 0 (silent / DC-rail frames), scipy returns NaN, and NaN > threshold is False, so the frame is not flagged.

v2 src/spyglass/spikesorting/v2/artifact.py:901-903:
  `ch_mean = traces_uv.mean(axis=1, keepdims=True)`
  `ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12`
  `zscores = _np.abs((traces_uv - ch_mean) / ch_std)`
Manually computes the same per-frame across-channe

## curation (15 findings)

### curation#1  [HIGH | INT-JUST] v2 get_sort_group_info returns ALL electrodes; v1 returns only one electrode per sort_group via fetch(limit=1)
- **v1**: `src/spyglass/spikesorting/v1/curation.py:288-302`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:1193-1236`
- v1 behavior: v1 get_sort_group_info (v1/curation.py:288-302) iterates `for entry in table:` and does `electrode_restrict_list.extend(((SortGroup.SortGroupElectrode() & entry) * Electrode).fetch(limit=1))` — only the FIRST electrode per sort group is included, so a multi-region probe surfaces only one region per group.
- v2 behavior: v2 get_sort_group_info (v2/curation.py:1193-1236) returns (SortGroupV2.SortGroupElectrode & sg_restriction) * _Electrode * BrainRegion — every electrode in the sort group joined to its BrainRegion. Returns a DataJoint relation, not a DataFrame.
- documented rationale: Documented in v2 docstring as an explicit bugfix.
- verifier reasoning: Verified directly at the cited lines.

v1 (src/spyglass/spikesorting/v1/curation.py:282-302) builds `electrode_restrict_list` by iterating `for entry in table` and calling `((SortGroup.SortGroupElectrode() & entry) * Electrode).fetch(limit=1)` — pulling only the FIRST electrode per sort group. The final return `(cls & key).proj() * sort_group_info` is then effectively single-row-per-sort-group.

v2 (src/spyglass/spikesorting/v2/curation.py:1193-1236) returns `(SortGroupV2.SortGroupElectrode & sg_restriction) * _Electrode * BrainRegion` — every electrode in the sort group, joined to BrainRegion

### curation#2  [HIGH | MISSING] v2 insert_curation drops the metrics parameter entirely; v1 wrote per-metric NWB unit columns
- **v1**: `src/spyglass/spikesorting/v1/curation.py:51, 416-428`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:132-219, 821-887`
- v1 behavior: v1 CurationV1.insert_curation (v1/curation.py:51) accepts metrics: Union[None, Dict[str, Dict[str, float]]] = None. _write_sorting_to_nwb_with_curation (v1/curation.py:416-428) iterates metrics.items() and writes one add_unit_column(name=metric, data=metric_values) per metric into the units NWB so downstream consumers can read e.g. nwbf.units['snr'].
- v2 behavior: v2 CurationV2.insert_curation (v2/curation.py:132-219) has NO metrics keyword argument. Only metrics_source exists, which is a 3-value enum recording provenance with no actual per-unit metric values stored anywhere. The NWB units table written by _stage_curated_units_nwb carries only spike_times + (optionally) a curation_label ragged column.
- documented rationale: Implicit in v2 design: MetricCuration is a stub. But the v2/curation.py docstring (lines 1-17) does not call out the dropped metrics parameter, and v1 callers passing metrics=... will fail with TypeError at the signature level.
- verifier reasoning: All cited code matches the reviewer's claim exactly.

v1 (src/spyglass/spikesorting/v1/curation.py):
- Line 51: `metrics: Union[None, Dict[str, Dict[str, float]]] = None` is a real parameter in `CurationV1.insert_curation`.
- Line 105: `metrics=metrics` is forwarded to `_write_sorting_to_nwb_with_curation`.
- Lines 416-428: `if metrics is not None: for metric, metric_dict in metrics.items(): ... nwbf.add_unit_column(name=metric, description=metric, data=metric_values)` — confirms per-metric NWB columns are written so downstream code can read `nwbf.units['snr']`.

v2 (src/spyglass/spikesorting/

### curation#3  [HIGH | INT-JUST] v2 passes delta_time_ms=None to MergeUnitsSorting; v1 uses the SI default which silently drops same-sample duplicate spikes
- **v1**: `src/spyglass/spikesorting/v1/curation.py:262-264`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:1098-1109`
- v1 behavior: v1 get_merged_sorting (v1/curation.py:262-264) does sc.MergeUnitsSorting(parent_sorting=si_sorting, units_to_merge=units_to_merge) with no delta_time_ms kwarg, so SI uses its default. Per the SI 0.104 contract (and v2 probe test at test_single_session_pipeline.py:4974-5021), the SI default drops exact same-sample coincident spikes from different merged units. v1's lazy-merged train != np.concatenate(...) of contributors.
- v2 behavior: v2 get_merged_sorting (v2/curation.py:1107-1109) passes delta_time_ms=None explicitly, the only value that disables the duplicate check entirely. The v2 lazy merge therefore matches np.concatenate semantics and matches v2's own apply_merge=True staging at v2/curation.py:796-804.
- documented rationale: Explicit in code comments and probe test. The v2 semantic is the correct intended behavior (v1's was an implicit SI-default surprise).
- verifier reasoning: Verified every element of the claim by reading the actual code and library.

1. v1 location (src/spyglass/spikesorting/v1/curation.py:262-264) calls:
   `sc.MergeUnitsSorting(parent_sorting=si_sorting, units_to_merge=units_to_merge)` with no `delta_time_ms` kwarg — exactly as claimed.

2. v2 location (src/spyglass/spikesorting/v2/curation.py:1107-1109) calls:
   `sc.MergeUnitsSorting(base, units_to_merge=units_to_merge, delta_time_ms=None)` — explicitly passing None, as claimed. Comment at 1098-1106 documents the SI semantic and rationale.

3. SI library semantic confirmed at spikeinterface/cu

### curation#4  [MEDIUM | INT-JUST] v2 dispatches sample-frame conversion through NwbSortingExtractor; v1 uses bespoke spike_times_to_valid_samples
- **v1**: `src/spyglass/spikesorting/v1/curation.py:213-225; src/spyglass/spikesorting/v1/sorting.py:29-70`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:982-988`
- v1 behavior: v1 get_sorting (v1/curation.py:213-225) calls spike_times_to_valid_samples(recording_times, unit.spike_times, n_samples, unit.Index) per-unit, which guards a documented searchsorted boundary bug at the last sample (v1/sorting.py:39-50): a spike time exactly equal to recording_times[-1] would otherwise map to index n_samples (out of bounds).
- v2 behavior: v2 get_sorting (v2/curation.py:986-988) constructs NwbSortingExtractor(file_path=abs_path, sampling_frequency=fs, t_start=t_start) which uses SI's internal sample conversion via t_start + sample/fs round-trip. The boundary-spike behavior depends on SI's NwbSortingExtractor implementation, not on spike_times_to_valid_samples.
- documented rationale: v2 moves to SortingAnalyzer API per the stated goal; documented in the get_sorting docstring.
- verifier reasoning: All cited code lines verified.

v1 (src/spyglass/spikesorting/v1/curation.py:211-225) calls per-unit:
  units_dict = {unit.Index: spike_times_to_valid_samples(recording_times, unit.spike_times, n_samples, unit.Index) for unit in units.itertuples()}
  return si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=sampling_frequency)

The helper at v1/sorting.py:29-79 documents the boundary bug guard. Its docstring says: "floating-point rounding in the seconds-to-samples round-trip can cause np.searchsorted to return an index equal to n_samples ... SpikeInterface rejects such a sorting w

### curation#5  [MEDIUM | INT-JUST] v2 get_sorting(as_dataframe=True) returns 2-column DataFrame; v1 returns the full NWB units DataFrame
- **v1**: `src/spyglass/spikesorting/v1/curation.py:197-209`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:990-1026`
- v1 behavior: v1 (v1/curation.py:197-209) opens the units NWB and returns nwbf.units.to_dataframe() directly — carries spike_times, curation_label, merge_groups, AND every per-metric column added at insert time (e.g. snr, isi_violation).
- v2 behavior: v2 (v2/curation.py:990-1026) builds a custom DataFrame with only spike_times and curation_label (joined from CurationV2.UnitLabel). No merge_groups column (it is in DataJoint, not the NWB), no metrics columns (none stored), no other NWB unit columns.
- documented rationale: Partially documented. The metric-columns gap follows from the missing-MetricCuration gap; the merge_groups gap is documented in the MergeGroup part-table docstring.
- verifier reasoning: Verified all claims directly against source.

v1 (src/spyglass/spikesorting/v1/curation.py:197-209):
```
analysis_file_name = (CurationV1 & key).fetch1("analysis_file_name")
...
with pynwb.NWBHDF5IO(analysis_file_abs_path, "r", load_namespaces=True) as io:
    nwbf = io.read()
    units = nwbf.units.to_dataframe()
if as_dataframe:
    return units
```
v1 returns the full NWB units DataFrame. Confirmed at v1/curation.py:390-428 that v1 writes `curation_label` (index=True), `merge_groups` (index=True), AND one column per metric (e.g. snr, isi_violation) to the NWB units table at insert_curation 

### curation#6  [MEDIUM | INT-JUST] v2 idempotency raises on non-default args; v1 silently ignores them
- **v1**: `src/spyglass/spikesorting/v1/curation.py:88-93`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:265-302`
- v1 behavior: v1 (v1/curation.py:88-93) for parent_curation_id == -1 with an existing root: logs 'Sorting has already been inserted.' and returns the existing key via parent_query.fetch('KEY'). Any new labels, merge_groups, metrics, description passed by the caller are silently dropped with no error.
- v2 behavior: v2 (v2/curation.py:272-302) detects the existing root, then if the caller passed truthy labels, merge_groups, or description AND reuse_existing is False (the default), raises ValueError telling them to pass reuse_existing=True or use parent_curation_id=<existing>. Only with reuse_existing=True does it silently return the existing key.
- documented rationale: Documented in code comments and tested.
- verifier reasoning: Verified all claims directly against the code.

v1 (src/spyglass/spikesorting/v1/curation.py:85-93):
```
sort_query = cls & {"sorting_id": sorting_id}
parent_curation_id = max(parent_curation_id, -1)
parent_query = sort_query & {"curation_id": parent_curation_id}
if parent_curation_id == -1 and len(parent_query):
    # check to see if this sorting with a parent of -1
    # has already been inserted and if so, warn the user
    logger.warning("Sorting has already been inserted.")
    return parent_query.fetch("KEY")
```
Confirms v1 silently ignores any caller-provided labels/merge_groups/metric

### curation#7  [MEDIUM | UNTESTED] v2 lazy-merge id assignment can diverge from applied-merge ids when user input order differs from kept-uid-ascending order
- **v1**: `src/spyglass/spikesorting/v1/curation.py:359-367`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:175-179, 590-598, 1041-1056, 1107-1109`
- v1 behavior: v1 only ever applies merges synchronously at insert; the v1 lazy path via get_merged_sorting reads merge_groups from NWB and reconstructs via _merge_dict_to_list which already loses input order. v1's applied-merge ids follow user-provided order via `for merge_group in merge_groups: new_unit_id = max(...) + 1` (v1/curation.py:359-367).
- v2 behavior: v2 applied path (apply_merge=True) assigns fresh merged ids in USER-PROVIDED group order (v2/curation.py:599-602, 652-658). v2 lazy path (get_merged_sorting on apply_merge=False) reads MergeGroup rows ordered by (unit_id, contributor_unit_id) (v2/curation.py:1051-1056), so the lazy path iterates by kept-uid-ascending. When user-given group order differs from kept-uid order, applied and lazy paths assign the same fresh ids to different content groups.
- documented rationale: Acknowledged in docstring as a known divergence between v2's own two paths (applied vs lazy). v1 parity is preserved only for the applied path.
- verifier reasoning: Verified all four cited locations:

1. v1 applied path at /cumulus/edeno/spyglass/src/spyglass/spikesorting/v1/curation.py:359-367 iterates merge groups in user-given order and assigns `np.max(list(units_dict.keys())) + 1` per group — confirmed.

2. v2 applied path at /cumulus/edeno/spyglass/src/spyglass/spikesorting/v2/curation.py:599-602, 650-658: `normalized_groups` preserves user order; `next_merged_id = max(by_id) + 1` and `for int_group in normalized_groups: if apply_merge and len(int_group) > 1: key = next_merged_id; next_merged_id += 1` — confirmed user-order assignment.

3. v2 lazy pa

### curation#8  [MEDIUM | INT-JUST] v2 omits the merge_groups NWB unit column that v1 wrote (with [""] sentinel)
- **v1**: `src/spyglass/spikesorting/v1/curation.py:404-415, 258-264`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:101-130, 821-887, 1068-1109`
- v1 behavior: v1 _write_sorting_to_nwb_with_curation (v1/curation.py:404-415) builds a per-unit merge-group list via _list_to_merge_dict and writes it as add_unit_column(name='merge_groups', data=merge_groups_list, index=True) where empty groups become [""] to satisfy pynwb's dtype inference. Downstream get_merged_sorting (v1/curation.py:258-264) reads this column with nwb_sorting.get('merge_groups') and reconstructs the merges via _merge_dict_to_list.
- v2 behavior: v2 _stage_curated_units_nwb (v2/curation.py:821-887) writes ONLY curation_label. Merge provenance lives in CurationV2.MergeGroup DataJoint part rows (v2/curation.py:101-130) instead of an NWB column. v2 get_merged_sorting (v2/curation.py:1068-1109) reads from CurationV2.MergeGroup via get_merge_groups, not from the NWB file.
- documented rationale: Explicit: v2 schema preferred queryable provenance over NWB-side encoding.
- verifier reasoning: Verified all four cited code regions directly.

v1 writes merge_groups NWB column (v1/curation.py:404-415):
```
if merge_groups is not None:
    merge_groups_dict = _list_to_merge_dict(merge_groups, unit_ids)
    merge_groups_list = [
        [""] if value == [] else value
        for value in merge_groups_dict.values()
    ]
    nwbf.add_unit_column(
        name="merge_groups",
        description="merge groups",
        data=merge_groups_list,
        index=True,
    )
```

v1 reads it (v1/curation.py:258-264):
```
nwb_sorting = nwbfile.objects[curation_key["object_id"]]
merge_groups = nwb_

### curation#9  [MEDIUM | INT-JUST] v2 raises on truly-stray label unit_ids; v1 silently ignored them
- **v1**: `src/spyglass/spikesorting/v1/curation.py:391-403`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:333-372`
- v1 behavior: v1 (v1/curation.py:391-403) iterates only unit_ids (final post-merge ids) and does `if unit_id not in labels: append([])`. Any keys in labels that aren't in unit_ids are silently never visited — no error, no log. A label keyed on a typo unit (999) just disappears.
- v2 behavior: v2 (v2/curation.py:346-364) computes truly_stray = label_keys - (source_unit_ids | written_unit_ids). If non-empty AND permissive_labels=False (default), raises ValueError listing the stray ids. With permissive_labels=True, logs a warning and continues. Absorbed-source labels (label on a contributor merged away) are still silently dropped (only a warning is emitted) to preserve v1 parity.
- documented rationale: Documented in code comments and tests.
- verifier reasoning: Verified all claims by reading the actual code and tests.

v1 (src/spyglass/spikesorting/v1/curation.py:391-403): iterates only `unit_ids` (final post-merge ids), appending `[]` for any unit_id not in `labels`. Label keys not present in `unit_ids` are silently never visited:
```
if labels is not None:
    label_values = []
    for unit_id in unit_ids:
        if unit_id not in labels:
            label_values.append([])
        else:
            label_values.append(labels[unit_id])
```

v2 (src/spyglass/spikesorting/v2/curation.py:346-372): explicitly computes truly_stray and absorbed_label_id

### curation#10  [MEDIUM | INT-JUST] v2 rejects empty and singleton merge groups; v1 silently accepted them as no-op / max+1 rename
- **v1**: `src/spyglass/spikesorting/v1/curation.py:359-367`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:609-642`
- v1 behavior: v1 _write_sorting_to_nwb_with_curation (v1/curation.py:359-367) iterates `for merge_group in merge_groups:` with no shape validation. An empty group [] would crash on np.concatenate of nothing. A singleton group [k] silently 'renames' unit k to max(units_dict.keys())+1 then pops the original — a typo turns into a renumbering.
- v2 behavior: v2 _build_curated_unit_rows (v2/curation.py:609-642) raises ValueError on any group with len(int_group) < 2, AND on intra-group duplicates ([0,0]), AND on cross-group overlap, AND on references to unknown unit_ids. Validation runs BEFORE the zero-unit early-return at line 644.
- documented rationale: Documented in docstrings + recent commit messages.
- verifier reasoning: Verified all four claims by reading the cited code directly.

v1 at src/spyglass/spikesorting/v1/curation.py:359-367:
```
if apply_merge and units_dict:
    for merge_group in merge_groups:
        new_unit_id = np.max(list(units_dict.keys())) + 1
        units_dict[new_unit_id] = np.concatenate(
            [units_dict[merge_unit_id] for merge_unit_id in merge_group]
        )
        for merge_unit_id in merge_group:
            units_dict.pop(merge_unit_id, None)
    merge_groups = None
```
No shape validation. Singleton [k] silently renames k -> max+1. Empty [] would fail at np.concatenate

### curation#11  [LOW | INT-UNJ] v2 description column widened from varchar(100) to varchar(255)
- **v1**: `src/spyglass/spikesorting/v1/curation.py:40`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:64`
- v1 behavior: v1 schema (v1/curation.py:40): description: varchar(100) — descriptions longer than 100 chars are truncated by MySQL (or rejected, depending on sql_mode).
- v2 behavior: v2 schema (v2/curation.py:64): description: varchar(255) — accepts up to 255 chars.
- verifier reasoning: Divergence confirmed by direct inspection.

v1 (src/spyglass/spikesorting/v1/curation.py:40): `description: varchar(100)`
v2 (src/spyglass/spikesorting/v2/curation.py:63): `description: varchar(255)` — note reviewer cited :64, actual line is :63 (off-by-one within the same definition block); the divergence itself is real.

Context:
- The widening is documented in the v2 design plan: `.claude/docs/plans/spikesorting-v2/designs.md:705` lists `description: varchar(255)` as the final-shape declaration, so this is not accidental drift — it was prescribed.
- However, `precondition-check.md:47` expli

### curation#12  [LOW | INT-JUST] v2 enforces parent_curation_id existence; v1 silently coerces and accepts phantom parents
- **v1**: `src/spyglass/spikesorting/v1/curation.py:86-88`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:251-264`
- v1 behavior: v1 (v1/curation.py:86) does `parent_curation_id = max(parent_curation_id, -1)` — any negative input (-5, -2) is silently clamped to -1, then NO FK existence check is performed. A caller passing parent_curation_id=999 (with no row 999) gets the row written with that phantom parent.
- v2 behavior: v2 (v2/curation.py:251-264) explicitly checks `if parent_curation_id != -1: if not (cls & {sorting_id, curation_id: parent_curation_id}): raise ValueError`. No silent clamping; -2 / -5 / 999 would all attempt to look up that row, fail, and raise.
- documented rationale: no documented rationale for the divergence from v1's clamp behavior, though the v2 behavior is clearly more correct.
- verifier reasoning: Verified both citations are accurate.

v1 at src/spyglass/spikesorting/v1/curation.py:86 explicitly does `parent_curation_id = max(parent_curation_id, -1)` — silently clamps negatives to -1. After clamping it queries `parent_query = sort_query & {"curation_id": parent_curation_id}` (line 88) but only USES that query for the -1 idempotency short-circuit (line 89). For parent_curation_id > 0 (e.g., 999), the FK existence is never checked and the row is written with that phantom parent at lines 117-126.

v2 at src/spyglass/spikesorting/v2/curation.py:251-264 explicitly performs the existence chec

### curation#13  [LOW | UNCERTAIN] v2 master insert is NOT skip_duplicates; v1 is
- **v1**: `src/spyglass/spikesorting/v1/curation.py:126`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:464-479`
- v1 behavior: v1 (v1/curation.py:126) cls.insert1(key, skip_duplicates=True) — a second insert with the same PK (sorting_id, curation_id) returns silently with no row written.
- v2 behavior: v2 (v2/curation.py:464) cls.insert1(master_row) — no skip_duplicates. The merge-table insert at line 470-479 uses skip_duplicates=True, but the master+Unit+UnitLabel inserts use plain insert.
- documented rationale: No documented rationale; v2's idempotency check makes skip_duplicates effectively unnecessary.
- verifier reasoning: Both citations verified directly.

v1/curation.py:126 — `cls.insert1(key, skip_duplicates=True)` (confirmed at the exact line).

v2/curation.py:464 — `cls.insert1(master_row)` with no skip_duplicates (confirmed). The merge-table insert at v2/curation.py:478 does use `skip_duplicates=True`. The master + Unit (line 465) + UnitLabel (line 467) + MergeGroup (line 469) inserts are plain inserts.

The whole v2 sequence is wrapped in `with transaction_or_noop(cls.connection):` at v2/curation.py:459, so a duplicate-key error on the master insert would raise and roll back the entire transaction (includ

### curation#14  [LOW | NEW-V2] v2 master schema adds metrics_source enum column (no v1 equivalent)
- **v1**: `src/spyglass/spikesorting/v1/curation.py:29-41`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:54-64, 62, 304-313`
- v1 behavior: v1 schema (v1/curation.py:29-41) has columns: sorting_id, curation_id, parent_curation_id, analysis_file_name, object_id, merges_applied, description. No metrics-provenance column.
- v2 behavior: v2 schema (v2/curation.py:54-64) adds `metrics_source = 'manual': enum('manual', 'analyzer_curation', 'figpack')` between merges_applied and description.
- documented rationale: Module docstring (v2/curation.py:6-9) explains: 'metrics_source is restricted to true CurationV2 provenance values; external or ground-truth NWB Units continue to use ImportedSpikeSorting'. The two non-default values reserve space for future MetricCuration / figpack workflows.
- verifier reasoning: All claims verified by direct read:

v1 schema at src/spyglass/spikesorting/v1/curation.py:29-41 has:
```
-> SpikeSorting
curation_id=0: int
---
parent_curation_id=-1: int
-> AnalysisNwbfile
object_id: varchar(72)
merges_applied: bool
description: varchar(100)
```
No metrics_source column.

v2 schema at src/spyglass/spikesorting/v2/curation.py:54-64 has the same columns plus a new line 62: `metrics_source = 'manual': enum('manual', 'analyzer_curation', 'figpack')` positioned between merges_applied and description. (Also note: v2 widens description to varchar(255) and defaults merges_applied=0 

### curation#15  [LOW | INT-JUST] v2 promotes curation labels to an enforced enum; v1 had a docstring-only list
- **v1**: `src/spyglass/spikesorting/v1/curation.py:26, 391-403`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:249, 499-524; src/spyglass/spikesorting/v2/utils.py:50-66`
- v1 behavior: v1 (v1/curation.py:26) declares valid_labels = ['reject', 'noise', 'artifact', 'mua', 'accept'] as a module-level list but never enforces it in insert_curation or _write_sorting_to_nwb_with_curation. A typo like 'noize' is written into the NWB units column with no error.
- v2 behavior: v2 (v2/curation.py:249, 499-524 + utils.py:50-66 CurationLabel enum) calls _validate_labels before any insert; each label must be a CurationLabel member ('accept', 'mua', 'noise', 'artifact', 'reject') or its string value, otherwise ValueError is raised. Label set matches v1's 5 values exactly.
- documented rationale: Documented in CurationLabel docstring.
- verifier reasoning: Verified all cited file:line references directly.

v1/curation.py:26 - `valid_labels = ["reject", "noise", "artifact", "mua", "accept"]` exists as a module-level list. grep confirms `valid_labels` appears ONLY on line 26 in the entire v1/curation.py file - never referenced by insert_curation or _write_sorting_to_nwb_with_curation. v1/curation.py:391-403 writes labels directly to `nwbf.add_unit_column(name="curation_label", ..., data=label_values)` with no validation, so a typo like 'noize' would be silently persisted.

v2/curation.py:249 - `cls._validate_labels(labels)` is called before insert

## utils (14 findings)

### utils#1  [HIGH | INT-JUST] v1 _consolidate_intervals has an end-frame off-by-one + a sort bug; v2 fixes both but documents only the first
- **v1**: `src/spyglass/spikesorting/v1/recording.py:715-765 (off-by-one at 742-744; bad sort at 734-735)`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:435-506`
- v1 behavior: stop_indices uses `np.searchsorted(..., side="right") - 1`, producing an INCLUSIVE stop index that is then passed to SI's frame_slice whose end_frame is EXCLUSIVE -- silently drops the last sample of every disjoint interval (~33 us @ 30 kHz). Additionally, the unsorted-fallback path runs `intervals = np.sort(intervals, axis=0)` which sorts EACH COLUMN INDEPENDENTLY, scrambling the (start, stop) pairing.
- v2 behavior: Uses `np.searchsorted(..., side="right")` directly as the exclusive end (matches SI convention) AND uses `intervals[np.argsort(intervals[:, 0])]` (stable order by start), preserving pair integrity. Adjacency condition becomes `next_start <= stop` (vs v1's `stop >= next_start - 1`).
- documented rationale: v2 docstring (utils.py:442-450) documents the off-by-one fix; the columnwise sort fix is undocumented but clearly intentional.
- verifier reasoning: Verified all claims directly against source.

v1 off-by-one (recording.py:742-744):
```
stop_indices = (
    np.searchsorted(timestamps, intervals[:, 1], side="right") - 1
)
```
This produces an INCLUSIVE end index, but it is passed at recording.py:569 as `end_frame=interval_indices[1]` to SI's `frame_slice`, whose end_frame is EXCLUSIVE. Confirmed: drops the last sample of every disjoint interval.

v1 columnwise sort bug (recording.py:735):
```
intervals = np.sort(intervals, axis=0)
```
`np.sort(..., axis=0)` sorts each column independently, scrambling (start, stop) pairs. Confirmed; only hit

### utils#2  [HIGH | INT-UNJ] v2 reimplements artifact detection in-memory, abandoning v1's ChunkRecordingExecutor + worker helpers
- **v1**: `src/spyglass/spikesorting/utils.py:141-205; src/spyglass/spikesorting/v1/artifact.py:281-308 (call site)`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:828-925`
- v1 behavior: v1 ships `_init_artifact_worker` (utils.py:141-158) and `_compute_artifact_chunk` (utils.py:161-205) for SI's ChunkRecordingExecutor parallel scan; combined-threshold mode uses `np.logical_or(above_z, above_a)`.
- v2 behavior: v2 inlines the entire scan in `ArtifactDetection._detect_artifacts` (artifact.py:828-925) using full `recording.get_traces()` in memory, no chunking, no parallel workers. Comment at v2/artifact.py:866-868 acknowledges 'In-memory scan: acceptable for recordings up to a few minutes; ... Chunked iteration is follow-up work alongside the recompute pipeline.'
- documented rationale: v2 author acknowledges it as 'follow-up work' in the comment; no further justification given.
- verifier reasoning: All claims verified directly against the code.

v1 (src/spyglass/spikesorting/utils.py:141-205): `_init_artifact_worker` and `_compute_artifact_chunk` are SI ChunkRecordingExecutor worker functions. Combined-threshold mode uses `np.logical_or(above_z, above_a)` at line 198. v1/artifact.py:278-307 wires these into ChunkRecordingExecutor with `n_jobs` from `job_kwargs`, supporting parallel chunked scans.

v2 (src/spyglass/spikesorting/v2/artifact.py:828-925): `_detect_artifacts` is a single staticmethod. Line 870: `traces = recording.get_traces(return_in_uV=False)` loads the ENTIRE recording int

### utils#3  [MEDIUM | INT-JUST] v1's get_spiking_sorting_v1_merge_ids has duplicated/dead branch; v2's get_spiking_sorting_v2_merge_ids delegates to merge table
- **v1**: `src/spyglass/spikesorting/v1/utils.py:78-91`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:536-568`
- v1 behavior: v1 helper (utils.py:37-109) has obviously copy-pasted dead code: lines 79-91 contain `if ... else ...` where BOTH branches are identical (`SpikeSortingSelection() & sorting_restriction & rec_dict`). The 'sorted without artifact detection' branch is unreachable because both queries are textually the same. When curation_id is unspecified, returns latest via `np.max(...)` per sorting_id.
- v2 behavior: v2 helper (utils.py:536-568) is a thin wrapper over `SpikeSortingOutput()._get_restricted_merge_ids_v2(restriction, as_dict=as_dict)`. Adds an `as_dict` parameter (default False matching v1's plain-list return). v2's helper validates unknown keys (per docstring: 'Unknown keys raise ValueError').
- documented rationale: Documented in v2/utils.py:541-562 docstring.
- verifier reasoning: Verified directly in source.

v1 (src/spyglass/spikesorting/v1/utils.py:78-91):
```
        # if sorted with artifact detection
        if SpikeSortingSelection() & sorting_restriction & rec_dict:
            sorting_id_list.append(
                (
                    SpikeSortingSelection() & sorting_restriction & rec_dict
                ).fetch1("sorting_id")
            )
        # if sorted without artifact detection
        else:
            sorting_id_list.append(
                (
                    SpikeSortingSelection() & sorting_restriction & rec_dict
                ).fetch1("s

### utils#4  [MEDIUM | NEW-V2] v2 _assert_v2_db_safe is import-time guard against non-test DB; v1 has no such guard
- **v1**: `(none -- no v1 analog)`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:151-199`
- v1 behavior: v1 has no DB-host safety check; v1 schemas register against whatever `dj.config['database.host']` points to.
- v2 behavior: v2 declares `_assert_v2_db_safe()` (utils.py:151-199) that hard-fails import unless `database.host in {localhost, 127.0.0.1, ::1}` or env var `SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1` is set. Called at module-import time from every v2 schema module (recording.py:40, sorting.py, artifact.py:55, curation.py:35, session_group.py:32).
- documented rationale: Explicitly justified in utils.py:156-182 docstring as a temporary safety net for active v2 development.
- verifier reasoning: Verified all claims by direct read.

(1) v2 utils.py:151-199 declares `_SAFE_DB_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})`, `_OVERRIDE_ENV = "SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB"`, and `_assert_v2_db_safe()` which returns early if the env var is "1" or if `dj.config['database.host']` is in `_SAFE_DB_HOSTS`, else raises `RuntimeError`. Code matches the reviewer's summary exactly.

(2) Import-time calls verified at the cited locations:
- recording.py:40 _assert_v2_db_safe()
- sorting.py:71 _assert_v2_db_safe() (reviewer cited "sorting.py" without a line; confirmed present)
- ar

### utils#5  [MEDIUM | INT-UNJ] v2 drops v1's per-group reference-electrode handling in shank grouping
- **v1**: `src/spyglass/spikesorting/utils.py:75-94 (references dict / original_reference_electrode); src/spyglass/spikesorting/v1/recording.py:88 (call site)`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:232-273 (set_group_by_shank signature + docstring line 262-263)`
- v1 behavior: `get_group_by_shank` (utils.py:13-138) supports two reference-resolution paths: (a) caller-supplied `references` dict mapping each electrode_group to its ref electrode id; (b) when None, reads `original_reference_electrode` from each Electrode row and asserts all shank electrodes share the same ref. Result: each sort group can have its own `sort_reference_electrode_id`.
- v2 behavior: `SortGroupV2.set_group_by_shank` only accepts a single scalar `sort_reference_electrode_id: int = -1` applied UNIFORMLY to every sort group. The docstring states `Per-group references are not supported in v2; if you need them, build the rows manually.` No code reads `original_reference_electrode`.
- verifier reasoning: Verified the divergence holds.

v1 (src/spyglass/spikesorting/utils.py:13-138, called from src/spyglass/spikesorting/v1/recording.py:88-93):
- `get_group_by_shank` accepts `references: dict = None` keyed by electrode_group.
- If `references` provided, looks up per-group ref (utils.py:76-82).
- If None, reads `original_reference_electrode` from Electrode rows (utils.py:83-93) and asserts all shank electrodes share the same ref. Result: each sort group can carry its own `sort_reference_electrode_id` (utils.py:94).
- v1 SortGroup.set_group_by_shank signature includes `references: dict = None` (v1

### utils#6  [MEDIUM | NEW-V2] v2 transaction_or_noop wraps source-part inserts but v1 had no equivalent; new contract assumed by callers
- **v1**: `(none -- per-call autocommit)`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:19-33`
- v1 behavior: v1 had no atomicity helper for master+part inserts; individual `insert_selection` methods relied on DataJoint's default per-call autocommit.
- v2 behavior: v2 `transaction_or_noop` (utils.py:19-33) wraps master + part inserts in one transaction unless already in one. CurationV2.insert_curation has an enforced contract: tests/spikesorting/v2/test_v1_parity.py:785-881 statically (AST) verifies `insert_curation` uses `transaction_or_noop` and forbids certain inserts outside the block.
- documented rationale: Documented in utils.py:21-28.
- verifier reasoning: Verified all three pillars of the claim by reading the actual code:

1) v2 helper exists as described. `transaction_or_noop` is defined at /cumulus/edeno/spyglass/src/spyglass/spikesorting/v2/utils.py:19-33 with the documented behavior:
```
if connection.in_transaction:
    yield
else:
    with connection.transaction:
        yield
```
The docstring at utils.py:21-28 explicitly explains the nested-transaction motivation: "source-part inserts and curation inserts both want to wrap their master + part rows in one transaction; but the same helpers may be called from inside an existing populate ca

### utils#7  [LOW | INT-JUST] v2 CurationLabel enum vs v1 valid_labels list -- enforcement scope differs
- **v1**: `src/spyglass/spikesorting/v1/curation.py:26`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:50-66`
- v1 behavior: v1 declares `valid_labels = ['reject', 'noise', 'artifact', 'mua', 'accept']` as a module-level list at v1/curation.py:26. No automated enforcement at insert time; the list is documented in the `insert_curation` docstring.
- v2 behavior: v2 promotes the labels to a `CurationLabel(str, Enum)` enum (utils.py:50-66) with values `accept, mua, noise, artifact, reject`. Enforced at insert time inside `CurationV2.insert_curation` (curation.py:507-524). Docstring (utils.py:55-60) explicitly mentions free-form `dj.Manual.insert1` bypassing the helper remains permitted (DataJoint cannot enforce enums on varchar columns), and downstream filters fall back to the v1 list for unrecognized labels.
- documented rationale: Explicitly justified in utils.py:50-60 docstring.
- verifier reasoning: Verified all reviewer claims at the cited lines.

v1 (v1/curation.py:26): `valid_labels = ["reject", "noise", "artifact", "mua", "accept"]` — module-level list, no automated enforcement, only documented in docstrings. Confirmed by direct read.

v2 (v2/utils.py:50-66): `class CurationLabel(str, Enum)` with members `accept, mua, noise, artifact, reject`. Same five strings as v1. Docstring at lines 51-60 explicitly states: "Members match the v1 convention list... v2 promotes the list from a docstring to an enforced enum so a typo raises at insert time. Free-form ``dj.Manual.insert1`` calls bypass

### utils#8  [LOW | NEW-V2] v2 SourceResolution dataclass is new, gating dispatch in Sorting.make / ArtifactDetection.make
- **v1**: `(none -- no v1 analog)`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:127-144`
- v1 behavior: v1 has no source-part pattern; SpikeSortingSelection and ArtifactDetectionSelection FK upstream tables directly via plain columns, no dispatch.
- v2 behavior: v2 introduces a frozen dataclass `SourceResolution(kind, key)` (utils.py:127-144) with `kind` as a Literal of three source-part kinds (`recording`, `concatenated_recording`, `shared_artifact_group`). Used by `<Master>.resolve_source(key)` to dispatch which source-part backs a master.
- documented rationale: Docstring at utils.py:128-137 explicitly justifies the dataclass.
- verifier reasoning: Verified all claims:

1. **SourceResolution dataclass exists at v2/utils.py:127-144** as a frozen dataclass with the documented docstring (lines 129-137) explicitly stating its purpose: "Used at the top of Sorting.make() and ArtifactDetection.make() to dispatch on which source-part row backs the master." The `kind` field is exactly `Literal["recording", "concatenated_recording", "shared_artifact_group"]` and `key` is `dict`.

2. **v1 has no analog**: `grep -rn "SourceResolution\|resolve_source" v1/` returns no matches. v1's `SpikeSortingSelection` (v1/sorting.py:198-206) FKs `SpikeSortingRecor

### utils#9  [LOW | INT-JUST] v2 _check_artifact_thresholds equivalent is absent; validation lives in Pydantic schema
- **v1**: `src/spyglass/spikesorting/utils.py:208-257; src/spyglass/spikesorting/v1/artifact.py:267-275 (call site)`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py (no helper; relies on PydanticArtifactDetectionParamsSchema)`
- v1 behavior: v1 has dedicated `_check_artifact_thresholds` helper (utils.py:208-257) that (a) raises ValueError for negative thresholds and (b) WARNS and clamps `proportion_above_thresh` if out of [0, 1] (clamps <0 to 0.01 with warning, >1 to 1 with warning) -- a forgiving runtime check.
- v2 behavior: v2 has no equivalent helper. Validation happens implicitly via `ArtifactDetectionParamsSchema` (Pydantic) at insert time. If schema constraints (which I did not re-verify here) reject out-of-range values, the v1 warn-and-clamp behavior is lost -- callers get a hard error instead of a clamp.
- documented rationale: Implied by v2's overall schema-first design; not explicitly justified for this case.
- verifier reasoning: Verified the claim directly against code.

v1 helper exists at /cumulus/edeno/spyglass/src/spyglass/spikesorting/utils.py:208-257 with exact described behavior:
- Lines 236-240: raises ValueError if amplitude/zscore thresholds are negative
- Lines 243-249: WARNS and clamps proportion_above_thresh < 0 to 0.01
- Lines 250-256: WARNS and clamps proportion_above_thresh > 1 to 1

v1 call site confirmed at /cumulus/edeno/spyglass/src/spyglass/spikesorting/v1/artifact.py:267-275.

v2 has no _check_artifact_thresholds helper (grep on v2/artifact.py confirms). Validation lives in /cumulus/edeno/spyglas

### utils#10  [LOW | INT-JUST] v2 _get_recording_timestamps adds optional override parameter and changes dtype handling
- **v1**: `src/spyglass/spikesorting/utils.py:260-279`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:366-432`
- v1 behavior: `_get_recording_timestamps(recording)` -- single arg; uses `np.zeros((total_frames,))` (default float64 already) and direct integer indexing from numpy cumsum (no int() casts).
- v2 behavior: `_get_recording_timestamps(recording, override=None)` -- adds `override` kwarg that short-circuits to caller-supplied repaired-timestamp array. Forces `dtype=_np.float64` explicitly and `int()`-casts the index bounds.
- documented rationale: Documented in v2/utils.py:386-394 docstring.
- verifier reasoning: Verified at the cited locations.

v1 (src/spyglass/spikesorting/utils.py:260-279):
```
def _get_recording_timestamps(recording):
    ...
    cumsum_frames = np.cumsum(frames_per_segment)
    total_frames = np.sum(frames_per_segment)
    timestamps = np.zeros((total_frames,))
    for i in range(num_segments):
        start_index = cumsum_frames[i]
        end_index = cumsum_frames[i + 1]
        timestamps[start_index:end_index] = recording.get_times(segment_index=i)
    return timestamps
```
Single arg, default-dtype zeros, no int() casts.

v2 (src/spyglass/spikesorting/v2/utils.py:366-432):
`

### utils#11  [LOW | NEW-V2] v2 introduces MetricsSource enum with no v1 analog; figpack member unused (no figpack code yet)
- **v1**: `(none -- no v1 analog)`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:36-47; src/spyglass/spikesorting/v2/curation.py:141, 307-312`
- v1 behavior: v1 has no MetricsSource concept. The provenance of curation metrics is implicit (CurationV1 stores a single metrics blob; downstream MetricCuration / FigURLCuration are separate tables with no provenance enum).
- v2 behavior: v2 introduces `MetricsSource(str, Enum)` (utils.py:36-47) with members `manual, analyzer_curation, figpack`. Used in CurationV2 (curation.py:141, 307-312) to validate the `metrics_source` column. `figpack` member is declared but v2/figpack_curation.py is a 3-line stub -- no code produces figpack-sourced metrics yet.
- documented rationale: Docstring at utils.py:37-43 documents the design choice (typo -> ValueError instead of MySQL enum-mismatch); figpack member is forward-declared.
- verifier reasoning: Verified all cited locations directly:

1. utils.py:36-47 defines `MetricsSource(str, Enum)` with members `manual`, `analyzer_curation`, `figpack`. Docstring at lines 37-43 explicitly documents the rationale: "Promoted from a runtime set check so a typo at insert time raises a clear ``ValueError`` instead of a DataJoint enum-mismatch error from MySQL."

2. curation.py:141 uses it as the type annotation/default: `metrics_source: str | MetricsSource = "manual"`.

3. curation.py:307-312 validates via `MetricsSource(metrics_source).value` with a comment "Coerce metrics_source through the enum so a

### utils#12  [LOW | NEW-V2] v2 introduces _resolved_job_kwargs priority-merge with three sources; v1 just took raw job_kwargs
- **v1**: `src/spyglass/spikesorting/v1/sorting.py (passes job_kwargs directly to run_sorter / ChunkRecordingExecutor)`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:297-322`
- v1 behavior: v1 took `job_kwargs` dict directly from caller and passed straight into ChunkRecordingExecutor / SI compute (no SI-global merge, no dj.config merge).
- v2 behavior: v2 `_resolved_job_kwargs(*row_job_kwargs)` (utils.py:297-322) merges in increasing precedence: SI global defaults -> `dj.config['custom']['spikesorting_v2_job_kwargs']` -> per-row blobs in order given. Empty/None entries skipped.
- documented rationale: Documented at utils.py:298-303.
- verifier reasoning: Verified all four claims against actual code:

1. v2 code at src/spyglass/spikesorting/v2/utils.py:297-322 matches the claim exactly:
```
def _resolved_job_kwargs(*row_job_kwargs: dict | None) -> dict:
    ...
    merged = dict(si.get_global_job_kwargs())
    custom = dj.config.get("custom", {}) or {}
    merged.update(custom.get("spikesorting_v2_job_kwargs", {}) or {})
    for override in row_job_kwargs:
        if override:
            merged.update(override)
    return merged
```
Precedence: SI-global -> dj.config custom -> per-row blobs (in order). Empty/None entries skipped (`if override:

### utils#13  [LOW | MISSING] v2 omits _reformat_metrics helper used by v1 FigURLCuration
- **v1**: `src/spyglass/spikesorting/utils.py:282-294`
- **v2**: `src/spyglass/spikesorting/v2/figpack_curation.py (stub; no helper)`
- v1 behavior: v1's shared utils provides `_reformat_metrics` (utils.py:282-294) which reshapes a Dict[metric_name, Dict[unit_id, value]] into the sortingview-expected list-of-dicts format with name/label/tooltip/data fields. Used by `v1/figurl_curation.py:194`.
- v2 behavior: v2 has no equivalent helper. v2/figpack_curation.py is a 3-line stub (per the known-gaps list), so the absent helper is consistent with the absent feature.
- documented rationale: FigURLCuration is a known-gap feature; helper porting is deferred with the feature.
- verifier reasoning: Directly verified all claims:

1. v1 `_reformat_metrics` exists at src/spyglass/spikesorting/utils.py:282-294 exactly as described:
```python
def _reformat_metrics(metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
    return [
        {
            "name": metric_name,
            "label": metric_name,
            "tooltip": metric_name,
            "data": {
                str(unit_id): metric_value
                for unit_id, metric_value in metric.items()
            },
        }
        for metric_name, metric in metrics.items()
    ]
```
Signature, fields (name/label/tooltip/data), a

### utils#14  [LOW | NEW-V2] v2 unit_brain_region_df returns empty df with full schema; v1 has no analog
- **v1**: `(none)`
- **v2**: `src/spyglass/spikesorting/v2/utils.py:94-124`
- v1 behavior: (none -- no v1 analog at all; v1 attached brain regions inline)
- v2 behavior: v2 `unit_brain_region_df(unit_relation, resolution)` (utils.py:94-124) joins Unit x Electrode x BrainRegion, returns DataFrame with columns `[unit_id, electrode_id, region_name, subregion_name, subsubregion_name, region_resolution]`. Passes `columns=` explicitly to pd.DataFrame so empty result keeps full schema -- explicitly defends against the `pd.DataFrame([])` zero-column gotcha.
- documented rationale: Documented in utils.py:118-123 inline comment.
- verifier reasoning: Verified all claims against actual code.

v2 utils.py:94-124 matches the reviewer's description exactly. Key lines:
- Line 116-118: `joined = (unit_relation * _Electrode * BrainRegion).fetch(*columns, as_dict=True)`
- Lines 119-121: inline comment documents the rationale: "Pass ``columns=`` so an empty result still carries the full schema; ``pd.DataFrame([])`` would otherwise drop every column and leave callers a frame with only ``region_resolution``."
- Line 122: `df = pd.DataFrame(joined, columns=columns)` -- explicit columns kwarg
- Line 123: `df["region_resolution"] = resolution`

Both cal

## pipeline (13 findings)

### pipeline#1  [HIGH | INT-UNJ] v2 default sorter for clusterless preset is 'default' (100 uV threshold) which yields zero units on real data
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:167-182`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:77-83, src/spyglass/spikesorting/v2/sorting.py:161-182`
- v1 behavior: v1's clusterless preset for the populator is selected by `sorter_params_name`; the populator does not pin a clusterless variant. The shipped clusterless v1 row is 'default_clusterless' with detect_threshold=100.0 uV (v1/sorting.py:170).
- v2 behavior: The `franklab_tetrode_clusterless_thresholder` preset hard-codes sorter_params_name='default' (pipeline.py:81), which maps to the SorterParameters row at sorting.py:163 with `noise_levels=[1.0]` and detect_threshold=100. Comment at sorting.py:166-174 admits this is the v1 raw-amplitude reading. The unit-confusion note at test_single_session_pipeline.py:2528-2542 documents that the threshold is interpreted by SI's detect_peaks in the recording's native units (raw counts), not microvolts, because preprocessing does not gain-scale. The shipped 100 uV default reliably finds zero peaks on real data — exercised by test_run_v2_pipeline_clusterless_default_handles_zero_units_gracefully.
- documented rationale: Partial: sorting.py:166-174 documents the noise_levels choice mirrors v1 for raw-amplitude interpretation. But the preset's silent-no-output behavior is not flagged in the pipeline.py docstring.
- verifier reasoning: Verified all cited code:

1) pipeline.py:77-83 — confirmed. The preset `franklab_tetrode_clusterless_thresholder` hard-codes `sorter_params_name="default"`:
```
"franklab_tetrode_clusterless_thresholder": _Preset(
    preproc_params_name="default_franklab",
    artifact_params_name="default",
    sorter="clusterless_thresholder",
    sorter_params_name="default",
),
```

2) sorting.py:161-182 — confirmed. The 'default' row for clusterless_thresholder uses `noise_levels=[1.0]` with explicit comment that this `mirrors v1's default_clusterless... making the shipped detect_threshold=100 read as mi

### pipeline#2  [MEDIUM | INT-JUST] v2 hard-codes artifact_params_name='default' (500 uV threshold); v1 populator default was 'ampl_2000_prop_75'
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:84; src/spyglass/spikesorting/v1/artifact.py:65-87`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:64-83; src/spyglass/spikesorting/v2/_params/artifact_detection.py:44-54`
- v1 behavior: v1 populator's default `artifact_parameters='ampl_2000_prop_75'` (spikesorting_populator.py:84) is NOT one of the rows shipped by ArtifactDetectionParameters.insert_default (v1/artifact.py:65-87 ships only 'default'=3000 uV and 'none'). The populator's default thus relied on an externally-inserted parameter row.
- v2 behavior: v2 preset hard-codes artifact_params_name='default' (pipeline.py:67, 73, 79). The v2 'default' row schema-validates to amplitude_thresh_uV=500.0 (_params/artifact_detection.py:54).
- documented rationale: Documented as 'bug-fix default' in _params/artifact_detection.py:44; rationale beyond label not stated inline.
- verifier reasoning: All cited lines verified directly.

(1) v1 populator default at /cumulus/edeno/spyglass/src/spyglass/spikesorting/v0/spikesorting_populator.py:84: `artifact_parameters: str = "ampl_2000_prop_75"`. (The reviewer's path string said "v1" but the line number is for v0/spikesorting_populator.py — actual content matches exactly.)

(2) v1 `ArtifactDetectionParameters.insert_default` at /cumulus/edeno/spyglass/src/spyglass/spikesorting/v1/artifact.py:65-87 ships only two rows: `"default"` with `amplitude_thresh_uV: 3000` and `"none"` with `amplitude_thresh_uV: None`. The populator default `"ampl_2000_

### pipeline#3  [MEDIUM | MISSING] v2 pipeline has no FigURL / sortingview / interactive curation hook; v1 emits curation URLs
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:309-330`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py (entire file)`
- v1 behavior: If `fig_url_repo` is truthy, v1 populator builds a kachery/figurl URL per AutomaticCuration sort_group_id and populates CurationFigurl (spikesorting_populator.py:309-330).
- v2 behavior: No FigURL hook exists. The orchestrator returns only the merge_id (no URL). v2/figpack_curation.py is a 3-line stub.
- documented rationale: Documented as deferred — pipeline.py:6-9 says 'richer surfaces (metrics + auto-curation, concat sorts, cross-session matching, UI hooks) come in later versions'.
- verifier reasoning: Verified all claims directly against the cited code.

v1 at spikesorting_populator.py:309-330 contains exactly the claimed code: `if fig_url_repo:` block that builds `gh_url = fig_url_repo + nwb_file_name + "_" + sort_interval_name + "/{}/curation.json"`, iterates `(AutomaticCuration() & sort_dict).fetch("auto_curation_key")`, inserts into `CurationFigurlSelection` and calls `CurationFigurl.populate(auto_curation_out_key)`. The reviewer's v1 description is accurate.

v2 pipeline.py: the orchestrator's `run_v2_pipeline` returns only `{preset, recording_id, artifact_id, sorting_id, curation_id, 

### pipeline#4  [MEDIUM | MISSING] v2 pipeline lacks metric_curation / waveform / automatic curation pass entirely
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:249-307`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:248-269`
- v1 behavior: When `len(waveform_params_name) > 0 and len(metric_params_name) > 0 and len(auto_curation_params_name) > 0`, v1 populator chains WaveformSelection → Waveforms.populate → MetricSelection → QualityMetrics.populate → AutomaticCurationSelection → AutomaticCuration.populate → CuratedSpikeSortingSelection → CuratedSpikeSorting.populate. The else-branch creates an empty CuratedSpikeSortingSelection for clusterless decoding.
- v2 behavior: v2 pipeline goes straight from Sorting → CurationV2.insert_curation (root, empty labels) → merge row. No quality-metric step, no auto-label/merge pass.
- documented rationale: Documented: pipeline.py:6-9 lists 'metrics + auto-curation' as deferred.
- verifier reasoning: Verified all cited code directly.

v1 (v0/spikesorting_populator.py:249-307): Confirmed. Lines 250-293 chain WaveformSelection.insert -> Waveforms.populate -> MetricSelection.insert -> QualityMetrics.populate -> AutomaticCurationSelection.insert -> AutomaticCuration.populate -> CuratedSpikeSortingSelection.insert1 when all three *_params_name arguments are non-empty. Lines 295-304 (the else branch) just insert curation_keys directly into CuratedSpikeSortingSelection.

v2 (pipeline.py:248-269): Confirmed. After Sorting.populate (line 220), the code checks for an existing root curation (parent_c

### pipeline#5  [MEDIUM | INT-UNJ] v2 uses reserve_jobs=False on every populate; v1 populator uses DJ defaults (reserve_jobs=True under SpyglassMixin.populate)
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:219, 228, 241`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:202, 210, 220`
- v1 behavior: v1 populator calls `SpikeSortingRecording.populate(sort_dict)`, etc., without overriding reserve_jobs.
- v2 behavior: v2 explicitly passes `reserve_jobs=False` to every populate call (pipeline.py:202, 210, 220).
- documented rationale: no documented rationale.
- verifier reasoning: Verified all cited lines directly:

v2/pipeline.py:202: `Recording.populate(rec_pk, reserve_jobs=False)`
v2/pipeline.py:210: `ArtifactDetection.populate(art_pk, reserve_jobs=False)`
v2/pipeline.py:220: `Sorting.populate(sort_pk, reserve_jobs=False)`

v0/spikesorting_populator.py:219: `SpikeSortingRecording.populate(sort_dict)`
v0/spikesorting_populator.py:228: `ArtifactDetection.populate(sort_dict)`
v0/spikesorting_populator.py:241: `SpikeSorting.populate(sort_dict)`

Note the reviewer wrote "v1" but cited v0 — that is the correct comparator because there is no orchestrator in src/spyglass/spi

### pipeline#6  [LOW | UNTESTED] Idempotency contract: existing_root short-circuit relies on parent_curation_id=-1; if user manually inserted a non-root first, behavior differs from comment claim
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:245-247`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:254-258`
- v1 behavior: v1 populator uses `if not (Curation() & sorting_key): Curation.insert_curation(sorting_key)` (spikesorting_populator.py:245-247), so the first curation_id=0 row is what's reused.
- v2 behavior: v2 pipeline restricts by `{parent_curation_id: -1}` only (pipeline.py:255). If a user inserted a child curation as the first row (parent_curation_id=0 referencing a root that was deleted, an exotic but possible state), or if multiple roots exist (CurationV2 normally rejects this at curation.py:288-296, but a direct dj.insert1 could bypass), `existing_root[0]` picks the first match by DJ row order, which is not guaranteed deterministic.
- documented rationale: no documented rationale; the comment at pipeline.py:249-253 only states the happy path.
- verifier reasoning: Verified the cited code directly. At src/spyglass/spikesorting/v2/pipeline.py:254-258:

```
existing_root = (
    CurationV2 & sort_pk & {"parent_curation_id": -1}
).fetch("KEY", as_dict=True)
if existing_root:
    curation_pk = existing_root[0]
```

The fetch has no `order_by` clause, so if more than one root row exists for the same sorting, the chosen row depends on DataJoint's unspecified row order. CurationV2.insert_curation at curation.py:285-302 normally enforces a single-root invariant (raises if labels/merge_groups/description are passed AND a root already exists; otherwise reuses), bu

### pipeline#7  [LOW | INT-JUST] Three of v2's six default SorterParameters rows are not reachable via _PRESETS
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:86-89 (sorter / sorter_params_name args)`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:64-83`
- v1 behavior: v1 populator accepts arbitrary sorter / sorter_params_name strings as arguments, so any default SpikeSorterParameters row (mountainsort4 tetrode + cortex + every SI-installed sorter via `sis.available_sorters()`, sorting.py:185-189) can be selected.
- v2 behavior: v2 pipeline accepts only a `preset` string; the three presets cover ms4-tetrode + ms5-tetrode + clusterless. The MS4 cortex row 'franklab_probe_ctx_30kHz_ms4' (sorting.py:124-132), the KS4 row 'franklab_neuropixels_default' (sorting.py:141), the SC2/TDC2 rows (sorting.py:148-159) are unreachable through the orchestrator — users must bypass it.
- documented rationale: no documented rationale for why only the franklab tetrode subset got presets.
- verifier reasoning: Verified all cited code locations.

v2/pipeline.py:64-83 contains _PRESETS with exactly three entries (franklab_tetrode_mountainsort4, franklab_tetrode_mountainsort5, franklab_tetrode_clusterless_thresholder). At line 181-187, run_v2_pipeline raises PipelineInputError if preset is not in _PRESETS — there is no arbitrary-string fallback for the orchestrator.

v2/sorting.py:106-183 confirms six default SorterParameters rows: mountainsort4/franklab_tetrode_hippocampus_30kHz_ms4 (108-116), mountainsort4/franklab_probe_ctx_30kHz_ms4 (117-132), mountainsort5/franklab_tetrode_hippocampus_30kHz_ms5 (1

### pipeline#8  [LOW | NEW-V2] v2 `n_units` field is fetched and exposed in the manifest; v1 populator never inspected sort unit count
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py (n_units never inspected)`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:225-247, 278`
- v1 behavior: v1 populator does not check `len(units)` or n_units; CuratedSpikeSorting holds units silently in NWB and downstream callers query as needed.
- v2 behavior: v2 pipeline fetches Sorting.n_units (pipeline.py:225) and branches on n_units==0 to either raise ZeroUnitSortError (require_units=True) or emit a warning and continue (pipeline.py:226-247). The n_units field is included in the returned manifest.
- documented rationale: Documented in the pipeline.py:127-149 docstring under `require_units`.
- verifier reasoning: Verified all four claims directly:

1) v2 pipeline.py:225 reads `n_units = int((Sorting & sort_pk).fetch1("n_units"))`; lines 226-247 branch on `n_units == 0` with `ZeroUnitSortError` (line 229) when `require_units=True`, otherwise emits a `logger.warning(...)` (line 242) and falls through. Line 278 puts `"n_units": n_units` in the returned manifest.

2) Confirmed v0/spikesorting_populator.py (330 lines) and v1/ directory have no occurrences of `n_units`, `len(units)`, `require_units`, or `ZeroUnit*` — v1 populator never inspects sort unit count. (Note: reviewer cited v0/spikesorting_populator

### pipeline#9  [LOW | INT-JUST] v2 does not auto-create the SortGroup; v1 populator inserts via set_group_by_shank when absent
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:167-169`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:107-113`
- v1 behavior: v1 populator runs `SortGroup().set_group_by_shank(nwb_file_name)` when no SortGroup rows match the session (spikesorting_populator.py:167-169).
- v2 behavior: `run_v2_pipeline` requires the caller to have already inserted the SortGroupV2 row. Docstring at pipeline.py:107-113 says 'the orchestrator does not auto-create them because sort-group structure is session-specific user input'.
- documented rationale: Documented in the pipeline docstring at pipeline.py:107-113.
- verifier reasoning: Verified both citations directly.

v1 (spikesorting_populator.py:165-169):
```
# make sort groups only if not currently available
# don't overwrite existing ones!
if not SortGroup() & nwbf_dict:
    logger.info("Generating sort groups")
    SortGroup().set_group_by_shank(nwb_file_name)
```
v1's populator auto-creates SortGroup rows via `set_group_by_shank` when no rows match the session.

v2 (pipeline.py:86-113):
- `run_v2_pipeline` takes `sort_group_id: int` as a required positional parameter (line 88).
- The docstring at lines 107-113 explicitly states: "Callers create sort groups via `SortG

### pipeline#10  [LOW | INT-JUST] v2 hard-codes labels={} and parent_curation_id=-1 — no path to write labels/merge_groups through the orchestrator
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:243-294`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:260-265`
- v1 behavior: v1 populator funnels through Curation.insert_curation and then AutomaticCuration, which can attach labels/merge groups derived from quality metrics + thresholds.
- v2 behavior: v2 always calls `CurationV2.insert_curation(sorting_key=sort_pk, labels={}, parent_curation_id=-1, description=...)` (pipeline.py:260-265). No way to pass merge_groups or labels through `run_v2_pipeline`; the orchestrator only ever produces an empty root curation.
- documented rationale: Documented: 'metrics + auto-curation' are deferred (pipeline.py:6-9). The orchestrator is intentionally minimum-viable.
- verifier reasoning: Verified the v2 code at src/spyglass/spikesorting/v2/pipeline.py:260-265:

```
curation_pk = CurationV2.insert_curation(
    sorting_key=sort_pk,
    labels={},
    parent_curation_id=-1,
    description=description or f"run_v2_pipeline preset={preset}",
)
```

The orchestrator does hardcode `labels={}` and `parent_curation_id=-1` with no parameter to expose merge_groups or labels through `run_v2_pipeline`. The reviewer's claim is factually correct.

The v2 pipeline docstring (pipeline.py:6-9) explicitly documents this deferral: "The orchestrator focuses on the minimum-viable single-session pa

### pipeline#11  [LOW | INT-JUST] v2 pipeline lacks per-sort-group fan-out; v1 loops over sort_group_id_list
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py:201-241`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:86-220`
- v1 behavior: v1 populator computes a list of sort group ids via `(SortGroup.SortGroupElectrode * ElectrodeGroup) & nwbf_dict & probe_restriction).fetch('sort_group_id')` and inserts/populates Selection rows for every matching sort group in a session (recording, artifact, sorting) in one call. The session is fully sorted at the end.
- v2 behavior: `run_v2_pipeline(sort_group_id: int, ...)` accepts a single int and inserts exactly one chain. The caller is responsible for looping over shanks (as e.g. test_mountainsort5_ground_truth_polymer_60s does at tests/spikesorting/v2/test_single_session_pipeline.py:2606-2638). There is no `probe_restriction` analog.
- documented rationale: no documented rationale for dropping the per-shank fan-out; the docstring at pipeline.py:6-9 only says richer surfaces come in later versions.
- verifier reasoning: Verified the cited code. The divergence in API shape is real but the reviewer's framing conflates v0 with v1.

v0 populator (src/spyglass/spikesorting/v0/spikesorting_populator.py:201-205) does fan out over sort groups:
```
sort_group_id_list = (
    (SortGroup.SortGroupElectrode * ElectrodeGroup)
    & nwbf_dict
    & probe_restriction
).fetch("sort_group_id")
```
and loops over it (lines 209-217, 232-241) inserting one `SpikeSortingRecordingSelection` and one `SpikeSortingSelection` per shank in a single call.

v2 pipeline (src/spyglass/spikesorting/v2/pipeline.py:86-220) signature is `run_v

### pipeline#12  [LOW | INT-JUST] v2 returns n_units but the manifest has no analysis_file_name / curated NWB path; v1 used CuratedSpikeSorting fetch1 for that
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py (no return value)`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:271-279`
- v1 behavior: v1 populator does not return a manifest. Downstream consumers fetch CuratedSpikeSorting rows by sort_dict and pick up analysis_file_name + object_id from them.
- v2 behavior: v2 returns `{preset, recording_id, artifact_id, sorting_id, curation_id, merge_id, n_units}` (pipeline.py:271-279). No analysis_file_name. Consumers must fetch the curated NWB via SpikeSortingOutput merge dispatch using merge_id.
- documented rationale: Documented: SpikeSortingOutput unified-output design routes through merge_id, not per-table analysis files. The merge-table API is the documented public surface.
- verifier reasoning: Verified the claim against the actual code.

v2 manifest (pipeline.py:271-279):
```
return {
    "preset": preset,
    "recording_id": rec_pk["recording_id"],
    "artifact_id": art_pk["artifact_id"],
    "sorting_id": sort_pk["sorting_id"],
    "curation_id": curation_pk["curation_id"],
    "merge_id": merge_id,
    "n_units": n_units,
}
```
No `analysis_file_name`, no curated NWB object_id.

Predecessor pipeline: The reviewer cited `v0/spikesorting_populator.py` (calling it "v1"). I verified that file: `spikesorting_pipeline_populator()` (line 76) runs through `CuratedSpikeSorting.populate(s

### pipeline#13  [LOW | NEW-V2] v2 silently ignores `description` parameter on rerun when a root curation already exists
- **v1**: `src/spyglass/spikesorting/v0/spikesorting_populator.py (no equivalent)`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:254-265`
- v1 behavior: v1 populator does not pass a description through the chain; CuratedSpikeSorting has no free-text description field.
- v2 behavior: On first call, `run_v2_pipeline` passes `description or f"run_v2_pipeline preset={preset}"` to `CurationV2.insert_curation` (pipeline.py:260-264). On reruns, the existing_root short-circuit at pipeline.py:254-258 returns the existing curation_pk WITHOUT calling insert_curation. The same `description` argument that would have raised inside insert_curation (curation.py:285-296 forbids parameter changes on reuse without `reuse_existing=True`) is instead silently dropped.
- documented rationale: no documented rationale; the docstring at pipeline.py:96-100 promises idempotency without flagging this corner.
- verifier reasoning: Verified the claim against the actual code.

pipeline.py:254-265 reads exactly as claimed:
```
existing_root = (
    CurationV2 & sort_pk & {"parent_curation_id": -1}
).fetch("KEY", as_dict=True)
if existing_root:
    curation_pk = existing_root[0]
else:
    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        parent_curation_id=-1,
        description=description or f"run_v2_pipeline preset={preset}",
    )
```
On the reuse branch, the `description` argument is never consulted.

curation.py:285-296 confirms the guard the helper enforces:
```
if (
 

## session_group (13 findings)

### session_group#1  [MEDIUM | UNCERTAIN] ConcatenatedRecording lacks 'timestamps_adjusted' and 'n_adjusted_samples' columns present on single-session Recording
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:195-205`
- v1 behavior: No v1 equivalent.
- v2 behavior: Recording (v2/recording.py:804-805) carries 'timestamps_adjusted=0: bool' and 'n_adjusted_samples=0: int' tracking upstream timestamp repair. ConcatenatedRecording (v2/session_group.py:195-205) has no equivalent — a concat over a session whose Recording had timestamps repaired would silently lose this provenance.
- documented rationale: No documented rationale. Final-shape claim at v2/session_group.py:3-7 implies these columns will not be added later without a migration.
- verifier reasoning: Verified the line numbers and content against the actual source. v2/recording.py:804-805 declares `timestamps_adjusted=0: bool   # source recording's timestamps were repaired` and `n_adjusted_samples=0: int     # source-wide count of repaired samples`. v2/session_group.py:195-205 declares ConcatenatedRecording with `cache_hash: char(64)` and no equivalent repair-provenance columns. The omission is real.

The v2 author makes the omission load-bearing: v2/session_group.py:3-7 says "The schema is frozen; the populate body of ConcatenatedRecording is gated behind NotImplementedError until the conc

### session_group#2  [MEDIUM | NEW-V2] ConcatenatedRecordingSelection.insert_selection is a stub — Manual table is writable but no validation gate exists
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:171-181`
- v1 behavior: No v1 equivalent.
- v2 behavior: insert_selection raises NotImplementedError. The class docstring at v2/session_group.py:158-160 promises 'enforces that every member has a populated Recording row matching the requested preprocessing parameters', but only via the unimplemented helper. Because the table is dj.Manual a user can still call insert1() and create a ConcatenatedRecordingSelection row without any Recording-coverage check.
- documented rationale: Module docstring v2/session_group.py:1-7 — zero-migration policy. The schema is intentionally final-shape; consumer logic comes later.
- verifier reasoning: Verified the cited code directly:

1. `src/spyglass/spikesorting/v2/session_group.py:153` declares `class ConcatenatedRecordingSelection(SpyglassMixin, dj.Manual)` — confirmed it is a `dj.Manual`, so `insert1()` is writable by default DataJoint semantics.

2. Lines 171-181: `insert_selection` body is exactly `raise NotImplementedError("ConcatenatedRecordingSelection.insert_selection is not yet implemented")`. The docstring at lines 173-177 says "Forward-declared; the schema is in place so single-session tables can FK ``concat_recording_id`` from day one."

3. The class docstring at lines 154-1

### session_group#3  [MEDIUM | UNTESTED] Every NotImplementedError gate in session_group has zero test coverage
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:84-86, 94-96, 178-181, 223-225`
- v1 behavior: No v1 equivalent.
- v2 behavior: SessionGroup.create_group, SessionGroup.is_multi_day, ConcatenatedRecordingSelection.insert_selection, and ConcatenatedRecording.make all raise NotImplementedError. None of these gates is asserted by a test in tests/spikesorting/v2/. Only the parallel gate on SortingSelection.insert_selection (in sorting.py, not session_group.py) is tested at tests/spikesorting/v2/test_single_session_pipeline.py:5082-5103.
- documented rationale: No documented rationale. The gates would prevent the schema being mistakenly assumed live, but a future refactor could quietly remove a 'raise NotImplementedError' without tripping a test.
- verifier reasoning: Verified all four NotImplementedError gates at the exact cited line numbers in /cumulus/edeno/spyglass/src/spyglass/spikesorting/v2/session_group.py:

- Line 84-86: `SessionGroup.create_group` raises `NotImplementedError("SessionGroup.create_group is not yet implemented")`
- Line 94-96: `SessionGroup.is_multi_day` raises `NotImplementedError("SessionGroup.is_multi_day is not yet implemented")`
- Line 178-181: `ConcatenatedRecordingSelection.insert_selection` raises `NotImplementedError("ConcatenatedRecordingSelection.insert_selection is not yet implemented")`
- Line 223-225: `ConcatenatedRecor

### session_group#4  [MEDIUM | DRIFT] MotionCorrectionParameters.insert1 skips _assert_schema_version_matches — drift between outer params_schema_version and inner schema_version is undetected
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:139-144`
- v1 behavior: No v1 equivalent.
- v2 behavior: MotionCorrectionParameters.insert1 (v2/session_group.py:139-144) calls _validate_params on the blob but does NOT call _assert_schema_version_matches. PreprocessingParameters (v2/recording.py:636-644), ArtifactDetectionParameters (v2/artifact.py:132-142), and SorterParameters all call both helpers; MotionCorrectionParameters is the only v2 Lookup that omits the outer/inner version check.
- documented rationale: No documented rationale. Looks like an omission given the consistent pattern across the other three Lookups.
- verifier reasoning: Verified the claim directly:

1. session_group.py:139-144 — MotionCorrectionParameters.insert1 only calls _validate_params, NOT _assert_schema_version_matches:
```
def insert1(self, row, **kwargs):
    row = dict(row)
    row["params"] = _validate_params(
        MotionCorrectionParamsSchema, row["params"]
    )
    super().insert1(row, **kwargs)
```
The module's import line (session_group.py:29) only imports `_assert_v2_db_safe, _validate_params` from utils — not `_assert_schema_version_matches`.

2. The three peer Lookups DO call both helpers:
   - recording.py:636-644 (PreprocessingParamete

### session_group#5  [MEDIUM | UNTESTED] SessionGroup.Member has no uniqueness constraint on its content tuple; member_index is the only discriminator
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:58-67`
- v1 behavior: No v1 equivalent.
- v2 behavior: Member's PK is (session_group_owner, session_group_name, member_index). The secondary key fields (nwb_file_name, sort_group_id, interval_list_name, team_name) carry the actual content. Two rows with different member_index can carry identical content, and DataJoint will not catch the duplicate. The docstring (v2/session_group.py:39-44) defines a member as a '(nwb_file_name, sort_group_id, interval_list_name, team_name) tuple' but the schema does not enforce uniqueness on that tuple.
- documented rationale: No documented rationale. member_index ordering may be needed for concat boundary alignment, which would justify positional keying, but uniqueness still needs to be enforced at insert helper time.
- verifier reasoning: Verified all claims against v2/session_group.py:

1. Schema at lines 58-67 exactly matches reviewer's claim — Member's PK is (master fields + member_index) and the content tuple (Session, SortGroupV2, IntervalList, LabTeam) is in secondary keys. DataJoint enforces uniqueness only on PK columns, so two Member rows with identical secondary content but different member_index are permitted.

2. Docstring at lines 40-44 says: "A member is a ``(nwb_file_name, sort_group_id, interval_list_name, team_name)`` tuple" — this defines the intended identity, but the schema does not enforce it.

3. create_gr

### session_group#6  [MEDIUM | NEW-V2] SessionGroup.create_group is unimplemented; documented contract for multi-day + motion-correction enforcement is unenforced
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:69-86`
- v1 behavior: No v1 equivalent — there is no cross-session grouping table in v1.
- v2 behavior: SessionGroup.create_group raises NotImplementedError unconditionally. The class docstring at v2/session_group.py:46-49 explicitly promises 'multi-day requires allow_multi_day=True AND forces an explicit MotionCorrectionParameters row (see create_group)', but the helper that would enforce this is just a stub.
- documented rationale: Module docstring v2/session_group.py:1-7 cites a zero-migration policy: schema is final-shape, populate/insert bodies are gated until materializer lands. This is justified for downstream FK stability but leaves the user-facing invariant unenforced.
- verifier reasoning: Verified all claims against the actual code.

1. v2/session_group.py:69-86 contains `create_group` which raises `NotImplementedError("SessionGroup.create_group is not yet implemented")` unconditionally — matches reviewer claim exactly.

2. The class docstring at lines 46-49 reads: "Same-day groups are the default; multi-day requires ``allow_multi_day=True`` AND forces an explicit ``MotionCorrectionParameters`` row (see ``create_group``)." This contract is real and unenforced.

3. `is_multi_day` at lines 88-96 is also a stub raising NotImplementedError.

4. `SessionGroup` is declared as `dj.Man

### session_group#7  [MEDIUM | NEW-V2] SessionGroup.is_multi_day is unimplemented — no way to gate multi-day workflows even after a group is inserted
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:88-96`
- v1 behavior: No v1 equivalent.
- v2 behavior: v2/session_group.py:88-96 raises NotImplementedError. Other v2 code paths that should branch on multi-day-ness (e.g. concat materializer auto-preset resolution, cross-day brain region ambiguity) have no callable contract to consult.
- documented rationale: Final-shape / zero-migration policy. Acceptable as a forward declaration but it means the multi-day contract is currently documentation-only.
- verifier reasoning: Verified the reviewer's claim at the cited file:line. session_group.py:88-96 contains `is_multi_day(cls, key)` whose body is `raise NotImplementedError("SessionGroup.is_multi_day is not yet implemented")`. Docstring on line 92 says "Implemented in a follow-up change." The cross-reference to the auto preset is also verified: motion_correction.py:67-69 documents preset `"auto"` as resolved inside `ConcatenatedRecording.make()` — "maps to `rigid_fast` for same-day groups, raises for multi-day" — but `ConcatenatedRecording.make()` itself is a NotImplementedError stub at session_group.py:215-225, s

### session_group#8  [LOW | INT-UNJ] ConcatenatedRecording uses 'total_duration_s'; single-session Recording uses 'duration_s' — duration column name diverges across the two FK targets of SortingSelection
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:802 vs src/spyglass/spikesorting/v2/session_group.py:203`
- v1 behavior: No v1 equivalent.
- v2 behavior: Recording (v2/recording.py:794-806) stores duration as 'duration_s: float'. ConcatenatedRecording (v2/session_group.py:195-205) stores it as 'total_duration_s: float'. Code that wants to read 'how long is the recording this sort came from' has to branch on source kind.
- documented rationale: No documented rationale. Possibly intentional to highlight that concat duration is a sum, but this is not stated anywhere in v2/session_group.py.
- verifier reasoning: Verified both line citations directly.

src/spyglass/spikesorting/v2/recording.py:794-806 (Recording table definition):
```
definition = """
-> RecordingSelection
---
-> AnalysisNwbfile
electrical_series_path: varchar(255)
object_id: varchar(72)
n_channels: int
sampling_frequency: float
duration_s: float          # <-- line 802
cache_hash: char(64)
...
"""
```

src/spyglass/spikesorting/v2/session_group.py:195-205 (ConcatenatedRecording table definition):
```
definition = """
-> ConcatenatedRecordingSelection
---
-> AnalysisNwbfile
electrical_series_path: varchar(255)
object_id: varchar(72)
n_

### session_group#9  [LOW | NEW-V2] ConcatenatedRecording.make body unimplemented — downstream FK target is permanently empty
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:215-225`
- v1 behavior: No v1 equivalent.
- v2 behavior: make() raises NotImplementedError. The class docstring (v2/session_group.py:186-193) describes the intended output but no code produces a row; SortingSelection.ConcatenatedRecordingSource and Sorting.make's concat branch can never be reached by real data.
- documented rationale: Module docstring v2/session_group.py:3-7 explicitly justifies this: 'tables are declared in their final-shape so SortingSelection can FK ConcatenatedRecording from day one (zero-migration policy)'. Intentional and documented.
- verifier reasoning: Verified all cited code locations exactly match the reviewer's claim:

1. /cumulus/edeno/spyglass/src/spyglass/spikesorting/v2/session_group.py:215-225 — ConcatenatedRecording.make() body is exactly:
   ```
   def make(self, key):
       """Materialize the concatenated recording cache. ..."""
       raise NotImplementedError(
           "ConcatenatedRecording.make() is not implemented yet"
       )
   ```

2. /cumulus/edeno/spyglass/src/spyglass/spikesorting/v2/sorting.py:271-276 — SortingSelection.insert_selection rejects concat sources:
   ```
   if has_concat:
       raise NotImplementedErr

### session_group#10  [LOW | NEW-V2] MotionCorrectionParameters Pydantic validation runs only via insert1 override, not via DataJoint .insert() — the path used by insert_default itself
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:146-149`
- v1 behavior: No v1 equivalent.
- v2 behavior: MotionCorrectionParameters.insert_default (v2/session_group.py:146-149) calls 'cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)' — DataJoint's bulk insert path, which bypasses the insert1 override. The default rows are pre-validated by calling .model_dump() on construction (v2/session_group.py:121, 127, 133), so today the bypass is benign; but any contributor adding a new tuple-form default row that omits .model_dump() would land an unvalidated row. Same pattern as PreprocessingParameters (v2/recording.py:649) but PreprocessingParameters _DEFAULT_CONTENTS rows are also always pre-validated through Pydantic dumps.
- documented rationale: Consistent with the pattern in the other v2 Lookup tables (PreprocessingParameters, ArtifactDetectionParameters, SorterParameters). No explicit rationale in docstrings.
- verifier reasoning: Verified the reviewer's claim against the code:

1. v2/session_group.py:139-149: `insert1` override (lines 139-144) calls `_validate_params(MotionCorrectionParamsSchema, row["params"])` before delegating to `super().insert1`. But `insert_default` (lines 146-149) calls `cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)` — DataJoint's bulk `.insert()`, which does NOT delegate to the subclass `insert1` override. Confirmed.

2. The three default rows (lines 118-137) are each pre-validated at module-import time via `MotionCorrectionParamsSchema().model_dump()`, `MotionCorrectionParamsSchema(p

### session_group#11  [LOW | INT-JUST] MotionCorrectionParameters defaults 'auto_default' and 'rigid_fast_default' depend on consumer-side resolution of 'auto', which doesn't exist yet
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:125-130, src/spyglass/spikesorting/v2/_params/motion_correction.py:65-69`
- v1 behavior: No v1 equivalent.
- v2 behavior: MotionCorrectionParameters._DEFAULT_CONTENTS (v2/session_group.py:118-137) seeds an 'auto_default' row with preset='auto'. The schema (_params/motion_correction.py:65-69) says 'auto' is a 'Spyglass alias resolved inside ConcatenatedRecording.make() — maps to rigid_fast for same-day groups, raises for multi-day'. ConcatenatedRecording.make() is unimplemented, and SessionGroup.is_multi_day() is unimplemented, so the 'auto' alias has no resolution path.
- documented rationale: _params/motion_correction.py:34-40 and the module docstring at v2/session_group.py:1-13 document the zero-migration / final-shape policy. Documented forward declaration.
- verifier reasoning: Verified all cited lines:

1. **v2/session_group.py:118-137** confirmed: `_DEFAULT_CONTENTS` ships three rows including `"auto_default"` at line 126-130 which constructs `MotionCorrectionParamsSchema(preset="auto").model_dump()`.

2. **_params/motion_correction.py:65-69** confirmed verbatim: docstring says preset 'auto' is a "Spyglass alias resolved inside `ConcatenatedRecording.make()` -- maps to `rigid_fast` for same-day groups, raises for multi-day".

3. **'auto' is a valid Literal value**: `MotionPreset` Literal at lines 43-53 includes "auto", so the schema validation at insert time accept

### session_group#12  [LOW | DRIFT] MotionCorrectionParameters.insert_default is NOT invoked by initialize_v2_defaults — motion presets ship missing by default
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/__init__.py:28-34 and src/spyglass/spikesorting/v2/session_group.py:146-149`
- v1 behavior: No v1 equivalent.
- v2 behavior: initialize_v2_defaults (v2/__init__.py:14-37) seeds PreprocessingParameters, ArtifactDetectionParameters, and SorterParameters but does NOT call MotionCorrectionParameters.insert_default(). A user who runs initialize_v2_defaults will get the v2 Lookup defaults for every other params table but no motion-correction rows; once ConcatenatedRecording lands, FK constraints would silently fail to find 'none' / 'auto_default' / 'rigid_fast_default' until the user remembers to call insert_default by hand.
- documented rationale: No documented rationale. The omission is consistent with the consumer being NotImplementedError-gated, but the docstring promise is broken.
- verifier reasoning: Verified all claims directly from the source.

1. v2/__init__.py:28-34 calls exactly three insert_default()s:
   - PreprocessingParameters.insert_default()
   - ArtifactDetectionParameters.insert_default()
   - SorterParameters.insert_default()
   MotionCorrectionParameters is NOT called. Confirmed.

2. v2/__init__.py:8-10 docstring says: "The ``initialize_v2_defaults`` helper is the only eager symbol: it installs every default Lookup row the pipeline needs in one call, removing the 'forgot to call insert_default' first-run friction." Confirmed broken promise.

3. v2/session_group.py:100-149: 

### session_group#13  [LOW | INT-JUST] SessionGroup.Member has a redundant LabTeam FK distinct from the master's session_group_owner
- **v1**: `n/a`
- **v2**: `src/spyglass/spikesorting/v2/session_group.py:51-67`
- v1 behavior: No v1 equivalent.
- v2 behavior: SessionGroup master PK includes 'session_group_owner' projected from LabTeam.team_name (v2/session_group.py:52). The Member part adds another FK '-> LabTeam' (line 66) which inherits team_name. So each member can carry a per-member team_name that differs from the group's owner team. The docstring (lines 39-44) defines a member tuple to include team_name independently, which justifies it, but there is no validator enforcing relationship between the two (e.g. owner == member team_name is not required).
- documented rationale: Module docstring v2/session_group.py:39-44 — 'A member is a (nwb_file_name, sort_group_id, interval_list_name, team_name) tuple, not necessarily a whole NWB file'. Documented.
- verifier reasoning: Verified all claims at cited file:lines.

v2/session_group.py:51-56 — master definition is exactly:
```
-> LabTeam.proj(session_group_owner='team_name')
session_group_name: varchar(64)
---
description: varchar(255)
```
Confirms master PK projects LabTeam.team_name as session_group_owner.

v2/session_group.py:58-67 — Member part definition is exactly:
```
-> master
member_index: int
---
-> Session
-> SortGroupV2
-> IntervalList
-> LabTeam
```
Confirms a second, independent LabTeam FK on the Member part, which carries its own team_name.

v2/session_group.py:38-49 (docstring) — explicitly states:

## exceptions (cross) (10 findings)

### exceptions (cross)#1  [HIGH | INT-JUST] DuplicateSelectionError replaces v1's silent warning + multi-row return
- **v1**: `src/spyglass/spikesorting/v1/recording.py:176-182; same pattern at src/spyglass/spikesorting/v1/artifact.py:123-129 and src/spyglass/spikesorting/v1/sorting.py:223-229.`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:700-713; src/spyglass/spikesorting/v2/artifact.py:421-475; src/spyglass/spikesorting/v2/sorting.py:254-319.`
- v1 behavior: Multiple matching selection rows: v1 emits logger.warning("Similar row(s) already inserted.") and returns query.fetch(as_dict=True) -- a LIST of full row dicts.
- v2 behavior: v2 raises DuplicateSelectionError (subclass of ValueError) if >1 rows match the logical-identity restriction; the find-existing branch returns exactly one PK-only dict, never a list.
- documented rationale: v2 commits to UUID find-existing-or-insert semantics; multi-row case treated as structural bug not user warning. Documented in v2/recording.py:684-686 docstring.
- verifier reasoning: Verified all citations directly against the code.

v1 silent-warning + list-return pattern confirmed:
- v1/recording.py:176-179: `if query: logger.warning("Similar row(s) already inserted."); return query.fetch(as_dict=True)` — returns list of full row dicts.
- v1/artifact.py:123-126: identical pattern with `logger.warning`.
- v1/sorting.py:223-226: same shape but uses `logger.info`, NOT `logger.warning` as the finding states. Minor inaccuracy in the finding, but does not change the substantive divergence: v1 still silently logs + returns `query.fetch(as_dict=True)` (list of full rows).

v2 ra

### exceptions (cross)#2  [HIGH | NEW-V2] RecordingTruncatedError adds a save-coverage guard that does not exist in v1
- **v1**: `src/spyglass/spikesorting/v1/recording.py:185-... (SpikeSortingRecording.make has no equivalent guard).`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1027-1068 (raise at line 1060).`
- v1 behavior: v1 SpikeSortingRecording writes the preprocessed ElectricalSeries to AnalysisNwbfile and inserts without comparing requested vs saved duration. No truncation check (grep "truncat" v1/recording.py finds only an unrelated KeyError at line 401).
- v2 behavior: v2 Recording.make_insert computes requested_total = sum(end - start for start, end in sort_valid_times), compares against saved_total with tolerance = 1.5 / sampling_frequency, and on excess deletes the orphan analysis NWB file before raising RecordingTruncatedError.
- documented rationale: Documented in exceptions.py:25-30. v2 NwbfileHasher replaces the v1 RecordingRecompute pipeline; this guard is the new in-pipeline integrity check.
- verifier reasoning: All claims verified directly against the code.

1) v1 has no truncation guard. grep for "truncat" in src/spyglass/spikesorting/v1/recording.py returns no hits. v1 SpikeSortingRecording.make_insert (lines 241-260) does only:
   - IntervalList.insert1(sort_interval_valid_times.as_dict, skip_duplicates=True)
   - AnalysisNwbfile().add(nwb_file_name, insert_key["analysis_file_name"])
   - self.insert1(insert_key); self._record_environment(insert_key)
No duration comparison, no tolerance, no raise.

2) v2 guard exists exactly as described. recording.py:1027-1068:
   - imports `from spyglass.spikeso

### exceptions (cross)#3  [MEDIUM | UNTESTED] ConcatBrainRegionAmbiguousError has two raise sites but zero test coverage
- **v1**: `n/a -- no v1 brain-region resolution at the sort level.`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:977-985; src/spyglass/spikesorting/v2/curation.py:1137-...`
- v1 behavior: v1 has no get_unit_brain_regions analogue at the Sorting/Curation level (grep get_unit_brain_regions v1/*.py returns nothing); the failure mode and exception class do not exist.
- v2 behavior: Sorting.get_unit_brain_regions and CurationV2.get_unit_brain_regions raise ConcatBrainRegionAmbiguousError when resolve_source(...).kind == "concatenated_recording" and allow_anchor_member=False. The branch is reachable only via a concat source-part row -- which itself is gated upstream by NotImplementedError in the concat materializer.
- documented rationale: Documented in exceptions.py:48-55 ("per-session regions require cross-session unit matching, not available in this build"). The branch requires a concat-source fixture that the upstream materializer cannot produce yet, so test coverage will arrive with the concat consumer.
- verifier reasoning: All claims verified against the code:

(1) Two raise sites confirmed: 
- src/spyglass/spikesorting/v2/sorting.py:977-985 raises ConcatBrainRegionAmbiguousError inside Sorting.get_unit_brain_regions when `source.kind == "concatenated_recording"` and `allow_anchor_member=False`.
- src/spyglass/spikesorting/v2/curation.py:1137-1143 raises the same exception in CurationV2.get_unit_brain_regions under identical conditions.

(2) Zero test coverage confirmed: `grep -rn "ConcatBrainRegion\|allow_anchor_member" tests/spikesorting/` returns no matches.

(3) v1 has no analogue: `grep -rn "get_unit_brain_

### exceptions (cross)#4  [MEDIUM | UNTESTED] NonIntegerUnitIDError fires where v1 would silently pass through to pynwb
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:592-599.`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:1609-1618 (raise at 1613, from exc).`
- v1 behavior: v1 _write_sorting_to_nwb_with_curation does `for unit_id in sorting.get_unit_ids(): nwbf.add_unit(... id=unit_id ...)` with no int coercion. Non-integer unit_id would be stored as-is or surface a confusing pynwb/HDF5 error downstream.
- v2 behavior: v2 _populate_unit_part wraps int(unit_id) in try/except and raises NonIntegerUnitIDError("... v2's Sorting.Unit stores int unit_ids; remap before insertion if the sorter emits non-convertible IDs.") if conversion fails.
- documented rationale: Documented in exceptions.py:33-36 and at the raise site (sorting.py:1614-1618: "v2's Sorting.Unit stores int unit_ids; remap before insertion"). No test exercises the branch.
- verifier reasoning: Verified directly against source.

v1 (src/spyglass/spikesorting/v1/sorting.py:592-599): `for unit_id in sorting.get_unit_ids(): ... nwbf.add_unit(spike_times=..., id=unit_id, obs_intervals=obs_interval, curation_label="uncurated")` — no int coercion, no try/except. A non-int unit_id would flow straight into pynwb.

v2 (src/spyglass/spikesorting/v2/sorting.py:1609-1618):
```
for unit_id in sorting.unit_ids:
    try:
        int_unit_id = int(unit_id)
    except (TypeError, ValueError) as exc:
        raise NonIntegerUnitIDError(
            f"Sorting.make: sorter returned unit_id {unit_id!r} "

### exceptions (cross)#5  [MEDIUM | NEW-V2] SchemaBypassError is a new structural check with no v1 precedent
- **v1**: `n/a -- v1 has no source-part schema (v1/sorting.py:198-206, v1/artifact.py:97-105 show single-FK selection tables).`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:383-397; src/spyglass/spikesorting/v2/artifact.py:551-565. Also consumed defensively in src/spyglass/spikesorting/v2/artifact.py:1099-1114.`
- v1 behavior: v1 uses single-FK Selection rows (no source-part pattern); the failure mode (master with 0 or 2 source-part rows) is structurally impossible in v1 and the exception class does not exist.
- v2 behavior: resolve_source() (called from make()) fetches both source parts and raises SchemaBypassError if total != 1. Message names the table, PK, and points the caller at insert_selection().
- documented rationale: Documented in exceptions.py:19-22 as source-part integrity invariant; consumer-side recovery in artifact.py:1099-1113 documents why a broad except would silently orphan IntervalList rows.
- verifier reasoning: Verified all claims by direct code inspection.

(1) v1 has no exceptions.py and no SchemaBypassError class — `grep -rn "SchemaBypassError" v1/` returns empty; v1 directory listing has no exceptions.py.

(2) v1 selection tables use a single FK to SpikeSortingRecording, making the source-part divergence structurally impossible:
- v1/sorting.py:198-206: `SpikeSortingSelection` definition with `-> SpikeSortingRecording` (single FK).
- v1/artifact.py:97-105: `ArtifactDetectionSelection` definition with `-> SpikeSortingRecording` (single FK).

(3) v2 introduces source-part subtables:
- v2/sorting.py

### exceptions (cross)#6  [MEDIUM | INT-JUST] ZeroUnitSortError converts v1's silent empty-units NWB into an opt-in hard error
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:578-581 (silent empty-units NWB write).`
- **v2**: `src/spyglass/spikesorting/v2/pipeline.py:226-235; src/spyglass/spikesorting/v2/exceptions.py:72-79.`
- v1 behavior: v1 SpikeSorting silently writes an empty pynwb.misc.Units table (nwbf.units = pynwb.misc.Units(name="units", description="Empty units table.")) when sorting.get_num_units() == 0 -- no warning, no error, no caller-facing signal.
- v2 behavior: v2 Sorting.make also writes an empty sort row (graceful by default); run_v2_pipeline only raises ZeroUnitSortError if the caller opted in with require_units=True (pipeline.py:228-235). Otherwise it logs a warning and proceeds to insert an empty curation+merge row to keep the result merge-keyable.
- documented rationale: Documented in exceptions.py:72-79 and pipeline.py:222-247 ("Fall through ... matching v1, which writes an empty Units table"). Default behavior preserved; opt-in raise is the only new branch.
- verifier reasoning: Verified all citations against the actual code.

v1 silent empty-Units write confirmed at src/spyglass/spikesorting/v1/sorting.py:578-581:
```
if sorting.get_num_units() == 0:
    nwbf.units = pynwb.misc.Units(
        name="units", description="Empty units table."
    )
```
No warning, no error, no caller-facing signal — silent.

v2 opt-in raise confirmed at src/spyglass/spikesorting/v2/pipeline.py:226-235:
```
n_units = int((Sorting & sort_pk).fetch1("n_units"))
if n_units == 0:
    recording_id = rec_pk["recording_id"]
    if require_units:
        raise ZeroUnitSortError(...)
```
Followed 

### exceptions (cross)#7  [LOW | UNTESTED] MissingRecordingForConcatError is declared but never raised and never tested
- **v1**: `n/a -- v1 has no concat pipeline.`
- **v2**: `src/spyglass/spikesorting/v2/exceptions.py:58-63 (definition); src/spyglass/spikesorting/v2/session_group.py:178-181, 223-225 (still NotImplementedError stubs).`
- v1 behavior: v1 has no concat-recording pipeline; the failure mode does not exist in v1.
- v2 behavior: Exception defined in v2/exceptions.py:58-63 with a docstring naming ConcatenatedRecordingSelection.insert_selection() and ConcatenatedRecording.make() as raisers. Both methods (session_group.py:171-181, 215-225) raise NotImplementedError instead; no per-member Recording check exists in current code.
- documented rationale: Forward-declared per schema-first / zero-migration policy (session_group.py:1-14).
- verifier reasoning: Verified directly:

1. exceptions.py:58-63 defines `MissingRecordingForConcatError(RuntimeError)` with docstring: "`ConcatenatedRecordingSelection.insert_selection()` or `ConcatenatedRecording.make()` cannot find a populated per-member `Recording` row with the shared `preproc_params_name`. Message lists the missing member keys and instructs the caller to populate `Recording` for those members first."

2. session_group.py:178-181: `ConcatenatedRecordingSelection.insert_selection` raises `NotImplementedError("ConcatenatedRecordingSelection.insert_selection is not yet implemented")` — does NOT ra

### exceptions (cross)#8  [LOW | INT-UNJ] PipelineInputError docstring overstates what is currently validated
- **v1**: `n/a.`
- **v2**: `src/spyglass/spikesorting/v2/exceptions.py:66-69 (docstring); src/spyglass/spikesorting/v2/pipeline.py:181-187 (only unknown-preset path).`
- v1 behavior: v1 has no run_v2_pipeline orchestrator; the failure mode is v2-specific.
- v2 behavior: Docstring (exceptions.py:66-69) promises: "run_v2_pipeline() receives zero, partial, or mixed input modes. Message says exactly one input mode is required and lists the required fields for each mode." Actual run_v2_pipeline (pipeline.py:181-187) only validates `preset not in _PRESETS` and raises with an "unknown preset" message; there is no input-mode XOR check.
- documented rationale: No documented justification for the docstring/behavior mismatch. Likely a placeholder for a future multi-input-mode signature (concat / session-group inputs); should be either narrowed to match current behavior or implemented.
- verifier reasoning: Verified directly against source.

1. exceptions.py:66-69 docstring claims PipelineInputError fires when "run_v2_pipeline() receives zero, partial, or mixed input modes. Message says exactly one input mode is required and lists the required fields for each mode."

2. pipeline.py:86-94 shows run_v2_pipeline has a fixed signature with no alternative input modes:
   def run_v2_pipeline(nwb_file_name: str, sort_group_id: int, interval_list_name: str, team_name: str, preset: str = "franklab_tetrode_mountainsort5", description: str = "", require_units: bool = False) -> dict[str, Any]
   All four dat

### exceptions (cross)#9  [LOW | UNTESTED] SessionGroupDateError is declared but never raised and never tested
- **v1**: `n/a -- no v1 module.`
- **v2**: `src/spyglass/spikesorting/v2/exceptions.py:39-45 (definition); src/spyglass/spikesorting/v2/session_group.py:69-96 (function bodies are NotImplementedError stubs).`
- v1 behavior: v1 has no SessionGroup concept at all; cross-session grouping is not a v1 feature.
- v2 behavior: Exception is defined with a detailed docstring describing SessionGroup.create_group multi-day enforcement. v2/session_group.py:84-86 and :94-96 (create_group and is_multi_day) raise plain NotImplementedError instead -- the documented exception is never wired up.
- documented rationale: Forward-declared per schema-first / zero-migration policy (session_group.py:1-14 docstring). No documented justification for shipping the exception class ahead of its raise site.
- verifier reasoning: Verified directly. `grep -rn "SessionGroupDateError"` over src/ and tests/ returns only the class definition at `src/spyglass/spikesorting/v2/exceptions.py:39` — zero `raise` sites, zero test references. The class is declared (lines 39-45) with a docstring promising it is raised by `SessionGroup.create_group()` for caller-supplied `recording_date` or for members spanning multiple dates without `allow_multi_day=True`. But `session_group.py:84-86` (`create_group`) and `session_group.py:94-96` (`is_multi_day`) both raise plain `NotImplementedError("... is not yet implemented")`, and `create_group

### exceptions (cross)#10  [LOW | INT-JUST] ZeroUnitAnalyzerError fires where v1's analyzer concept does not exist
- **v1**: `n/a -- no v1 SortingAnalyzer surface.`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:841-848 (raise ZeroUnitAnalyzerError); the parallel get_sorting zero-unit handler at sorting.py:749-757 returns empty NumpySorting and does NOT raise.`
- v1 behavior: v1 has no SortingAnalyzer concept (uses the legacy SpikeInterface waveform extractor path); no equivalent get_analyzer method on v1 SpikeSorting (grep get_analyzer v1/sorting.py returns nothing).
- v2 behavior: v2 Sorting.get_analyzer checks int((self & key).fetch1("n_units")) == 0 and raises ZeroUnitAnalyzerError instead of attempting to load a never-written analyzer folder (SI cannot build a SortingAnalyzer over zero units -- estimate_sparsity crashes on np.concatenate([])). get_sorting() still returns an empty NumpySorting for callers that only need the unit list.
- documented rationale: Documented in both the raise-site comment (sorting.py:827-832) and exceptions.py:82-89: loading a phantom folder would surface a confusing SI error -- this is the clean signal.
- verifier reasoning: Verified by direct code inspection.

1. v1 has no SortingAnalyzer surface: `grep -rn "SortingAnalyzer\|analyzer\|get_analyzer" src/spyglass/spikesorting/v1/` returns zero matches. v1's `Sorting.get_sorting` at v1/sorting.py:499 is the only retrieval API. The claim "v1 has no analyzer concept" holds.

2. v2 raise site confirmed at src/spyglass/spikesorting/v2/sorting.py:841-848:
```
if int((self & key).fetch1("n_units")) == 0:
    raise ZeroUnitAnalyzerError(
        "Sorting.get_analyzer: sorting_id="
        f"{key['sorting_id']!r} has zero units; no "
        "SortingAnalyzer exists (SI cann

## nwb-io (cross) (10 findings)

### nwb-io (cross)#1  [HIGH | INT-JUST] v2 ElectricalSeries write raises on heterogeneous channel gains; v1 silently picks gains[0]
- **v1**: `src/spyglass/spikesorting/v1/recording.py:858`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1862-1870`
- v1 behavior: _write_recording_to_nwb passes `conversion=np.unique(recording.get_channel_gains())[0] * 1e-6` -- picks the first unique gain even when channels have heterogeneous gains. The resulting ElectricalSeries.conversion is applied uniformly across channels, silently mis-scaling channels whose gains differ from the chosen one.
- v2 behavior: _write_nwb_artifact computes `gains = _np.unique(recording.get_channel_gains())`; if `len(gains) != 1` it raises ValueError('Recording.make: recording has heterogeneous channel gains ...'). Otherwise `conversion = float(gains[0]) * 1e-6`.
- documented rationale: documented inline at src/spyglass/spikesorting/v2/recording.py:1853-1869 and referenced to shared-contracts.md
- verifier reasoning: Verified both code locations directly.

v1 at src/spyglass/spikesorting/v1/recording.py:858 confirms:
`conversion=np.unique(recording.get_channel_gains())[0] * 1e-6,`
This silently picks the first unique gain. If channels have heterogeneous gains, np.unique returns multiple values and [0] selects only one, which becomes the universal conversion factor for the ElectricalSeries -- silently mis-scaling other channels.

v2 at src/spyglass/spikesorting/v2/recording.py:1862-1870 confirms:
```
gains = _np.unique(recording.get_channel_gains())
if len(gains) != 1:
    raise ValueError(
        "Recordi

### nwb-io (cross)#2  [HIGH | INT-JUST] v2 curation no longer writes the 'merge_groups' NWB column; provenance lives only in DataJoint MergeGroup part
- **v1**: `src/spyglass/spikesorting/v1/curation.py:404-415`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:101-130 (MergeGroup part definition); v2/curation.py:864-876 (only curation_label written, no merge_groups branch); v2/curation.py:104-110 docstring 'replaced v1's merge_groups NWB column with a per-unit kept-set'`
- v1 behavior: _write_sorting_to_nwb_with_curation calls nwbf.add_unit_column(name='merge_groups', description='merge groups', data=merge_groups_list, index=True) -- a per-unit ragged column on the curated units NWB. External readers (v1/figurl_curation.py reads via nwb_sorting.get('merge_groups')) can recover the merge dictionary from the NWB alone.
- v2 behavior: CurationV2._write_curated_nwb adds only the 'curation_label' column (when non-empty). The MergeGroup part table on CurationV2 stores merge provenance as (kept_unit_id, contributor_unit_id) pairs but NOTHING about merges is written into the AnalysisNwbfile. A consumer that opens the curated NWB without a DB connection cannot reconstruct merge groups.
- documented rationale: documented in src/spyglass/spikesorting/v2/curation.py:101-130 and 104-110; rationale is 'queryable provenance' over file-level provenance
- verifier reasoning: Verified directly against code.

v1 writes merge_groups as a per-unit ragged NWB column (src/spyglass/spikesorting/v1/curation.py:404-415):
```
if merge_groups is not None:
    merge_groups_dict = _list_to_merge_dict(merge_groups, unit_ids)
    merge_groups_list = [[""] if value == [] else value for value in merge_groups_dict.values()]
    nwbf.add_unit_column(name="merge_groups", description="merge groups", data=merge_groups_list, index=True)
```

v1 reads it back from the NWB alone (src/spyglass/spikesorting/v1/figurl_curation.py:85):
```
merge_groups = list(nwb_sorting.get("merge_groups", [

### nwb-io (cross)#3  [MEDIUM | UNTESTED] v2 ElectricalSeries write tests assert specific shape, but parity tests (test_v1_parity.py) do not verify ProcessedElectricalSeries metadata vs v1
- **v1**: `tests/spikesorting/v1/ (1.1k lines, light); no ElectricalSeries metadata assertions`
- **v2**: `tests/spikesorting/v2/test_v1_parity.py:521 (sole 'merge_groups' reference); tests/spikesorting/v2/test_single_session_pipeline.py:2865-2887 (regression fixture not parity)`
- v1 behavior: v1 NWB writes filtering / description / conversion / electrodes_id / object_id; no test in v1 enforces the exact string values of filtering or description.
- v2 behavior: test_single_session_pipeline.py line 2865-2879 constructs a ProcessedElectricalSeries with `filtering='bandpass'` and `conversion=1e-6` -- but this is for a regression fixture, not a parity check. test_v1_parity.py (896 lines) only references 'get_merge_groups' once and has zero assertions about ElectricalSeries filtering/description/conversion/object_id parity with v1's strings.
- verifier reasoning: Verified the finding by reading code directly.

1) The v2 NWB-write metadata for ProcessedElectricalSeries diverges from v1:
   - v1 src/spyglass/spikesorting/v1/recording.py:855 sets `filtering="Bandpass filtered for spike band"`; v2 src/spyglass/spikesorting/v2/recording.py:1904 sets `filtering="Bandpass filter + common reference"`.
   - v1 recording.py:856-857 sets description=`"Referenced and filtered recording from {nwb_file_name} for spike sorting"`; v2 recording.py:1905-1908 sets description=`"Pre-motion preprocessed recording from {nwb_file_name} for spike sorting"`.
   - v1 recording.

### nwb-io (cross)#4  [MEDIUM | INT-JUST] v2 Recording does not persist 'electrodes_id' (object_id of the electrodes sub-object); v1 stores both
- **v1**: `src/spyglass/spikesorting/v1/recording.py:187-195 (definition); v1/recording.py:864 (electrodes_id capture)`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:794-806 (definition); v2/recording.py:1899-1915 (only ElectricalSeries object_id captured)`
- v1 behavior: SpikeSortingRecording definition stores three identity fields: object_id (varchar(40)) for the ProcessedElectricalSeries, electrodes_id (varchar(40)) for ProcessedElectricalSeries/electrodes, hash (varchar(32)). _write_recording_to_nwb captures electrodes_id = nwbfile.acquisition[series_name].electrodes.object_id and the recompute machinery (v1/recompute.py + RecordingRecompute) uses electrodes_id to verify the electrodes sub-object matches across rebuilds.
- v2 behavior: Recording definition stores object_id (varchar(72)) plus electrical_series_path (varchar(255)), n_channels, sampling_frequency, duration_s, cache_hash (char(64)), timestamps_adjusted, n_adjusted_samples. There is no electrodes_id column and _write_nwb_artifact does not capture or persist nwbfile.acquisition[...].electrodes.object_id.
- documented rationale: implicit: v2 design drops the recompute chain and uses whole-file NwbfileHasher (see v2/recording.py:1797-1803 'shared-contracts.md Recording Cache Format. The v1 recompute machinery uses the same hashing path, so v2 verification does not maintain a parallel implementation.')
- verifier reasoning: Verified all claims against source.

v1 schema at src/spyglass/spikesorting/v1/recording.py:187-195:
```
definition = """
# Processed recording.
-> SpikeSortingRecordingSelection
---
-> AnalysisNwbfile
object_id: varchar(40) # Object ID for the processed recording in NWB file
electrodes_id=null: varchar(40) # Object ID for the processed electrodes
hash=null: varchar(32) # Hash of the NWB file
"""
```

v1 capture at v1/recording.py:863-864:
```
recording_object_id = nwbfile.acquisition[series_name].object_id
electrodes_id = nwbfile.acquisition[series_name].electrodes.object_id
```

v2 schema at

### nwb-io (cross)#5  [MEDIUM | INT-UNJ] v2 Recording does not register sort_valid_times to IntervalList (v1 always did)
- **v1**: `src/spyglass/spikesorting/v1/recording.py:211-217 (set_key + pipeline='spikesorting_recording_v1'); src/spyglass/spikesorting/v1/recording.py:255-257 (IntervalList.insert1)`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1077-1090 (insert1 only); grep IntervalList.insert in v2/recording.py returns nothing`
- v1 behavior: SpikeSortingRecording.make_insert writes a new IntervalList row with interval_list_name=key['recording_id'], pipeline='spikesorting_recording_v1', valid_times=sort_interval_valid_times. Downstream callers can then look up the sort interval by recording_id (v1/utils.py:77 uses this convention).
- v2 behavior: Recording.make_insert only writes the master row (with electrical_series_path, object_id, n_channels, etc.). No IntervalList row keyed by recording_id is ever created. The sort_valid_times array is fetched per-call from the upstream IntervalList row identified by RecordingSelection.interval_list_name, but no recording_id-scoped IntervalList row is materialized.
- verifier reasoning: Verified the core behavioral divergence: v1 writes a recording_id-keyed IntervalList row, v2 does not.

v1 evidence — src/spyglass/spikesorting/v1/recording.py:211-217 sets the IntervalList key with `interval_list_name=key["recording_id"]` and `pipeline="spikesorting_recording_v1"`; lines 255-257 insert it:
```
IntervalList.insert1(
    sort_interval_valid_times.as_dict, skip_duplicates=True
)
```

v2 evidence — src/spyglass/spikesorting/v2/recording.py:1077-1090 only inserts the master Recording row (electrical_series_path, object_id, n_channels, sampling_frequency, duration_s, cache_hash, ti

### nwb-io (cross)#6  [MEDIUM | INT-JUST] v2 artifact IntervalList uses 'artifact_{uuid}' prefix instead of v1's bare str(uuid)
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:194-205`
- **v2**: `src/spyglass/spikesorting/v2/artifact.py:807-826; helper at src/spyglass/spikesorting/v2/utils.py:512-519`
- v1 behavior: ArtifactDetection.make inserts IntervalList row with interval_list_name=str(key['artifact_id']) (the bare UUID string) and pipeline='spikesorting_artifact_v1'.
- v2 behavior: ArtifactDetection.make_insert writes interval_list_name=artifact_interval_list_name(artifact_id) which returns f'artifact_{artifact_id}' (prefixed) and pipeline='spikesorting_artifact_v2'.
- documented rationale: documented in v2/utils.py:512-519 ('Centralizes the v2 convention'); merge-dispatcher contract uses parse_artifact_interval_list_name as inverse
- verifier reasoning: Verified all three claims by reading the code directly.

(1) v1 artifact.py:194-205 inserts the IntervalList row with `interval_list_name=str(key["artifact_id"])` (line 200, bare UUID string) and `pipeline="spikesorting_artifact_v1"` (line 202).

(2) v2 artifact.py:807 computes `interval_list_name = artifact_interval_list_name(key["artifact_id"])`, and the insert rows on lines 813-821 set `pipeline="spikesorting_artifact_v2"` (line 818). The v2 version also writes one row per member nwb_file_name (multi-row) rather than v1's single row.

(3) v2 utils.py:509-519 defines `_ARTIFACT_INTERVAL_LIST

### nwb-io (cross)#7  [LOW | INT-JUST] v1 hardcodes 30 kHz in TimestampsExtractor; v2 propagates the actual recording sampling_frequency
- **v1**: `src/spyglass/spikesorting/v1/recording.py:973 (default 30e3); v1/recording.py:847-849 (no sampling_frequency override at call site)`
- **v2**: `src/spyglass/spikesorting/v2/_nwb_iterators.py:127 (default 30e3 retained in private extractor); src/spyglass/spikesorting/v2/recording.py:1878 (caller resolves true rate); v2/recording.py:1885-1889 (passes sampling_frequency)`
- v1 behavior: TimestampsExtractor.__init__ defaults `sampling_frequency=30e3`. _write_recording_to_nwb constructs `TimestampsDataChunkIterator(recording=TimestampsExtractor(timestamps), buffer_gb=5)` -- using the default 30 kHz regardless of recording's actual sampling rate.
- v2 behavior: _TimestampsExtractor still defaults 30e3, but TimestampsDataChunkIterator takes a `sampling_frequency` param and the v2 writer passes `sampling_frequency = float(recording.get_sampling_frequency())` (the real rate).
- documented rationale: documented in src/spyglass/spikesorting/v2/_nwb_iterators.py:17-24
- verifier reasoning: Verified by direct read of all four cited locations.

(1) v1/recording.py:969-984 defines `TimestampsExtractor.__init__(self, timestamps, sampling_frequency=30e3)` — default 30e3 confirmed at line 973.

(2) v1/recording.py:847-848 calls `TimestampsDataChunkIterator(recording=TimestampsExtractor(timestamps), buffer_gb=5)` — no `sampling_frequency` override at call site, so the 30e3 default is always used in v1 regardless of `recording.get_sampling_frequency()`.

(3) v2/_nwb_iterators.py:124-138 keeps a private `_TimestampsExtractor` with the same `sampling_frequency=30e3` default — confirmed at

### nwb-io (cross)#8  [LOW | INT-JUST] v2 ElectricalSeries 'filtering' / 'description' attribute strings differ from v1
- **v1**: `src/spyglass/spikesorting/v1/recording.py:840-857`
- **v2**: `src/spyglass/spikesorting/v2/recording.py:1895-1909`
- v1 behavior: filtering='Bandpass filtered for spike band'; description=f'Referenced and filtered recording from {nwb_file_name} for spike sorting'; table_region description='Sort group'.
- v2 behavior: filtering='Bandpass filter + common reference'; description=f'Pre-motion preprocessed recording from {nwb_file_name} for spike sorting'; table_region description='Sort group electrodes'.
- documented rationale: no documented rationale; 'Pre-motion' likely reflects the SortingAnalyzer pipeline where motion correction is deferred to sorter stage
- verifier reasoning: Citations verified exactly as stated.

v1 (src/spyglass/spikesorting/v1/recording.py:840-857):
- table_region description="Sort group"
- filtering="Bandpass filtered for spike band"
- description="Referenced and filtered recording from {nwb_file_name} for spike sorting"

v2 (src/spyglass/spikesorting/v2/recording.py:1895-1909):
- table_region description="Sort group electrodes"
- filtering="Bandpass filter + common reference"
- description="Pre-motion preprocessed recording from {nwb_file_name} for spike sorting"

Grep confirms these strings are SET but never READ by any code in the repo. No c

### nwb-io (cross)#9  [LOW | INT-UNJ] v2 Sorting stores time_of_sort as datetime; v1 stores as Unix int seconds
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:239 (definition); v1/sorting.py:295 (int(time.time()))`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:429 (definition); v2/sorting.py:676 (_dt.datetime.now())`
- v1 behavior: SpikeSorting definition: `time_of_sort: int  # in Unix time, to the nearest second`. make_compute sets `time_of_sort = int(time.time())`.
- v2 behavior: Sorting definition: `time_of_sort: datetime`. make_insert sets `time_of_sort = _dt.datetime.now()`.
- verifier reasoning: Directly verified all cited locations. v1/sorting.py:239 declares `time_of_sort: int               # in Unix time, to the nearest second`; v1/sorting.py:295 sets `time_of_sort = int(time.time())`. v2/sorting.py:429 declares `time_of_sort: datetime`; v2/sorting.py:676 sets `"time_of_sort": _dt.datetime.now()` with `import datetime as _dt` at line 660. The `datetime.now()` call is naive (no tz=), confirming the timezone-awareness risk. v0 also uses int Unix time (v0/spikesorting_sorting.py:181), so v2 is the lone outlier. No comment, docstring, or commit message explains the migration at either 

### nwb-io (cross)#10  [LOW | INT-JUST] v2 sorting writes obs_intervals fallback to full timestamps envelope when artifact_id is None; v1 always passes artifact-removed intervals
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:587-598`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:499-514 (None pathway), 1480-1488 (fallback envelope)`
- v1 behavior: _write_sorting_to_nwb always receives `sort_interval` = artifact_removed_intervals from the artifact IntervalList; if 1D it is reshaped to (1,2); nwbf.add_unit(obs_intervals=obs_interval, ...) for every unit.
- v2 behavior: _write_units_nwb receives obs_intervals=None when sel_row['artifact_id'] is None (line 514) and falls back to obs_intervals_arr = _np.asarray([[float(timestamps[0]), float(timestamps[-1])]]) -- the full recording envelope. The fallback envelope is wider than the artifact-removed intervals, so downstream firing-rate calcs over an unmasked sort report observation across the whole recording.
- documented rationale: documented in src/spyglass/spikesorting/v2/sorting.py:1462-1469 ('no-mask sort semantics (the sort observed every sample)')
- verifier reasoning: Verified all cited code locations.

v1 evidence (src/spyglass/spikesorting/v1/sorting.py):
- Lines 198-206: SpikeSortingSelection has a required `-> IntervalList` FK -- non-nullable.
- Lines 259-265: `make_fetch` always fetches `artifact_removed_intervals` from the IntervalList associated with `recording_key["interval_list_name"]`.
- Lines 587-598: `_write_sorting_to_nwb` reshapes `sort_interval` to (1,2) if 1D and passes `obs_intervals=obs_interval` for every unit. No None code path exists.

v2 evidence (src/spyglass/spikesorting/v2/sorting.py):
- Lines 215-220: `SortingSelection` definition 

## params (cross) (10 findings)

### params (cross)#1  [HIGH | INT-JUST] v2 ClusterlessThresholderSchema noise_levels default is None (schema), [1.0] (shipped row); v1 inline default was [1.0]
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:169-181`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:135-159, src/spyglass/spikesorting/v2/sorting.py:162-183 (shipped row sets [1.0]), src/spyglass/spikesorting/v2/sorting.py:1146-1167 (runtime branching)`
- v1 behavior: SpikeSorterParameters.contents 'default_clusterless' row hard-codes noise_levels=np.asarray([1.0]) with the comment 'noise levels needs to be 1.0 so the units are in uV and not MAD'. The field is always present.
- v2 behavior: ClusterlessThresholderSchema.noise_levels: list[float] | None = Field(default=None) — when None, the runtime drops the key so detect_peaks falls back to SI's per-channel MAD estimation. The shipped 'default' SorterParameters row EXPLICITLY passes noise_levels=[1.0], preserving v1's uV interpretation for the shipped preset.
- documented rationale: Documented at _params/sorter.py:135-149 (schema_version=3 rationale) and v2/sorting.py:166-180 (shipped row mirrors v1).
- verifier reasoning: All four cited locations match the finding exactly.

v1 (src/spyglass/spikesorting/v1/sorting.py:169-181): The 'default_clusterless' row hard-codes `"noise_levels": np.asarray([1.0])` with comment "noise levels needs to be 1.0 so the units are in uV and not MAD" (line 176-177). Field is always present.

v2 schema (src/spyglass/spikesorting/v2/_params/sorter.py:159): `noise_levels: list[float] | None = Field(default=None)`. The default is the immutable singleton `None`, not a mutable list — avoiding the Pydantic mutable-default bug class. Lines 135-149 document: "noise_levels defaults to None (

### params (cross)#2  [HIGH | INT-JUST] v2 artifact-detection ships amplitude_thresh_uV=500 vs v1's 3000
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:65-77 (default row), src/spyglass/spikesorting/v1/recording.py:845 (return_scaled=False), src/spyglass/spikesorting/utils.py:178-183 (threshold compared against unscaled traces)`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:54 (default=500.0), src/spyglass/spikesorting/v2/_params/artifact_detection.py:44-52 (rationale)`
- v1 behavior: ArtifactDetectionParameters 'default' row sets amplitude_thresh_uV=3000 (units labelled microvolts in docstring) and is compared against recording.get_traces() with return_scaled=False, so the threshold actually compares against raw int16 ADC counts — not microvolts.
- v2 behavior: ArtifactDetectionParamsSchema.amplitude_thresh_uV defaults to 500.0 (documented as v1's effective Intan-probe behavior within ~15%). The schema docstring explicitly calls this a bug-fix.
- documented rationale: Documented in src/spyglass/spikesorting/v2/_params/artifact_detection.py:44-52 — explicitly a bug-fix for v1's unit-conversion error.
- verifier reasoning: Verified the finding by reading the actual code in both v1 and v2.

v1 evidence (all confirmed):
- src/spyglass/spikesorting/v1/artifact.py:70 — `"amplitude_thresh_uV": 3000` in the `default` row of `ArtifactDetectionParameters.contents`.
- src/spyglass/spikesorting/v1/recording.py:844-846 — preprocessed recording is written with `return_scaled=False`: `SpikeInterfaceRecordingDataChunkIterator(recording=recording, return_scaled=False, buffer_gb=5)`. The NWB `ElectricalSeries.conversion` is set to `np.unique(recording.get_channel_gains())[0] * 1e-6` (line 858), i.e., gain encoded as a conversio

### params (cross)#3  [MEDIUM | INT-JUST] v2 Kilosort4Schema curates a fixed default set; v1 dynamically uses sis.get_default_sorter_params(sorter)
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:184-189`
- **v2**: `src/spyglass/spikesorting/v2/_params/sorter.py:91-112`
- v1 behavior: v1's SpikeSorterParameters.contents.extend uses `sis.get_default_sorter_params(sorter)` for every installed SI sorter including KS4 (v1/sorting.py:184-189). The actual default values depend on the installed SI version at insert_default() time.
- v2 behavior: Kilosort4Schema (v2/_params/sorter.py:91-112) curates explicit defaults Th_universal=9.0, Th_learned=8.0, nblocks=1, max_cluster_subset=25_000, do_CAR=True with extra='allow' for SI's other kwargs. Decoupled from sis.get_default_sorter_params().
- documented rationale: Documented at _params/sorter.py:94-104 — v2 curates the most-used knobs; allows escape hatch. KS4 non-determinism is also called out (CPU/GPU runtime differences).
- verifier reasoning: Confirmed both code references directly:

v1/sorting.py:184-189 reads:
```
contents.extend(
    [
        [sorter, "default", sis.get_default_sorter_params(sorter)]
        for sorter in sis.available_sorters()
    ]
)
```
So v1 dynamically loads SI's defaults for every installed sorter (including KS4) at insert_default() time — defaults silently track whatever SI version is installed.

v2/_params/sorter.py:91-112 reads:
```
class Kilosort4Schema(BaseModel):
    ...
    model_config = ConfigDict(extra="allow")
    schema_version: int = 1
    Th_universal: float = Field(default=9.0, gt=0.0)
   

### params (cross)#4  [MEDIUM | NEW-V2] v2 adds join_window_ms field with no v1 equivalent
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:69-87 (no join_window_ms in either preset)`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:70`
- v1 behavior: v1 has no concept of joining nearby artifact intervals — removal_window_ms is the only window-shaping knob (v1/artifact.py:324 via add_removal_window).
- v2 behavior: ArtifactDetectionParamsSchema.join_window_ms: float = Field(default=1.0, ge=0.0) — described as 'artifact intervals closer than this are merged into one'. This is a NEW field with no v1 default.
- documented rationale: No inline rationale comment in the schema for the choice of 1.0 ms specifically. Field is mentioned in the schema docstring (_params/artifact_detection.py:21) as a v2 addition but no quantitative justification.
- verifier reasoning: Finding holds up under direct code inspection.

(1) v2 schema field at src/spyglass/spikesorting/v2/_params/artifact_detection.py:70 reads `join_window_ms: float = Field(default=1.0, ge=0.0)`. Schema docstring at lines 18-21 calls it "artifact intervals closer than this are merged into one" and explicitly labels it a v2 addition.

(2) v1 has NO equivalent. Both v1 presets at src/spyglass/spikesorting/v1/artifact.py:65-87 (`default` and `none`) contain only `zscore_thresh`, `amplitude_thresh_uV`, `proportion_above_thresh`, `removal_window_ms`, `chunk_duration`, `n_jobs`, `progress_bar`. The `_g

### params (cross)#5  [MEDIUM | INT-JUST] v2 renames v1 keys frequency_min/frequency_max → freq_min/freq_max
- **v1**: `src/spyglass/spikesorting/v1/recording.py:131-132`
- **v2**: `src/spyglass/spikesorting/v2/_params/preprocessing.py:22-23`
- v1 behavior: v1 default preprocessing row uses keys 'frequency_min' and 'frequency_max' (v1/recording.py:131-132) and reads them via filter_params['frequency_min'] (v1/recording.py:623).
- v2 behavior: v2 PreprocessingParamsSchema.BandpassFilterParams uses keys 'freq_min' and 'freq_max' (matching SI's sip.bandpass_filter signature). Custom v1 rows would fail Pydantic validation on insert into v2.
- documented rationale: BandpassFilterParams docstring at _params/preprocessing.py:9-19 documents that v1 always REQUIRED the keys and v2 promotes them to schema-level defaults; the rename to match SI is implicit but consistent with extra='forbid' policy.
- verifier reasoning: Verified all cited lines directly.

v1/recording.py:131-132 confirms the default row uses 'frequency_min'/'frequency_max':
    "frequency_min": 300,  # high pass filter value
    "frequency_max": 6000,  # low pass filter value

v1/recording.py:623-624 confirms v1 reads these keys from the blob:
    freq_min=filter_params["frequency_min"],
    freq_max=filter_params["frequency_max"],
(v1 itself translates to SI's freq_min/freq_max API at the call site.)

v2/_params/preprocessing.py:21-23 confirms the rename plus extra='forbid':
    model_config = ConfigDict(extra="forbid")
    freq_min: float =

### params (cross)#6  [LOW | INT-JUST] v2 ArtifactDetectionParamsSchema rejects detect=True with both thresholds None; v1 silently skips
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:69-87 (presets), v1/artifact.py:256-264 (runtime skip)`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:53 (detect field), :73-80 (validator), v2/artifact.py:118-120 ('none' preset)`
- v1 behavior: _get_artifact_times (v1/artifact.py:256-264) checks `if amplitude_thresh_uV is zscore_thresh is None` and silently SKIPS detection at runtime, returning the unmasked recording. v1's 'none' preset (v1/artifact.py:78-87) relies on this fall-through.
- v2 behavior: ArtifactDetectionParamsSchema._check_thresholds (v2/_params/artifact_detection.py:73-80) raises ValueError at insert time if detect=True and both thresholds are None. The 'none' preset works via an explicit detect=False field (v2/artifact.py:118-120).
- documented rationale: Implicit from the 'detect=False' design + extra='forbid' on the schema; no inline rationale comment beyond field naming.
- verifier reasoning: Verified all claims at the cited lines:

**v1 silent-skip behavior confirmed** at `src/spyglass/spikesorting/v1/artifact.py:255-264`:
```python
# if both thresholds are None, we skip artifract detection
if amplitude_thresh_uV is zscore_thresh is None:
    if verbose:
        logger.info(
            "Amplitude and zscore thresholds are both None, "
            + "skipping artifact detection"
        )
    return np.asarray(
        [valid_timestamps[0], valid_timestamps[-1]]
    ), np.asarray([])
```
This is a runtime fall-through: returns the full session as one valid interval, no detection.


### params (cross)#7  [LOW | INT-JUST] v2 CommonReferenceParams drops v1's 'reference' field and exposes 'operator' as a user knob
- **v1**: `src/spyglass/spikesorting/v1/recording.py:597-619 (no 'reference' field consumed; operator hardcoded median)`
- **v2**: `src/spyglass/spikesorting/v2/_params/preprocessing.py:35-56`
- v1 behavior: v1's preprocessing default row has no 'reference' field but the runtime dispatches based on sort_reference_electrode_id only (recording.py:597-619): -1 = no reference, -2 = global median, positive = single-channel. operator='median' is hardcoded at recording.py:611.
- v2 behavior: v2 drops the inert 'reference' field entirely (extra='forbid' rejects it). 'operator' is exposed as a Literal['median', 'average'] with default 'median' — passing 'average' is a v2-only capability.
- documented rationale: Documented at _params/preprocessing.py:38-57.
- verifier reasoning: Verified by direct code reading.

v1 default preproc_params row (src/spyglass/spikesorting/v1/recording.py:127-138) contains only frequency_min/frequency_max/margin_ms/seed/min_segment_length — NO 'reference' field. The default row never carried it.

v1 runtime dispatch (recording.py:597-619) branches purely on ref_channel_id (the sort_reference_electrode_id column, not filter_params):
- `if ref_channel_id >= 0`: `reference="single"`, ref_channel_ids=ref_channel_id
- `elif ref_channel_id == -2`: `reference="global"`, `operator="median"` HARDCODED at line 611
- `elif ref_channel_id != -1`: rais

### params (cross)#8  [LOW | INT-JUST] v2 adds min_length_s field to artifact schema (was hardcoded in v1)
- **v1**: `src/spyglass/spikesorting/v1/artifact.py:327-328`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:71`
- v1 behavior: v1's _get_artifact_times calls sort_interval_valid_times.subtract(artifact_intervals_s, min_length=1) (v1/artifact.py:327-328) — the 1-second minimum length is hardcoded with no user override.
- v2 behavior: ArtifactDetectionParamsSchema.min_length_s: float = Field(default=1.0, gt=0.0). Default 1.0 matches v1's hardcoded value; users can override per-row.
- documented rationale: Documented at _params/artifact_detection.py:37-43.
- verifier reasoning: All four cited claims verified directly in source.

1. v1/artifact.py:327-328 confirmed: `sort_interval_valid_times.subtract(artifact_intervals_s, min_length=1).union_consolidate()` — hardcoded `min_length=1`, no user override.

2. v2/_params/artifact_detection.py:71 confirmed: `min_length_s: float = Field(default=1.0, gt=0.0)` — schema-tunable with default matching v1's hardcoded value.

3. v2/_params/artifact_detection.py:37-43 confirmed: `schema_version: int = 2` with explicit comment "Bumped to 2 by adding ``min_length_s``: the artifact-removed valid_times are filtered to drop slivers shor

### params (cross)#9  [LOW | INT-JUST] v2 drops v1 preprocessing fields 'margin_ms' and 'seed' from the schema
- **v1**: `src/spyglass/spikesorting/v1/recording.py:131-135 (default row), v1/recording.py:621-626 (only freq_min/freq_max used)`
- **v2**: `src/spyglass/spikesorting/v2/_params/preprocessing.py:74-122 (no margin_ms/seed fields), src/spyglass/spikesorting/v2/sorting.py:1281 (whitening seed pinned to 0)`
- v1 behavior: SpikeSortingPreprocessingParameters 'default' row contains margin_ms=5 and seed=0 in its params blob. v1's filter_params accessor reads only frequency_min/frequency_max (recording.py:623-624), so margin_ms and seed are never actually consumed — they are documented but inert.
- v2 behavior: PreprocessingParamsSchema has no margin_ms field and no seed field; the seed for the externalized whitening is hard-coded to 0 (overrideable via job_kwargs.random_seed) at v2/sorting.py:1281.
- documented rationale: Schema-first design rationale: 'Removed CommonReferenceParams.reference -- dead field' documented in PreprocessingParamsSchema docstring (_params/preprocessing.py:88-93). Same principle applies to margin_ms/seed though it is not called out as explicitly.
- verifier reasoning: All claims in the finding hold up under direct code inspection.

(1) v1 default row at src/spyglass/spikesorting/v1/recording.py:127-138 contains `"margin_ms": 5` and `"seed": 0` in the params blob.

(2) v1's filter_params accessor at src/spyglass/spikesorting/v1/recording.py:621-626 only consumes `filter_params["frequency_min"]` and `filter_params["frequency_max"]`:
    recording = si.preprocessing.bandpass_filter(
        recording,
        freq_min=filter_params["frequency_min"],
        freq_max=filter_params["frequency_max"],
        dtype=np.float64,
    )
Grep across v1/recording.py con

### params (cross)#10  [LOW | INT-JUST] v2 promotes proportion_above_thresh constraints from runtime warnings to insert-time errors
- **v1**: `src/spyglass/spikesorting/utils.py:242-249`
- **v2**: `src/spyglass/spikesorting/v2/_params/artifact_detection.py:68`
- v1 behavior: v1's _check_artifact_thresholds (utils.py:242-249) warns and silently clamps proportion_above_thresh to 0.01 if <0, and clamps to 1.0 if >1. Runtime fallback, no insert-time rejection.
- v2 behavior: ArtifactDetectionParamsSchema.proportion_above_thresh: float = Field(default=1.0, gt=0.0, le=1.0) — rejects 0.0 and >1.0 at INSERT time (not runtime). Stricter contract.
- documented rationale: Implicit: schema-first design philosophy. No inline rationale comment.
- verifier reasoning: Verified directly against both files.

v1 at src/spyglass/spikesorting/utils.py:242-256:
```
if proportion_above_thresh < 0:
    warnings.warn(... "Using proportion_above_thresh = 0.01 instead of ...")
    proportion_above_thresh = 0.01
elif proportion_above_thresh > 1:
    warnings.warn(... "Using proportion_above_thresh = 1 instead of ...")
    proportion_above_thresh = 1
```
This is a runtime warn-and-clamp, not an insert-time check.

v2 at src/spyglass/spikesorting/v2/_params/artifact_detection.py:68:
```
proportion_above_thresh: float = Field(default=1.0, gt=0.0, le=1.0)
```
Pydantic reje

## consumers (cross) (8 findings)

### consumers (cross)#1  [HIGH | INT-JUST] CurationV2.get_sort_group_info returns ALL electrodes per sort group; v1 returns one row per group (limit=1)
- **v1**: `src/spyglass/spikesorting/v1/curation.py:269-302`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:1193-1236`
- v1 behavior: CurationV1.get_sort_group_info builds electrode_restrict_list by iterating sort groups and doing `(SortGroup.SortGroupElectrode & entry * Electrode).fetch(limit=1)` -- one electrode per group -- then joins to BrainRegion and returns a dj.Table with one (group, region) row.
- v2 behavior: CurationV2.get_sort_group_info returns the relation `(SortGroupV2.SortGroupElectrode & sg_restriction) * Electrode * BrainRegion` -- one row per electrode, not per group. Multi-electrode probes produce multiple rows.
- documented rationale: Documented in v2/curation.py:1196-1201 as a deliberate fix for v1's multi-region under-reporting. The shape change is a breaking API change for any caller relying on single-row return; no migration shim provided.
- verifier reasoning: All claims verified in code.

v1 (src/spyglass/spikesorting/v1/curation.py:288-295) explicitly iterates sort groups and does `((SortGroup.SortGroupElectrode() & entry) * Electrode).fetch(limit=1)` -- one row per group. The trailing `(cls & key).proj() * sort_group_info` returns that join.

v2 (src/spyglass/spikesorting/v2/curation.py:1234-1236) returns `(SortGroupV2.SortGroupElectrode & sg_restriction) * _Electrode * BrainRegion` with NO limit -- one row per electrode. Docstring at line 1195-1207 explicitly says: "Fix for the v1 ``fetch(limit=1)`` multi-region under-reporting bug ... returns a

### consumers (cross)#2  [MEDIUM | UNTESTED] test_get_restricted_merge_ids_default_sources_includes_v2 asserts wrong default; signature is sources=None not [v0, v1, v2]
- **v1**: `n/a`
- **v2**: `tests/spikesorting/v2/test_v1_parity.py:374-380; src/spyglass/spikesorting/spikesorting_merge.py:71-83, 298-334`
- v1 behavior: N/A -- this is a v2-side test for the merge dispatcher API.
- v2 behavior: Test at tests/spikesorting/v2/test_v1_parity.py:374-380 asserts `sig.parameters['sources'].default == ['v0', 'v1', 'v2']`. But the actual signature at spikesorting_merge.py:301 is `sources: list | None = None`; the literal `['v0','v1','v2']` is built INSIDE the function via `_available_merge_sources()` (line 71-83) which conditionally appends 'v2'.
- documented rationale: No documented rationale for the asymmetry. The test appears stale -- written against an earlier branch where the default was a literal list -- and was not updated when the lazy `_available_merge_sources()` indirection was added.
- verifier reasoning: Direct read confirms the finding. At src/spyglass/spikesorting/spikesorting_merge.py:301 the signature is `sources: list | None = None`. The literal `["v0", "v1", "v2"]` does not appear in the signature; it is built lazily inside the function via `_available_merge_sources()` at lines 71-83 (`sources = ["v0", "v1"]; if CurationV2 is not None: sources.append("v2"); return sources`), invoked at line 334 only when `sources is None`. The test at tests/spikesorting/v2/test_v1_parity.py:374-380 asserts `sig.parameters["sources"].default == ["v0", "v1", "v2"]` — under inspect.signature this default is

### consumers (cross)#3  [MEDIUM | INT-JUST] v2 CurationV2 omits NWB `curation_label` column when all labels are empty; v1 always writes column when labels arg is not None
- **v1**: `src/spyglass/spikesorting/v1/curation.py:391-403`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:857-876`
- v1 behavior: `_write_sorting_to_nwb_with_curation` at v1/curation.py:391-403 writes the `curation_label` indexed column whenever `labels is not None` -- even if every per-unit label list is empty. With `labels={}` v1 writes one empty list per unit.
- v2 behavior: `_stage_curated_units_nwb` at v2/curation.py:857-876 only adds the `curation_label` column when `any(all_labels)` is true. With `labels={}` or all-empty-lists v2 writes NO column.
- documented rationale: pynwb dtype-inference limitation documented in source. The shape divergence on edge cases is not documented as a contract for downstream consumers; the test corpus does not include a v1-parity test that constructs the same labels arg on both pipelines and compares the resulting NWB column presence.
- verifier reasoning: Verified directly against source.

v1 at src/spyglass/spikesorting/v1/curation.py:391-403 unconditionally writes the `curation_label` indexed column whenever `labels is not None`. The inner loop appends `[]` for unit_ids missing from the dict, so `labels={}` produces a column of empty lists, one per unit:
```
if labels is not None:
    label_values = []
    for unit_id in unit_ids:
        if unit_id not in labels:
            label_values.append([])
        else:
            label_values.append(labels[unit_id])
    nwbf.add_unit_column(name="curation_label", ..., data=label_values, index=True

### consumers (cross)#4  [MEDIUM | INT-JUST] v2 does not write the `merge_groups` NWB column; v1 stores merge groups as a units-table column
- **v1**: `src/spyglass/spikesorting/v1/curation.py:404-415, 258-265`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:821-888, 1067-1109`
- v1 behavior: v1/curation.py:404-415 writes `merge_groups` as an `index=True` (ragged) column on the units table whenever `merge_groups is not None`. v1/curation.py:258-265 (`get_merged_sorting`) and v1/figurl_curation.py:85 read this column to recover the merge groups directly from the NWB.
- v2 behavior: v2/curation.py:_stage_curated_units_nwb writes ONLY `spike_times` and (conditionally) `curation_label` columns. Merge groups live exclusively in the DataJoint `CurationV2.MergeGroup` part table; `get_merged_sorting` reads from the part table, never from NWB.
- documented rationale: Schema rationale documented in curation.py:101-124 (queryable bulk-audit + provenance retrieval). Since no cross-pipeline consumer reads the NWB column, this is API-compatible; however, any user-script that hand-rolled `nwb_file['object_id'].get('merge_groups')` on v1 NWBs will silently see empty results on v2 NWBs.
- verifier reasoning: Verified the finding by reading both implementations directly.

V1 WRITES merge_groups NWB column (src/spyglass/spikesorting/v1/curation.py:404-415):
```
if merge_groups is not None:
    merge_groups_dict = _list_to_merge_dict(merge_groups, unit_ids)
    merge_groups_list = [...]
    nwbf.add_unit_column(
        name="merge_groups",
        description="merge groups",
        data=merge_groups_list,
        index=True,
    )
```

V1 READS it from NWB to reconstruct merges (src/spyglass/spikesorting/v1/curation.py:258-261):
```
merge_groups = nwb_sorting.get("merge_groups")
if merge_groups:
  

### consumers (cross)#5  [MEDIUM | INT-JUST] v2 raises on unknown restriction keys; v1 silently drops them -- asymmetric semantics across mixed-source calls
- **v1**: `src/spyglass/spikesorting/spikesorting_merge.py:250-296`
- **v2**: `src/spyglass/spikesorting/spikesorting_merge.py:189-197, 364-371`
- v1 behavior: `_get_restricted_merge_ids_v1` at spikesorting_merge.py:250-296 silently ignores restriction keys that don't match any v1 attribute (DataJoint restriction with unknown attribute is a no-op on the projected table). A typo like `recording_id=...` against a v1 chain that doesn't carry recording_id would return the unrestricted v1 set.
- v2 behavior: `_get_restricted_merge_ids_v2` at spikesorting_merge.py:190-197 explicitly checks `unknown = set(key) - allowed` and raises ValueError listing the offenders. The `allowed` set is the union of rec_keys, sort_keys, curation_keys.
- documented rationale: Intentional v2 hardening documented in source. The asymmetry across sources is not explicitly addressed -- a mixed call short-circuits on v2's raise before v1 results can be returned, even when the user only cared about v1 rows.
- verifier reasoning: Verified directly against source.

v2 raise behavior (confirmed at src/spyglass/spikesorting/spikesorting_merge.py:190-197):
```
allowed = set(rec_keys) | set(sort_keys) | set(curation_keys)
unknown = set(key) - allowed
if unknown:
    raise ValueError(
        "SpikeSortingOutput._get_restricted_merge_ids_v2: "
        f"unknown restriction keys {sorted(unknown)}. Allowed: "
        f"{sorted(allowed)}."
    )
```

v1 helper (lines 250-296) performs NO equivalent unknown-key check; it only does `table & key` style restrictions on join projections, so unknown keys are handled by DataJoint's re

### consumers (cross)#6  [LOW | INT-UNJ] CurationV2.insert_curation returns PK-only dict; CurationV1.insert_curation returns the full row
- **v1**: `src/spyglass/spikesorting/v1/curation.py:117-128`
- **v2**: `src/spyglass/spikesorting/v2/curation.py:495`
- v1 behavior: CurationV1.insert_curation at v1/curation.py:117-128 returns `{sorting_id, curation_id, parent_curation_id, analysis_file_name, object_id, merges_applied, description}` -- the full row dict.
- v2 behavior: CurationV2.insert_curation at v2/curation.py:495 returns `{"sorting_id": ..., "curation_id": ...}` -- PK-only.
- documented rationale: no documented rationale; the docstring states the new contract but does not explain why it differs from v1.
- verifier reasoning: Verified both return statements directly.

v1 (src/spyglass/spikesorting/v1/curation.py:117-128) on the new-row path returns the full row dict:
```python
key = {
    "sorting_id": sorting_id,
    "curation_id": curation_id,
    "parent_curation_id": parent_curation_id,
    "analysis_file_name": analysis_file_name,
    "object_id": object_id,
    "merges_applied": apply_merge,
    "description": description,
}
cls.insert1(key, skip_duplicates=True)
return key
```

v2 (src/spyglass/spikesorting/v2/curation.py:495) returns PK-only:
```python
return {"sorting_id": sorting_id, "curation_id": curati

### consumers (cross)#7  [LOW | INT-JUST] Sorting table `object_id` is varchar(72) in v2 vs varchar(40) in v1 SpikeSorting; both share the same NWB-object-key dict shape after `_get_nwb_object` substring trim, but inserting a v1 SpikeSorting analysis_file_name's object_id into v2 schema requires 72-char tolerance
- **v1**: `src/spyglass/spikesorting/v1/sorting.py:238; src/spyglass/spikesorting/v1/curation.py:38`
- **v2**: `src/spyglass/spikesorting/v2/sorting.py:426; src/spyglass/spikesorting/v2/curation.py:60`
- v1 behavior: v1/sorting.py:238 declares `object_id: varchar(40)` on SpikeSorting; v1/curation.py:38 declares `object_id: varchar(72)` on CurationV1.
- v2 behavior: v2/sorting.py:426 declares `object_id: varchar(72)`; v2/curation.py:60 declares `object_id: varchar(72)`.
- documented rationale: Implicit consistency with CurationV1's pre-existing varchar(72) width; not explicitly documented.
- verifier reasoning: Directly verified all four cited declarations: v1/sorting.py:238 (`object_id: varchar(40)`), v1/curation.py:38 (`object_id: varchar(72)`), v2/sorting.py:426 (`object_id: varchar(72)`), v2/curation.py:60 (`object_id: varchar(72)`). The `_get_nwb_object` dispatcher at utils/dj_helper_fn.py:429-435 keys on `id_attr.replace("_object_id", "")` — both v1 and v2 use the literal column name `object_id` (no prefix), so the dict key downstream is identical (`object_id`) and `_get_nwb_object` resolves via `nwbf.objects[object_id]` the same way. A codebase-wide grep confirms `varchar(40)` is the dominant 

### consumers (cross)#8  [LOW | INT-JUST] v2 only includes the CurationV2 part-table when v2 imports successfully -- conditional declaration ties FK availability to runtime import gate
- **v1**: `src/spyglass/spikesorting/spikesorting_merge.py:105-110`
- **v2**: `src/spyglass/spikesorting/spikesorting_merge.py:36-44, 126-133`
- v1 behavior: v1's CurationV1 part on SpikeSortingOutput at spikesorting_merge.py:105-110 is ALWAYS declared.
- v2 behavior: v2's CurationV2 part at spikesorting_merge.py:126-133 is conditional: `if CurationV2 is not None: class CurationV2(...)`. In v0/v1-only deployments or when the non-localhost-DB safety guard blocks the v2 import, the part is missing entirely.
- documented rationale: Documented schema-compile-time FK constraint.
- verifier reasoning: Verified all claims by reading the actual code.

(1) v1 part-table unconditional declaration confirmed at src/spyglass/spikesorting/spikesorting_merge.py:105-110:
```
class CurationV1(SpyglassMixin, dj.Part):  # noqa: F811
    definition = """
    -> master
    ---
    -> CurationV1
    """
```

(2) v2 part-table conditional declaration confirmed at spikesorting_merge.py:126-133:
```
if CurationV2 is not None:
    class CurationV2(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CurationV2
        """
```

(3) Rationale documented explici

## Untested branches (70)

### artifact.py (15 branches)

- [HIGH] `src/spyglass/spikesorting/v2/artifact.py:470-475` — ArtifactSelection.insert_selection raises DuplicateSelectionError when the join surfaces multiple master rows for the same logical identity (the 'len(unique) > 1' branch).
    why risky: The DuplicateSelectionError test at test_single_session_pipeline.py:5384 covers RecordingSelection, NOT ArtifactSelection. If a raw dj.insert1 (bypassing the helper) plants duplicate master rows + source parts, ArtifactSelection.insert_selection should raise; instead a silent regression to 'return first match' would let downstream Sorting/Curation consumers index a non-deterministic master, producing different artifact_ids per call.
- [HIGH] `src/spyglass/spikesorting/v2/artifact.py:276-284` — SharedArtifactGroup.insert_group raises ValueError when members span more than one session ('members span N sessions').
    why risky: No test creates members across two nwb_file_name sessions (the smoke fixture is single-session, and test_shared_artifact_group_insert_rejects_mismatched_durations / _dtypes both reuse the same nwb_file_name). This invariant is the foundational guarantee that make_compute's aggregate_channels won't crash on incompatible time axes — the per-frame z-score and proportion-above-threshold arithmetic in _detect_artifacts depends on it. If the check silently breaks, populate fails deep inside SI with an opaque shape mismatch.
- [MEDIUM] `src/spyglass/spikesorting/v2/artifact.py:479-486` — _ensure_lookup_row_exists branch: when ArtifactDetectionParameters has no row matching artifact_params_name, insert_selection raises the clear 'run insert_default()' ValueError instead of an opaque FK IntegrityError.
    why risky: No test grep-matches '_ensure_lookup_row_exists' for ArtifactSelection nor 'required Lookup row not found' / 'insert_default()' messages tied to ArtifactDetectionParameters. Every test calls ArtifactDetectionParameters.insert_default() in setup before insert_selection, so this user-facing diagnostic is dead code from the test suite's perspective. If a refactor removes the pre-check, fresh installs hit cryptic 'foreign key constraint fails' errors.
- [MEDIUM] `src/spyglass/spikesorting/v2/artifact.py:290-300` — SharedArtifactGroup.insert_group raises ValueError when members have differing sampling_frequency values.
    why risky: The duration/n_samples mismatch (line 328-337) and dtype mismatch (line 339-344) branches each have dedicated tests, but no test surfaces members with distinct sampling_frequency. A drift here would let SI.aggregate_channels crash at populate time with an opaque error — exactly the failure mode this guard was designed to prevent (the same pattern as the documented n_samples / dtype checks).
- [MEDIUM] `src/spyglass/spikesorting/v2/artifact.py:1004` — _detect_artifacts returns _np.empty((0, 2)) when the sliver-filter (min_length_s threshold) removes every kept interval.
    why risky: OR-mode test at test_single_session_pipeline.py:4250 produces shape (0, 2) by flagging the entire recording (so 'kept' itself is empty BEFORE the sliver filter). No test takes a non-empty kept list and watches the min_length_s filter wipe all entries (e.g., min_length_s greater than every gap between detected artifact runs). Downstream consumers (Sorting._apply_artifact_mask) may not handle a (0, 2) valid_times produced via the sliver path identically to one produced via 'everything flagged' — silent regressions in either path would surface as empty sorts.
- [MEDIUM] `src/spyglass/spikesorting/v2/artifact.py:1167-1171` — ArtifactDetection.delete IntervalList cleanup: 'if len(rows) == 0: continue' branch skips already-missing IntervalList rows, then delegates to rows.delete(safemode=False).
    why risky: Tests cover the happy path (rows present, deleted) and the SchemaBypassError-orphan path (logged-and-skipped). No test exercises the case where resolve_source succeeded but the corresponding IntervalList row was already removed by a separate cleanup script (race). A regression that called rows.delete() on an empty restriction would attempt to cautious_delete the entire IntervalList table.
- [MEDIUM] `src/spyglass/spikesorting/v2/artifact.py:531-535` — prune_orphaned_selections(dry_run=False) branch where 'orphans' is empty: returns empty list without entering the for-loop deletion.
    why risky: test_prune_orphaned_selections_finds_and_cleans (line 1706) injects an orphan first, so the empty-orphans path is not directly tested. A regression that mistakenly cautious_delete'd the master under empty restriction would wipe ArtifactSelection. The 'or not orphans' short-circuit on line 531 is what prevents that.
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:438-442` — ArtifactSelection.insert_selection raises ValueError when 'artifact_params_name' is absent from key (only one source key given, but no params name).
    why risky: A caller that passes recording_id but forgets artifact_params_name gets this targeted error instead of an opaque KeyError. The branch was never grep-matched in tests/spikesorting/v2/ — the 'exactly one source key' test only validates the source-key XOR (lines 633-655), not the master-field check. If the docstring identity of this branch silently changed (e.g., to a KeyError) downstream notebooks would lose the helpful diagnostic.
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:313-322` — SharedArtifactGroup.insert_group wraps Recording.get_recording() failures in a RuntimeError with diagnostic context ('failed to load preprocessed recording for recording_id=...').
    why risky: No test exercises this except block. If a recording_id passes the (Recording & {...}) existence check at lines 233-243 but get_recording() raises (e.g., the cached preprocessed NWB on disk has been moved/deleted), the user should get the diagnostic-wrapped RuntimeError. Without coverage, a refactor that swallows or strips the wrapper would re-introduce opaque pickling/NWB errors at insert time.
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:654-660` — ArtifactDetection.make_fetch raises RuntimeError when SharedArtifactGroup has zero members ('insert_group should have rejected the empty case').
    why risky: The check is a defense-in-depth re-verification of an invariant that insert_group already enforces. No test plants an empty SharedArtifactGroup (via super_delete on Member rows after the master is created) and then calls populate. If insert_group's empty-list check regressed and this guard also regressed silently, ArtifactDetection.make would silently process a 0-channel union or fail later in SI.aggregate_channels.
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:681-683` — ArtifactDetection.make_fetch raises RuntimeError on an unexpected source.kind ('unexpected source kind ...').
    why risky: Catch-all guard — only fires if SourceResolution.kind grows a new variant that make_fetch is not aware of. No test mocks resolve_source to return an unknown kind. If a future contributor adds a third source part (e.g., probe-level shared artifact) and updates resolve_source but forgets the make_fetch dispatch, the failure would only surface at populate time on a real recording.
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:744-749` — ArtifactDetection.make_compute raises RuntimeError when source.kind=='shared_artifact_group' but member_recording_ids is falsy ('make_fetch contract violated').
    why risky: Contract-violation guard against make_fetch returning bad shape. No test calls make_compute directly with the shared-group kind and member_recording_ids=None. Because the integration test (test_shared_artifact_group_populate_end_to_end) always exercises the happy path via populate, a regression that decoupled the two methods could silently feed empty member_recording_ids to an si.aggregate_channels call on an empty list.
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:1028-1032` — get_artifact_removed_intervals raises ValueError when key lacks 'artifact_id'.
    why risky: No test passes a key without artifact_id; the check sits below resolve_source which would also raise (SchemaBypassError) on an empty key. If a caller passes only a partial restriction (e.g., {'shared_artifact_group_name': X}) expecting bulk fetch semantics, the diagnostic error message guides them; without test coverage, a refactor that re-orders the call site could regress to a confusing fetch1 error.
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:885` — _detect_artifacts amplitude_thresh_uV is None branch ('above_amp = zeros'), exercised only when zscore_thresh is set alone.
    why risky: test_detect_artifacts_zscore_only (line 4128) does exercise this path on synthetic data, BUT the combined-mode invariant the code comment depends on ('zeros | above_x == above_x') is not regression-tested via a direct comparison between (amplitude=None, zscore=X) and (amplitude=X, zscore=None) producing matching outputs on a single-channel-amplitude trace. A silent regression to AND-combine semantics inside _detect_artifacts would slip through the OR test (line 4250) because that test only asserts shape (0,2).
- [LOW] `src/spyglass/spikesorting/v2/artifact.py:1156-1159` — delete() safemode dispatch: passes through to super().delete with kwargs preserved when safemode is None vs explicit safemode.
    why risky: Tests cover delete() (line 954, implicit safemode=None) and delete(safemode=False) (lines 5447, 6384, 8329) but NOT delete(safemode=True). Spyglass's cautious_delete reads safemode from dj.config; a regression that always set safemode=False would silently strip the interactive confirmation that safemode=True provides for shared-lab sessions.

### sorting.py (15 branches)

- [HIGH] `src/spyglass/spikesorting/v2/sorting.py:976-985` — Sorting.get_unit_brain_regions concat-source dispatch: when source.kind == 'concatenated_recording' and allow_anchor_member=False, it raises ConcatBrainRegionAmbiguousError; with allow_anchor_member=True it sets resolution='anchor_member' and dispatches to unit_brain_region_df.
    why risky: Neither the error path nor the anchor_member resolution label is exercised in tests (grep finds zero references to ConcatBrainRegionAmbiguousError, allow_anchor_member, or 'anchor_member' in tests/spikesorting/v2/). For concat-backed sorts this is the entire brain-region accessor; a silent rename or label drift (e.g. resolution string typo, missing kwarg) would not surface until a concat-pipeline notebook breaks. Since the concat schema is declared 'final under the zero-migration policy', wrong labels could lock in.
- [HIGH] `src/spyglass/spikesorting/v2/sorting.py:1620-1626` — _populate_unit_part raises RuntimeError 'peak channel ... not in sort group ...' when the analyzer's extremum channel id is not in the sort group's electrode_by_id map.
    why risky: No test exercises this path (grep finds no reference to 'peak channel' or the RuntimeError message). This is the only guard against a sort-group/recording channel-id mismatch silently producing rows pointing to wrong electrodes. If the dict-lookup ever changes (e.g. peak_chan typed as numpy int and lookup keys typed as Python int, or off-by-one channel re-id after probe remap), it could either crash unexpectedly or stop raising on real mismatches. Wrong-electrode unit rows corrupt downstream brain-region joins.
- [HIGH] `src/spyglass/spikesorting/v2/sorting.py:855-922 (_rebuild_analyzer_folder)` — Full rebuild path: triggered by Sorting.get_analyzer when the analyzer folder is missing on disk. Includes a concat-source NotImplementedError, an artifact-mask re-application using a re-fetched IntervalList valid_times, and a try/except that rmtrees a partial folder on rebuild failure.
    why risky: No tests exercise _rebuild_analyzer_folder (grep finds zero matches). Three independent branches lack coverage: (1) the missing-folder trigger inside get_analyzer (line 851-852), (2) the artifact-mask re-application path (line 878-901) which re-issues DB I/O that bypasses make_fetch's pre-fetched obs_intervals, and (3) the partial-folder cleanup on rebuild failure (913-922). A silent change to _analyzer_path semantics or to load_sorting_analyzer error behavior could turn the rebuild path into either a permanent KeyError or a silent stale-folder load.
- [HIGH] `src/spyglass/spikesorting/v2/sorting.py:749-769 (get_sorting, zero-unit branch + ID-cast branch)` — get_sorting returns an empty NumpySorting for zero-unit sorts, AND has an unit_ids = [int(uid) for uid in si_sorting.unit_ids] cast that the docstring says 'would raise on string-typed unit IDs'.
    why risky: The zero-unit empty-NumpySorting path IS covered at line 3518. However the as_dataframe=True path on a zero-unit sort is not (the test only checks .get_num_units()==0). More importantly, the as_dataframe=True path on a NON-empty sort is also not covered: grep finds zero calls of Sorting().get_sorting(..., as_dataframe=True); the only as_dataframe test (test_v1_parity.py:533) just inspects the signature. The pandas DataFrame construction (line 770-796) including the int(uid) cast, the spike_times list comprehension, and the unit_id-indexed DataFrame shape are all unverified behaviorally. A regression that returns rows positionally indexed (the bug the comment warns against) would not be caught.
- [HIGH] `src/spyglass/spikesorting/v2/sorting.py:1146-1167 (clusterless noise_levels=None path with random_seed pin)` — When noise_levels is None in params, the key is removed and random_slices_kwargs={'seed': _random_seed} is installed; _random_seed comes from job_kwargs.get('random_seed', 0).
    why risky: The strip-random_seed test at line 4042 covers noise_levels=[1.0] explicit path (line 4069). The auto-MAD path (noise_levels=None) where the explicit seed pin is installed is not directly verified to write the pinned seed into random_slices_kwargs. A regression that drops the seed pin (or pins to None instead of 0) would silently reintroduce the non-determinism that PR #3359 introduced -- the very bug this code exists to neutralize. The shipped smoke/synthetic-fixture rows omit noise_levels, so this branch IS what production hits, yet it is the less-tested of the two.
- [MEDIUM] `src/spyglass/spikesorting/v2/sorting.py:1610-1618` — _populate_unit_part raises NonIntegerUnitIDError when a sorter emits a unit_id that does not convert to int.
    why risky: Zero test coverage (grep finds no NonIntegerUnitIDError reference in v2 tests). The contract is exception name + helpful message; a future refactor that swaps in a plain TypeError, drops the helpful message, or silently coerces a non-int via int() of a float NaN (raises ValueError that does fall through, but message would not match) would not be caught. Important because external sorters routinely emit string IDs.
- [MEDIUM] `src/spyglass/spikesorting/v2/sorting.py:614-626 (make_compute except / Mode A cleanup)` — When _write_units_nwb raises AFTER _build_analyzer succeeded, the except block rmtrees the analyzer folder before re-raising.
    why risky: The existing rollback test (line 3534-3679) covers Mode B (failure inside make_insert's transaction) by patching _populate_unit_part. It does not cover Mode A: an exception thrown specifically from _write_units_nwb after _build_analyzer wrote a folder. A 5-50 GB analyzer-folder leak per failed NWB write is precisely the failure mode the cleanup guards against. The inner except for cleanup_exc at line 621 is also untested and marked pragma: no cover.
- [MEDIUM] `src/spyglass/spikesorting/v2/sorting.py:323-333 (_ensure_lookup_row_exists call after no-existing-row)` — Before creating a new master row, insert_selection calls _ensure_lookup_row_exists to translate a would-be FK IntegrityError into a clear 'missing default row' message.
    why risky: Zero coverage (grep finds no test for the missing-default-row path in v2/sorting.py). A user who skips SorterParameters.insert_default() and immediately calls insert_selection should get the helpful error rather than a raw IntegrityError. Refactors to _ensure_lookup_row_exists semantics (return-vs-raise, message format) could regress silently.
- [MEDIUM] `src/spyglass/spikesorting/v2/sorting.py:1313-1319 (MATLAB sorter carve-out)` — When sorter.lower() in _MATLAB_SORTERS (kilosort2_5, kilosort3, ironclust), run_kwargs['singularity_image']=True is set AND a strip-kwargs filter removes tempdir/mp_context/max_threads_per_process from sorter_params.
    why risky: Zero test coverage (grep finds no test references to _MATLAB_SORTERS, kilosort2_5, kilosort3, ironclust, or singularity_image in v2 tests). If a user inserts a custom SorterParameters row for one of these MATLAB sorters and one of the stripped kwargs is silently dropped (or, conversely, a new kwarg should be stripped but isn't), the sort either fails inside the container or runs with wrong params. The docstring explicitly says users 'can extend this set by subclassing' but the extension mechanism is not tested either.
- [MEDIUM] `src/spyglass/spikesorting/v2/sorting.py:1029-1030 (artifact mask empty-frame-ranges short-circuit)` — _apply_artifact_mask returns the unmodified recording when frame_ranges is empty (no artifact gaps fall within the recording window).
    why risky: The general artifact-mask test (line 2905-2998) constructs a single carved-out gap; it does not exercise the empty-frame-ranges case where valid_times completely cover the recording (so no artifact frames remain). This is the 'no-op' code path and a regression that erroneously feeds an empty np.concatenate to sip.remove_artifacts could silently raise or alter the signal. Important because the 'none' artifact preset writes exactly this shape (single all-valid interval), and Sorting routinely runs against it.
- [MEDIUM] `src/spyglass/spikesorting/v2/sorting.py:1168-1176 (noise_levels scalar broadcast)` — When noise_levels is supplied as a singleton (e.g. [1.0]), it is broadcast via np.full to length recording.get_num_channels() because SI's locally_exclusive indexes noise_levels[chan] per channel.
    why risky: No test directly verifies the broadcast (grep finds no test referencing nl.size == 1, np.full, or broadcast-to-n_channels in tests). The detect-peaks test at line 4069 supplies noise_levels=[1.0] but only checks the random_seed stripping, not whether the array reaches SI as length n_channels. A change that drops the broadcast (e.g. asarray without the size==1 check) would crash SI mid-detection only on real multi-channel recordings, not on a single-channel smoke test.
- [LOW] `src/spyglass/spikesorting/v2/sorting.py:262-276 (insert_selection both-sources gate + concat NotImplementedError)` — insert_selection raises ValueError on zero/both source keys, and NotImplementedError when concat_recording_id is supplied.
    why risky: The both-sources/zero-sources ValueError IS covered (test at line 635-648). The concat NotImplementedError IS covered (test at line 5081). However the message 'recording_id={set/unset}' formatting is unverified for the both-sources case (the test passes neither; there is no test that passes both). A user passing both keys would get the ValueError but the helpful 'both supplied' diagnostic could silently drift.
- [LOW] `src/spyglass/spikesorting/v2/sorting.py:278-283 (required-keys validation)` — insert_selection raises ValueError when 'sorter' or 'sorter_params_name' is missing from the key.
    why risky: No test covers the missing-sorter or missing-sorter_params_name branch (grep finds no test matching 'requires.*sorter' for this path). Users who forget either key currently get a clear error; a refactor that defers the check to the FK insertion would produce a confusing IntegrityError. Low impact because the error path is simple, but it is the only contract-enforcement before the FK insert.
- [LOW] `src/spyglass/spikesorting/v2/sorting.py:1257-1258 (numpy.Inf alias restore)` — When sorter == 'mountainsort4' and numpy lacks the 'Inf' attribute, _np.Inf = _np.inf is assigned to work around a removed numpy 2.0 alias used by spikeextractors 0.9.11.
    why risky: Zero test coverage (grep finds no references to np.Inf or numpy.Inf in v2 tests). The branch is gated on (a) sorter='mountainsort4', and (b) numpy>=2.0 (the only build where hasattr returns False). If MS4 is dropped from the install matrix, this branch silently becomes dead code; if numpy < 2.0 is the test environment, the workaround is never exercised. A regression in the workaround (wrong attribute name, conditional inversion) only fails on a production NWB sort.
- [LOW] `src/spyglass/spikesorting/v2/sorting.py:924-958 (Sorting.delete with safemode passthrough)` — delete() collects analyzer_folder paths BEFORE the cascade delete, then passes safemode-or-defaults through super().delete(), then rmtrees the collected paths. Includes a safemode is None vs not-None branch.
    why risky: The existing test at line 6059-6099 invokes delete(safemode=False) only. The safemode=None branch (line 952-953) where safemode is omitted -- the default path -- is not tested. The folders_to_remove pre-fetch is also unverified for empty restrictions (delete on empty set). The post-cascade rmtree must check folder.exists() because the analyzer may have been removed elsewhere; the False branch (folder already gone) is uncovered.

### recording.py (14 branches)

- [HIGH] `src/spyglass/spikesorting/v2/recording.py:1114-1117` — Recording.get_recording rebuild path when the cached AnalysisNwbfile is missing on disk: `if not Path(abs_path).exists(): self._rebuild_nwb_artifact(key)`
    why risky: No test deletes the on-disk artifact and re-fetches via get_recording. A regression here (silent rebuild path becoming a hard error, or recomputing into the wrong slot) would only surface in production when a user moved/lost an analysis NWB. The rebuild also re-runs the full preprocessing pipeline, which silently masks data divergence if the raw NWB has changed.
- [HIGH] `src/spyglass/spikesorting/v2/recording.py:1138-1179 (_rebuild_nwb_artifact, hash-mismatch warning at 1171-1179)` — The entire _rebuild_nwb_artifact method and its hash-mismatch warning branch (`if rebuilt_hash != row['cache_hash']`) have no direct test coverage. No test triggers a hash drift to verify the user-facing logger.warning fires with the right content.
    why risky: Hash mismatch is the primary integrity signal for upstream raw-NWB drift or SI version skew. If the comparison silently inverts or the message format changes, users would not see the warning and would silently consume drifted data. Also: the row is intentionally NOT auto-deleted, so a regression that DOES delete it would silently destroy curation lineage.
- [HIGH] `src/spyglass/spikesorting/v2/recording.py:1510-1513 (_spikeinterface_channel_ids channel_name path)` — Two-branch logic mapping Spyglass electrode_ids to SpikeInterface channel ids based on whether the raw NWB carries a `channel_name` column. The test fixtures (MEArec) do not include channel_name, so only the integer-fallback path (line 1511) is exercised; the `channel_names[int(c)]` path (line 1513) is dead in tests.
    why risky: Production Frank-lab NWBs DO carry channel_name; v1 parity at v1/recording.py:683-712 depends on this branch resolving correctly. A regression (e.g. dropping the int() cast, or returning a numpy scalar that SI silently mishandles) would only surface on real lab data, never on the test fixtures.
- [HIGH] `src/spyglass/spikesorting/v2/recording.py:1681-1722 (_maybe_apply_tetrode_geometry gating predicate)` — The 4-condition gate (`len(unique_probes)==1 and probe=='tetrode_12.5' and len(channel_ids)==4 and len(unique_groups)==1`) decides whether the legacy tetrode patch applies. Only the all-true path is exercised by the tetrode_60s fixture; the false-by-any-of-4-reasons fall-through has no negative test (e.g. tetrode with 3 channels after a bad-channel drop, or mixed probe types).
    why risky: Silent geometry attachment on a non-tetrode-12.5 group would feed Kilosort/MS5 the wrong contact positions, producing plausible but wrong sort output. The condition is brittle (string compare 'tetrode_12.5'; len==4 hardcoded) and a regression flipping any inequality would never trip a test.
- [MEDIUM] `src/spyglass/spikesorting/v2/recording.py:1765-1770` — Invalid sort_reference_electrode_id branch in _apply_pre_motion_preprocessing: `elif ref_channel_id != -1: raise ValueError(...)`. Only -1 (skip), -2 (global), and positive ids are exercised; values like -3 or -5 are never tested.
    why risky: SortGroupV2 stores this as a plain int with no schema validation, so a user/automation could plausibly write -3 or -10 (e.g. a typo on a negative sentinel). The guard exists precisely because the chain of elifs silently falls through with no preprocessing applied for any negative value other than -2 -- but no test confirms the guard actually fires.
- [MEDIUM] `src/spyglass/spikesorting/v2/recording.py:338-345` — set_group_by_shank `if not proposed: raise ValueError(...)` branch when every shank is filtered out (e.g. all unitrode + ref-electrode-group skip combination).
    why risky: Tests cover unitrode-skip (line 8267) returning a non-empty result, but none drives the filter all the way to zero proposed groups. A regression that changes the filter ordering (e.g. checking omit_unitrode after omit_ref) could silently produce empty group inserts before the guard catches them.
- [MEDIUM] `src/spyglass/spikesorting/v2/recording.py:313-335 (omit_ref_electrode_group skip branch in set_group_by_shank)` — The omit_ref_electrode_group=True path that skips the electrode group containing the reference electrode. No test sets omit_ref_electrode_group=True, so the `if ref_match.any():` + `if str(ref_group) == str(e_group):` nested logic, including the skipped-list entry with `reason: reference_electrode_group`, is unexercised.
    why risky: The string-comparison `str(ref_group) == str(e_group)` is fragile (electrode_group_name can be numpy string vs Python str). A regression that changes the comparison semantics or fails to populate the skipped entry would silently include the ref-group when the user explicitly asked to omit it.
- [MEDIUM] `src/spyglass/spikesorting/v2/recording.py:1297-1302 (single vs multi-interval saved_times derivation)` — Branch picking timestamps_override vs recording.get_times() based on n_selected_intervals. The single-interval persisted-override path is covered by test_recording_single_interval_persists_repaired_timestamps; the multi-interval `else: saved_times = _get_recording_timestamps(recording)` branch (line 1302) has no test that simultaneously triggers the repair AND the multi-interval concat path.
    why risky: If the multi-interval saved-times derivation accidentally used the override (gap-spanning envelope), the truncation guard at make_insert would spuriously fire on every disjoint sort. Conversely a regression here could silently store wrong duration_s on disjoint-interval rows.
- [MEDIUM] `src/spyglass/spikesorting/v2/recording.py:1311-1323 (cleanup-on-fresh-write-failure in _compute_recording_artifact)` — The `existing_analysis_file_name is None and analysis_file_name is not None` cleanup branch that unlinks the staged file when a post-write step (e.g. _get_recording_timestamps) raises. Cleanup-on-insert-failure (line 1091-1100) is tested; cleanup-on-compute-failure here is not.
    why risky: If gain extraction or timestamp resolution raises after _write_nwb_artifact succeeds, this branch is the only thing preventing an orphan AnalysisNwbfile. A regression (swapping the `is None` to `is not None`) would silently leak files OR delete the canonical rebuild cache.
- [MEDIUM] `src/spyglass/spikesorting/v2/recording.py:204-217 (additive-insert opt-in via explicit non-overlapping sort_group_ids)` — The branch that allows additive insert when `explicit_sort_group_ids and not overlap`. test_set_group_by_shank_refuses_overlapping_rerun (line 80) covers the refusal path; no test exercises a successful additive insert by passing explicit non-overlapping sort_group_ids to set_group_by_shank against an already-populated session.
    why risky: This is the documented escape hatch for rerunning the constructor without delete_existing_entries. A regression here would silently force users into the destructive overwrite path, exactly the v1 regression the gate was designed to fix.
- [LOW] `src/spyglass/spikesorting/v2/recording.py:358-363` — set_group_by_shank length-mismatch validation: `elif len(sort_group_ids) != len(proposed): raise ValueError(...)`. Same check at line 491-496 in set_group_by_electrode_table_column.
    why risky: Grep finds no test passing explicit sort_group_ids with wrong length. This is a user-supplied parameter and the off-by-one is a natural mistake; without coverage the message wording or the trigger condition could regress silently.
- [LOW] `src/spyglass/spikesorting/v2/recording.py:503-508` — set_group_by_electrode_table_column empty-match ValueError: `if len(group_elecs) == 0: raise ValueError(...)` when a user-supplied values sublist matches zero electrodes.
    why risky: Common user error (typo in shank numbering, off-by-one column values). No test verifies the error fires with the expected message. The 8267 test only covers the unitrode skip case where len==1.
- [LOW] `src/spyglass/spikesorting/v2/recording.py:707-713 (RecordingSelection duplicate-rows DuplicateSelectionError)` — Branch raising `DuplicateSelectionError` when `len(existing) > 1` for a logical identity. test_single_session_pipeline.py:5384 tests DuplicateSelectionError but only against SortingSelection, not against RecordingSelection.insert_selection specifically.
    why risky: This guard is the only thing that catches direct dj.insert duplicates bypassing the helper. The error wording calls it 'an integrity bug', meaning the right detection path matters for forensics. No test bypasses the helper to verify the trap fires for RecordingSelection.
- [LOW] `src/spyglass/spikesorting/v2/recording.py:75-87 (_electrode_group_sort_key)` — Sort-key tolerance for non-numeric electrode_group_name strings. test_electrode_group_sort_key directly tests the helper at test_single_session_pipeline.py:2081-2102, but no end-to-end test runs set_group_by_shank against an NWB with non-decimal electrode_group_name (e.g. 'probeA').
    why risky: The unit test catches a regression in the key itself, but a regression in how set_group_by_shank passes the key (e.g. forgetting `key=_electrode_group_sort_key` on line 289) would silently revert to `int()`-fails on the first non-numeric group name. No integration test exercises this on a real NWB shape.

### curation.py (12 branches)

- [HIGH] `src/spyglass/spikesorting/v2/curation.py:306-313` — Invalid `metrics_source` value triggers the `except ValueError` from `MetricsSource(metrics_source)` and is re-raised with a guidance message. Search for 'MetricsSource' and 'metrics_source' in tests/spikesorting/v2/ shows only the default `'manual'` is ever used (one assertion at line 1847 of test_single_session_pipeline.py). No test passes 'analyzer_curation', 'figpack', or an invalid string.
    why risky: Three of the four documented MetricsSource enum values (`analyzer_curation`, `figpack`, plus the typo path) have ZERO behavioral coverage. A regression in the enum coercion (e.g., import failure, accepting bad strings) would only surface when downstream code (figpack_curation, metric_curation stubs) eventually integrates. Schema-first defaults were the cause of the prior 1400x divergence bug; this is the same shape of risk.
- [HIGH] `src/spyglass/spikesorting/v2/curation.py:297-302` — Idempotent-root WARN-AND-RETURN path: when a root curation already exists AND the caller passes NO non-default args (labels/merge_groups/description all falsy), the function logs a warning and returns existing_root[0] without inserting. The companion error path (line 285-296, raises when caller passed args) is tested by test_curation_insert_idempotent_rejects_new_args (line 8376). The all-defaults silent-return branch is referenced in a docstring at line 1891 as `test_root_curation_idempotent` but that test does not exist.
    why risky: A regression could swap the condition (e.g., always raise on repeat, or always return) and silently break user workflows that repeatedly call insert_curation(sorting_key=sk) as a guard. The dangling docstring reference suggests the test was renamed or removed; the branch is now uncovered. Severity bumped because the comment claims this is parity with `v1/curation.py:88-93`.
- [HIGH] `src/spyglass/spikesorting/v2/curation.py:635-641` — Across-group overlap validation raises when the same unit appears in two merge groups (e.g., merge_groups=[[0,1],[1,2]]). test_curation_rejects_invalid_merge_groups (line 4924) exercises the shape/duplicate-member/missing-id paths but NOT the across-group overlap path. No grep hit for 'overlap' or 'belong to at most one merge group'.
    why risky: Across-group overlap is silently corrupting: without the guard, contributor 1 would be absorbed into TWO merged units, double-counting spikes and ambiguating the kept-unit choice. This is a true data-integrity guard; a regression would not crash but produce subtly wrong curated outputs. Pattern matches the prior 'mutable defaults silently diverge' class of v2 bugs.
- [HIGH] `src/spyglass/spikesorting/v2/curation.py:931-938 and curation.py:1218-1222 and curation.py:754-759` — Three separate `NotImplementedError` raises for concat-source sortings: in `get_recording` (931), `_stage_curated_units_nwb` (754), and `get_sort_group_info` (1218). The only concat-source test (test_sorting_selection_rejects_concat_source at line 5082) exercises the SortingSelection layer NOT these three CurationV2 surfaces. No grep hit for 'concat-source sorts are not yet supported' or analogous CurationV2-side concat assertions.
    why risky: If a future commit wires up concat sortings through SortingSelection without these three guards firing, CurationV2 would silently try to fetch a non-existent `recording_id` and either crash with a confusing fetch1 error or produce a malformed curation. The three guards are the only line of defense at the CurationV2 surface; one wrong edit removes correctness without breaking any test.
- [HIGH] `src/spyglass/spikesorting/v2/curation.py:1148-1159 (get_unit_brain_regions: include_labels filter)` — When `include_labels` is supplied, restricts `unit_restriction` to units carrying at least one of the listed labels by joining UnitLabel. Grep for 'include_labels' across tests returns zero hits. The only get_unit_brain_regions coverage (test_curation_v2_auto_registers_in_merge_table at line 2022) goes through the merge dispatcher with the default include_labels=None.
    why risky: Documented feature gated by a parameter with NO behavioral test. A regression in the join logic (e.g., changing AND vs OR semantics, missing units that carry an excluded plus included label) would silently change which units a paper attributes to which regions. This is the kind of indexing/masking change the scientific-code-change-audit warns about: plausible-looking numbers can be wrong.
- [HIGH] `src/spyglass/spikesorting/v2/curation.py:1136-1146 (get_unit_brain_regions: ConcatBrainRegionAmbiguousError + allow_anchor_member)` — Concat-source guard: raises ConcatBrainRegionAmbiguousError unless `allow_anchor_member=True`, in which case it sets resolution='anchor_member'. Greps for 'allow_anchor_member' and 'ConcatBrainRegionAmbiguousError' across tests/spikesorting/v2/ return zero hits.
    why risky: Both branches (the raise AND the anchor-member opt-in) are uncovered. The error path is supposed to prevent silently wrong brain-region assignments on concat-backed sortings; the opt-in is the documented escape hatch. A regression that swaps the conditional or drops the exception would let users silently produce wrong per-unit region assignments on concat sorts -- the exact correctness failure mode the exception was added to prevent.
- [MEDIUM] `src/spyglass/spikesorting/v2/curation.py:234-238` — ValueError raised when sorting_id has no rows in Sorting (both Sorting.Unit AND Sorting are empty). Tests grep negative for the string 'Populate Sorting first' or any insert_curation call with a non-existent sorting_id.
    why risky: If a caller passes a bogus sorting_id, this guard is supposed to fail loudly with a clear instruction. A regression that breaks the message (or the check entirely) would either crash deeper in `_build_curated_unit_rows`/`_stage_curated_units_nwb` with a less helpful error, or silently produce a malformed curation. Tests should pass `sorting_key={'sorting_id': <missing-uuid>}` and assert this specific ValueError.
- [MEDIUM] `src/spyglass/spikesorting/v2/curation.py:509-514` — _validate_labels rejects when `labels[unit_id]` is not a list/tuple (e.g., a bare string like `{0: 'mua'}` instead of `{0: ['mua']}`). Greps for 'list of labels', 'must be a list', and label-value-typing all return no test hits.
    why risky: Easy user mistake: callers from a v1 workflow often pass `labels={uid: 'mua'}` (scalar). Without this guard a regression would let the bare string into `_build_curated_unit_rows`/NWB write and then crash deep with an unhelpful 'CurationLabel value' or iterator error. Defense-in-depth principle: this validation is the only barrier.
- [MEDIUM] `src/spyglass/spikesorting/v2/curation.py:1086-1097 (get_merged_sorting)` — Two early-return branches in `get_merged_sorting`: (a) line 1088-1089 returns base verbatim when `merges_applied=True`; (b) line 1096-1097 returns base when no merge group has >1 contributor. The only `get_merged_sorting` test (line 4784) exercises the apply_merge=False + non-trivial merge path. No test calls `get_merged_sorting` on an `apply_merge=True` curation or on one with no real merge groups.
    why risky: If branch (a) regresses (e.g., the bool() check inverts), v2 would re-apply MergeUnitsSorting over already-merged units, asking SI to merge contributors that no longer exist in the base sorting -- raising an opaque SI error. If branch (b) regresses, every unmerged curation would unnecessarily wrap MergeUnitsSorting with empty units_to_merge, possibly accepted by SI but a perf regression. Both are silent: same correctness shape as the 1400x defaults bug.
- [MEDIUM] `src/spyglass/spikesorting/v2/curation.py:864-876 (curation_label column add path)` — The `if any(all_labels)` branch -- add the indexed curation_label column ONLY when at least one unit has non-empty labels. The 'any labels present' branch is covered by test_curation_label_post_curation_is_indexed_ragged_list (line 6004). The companion 'all-empty labels' branch (skip add_unit_column entirely) is implicitly exercised when labels={} but no test asserts the column is ABSENT in that case; an external reader doing `nwb_sorting.get('curation_label', [])` (cited in the code comment at line 826-828) depends on the column being missing, not empty.
    why risky: If a regression changes the condition to `if all_labels:` (always true since `all_labels` is a non-empty list-of-lists), pynwb would try to write an all-empty ragged column and either fail dtype inference (loud crash) or write an empty column that misparses on read. No test pins the 'absent column on no-labels' invariant.
- [LOW] `src/spyglass/spikesorting/v2/curation.py:466-469 (unit_label_rows and merge_group_rows empty guards)` — Two `if X: insert(X)` guards: skip `cls.UnitLabel.insert(unit_label_rows)` when empty (line 466-467) and skip `cls.MergeGroup.insert(merge_group_rows)` when empty (line 468-469). The merge_group_rows path always has at least the self-entries (lines 670-679 ensure every Unit has >=1 row), so the empty-guard for MergeGroup only fires when there are zero units AND zero merge groups -- which is the empty-curation case tested at line 3243. No test specifically calls labels={} on a sorting with units to assert no UnitLabel rows are inserted (only happy path through curation_id=0 fetches at line 1822 cover this implicitly).
    why risky: Lower severity than the others because the surrounding behavior is exercised. But a regression that drops the `if unit_label_rows:` guard would call `Part.insert([])` which on some DataJoint versions raises -- breaking ALL no-label curations (the default case). Worth a one-line assertion test.
- [LOW] `src/spyglass/spikesorting/v2/curation.py:651-658 (next_merged_id allocation for singleton-as-merge-group inside apply_merge=True)` — When apply_merge=True the fresh id is allocated ONLY when `len(int_group) > 1`; the singleton case is rejected upstream by the >=2 validation (line 610-618). The `else: key = min(int_group)` branch on line 657 is only reachable when apply_merge=False (singleton case is already rejected). The test at line 4885 covers apply_merge=True with multi-merge-only groups; no test exercises a mix of multi-and-non-multi groups under apply_merge.
    why risky: The dead-looking `else` branch at line 657 is reachable in practice only via apply_merge=False. If a future commit relaxes the >=2 validation, the apply_merge=True path would silently pick min(group) for a singleton instead of the previously-rejected behavior, and the curation would still write but with a duplicate-id collision against a non-merged unit. Pattern-match: the same 'silent contributor double-counting' shape as the across-group overlap bug.

### utils.py (10 branches)

- [CRITICAL] `src/spyglass/spikesorting/v2/utils.py:417-432` — _get_recording_timestamps multi-segment branch: when recording.get_num_segments() > 1, concatenate per-segment times into one (total_frames,) array.
    why risky: This branch is the entire reason the helper exists (the docstring lists 'Multi-segment NWB support' as responsibility #1). All tests in test_v1_parity.py use fake recordings that return get_num_segments()=1, and no other test exercises the multi-segment path. An epoch-stitched recording would silently report just segment 0's times, corrupting timestamp alignment for all downstream artifact/sort intervals. No test would catch a regression in the cumsum_frames slicing.
- [CRITICAL] `src/spyglass/spikesorting/v2/utils.py:155-199` — _assert_v2_db_safe: all three branches (env-var override at :184-185, safe-host pass at :187-189, RuntimeError raise at :191-198) are untested.
    why risky: This is the LAST LINE OF DEFENSE preventing v2 schema registration from clobbering a production database. A refactor that broke the host check (typo in the frozenset, wrong dj.config key, inverted condition) or broke the override env-var read would either deadlock test imports or, much worse, silently allow non-local writes. The docstring explicitly calls this 'the last line of defense if some other code path repointed dj.config'.
- [HIGH] `src/spyglass/spikesorting/v2/utils.py:258-267` — _assert_schema_version_matches mismatch ValueError branch: raises when row['params_schema_version'] (outer) disagrees with row['params']['schema_version'] (inner).
    why risky: test_integrity.py:240-287 exercises the surrounding _validate_params plumbing but only with consistent outer+inner versions, and tests/spikesorting/v2/ contains no row insert where outer != inner. The docstring explicitly warns 'downstream code that branches on the outer column will silently route v2 rows to v1 behavior (or vice versa)'. If this guard regressed (e.g. a typo flipping the !=), bogus rows would land and the silent-routing bug it was designed to prevent would resurface — exactly the kind of cascading divergence the schema-first design exists to prevent.
- [HIGH] `src/spyglass/spikesorting/v2/utils.py:29-33` — transaction_or_noop already-in-transaction branch (line 29 yields bare, no nested dj transaction).
    why risky: All five call sites (recording.py:398, :556, :1075, etc.) wrap source-part-master inserts; an incorrect inversion (e.g. unconditionally opening connection.transaction) would crash with 'transactions cannot be nested' inside a populate cascade. No test in tests/spikesorting/v2/ asserts the nested-call behavior — tests only confirm the context manager is *used* (AST scan in test_v1_parity.py:785-868). A regression here would corrupt the source-part atomicity guarantee at populate time only.
- [MEDIUM] `src/spyglass/spikesorting/v2/utils.py:325-363` — _ensure_lookup_row_exists ValueError branch (lines 355-363): raised when a required Lookup row is missing before insert_selection.
    why risky: Called from artifact.py:481, sorting.py:325, recording.py:717 to convert opaque DataJoint IntegrityError ('foreign key constraint fails') into actionable error messages with 'Run X.insert_default()' guidance. No test exercises the missing-row path — all v2 tests call insert_default() first. A regression (e.g. wrong helper_name in the error string, or skipping the check) would re-surface the opaque MySQL error that this helper exists to suppress, degrading the notebook UX precisely when users have a setup bug.
- [MEDIUM] `src/spyglass/spikesorting/v2/utils.py:473-474` — _consolidate_intervals ValueError branch: 'Input array must have shape (N_Intervals, 2)' raised when intervals.shape[1] != 2.
    why risky: No test in tests/spikesorting/v2/ asserts the shape rejection. Indirect coverage via test_disjoint_sort_intervals_concatenated only feeds well-shaped (N, 2) inputs. A malformed valid_times blob (e.g. (N, 3) from an upstream schema drift) would bypass the guard if the check were removed, yielding wrong frame indices and silently corrupted recordings.
- [MEDIUM] `src/spyglass/spikesorting/v2/utils.py:477-478` — _consolidate_intervals defensive sort branch: when intervals are not already monotonic, argsort by start.
    why risky: test_disjoint_sort_intervals_concatenated passes pre-sorted disjoint chunks (test_single_session_pipeline.py:5759 builds them in order). No test feeds unsorted intervals. If the sort were broken (e.g. argsort on the wrong axis), overlapping intervals would silently merge incorrectly, producing wrong frame slices for upstream IntervalList rows that aren't pre-sorted.
- [MEDIUM] `src/spyglass/spikesorting/v2/utils.py:36-47` — MetricsSource enum members 'analyzer_curation' and 'figpack': only the 'manual' value is tested.
    why risky: tests/spikesorting/v2/test_single_session_pipeline.py:1847 asserts metrics_source == 'manual'; no test inserts a CurationV2 row with metrics_source='analyzer_curation' or 'figpack'. curation.py:307 calls MetricsSource(metrics_source).value, which would reject a typo, but if either enum value were renamed/deleted in a refactor, the downstream metric_curation.py and figpack_curation.py stub ports would silently lose their provenance gate. Given those two modules are still stubs (per branch context), no integration test will catch the drift either.
- [MEDIUM] `src/spyglass/spikesorting/v2/utils.py:529-533` — parse_artifact_interval_list_name: both the str-prefix-match branch and the None-return branch are completely untested, and the function has NO callers in src/spyglass/spikesorting/v2/.
    why risky: The docstring claims it is 'the inverse' of artifact_interval_list_name and quotes a 'merge-dispatcher leave non-artifact names alone' contract — but no merge dispatcher actually invokes it. Either this is dead code that should be removed, or there is a missing call site that the v2 merge integration expects. Either way the public API is shipped untested.
- [MEDIUM] `src/spyglass/spikesorting/v2/utils.py:536-568` — get_spiking_sorting_v2_merge_ids: only its function signature is asserted (test_v1_parity.py:294-302). Neither as_dict=True nor as_dict=False return paths are executed in any test.
    why risky: This is the user-facing v2 parallel of v1's get_spiking_sorting_v1_merge_ids. The docstring promises 'Unknown keys raise ValueError' — but no test passes an unknown key. A regression in SpikeSortingOutput()._get_restricted_merge_ids_v2 (the delegate) would not be caught at this layer; notebook users would hit it directly.

### pipeline.py (3 branches)

- [MEDIUM] `src/spyglass/spikesorting/v2/pipeline.py:65-70` — franklab_tetrode_mountainsort4 preset registration: the bundle is never end-to-end populated by any test.
    why risky: Only the ms5 preset (test_single_session_pipeline.py:2330) and clusterless_thresholder preset (:3450) are exercised by run_v2_pipeline. The ms4 preset string is asserted to exist in list_presets() output indirectly, but no test inserts the ms4 bundle's Lookup rows (preproc_params_name + 'franklab_tetrode_hippocampus_30kHz_ms4') through run_v2_pipeline. If the preset bundle references a Lookup row name that was renamed or never installed by insert_default, the failure surfaces only on first user invocation.
- [LOW] `src/spyglass/spikesorting/v2/pipeline.py:42-83` — list_presets() helper: never called by any test.
    why risky: Notebook-discoverable accessor that returns sorted(_PRESETS). If a future refactor accidentally returns the dict instead of sorted names, or hides a preset, no test catches it. Low severity because the docstring example is its only contract — but the docstring lists the exact three names as expected output, and a regression that drops one would silently break notebooks.
- [LOW] `src/spyglass/spikesorting/v2/pipeline.py:264` — run_v2_pipeline empty-description fallback: 'description or f"run_v2_pipeline preset={preset}"' substitutes a generated description when caller passes description="".
    why risky: The end-to-end test (test_single_session_pipeline.py:2331) always passes description='pipeline e2e test', and the zero-units test (:3445) omits description (uses default ''). The fallback path may or may not execute depending on how CurationV2.insert_curation handles the synthesized string — no assertion ever verifies that the generated description lands in the row. A subtle truthiness bug (e.g. empty dict vs empty string semantics) would be invisible.

### session_group.py (entire module) (1 branches)

- [HIGH] `src/spyglass/spikesorting/v2/session_group.py (entire module)` — ALL non-stub branches in session_group.py are untested: SessionGroup.create_group (raises NotImplementedError), SessionGroup.is_multi_day (raises NotImplementedError), MotionCorrectionParameters.insert1 (Pydantic-validates row['params']), MotionCorrectionParameters.insert_default, ConcatenatedRecordingSelection.insert_selection (NotImplementedError), ConcatenatedRecording.make (NotImplementedError). No test in tests/spikesorting/v2/ imports from spyglass.spikesorting.v2.session_group at all.
    why risky: MotionCorrectionParameters.insert1 (line 139-144) WILL be called in production once the concat consumer lands -- its Pydantic validation is the only thing preventing bad blobs from being persisted, and _DEFAULT_CONTENTS at module-import time invokes MotionCorrectionParamsSchema(preset=...).model_dump() three times. A regression in the schema or in insert1 would surface only when a user inserts a custom preset, with no test catching it. The NotImplementedError sentinels are also not tested -- a typo silently swallowing the exception would let users believe a no-op succeeded.

## Refuted findings (for awareness)

- **utils**: v2 parse_artifact_interval_list_name helper exists but is never called
  - why refuted: The reviewer's claim that `parse_artifact_interval_list_name` has "zero call sites in src or tests" is REFUTED. The function IS actively used by the merge-table dispatcher at `src/spyglass/spikesorting/spikesorting_merge.py:199-214`:

```
from spyglass.spikesorting.v2.utils import (
    parse_artifact_interval_list_name,
)
...
if restrict_by_artifact and "interval_list_name" in key:
    artifact_i
- **session_group**: MotionCorrectionParameters.job_kwargs field is wired up but completely undocumented and unused
  - why refuted: The reviewer's factual claims about the code are accurate: `job_kwargs=null: blob` exists on line 115 of `src/spyglass/spikesorting/v2/session_group.py`, defaults pass `None` at lines 123/129/135, and `MotionCorrectionParamsSchema` (in `_params/motion_correction.py`) does not include a `job_kwargs` field. The consumer `ConcatenatedRecording.make()` is gated with `NotImplementedError` (session_grou
- **pipeline**: v2 interval_list_name argument is passed through unchanged; v1 populator could auto-create a SortInterval from the IntervalList
  - why refuted: The reviewer's citation is mis-attributed. They cite v1 behavior at `src/spyglass/spikesorting/v0/spikesorting_populator.py:171-198` — but that is the v0 module, not v1. The actual v1 code at `src/spyglass/spikesorting/v1/` has no populator and no SortInterval table.

v0 (cited as "v1"): `v0/spikesorting_populator.py:184` does pull `valid_times[0]` and `:189-196` inserts a SortInterval row when `s
- **pipeline**: v2 has no equivalent of v1's SpikeSortingPipelineParameters lookup table
  - why refuted: The claim conflates v0 with v1, mis-states the rationale, and ignores existing test coverage.

1. Wrong baseline (v0 vs v1). `SpikeSortingPipelineParameters` lives in `src/spyglass/spikesorting/v0/spikesorting_populator.py:40` (confirmed by grep — only v0 has it; `__init__.py:35,82` re-exports it). It does NOT exist in v1 at all. `ls src/spyglass/spikesorting/v1/` shows no populator file, and `gre
- **?**: get_sorting zero-unit path uses NumpySorting.from_unit_dict({}, ...) instead of [{}] -- API-shape divergence vs tests
  - why refuted: Cited lines match: v1 sorting.py:539-541 uses [units_dict] (list); v2 sorting.py:763-765 uses bare {} positionally. But this is NOT a behavioral divergence. SpikeInterface 0.104.0 (the pinned version, verified against the GitHub source) explicitly normalizes the input inside NumpySorting.from_unit_dict: 'if isinstance(units_dict_list, dict): units_dict_list = [units_dict_list]'. So from_unit_dict(
