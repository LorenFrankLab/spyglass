# PR #1609 review — spikesorting-v2 vs master (head cc9a7953, 2026-09-18, revised after owner meta-review)

Scope: 1,442 commits, 415 files, +163K lines. 98 new files under src/spyglass/spikesorting/v2 (58K lines), 41 modified shared modules (2.5K lines), pyproject, 7 environment files, 2 workflows. Nine review agents; owner meta-review on 2026-09-18 re-tested the numbered findings. Items marked [v] were reproduced in this session; [owner] were reproduced by the owner; [agent] rest on an agent's report only.

Recommendation: REQUEST CHANGES. Priority order: scientific correctness → compatibility/dependency failures → validation gaps → CI gate → docs.

## Tier 1 — Scientific correctness (silent wrong results)

1. **Noise levels and whitening are estimated on the artifact-ZEROED recording.** sorting.py:1772 rebinds to the masked recording (silence_periods mode="zeros", _sorting_artifact_mask.py:323); _sorting_dispatch whitens it; _sorting_analyzer.py:942-954 estimates noise_levels from it. Bias magnitude is layout-dependent: agent measured SNR +11% at 5% masked / +44% at 50%; owner measured +35% to +85% at 30% masked depending on mask layout. MS5 detect_threshold is in whitened-noise units, so it is effectively lowered. nn_noise_overlap's noise cluster also samples masked data. Docstring at _sorting_dispatch.py:618-621 asserts the opposite. [v code path; owner + agent measurements]
   **Fix:** estimate W/M and noise_levels from valid samples only (valid-times-restricted view; SI 0.104 accepts explicit W, M, noise_levels). Do NOT substitute mode="noise": SI generates replacement noise from parent-recording noise levels and does not preserve cross-channel covariance. Log the realized masked fraction.

2. **Bandpass filter runs AFTER disjoint intervals are concatenated, and shipped recipes floor segments at 1.5 ms.** recording.py:2109-2131 (restrict_recording → apply_pre_motion_preprocessing); _recipe_catalog.py:180,192 `min_segment_length: 0.0015`. v1 also filtered after concatenation (floor 1.0 s); the 45-sample floor turns a bounded edge effect into whole-segment filter transient (agent: 3.8x RMS, 96 µV peak error at the floor). [v order + floor; owner + agent reproduced distortion]
   **Fix:** filter continuous data before restriction, or keep filter-length context around each interval and crop after filtering. Raising the floor alone does not remove the discontinuity between longer intervals.

3. **UnitMatch cross-validation halves are cut by recording time.** _unitmatch_backend.py:208-231; SI templates zero-fill and `continue` for units with no sampled spikes in a half (analyzer_extension_core.py:505,517-518). Units absent from one half get an all-zero template half. [v zero template; owner reproduced]
   **End-to-end experiment (10 seeds, synthetic drift-free 16-ch/20-unit two-session data, UnitMatchPy 3.2.7, real `extract_unitmatch_bundle` vs a per-unit temporal-split bundle; scratchpad/unitmatch_exp/):**
   - Affected units are lost entirely: 0/50 true matches under the current bundle vs 42–45/50 with the per-unit split (baseline 87–89% on this generator). Any directed pair touching a zero half gets P(match) < 1e-6 because five of six metric scores collapse to 0 (NaN coerced).
   - Calibration does shift session-wide: those pairs (13–23% of directed pairs) are labelled non-match and feed the non-match kernels (e.g. waveform-score non-match mean 0.47→0.32, trajectory 0.57→0.65), n_expected_matches 86→70, match prior 0.054→0.043. Healthy true-pair matching was not measurably degraded (133→134/150), but healthy false positives rose 10→28 of 2100 when both sessions carry zero halves (|Δp| up to 0.97) and fell 10→5 with one session.
   - The earlier "zero-half units score as mutually similar" claim is NOT supported: 0/200 S×S false matches in every run; they match nothing.
   Net: silent loss of every drift-in/drift-out unit plus a calibration shift that changes false-positive counts among healthy units. Caveat: synthetic, drift-free, one FIXED construction.
   **Fix:** split each unit's own sampled spikes into first-half/second-half by temporal order, as UnitMatchPy's extract_raw_data does. Do not use random halves (changes what the within-unit distance calibrates).

4. **`observed_intervals` derives interval END from nominal fs and can exclude real data.** _observed_time.py:74-75 `kept.append((t, t + (end - first) / fs))`. Probe: 1000 samples at 990 Hz spacing under a declared 1000 Hz rate → interval [0, 1), last 10 recorded samples excluded; with drift the intervals can also nest, and contains_times then drops spikes from observed_firing_rate_hz, observed_presence_ratio, and the decoder input (decoding/v1/sorted_spikes.py:120-122). [v]
   **Fix:** take the end from the data (`_segment_times_at(recording, [end-1]) + 1/fs`) and assert sorted/disjoint.

5. **NaN from `get_spike_indicator` corrupts the MUA detector.** analysis/v1/group.py:623 emits NaN for unobserved bins (v2-containing groups only); mua/v1/mua.py:89-111 sums it and passes it to multiunit_HSE_detector unguarded. Owner probe: a synthetic event next to a gap disappears (1 → 0). This session's probe (ripple_detection in the v2 env): a 20-bin NaN gap adjacent to a 30–100 ms event yields 31–35 detected events instead of 1–2; a NaN gap inside a 200 ms event splits it (1 → 2); NaN far from the event has no effect. Loss or fabrication depends on layout; both are silent. [v]
   **Fix:** drop or mask unobserved bins before smoothing/z-scoring (restrict to observed intervals, as the decoder path now does), never pass NaN into the detector.

## Tier 2 — Compatibility and dependency failures

6. **Three env files are unsolvable**: environment_dlc.yml, environment_moseq_cpu.yml, environment_moseq_gpu.yml keep `pytorch<1.12.0` (line 32) while numpy moved to `>=2,<3` (line 26). master had numpy<2 alongside the torch pin. Reproduced on osx-arm64: `conda create --dry-run --override-channels -c conda-forge "python=3.11" "numpy>=2,<3" "pytorch<1.12.0"` → "pytorch <1.12.0 is not installable … pytorch 1.11.0 would require numpy >=1.21.6,<2.0a0, which conflicts". Without --override-channels the defaults channel needs an anaconda token, which is why plain `conda create` fails locally with an auth error instead. environment.yml already has the fix (pip torch>=2). [v]

7. **`[dlc]` extra is broken; `[moseq-cpu]` silently backtracks.** Under numpy>=2 the resolver backtracks (macOS arm64 / py3.11 / uv 0.12.15 / 2026-09-18: deeplabcut 2.2.3, tensorflow 2.20.0, numpy 2.4.6; owner on Linux / py3.10: deeplabcut 2.3.10, keypoint-moseq 0.0.5; agent: keypoint-moseq 0.1.5). Verified in a throwaway venv: `uv pip install ".[dlc]"` succeeds, then `import deeplabcut` (2.2.3) raises `AttributeError: np.sctypes was removed in the NumPy 2.0 release`. So on this platform the extra installs cleanly and cannot be imported. Whether 2.3.10 imports under numpy 2 is untested. Base was 3.0.1 / 0.2.5. Not in CHANGELOG. [v import failure]

8. **v0 `Waveforms.load_waveforms` is unconditionally gated** by the legacy-runtime guard (v0/spikesorting_curation.py:498) though its body is a pure load. Qualification: SI 0.104's `load_waveforms` handles legacy binary-folder extractors (agent verified a 0.99-written folder loads as MockWaveformExtractor) but raises NotImplementedError for legacy Zarr (waveforms_extractor_backwards_compatibility.py:419). v1 `MetricCuration.get_waveforms` (v1/metric_curation.py:370) extracts by default (overwrite=True), so only its cached-read branch (`si.load_waveforms`, line 414) is a candidate for ungating. [v]
   **Fix:** gate extraction, not cached binary-folder reads; keep the guard for Zarr.

9. **`migrate_positional_unit_ids` can reinterpret annotations written under the new true-id contract.** unit_annotation.py:193-197 remaps stored_id → true_ids[stored_id]. A durable per-merge marker table exists (UnitAnnotationPositionalIdMigration, :331) and the docstring requires migrating before writing new annotations (:155), but add_annotation (:70-92) does not write the marker, so migration run after new true-id writes reinterprets them. Narrower trigger than "silent re-point": requires ignoring the documented order. [v]
   **Fix:** an explicit migration boundary (e.g., add_annotation inserts the marker for its merge_id on first write, or migration refuses merge_ids with rows newer than the marker). Membership of stored ids in the true-id set cannot establish provenance (legit positional ids can also be valid true ids).

10. **`insert_default_legacy_si_sorters()` hard-fails for any installed binary-data sorter** (sorting.py:584-623): SI folds job kwargs into get_default_sorter_params; wrapper vocabulary excludes them; validation raises outside the try/except. Reproduced for tridesclous/yass/pykilosort/kilosort2 on SI 0.104. [owner + agent]

11a. **Real Frank-lab tetrode geometry breaks v2 after reload.** The real file tests/_data/raw/minirec20230622.nwb (probe_type tetrode_12.5, 2023) stores contacts in the x-z plane: rel_x = ±6.25, rel_z = ±6.25, rel_y = 0 for all 128 electrodes (h5py). `maybe_apply_tetrode_geometry` (recording.py:2132, _recording_geometry.py:166-252) attaches a 2D square in memory at Recording.make, but the artifact writer copies the raw rel_x/rel_y/rel_z through the electrodes table; a recording artifact derived from this file (tests/_data/analysis/minirec20230622/minirec20230622_0Q58PXTDC1.nwb, 3-ch ProcessedElectricalSeries) persists rel_y = 0 / rel_z = ±6.25. v2's `read_recording_nwb` then yields locations [[-6.25,0],[-6.25,0],[6.25,0]] and `get_probe()` raises `ValueError: Contact positions must be unique within a probe`. Consequence: the sort stage runs on a degenerate two-position geometry (SI's basesorter uses get_channel_locations, which does not check uniqueness), and the pipeline then crashes at `build_analyzer` (_sorting_analyzer.py:844 `recording.get_probe()`). Loud in the end, but only after a wasted degenerate sort. The v2 test fixtures (chronic_minirec_*) carry x-y geometry (rel_y = ±6.25, rel_z = 0), which is why the suite passes. The lab DB could not be queried (no credentials on this machine), so how many production files share the x-z layout is unknown; the canonical 2023 test file does. [v]
    **Fix:** persist the patched geometry (or re-apply the patch in get_recording), and pick the projection plane from the non-degenerate axes rather than assuming x-y; add a preflight assertion that a multi-channel group has ≥2 distinct 2D locations.

11. **scipy is no longer declared in pyproject** (master: scipy<1.13; head: none) while 11 non-v2 modules import it at top level. Downgraded to packaging cleanup: scipy is required by jax (>=1.13), position-tools, ripple-detection, and non-local-detector, so a missing-scipy install is not demonstrated. Declare `scipy>=1.13` to make the contract explicit. [v]

## Tier 3 — Validation gaps (tests / CI)

12. v0/v1 read paths under SI 0.104 are proven only by monkeypatched stubs (test_legacy_modern_si_coexistence.py:115-148); no 0.104-tier test loads a real 0.99-serialized extractor.
13. v0/v1 `UnitWaveformFeatures.make` has zero real CI coverage: tests/decoding/conftest.py now direct-inserts with a `_StubWaveforms`, and pytest-legacy does not run tests/decoding. The docstring at test_clusterless_waveform_features.py:708-730 claiming otherwise is false.
14. "Release acceptance probes" (cc9a7953) live in scripts/audit_*.py, are not collected by pytest, need a Docker client and a manual-dispatch fixture; one has no assertion.
15. Every sorting-quality known-answer gate is nightly/manual or never runs: UnitMatch AUC fixtures unhosted (`|| true` fetch, absent from SPYGLASS_V2_REQUIRE_FIXTURES); v1-parity baseline env var never set in any workflow.
16. Test-integrity: pinned_whiten seed untested (deleting `seed=` breaks nothing); artifact oracle is gain-only with zero offsets everywhere; concat member order unasserted; `pytest.skip` on SpikeSortingError (test_pipeline_run.py:829-836); `pytest.raises(Exception)` x5; `_too_recent_or_unknown` only ever mocked to False; four `UnsupportedDirectInsertError` guards and metric_curation.py:1696/1712 namespace invariants untested.

## Tier 4 — CI gate
17. black fails on 9 files (`uvx black@26.5.1 --line-length 80 --check .`): src v2/_recording_restriction.py, analysis/v1/group.py; tests test_review_browser.py, test_recording_services.py, scripts/audit_{branch_scale,handoff,lifecycle}.py; notebooks/py_scripts/10_Spike_SortingV2_{Presets,Curation}.py. [v]

## Important (should fix before release)

### Compute / science
- Shipped auto-curation rules all use missing_policy="pass": an all-NaN metric yields "no units flagged" with only a warning; "fail" labels every unit with no warning (_metric_curation.py:189-215; metric_curation.py:594-654). All-NaN can be legitimate (every unit below min_spikes), so the fix must distinguish computation failure from expected missingness. Turning SI's "Error computing metric" warning into an error is insufficient: SI's nn code catches internally and returns NaN without warning (pca_metrics.py:185-194). Suggested: after compute, assert each requested column is finite for every unit that meets its spike-count precondition; raise otherwise.
- No finiteness guard on traces anywhere in the signal path; a NaN sample propagates through filtfilt within its chunk and across channels via median referencing (agent measured on a single-chunk recording; the contamination is chunk-scoped, not necessarily recording-wide). Cheap to check in _compute_artifact_chunk.
- `proportion_above_threshold=1.0` (shipped `default` recipe) + one unflagged dead channel makes artifact detection inert; warning fires only for proportion<1.0 (_artifact_intervals.py:317-326).
- `effective_random_seed` never reaches any sorter's own seed param; TDC2 rows record seed=0 while the sort is non-reproducible (sorting.py:424-429; _recipe_catalog.py:427-436).
- `intersect_intervals` lacks a guard for its documented sorted/disjoint precondition (_signal_math.py:201-220); raw IntervalList.valid_times reaches it via decoding/v1/sorted_spikes.py:133,173 unchecked. A missing guard, not an algorithm defect.
- `tetrode_12.5` geometry patch applied in memory (recording.py:2132) but not persisted; get_recording (1808-1818) never re-applies. PROMOTED to item 11a (verified on the real 2023 file: x-z geometry → duplicate xy positions → get_probe ValueError at analyzer build).
- Zero-spike units survive remove_excess_spikes (_sorting_dispatch.py:835-852) with an arbitrary extremum Electrode FK and peak_amplitude 0.0.
- `threshold_unit=="uv"` still scales an explicit noise_levels override (_sorting_dispatch.py:562-580).
- nn-metric seed defaults to None for user MetricParameters rows (_si_metric_patches.py:50).
- `set_channel_offsets(0.0)` bypasses the heterogeneous-offset guard (_recording_preprocessing.py:218,242).
- `hash_extension_data` materializes whole waveform memmaps twice (_recompute.py:47-57).
- `.trash-*` rollback copy is reclaimed by cleanup_analyzer_staging (_analyzer_cache.py:648-650 vs 728-748); dry_run still unlinks lock files.

### Shared modules
- `Merge.fetch_nwb` silently drops sources whose parent lacks the restricted attribute: `{"nwb_file_name": X}` drops all v1 curations (CurationV1 keyed by sorting_id only); warning only when no source matches (dj_merge_tables.py:823-854). Test fixture gives both parents the attribute.
- `fetch_spike_data(return_unit_ids=True)` changes unit_id VALUES for v1 apply_merge=True curations (group.py:418-421); CHANGELOG says dense namespaces unaffected but v1 merged output isn't dense.
- `SortedSpikesDecodingV1` ORs user `is_missing` into the computed mask and can newly raise (sorted_spikes.py:311-325); not in CHANGELOG.
- `get_raw_eseries_path` raises on >1 matching series while ingestion accepts (`_only_ingest_first`); remedy kwarg unreachable from v0/v1 callers (nwb_helper_fn.py:401-408).
- `dj_graph._bridge_restr` swallows "already exists" edges silently when not verbose (484-503).
- recompute.py:133 env gate loosened to 4 namespaces; comment lists ndx_optogenetics/ndx_pose too.
- environment.yml dropped jax<0.7.2; conda section resolves jax 0.10.2 vs pyproject jax<0.10 (pip step corrects it; `mamba env update` without pip does not). non_local_detector unpinned in ymls vs ==0.6.9.

### Curation / params / orchestration / UX
- Review browser: a failed `api/operation` poll leaves session.operation "running" with no retry; every button that could reset it is disabled while busy; `inspect` does not pre-save, so a reload loses edits (_review_controls.js:99-124,140-172). The 409 path deliberately offers "Open latest draft in a new tab" (line 164) with edits retained in the tab, so it is a designed recovery, not a lock; but editing stays enabled in a tab whose saves will all be rejected.
- `clone_pipeline_preset` with a preprocessing override yields a new preprocessing name absent from `_PREPROC_WAVEFORM_PARAMS` → silent cortex 1.0/2.0 ms waveform window (_recipe_catalog.py:521-524; _pipeline_presets.py:1091-1094).
- `QualityMetricParameters`: insert_default keeps a stale same-name default silently (metric_curation.py:386-390); `insert(replace=True)` bypasses immutability (360-377; unit_annotation.py:123-130 already rejects it); `observed_presence_bin_duration_s` float32 column in an exact fingerprint (_lookup_validation.py:348-350).
- `save_curation_from_uri` defaults label_policy="inherit" over a full-state draft while review_api.py:1278 hard-codes "replace" (figpack_curation.py:1057,1099).
- `select_units_for_analysis` validates group_name ≤80 before appending `_m{i}` into varchar(80) (analysis_selection.py:469-470,516-520); also inserts on a read path (284).
- Cache loader `except Exception` → rmtree + rebuild on transient I/O/MemoryError (_sorting_analyzer.py:199-227); curation cache validation except → full republish (_curation_analyzer.py:477-481); incomplete `waveforms` extension returned silently (_analyzer_cache.py:303-307).
- SI-version drift guard on nn_noise_overlap shim only warns (_si_metric_patches.py:192-200).
- Preflight never runs resolve_effective_seed(reject_ambient_seed=True); run_summary reports the ambient seed the sort will reject.
- Container sorts record host SI/sorter versions (sorting.py:1817-1818).
- Recompute: storage fingerprint hashes every file → unrelated extension writes cascade-delete matched=1 verdicts (recompute.py:940-953); invalidate→repopulate not atomic (1211-1233); inventory hashes read outside lock (850-870); recording verify re-materializes reclaimed files (586-598).
- `UnitMatch.Pair.insert` setdefault evaluates `_existing_pair_edges` eagerly per row (unit_matching.py:763-765). `manual` strategy rejects string sorting_id (_unit_match_planning.py:218-235). Concat member-artifact stage does per-member I/O on reused runs (_pipeline_run.py:835-869).

### Types
- `ObservationAvailability` assumes sorted/disjoint (n,2) intervals with no check. Exception hierarchy has no common base; DuplicateSelectionError promises a message substring. `MatchPair.match_probability` unchecked until Pair.insert. Three string vocabularies for source-kind; four str/UUID shapes for cross-session curation keys.

## Docs (separate priority)
- README quick-example reads `run_summary["analysis_merge_id"]` (README.md:167,172,181); key is `auto_labeled_merge_id` → KeyError. [v]
- CHANGELOG [Unreleased] contradicts itself and code (MS4 in extra 1182-1189; SI pin 1551; .zarr caches 468,498; MAD multiplier 1717; default preset 1250; NwbfileHasher 1010) and documents deleted ArtifactDetection/ArtifactDetectionSelection tables (623, 643-646, 1155-1166, 1591-1596). Omits networkx hard dep, scipy removal, torch move, ipywidgets extra, decoder masking change.
- pyproject.toml:120-121 comment references deleted AnalyzerCuration and renamed lock. Residual `ArtifactDetection` references in v2 docstrings/errors (artifact.py:90,123,221; _selection_identity.py:291; _shared_artifact_group.py:71-120; utils.py:652; sorting.py:1452,2493,2514; metric_curation.py:1077; _sorting_artifact_mask.py:112).
- Scaffolding: `OP-3`/`OP-4` in _selection_identity.py:331,334 (leakage test regex misses two-letter codes); `phase4_*` in test_unit_annotation_integration.py / test_review_api_integration.py; TODO.md at repo root.
- waveform_features.py `_fetch_waveform_v2` docstring cites max_spikes_per_unit=500; recipes use 20000. 36 public symbols without docstrings; 40 without Parameters/Returns.

## Suggestions (selected)
- Log concat preflight warnings like the single-session branch (_pipeline_run.py:594-610). Custom presets without sampling_rate_hz skip the rate check silently. Corrupt analyzer folder churns recompute audit history. MountainSort4Schema docstring claims filter defaults matter but filter=False. AutoCurationRuleSchema.threshold accepts NaN. `weights/np.sum(weights)` in _si_metric_patches.py:150-153 is scale-invariant downstream (an "explodes" claim was checked and is WRONG); only 0/0 on all-zero clips is real. Literal types for status/role/kind strings. `sampling_frequency`/`total_duration_s` single-precision columns.

## Verified OK (with limits)
- Artifact detect→mask round-trip exact at the nominal rate; uses recording fs; µV thresholding genuine; no int16 overflow; chunked timestamp helpers == searchsorted (300 trials); merge dedup == SI (500 trials); masking preserves sample indices. These checks assume timestamps consistent with the declared rate; the clock-drift probe (item 4) shows the observed-time layer is NOT covered by that assurance.
- Prior 2026-09-01 merge-blockers: UnitAnnotation re-point FIXED (residual: item 9); concat synthetic timeline FIXED; Export.File delete FIXED; ES selection FIXED as designed; SI pin vs v0 reads PARTIAL (item 8).
- Batched lineage (36aec5f4) equivalent to per-row; merged unit ids agree between applied and lazy paths; metrics scored on curation-scoped analyzer; nn_noise_overlap always-NaN fix present; review state round-trip and identity pinning; lineage delete guards; concat split-back; hidden-sibling atomic publish; selection identity hashing stable; ImmutableParamsLookup contract; lab recipes match docs; frozen universe enforced at 3 layers; no v0/v1 schema definition changes; fixture determinism closed by hash-gated download.

## Strengths
Identity layer (content-addressed selections, generation-pinned CurationRef, EffectiveSortConfig as single resolver), atomic cache publish with trash rollback, DB-free pure-logic modules, and comments that explain why at the point of dependence.

## Reproduction notes
- conda: osx-arm64, conda-forge only, 2026-09-18 (output above).
- observed_intervals probe: scratchpad/probe_obs.py (v2 env, SI 0.104.3).
- MUA probe: scratchpad/probe_mua{,2,3}.py (ripple_detection in v2 env).
- black: 26.5.1 via uvx, line length 80, repo root.
- DLC: `uv venv --python 3.11` + `uv pip install ".[dlc]"` in scratchpad/dlc_venv, then `import deeplabcut` (macOS arm64, 2026-09-18).
- UnitMatch: scratchpad/unitmatch_exp/ (run_experiment.py, zeroed_experiment.py, pooled_tables.md, analysis_tables.md; ~35 s for 10 seeds).
- Geometry: h5py on tests/_data/raw/minirec20230622.nwb and tests/_data/analysis/minirec20230622/minirec20230622_0Q58PXTDC1.nwb; v2 `_recording_nwb.read_recording_nwb` + `get_probe()`.
- Not run: full pytest (DB), sorting-quality suites, lab DB Probe.Electrode survey (no credentials here; lmf-db.cin.ucsf.edu:3306 is reachable), DLC 2.3.10 import.
