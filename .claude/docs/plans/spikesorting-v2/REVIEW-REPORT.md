I'll synthesize the final report. The inputs are already verified, so I'll dedup, rank, and structure them without re-running checks. Let me note a few cross-references first (several findings point at the same root cause).

Key dedup observations:
- The v2 artifact-restriction silent-drop appears **4 times** (spikesorting_merge.py:233-248 / 236 / 237-238, plus the cross-cutting copy) — one root cause.
- The `fetch_nwb(return_merge_ids=True)` multi-source bug appears **3 times** (dj_merge_tables.py:563) — one root cause (verdicts split high/medium on whether it silently misaligns vs. raises; I'll reconcile).
- `deprecate_log(alternate=...)` appears **twice** (dj_merge_tables.py:958) — one root cause.
- `SortedSpikesGroup` test_mode label-filter appears **4 times** (group.py:219) — one root cause.
- `UnitWaveformFeatures` unreachable v2 branch appears **twice** (waveform_features.py:137).
- The MS5 finding subsumes a generalizable "schema omits SI preprocessing toggles" pattern.

Here is the report.

---

# Spikesorting-v2 Review — Synthesis Report

Branch under review: `spikesorting-v2` vs `master`. Core: `src/spyglass/spikesorting/v2/`. All findings below were adversarially verified (the `verdict` block) unless tagged `(unverified)` — and there were **zero** unverified findings and **zero** unverified-risky divergences in this batch.

---

## VERDICT

The v2 single-session spike-sorting core is **correct, well-reasoned, and unusually well-tested for its happy paths and documented v1 divergences** — the scientifically load-bearing fixes (gain-aware artifact thresholds, off-by-one interval consolidation, searchsorted gap-safe readback, deterministic seeds, fail-loud guards) are real, sound, and mostly tested. **It is reliable enough to use for fresh single-session curations.**

However, **it is NOT yet export-safe and has two genuine silent-wrong-result bugs at the consumer boundary** that must be fixed before relying on it for downstream decoding/export:

1. **Merge-ID resolution silently ignores `artifact_id`** (`spikesorting_merge.py`) — a high-severity silent-error that can feed the wrong sorting into decoding, and in the primary documented path can return *all* v2 merge_ids.
2. **`fetch_nwb(return_merge_ids=True)` breaks for multi-source restrictions** (`dj_merge_tables.py:563`) — exactly the v1+v2 migration scenario; either misaligns merge_ids or raises mid-fetch.
3. **Export completeness is entirely unverified** — no test asserts a v2 merge_id's NWB files land in `ExportSelection.File`; helper accessors bypass `fetch_nwb` (the only export-logging hook).
4. One **MS5 reproducibility/double-filter** high-severity issue (schema omits `filter`/`whiten`).
5. The **DANDI/content-hash digest excludes dataset values** (pre-existing upstream, but live in v2's `cache_hash` path; warn-only mitigates).

Efficiency: no efficiency regressions found; chunked artifact detection and sparse analyzers are appropriately bounded.

### TOP 5 highest-value actions
1. **Join `SortingSelection.ArtifactSource` into `sort_master`** so `artifact_id` is an actual heading column (and cast parsed artifact id to UUID). `spikesorting_merge.py:233-248`. Add an exclusivity test (two sorts differing only by artifact pass → exactly one merge_id). **[A]**
2. **Fix `fetch_nwb(return_merge_ids=True)`** to iterate only the current source's freshly-fetched files and restrict by PK-only. `dj_merge_tables.py:549-577`. Add a ≥2-source test. **[A]**
3. **Add `filter:bool=False, whiten:bool=True` to `MountainSort5Schema`** (mirror MS4). `_params/sorter.py:90-110`. **[A]**
4. **Add an export-completeness DB test**: start an export, `fetch_nwb({'merge_id': v2_id})`, assert the CurationV2 + Recording cache files are captured in `ExportSelection.File`. Decide/document whether helper accessors (`CurationV2.get_sorting`, `Recording.get_recording`) must route through `fetch_nwb` for export. **[C/D]**
5. **Seed the analyzer `random_spikes` extension** so persisted `peak_amplitude_uv`/peak_channel are reproducible across rebuilds. `sorting.py:2087-2097`. Fix the one-token notebook typo `plot_by_sorting_ids → plot_by_sort_group_ids`. **[A/D]**

### Strengths (what is genuinely well-done)
- **Fail-loud safety guards** replace v1's silent footguns throughout: heterogeneous-gain rejection, `RecordingTruncatedError`, `EmptyArtifactValidTimesError`, `ZeroUnitSortError`, peak-channel-mismatch `RuntimeError`, `NonIntegerUnitIDError`, overlapping-merge-group rejection, and inspect-before-destroy `set_group_by_shank`.
- **The headline scientific fixes are real and sound**: gain-aware artifact thresholds in µV (`artifact.py:119-148`), exclusive-end interval consolidation matching SI `frame_slice` (`utils.py:550-563`), searchsorted (not affine) gap-safe spike readback, abs-time merge dedup that is gap-correct, and explicit `seed=0` pins (SI PR #3359) for whitening and noise levels.
- **Reproducibility discipline**: tri-part `make` moves the long sort outside the transaction with cleanup-on-failure; global job_kwargs install/restore in a `finally`; scoped `np.Inf` monkeypatch.
- **Strong, exact-value test pinning** for SI sorter defaults (KS4/MS5/SC2/TDC2), preprocessing default blob + stage split, the gain gate, brain-region attribution (raise + anchor-member df on both Sorting and CurationV2 sides), zero-unit graceful handling, and disjoint-recording readback.
- **Pydantic validation on every param `insert1`** with `extra='forbid'` for curated schemas and a documented escape hatch for generic sorters.
- **Type-stable, idempotent `insert_selection`/`insert_curation`** PK-only return contract; source-part pattern makes "no artifact pass" a queryable absent row.

---

## (A) Correctness & silent-error bugs — ranked by severity × confidence

| # | Title | File:line | Sev | Conf | Evidence → Fix |
|---|---|---|---|---|---|
| A1 | **`_get_restricted_merge_ids_v2` silently drops `artifact_id` restriction** (DEDUP: reported 4×) | `spikesorting/spikesorting_merge.py:183-188, 233-248` | **High** | High | `artifact_id` lives only on `SortingSelection.ArtifactSource`, not in `sort_master = SortingSelection * RecordingSource`. DataJoint dict-restriction silently drops keys absent from the heading (`condition.py:228-242`), so `& {'artifact_id':X}` is a no-op. The documented primary path (`restrict_by_artifact=True`, `artifact_{uuid}` interval) maps to this dead restriction; in that path `rec`/`curation` restrictions are also empty → returns **all** v2 merge_ids. Parity regression vs v1's `_get_restricted_merge_ids_v1`. Existing test asserts membership only. **Fix:** join `SortingSelection.ArtifactSource` into `sort_master` (or resolve `artifact_id→sorting_id` via the part); cast parsed artifact id `str→uuid`; add an exclusivity test. |
| A2 | **`Merge.fetch_nwb(return_merge_ids=True)` over-accumulates/misaligns merge_ids across multiple sources** (DEDUP: reported 3×) | `utils/dj_merge_tables.py:549-577` (regressed line 563) | **High** | High/Med | Loop builds `merge_ids.extend([... for file in nwb_list])` over the **cumulative** `nwb_list`, not the current source's files; master did `extend([k[pk] for k in source_restr])`. With ≥2 sources, `len(merge_ids) > len(nwb_list)` and ordering scrambles; consumers `zip(nwb_file_list, merge_ids)` (`analysis/v1/group.py:190-193`, `unit_annotation.py:140`). Two verdicts split on outcome: in-loop AND-of-other-source's `source_restr` with a PK-carrying `fetch_nwb` dict can make `fetch1` raise on a 0-tuple (loud) rather than silently misalign — but for ≥2 sources each contributing ≥1 file the over-accumulation is library-independent. Exactly the v1+v2 migration scenario. Test covers single source only. **Fix:** iterate only the current source's freshly-fetched files; restrict by PK-only (strip NWB-object values), mirroring `_execute_nwb_query`'s `rec_only_pk`. |
| A3 | **MS5 schema omits `filter`/`whiten` → double bandpass + non-deterministic internal whitening (bypasses the seed pin)** | `spikesorting/v2/_params/sorter.py:90-110` | **High** | High | `MountainSort5Schema` (`extra='forbid'`) defines no `filter`/`whiten`; shipped `{}` → SI MS5 wrapper applies its `_default_params` `filter=True, whiten=True` unconditionally (`mountainsort5.py:133-145`). (1) Recording is already bandpassed (`recording.py:1929-1935`) → MS5 re-filters 300-6000. (2) `_run_si_sorter`'s external-whitening seed pin is gated on `sorter_params.get("whiten")` (`sorting.py:1892`), which is `False` for MS5 → SI's internal `whiten(seed=None)` runs. MS4 correctly sets `filter=False, whiten=True`. **Fix:** add `filter: bool=False; whiten: bool=True` to `MountainSort5Schema`. |
| A4 | **`Curation.get_curated_sorting` changed `@staticmethod`→instance method; v0 clusterless caller still unbound → crashes under legacy SI** | `spikesorting/v0/spikesorting_curation.py:205`; caller `decoding/v0/clusterless.py:201` | **High** | High | Branch refactored to `def get_curated_sorting(self, key)` calling `self._load_sorting_info`; sibling `get_recording` kept static. All other callers updated to `Curation()....`; `clusterless.py:201` left as `Curation.get_curated_sorting(key)` → `TypeError` (missing positional) under legacy SI 0.99 (masked in CI by the SI≥0.101 guard). **Fix:** restore `@staticmethod` (lower-risk, preserves external API) or update the caller to `Curation().get_curated_sorting(key)`. |
| A5 | **Analyzer `random_spikes` computed without a seed → persisted `peak_amplitude_uv`/peak_channel non-deterministic for >500-spike units** | `spikesorting/v2/sorting.py:2087-2097` | **Med** | High | `_build_analyzer` strips `random_seed` then `compute([...random_spikes...])` with no `seed`; SI `random_spikes_selection` uses `default_rng(None)`. templates/waveforms (→ `get_template_extremum_amplitude`/`_channel`, persisted at `sorting.py:2317`) depend on the subset. Rebuild path re-runs unseeded yet docstring claims "bit-equivalent." **Fix:** `extension_params['random_spikes']['seed'] = (job_kwargs or {}).get('random_seed', 0)`; pass the same seed on rebuild. |
| A6 | **`deprecate_log(alternate=...)` invalid kwarg → `TypeError` on `delete_downstream_merge`** (DEDUP: reported 2×) | `utils/dj_merge_tables.py:958-961` | **Med** | High | Signature is `deprecate_log(cls, name, alt=None, warning=True)` (`common_usage.py:70`); `alternate=` raises on the first line of the shim before the passthrough. Branch regression (master called it positionally). Only the deprecated public shim is affected (no internal callers). **Fix:** `alternate=` → `alt=`. |
| A7 | **`UnitWaveformFeatures.make` v2 dispatch branch is unreachable (legacy guard fires first)** (DEDUP: reported 2×) | `decoding/v1/waveform_features.py:137, 168-194` | Med→**Low** | High | `make()` opens with `_require_legacy_si_environment(...)` → raises under SI≥0.101; the `__curation_v2` branch is only reachable where v2 data doesn't exist. `_fetch_waveform` uses removed `si.WaveformExtractor`/`extract_waveforms(overwrite=True)` (AssertionError under 0.104). Dead code; v2 user gets a clear RuntimeError, not a wrong answer. **Fix:** remove the v2 branch or replace with explicit `NotImplementedError`; document the clusterless-v2 feature-extraction gap (see F). |
| A8 | **`NwbfileHasher` excludes dataset *values* from the file hash** (pre-existing upstream PR #1093; lives in v2 `cache_hash` path) | `utils/nwb_hash.py:326-327` | Crit→**High** | High | `_ = self.hash_dataset(obj)` discards the per-dataset digest; only name+attrs fold into `self.hashed`. Two NWBs identical in shape/dtype/attrs but differing in `ElectricalSeries` values hash identically (empirically reproduced). v2 stores it as `cache_hash` (`recording.py:824`) and compares on rebuild (`recording.py:1225`) — **warn-only**, which softens "silently accepts." Shape/dtype/attr changes *are* still detected. **Fix:** fold `hash_dataset(obj)` into the per-object hash; add a value-difference regression test. (Not a v2 regression, but a real content-hash gap.) |
| A9 | **`LFPElectrodeGroup.cautious_insert` silently drops requested electrodes missing from `Electrode`** (not v2; PR #1302) | `lfp/lfp_electrode.py:172-180` | **Med** | High | Branch fetches `electrode_keys` then builds inserts only from returned keys with **no `len == len(e_ids)` check**; master would have hit a loud FK error. Silently-wrong electrode set feeds LFP/ripple/decoding. **Fix:** assert/warn on `len(electrode_keys) != len(e_ids)` naming the missing ids. |

### Lower-severity correctness / robustness (Low, all confirmed)
- **`min_length_s` uses `>=` where v1 used strict `>`** — `artifact.py:1226`. Measure-zero edge (interval duration exactly 1.0 s); the "matches v1" comment is overstated. Fix operator or comment.
- **Idempotency dedup compares `artifact_id` str-vs-UUID** — `sorting.py:541-558`. Hand-built string keys defeat find-existing → duplicate analyzer build. Normalize both sides with `str(...)` (mirror lines 640-643).
- **Reuse guard omits `apply_merge=True`** — `curation.py:397`. `apply_merge=True` with empty merge_groups silently ignored on existing root (no-op content; only `merges_applied` flag diverges). Add `or bool(apply_merge)` or document.
- **Clusterless `noise_levels` of length ∉ {1, n_channels} passes through unchecked** — `sorting.py:1793`. Silent OOB read (short) / truncation (long) in SI numba kernel. Add a length-in-{1, n_channels} guard. (See pattern P3.)
- **`_consolidate_intervals` sortedness guard tests both columns but sorts only col 0** — `utils.py:541`. Over-triggers a harmless re-sort; output always correct. Test only col 0.
- **`Interval.contains(padding=...)` `IndexError` on empty result** — `common_interval.py:584`. No in-tree caller; latent. Guard `if padding and len(ret):`.
- **`Interval.__init__` uses `valid_times or interval_list`** — `common_interval.py:377`. `Interval(valid_times=arr)` raises numpy ambiguity ValueError; latent (no caller passes `valid_times=` to the class). Use explicit `is not None`.
- **`IntervalList.insert` override ignores `replace=True`** — `common_interval.py:250`. Re-rated **low**: not silent — same-times re-populate still no-ops; differing-times raises a *loud, tested, intended* `ValueError`. Map `replace=True`→`cautious_insert(update=True)` or document.
- **Enum-coercion re-raise drops `__cause__`** — `curation.py:418-425`. Use `from exc` (mirror `sorting.py:2287`).
- **v0 figurl `SpikeSortingView` restricts by non-existent `recording_id`** — `v0/figurl_views/SpikeSortingView.py:52`. Pre-existing dead code; edited but still cannot succeed.
- **Position (not v2, on-branch):** `ImportedPose.fetch_pose_dataframe` drops index name `'time'` → breaks MoSeq `set_index('time')` (`position/v1/imported_pose.py:136`, Med); `DLCPosVideo` inserts a debug row in `limit` mode (`position_dlc_selection.py:514`, Low); `DLCModelTraining` `UnboundLocalError` on no-snapshot (`position_dlc_training.py:260`, Low, pre-existing).

---

## (B) Efficiency

No efficiency *regressions* were found. Relevant positives confirmed: chunked artifact detection (`ChunkRecordingExecutor`, default `chunk_duration='1s'`) is memory-bounded and consumes `job_kwargs`; sparse SortingAnalyzer (`sparse=True`) gives the documented 5-10× storage savings on dense probes; the long sort runs outside the DB transaction. The only adjacent concern is **A1's** over-broad merge-id return causing downstream consumers to load extra sortings — that's filed as a correctness issue, not pure efficiency.

---

## (C) Test coverage gaps & weak tests — highest-value first

**Test-integrity (production code branches on the global test flag) — DEDUP: `group.py:219` reported 4×, one root cause.** `SortedSpikesGroup.fetch_spike_data` gained `and not test_mode`, disabling include/exclude label filtering under pytest. Verified branch-introduced (commit be74ae1c, #1209, not on this repo's master via that lineage). **Net rating Low–Medium:** no production impact (`test_mode=False` in prod; spike_times/unit_ids stay aligned), and v2 label behavior is covered via `CurationV2.get_unit_brain_regions` (no guard there). But the `fetch_spike_data` label-filter path is untested — a `filter_units` regression would pass CI. **Fix:** drop `and not test_mode`; give a fixture real labels, or pass empty include/exclude explicitly. Same anti-pattern to audit: `unit_annotation.py:75-83`, `group.py:88-93`.

**High-severity coverage gaps (behavior shipped, not pinned):**

| Area | Gap | File | Fix |
|---|---|---|---|
| Recording | `_consolidate_intervals` off-by-one fix tested only via integration with ±5% (≈±1500 samples) tolerance — 3 orders too loose to catch a 1-sample/interval drift | `utils.py:499-560` | Hermetic test asserting **exact** `(start, end_frame_exclusive)` pairs incl. adjacency-merge & unsorted-reorder cases |
| Artifact | `amplitude_thresh_uV` gain conversion: every direct detection test uses gain=1.0 (no-op); the one heterogeneous-gain test compares chunked vs a **copy** of the same math | `artifact.py:125` | Hermetic test with known gain (e.g. 0.195 µV/count) asserting an absolute expected detect/no-detect outcome |
| Artifact | Multi-member `SharedArtifactGroup` union scan never exercised (only test is 1-member + `none` preset) | `artifact.py:884-924` | 2-member time-aligned recordings, artifact on one member's channels; assert union scan sees it + identical valid_times per member |
| Curation | Lazy `get_merged_sorting` (preview) vs `apply_merge=True` stored train asserted by **count only**, not frames; docstring claims identical | `curation.py:957-959,1265-1275` | `np.array_equal` of merged unit frames/times between the two paths, contiguous **and** disjoint |
| Curation | `apply_merge=True` cross-gap stored train untested (disjoint test only covers lazy path) | `sorting.py`/`curation.py` | Apply merge on the disjoint 2-unit fixture; assert both gap-boundary frames survive |
| Sorting/SI | Analyzer extension SET & params (`['random_spikes','noise_levels','templates','waveforms']`, `max_spikes_per_unit=500`, `ms_before/after`) + `create_sorting_analyzer(sparse=True, return_in_uV=True)` not asserted | `sorting.py:2069-2097` | Capture `compute` args + `create_sorting_analyzer` kwargs in the existing fake-analyzer test |
| Pipeline | Idempotency tests assert PK equality only, never **row counts == 1** in Selection/Curation tables | `pipeline.py` | Add `len(...)==1` assertions after a second `run_v2_pipeline` |
| Type-design | Param Lookups override only `insert1`, not bulk `insert([...])` → plural insert bypasses Pydantic + schema-version check (DEDUP: reported 2× — robustness finding + coverage gap) | `sorting.py:117`, `recording.py:656`, `artifact.py:234`, `session_group.py:143` | Add `insert` override mapping `_validate_params` over rows (mirror `SortGroupV2`/`UnitLabel`); test plural insert raises |
| Export | **No export-completeness test for v2 at all** (see D) | — | DB test capturing v2 NWB files in `ExportSelection.File` |

**Notable weak tests (pass with a plausible bug present):**
- `test_disjoint_sort_intervals_concatenated` — ±5% length tolerance can't catch the 1-sample/interval fix it guards.
- `test_chunked_artifact_matches_in_memory_reference` — reference is a line-for-line copy of the production math (self-consistency, not correctness).
- `test_detect_artifacts_amplitude_and_zscore_combined` — baseline trips amplitude everywhere, so AND-vs-OR is untestable as written.
- `test_sorting_get_sorting_round_trips` — contiguous grid; affine and searchsorted readback agree, so it can't guard the parity fix (only the slow disjoint test does).
- `test_boundary_spike_round_trip_does_not_raise` — tests only the in-bounds boundary; never plants an out-of-bounds spike to confirm `_remove_excess_spikes` drops it.
- `test_build_analyzer_strips_random_seed` — stubs `compute` and ignores its args (the actual contract).
- `test_rebuild_analyzer_folder_recreates_on_missing` — asserts unit_ids only, far weaker than the docstring's "bit-equivalent."
- Consumer shape tests (`test_consumer_api_shape_contract`, MUA tests) assert shape/sign/finite only — a sparse-unit_id column misalignment passes. **Fix:** assert `get_spike_indicator(...).sum(axis=0)[j]` equals the count of `get_spike_times()[j]` in window.
- `test_curation_v2_auto_registers_in_merge_table` — never asserts the `region_resolution` value or region_name through the merge dispatch.
- Concat schema test pins one column; the zero-migration freeze needs full PK/FK/`end_sample` bigint shape pinned.

---

## (D) Cross-cutting & export/downstream risks

| # | Risk | File:line | Sev | Note |
|---|---|---|---|---|
| D1 | **A1 restated as cross-cutting** — artifact restriction no-op surfaces on the public `get_spiking_sorting_v2_merge_ids` API too | `spikesorting_merge.py:236`; `v2/utils.py:764` | High | Same root cause as A1; the public notebook wrapper advertises `artifact_id` as a valid restriction. |
| D2 | **Export completeness unverified (critical gap)** — no test starts an export and asserts v2-owned AnalysisNwbfiles (CurationV2 units + Recording cache) land in `ExportSelection.File` | `utils/mixins/export.py:434-480` | **Critical (coverage)** | Export logging only fires through `fetch_nwb`. A missing file → incomplete DANDI/Kachery bundle, invisible until off-site reload. |
| D3 | **Helper accessors bypass `fetch_nwb`** — `CurationV2.get_sorting/get_merged_sorting/get_unit_brain_regions/get_recording` and `Recording.get_recording` read via `get_abs_path()` directly | `curation.py:1130,1156-1159,1278`; `recording.py:1146-1168` | High | Files reached only via these documented notebook accessors are **not** registered during an active export. Either make them export-capturing or document that export must go through `fetch_nwb` / `get_spike_times`. |
| D4 | **Zero-unit export path untested** — empty-but-real units NWB through `_log_fetch_nwb` (IN/= restriction from possibly-zero-row fetch) | `curation.py:1042-1048`; `export.py:444-480` | High | The #1532/#1154 edge v2 promised end-to-end; likely place for `KeyError: 'spike_times'`/empty-tuple SQL. |
| D5 | **Legacy SI 0.99 import break (unverified-needs-DB but high-confidence static):** three v0/v1 modules now `import spikeinterface.metrics.quality as sq` (the 0.10x path), which doesn't exist under `>=0.99,<0.101`; `v1/__init__.py:17` imports `metric_curation` eagerly → `ModuleNotFoundError` at collection in `pytest-legacy` | `v1/metric_utils.py:5`, `v1/metric_curation.py:12`, `v0/spikesorting_curation.py:15` | High | **Fix:** `try: import spikeinterface.metrics.quality as sq / except ModuleNotFoundError: import spikeinterface.qualitymetrics as sq`. |
| D6 | **Legacy CI env is unsatisfiable** — pip resolves `..[test]` (hard `spikeinterface==0.104.3`, pyproject:69) together with the override `spikeinterface>=0.99,<0.101` in one resolve → `ResolutionImpossible`; `pytest-legacy` job fails before any test | `environments/environment_spikesorting_legacy.yml:45-48`; `.github/workflows/test-conda.yml:328-333` | High | **Fix:** install legacy SI first + `pip install -e "..[test]" --no-deps`, or relax the pyproject pin. |
| D7 | **Burst-merge tutorial calls nonexistent `plot_by_sorting_ids`** (v1 class defines `plot_by_sort_group_ids`) | `notebooks/py_scripts/12_Burst_Merge_Curation.py:80`; `.ipynb:219` | Med | One-token rename in both paired files. |
| D8 | **`skip_if_no_dlc` uses a lambda condition → ALL 15 DLC tests unconditionally skipped** even with DLC installed (skipif `bool()`s the lambda, always truthy) | `tests/conftest.py:656` | High (test-integrity) | Not v2 code, but on-branch. **Fix:** `condition=getattr(pytest,"NO_DLC",False)` (module-level bool) or string form. |
| D9 | Six sorted-spikes decoding integration tests newly `@pytest.mark.skip("JAX issues")` (incl. pure-pandas `test_get_orientation_col`) | `tests/decoding/test_spikes.py:14` | Low | Pre-existing test-speed PR #1440; rooted in a real JAX/CI limit. Prefer `importorskip`/`xfail`. |

**Stale doc / minor infra (Low):** CHANGELOG self-contradicts on artifact chunking (`:200` stale vs `:434` current); `run-tests` CI passes `--cov=spyglass-neuro` (dead arg, works only via addopts `--cov=spyglass`); trailing-whitespace pre-commit hook both excludes `*.md` and passes a markdown-only arg; `list_presets()` doctest shows wrapped output vs single-line repr; `run_v2_pipeline` returns undocumented `n_units` key; comment-rot at `recording.py:646,615,624` and `artifact.py:860` (transitional "now N"/"formerly-dead" narration violating the project comment rule).

---

## (E) v1/v2 divergences

### Concerning (verified, undocumented/contradictory or with real risk)

| Aspect | v1 → v2 | Why it's a problem | Fix |
|---|---|---|---|
| **CHANGELOG self-contradiction on artifact chunking** | n/a (doc defect) | `CHANGELOG.md:200-205` says detection loads fully into RAM and `job_kwargs` is unconsumed; shipped code chunks via `ChunkRecordingExecutor` and consumes `job_kwargs` (`artifact.py:863`), per `CHANGELOG.md:434-440`. An operator sizing memory for chronic-scale runs is misled. | Delete/annotate the `:200` bullet as superseded. |

No **unverified-risky** divergences exist in this batch (the unverified-risky list was empty).

Two additional verified divergences that are *justified but documented inconsistently* and warrant a doc fix (carried into F):
- **Clusterless `detect_threshold` is not truly µV** — docstrings (`sorter.py:169-173`, `sorting.py:1730-1732`) say "microvolts" but the recording is in raw counts; `CHANGELOG.md:273-285` admits it's an unfixed v1-inherited confusion. Reconcile docstring vs CHANGELOG. (Default-row users protected.)
- **Zero-unit pipeline return contract** — code builds a real empty merge/curation row (`pipeline.py:222-278`), but `CHANGELOG.md:259-264`, `exceptions.py:86` (user-facing docstring), and `phase-1-...:171` still describe the abandoned `curation_id=None/merge_id=None` manifest. Correct the three doc sources.

### Justified divergences (for the record) — all verified sound

These are intentional, documented improvements/bugfixes/design choices; no action needed beyond the doc fixes noted above.

| Theme | Representative aspects (all "justified") |
|---|---|
| **Recording/preproc** | PK-only `insert_selection`; heterogeneous-gain → raise; `reference_mode` split from sentinel int; `_consolidate_intervals` exclusive-end fix; always-on non-monotonic timestamp repair w/ provenance; dropped dead `margin_ms`/`seed`/`reference`; `no_filter`=None real disable; `electrodes_id`/`update_ids` not ported; `channel_slice`→`ChannelSliceRecording`; `return_scaled`→`return_in_uV`; tetrode contact-id int→str + skip logging; `RecordingTruncatedError`; guarded `set_group_by_shank`. |
| **Artifact** | gain-aware µV thresholds (the headline fix); default 3000→500; `artifact_{uuid}` IntervalList naming, no `skip_duplicates`; z-score on µV + epsilon (scale-invariant, equivalent under uniform gain); new `join_window_ms`; per-chunk base intervals (gap-safe); frames(ceil) + half-open `end_f+1`; PK-only idempotent `insert_selection`; proportion out-of-range now rejected (was clamped); new `SharedArtifactGroup`. |
| **Sorting** | tri-part `make` (sort outside txn + cleanup); `time_of_sort` int→datetime; per-unit metadata persisted; gap-safe artifact mask reimpl; `EmptyArtifactValidTimesError`; seed=0 pins; global job_kwargs restore; MATLAB-sorter carve-out; `np.Inf` monkeypatch; searchsorted readback + `as_dataframe`; zero-unit real empty row; `params_schema_version` sentinel-0; default-row install-gate via `installed_sorters()`; MS4 rename + back-compat aliases; clusterless `threshold_unit` knob; source-part pattern + optional ArtifactSource; concat `key_source` antijoin. |
| **Curation/merge** | applied-merge 0.4 ms dedup (fixes v1 inconsistency); overlapping/singleton/dup merge-group rejection; lazy dedup in abs-time (gap-safe); root-reuse `reuse_existing` guard (raises on conflicting params — **needs migration note, see F**); always-dict return; parent-existence validation; UnitLabel enum validation; int-key/scalar-string label coercion; `curation_label` omitted when unlabeled; merge_groups/metrics moved to MergeGroup part; `get_sort_group_info` all-electrodes fix; `get_restricted_merge_ids` raises on unknown keys. |
| **Params/schema** | Pydantic on every Lookup insert; schema-version enforcement asymmetry (justified by table cardinality); generic-sorter escape hatch; Motion schema lands before consumer; preprocessing field flattening + whiten=None; WhitenParams inert scaffolding. |
| **Downstream/UX** | PK-only `insert_selection` across selections; positional→true-NWB-unit_id indexing (fixes sparse-id mis-attribution in group/unit_annotation/waveform_features); `get_restricted_merge_ids` conditional v2 default source. |

---

## (F) Improvement gaps over v1

**Should do now (true):**
- **F1 — Document `reuse_existing` raise-on-conflicting-params** in CHANGELOG migration notes (`curation.py:391-414`). v1 silently returned; v2 raises `ValueError` when labels/merge_groups/description passed to an existing root. Code correct; docs lag.
- **F2 — Correct the stale zero-unit return-contract docs** (`CHANGELOG.md:259-264`, `exceptions.py:86`, `phase-1-...:171`) to match the shipped real-empty-merge-row behavior.
- **F3 — Clusterless waveform-feature extraction unavailable for v2 under SI 0.104** (`waveform_features.py:137` + dead v2 branch). Either provide a v2-native SortingAnalyzer-based path, or document a two-environment workflow, and remove/`NotImplementedError`-mark the dead branch. Clusterless decoding is a primary consumer; the limitation is non-obvious.

**Stubbed-roadmap (do NOT flag as bugs — these raise clean ImportError on attribute access):**
- `unit_matching.py`, `matcher_protocol.py` (Phase 4 cross-session unit matching / `TrackedUnit` / `TrackedUnitBudgetExceededError`).
- `metric_curation.py` (Phase 2 `AnalyzerCuration.materialize_curation`, BurstPair auto-merge).
- `figpack_curation.py` (Phase 5 FigPack UI; v1 FigURL remains for v1 data).
- Concatenated/multi-session sorting (schema final-shape, all paths `NotImplementedError`-gated, `key_source` antijoin) — Phase 3.

**Not-addressed / partial, correctly deferred (should_do_now=false):**
- v1↔v2 `cache_hash` byte-equivalence (intentional, by-design divergence).
- v1↔v2 byte-level clusterless/MS4 parity (intentional; bounded by documented asymmetric tolerance bands; run-to-run reproducibility delivered).
- True-µV clusterless `detect_threshold` (roadmap; but reconcile docstring↔CHANGELOG — see E).
- `EmptyArtifactValidTimesError` not in the contract's §27 exception list (doc gap).
- Cortex/Neuropixels orchestrator presets (only 3 tetrode presets ship; manual path documented).
- Curated NWB lacking merge_groups/metric columns for external DANDI tooling (internal consumers use `.get`; confirm export path does too).

---

## Named patterns (generalizations) — fix once, audit all sites

- **P1 — Dict restriction on a query whose heading lacks the key is a silent no-op.** Root of A1/D1. Audit every `query & {k: v}` where `k` comes from a multi-table key but the query joins only some tables: `_get_restricted_merge_ids_v2` rec/curation restrictions; `resolve_artifact`/`resolve_source` (these read through the part *correctly*); any direct `& {'artifact_id':...}`/`& {'sorting_id':...}` on a master.
- **P2 — `insert1` overridden for validation but bulk `insert` not** (DataJoint `insert1`→`insert`, never the reverse). Sites: the four param Lookups (`sorting.py:117`, `recording.py:656`, `artifact.py:234`, `session_group.py:143`). Correct exemplars guarding both: `SortGroupV2` (`recording.py:135-143`), `CurationV2.UnitLabel` (`curation.py:169-189`).
- **P3 — Per-channel array params forwarded to SI without a length-vs-n_channels guard.** Sites: clusterless `noise_levels` (`sorting.py:1793`); check artifact `gains` usage.
- **P4 — `pytest.approx(0, rel=...)` / shape-only assertions can't distinguish the intended output from a real bug.** Sites: `test_position.py:66` (sum==0 passes for NaN *and* all-zeros); consumer shape/sign/finite tests (sparse-unit_id misalignment passes). Assert the actual property (`isna().all()`, per-unit count alignment).
- **P5 — Production code branches on the global `test_mode` flag to skip scientific logic.** Sites: `group.py:219` (label filter), `group.py:88-93` (duplicate-group guard), `unit_annotation.py:75-83` (unit-id validity). Skip I/O, not correctness checks.
- **P6 — Comment-rot via transitional "now N"/"formerly-dead" narration** (violates the project comment-style rule). Sites: `recording.py:615,624,646`; `artifact.py:860`.

---

## What was NOT checked / remaining risks (completeness critic)

Static analysis was sufficient for almost everything; nothing below was executed against a live DB or a real sort (per the no-spawn/no-hang constraints). Highest-value remaining checks, in order:

1. **Export end-to-end (D2-D4)** — *no agent ran an export.* Whether v2 NWB files (CurationV2 units + Recording cache) are captured in `ExportSelection.File`, whether helper accessors register during export, and whether the zero-unit path survives `_log_fetch_nwb` are all **unverified**. This is the single biggest unknown for "export-safe." Next check: the DB test in TOP-5 #4 / D2.
2. **Legacy SI 0.99 import + env build (D5/D6)** — confirmed statically (import path, pip resolution mechanism) but *not executed under SI 0.99 / a real `pytest-legacy` run*. Next check: build the legacy env and `import spyglass.spikesorting.v1` under SI 0.99/0.100.
3. **A2 runtime outcome** — whether multi-source `fetch_nwb(return_merge_ids=True)` *raises* (fetch1 on 0-tuple) vs *silently misaligns* depends on populated DB state; verified from code + DataJoint source, not a populated run. Either way it's broken for ≥2 sources.
4. **MS5 empirical magnitude (A3)** — the double-filter/unseeded-whitening code path is unambiguous, but the actual spectral/unit-set divergence was not measured (needs live DB + mountainsort5). Run two MS5 populates and diff unit sets to quantify.
5. **Analyzer rebuild determinism (A5)** — the unseeded `random_spikes` chain is confirmed, but the resulting peak_amplitude/peak_channel drift across rebuilds was not measured on real >500-spike units (smoke fixture units may be <500).
6. **Multi-segment / 3+-chunk disjoint recordings** — all disjoint fixtures are 2-chunk/single-gap; `_base_intervals_from_timestamps` zip-indexing and the 1.5/fs gap threshold/jitter-robustness are untested at the decision boundary and with ≥2 gaps.
7. **`NwbfileHasher` consumers beyond v2** (`v1/recompute.py`, `common_nwbfile.get_hash`) — the value-exclusion (A8) was confirmed; downstream reliance on detecting value changes elsewhere was not traced.

Concurrency note honored: no Docker MySQL container was started and the MEArec fixture was not (re)generated by this synthesis; all DB-tier items above are reported as needing a live run rather than executed.