# Spike sorting v2 in shared code

How code outside `spyglass.spikesorting.v2` supports v2 sources while v0/v1 keep
working.

## Optional v2 layer in `SpikeSortingOutput`

`spikesorting_merge.py`:

- `CurationV2` and `ConcatMemberCuration` are imported in `try`/`except`, so
    environments without the v2 dependencies still load the merge table. Their
    part tables are declared only when the import succeeds, because DataJoint
    needs the foreign-key target at declaration time.
- The probe catches `Exception`, not only `ImportError`, because v2 can also
    fail on a dependency version skew. The error is logged and kept in
    `_v2_import_error`; `_raise_v2_unavailable` re-raises it when a v2-only
    helper is called, so users see the real cause.
- `_available_merge_sources()` lists `"v2"` only when it imported, so
    `get_restricted_merge_ids()` with default sources works without v2. With
    default sources the v2 resolver ignores keys meant for v0/v1; an explicit
    `sources` list keeps it strict so typos raise.

## Warn or raise at the consumer boundary

- Query helpers warn: `get_restricted_merge_ids` on several curations of one
    sorting, `CurationV2.get_sorting` and the spike accessors on preview
    curations, and the spike accessors on merges that span pipeline sources.
    They stay usable for inspection.
- Decoding-group creation raises (`assert_decoding_merge_ids_ok`), so a
    dangerous set never reaches a decoder. It raises on a `CurationV2` made with
    `apply_merge=False` whose proposed merges are unapplied (concat members
    inherit their parent curation's state), and on two merge ids for the same
    sorting or the same concat member. v0/v1 sources are skipped.
- `get_spike_times_by_unit` is the common sink for `get_firing_rate`,
    `get_spike_indicator` and `get_spike_times`, so it warns once. It warns on
    the merge ids that `fetch_nwb(return_merge_ids=True)` resolved, so the
    warnings match what was read. It passes `multi_source=True`, so
    `_warn_multi_source_merge_ids` replaces `fetch_nwb`'s multi-source error
    with a warning.

## Legacy SpikeInterface guard

`spikesorting/_legacy_runtime.py`:

- v0/v1 compute paths that call SI APIs removed or renamed in 0.101 (sorting,
    waveforms, quality metrics, artifact detection, burst curation, clusterless
    features) call `_require_legacy_si_environment()` first. It raises a clear
    error instead of an opaque failure inside SI, and does nothing before SI
    0.101.
- Read paths are not guarded. Renamed loading and `NumpySorting` APIs go through
    `_si_compat`; saved binary waveform folders load as a
    `MockWaveformExtractor` under SI 0.101+ (`_si_compat.load_waveforms`).
- `clusterless_thresholder` sorting is exempt: it uses `detect_peaks`, not
    `run_sorter`, whose `output_folder` argument SI 0.101+ rejects.

## MUA detection over observed time

`mua/v1/_detection.py`:

- A group whose units were not observed for the whole time axis carries NaN in
    those bins. Passed straight to `ripple_detection.multiunit_HSE_detector`,
    `gaussian_smooth` (`truncate=8`) spreads each NaN bin across the smoothed
    rate, and the z-score's `nan_policy="omit"` then renormalizes over what
    survives, which can erase real events and invent others.
- `detect_multiunit_events_in_observed_runs` runs the detector's steps on each
    contiguous observed run with one normalization shared across all observed
    samples, including runs too short to hold an event. A timestamp jump over
    1.5 sample periods also ends a run; `close_event_threshold` merges only
    within a run.
- The module holds no DataJoint tables, so it is unit testable without a
    database.
- `firing_rate_over_observed_runs` and the MUA FigURL split runs the same way.
    Unobserved bins are NaN (missing evidence), not zero (silence).

## Clusterless waveform features from v2 sources

`decoding/v1/waveform_features.py`:

- v2 marks come from a fresh `SortingAnalyzer` over the merge source's recording
    and sorting (`random_spikes` with `method="all"`, then `waveforms`), wrapped
    in `_AnalyzerWaveformAccessor` so the shared feature helpers and the NWB
    writer serve both v0/v1 `WaveformExtractor`s and v2.
- The persisted v2 analyzer is not reused: it subsamples spikes
    (`max_spikes_per_unit`, 20000 in the shipped recipes), while clusterless
    decoding needs every spike's amplitude aligned 1:1 with `spike_times`.
    `sparse=False` keeps every channel.
- A finite `max_spikes_per_unit` is rejected before the analyzer build.
    `_write_waveform_features_to_nwb` rechecks 1:1 alignment for v2 because
    features and `spike_times` are separate reads.
- Amplitudes are in microvolts (`return_in_uV=True`); v0/v1 marks are raw ADC
    counts. A model trained on v1 marks cannot be applied to v2 marks without
    rescaling.
- `format="binary_folder"` memmaps the waveforms, so memory stays bounded for
    millions of spikes.
- The accessor holds strong references to the analyzer (SI extensions only
    weakref it) and the `TemporaryDirectory`. `__del__` drops the memmap first
    because NFS blocks removing a still-mapped file.
- A zero-unit sort gets an accessor with `analyzer=None`.
- `spike_location` features need a v2 source; v0/v1 sources raise
    `NotImplementedError`.

## `UnitAnnotation` unit ids

`spikesorting/analysis/v1/unit_annotation.py`:

- `UnitAnnotation.unit_id` is the NWB units-table id, not a position in the
    spike-times list. They differ for sparse id sets: v2 merges, and v1 merges,
    which assign `max(ids) + 1` and drop the constituents.
- Rows written with positional ids are found by `audit_positional_unit_ids`
    (merges whose NWB ids are not `0..n-1`) and remapped by
    `migrate_positional_unit_ids`. The migration validates the full plan first,
    then inserts a per-merge marker (`UnitAnnotationPositionalIdMigration`) in
    the same transaction as the rewrite. The marker goes first so a concurrent
    migration collides before rows are rewritten.
- `add_annotation` refuses merges with unmigrated rows. Any write it makes marks
    the merge in the same transaction, so later writes skip the audit's NWB
    read. It checks the unrestricted table, since a restricted `self` could hide
    rows and mark an unmigrated merge. Inside a caller's transaction it joins
    that transaction; the caller must roll back on failure or the merge stays
    marked without rows.
- `fetch_unit_spikes` reads markers before rows: a marker guarantees NWB ids,
    and a missing marker refuses a sparse read even if a migration commits
    between the two queries.
