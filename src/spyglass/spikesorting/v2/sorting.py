"""Spike sorting and per-unit brain-region metadata.

Tables:
    SorterParameters          -- Per-sorter Pydantic-validated params.
    SortingSelection          -- Source-polymorphic sorting request.
        .RecordingSource          -- single-session source.
        .ConcatenatedRecordingSource -- concat source (same-day chronic).
        .ArtifactDetectionSource  -- optional artifact mask (single source).
        .MotionCorrectionSource   -- optional motion-corrected recording of
                                     the source, sorted in its place.
    Sorting (+ Unit)          -- Sorted units NWB + SortingAnalyzer folder.

``SorterParameters.insert1`` dispatches to the per-sorter Pydantic
schema via ``_get_sorter_schema``. ``insert_selection`` resolves a
sorting request to a single ``sorting_id``, ``make`` runs the
sorter and writes the units NWB + analyzer, and the accessor methods
(``get_sorting``, ``get_analyzer``) read those back. Both source kinds
populate: a recording source loads ``Recording``, a concatenated-recording
source loads ``ConcatenatedRecording`` and anchors the per-unit Electrode FK
and the analysis-NWB parent to the first frozen
``ConcatenatedRecordingSelection.MemberSnapshot`` member.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common import IntervalList  # noqa: F401
from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._params.analyzer_waveform import (
    ANALYZER_WAVEFORM_SCHEMA_VERSION,
    AnalyzerWaveformParamsSchema,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    sorter_default_contents,
    waveform_params_default_contents,
)
from spyglass.spikesorting.v2._sorting.analyzer import (
    build_analyzer,
    load_or_rebuild_analyzer,
    rebuild_analyzer_folder,
)
from spyglass.spikesorting.v2._sorting.artifact_mask import (
    apply_artifact_mask,
    sorting_statistics_spans,
)
from spyglass.spikesorting.v2._sorting.dispatch import (
    remove_excess_spikes,
    run_clusterless_thresholder,
    run_si_sorter,
    sort_runtime_versions,
)
from spyglass.spikesorting.v2._storage import analyzer_cache as _analyzer_cache
from spyglass.spikesorting.v2._sorting import (
    parameters as _sorter_parameters,
    fetch as _sorting_fetch,
    selection as _sorting_selection_insert,
    units as _sorting_units,
)
from spyglass.spikesorting.v2._recording.source import (
    EffectiveSource,
    EffectiveTraces,
    SourceLineage,
    correction_lineage_mismatch,
    effective_source_from_base,
    effective_source_from_correction,
    sorting_parts_mismatch,
)
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
    StagedOutputs,
)
from spyglass.spikesorting.v2._storage.units_nwb import (
    STATISTICS_SPANS_FIELD,
    StoredUnits,
    abs_spike_times_dataframe,
    empty_spike_times_dataframe,
    read_units_abs_spike_times,
    read_sorting_statistics_spans,
    sorting_from_units_nwb,
    write_sorting_units_nwb,
)
from spyglass.spikesorting.v2.artifact_output import (
    ArtifactDetectionOutput,
)
from spyglass.spikesorting.v2.motion import (
    MotionCorrectedRecording,
    MotionCorrectedRecordingSelection,
)
from spyglass.spikesorting.v2.recording import Recording  # noqa: F401
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,  # noqa: F401
)
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _insert_parameter_rows,
    find_orphaned_masters,
    split_leading_restrictions,
)
from spyglass.spikesorting.v2._recording.source import SourceResolution
from spyglass.spikesorting.v2._core.job_config import resolve_effective_seed
from spyglass.spikesorting.v2._recording.unit_metadata import (
    unit_brain_region_df,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import pandas as pd
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.analyzer_cache import StagedAnalyzer

#: The table owning each effective-traces kind's cached NWB artifact.
_TRACE_TABLES = {
    "recording": Recording,
    "concatenated_recording": ConcatenatedRecording,
    "motion_corrected_recording": MotionCorrectedRecording,
}


class SortingFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`Sorting.make_fetch`.

    Attributes
    ----------
    source : SourceResolution
        Resolved sort input source (``kind == "recording"`` or
        ``"concatenated_recording"``).
    recording_id : str
        The anchor ``recording_id``: the sort's own recording for a
        single-recording source, or the FIRST frozen ``MemberSnapshot``
        member's recording for a concat source (the deterministic parent anchor
        for the per-unit ``Electrode`` FK and the analysis-NWB parent).
    sel_row : dict
        The ``SortingSelection`` row, with ``artifact_detection_id`` resolved
        and stashed on it for downstream readers.
    sorter_row : dict
        The matching ``SorterParameters`` row (``sorter``, ``params``,
        ``job_kwargs``).
    nwb_file_name : str
        Anchor NWB file (the single-recording session, or the first concat
        member's session).
    obs_intervals : numpy.ndarray or None
        Valid observation intervals on the source's timeline. Concat sources
        supply their stored synthetic-time intervals; standalone sources
        supply detected valid times, or ``None`` to derive recording intervals
        when no artifact detection was selected.
    display_waveform_params_name : str
        The DISPLAY ``AnalyzerWaveformParameters`` recipe resolved from the
        source preprocessing recipe (region), stored on the ``Sorting`` row.
    display_waveform_params : dict
        That recipe's resolved params blob, threaded into ``_build_analyzer``
        so ``make_compute`` receives the resolved parameter input.
    execution_params : dict
        The validated ``SorterParameters.execution_params`` blob (sorter
        execution backend + container provenance), resolved here so
        ``make_compute`` can pass it directly to the sorter dispatch.
    """

    source: SourceResolution
    recording_id: str
    sel_row: dict
    sorter_row: dict
    nwb_file_name: str
    obs_intervals: np.ndarray | None
    display_waveform_params_name: str
    display_waveform_params: dict
    execution_params: dict
    # Per-unit Electrode FK resolution, threaded so make_compute builds the
    # Sorting.Unit rows without DB writes (anchored to the recording / first
    # concat member). ``region_by_electrode`` maps electrode_id -> brain region
    # for the NWB ``brain_region`` column.
    sort_group_id: int
    electrode_by_id: dict
    region_by_electrode: dict
    # A concat source's stored ``ConcatenatedRecording.statistics_spans``
    # (``(n, 2)`` int64 concat frame ranges); ``None`` for a single-recording
    # source, whose spans ``make_compute`` derives from the reloaded recording.
    concat_statistics_spans: np.ndarray | None
    # The sort's effective traces from
    # ``SortingSelection.resolve_effective_source``: the cached artifact
    # ``make_compute`` loads as the sorter input, and its absolute path
    # (the file rebuilt here if it was missing).
    traces: EffectiveTraces
    traces_abs_path: str
    # For a sort of a motion-corrected recording: the correction's ids and
    # recipe names written to the units NWB provenance, and the source's frame
    # count the corrected traces must keep. ``None`` for an uncorrected sort.
    motion_correction_provenance: dict | None
    source_n_samples: int | None


class SortingComputed(NamedTuple):
    """Compute -> insert carrier for :meth:`Sorting.make_compute`.

    NONE of these fields are ``Sorting`` columns -- they are values threaded
    from ``make_compute`` into ``make_insert`` (NWB staging, lookups,
    unit-part inserts). ``staged_analyzer`` retains the private build and its
    ownership lock until insertion either publishes or discards it. The cache
    folder is not a DB column; every reader resolves the
    canonical location from ``sorting_id`` via
    ``_analyzer_cache.analyzer_path``.

    NOTE: the field ORDER here is a positional wire contract -- the tri-part
    dispatch unpacks this tuple positionally into ``make_insert``
    (``make_insert(key, *make_compute_result)``). Keep it in sync with
    ``make_insert``'s parameter order;
    ``test_sorting_computed_matches_make_insert_signature`` pins the
    alignment.

    Attributes
    ----------
    sorting_obj : spikeinterface.BaseSorting
        The computed sorting (unit ids + spike trains).
    analysis_file_name : str
        Staged-but-unregistered AnalysisNwbfile holding the units.
    units_object_id : str
        NWB object id of the units table inside that file.
    nwb_file_name : str
        Source NWB file backing the recording selection.
    display_waveform_params_name : str
        The resolved DISPLAY recipe name; stored on the ``Sorting`` row by
        ``make_insert`` so every later rebuild reads it back deterministically.
    effective_random_seed : int
        The random seed the sort actually used (``resolve_effective_seed``),
        recorded as secondary provenance -- NOT identity.
    spikeinterface_version : str
        Producing runtime's ``spikeinterface.__version__`` (host for local
        execution, container for Docker/Singularity; secondary provenance).
    sorter_version : str or None
        Producing sorter version (container sorter reports its runtime version;
        local execution uses the installed distribution). ``None`` for local
        in-process / SI-internal sorters (secondary provenance).
    unit_rows : list of dict
        The ``Sorting.Unit`` rows built ONCE in ``make_compute`` (peak channel /
        amplitude / spike count), reused by ``make_insert`` for the part insert
        so the DB and the NWB cannot drift. Empty for a zero-unit sort.
    staged_analyzer : StagedAnalyzer
        Private analyzer and cleanup ownership, transferred to ``make_insert``.
    """

    sorting_obj: "si.BaseSorting"
    analysis_file_name: str
    units_object_id: str
    nwb_file_name: str
    display_waveform_params_name: str
    effective_random_seed: int
    spikeinterface_version: str
    sorter_version: str | None
    unit_rows: list[dict]
    staged_analyzer: StagedAnalyzer

    def staged_outputs(self) -> StagedOutputs:
        """The staged units NWB and private analyzer ``make_insert`` takes."""
        return StagedOutputs(
            analysis_file_names=(self.analysis_file_name,),
            owners=(self.staged_analyzer,),
        )


schema = dj.schema("spikesorting_v2_sorting")


@schema
class SorterParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Per-sorter Pydantic-validated parameter blob.

    The ``params`` blob is validated by the per-sorter schema returned by
    ``_get_sorter_schema(sorter)``. ``insert_default`` ships explicit
    default rows for MS4, MS5, KS4, SC2, TDC2, and
    ``clusterless_thresholder``. Users can insert additional rows for any
    installed SI sorter; the generic ``extra="allow"`` schema is the
    fallback dispatch for non-default sorters, the "try any installed
    sorter" escape hatch.
    """

    definition = """
    sorter: varchar(64)
    sorter_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=0: int
    job_kwargs=null: blob
    execution_params: blob               # validated SorterExecutionParamsSchema dump
    execution_params_schema_version=1: int
    """
    # ``execution_params`` records the sorter EXECUTION backend (local vs
    # Docker/Singularity) + container-side install provenance, validated by
    # ``SorterExecutionParamsSchema``. It is tracked here -- not in the scientific
    # ``params`` blob and not in ``job_kwargs`` -- because the backend can change
    # the sort output and ``sorter_params_name`` already flows into ``sorting_id``.
    # A row that omits it backfills the schema default (``backend="local"``).
    # Switching backend for an existing logical row therefore requires a NEW
    # ``sorter_params_name`` (a local MS4 row and a containerized MS4 row are
    # distinct named rows), not an in-place mutation.

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Validate every row against its per-sorter schema, then insert.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob, scoped per sorter); see
        ``reject_duplicate_parameter_content``.
        """

        _insert_parameter_rows(
            self,
            rows,
            insert_rows=super().insert,
            validate_rows=lambda batch, names: _sorter_parameters.validate_sorter_rows(
                batch, names, self._NON_SI_SORTERS
            ),
            table_name="SorterParameters",
            name_attr="sorter_params_name",
            sorter_keyed=True,
            allow_duplicate_params=allow_duplicate_params,
            **kwargs,
        )

    # The shipped rows are defined in
    # ``_recipe_catalog.sorter_default_contents`` (single source).
    _DEFAULT_CONTENTS: tuple = sorter_default_contents()

    # Sorter names in ``_DEFAULT_CONTENTS`` that are NOT SpikeInterface
    # registered sorters and so must never be gated on
    # ``sis.installed_sorters()``. ``clusterless_thresholder`` is a
    # Spyglass-internal peak detector built on ``detect_peaks``.
    _NON_SI_SORTERS: frozenset[str] = frozenset({"clusterless_thresholder"})

    @classmethod
    def _is_seed_dependent_sort(cls, sorter: str, sorter_params: dict) -> bool:
        """Whether the sort's output depends on ``random_seed``.

        A seed-dependent sort must reject an ambient-only seed (present in the
        SI-global / ``dj.config`` layer but not in the ``SorterParameters`` row):
        the seed changes the output but is not folded into ``sorting_id``, so a
        later ambient-seed change would silently reuse this sort.

        SI sorters cluster stochastically, so they are always seed-dependent.
        The ``clusterless_thresholder`` peak detector is deterministic EXCEPT on
        its per-channel MAD noise path (``threshold_unit='mad'`` with no explicit
        ``noise_levels``), which estimates noise from randomly sampled chunks;
        the ``'uv'`` path and an explicit ``noise_levels`` are deterministic.
        """
        if sorter not in cls._NON_SI_SORTERS:
            return True
        return (
            sorter_params.get("threshold_unit") == "mad"
            and sorter_params.get("noise_levels") is None
        )

    @classmethod
    def insert_default(cls):
        """Insert v2 default sorter rows if missing.

        The default-content catalog includes MS4, MS5, KS4, SC2,
        TDC2, and clusterless_thresholder. Rows whose SpikeInterface
        sorter is NOT in ``spikeinterface.sorters.installed_sorters()``
        are skipped (logged at INFO) -- otherwise a user who inserts an
        uninstalled sorter's default row and then populates ``Sorting``
        hits an unhelpful "sorter not registered" error from SI. MS4 and
        KS4 are the common uninstalled cases (their Python wrappers exist
        even when the runtime/binary is absent, so ``available_sorters``
        is too lax). ``installed_sorters`` gates the row INSERT here -- a
        mild check that only decides whether to ship the default params
        row. It is NOT proof a sort will run: the mountainsort4 check only
        locates the package without importing its compiled dependencies.
        ``preflight_v2_pipeline``'s ``sorter_runtime_available`` check is
        the actual runtime gate.

        ``clusterless_thresholder`` is never gated (it is a Spyglass
        peak-detection special case, not an SI registered sorter). The
        full catalog stays in ``_DEFAULT_CONTENTS`` for introspection;
        only the insert is gated. See the per-sorter Pydantic schemas in
        ``spyglass.spikesorting.v2._params.sorter`` for the validated
        field surface.
        """
        insertable, skipped = cls._gated_default_rows()
        for row in skipped:
            logger.info(
                "SorterParameters.insert_default: skipping default "
                f"row {row[1]!r} -- sorter {row[0]!r} is not in "
                "spikeinterface.sorters.installed_sorters() on this "
                "platform."
            )
        cls.insert(insertable, skip_duplicates=True)

    @classmethod
    def _gated_default_rows(cls):
        """Split ``_DEFAULT_CONTENTS`` into (insertable, skipped) by install.

        A default row is insertable when its sorter is a Spyglass-internal
        sorter (``_NON_SI_SORTERS``, never gated), is in
        ``spikeinterface.sorters.installed_sorters()``, OR runs on a tracked
        container backend (``execution_params.backend`` in
        ``{"docker", "singularity"}``). Container rows ship even when the local
        sorter runtime is unavailable -- their runtime lives in the image, so a
        selected container row is gated by preflight (the container runtime
        check), not by local-install at default-row insertion. Returns
        ``(insertable, skipped)`` so ``insert_default`` can log the skips and
        tests can assert the gating decision without depending on the live
        ``SorterParameters`` table state.
        """
        import spikeinterface.sorters as sis

        from spyglass.spikesorting.v2._sorting.dispatch import (
            is_container_backend,
        )

        installed = set(sis.installed_sorters())
        insertable: list = []
        skipped: list = []
        for row in cls._DEFAULT_CONTENTS:
            sorter = row[0]
            # row = (sorter, name, params, params_sv, job_kwargs,
            #        execution_params, execution_params_sv)
            if (
                sorter in cls._NON_SI_SORTERS
                or sorter in installed
                or is_container_backend(row[5])
            ):
                insertable.append(row)
            else:
                skipped.append(row)
        return insertable, skipped

    @classmethod
    def insert_default_legacy_si_sorters(cls):
        """Insert ('sorter','default') rows for installed non-curated sorters.

        Opt-in helper for users porting v1 workflows that name a non-curated
        sorter via ``('kilosort2_5','default')`` or similar. Each row holds
        the SI wrapper's own algorithm defaults (without SI's global job
        kwargs), validated through ``GenericSorterParamsSchema`` and the
        wrapper vocabulary; a row that would be rejected at insert is skipped
        with a warning instead of aborting the batch.

        Three classes of sorter are skipped:

        - **Not installed** (not in ``installed_sorters()``, the same gate as
          :meth:`insert_default`; logged at INFO). A wrapper exposes defaults
          even when its binary is absent, and such a row would only fail at
          ``Sorting.populate``.
        - **MATLAB** sorters, which need a container ``execution_params``; a
          local ``'default'`` row would be rejected at populate.
        - **Curated** (mountainsort4, mountainsort5, kilosort4,
          spykingcircus2, tridesclous2, clusterless_thresholder). Their typed
          ``extra='forbid'`` schemas intentionally strip some SI defaults
          (e.g. ``MountainSort5Schema`` drops ``filter`` / ``freq_min``
          because the recording is already filtered), so SI's full default
          dict would fail validation or lose keys.

        Not called by ``initialize_v2_defaults``, so users who do not need v1
        sorter names do not pay for the inserts. Idempotent via
        ``skip_duplicates=True``.

        Examples
        --------
        >>> from spyglass.spikesorting.v2.sorting import SorterParameters
        >>> SorterParameters.insert_default()
        >>> SorterParameters.insert_default_legacy_si_sorters()
        """

        cls.insert(
            _sorter_parameters.legacy_si_sorter_rows(), skip_duplicates=True
        )


def _reject_unsafe_waveform_params_name(row, _schema_cls) -> None:
    """Reject a non-path-safe ``waveform_params_name`` at insert.

    A ``per_row_hook`` for ``validate_lookup_rows``; delegates to the DB-free
    :func:`._analyzer_cache.assert_path_safe_waveform_params_name` (the owner of
    the analyzer-path policy), which the load path reuses.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        assert_path_safe_waveform_params_name,
    )

    assert_path_safe_waveform_params_name(row["waveform_params_name"])


@schema
class AnalyzerWaveformParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Tracked window / subsample / whitening for a sort's analyzer recipe.

    Mirrors v1's ``WaveformParameters`` so the settings that produced an
    analyzer (``ms_before`` / ``ms_after`` / ``max_spikes_per_unit`` / whitening)
    are recorded in the DB rather than hardcoded in the analyzer build. The
    ``params`` blob is validated by :class:`AnalyzerWaveformParamsSchema`;
    ``insert_default`` ships the region-specific display/metric rows
    (hippocampus 0.5/0.5 ms, cortex 1.0/2.0 ms; both 20000 spikes).

    A sort's DISPLAY recipe is resolved from its source preprocessing recipe
    (region) and persisted on ``Sorting.display_waveform_params_name`` -- the
    row name is not a free per-sort knob and is not part of ``sorting_id``
    identity. ``analyzer_path`` embeds ``waveform_params_name`` in the cache
    folder name, so the name is validated path-safe (``^[A-Za-z0-9_]+$``) at
    insert time.
    """

    definition = f"""
    waveform_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={ANALYZER_WAVEFORM_SCHEMA_VERSION}: int
    """

    # The shipped region rows are defined in
    # ``_recipe_catalog.waveform_params_default_contents`` (single source).
    _DEFAULT_CONTENTS: tuple = waveform_params_default_contents()

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Validate each ``params`` blob + path-safe name, then insert.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob); see
        ``reject_duplicate_parameter_content``.
        """
        _insert_parameter_rows(
            self,
            rows,
            insert_rows=super().insert,
            schema_for=lambda _row: AnalyzerWaveformParamsSchema,
            table_name="AnalyzerWaveformParameters",
            per_row_hook=_reject_unsafe_waveform_params_name,
            name_attr="waveform_params_name",
            allow_duplicate_params=allow_duplicate_params,
            **kwargs,
        )

    @classmethod
    def insert_default(cls):
        """Insert the region display/metric waveform recipes (idempotent)."""
        cls().insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class SortingSelection(SelectionMasterInsertGuard, SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` (single-session) or ``ConcatenatedRecordingSource``
    (same-day chronic concatenation) exists for each selection row.
    ``insert_selection`` dispatches on the requested source and both kinds
    populate through ``Sorting.make``.

    Whether an artifact-detection pass was applied is recorded by the
    presence or absence of an ``ArtifactDetectionSource`` part row
    (zero-or-one), NOT by a nullable FK on the master. A nullable FK
    conflates "no artifact-detection pass" with "match anything" in a
    restriction and forces every reader to special-case ``None``; the part
    row makes "no ``ArtifactDetectionSource`` row" queryable, joinable, and
    impossible to alias. ``ArtifactDetectionSource`` is independent of the
    recording-source parts -- it is NOT counted by ``resolve_source``
    (a sort still has exactly one *recording* source) nor by
    ``prune_orphaned_selections``.

    A ``MotionCorrectionSource`` part (zero-or-one) makes the sort read a
    ``MotionCorrectedRecording`` of its source instead of the source's own
    traces. The source parts still record the lineage; the corrected
    recording must have been estimated on that source under the sort's
    artifact mask, and its id is part of ``sorting_id``.
    """

    definition = """
    sorting_id: uuid
    ---
    -> SorterParameters
    """

    class RecordingSource(SpyglassMixinPart):
        """Single-session recording source for a sorting selection."""

        definition = """
        -> master
        ---
        -> Recording
        """

    class ConcatenatedRecordingSource(SpyglassMixinPart):
        """Concatenated-recording source for a sorting selection."""

        definition = """
        -> master
        ---
        -> ConcatenatedRecording
        """

    class ArtifactDetectionSource(SpyglassMixinPart):
        """Optional artifact-detection pass for a sorting selection.

        Present iff an artifact detection was configured for the sort;
        absent means "no artifact-detection pass." Deliberately separate from the
        recording-source parts so ``resolve_source``'s "exactly one
        recording source" invariant is unaffected -- read it through
        :meth:`SortingSelection.resolve_artifact_detection`.
        """

        definition = """
        -> master
        ---
        -> ArtifactDetectionOutput.proj(artifact_detection_merge_id='merge_id')
        """

    class MotionCorrectionSource(SpyglassMixinPart):
        """Optional motion-corrected recording a sort reads.

        Present iff the sort reads a ``MotionCorrectedRecording`` of its
        source instead of the source's own traces. Separate from the source
        parts, which keep recording the sort's lineage; read it through
        :meth:`SortingSelection.resolve_motion_correction`.
        """

        definition = """
        -> master
        ---
        -> MotionCorrectedRecording
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert master + exactly one source part; return PK-only dict.

        Reads exactly one of ``recording_id`` (single-session) or
        ``concat_recording_id`` (same-day chronic concatenation) from ``key``;
        raises ValueError on zero or two sources, and inserts the matching
        source part (``RecordingSource`` or ``ConcatenatedRecordingSource``).
        ``artifact_detection_id`` is optional: when supplied (non-None), an
        ``ArtifactDetectionSource`` part row records the artifact-detection
        pass; when omitted/None, no ``ArtifactDetectionSource`` row is created.
        The find-existing path keys on the presence/absence and identity
        of that part row, so an artifact-backed and an artifact-free
        selection for the same ``(recording_id, sorter,
        sorter_params_name)`` are distinct, idempotent rows.
        ``motion_corrected_recording_id`` is optional too: when supplied, a
        ``MotionCorrectionSource`` part row makes the sort read that
        populated ``MotionCorrectedRecording``, which must have been
        estimated on this source under this artifact detection, and the
        sorter must not correct motion itself.

        Parameters
        ----------
        key : dict
            Selection request. Must carry exactly one of ``recording_id`` or
            ``concat_recording_id``, plus ``sorter`` and
            ``sorter_params_name``. ``artifact_detection_id`` and
            ``motion_corrected_recording_id`` are optional; an explicit
            ``sorting_id`` is cross-checked against the derived deterministic
            id.

        Returns
        -------
        dict
            Primary-key-only dict (``{"sorting_id": ...}``) for the
            inserted-or-existing master row.

        Raises
        ------
        ValueError
            If ``key`` carries a field other than those listed above (a
            misspelled key is refused, not dropped), if zero or both source
            keys are supplied, if a concat source also supplies an
            ``artifact_detection_id`` (concat member masks
            are configured on ``ConcatenatedRecordingSelection``), or if a
            motion-corrected recording is not populated, was estimated on
            another source or mask, or is paired with a sorter that corrects
            motion itself.
        DuplicateSelectionError
            If any matching master has a non-deterministic ``sorting_id``
            (a raw ``insert`` bypass or a legacy non-content-addressed row) --
            even a single one; an integrity bug, not user error.
        SchemaBypassError
            If a deterministic master exists but its recording/artifact-detection
            source parts are missing/mismatched (a raw-insert orphan).
        """

        return _sorting_selection_insert.insert_selection(cls, key)

    @classmethod
    def _find_existing_pk(
        cls,
        master_restriction,
        source_restriction,
        artifact_detection_id,
        deterministic_id,
        source_part,
        motion_corrected_recording_id,
    ) -> dict | None:
        """Return the canonical master PK for this sort selection, or None.

        See :func:`._sorting_selection_insert.find_existing_pk`. Tests patch
        it, so it stays on the class and ``insert_selection`` calls it through
        the class.
        """
        return _sorting_selection_insert.find_existing_pk(
            cls,
            master_restriction,
            source_restriction,
            artifact_detection_id,
            deterministic_id,
            source_part,
            motion_corrected_recording_id,
        )

    @classmethod
    def prune_orphaned_selections(cls, dry_run: bool = True) -> list[dict]:
        """Find or delete master rows that have no source-part row.

        DataJoint cannot enforce "exactly one recording source per master"
        across the two XOR source parts, so an upstream cascade-delete from
        ``Recording`` / ``ConcatenatedRecording`` can leave a master row with no
        source child. The optional ``ArtifactDetectionSource`` and
        ``MotionCorrectionSource`` parts are not sources: a master left with
        only those is an orphan too, and deleting it removes them. Dry-run by default; with ``dry_run=False`` runs
        cautious_delete on each orphan so the cascade preview shows downstream
        ``Sorting`` / ``CurationV2`` / ``SpikeSortingOutput.CurationV2`` impact.
        """
        orphans = find_orphaned_masters(
            cls,
            [cls.RecordingSource, cls.ConcatenatedRecordingSource],
        )
        if dry_run or not orphans:
            return orphans
        for orphan in orphans:
            (cls & orphan).cautious_delete()
        return orphans

    @classmethod
    def resolve_source(cls, key: dict) -> SourceResolution:
        """Return the source-resolution record for a sorting selection.

        Layer 2 of the source-part pattern: fetches source-part rows for
        the master key, asserts exactly one exists across both source
        parts, and returns a ``SourceResolution(kind, key)`` so
        ``Sorting.make`` can dispatch on source shape without inspecting
        the raw part tables.

        Raises
        ------
        SchemaBypassError
            If zero or multiple source part rows exist for ``key``.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        rec_rows = (cls.RecordingSource & master_key).fetch(as_dict=True)
        concat_rows = (cls.ConcatenatedRecordingSource & master_key).fetch(
            as_dict=True
        )
        total = len(rec_rows) + len(concat_rows)
        if total != 1:
            raise SchemaBypassError(
                f"SortingSelection {master_key} has {total} source part "
                "rows; expected exactly one. Use "
                "SortingSelection.insert_selection() to add or remove "
                "this selection."
            )
        if rec_rows:
            return SourceResolution(
                kind="recording",
                key={"recording_id": rec_rows[0]["recording_id"]},
            )
        return SourceResolution(
            kind="concatenated_recording",
            key={"concat_recording_id": concat_rows[0]["concat_recording_id"]},
        )

    @classmethod
    def resolve_artifact_detection(cls, key: dict):
        """Return the ``artifact_detection_id`` for a selection, or ``None``.

        Reads the optional ``ArtifactDetectionSource`` part row -- which stores
        the ``ArtifactDetectionOutput`` ``artifact_detection_merge_id`` -- and
        resolves it back to the per-source ``artifact_detection_id`` (the
        id-of-record folded into ``sorting_id`` and naming the IntervalList).
        Returns ``None`` when no sorting-stage artifact input was configured.
        This includes concat sorts, whose detections instead live on frozen
        ``ConcatenatedRecordingSelection.MemberSnapshot`` rows. ``None`` here
        does not imply that a concat source is unmasked. This is the accessor
        for the optional sorting-stage input; it returns the natural
        ``artifact_detection_id`` (or ``None``), never the merge id.

        Raises
        ------
        SchemaBypassError
            If more than one ``ArtifactDetectionSource`` row exists for ``key``
            (the part is zero-or-one by construction).
        """
        merge_id = cls._optional_part_value(
            key, cls.ArtifactDetectionSource, "artifact_detection_merge_id"
        )
        if merge_id is None:
            return None
        return ArtifactDetectionOutput.resolve_artifact_detection_id(merge_id)

    @classmethod
    def resolve_motion_correction(cls, key: dict):
        """Return the sort's ``motion_corrected_recording_id``, or ``None``.

        Reads the optional ``MotionCorrectionSource`` part; ``None`` means the
        sort reads its source's own traces.

        Raises
        ------
        SchemaBypassError
            If more than one ``MotionCorrectionSource`` row exists for ``key``.
        """
        return cls._optional_part_value(
            key, cls.MotionCorrectionSource, "motion_corrected_recording_id"
        )

    @classmethod
    def _optional_part_value(cls, key: dict, part, column: str):
        """Return ``column`` of a selection's zero-or-one part row, or ``None``.

        Raises
        ------
        SchemaBypassError
            If more than one ``part`` row exists for ``key``.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        rows = (part & master_key).fetch(column)
        if len(rows) > 1:
            raise SchemaBypassError(
                f"SortingSelection {master_key} has {len(rows)} "
                f"{part.__name__} rows; expected zero or one."
            )
        return rows[0] if len(rows) else None

    @classmethod
    def resolve_effective_source(cls, key: dict) -> EffectiveSource:
        """Return a sort's lineage and the traces its consumers must read.

        Built on :meth:`resolve_source` and :meth:`resolve_artifact_detection`.
        Every consumer that loads the traces a sort ran on (sorter input,
        analyzer builds and rebuilds, metric curation, the recompute audit,
        the curation recording accessor) reads ``traces``; metadata consumers
        keep using :meth:`resolve_source`. The effective traces are the lineage
        source's own cached artifact, or the ``MotionCorrectedRecording`` the
        sort selected, fetched here; no trace file is opened.

        Parameters
        ----------
        key : dict
            Restriction carrying the ``SortingSelection`` primary key.

        Returns
        -------
        EffectiveSource
            ``lineage`` (source kind, source key, pinned artifact detection)
            and ``traces`` (owning table, key, fetched row, whether the loaded
            traces must still be artifact-masked).

        Raises
        ------
        SchemaBypassError
            If the selected corrected recording was not made from the sort's
            source and artifact mask, or the current source, artifact
            detection and motion-correction parts do not give the stored
            ``sorting_id`` (a part row inserted or deleted without
            :meth:`insert_selection`).
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        source = cls.resolve_source(key)
        lineage = SourceLineage(
            kind=source.kind,
            key=source.key,
            artifact_detection_id=cls.resolve_artifact_detection(key),
        )
        corrected_id = cls.resolve_motion_correction(key)
        if corrected_id is not None:
            mismatches = correction_lineage_mismatch(
                lineage,
                MotionCorrectedRecordingSelection.resolve_source(
                    {"motion_corrected_recording_id": corrected_id}
                ),
            )
            if mismatches:
                raise SchemaBypassError(
                    f"SortingSelection {dict(key)} reads motion-corrected "
                    f"recording {corrected_id}, which was not made from its "
                    f"source and mask ({'; '.join(mismatches)}). The "
                    "MotionCorrectionSource part was inserted without "
                    "SortingSelection.insert_selection; drop the selection "
                    "and re-insert it."
                )
        # sorting_id folds in every part, so a part inserted or deleted after
        # the selection (and after its sort) shows up as an id mismatch.
        master = (
            cls & {k: v for k, v in key.items() if k in cls.primary_key}
        ).fetch1()
        mismatch = sorting_parts_mismatch(
            master["sorting_id"],
            lineage,
            sorter=master["sorter"],
            sorter_params_name=master["sorter_params_name"],
            motion_corrected_recording_id=corrected_id,
        )
        if mismatch is not None:
            raise SchemaBypassError(
                f"SortingSelection {master['sorting_id']}: its current parts "
                f"({mismatch}) are not the ones it was selected with. A "
                "source, ArtifactDetectionSource or MotionCorrectionSource "
                "part was inserted or deleted without "
                "SortingSelection.insert_selection, so its consumers would "
                "read other traces than its sorter did. Drop the selection "
                "(and its sort) and re-insert it with insert_selection."
            )
        if corrected_id is None:
            row = (_TRACE_TABLES[source.kind] & source.key).fetch1()
            return effective_source_from_base(lineage, row)
        corrected_key = {"motion_corrected_recording_id": corrected_id}
        row = (MotionCorrectedRecording & corrected_key).fetch1()
        return effective_source_from_correction(lineage, corrected_key, row)

    @staticmethod
    def ensure_effective_traces(traces: EffectiveTraces) -> str:
        """Rebuild the effective traces' cached NWB file if it is missing.

        The same self-heal the owning table's ``get_recording`` performs
        (``Recording``, ``ConcatenatedRecording`` or
        ``MotionCorrectedRecording``): a locked, verified rebuild whose
        content must match the stored ``content_hash``. Call it before
        :func:`._source_resolution.load_effective_recording`, or use
        :meth:`load_stored_traces`.

        Parameters
        ----------
        traces : EffectiveTraces
            The ``traces`` of :meth:`resolve_effective_source`.

        Returns
        -------
        str
            Absolute path of the (present) file, for
            :func:`._source_resolution.read_effective_recording`.
        """
        from spyglass.spikesorting.v2._storage.rebuilds import (
            ensure_artifact_file,
        )

        return ensure_artifact_file(
            _TRACE_TABLES[traces.kind],
            traces.key,
            traces.row["analysis_file_name"],
        )

    @staticmethod
    def resolve_stored_units(
        units_analysis_file_name: str,
        source: EffectiveSource,
    ) -> StoredUnits:
        """Resolve a v2 Units file and its source sampling rate for readback.

        Stored sample frames make recording artifacts unnecessary. A corrected
        recording preserves its lineage source's frames and sampling rate.
        """
        lineage, traces = source
        recording_row = (
            traces.row
            if traces.kind == lineage.kind
            else (_TRACE_TABLES[lineage.kind] & lineage.key).fetch1()
        )
        return StoredUnits(
            abs_path=AnalysisNwbfile.get_abs_path(units_analysis_file_name),
            sampling_frequency=float(recording_row["sampling_frequency"]),
        )

    @staticmethod
    def load_stored_traces(traces: EffectiveTraces) -> "si.BaseRecording":
        """Open the effective traces as persisted, rebuilding a missing file.

        Self-heals through :meth:`ensure_effective_traces`, then opens the
        file it resolved with
        :func:`._source_resolution.read_persisted_traces` (no artifact mask
        applied at load).

        Parameters
        ----------
        traces : EffectiveTraces
            The ``traces`` of :meth:`resolve_effective_source`.

        Returns
        -------
        si.BaseRecording
            The persisted traces, annotated ``is_filtered=True``.
        """
        from spyglass.spikesorting.v2._recording.source import (
            read_persisted_traces,
        )

        return read_persisted_traces(
            SortingSelection.ensure_effective_traces(traces), traces
        )

    @classmethod
    def resolve_source_preprocessing_params_name(cls, key: dict) -> str:
        """Return the sort source's ``preprocessing_params_name``.

        Region resolution input: a sort's analyzer waveform window is keyed on
        the SAME signal that sets the region filter cutoff -- the source
        preprocessing recipe. Reuses :meth:`resolve_source` for source
        detection (the single source-part integrity check) rather than
        re-inspecting the source part tables, then reads the source's
        ``preprocessing_params_name``:

        - ``RecordingSource`` -> the upstream ``RecordingSelection`` row.
        - ``ConcatenatedRecordingSource`` -> the upstream
          ``ConcatenatedRecordingSelection`` row (its
          ``-> PreprocessingParameters`` FK's primary key, not a literal column
          in the class definition).

        Does NOT query ``RecordingSelection`` for a concat source: concat member
        recordings are provenance inputs, but the concatenated source row is the
        sort input.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecordingSelection,
        )

        source = cls.resolve_source(key)
        if source.kind == "recording":
            return (
                RecordingSelection
                & {"recording_id": source.key["recording_id"]}
            ).fetch1("preprocessing_params_name")
        return (
            ConcatenatedRecordingSelection
            & {"concat_recording_id": source.key["concat_recording_id"]}
        ).fetch1("preprocessing_params_name")


@schema
class Sorting(
    _analyzer_cache.AnalyzerPublicationMixin,
    StagedOutputCleanupMixin,
    SpyglassMixin,
    dj.Computed,
):
    """Sorted units NWB + SortingAnalyzer folder.

    ``make()`` resolves the source recording, applies sorter-owned
    preprocessing such as external MS4/MS5 whitening when requested,
    runs the sorter, removes excess spikes, builds a
    ``SortingAnalyzer(format="zarr", sparse=True)``, computes
    the base extensions (``random_spikes``, ``noise_levels``,
    ``templates``, ``waveforms``), writes a fresh/whitelisted
    ``AnalysisNwbfile`` containing only the v2 sorting Units (NOT the
    parent NWB's units), and populates ``Sorting.Unit``.
    """

    definition = """
    -> SortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(72)
    n_units: int
    time_of_sort: datetime    # wall-clock time the sort was populated
    -> AnalyzerWaveformParameters.proj(display_waveform_params_name="waveform_params_name")
    effective_random_seed=null: int     # seed actually used (resolve_effective_seed); provenance, NOT identity
    spikeinterface_version: varchar(32) # producing runtime's spikeinterface.__version__
    sorter_version=null: varchar(64)    # producing sorter version, NULL when no separate local distribution
    """
    # ``display_waveform_params_name`` is a secondary FK to
    # ``AnalyzerWaveformParameters``: it records which DISPLAY recipe produced
    # this sort's analyzer + peak_amplitude_uv, and the FK enforces the
    # provenance in the database -- a sort cannot be inserted referencing a
    # recipe that is not tracked, and a referenced recipe row cannot be deleted.
    # The name is resolved at sort time from the source preprocessing recipe
    # (region; see _recipe_catalog.waveform_params_for_preprocessing) and read
    # back -- never re-resolved -- on every later rebuild, so the analyzer is
    # deterministic for a sorting_id. It is NOT a free per-sort knob and is NOT
    # part of sorting_id identity. The whitened METRIC recipe is carried on
    # CurationEvaluationSelection.metric_waveform_params_name, not here.
    # The SortingAnalyzer cache folder is intentionally NOT a column: it is
    # large (5-50 GB) regeneratable scratch resolved at runtime from
    # (sorting_id, display_waveform_params_name) via _analyzer_cache.analyzer_path.
    # Persisting an absolute path here would drift from the
    # accessor-computed path whenever temp_dir changes between runs.

    class Unit(SpyglassMixinPart):
        """Per-unit metadata persisted at sort time.

        Brain region is reached through the ``Sorting.Unit * Electrode *
        BrainRegion`` join (``Electrode`` carries a NON-NULL ``BrainRegion``
        FK in Spyglass). For concat sorts the Electrode FK is anchored to
        the FIRST member's row.
        """

        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uv: float    # peak template amplitude in microvolts
        n_spikes: int
        """

    # Both source kinds populate: ``make_fetch`` resolves the source (recording
    # vs concatenated_recording) and dispatches accordingly, so the default
    # ``key_source`` (the full ``SortingSelection``) is correct -- no antijoin.

    # Tri-part make keeps the sort (routinely 5-20 minutes) outside the
    # populate transaction; ``_parallel_make`` allows parallel populate via
    # the non-daemon process pool.
    _parallel_make = True

    def make_fetch(self, key):
        """Read every DB input the compute step needs.

        Layer-2 source re-check fires here (``resolve_source`` asserts exactly
        one input source). Dispatches on the source: a single-recording source
        anchors to its own ``RecordingSelection``; a concat source anchors
        deterministically to the FIRST frozen ``MemberSnapshot`` member (so the
        per-unit ``Electrode`` FK and the analysis-NWB parent both resolve to the
        anchor member) and reads the concat-owned observation intervals. A sort
        of a motion-corrected recording also resolves the correction's
        provenance and its source's frame count. All returned values are
        deterministic bytes (DataJoint fetches inline dicts) so DataJoint's
        tri-part DeepHash integrity check across the two fetches stays stable.

        Parameters
        ----------
        key : dict
            Primary key restricting to one ``SortingSelection`` row.

        Returns
        -------
        SortingFetched
            DB inputs (source, anchor ``recording_id``, ``sel_row``,
            ``sorter_row``, anchor ``nwb_file_name``, ``obs_intervals``,
            display recipe, execution params) for the compute step.
        """

        return _sorting_fetch.fetch_sorting_inputs(key)

    @staticmethod
    def resolve_anchor_nwb_file_name(key) -> str:
        """Return the analysis-NWB parent ``nwb_file_name`` for a sort.

        Source-agnostic: a single-recording sort anchors to its own
        ``RecordingSelection``; a concat sort anchors to the FIRST frozen
        ``MemberSnapshot`` member (the deterministic parent the per-unit Electrode
        FK and the curated/analyzer NWBs all use). Centralizes the
        unwrap-to-nwb dispatch that several reporting / curation accessors need,
        so the "which member is the anchor" decision lives in exactly one place.

        Parameters
        ----------
        key : dict
            Restriction carrying ``sorting_id``.

        Returns
        -------
        str
            The anchor session's ``nwb_file_name``.
        """

        return _sorting_fetch.resolve_anchor_nwb_file_name(key)

    def make_compute(
        self,
        key,
        source,
        recording_id,
        sel_row,
        sorter_row,
        nwb_file_name,
        obs_intervals,
        display_waveform_params_name,
        display_waveform_params,
        execution_params,
        sort_group_id,
        electrode_by_id,
        region_by_electrode,
        concat_statistics_spans,
        traces,
        traces_abs_path,
        motion_correction_provenance,
        source_n_samples,
    ) -> SortingComputed:
        """Sort, build analyzer, stage Units NWB outside any DB transaction.

        Reads only the inputs ``make_fetch`` resolved; the one DB access left
        is staging the units NWB file (see :mod:`._recording_nwb`). Loads the
        effective traces, silences a single recording's artifact frames at
        0 uV and resolves the statistics spans
        (:func:`._sorting_artifact_mask.sorting_statistics_spans`), runs the
        sorter, removes excess spikes, builds the analyzer in a private staged
        folder, and stages the units NWB without registering it.

        DataJoint does not call ``make_insert`` once this raises, so cleanup
        happens here: any exception after the analyzer is staged discards the
        staged analyzer, and a failed units-NWB write removes its own file.

        Parameters
        ----------
        key : dict
            Primary key of the sorting being populated.
        source : SourceResolution
            Resolved sort input source from ``make_fetch`` (selects whether the
            statistics spans are derived here or read from the concat row).
        recording_id : str
            The anchor ``recording_id`` (the sort's own recording, or the first
            concat member's) that the ``Sorting.Unit`` Electrode FK uses.
        sel_row : dict
            The ``SortingSelection`` row (with ``artifact_detection_id``).
        sorter_row : dict
            The ``SorterParameters`` row (``sorter``, ``params``,
            ``job_kwargs``).
        nwb_file_name : str
            Source NWB file backing the recording selection.
        obs_intervals : numpy.ndarray or None
            Artifact-removed valid-times window, or ``None`` when no
            artifact-detection pass is configured.
        display_waveform_params_name : str
            The DISPLAY recipe; selects the analyzer cache folder and is
            persisted by ``make_insert``.
        display_waveform_params : dict
            That recipe's resolved params blob, passed to ``_build_analyzer``.
        execution_params : dict
            The validated sorter execution backend / container provenance,
            passed to the sorter dispatch.
        sort_group_id : int
            The anchor recording's sort group.
        electrode_by_id : dict
            ``{electrode_id: SortGroupElectrode row}`` for that sort group.
        region_by_electrode : dict
            ``{electrode_id: region_name}``; an electrode without a region has
            no entry.
        concat_statistics_spans : numpy.ndarray or None
            A concat source's stored statistics spans, used as is; ``None``
            for a single-recording source.
        traces : EffectiveTraces
            The sort's effective traces from ``make_fetch``: the cached
            artifact loaded as the sorter input.
        traces_abs_path : str
            That artifact's absolute path, resolved (and the file rebuilt if
            missing) in ``make_fetch``.
        motion_correction_provenance : dict or None
            For a sort of a motion-corrected recording, the correction's ids
            and recipe names, written to the units NWB provenance; ``None``
            otherwise.
        source_n_samples : int or None
            The source's frame count a corrected recording must keep;
            ``None`` for an uncorrected sort.

        Returns
        -------
        SortingComputed
            Carrier of the computed sorting, staged units NWB, analyzer
            folder, and lookups threaded into ``make_insert``.
        """
        # Read by path with no DB access. A single recording's traces load
        # unmasked: sorting_statistics_spans applies its artifact mask, whose
        # excluded ranges also feed the statistics spans. Concat and
        # motion-corrected traces are stored already masked.
        from spyglass.spikesorting.v2._recording.source import (
            read_persisted_traces,
        )

        recording = read_persisted_traces(traces_abs_path, traces)

        # Statistics spans: the artifact-free frame ranges every noise and
        # whitening estimate samples from, persisted with the sort so each
        # later analyzer rebuild reuses them. A masked single recording is
        # silenced over its artifact frames here.
        recording, statistics_spans = sorting_statistics_spans(
            recording,
            traces,
            source_kind=source.kind,
            recording_id=recording_id,
            artifact_detection_id=sel_row.get("artifact_detection_id"),
            obs_intervals=obs_intervals,
            concat_statistics_spans=concat_statistics_spans,
            source_n_samples=source_n_samples,
        )

        sorter = sorter_row["sorter"]
        sorter_params = dict(sorter_row["params"])
        # ``schema_version`` is Pydantic bookkeeping; the SI sorter
        # wrapper does not accept it.
        sorter_params.pop("schema_version", None)

        # One resolution feeds both the sorter and the analyzer build, so an
        # n_jobs override (dj.config or the row's job_kwargs) reaches both.
        from spyglass.spikesorting.v2._core.job_config import (
            _resolved_job_kwargs,
        )

        job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])
        import spikeinterface as si

        # The stored seed is resolved from the same row blob as job_kwargs, so
        # it equals the job_kwargs['random_seed'] the sort consumes.
        # reject_ambient_seed: a seed-dependent sort's seed MUST live in the
        # SorterParameters row (part of sorter_params_name -> sorting_id), never
        # the ambient dj.config layer, or a later ambient-seed change would
        # silently reuse this sort under an unchanged id. The clusterless
        # thresholder is exempt ONLY on its deterministic paths -- its per-channel
        # MAD noise estimate (threshold_unit='mad', no explicit noise_levels)
        # samples random chunks and IS seed-dependent (see
        # _is_seed_dependent_sort), so that path is NOT exempt.
        effective_random_seed = resolve_effective_seed(
            sorter_row["job_kwargs"],
            reject_ambient_seed=SorterParameters._is_seed_dependent_sort(
                sorter, sorter_params
            ),
        )
        sorting_obj = self._run_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=key["sorting_id"],
            job_kwargs=job_kwargs,
            execution_params=execution_params,
            statistics_spans=statistics_spans,
        )
        spikeinterface_version, sorter_version = sort_runtime_versions(
            sorting_obj, sorter, execution_params
        )
        sorting_obj = self._remove_excess_spikes(sorting_obj, recording)

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            StagedAnalyzer,
            analyzer_path,
        )

        # Each compute owns its analyzer until its database insert succeeds.
        # Metadata must come from this attempt, never a concurrent publication.
        staged_analyzer = StagedAnalyzer(
            analyzer_path(key["sorting_id"], display_waveform_params_name)
        )
        try:
            self._build_analyzer(
                sorting=sorting_obj,
                recording=recording,
                key=key,
                sorter_row=sorter_row,
                job_kwargs=job_kwargs,
                analyzer_folder=staged_analyzer.folder,
                waveform_params=display_waveform_params,
                statistics_spans=statistics_spans,
            )
            # Compute the per-unit rows ONCE here (from the analyzer just built) and
            # reuse them for BOTH the NWB unit columns and the Sorting.Unit insert in
            # make_insert -- so the file and the DB cannot drift, and the peak
            # channel/amplitude are not computed twice.
            unit_rows = _sorting_units.build_unit_rows_from_analyzer(
                sorting=sorting_obj,
                analyzer_folder=staged_analyzer.folder,
                sorter_row=sorter_row,
                electrode_by_id=electrode_by_id,
                sort_group_id=sort_group_id,
                nwb_file_name=nwb_file_name,
                key=key,
            )
            unit_metadata = {
                int(row["unit_id"]): {
                    "peak_amplitude_uv": row["peak_amplitude_uv"],
                    "peak_electrode_id": int(row["electrode_id"]),
                    "n_spikes": int(row["n_spikes"]),
                    "brain_region": region_by_electrode.get(
                        int(row["electrode_id"])
                    ),
                }
                for row in unit_rows
            }
            concat_recording_id = (
                source.key["concat_recording_id"]
                if source.kind == "concatenated_recording"
                else None
            )
            source_provenance = {
                "sorting_id": str(key["sorting_id"]),
                "recording_id": (
                    source.key["recording_id"]
                    if source.kind == "recording"
                    else None
                ),
                "concat_recording_id": concat_recording_id,
                "sorter": sorter_row["sorter"],
                "sorter_params_name": sorter_row["sorter_params_name"],
                "sorter_params": sorter_row["params"],
                # The execution backend (container / engine) can change the sorter
                # output, so it is part of the named-parameter-row provenance.
                "execution_params": execution_params,
                "artifact_detection_id": sel_row.get("artifact_detection_id"),
                "display_waveform_params_name": display_waveform_params_name,
                "effective_random_seed": effective_random_seed,
                "spikeinterface_version": spikeinterface_version,
                "sorter_version": sorter_version,
                "analyzer_spikeinterface_version": si.__version__,
                STATISTICS_SPANS_FIELD: [
                    [int(a), int(b)] for a, b in statistics_spans
                ],
            }
            if motion_correction_provenance is not None:
                source_provenance.update(motion_correction_provenance)
            from spyglass.spikesorting.v2._core.runtime import (
                runtime_environment_provenance,
            )

            source_provenance.update(
                runtime_environment_provenance(
                    job_kwargs=job_kwargs, execution_params=execution_params
                )
            )
            analysis_file_name, units_object_id = self._stage_sorting_artifact(
                sorting=sorting_obj,
                recording=recording,
                nwb_file_name=nwb_file_name,
                obs_intervals=obs_intervals,
                unit_metadata=unit_metadata,
                source_provenance=source_provenance,
            )

            return SortingComputed(
                sorting_obj=sorting_obj,
                analysis_file_name=analysis_file_name,
                units_object_id=units_object_id,
                nwb_file_name=nwb_file_name,
                display_waveform_params_name=display_waveform_params_name,
                effective_random_seed=effective_random_seed,
                spikeinterface_version=spikeinterface_version,
                sorter_version=sorter_version,
                unit_rows=unit_rows,
                staged_analyzer=staged_analyzer,
            )
        except BaseException:
            staged_analyzer.close()
            raise

    def make_insert(
        self,
        key,
        sorting_obj,
        analysis_file_name,
        units_object_id,
        nwb_file_name,
        display_waveform_params_name,
        effective_random_seed,
        spikeinterface_version,
        sorter_version,
        unit_rows,
        staged_analyzer,
    ):
        """Atomic registration of the AnalysisNwbfile + master + Unit rows.

        DataJoint's tri-part dispatch wraps this method in the
        framework transaction; the inner ``_safe_context()`` is
        a no-op there (kept defensively). ``_populate_unit_part``
        runs INSIDE the transaction so its unit-part rows commit
        atomically with the master row; splitting it across stages
        is explicitly forbidden.

        The successful master insert establishes publication ownership. Only
        that attempt renames its completed analyzer into the shared cache,
        before the surrounding transaction commits. A duplicate insert never
        publishes. Every exit closes this attempt's staged analyzer. Removing
        a failed attempt's staged NWB is ``StagedOutputCleanupMixin``'s job
        during ``populate()``; a direct call leaves that to its caller.
        Previously published caches are never deleted by a losing attempt.
        The publication lock covers the framework's commit via
        ``AnalyzerPublicationMixin``; a direct call holds it through its own
        transaction. Direct calls inside a caller-owned transaction are refused
        because this method cannot keep that lock until the later commit.

        Parameters
        ----------
        key : dict
            Primary key of the sorting being populated.
        sorting_obj : spikeinterface.BaseSorting
            The computed sorting (unit ids + spike trains).
        analysis_file_name : str
            Staged AnalysisNwbfile registered here.
        units_object_id : str
            NWB object id of the units table inside that file.
        nwb_file_name : str
            Source NWB file backing the recording selection.
        display_waveform_params_name : str
            The resolved DISPLAY recipe, persisted on the master row so every
            later rebuild reads it back deterministically.
        effective_random_seed : int
            Seed the sort actually used; secondary provenance, not identity.
        spikeinterface_version : str
            ``spikeinterface.__version__`` at sort time.
        sorter_version : str or None
            Sorter package distribution version, ``None`` for in-process sorters.
        staged_analyzer : StagedAnalyzer
            Private build to publish after insertion and close on every exit.

        Returns
        -------
        None
        """
        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_publication_transaction,
        )

        try:
            with analyzer_publication_transaction(self, key["sorting_id"]):
                self._insert_sorting_rows_transaction(
                    key=key,
                    sorting_obj=sorting_obj,
                    analysis_file_name=analysis_file_name,
                    units_object_id=units_object_id,
                    nwb_file_name=nwb_file_name,
                    display_waveform_params_name=display_waveform_params_name,
                    effective_random_seed=effective_random_seed,
                    spikeinterface_version=spikeinterface_version,
                    sorter_version=sorter_version,
                    unit_rows=unit_rows,
                )
                # The successful INSERT owns this sorting_id until commit or
                # rollback. A duplicate worker cannot reach publication. Only
                # a rename happens here; waveform extraction was in compute.
                staged_analyzer.publish()
        finally:
            staged_analyzer.close()

    def _stage_sorting_artifact(
        self,
        *,
        sorting,
        recording,
        nwb_file_name,
        obs_intervals,
        unit_metadata=None,
        source_provenance,
    ):
        """Stage the units NWB; return ``(analysis_file_name, units_object_id)``.

        ``_write_units_nwb`` self-cleans its staged file on a write failure;
        ``make_compute`` discards the private analyzer on that failure.
        ``unit_metadata`` /
        ``source_provenance`` are the compute-once per-unit columns + source
        header embedded in the NWB.
        """
        return self._write_units_nwb(
            sorting=sorting,
            recording=recording,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            unit_metadata=unit_metadata,
            source_provenance=source_provenance,
        )

    def _insert_sorting_rows_transaction(
        self,
        *,
        key,
        sorting_obj,
        analysis_file_name,
        units_object_id,
        nwb_file_name,
        display_waveform_params_name,
        effective_random_seed,
        spikeinterface_version,
        sorter_version,
        unit_rows,
    ):
        """Register the AnalysisNwbfile + master + Unit rows atomically.

        Runs the ``_safe_context()`` block: registers the staged
        AnalysisNwbfile, inserts the Sorting master, and populates the Unit
        part rows (built once in ``make_compute``) INSIDE the transaction so
        they commit atomically with the master (splitting them across stages is
        forbidden). ``_safe_context()`` is a no-op when the framework
        transaction is already active (the tri-part dispatch path); it is kept
        so an out-of-populate caller still gets atomic registration.
        """
        import datetime as dt

        with self._safe_context():
            AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
            self.insert1(
                {
                    **key,
                    "analysis_file_name": analysis_file_name,
                    "object_id": units_object_id,
                    "n_units": len(sorting_obj.unit_ids),
                    "time_of_sort": dt.datetime.now(),
                    "display_waveform_params_name": (
                        display_waveform_params_name
                    ),
                    "effective_random_seed": effective_random_seed,
                    "spikeinterface_version": spikeinterface_version,
                    "sorter_version": sorter_version,
                }
            )
            self._populate_unit_part(unit_rows)

    # ---- Accessors -------------------------------------------------------

    def get_sorting(
        self, key: dict, as_dataframe: bool = False
    ) -> "si.BaseSorting | pd.DataFrame":
        """Return the SpikeInterface BaseSorting backed by the units NWB.

        Spike times are persisted by ``_write_units_nwb`` in two forms:
        absolute ``spike_times`` for NWB interoperability, and Spyglass's
        ``spike_sample_index`` sidecar for efficient frame-based readback.
        Readback uses the stored sample frames directly. Populated v2 Units
        tables require sample frames and observation intervals.

        Returns a ``NumpySorting`` (segment frame indices, ``t_start=0``),
        so ``get_unit_spike_train(uid)`` yields the original recording
        frames and a downstream ``extract_waveforms`` / analyzer build
        aligns to the right samples.

        ``as_dataframe=True`` returns a pandas DataFrame whose
        **index is the unit_id** and which carries a ``spike_times``
        column (the stored ABSOLUTE seconds, read straight from the
        units NWB). The ``CurationV2.get_sorting``
        accessor uses the same flag + index and adds a
        ``curation_label`` column joined from ``CurationV2.UnitLabel``.

        A zero-unit sort returns an empty sorting (with a warning);
        ``get_analyzer`` raises ``ZeroUnitAnalyzerError`` instead.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        as_dataframe : bool, optional
            If ``True``, return a per-unit DataFrame instead of an SI
            sorting object. Defaults to ``False``.

        Returns
        -------
        si.BaseSorting or pd.DataFrame
            The sorting (a ``NumpySorting``) when ``as_dataframe`` is
            ``False``; otherwise a DataFrame indexed by ``unit_id`` with
            a ``spike_times`` column.
        """
        import spikeinterface as si

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        # Resolve the source sampling rate. A motion-corrected recording keeps
        # its source's frames and rate, so the same row serves a corrected sort.
        source = SortingSelection.resolve_source(key)
        if source.kind == "recording":
            rec_row = (
                Recording & {"recording_id": source.key["recording_id"]}
            ).fetch1()
        else:  # concatenated_recording
            rec_row = (ConcatenatedRecording & source.key).fetch1()
        fs = float(rec_row["sampling_frequency"])

        if int(row["n_units"]) == 0:
            # Zero units is a valid result; return an empty sorting /
            # frame rather than crashing. (``get_analyzer`` on a
            # zero-unit sort raises instead, since an analyzer is not
            # representable over zero units.)
            logger.warning(
                f"Sorting.get_sorting: sorting_id={row['sorting_id']!r} "
                "has zero units; returning an empty sorting."
            )
            if as_dataframe:
                return empty_spike_times_dataframe()
            return si.NumpySorting.from_unit_dict({}, sampling_frequency=fs)

        if as_dataframe:
            abs_times = read_units_abs_spike_times(abs_path)
            return abs_spike_times_dataframe(abs_times)
        return sorting_from_units_nwb(abs_path, fs)

    def get_statistics_spans(self, key: dict) -> list[tuple[int, int]]:
        """Return the statistics spans persisted with a sort.

        The artifact-free half-open frame ranges of the sorted recording that
        never cross a selection join, a member join, or a member-internal
        timestamp gap. They were computed once when the sort ran; every
        analyzer build for the sort (sort time, self-heal rebuild, curation
        evaluation, merged-curation analyzers, the recompute audit) estimates
        noise levels and whitening from samples inside them, so all builds
        agree.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.

        Returns
        -------
        list[tuple[int, int]]
            Sorted half-open frame spans.

        Raises
        ------
        RuntimeError
            If the sort's units NWB has no persisted spans; repopulate the
            sort.
        """
        sorting_id, analysis_file_name = (self & key).fetch1(
            "sorting_id", "analysis_file_name"
        )
        return read_sorting_statistics_spans(
            AnalysisNwbfile.get_abs_path(analysis_file_name),
            sorting_id=sorting_id,
        )

    def get_analyzer(
        self,
        key: dict,
        waveform_params_name: str | None = None,
        *,
        rebuild: bool = True,
        load_extensions: bool = True,
    ) -> "si.SortingAnalyzer":
        """Return the SortingAnalyzer; rebuild on missing or invalid folder.

        Recompute is in-place; the DataJoint row is not deleted on a
        missing analyzer folder.

        A zero-unit sort has no analyzer (SI cannot build one over zero
        units), so this raises ``ZeroUnitAnalyzerError``, whereas
        ``get_sorting`` returns the valid empty sorting with a warning. To
        branch without catching, precheck
        ``(Sorting & key).fetch1("n_units") > 0``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        waveform_params_name : str, optional
            The analyzer recipe to load. ``None`` (default) loads the sort's
            stored DISPLAY recipe (``display_waveform_params_name``); a caller
            needing the whitened metric recipe (e.g. the PC/NN cluster-
            separation metrics in ``CurationEvaluation``) passes it explicitly.
            A missing or invalid folder is rebuilt for whichever recipe is
            requested (unless ``rebuild=False``).
        rebuild : bool, optional
            If ``True`` (default), a missing or invalid analyzer folder is
            rebuilt in place (the self-healing cache). If ``False``, a missing
            folder raises ``AnalyzerFolderMissingError`` and an unloadable folder
            raises ``AnalyzerFolderInvalidError`` instead -- the recompute audit
            uses this to OBSERVE a missing/reclaimed/corrupt analyzer rather
            than silently rebuild-then-hash it.
        load_extensions : bool, optional
            Load all saved extensions by default so SI save/select/merge methods
            retain them. Internal read-only inspection may pass False to load
            arrays on demand. Waveforms remain memory-mapped in either case.

        Returns
        -------
        si.SortingAnalyzer
            The loaded ``SortingAnalyzer`` for the sort, rebuilt in
            place if its folder was missing or invalid (when ``rebuild=True``).

        Raises
        ------
        ZeroUnitAnalyzerError
            If the sort has zero units.
        AnalyzerFolderMissingError
            If ``rebuild=False`` and the analyzer folder is absent on disk.
        AnalyzerFolderInvalidError
            If ``rebuild=False`` and the analyzer folder exists but cannot be
            loaded.
        """
        return load_or_rebuild_analyzer(
            self,
            key,
            waveform_params_name=waveform_params_name,
            rebuild=rebuild,
            load_extensions=load_extensions,
        )

    def add_extensions(
        self,
        key: dict,
        extensions: list[str],
        *,
        waveform_params_name: str | None = None,
        extension_params: dict[str, dict] | None = None,
        **kwargs,
    ) -> list[str]:
        """Add SortingAnalyzer extensions in place; return the ones computed.

        Convenience for callers (and ``CurationEvaluation``) that need
        extensions beyond the sort-time base set. Only extensions NOT already
        present are computed, so the call is idempotent and never recomputes
        ``waveforms`` / ``templates`` (recomputing a parent cascade-deletes its
        derived extensions and rewrites the committed ``peak_amplitude_uv``). A
        different waveform window is a sort-time (``SorterParameters``)
        decision, not an analyzer-curation recompute.

        Job kwargs are resolved from this sort's ``SorterParameters`` row
        (``_resolved_job_kwargs``: SpikeInterface globals, then ``dj.config``,
        then the row blob); explicit ``kwargs`` win on conflict. The computed extensions persist to the on-disk analyzer
        folder (SI's ``binary_folder`` format saves them automatically).

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        extensions : list of str
            SortingAnalyzer extension names to add.
        waveform_params_name : str, optional
            Exact analyzer recipe to mutate. ``None`` uses the sort's stored
            display recipe.
        extension_params : dict, optional
            Per-extension parameter dictionaries forwarded to
            :func:`ensure_extensions`.
        **kwargs
            Job kwargs that override the resolved per-row defaults.

        Returns
        -------
        list of str
            The extensions actually computed (already-present ones are
            skipped); empty when every requested extension already exists.
        """
        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
        )
        from spyglass.spikesorting.v2._sorting.analyzer import ensure_extensions
        from spyglass.spikesorting.v2._core.job_config import (
            _resolved_job_kwargs,
        )

        sorting_id = (self & key).fetch1("sorting_id")
        sorter_job_kwargs = (
            SorterParameters & (SortingSelection & key)
        ).fetch1("job_kwargs")
        resolved = _resolved_job_kwargs(sorter_job_kwargs)
        resolved.update(kwargs)
        # Hold the per-sort lock across BOTH the load and the in-place extension
        # compute: ``ensure_extensions`` persists into the canonical analyzer
        # folder, so loading under the lock and releasing it before the compute
        # would leave the mutation unguarded. The lock is reentrant, so the
        # nested ``get_analyzer`` load does not self-deadlock.
        with analyzer_cache_lock(sorting_id):
            analyzer = self.get_analyzer(
                key, waveform_params_name=waveform_params_name
            )
            return ensure_extensions(
                analyzer,
                extensions,
                job_kwargs=resolved,
                extension_params=extension_params,
            )

    # ---- visualization / export delegates (see v2.visualization facade) ---

    def root_curation(self, key):
        """Return the generation-pinned ``CurationRef`` of this sort's root.

        The raw sort's units are addressed through the root curation
        (``parent_curation_id=-1``): every unit-level plot / export in
        ``v2.visualization`` takes a curation, so the delegates below resolve
        the root and hand it over. Raises ``ValueError`` if no root curation
        exists yet (``CurationV2.insert_curation(sorting_key)`` creates it).
        """
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.curation_api import CurationRef

        sorting_id = (self & key).fetch1("sorting_id")
        root = CurationV2 & {"sorting_id": sorting_id, "parent_curation_id": -1}
        if len(root) != 1:
            raise ValueError(
                f"Sorting {sorting_id} has no root curation yet; create it "
                "with CurationV2.insert_curation(sorting_key) (run_v2_pipeline "
                "does this) before plotting or exporting its units."
            )
        return CurationRef.from_key(root.fetch1("KEY"))

    def plot_summary(
        self, key, *, compute_missing=False, backend=None, **kwargs
    ):
        """Delegate to ``visualization.plot_sorting_summary`` for this sort's
        root curation (the raw units).

        A local-discoverability one-liner; the display-analyzer routing and
        extension policy live in the ``v2.visualization`` facade, which the
        notebook/docs teach as the primary surface. ``backend`` is required (SI's
        ``SortingSummaryWidget`` has no matplotlib backend); see the facade.
        """
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_sorting_summary(
            self.root_curation(key),
            compute_missing=compute_missing,
            backend=backend,
            **kwargs,
        )

    def plot_unit_summary(
        self,
        key,
        unit_id,
        *,
        compute_missing=False,
        backend="matplotlib",
        **kwargs,
    ):
        """Delegate to ``visualization.plot_unit_summary`` (root curation)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_unit_summary(
            self.root_curation(key),
            unit_id,
            compute_missing=compute_missing,
            backend=backend,
            **kwargs,
        )

    def plot_waveforms(
        self, key, unit_ids=None, *, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_waveforms`` (root curation)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_waveforms(
            self.root_curation(key),
            unit_ids=unit_ids,
            backend=backend,
            **kwargs,
        )

    def plot_spikes_on_traces(
        self, key, *, compute_missing=False, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_spikes_on_traces`` (root curation)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_spikes_on_traces(
            self.root_curation(key),
            compute_missing=compute_missing,
            backend=backend,
            **kwargs,
        )

    def plot_unit_locations(
        self, key, *, compute_missing=False, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_unit_locations`` (root curation)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_unit_locations(
            self.root_curation(key),
            compute_missing=compute_missing,
            backend=backend,
            **kwargs,
        )

    def export_si_report(
        self, key, output_folder, *, compute_missing=False, **kwargs
    ):
        """Delegate to ``visualization.export_si_report`` (root curation)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.export_si_report(
            self.root_curation(key),
            output_folder,
            compute_missing=compute_missing,
            **kwargs,
        )

    def export_to_phy(self, key, output_folder, **kwargs):
        """Delegate to ``visualization.export_to_phy`` (root curation)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.export_to_phy(
            self.root_curation(key), output_folder, **kwargs
        )

    def _rebuild_analyzer_folder(self, key) -> None:
        """Rebuild the analyzer folder for an existing Sorting row.

        Reloads the canonical sorting from the units NWB so the
        rebuilt analyzer is bit-equivalent to the one Sorting.make
        wrote -- not a fresh, possibly nondeterministic, sort.

        ``key`` must carry a literal ``sorting_id`` (it is used directly
        for ``analyzer_path`` and the ``SortingSelection`` fetches). The
        public ``get_analyzer`` resolves the canonical id from a general
        restriction and hands this private helper a normalized
        ``{"sorting_id": ...}``; callers should do the same.
        """
        return rebuild_analyzer_folder(self, key)

    def delete(self, *args, safemode=None, **kwargs):
        """Cascade-delete + analyzer-cache cleanup on disk.

        The analyzer cache folder is regeneratable scratch resolved from
        ``sorting_id`` (not a DataJoint-tracked column), so a plain
        ``.delete()`` would leave the 5-50 GB folder on disk per row.
        Mirrors the artifact-detection tables' ``delete`` IntervalList cleanup
        pattern: snapshot every ``sorting_id`` BEFORE the cascade delete (it
        can no longer be fetched once the row is gone), call ``super().delete()``,
        then ``remove_analyzer_cache`` each (which resolves the path from
        ``sorting_id``, no-ops a missing folder, and surfaces a permission
        error loudly rather than swallowing it).

        Two cleanup paths already cover other points in the analyzer-cache
        lifecycle: ``_run_sorter`` cleans the
        sorter scratch ``TemporaryDirectory`` on successful sort,
        and the make_compute / make_insert except blocks clean the
        folder on populate failure. This override closes the third
        lifecycle event: row deletion.

        A leading positional restriction is accepted as a compatibility guard
        for the easy-to-mistype ``Sorting().delete(restriction)`` form. DataJoint's
        own ``delete`` does not take restrictions positionally, and Spyglass's
        cautious-delete layer would otherwise read that dict as a truthy
        ``force_permission`` (``cautious_delete(self, force_permission=False,
        ...)``) and cascade-delete EVERY row of the unrestricted instance --
        destroying each row's 5-50 GB analyzer folder. Mirrors
        the artifact-detection tables' ``delete`` guard.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``super().delete``.
        safemode : bool or None, optional
            Forwarded to ``super().delete`` to control the confirmation
            prompt. ``None`` (default) omits the argument so DataJoint's
            own default applies.
        **kwargs
            Keyword arguments forwarded to ``super().delete``.
        """
        restriction_args, args = split_leading_restrictions(args)
        if restriction_args:
            target = self
            for restriction in restriction_args:
                target = target & restriction
            return target.delete(*args, safemode=safemode, **kwargs)

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
            remove_analyzer_cache,
        )

        # Snapshot the PKs BEFORE the cascade -- after deletion the row is
        # gone, but the cache path is a pure function of sorting_id, so this
        # delete is one of the sites that resolves it from sorting_id (vs the
        # populate/rebuild cleanups, which rmtree the EXACT transient folder
        # they built to avoid a recompute).
        rows = self.fetch("KEY", as_dict=True)
        if safemode is None:
            super().delete(*args, **kwargs)
        else:
            super().delete(*args, safemode=safemode, **kwargs)
        # Only remove a folder whose DB row was ACTUALLY deleted. A cancelled
        # confirmation prompt (user answers "no") or an empty restriction
        # leaves the rows in place and returns normally -- removing their
        # 5-50 GB analyzer scratch then would destroy data for a row the user
        # chose to keep. ``remove_analyzer_cache`` no-ops a missing folder and
        # propagates a real removal error (``ignore_errors=False``).
        for row in rows:
            if not (Sorting & row):
                # Remove the regeneratable cache UNDER the per-sort lock so a
                # concurrent reader / rebuild never races the rmtree.
                with analyzer_cache_lock(row["sorting_id"]):
                    remove_analyzer_cache(row["sorting_id"], missing_ok=True)

    @classmethod
    def find_orphaned_analyzer_folders(
        cls, *, sorting_id=None, dry_run: bool = True
    ) -> dict:
        """Audit 5-50 GB analyzer-folder disk leaks; never auto-delete DB rows.

        Each populated sort writes a 5-50 GB ``analyzer_folder`` of
        regeneratable scratch outside the DataJoint-tracked store. The
        ``Sorting.delete`` override cleans it up on row delete. The common leak
        path is a delete that starts at ``Recording``, ``RecordingSelection``,
        or ``SortGroupV2``: the cascade reaches ``Sorting`` through DataJoint
        ``FreeTable`` objects and never dispatches this override. Raw SQL or a
        scripted ``dj.Table.connection.query`` also bypasses it. This periodic
        audit mirrors ``prune_orphaned_selections`` and reports four classes:

        - **DB-side orphan**: a ``Sorting`` row whose computed analyzer cache
          folder (``analyzer_path(sorting_id, display_waveform_params_name)``)
          no longer exists on disk (the
          regeneratable scratch was removed out of band). Reported only --
          deleting the *row* is a destructive DB operation left to the user;
          this method NEVER auto-deletes a row.
        - **Reclaimed**: a missing analyzer folder with a
          ``SortingAnalyzerRecompute.deleted=1`` audit trail. This is expected
          storage reclamation, not an unexpected DB-side orphan.
        - **Disk-side orphan**: an on-disk raw- or curation-kind folder under
          the analyzer root that no live sort/evaluation/curation generation
          references. Safe to delete after inspection.
        - **Staging**: a build/trash folder whose ownership locks are free.
          Active attempts are skipped, including attempts on another host using
          the supported shared-filesystem locking contract.

        **Zero-unit carve-out.** Rows with ``n_units == 0`` are NOT DB-side
        orphans: ``_build_analyzer`` short-circuits before writing a folder and
        ``get_analyzer`` raises ``ZeroUnitAnalyzerError`` before reading the
        path, so an absent folder is expected. The cache path is COMPUTED from
        ``sorting_id`` (not a stored column), so the carve-out is
        keyed on ``(Sorting & {"n_units": 0})``, NOT on any column value.

        Parameters
        ----------
        sorting_id : UUID or str, optional
            Restrict database queries and cache candidates to this sorting.
            Omit to audit all sortings, including folders for deleted sorts.
        dry_run : bool, optional
            When True (default) only report. When False,
            after interactive confirmation (``dj.utils.user_choice``), delete
            disk-side orphans and abandoned staging; DB-side rows are never
            deleted. Staging ownership is rechecked before removal.

        Returns
        -------
        dict
            ``{"db_side": [{"sorting_id", "computed_analyzer_path"}, ...],
            "disk_side": [folder_path_str, ...],
            "reclaimed": [{"sorting_id", "computed_analyzer_path"}, ...],
            "staging": [folder_path_str, ...]}``.
            ``computed_analyzer_path`` is resolved from ``sorting_id`` (there is
            no stored ``analyzer_folder`` column).
        """

        return _analyzer_cache.find_orphaned_analyzer_folders(
            cls, sorting_id=sorting_id, dry_run=dry_run
        )

    def get_unit_brain_regions(
        self, key: dict, *, allow_anchor_member: bool = False
    ) -> "pd.DataFrame":
        """Per-unit brain regions via Sorting.Unit * Electrode * BrainRegion.

        Single-session sorts return ``region_resolution='single_session'``.
        Concat sorts raise ``ConcatBrainRegionAmbiguousError`` unless
        ``allow_anchor_member=True``; the anchor-member output is
        labeled ``region_resolution='anchor_member'``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        allow_anchor_member : bool, optional
            If ``True``, return anchor-member regions for concat-backed
            sortings instead of raising. Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            One row per (unit, electrode) with the brain-region columns
            and a ``region_resolution`` label.
        """
        from spyglass.spikesorting.v2.exceptions import (
            ConcatBrainRegionAmbiguousError,
        )

        source = SortingSelection.resolve_source(key)
        if source.kind == "concatenated_recording":
            if not allow_anchor_member:
                raise ConcatBrainRegionAmbiguousError(
                    f"Sorting.get_unit_brain_regions: sorting_id "
                    f"{key['sorting_id']} is concat-backed; the unit peak "
                    "channel maps to multiple Electrode rows (one per "
                    "SessionGroup.Member). Pass allow_anchor_member=True "
                    "to return anchor-member regions, or match a curation of "
                    "this sort with UnitMatch and use "
                    "TrackedUnit.get_unit_brain_regions, which resolves each "
                    "unit's region in every member recording from that "
                    "member's own session."
                )
            resolution = "anchor_member"
        else:
            resolution = "single_session"
        return unit_brain_region_df(self.Unit & key, resolution)

    # ---- Implementation helpers -----------------------------------------

    @staticmethod
    def _apply_artifact_mask(
        recording, valid_times, *, artifact_detection_id=None, recording_id=None
    ):
        """Zero out the complement of ``valid_times`` on the recording.

        Thin delegator to :func:`._sorting_artifact_mask.apply_artifact_mask`;
        kept as a ``Sorting`` staticmethod because the v2 tests call
        ``Sorting._apply_artifact_mask`` directly. The analyzer reconstruction
        paths mask through
        :func:`._source_resolution.load_effective_recording`.
        ``make_compute`` masks through
        :func:`._sorting_artifact_mask.sorting_statistics_spans`
        (``artifact_frame_ranges`` / ``silence_frame_ranges``) so it keeps the
        excluded ranges for the statistics spans. The complement-walk
        masking + input validation (empty/shape/order checks, the
        disjoint-gap boundary carve-out) live in the service module.
        """
        return apply_artifact_mask(
            recording,
            valid_times,
            artifact_detection_id=artifact_detection_id,
            recording_id=recording_id,
        )

    @staticmethod
    def _run_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        """Dispatch sort execution; clusterless_thresholder vs SI sorters.

        Clusterless is a Spyglass-specific peak-detection path with no
        SI scratch directory, external whitening, or container backend, so
        ``execution_params`` does not apply to it. SI sorters get per-sort
        scratch, external whitening, and the tracked container-execution
        backend. The two paths share nothing but the signature; dispatch routes
        each to its own helper. ``statistics_spans`` (artifact-free frame
        spans of ``recording``; ``None`` means the whole recording) reach both:
        the clusterless MAD and the external whitening estimate from them.
        """
        if sorter == "clusterless_thresholder":
            return Sorting._run_clusterless_thresholder(
                sorter_params=sorter_params,
                recording=recording,
                job_kwargs=job_kwargs,
                statistics_spans=statistics_spans,
            )
        return Sorting._run_si_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=sorting_id,
            job_kwargs=job_kwargs,
            execution_params=execution_params,
            statistics_spans=statistics_spans,
        )

    @staticmethod
    def _run_clusterless_thresholder(
        sorter_params,
        recording,
        job_kwargs,
        statistics_spans=None,
    ):
        """Run Spyglass's clusterless-thresholder peak-detection path.

        Thin delegator to
        :func:`._sorting_dispatch.run_clusterless_thresholder`; kept as a
        ``Sorting`` staticmethod because ``_run_sorter`` dispatches to
        ``Sorting._run_clusterless_thresholder`` and the v2 tests call it
        directly. The detect_peaks pipeline -- noise_levels/threshold_unit
        precedence and validation, the uV ``scale_to_uV`` carve-out, and
        the deterministic seeding -- lives in the service module.
        """
        return run_clusterless_thresholder(
            sorter_params=sorter_params,
            recording=recording,
            job_kwargs=job_kwargs,
            statistics_spans=statistics_spans,
        )

    @staticmethod
    def _run_si_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        job_kwargs,
        execution_params=None,
        statistics_spans=None,
    ):
        """Run an SI registered sorter under a managed scratch dir.

        Thin delegator to :func:`._sorting_dispatch.run_si_sorter`; kept as
        a ``Sorting`` staticmethod because ``_run_sorter`` dispatches to
        ``Sorting._run_si_sorter`` and the v2 tests call it directly. The
        managed ``TemporaryDirectory`` scratch, external float64 whitening,
        scoped ``np.Inf`` patch, container-execution backend (local vs
        Docker/Singularity + the MATLAB-sorter container policy), and the
        global-job-kwargs save/restore live in the service module.
        ``execution_params`` defaults to ``None`` (resolved to local) so direct
        test callers and the clusterless path stay unchanged.
        """
        return run_si_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=sorting_id,
            job_kwargs=job_kwargs,
            execution_params=execution_params,
            statistics_spans=statistics_spans,
        )

    @staticmethod
    def _remove_excess_spikes(sorting, recording):
        """Drop spikes whose sample index is outside the recording window.

        Thin delegator to :func:`._sorting_dispatch.remove_excess_spikes`;
        kept as a ``Sorting`` staticmethod because ``make_compute`` calls
        ``self._remove_excess_spikes(...)``.
        """
        return remove_excess_spikes(sorting, recording)

    @staticmethod
    def _build_analyzer(
        sorting,
        recording,
        key,
        *,
        sorter_row,
        job_kwargs,
        analyzer_folder=None,
        waveform_params=None,
        statistics_spans=None,
    ):
        """Build the binary-folder SortingAnalyzer + base extensions.

        Thin delegator to :func:`._sorting_analyzer.build_analyzer`; kept as
        a ``Sorting`` staticmethod because ``make_compute`` /
        ``_rebuild_analyzer_folder`` call ``self._build_analyzer(...)`` and
        the v2 tests call it directly. The analyzer creation, seeded
        extension compute, zero-unit short-circuit, and partial-folder
        cleanup live in the service module. All database inputs and execution
        kwargs must be resolved before this call. ``waveform_params`` is the
        resolved analyzer-waveform params blob (window / subsample); ``None``
        is invalid and the service raises ``ValueError``.
        """
        return build_analyzer(
            sorting,
            recording,
            key,
            sorter_row=sorter_row,
            job_kwargs=job_kwargs,
            analyzer_folder=analyzer_folder,
            waveform_params=waveform_params,
            statistics_spans=statistics_spans,
        )

    @staticmethod
    def _write_units_nwb(
        sorting,
        recording,
        nwb_file_name,
        obs_intervals=None,
        *,
        unit_metadata=None,
        source_provenance,
    ):
        """Write a fresh AnalysisNwbfile containing only the v2 Units table.

        Thin delegator to :func:`._units_nwb.write_sorting_units_nwb`;
        kept as a ``Sorting`` staticmethod because ``make_insert`` calls
        ``self._write_units_nwb(...)`` and the v2 tests both monkeypatch
        ``Sorting._write_units_nwb`` (to check analyzer cleanup when the units
        write fails) and call it directly (the zero-unit guard test). The NWB staging
        IO -- absolute-timeline spike times, the ``obs_intervals`` +
        ``curation_label`` columns, the per-unit metadata + source-provenance
        scratch, and the zero-unit empty-Units guard -- lives in the service
        module.
        """
        return write_sorting_units_nwb(
            sorting=sorting,
            recording=recording,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            unit_metadata=unit_metadata,
            source_provenance=source_provenance,
        )

    @staticmethod
    def _populate_unit_part(unit_rows):
        """Insert the pre-built ``Sorting.Unit`` rows.

        The rows are built ONCE in ``make_compute``
        (:func:`._sorting_units.build_unit_rows_from_analyzer`) and threaded through, so this
        is a pure insert with no analyzer load or DB read. An empty list (a
        zero-unit sort) is a no-op.
        """
        Sorting.Unit.insert(unit_rows)
