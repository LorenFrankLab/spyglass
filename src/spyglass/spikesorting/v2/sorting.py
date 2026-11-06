"""Spike sorting and per-unit brain-region metadata.

Tables:
    SorterParameters: per-sorter Pydantic-validated params.
    SortingSelection: sorting request for one source.
        .RecordingSource: single-session source.
        .ConcatenatedRecordingSource: same-day concat source.
        .ArtifactDetectionSource: optional artifact mask (single source).
        .MotionCorrectionSource: optional motion-corrected recording, sorted
            in place of the source.
    Sorting (+ Unit): sorted units NWB and SortingAnalyzer folder.

A concat sort anchors the per-unit Electrode FK and the analysis-NWB parent to
the first frozen ``ConcatenatedRecordingSelection.MemberSnapshot`` member.
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
    load_or_rebuild_analyzer,
)
from spyglass.spikesorting.v2._sorting.artifact_mask import (
    sorting_statistics_spans,
)
from spyglass.spikesorting.v2._sorting.dispatch import (
    remove_excess_spikes,
    run_clusterless_thresholder,
    run_si_sorter,
    sort_runtime_versions,
)
from spyglass.spikesorting.v2._storage import (
    analyzer_cache as _analyzer_cache,
    units_nwb as _units_nwb,
)
from spyglass.spikesorting.v2._sorting import (
    analyzer as _sorting_analyzer,
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
)
from spyglass.spikesorting.v2.artifact_output import (
    ArtifactDetectionOutput,
)
from spyglass.spikesorting.v2.motion import (
    MotionCorrectedRecording,
    MotionCorrectedRecordingSelection,
)
from spyglass.spikesorting.v2.recording import (  # noqa: F401
    PreprocessingParameters,
    Recording,
)
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,  # noqa: F401
)
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _insert_parameter_rows,
    optional_part_value,
    restrict_by_leading_args,
    single_source_part,
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
        The sort's recording, or the first frozen member's recording for a
        concat source (the anchor).
    sel_row : dict
        The ``SortingSelection`` row, with ``artifact_detection_id`` added.
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
        The display ``AnalyzerWaveformParameters`` recipe resolved from the
        source preprocessing recipe.
    display_waveform_params : dict
        That recipe's params blob.
    execution_params : dict
        The validated ``SorterParameters.execution_params`` blob.
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
    # Lets make_compute build Sorting.Unit rows without DB access.
    sort_group_id: int
    electrode_by_id: dict
    region_by_electrode: dict
    # (n, 2) int64 concat frame ranges; None for a single recording.
    concat_statistics_spans: np.ndarray | None
    # Sorter input and its path (rebuilt if missing).
    traces: EffectiveTraces
    traces_abs_path: str
    # Motion-corrected sorts only: NWB provenance and the source frame count
    # the corrected traces must keep.
    motion_correction_provenance: dict | None
    source_n_samples: int | None


class SortingComputed(NamedTuple):
    """Compute -> insert carrier for :meth:`Sorting.make_compute`.

    Field order must match ``make_insert``'s parameters: the tuple is
    unpacked positionally into ``make_insert(key, *result)``.

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
        The random seed the sort used. Provenance, not identity.
    spikeinterface_version : str
        ``spikeinterface.__version__`` of the host or container that ran the
        sort.
    sorter_version : str or None
        Sorter version; ``None`` for in-process SI-internal sorters.
    unit_rows : list of dict
        The ``Sorting.Unit`` rows, shared with the NWB write. Empty for a
        zero-unit sort.
    staged_analyzer : StagedAnalyzer
        Private analyzer and its lock until ``make_insert`` publishes or
        discards it.
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

    ``insert_default`` ships rows for MS4, MS5, KS4, SC2, TDC2, and
    ``clusterless_thresholder``. Other installed SI sorters validate against
    a generic ``extra="allow"`` schema.
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
    # execution_params (local/Docker/Singularity) lives here because the
    # backend can change the output; changing it needs a new name.

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

    _DEFAULT_CONTENTS: tuple = sorter_default_contents()

    # Spyglass-internal sorters, never gated on ``installed_sorters()``.
    _NON_SI_SORTERS: frozenset[str] = frozenset({"clusterless_thresholder"})

    @classmethod
    def _is_seed_dependent_sort(cls, sorter: str, sorter_params: dict) -> bool:
        """Whether the sort's output depends on ``random_seed``.

        Such a sort rejects a seed set only in SI or ``dj.config``, since that
        seed is not part of ``sorting_id``. SI sorters always depend on the
        seed. ``clusterless_thresholder`` does only with
        ``threshold_unit='mad'`` and no ``noise_levels`` (random noise chunks).
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

        Covers MS4, MS5, KS4, SC2, TDC2, and ``clusterless_thresholder``.
        Skips (logged at INFO) rows whose sorter is not in
        ``installed_sorters()``. This does not prove a sort will run;
        ``preflight_v2_pipeline`` checks the runtime.
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

        Insertable: a Spyglass-internal sorter, an installed SI sorter, or a
        Docker/Singularity backend (its runtime is in the image).
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

        For v1 workflows that name e.g. ``('kilosort2_5','default')``. Each
        row holds the SI wrapper's defaults (without global job kwargs); a row
        that fails validation is skipped with a warning. Also skips:

        - sorters not in ``installed_sorters()`` (logged at INFO);
        - MATLAB sorters, which need a container ``execution_params``;
        - curated sorters, whose typed schemas differ from SI's defaults.

        Not called by ``initialize_v2_defaults``. Idempotent.

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

    A ``per_row_hook`` for ``validate_lookup_rows``.
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

    - ``params`` is validated by :class:`AnalyzerWaveformParamsSchema`.
    - ``insert_default`` ships region rows: hippocampus 0.5/0.5 ms, cortex
      1.0/2.0 ms, both 20000 spikes.
    - A sort's display recipe follows from its preprocessing recipe, is
      stored on ``Sorting.display_waveform_params_name``, and is not part of
      ``sorting_id``.
    - The name must match ``^[A-Za-z0-9_]+$``: it is part of the analyzer
      folder path.
    """

    definition = f"""
    waveform_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={ANALYZER_WAVEFORM_SCHEMA_VERSION}: int
    """

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
class PreprocessingWaveformParameters(SpyglassMixin, dj.Manual):
    """Analyzer windows for a preprocessing row derived from a region recipe.

    ``clone_pipeline_preset`` records the base recipe's ``(display,
    metric)`` pair here for a renamed preprocessing row, which would
    otherwise get the cortex windows.
    """

    definition = """
    -> PreprocessingParameters
    ---
    -> AnalyzerWaveformParameters.proj(display_waveform_params_name="waveform_params_name")
    -> AnalyzerWaveformParameters.proj(metric_waveform_params_name="waveform_params_name")
    """


@schema
class SortingSelection(SelectionMasterInsertGuard, SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    - Exactly one of ``RecordingSource`` or ``ConcatenatedRecordingSource``.
    - Optional ``ArtifactDetectionSource``. A part row rather than a nullable
      FK, so "no detection" does not act as a wildcard in restrictions.
    - Optional ``MotionCorrectionSource``: the sort reads that
      ``MotionCorrectedRecording``, made from the same source and mask,
      instead of the source traces. Its id is part of ``sorting_id``.
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

        Read through :meth:`SortingSelection.resolve_artifact_detection`.
        """

        definition = """
        -> master
        ---
        -> ArtifactDetectionOutput.proj(artifact_detection_merge_id='merge_id')
        """

    class MotionCorrectionSource(SpyglassMixinPart):
        """Optional motion-corrected recording a sort reads.

        Read through :meth:`SortingSelection.resolve_motion_correction`.
        """

        definition = """
        -> master
        ---
        -> MotionCorrectedRecording
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert master + exactly one source part; return PK-only dict.

        Selections with and without an artifact detection are distinct
        rows. Re-inserting the same request returns the existing key.

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
            If ``key`` has an unknown field, zero or both source keys, a
            concat source with an ``artifact_detection_id`` (concat masks
            live on ``ConcatenatedRecordingSelection``), or a motion-corrected
            recording that is unpopulated, made from another source or mask,
            or paired with a sorter that corrects motion itself.
        DuplicateSelectionError
            If a matching master has a non-deterministic ``sorting_id`` (a
            raw ``insert`` bypass).
        SchemaBypassError
            If a deterministic master exists but its source parts are missing
            or mismatched.
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

        See :func:`._sorting.selection.find_existing_pk`. Kept on the class
        so tests can patch it.
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
    def resolve_source(cls, key: dict) -> SourceResolution:
        """Return the source-resolution record for a sorting selection.

        Returns ``SourceResolution(kind, key)`` for the one source part.

        Raises
        ------
        SchemaBypassError
            If zero or multiple source part rows exist for ``key``.
        """
        kind, source_key = single_source_part(cls, key)
        return SourceResolution(kind=kind, key=source_key)

    @classmethod
    def resolve_artifact_detection(cls, key: dict):
        """Return the ``artifact_detection_id`` for a selection, or ``None``.

        Resolves the part's merge id to ``artifact_detection_id``. Concat
        sorts return ``None`` but may still be masked: their detections live
        on ``ConcatenatedRecordingSelection.MemberSnapshot``.

        Raises
        ------
        SchemaBypassError
            If more than one ``ArtifactDetectionSource`` row exists for ``key``.
        """
        merge_id = optional_part_value(
            cls, key, cls.ArtifactDetectionSource, "artifact_detection_merge_id"
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
        return optional_part_value(
            cls,
            key,
            cls.MotionCorrectionSource,
            "motion_corrected_recording_id",
        )

    @classmethod
    def resolve_effective_source(cls, key: dict) -> EffectiveSource:
        """Return a sort's lineage and the traces its consumers must read.

        Every consumer that loads the sorted traces reads ``traces``:
        the source's cached recording, or the selected
        ``MotionCorrectedRecording``. No trace file is opened.

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
        :func:`._recording.source.read_effective_recording`, or use
        :meth:`load_stored_traces`.

        Parameters
        ----------
        traces : EffectiveTraces
            The ``traces`` of :meth:`resolve_effective_source`.

        Returns
        -------
        str
            Absolute path of the (present) file, for
            :func:`._recording.source.read_effective_recording`.
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

        A corrected recording keeps its source's frames and sampling rate.
        """
        lineage, traces = source
        sampling_frequency = (
            traces.row["sampling_frequency"]
            if traces.kind == lineage.kind
            else (_TRACE_TABLES[lineage.kind] & lineage.key).fetch1(
                "sampling_frequency"
            )
        )
        return StoredUnits(
            abs_path=AnalysisNwbfile.get_abs_path(units_analysis_file_name),
            sampling_frequency=float(sampling_frequency),
        )

    @staticmethod
    def load_stored_traces(traces: EffectiveTraces) -> "si.BaseRecording":
        """Open the effective traces as persisted, rebuilding a missing file.

        No artifact mask is applied at load.

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

        The analyzer waveform window is keyed on this recipe. Read from
        ``RecordingSelection`` or, for a concat source,
        ``ConcatenatedRecordingSelection``.
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

    ``make()`` runs the sorter (with sorter-owned whitening when requested),
    removes excess spikes, builds a ``binary_folder`` ``SortingAnalyzer``
    with the base extensions, writes an ``AnalysisNwbfile`` holding only
    this sort's units, and populates ``Sorting.Unit``.
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
    # display_waveform_params_name is resolved once at sort time and read
    # back on rebuilds, so the analyzer is deterministic per sorting_id.
    # The analyzer folder is not stored: its path depends on temp_dir.

    class Unit(SpyglassMixinPart):
        """Per-unit metadata persisted at sort time.

        For concat sorts the Electrode FK points at the first member.
        """

        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uv: float    # peak template amplitude in microvolts
        n_spikes: int
        """

    # Tri-part make keeps the long sort outside the populate transaction.
    _parallel_make = True

    def make_fetch(self, key):
        """Read every DB input the compute step needs.

        A concat source anchors to its first frozen member. Return values
        must be deterministic: DataJoint hashes two fetches and compares
        them.

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

        A concat sort anchors to its first frozen member.

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

        The only DB access is staging the units NWB file. Silences a single
        recording's artifact frames, sorts, and builds the analyzer in a
        private staged folder. Cleans up its own staged outputs on error,
        since ``make_insert`` will not run.

        Parameters
        ----------
        key : dict
            Primary key of the sorting being populated.
        source : SourceResolution
            Resolved sort input source.
        recording_id : str
            The anchor ``recording_id``.
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
            The display recipe; selects the analyzer cache folder.
        display_waveform_params : dict
            That recipe's resolved params blob, passed to
            :func:`._sorting.analyzer.build_analyzer`.
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
            The sorter input.
        traces_abs_path : str
            Its absolute path.
        motion_correction_provenance : dict or None
            Motion correction ids and recipe names for the units NWB;
            ``None`` for an uncorrected sort.
        source_n_samples : int or None
            The source's frame count a corrected recording must keep;
            ``None`` for an uncorrected sort.

        Returns
        -------
        SortingComputed
            Carrier of the computed sorting, staged units NWB, analyzer
            folder, and lookups threaded into ``make_insert``.
        """
        # Concat and motion-corrected traces are stored masked; a single
        # recording is masked by sorting_statistics_spans below.
        from spyglass.spikesorting.v2._recording.source import (
            read_persisted_traces,
        )

        recording = read_persisted_traces(traces_abs_path, traces)

        # Artifact-free frame ranges for noise and whitening estimates,
        # stored so analyzer rebuilds reuse them.
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

        # Shared by the sorter and the analyzer build.
        from spyglass.spikesorting.v2._core.job_config import (
            _resolved_job_kwargs,
        )

        job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])
        import spikeinterface as si

        # A seed outside the row is not part of sorting_id, so reject it.
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
        sorting_obj = remove_excess_spikes(sorting_obj, recording)

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            StagedAnalyzer,
            analyzer_path,
        )

        # Owned by this attempt until insert, not a concurrent publication.
        staged_analyzer = StagedAnalyzer(
            analyzer_path(key["sorting_id"], display_waveform_params_name)
        )
        try:
            _sorting_analyzer.build_analyzer(
                sorting=sorting_obj,
                recording=recording,
                key=key,
                sorter_row=sorter_row,
                job_kwargs=job_kwargs,
                analyzer_folder=staged_analyzer.folder,
                waveform_params=display_waveform_params,
                statistics_spans=statistics_spans,
            )
            # Shared by the NWB columns and the Sorting.Unit insert.
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
                # The backend can change the output.
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
            # A failed write removes its own file.
            analysis_file_name, units_object_id = (
                _units_nwb.write_sorting_units_nwb(
                    sorting=sorting_obj,
                    recording=recording,
                    nwb_file_name=nwb_file_name,
                    obs_intervals=obs_intervals,
                    unit_metadata=unit_metadata,
                    source_provenance=source_provenance,
                )
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

        - Only the attempt whose master insert succeeds publishes its
          analyzer into the shared cache, before commit.
        - Every exit closes the staged analyzer. During ``populate()``,
          ``StagedOutputCleanupMixin`` removes a failed attempt's staged NWB;
          a direct caller must do so itself.
        - A direct call inside a caller-owned transaction is refused: the
          publication lock cannot be held until that later commit.

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
            The resolved display recipe, stored for later rebuilds.
        effective_random_seed : int
            Seed the sort actually used; secondary provenance, not identity.
        spikeinterface_version : str
            ``spikeinterface.__version__`` at sort time.
        sorter_version : str or None
            Sorter package distribution version, ``None`` for in-process sorters.
        unit_rows : list of dict
            ``Sorting.Unit`` rows built in ``make_compute``.
        staged_analyzer : StagedAnalyzer
            Private build to publish after insertion and close on every exit.

        Returns
        -------
        None
        """
        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_publication_transaction,
            remove_previous_sort_analyzer_caches,
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
                # The successful insert owns this sorting_id; only a rename
                # happens here.
                remove_previous_sort_analyzer_caches(
                    key["sorting_id"], keep=staged_analyzer.canonical_folder
                )
                staged_analyzer.publish()
        finally:
            staged_analyzer.close()

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

        ``_safe_context()`` is a no-op inside ``populate()``; it makes a
        direct call atomic.
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

        - The SI path reads the stored ``spike_sample_index`` frames, so
          spike trains are original recording frames (``t_start=0``).
        - The DataFrame path returns absolute ``spike_times`` in seconds.
        - A zero-unit sort returns an empty sorting and warns.

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
        # A motion-corrected recording keeps its source's sampling rate.
        source = SortingSelection.resolve_source(key)
        fs = float(
            (_TRACE_TABLES[source.kind] & source.key).fetch1(
                "sampling_frequency"
            )
        )

        if int(row["n_units"]) == 0:
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

        Artifact-free half-open frame ranges that cross no selection join,
        member join, or timestamp gap. Every analyzer build for the sort
        estimates noise and whitening inside them, so all builds agree.

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

        Rebuilds in place. To avoid ``ZeroUnitAnalyzerError``, check
        ``(Sorting & key).fetch1("n_units") > 0`` first.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        waveform_params_name : str, optional
            The analyzer recipe to load. ``None`` (default) loads the sort's
            display recipe; pass the whitened metric recipe explicitly when
            needed.
        rebuild : bool, optional
            If ``True`` (default), rebuild a missing or invalid folder in
            place. If ``False``, raise instead (used by the recompute audit).
        load_extensions : bool, optional
            If ``True`` (default), load all saved extensions so SI
            save/select/merge keep them. ``False`` loads them on demand.
            Waveforms stay memory-mapped either way.

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

        Computes only missing extensions, so it never recomputes
        ``waveforms`` or ``templates`` (that would delete their child
        extensions and change ``peak_amplitude_uv``). Results persist in the
        analyzer folder.

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
            :func:`ensure_extensions`. Amplitude and location extensions
            always use the sort's polarity (``resolve_peak_sign``).
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

        from spyglass.spikesorting.v2._core.signal_math import (
            resolve_peak_sign,
        )

        sorting_id = (self & key).fetch1("sorting_id")
        sorter_params, sorter_job_kwargs = (
            SorterParameters & (SortingSelection & key)
        ).fetch1("params", "job_kwargs")
        resolved = _resolved_job_kwargs(sorter_job_kwargs)
        resolved.update(kwargs)
        # Lock covers load and in-place compute; reentrant for get_analyzer.
        with analyzer_cache_lock(sorting_id):
            analyzer = self.get_analyzer(
                key, waveform_params_name=waveform_params_name
            )
            return ensure_extensions(
                analyzer,
                extensions,
                peak_sign=resolve_peak_sign(sorter_params),
                job_kwargs=resolved,
                extension_params=extension_params,
            )

    # ---- visualization / export delegates (see v2.visualization facade) ---

    def root_curation(self, key):
        """Return the generation-pinned ``CurationRef`` of this sort's root.

        The plot and export delegates below pass it to ``v2.visualization``.

        Raises
        ------
        ValueError
            If no root curation exists.
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

    def delete(self, *args, safemode=None, **kwargs):
        """Cascade-delete + analyzer-cache cleanup on disk.

        The analyzer folder (5-50 GB) is not a tracked column, so this
        removes it for each deleted row. A leading positional restriction is
        applied as a restriction; ``cautious_delete`` would otherwise read it
        as ``force_permission`` and delete every row.

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
        restricted, args = restrict_by_leading_args(self, args)
        if restricted is not None:
            return restricted.delete(*args, safemode=safemode, **kwargs)

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
            remove_analyzer_cache,
        )

        # Fetch before the cascade removes the rows.
        rows = self.fetch("KEY", as_dict=True)
        if safemode is None:
            super().delete(*args, **kwargs)
        else:
            super().delete(*args, safemode=safemode, **kwargs)
        # A cancelled prompt returns normally; keep folders of surviving rows.
        surviving = set((Sorting & rows).fetch("sorting_id"))
        for row in rows:
            if row["sorting_id"] not in surviving:
                with analyzer_cache_lock(row["sorting_id"]):
                    remove_analyzer_cache(row["sorting_id"], missing_ok=True)

    @classmethod
    def find_orphaned_analyzer_folders(
        cls, *, sorting_id=None, dry_run: bool = True
    ) -> dict:
        """Audit 5-50 GB analyzer-folder disk leaks; never auto-delete DB rows.

        Cascades from upstream tables (``Recording``, ``SortGroupV2``) and
        raw SQL skip :meth:`delete`, leaving folders behind. Reports:

        - **DB-side orphan**: a ``Sorting`` row whose analyzer folder is
          missing. Reported only; rows are never deleted.
        - **Reclaimed**: a missing folder with a
          ``SortingAnalyzerRecompute.deleted=1`` record.
        - **Disk-side orphan**: a folder no live sort, evaluation, or
          curation references.
        - **Staging**: a build or trash folder whose locks are free. Active
          attempts, including on other hosts, are skipped.

        Zero-unit sorts have no folder and are not orphans.

        Parameters
        ----------
        sorting_id : UUID or str, optional
            Restrict database queries and cache candidates to this sorting.
            Omit to audit all sortings, including folders for deleted sorts.
        dry_run : bool, optional
            When True (default) only report. When False, after confirmation,
            delete disk-side orphans and abandoned staging. Staging ownership
            is rechecked before removal.

        Returns
        -------
        dict
            ``{"db_side": [{"sorting_id", "computed_analyzer_path"}, ...],
            "disk_side": [folder_path_str, ...],
            "reclaimed": [{"sorting_id", "computed_analyzer_path"}, ...],
            "staging": [folder_path_str, ...]}``.
        """

        return _analyzer_cache.find_orphaned_analyzer_folders(
            cls, sorting_id=sorting_id, dry_run=dry_run
        )

    def get_unit_brain_regions(
        self, key: dict, *, allow_anchor_member: bool = False
    ) -> "pd.DataFrame":
        """Per-unit brain regions via Sorting.Unit * Electrode * BrainRegion.

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
            and ``region_resolution`` (``'single_session'`` or
            ``'anchor_member'``).

        Raises
        ------
        ConcatBrainRegionAmbiguousError
            For a concat sort unless ``allow_anchor_member=True``.
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

        ``execution_params`` applies only to SI sorters. ``statistics_spans``
        (``None`` means the whole recording) feed the clusterless MAD and
        external whitening estimates.
        """
        if sorter == "clusterless_thresholder":
            return run_clusterless_thresholder(
                sorter_params=sorter_params,
                recording=recording,
                job_kwargs=job_kwargs,
                statistics_spans=statistics_spans,
            )
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
    def _populate_unit_part(unit_rows):
        """Insert the pre-built ``Sorting.Unit`` rows.

        An empty list (zero-unit sort) is a no-op.
        """
        Sorting.Unit.insert(unit_rows)
