"""Motion estimation for spike-sorting v2, independent of concatenation.

Tables:
    MotionEstimationParameters: named, validated SpikeInterface motion recipes.
    MotionEstimateSelection: one source (a Recording or an uncorrected
        ConcatenatedRecording), an optional artifact mask, and a recipe.
        .RecordingSource: single-recording source.
        .ConcatenatedRecordingSource: concatenated-recording source.
        .ArtifactDetectionSource: optional mask (single recording only).
    MotionEstimate: the saved SpikeInterface ``Motion``, its estimation clock,
        resolved configuration and diagnostics.
    MotionInterpolationParameters: named interpolation recipes, every
        ``interpolate_motion`` argument explicit.
    MotionCorrectionParameters: public recipe pairing one estimation and one
        interpolation recipe.
    MotionCorrectedRecordingSelection: one saved estimate and an interpolation
        recipe.
    MotionCorrectedRecording: the corrected, masked, unwhitened traces written
        with the source's own timestamps.

``SortingSelection.MotionCorrectionSource`` selects a corrected recording as
a sort's input. Computation lives in ``_motion``.
"""

from __future__ import annotations

from typing import NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._motion import (
    compute as _motion_compute,
    report_inputs as _motion_report_inputs,
    selection as _motion_selection_insert,
)
from spyglass.spikesorting.v2._params.motion_estimation import (
    MOTION_ESTIMATION_SCHEMA_VERSION,
    MotionEstimationParamsSchema,
)
from spyglass.spikesorting.v2._params.motion_interpolation import (
    MOTION_INTERPOLATION_SCHEMA_VERSION,
    MotionInterpolationParamsSchema,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    motion_correction_default_contents,
    motion_estimation_default_contents,
    motion_interpolation_default_contents,
)
from spyglass.spikesorting.v2._recording.source import SourceLineage
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
)
from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
from spyglass.spikesorting.v2.recording import (
    PreprocessingParameters,
    Recording,
    RecordingSelection,
)
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,
    ConcatenatedRecordingSelection,
)
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _insert_parameter_rows,
    optional_part_value,
    single_source_part,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart
from spyglass.utils.mixins.populate import StagedOutputs

schema = dj.schema("spikesorting_v2_motion")


@schema
class MotionEstimationParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Named motion-estimation recipes: a SpikeInterface preset plus overrides.

    ``params`` is validated by :class:`MotionEstimationParamsSchema`; an
    override key the selected SpikeInterface method does not accept is
    rejected at insert. ``insert_default`` ships ``dredge_v1`` and
    ``dredge_fast_v1``. No recipe here is validated for a particular probe.

    ``job_kwargs`` (detect-and-localize pass) is not part of the estimate's
    identity. The seed belongs in ``params.noise_levels_seed``, so a
    ``random_seed`` job kwarg is rejected.
    """

    definition = f"""
    motion_estimation_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={MOTION_ESTIMATION_SCHEMA_VERSION}: int
    job_kwargs=null: blob  # SI job kwargs for detection and localization
    """

    _DEFAULT_CONTENTS: tuple = motion_estimation_default_contents()

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert motion-estimation parameter rows after validation.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard; see ``reject_duplicate_parameter_content``.

        Raises
        ------
        ValueError
            If ``params`` has an unknown override key or ``job_kwargs``
            contains ``random_seed``.
        """
        from spyglass.spikesorting.v2._motion.estimation import (
            resolve_estimation_params,
        )

        def _resolve_and_check_job_kwargs(row, _schema_cls):
            resolve_estimation_params(row["params"])
            if "random_seed" in (row.get("job_kwargs") or {}):
                raise ValueError(
                    "MotionEstimationParameters.job_kwargs must not contain "
                    "'random_seed': the noise-estimate seed is the identity-"
                    "bearing params field 'noise_levels_seed'."
                )

        _insert_parameter_rows(
            self,
            rows,
            insert_rows=super().insert,
            schema_for=lambda _row: MotionEstimationParamsSchema,
            table_name="MotionEstimationParameters",
            per_row_hook=_resolve_and_check_job_kwargs,
            name_attr="motion_estimation_params_name",
            allow_duplicate_params=allow_duplicate_params,
            **kwargs,
        )

    @classmethod
    def insert_default(cls):
        """Insert the shipped motion-estimation recipes if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class MotionInterpolationParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Named motion-interpolation recipes for applying a saved estimate.

    Validated by :class:`MotionInterpolationParamsSchema`. Every argument
    (``border_mode``, ``spatial_interpolation_method``, ``sigma_um``, ``p``,
    ``num_closest``) is required so no SpikeInterface default is relied on.
    ``insert_default`` ships ``kriging_force_extrapolate_v1`` (the
    ``dredge`` presets' interpolation) and ``kriging_remove_channels_v1``.
    """

    definition = f"""
    motion_interpolation_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={MOTION_INTERPOLATION_SCHEMA_VERSION}: int
    """

    _DEFAULT_CONTENTS: tuple = motion_interpolation_default_contents()

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert motion-interpolation parameter rows after validation.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard; see ``reject_duplicate_parameter_content``.
        """
        _insert_parameter_rows(
            self,
            rows,
            insert_rows=super().insert,
            schema_for=lambda _row: MotionInterpolationParamsSchema,
            table_name="MotionInterpolationParameters",
            name_attr="motion_interpolation_params_name",
            allow_duplicate_params=allow_duplicate_params,
            **kwargs,
        )

    @classmethod
    def insert_default(cls):
        """Insert the shipped motion-interpolation recipes if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class MotionCorrectionParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Named public motion-correction recipes: estimation plus interpolation.

    Recipes sharing an estimation row reuse the same saved
    ``MotionEstimate``. ``insert_default`` ships ``dredge_v1`` and
    ``dredge_fast_v1``; neither is validated for a probe.
    """

    definition = """
    motion_correction_params_name: varchar(64)
    ---
    -> MotionEstimationParameters
    -> MotionInterpolationParameters
    """

    _DEFAULT_CONTENTS: tuple = motion_correction_default_contents()

    @classmethod
    def insert_default(cls):
        """Insert the shipped recipes (and the rows they name) if missing."""
        MotionEstimationParameters.insert_default()
        MotionInterpolationParameters.insert_default()
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


#: The table owning each source kind's cached trace artifact.
_SOURCE_TABLES = {
    "recording": Recording,
    "concatenated_recording": ConcatenatedRecording,
}

#: The selection table naming each source kind's preprocessing recipe.
_SOURCE_SELECTION_TABLES = {
    "recording": RecordingSelection,
    "concatenated_recording": ConcatenatedRecordingSelection,
}


def preprocessing_filter_problem(preprocessing_params_name: str) -> str | None:
    """Say why a preprocessing recipe's traces cannot be motion-estimated.

    Fetches the recipe's ``params`` and applies
    :func:`._motion.unfiltered_source_problem`.

    Parameters
    ----------
    preprocessing_params_name : str
        A ``PreprocessingParameters`` row name.

    Returns
    -------
    str or None
        The problem, or ``None`` when the recipe applies a temporal filter.
    """
    from spyglass.spikesorting.v2._motion.estimation import (
        unfiltered_source_problem,
    )

    return unfiltered_source_problem(
        preprocessing_params_name,
        (
            PreprocessingParameters
            & {"preprocessing_params_name": preprocessing_params_name}
        ).fetch1("params"),
    )


def _source_filter_problem(source_kind: str, source_key: dict) -> str | None:
    """:func:`preprocessing_filter_problem` of a motion source's recipe.

    A concatenation's recipe is the one it was built with.
    """
    return preprocessing_filter_problem(
        (_SOURCE_SELECTION_TABLES[source_kind] & source_key).fetch1(
            "preprocessing_params_name"
        )
    )


@schema
class MotionEstimateSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One motion estimate to compute: a source, an optional mask, a recipe.

    Exactly one of ``RecordingSource`` / ``ConcatenatedRecordingSource``
    exists per row. ``ArtifactDetectionSource`` applies to single recordings
    only; a concatenation carries its members' masks. ``motion_estimate_id``
    hashes the source, its ``content_hash``, the artifact detection, the
    recipe, the resolved configuration, and the SpikeInterface and algorithm
    versions. Create rows with :meth:`insert_selection`.
    """

    definition = """
    motion_estimate_id: uuid
    ---
    -> MotionEstimationParameters
    resolved_params_hash: char(64)       # SHA-256 of the resolved estimation configuration
    spikeinterface_version: varchar(32)  # SpikeInterface the configuration was resolved with
    motion_algorithm_version: int        # estimation algorithm version at selection
    source_content_hash: char(64)        # content_hash of the source trace artifact
    """

    class RecordingSource(SpyglassMixinPart):
        """Single-recording source of a motion estimate."""

        definition = """
        -> master
        ---
        -> Recording
        """

    class ConcatenatedRecordingSource(SpyglassMixinPart):
        """Concatenated-recording source of a motion estimate."""

        definition = """
        -> master
        ---
        -> ConcatenatedRecording
        """

    class ArtifactDetectionSource(SpyglassMixinPart):
        """Optional artifact mask of a single-recording motion estimate."""

        definition = """
        -> master
        ---
        -> ArtifactDetectionOutput.proj(artifact_detection_merge_id='merge_id')
        """

    _INPUT_FIELDS = frozenset(
        {
            "recording_id",
            "concat_recording_id",
            "artifact_detection_id",
            "motion_estimation_params_name",
            "motion_estimate_id",
        }
    )

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert (or find) the selection for a source, mask and recipe.

        Parameters
        ----------
        key : dict
            Exactly one of ``recording_id`` / ``concat_recording_id``,
            ``motion_estimation_params_name``, and optionally
            ``artifact_detection_id`` (single recording only). An explicit
            ``motion_estimate_id`` is cross-checked against the derived id.

        Returns
        -------
        dict
            ``{"motion_estimate_id": ...}`` of the inserted-or-existing row.

        Raises
        ------
        ValueError
            On unknown fields, zero or two sources, an artifact detection on
            a concat source, a missing recipe or unpopulated source, a source
            whose preprocessing recipe applies no temporal filter
            (:func:`._motion.unfiltered_source_problem`; for a concat, the
            concatenation's recipe), an artifact detection that does not
            belong to the recording, or a mismatched explicit
            ``motion_estimate_id``.
        DuplicateSelectionError
            If a matching row has a non-deterministic ``motion_estimate_id``.
        SchemaBypassError
            If the deterministic master exists with other source parts.
        """
        return _motion_selection_insert.insert_estimate_selection(cls, key)

    @classmethod
    def _find_existing_pk(
        cls,
        master_row,
        source_part,
        source_key,
        artifact_detection_id,
        deterministic_id,
    ) -> dict | None:
        """Return the canonical PK for this selection, or ``None``.

        Matches masters with the same identity columns, source part and
        artifact state; any match whose id is not the deterministic one is a
        raw-insert bypass.

        Raises
        ------
        DuplicateSelectionError
            If a match has a non-deterministic ``motion_estimate_id``.
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            existing_selection_pk,
        )

        candidates = ((cls * source_part) & master_row & source_key).fetch(
            "motion_estimate_id"
        )
        matching = {
            candidate
            for candidate in candidates
            if cls.resolve_artifact_detection({"motion_estimate_id": candidate})
            == artifact_detection_id
        }
        return existing_selection_pk(
            matching,
            deterministic_id,
            pk_field="motion_estimate_id",
            bypass_message=lambda bypassed: (
                "MotionEstimateSelection has rows for "
                f"{source_key} / {master_row['motion_estimation_params_name']} "
                "whose motion_estimate_id is not the deterministic id "
                f"{deterministic_id}: {bypassed}. Drop them and re-insert via "
                "insert_selection."
            ),
        )

    @classmethod
    def resolve_artifact_detection(cls, key: dict):
        """Return the pinned ``artifact_detection_id``, or ``None``.

        Raises
        ------
        SchemaBypassError
            If more than one ``ArtifactDetectionSource`` row exists.
        """
        merge_id = optional_part_value(
            cls, key, cls.ArtifactDetectionSource, "artifact_detection_merge_id"
        )
        if merge_id is None:
            return None
        return ArtifactDetectionOutput.resolve_artifact_detection_id(merge_id)

    @classmethod
    def resolve_source(cls, key: dict) -> SourceLineage:
        """Return the selection's source and pinned artifact detection.

        Re-derives ``motion_estimate_id`` from the current parts to catch
        parts inserted or deleted after selection.

        Raises
        ------
        SchemaBypassError
            If the row has zero or two source parts, a concat source
            carries an artifact detection, or the current parts do not give
            the stored ``motion_estimate_id`` (a raw insert or part-only
            delete bypassing :meth:`insert_selection`).
        """
        from spyglass.spikesorting.v2._motion.estimation import (
            motion_estimate_parts_mismatch,
        )
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        kind, source_key = single_source_part(cls, master_key)
        artifact_detection_id = cls.resolve_artifact_detection(master_key)
        if (
            kind == "concatenated_recording"
            and artifact_detection_id is not None
        ):
            raise SchemaBypassError(
                f"MotionEstimateSelection {master_key} pairs a concatenated "
                "recording with an ArtifactDetectionSource; a concat source "
                "owns its member masks. Drop the stray part row."
            )
        lineage = SourceLineage(
            kind=kind,
            key=source_key,
            artifact_detection_id=artifact_detection_id,
        )
        master = (cls & master_key).fetch1()
        mismatch = motion_estimate_parts_mismatch(
            master["motion_estimate_id"], lineage, master
        )
        if mismatch is not None:
            raise SchemaBypassError(
                f"MotionEstimateSelection {master['motion_estimate_id']}: its "
                f"current parts ({mismatch}) are not the ones it was selected "
                "with. A source or ArtifactDetectionSource part was inserted "
                "or deleted without MotionEstimateSelection.insert_selection, "
                "so the estimate would describe another mask or source than "
                "the one selected. Drop the selection (and everything made "
                "from it) and re-insert it with insert_selection."
            )
        return lineage


class MotionEstimateFetched(NamedTuple):
    """DB inputs of :meth:`MotionEstimate.make_compute`.

    Attributes
    ----------
    lineage : SourceLineage
        Source kind, key and pinned artifact detection.
    selection : dict
        The ``MotionEstimateSelection`` row.
    params : dict
        The recipe's ``params`` blob.
    job_kwargs : dict
        Resolved SpikeInterface job kwargs.
    source_row : dict
        The ``Recording`` / ``ConcatenatedRecording`` row.
    source_path : str
        Absolute path of the source's analysis NWB (rebuilt if it was missing).
    artifact_valid_times : numpy.ndarray or None
        ``(n_intervals, 2)`` artifact-removed valid times in seconds, for a
        masked single-recording source.
    """

    lineage: SourceLineage
    selection: dict
    params: dict
    job_kwargs: dict
    source_row: dict
    source_path: str
    artifact_valid_times: np.ndarray | None


class MotionEstimateComputed(NamedTuple):
    """``MotionEstimate`` row fields returned by ``make_compute``."""

    motion: dict
    resolved_params: dict
    n_samples: int
    sampling_frequency: float
    continuity_spans: np.ndarray
    continuity_start_s: np.ndarray
    continuity_end_s: np.ndarray
    estimation_start_s: np.ndarray
    statistics_spans: np.ndarray
    channel_ids: list
    channel_locations: np.ndarray
    max_abs_displacement_um: float
    n_temporal_bins: int
    n_peaks_detected: int
    n_peaks_kept: int
    peaks_per_temporal_bin: np.ndarray
    peaks_per_continuity_span: np.ndarray
    noise_levels: np.ndarray
    input_fingerprint: str


#: ``EstimationClock`` field -> the ``MotionEstimate`` column that stores it.
_ESTIMATION_CLOCK_COLUMNS = {
    "spans": "continuity_spans",
    "source_start_s": "continuity_start_s",
    "source_end_s": "continuity_end_s",
    "estimation_start_s": "estimation_start_s",
    "sampling_frequency": "sampling_frequency",
}


def _estimation_clock_of(row: dict):
    """The ``_motion.EstimationClock`` stored in a ``MotionEstimate`` row.

    ``row`` needs only the time-map columns (``_ESTIMATION_CLOCK_COLUMNS``).
    """
    from spyglass.spikesorting.v2._motion.estimation import (
        estimation_clock_from_blob,
    )

    return estimation_clock_from_blob(
        {
            field: row[column]
            for field, column in _ESTIMATION_CLOCK_COLUMNS.items()
        }
    )


@schema
class MotionEstimate(SpyglassMixin, dj.Computed):
    """A saved motion estimate with its resolved configuration and evidence.

    ``motion`` is the SpikeInterface ``Motion`` as a blob dict; read it with
    :meth:`get_motion`. ``resolved_params`` is exactly the configuration
    passed to SpikeInterface.

    The ``motion`` bins are on the source's *estimation clock*: time
    advances ``1 / fs`` per frame within each continuity span, and the real
    gap between spans is kept up to the recipe's ``max_gap_s``.
    :meth:`get_estimation_clock` returns the time map and
    :meth:`get_displacement_on_source_clock` maps bins back to source time.

    Populated explicitly. If the recipe's resolution, SpikeInterface version
    or algorithm version changed since selection, populate raises; recreate
    the selection.
    """

    _populate_key_lock = True

    definition = """
    -> MotionEstimateSelection
    ---
    motion: longblob                  # SpikeInterface Motion as a blob dict, temporal bins on the estimation clock; read with get_motion
    resolved_params: longblob         # the resolved estimation configuration passed to SpikeInterface
    n_samples: bigint                 # frames of the estimated recording
    sampling_frequency: double        # Hz
    continuity_spans: longblob        # (n, 2) int64 half-open frame ranges of uninterrupted acquisition
    continuity_start_s: longblob      # (n,) float64 first timestamp of each continuity span on the source's own clock, in seconds
    continuity_end_s: longblob        # (n,) float64 last timestamp of each continuity span on the source's own clock, in seconds
    estimation_start_s: longblob      # (n,) float64 start of each continuity span on the estimation clock, in seconds
    statistics_spans: longblob        # (n, 2) int64 half-open frame ranges of the valid samples used
    channel_ids: longblob             # estimation channel ids in recording order
    channel_locations: longblob       # (n_channels, 2) float64 contact positions in um
    max_abs_displacement_um: double   # largest absolute displacement over all bins and windows
    n_temporal_bins: int              # temporal bins of the estimate
    n_peaks_detected: int             # peaks detected on the masked recording
    n_peaks_kept: int                 # peaks localized inside one statistics span, detected clear of span joins
    peaks_per_temporal_bin: longblob  # (n_temporal_bins,) int64 kept peaks per temporal bin
    peaks_per_continuity_span: longblob  # (n_spans,) int64 kept peaks per continuity span; 0 marks a span with no evidence
    noise_levels: longblob            # (n_channels,) float64 detection noise in microvolts
    input_fingerprint: char(64)       # SHA-256 of source content, spans, channels and configuration
    """

    _parallel_make = True

    def make_fetch(self, key) -> MotionEstimateFetched:
        """Resolve the selection, recipe, source artifact and mask.

        Rebuilds a missing source NWB.

        Raises
        ------
        ValueError
            If the source's ``content_hash`` changed since selection, or its
            preprocessing recipe applies no temporal filter (re-checked here
            for rows inserted with ``allow_direct_insert=True``).
        """
        from spyglass.spikesorting.v2._artifacts.readers import (
            read_recording_artifact_valid_times,
        )
        from spyglass.spikesorting.v2._storage.rebuilds import (
            ensure_artifact_file,
        )
        from spyglass.spikesorting.v2._core.job_config import (
            _resolved_job_kwargs,
        )

        lineage = MotionEstimateSelection.resolve_source(key)
        problem = _source_filter_problem(lineage.kind, lineage.key)
        if problem is not None:
            raise ValueError(
                f"MotionEstimate: {lineage.key} cannot be motion-estimated: "
                f"{problem}"
            )
        selection = (MotionEstimateSelection & key).fetch1()
        params, job_kwargs = (
            MotionEstimationParameters
            & {
                "motion_estimation_params_name": selection[
                    "motion_estimation_params_name"
                ]
            }
        ).fetch1("params", "job_kwargs")
        table, source_row = _live_source_row(
            lineage, key["motion_estimate_id"], selection["source_content_hash"]
        )
        source_path = ensure_artifact_file(
            table, lineage.key, source_row["analysis_file_name"]
        )

        artifact_valid_times = None
        if lineage.artifact_detection_id is not None:
            artifact_valid_times = read_recording_artifact_valid_times(
                lineage.artifact_detection_id,
                (RecordingSelection & lineage.key).fetch1("nwb_file_name"),
                caller="MotionEstimate.make_fetch",
            )

        return MotionEstimateFetched(
            lineage=lineage,
            selection=selection,
            params=params,
            job_kwargs=_resolved_job_kwargs(job_kwargs),
            source_row=source_row,
            source_path=source_path,
            artifact_valid_times=artifact_valid_times,
        )

    def make_compute(
        self,
        key,
        lineage,
        selection,
        params,
        job_kwargs,
        source_row,
        source_path,
        artifact_valid_times,
    ) -> MotionEstimateComputed:
        """Estimate motion from the resolved files; no DB access.

        Frames outside the statistics spans are silenced (after conversion
        to microvolts) before one ``estimate_motion_in_spans`` call. A
        concat source's spans come from its row.

        Raises
        ------
        ValueError
            On a stale selection, spans out of acquisition order, or any
            estimation failure (see ``estimate_motion_in_spans``).
        """
        from spyglass.spikesorting.v2._motion import estimation as _motion
        from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb

        resolved = _motion.resolve_estimation_params(params)
        resolved_hash = _motion.resolved_params_hash(resolved)
        stale = _motion_compute.stale_estimate_selection_fields(
            selection, resolved_hash
        )
        if stale:
            raise ValueError(
                f"MotionEstimate {key}: the selection is stale ({'; '.join(stale)}). "
                "Recreate it with MotionEstimateSelection.insert_selection, "
                "which derives a new motion_estimate_id."
            )

        recording = read_recording_nwb(
            source_path,
            electrical_series_path=source_row["electrical_series_path"],
        )
        recording.annotate(is_filtered=True)
        n_samples = int(recording.get_num_samples())
        continuity, continuity_start_s, continuity_end_s, statistics = (
            _motion_compute.estimation_spans(
                recording, lineage, source_row, artifact_valid_times, n_samples
            )
        )

        sampling_frequency = float(recording.get_sampling_frequency())
        clock = _motion.build_estimation_clock(
            continuity,
            continuity_start_s,
            continuity_end_s,
            sampling_frequency,
            resolved["max_gap_s"],
        )
        motion, diagnostics = _motion.estimate_motion_in_spans(
            recording,
            statistics_spans=statistics,
            clock=clock,
            resolved_params=resolved,
            job_kwargs=job_kwargs,
        )
        channel_ids = recording.channel_ids.tolist()
        channel_locations = np.asarray(
            recording.get_channel_locations(), dtype=np.float64
        )
        statistics_arr = np.asarray(statistics, dtype=np.int64).reshape(-1, 2)
        return MotionEstimateComputed(
            motion=_motion.motion_to_storage_dict(motion),
            resolved_params=resolved,
            n_samples=n_samples,
            sampling_frequency=sampling_frequency,
            continuity_spans=clock.spans,
            continuity_start_s=clock.source_start_s,
            continuity_end_s=clock.source_end_s,
            estimation_start_s=clock.estimation_start_s,
            statistics_spans=statistics_arr,
            channel_ids=channel_ids,
            channel_locations=channel_locations,
            max_abs_displacement_um=_motion.motion_max_abs_displacement_um(
                motion
            ),
            n_temporal_bins=_motion.motion_n_temporal_bins(motion),
            n_peaks_detected=diagnostics.n_peaks_detected,
            n_peaks_kept=diagnostics.n_peaks_kept,
            peaks_per_temporal_bin=diagnostics.peaks_per_temporal_bin,
            peaks_per_continuity_span=diagnostics.peaks_per_continuity_span,
            noise_levels=diagnostics.noise_levels,
            input_fingerprint=_motion.motion_input_fingerprint(
                source_content_hash=selection["source_content_hash"],
                artifact_detection_id=lineage.artifact_detection_id,
                n_samples=n_samples,
                sampling_frequency=sampling_frequency,
                continuity_spans=clock.spans,
                continuity_start_s=clock.source_start_s,
                continuity_end_s=clock.source_end_s,
                statistics_spans=statistics_arr,
                channel_ids=channel_ids,
                channel_locations=channel_locations,
                resolved_params_hash=resolved_hash,
            ),
        )

    def make_insert(self, key, *computed) -> None:
        """Insert the estimate row inside the framework transaction."""
        self.insert1({**key, **MotionEstimateComputed(*computed)._asdict()})

    def get_motion(self, key: dict):
        """Rebuild the saved SpikeInterface ``Motion`` for one estimate.

        Its temporal bins are on the source's estimation clock, not the
        source's own clock: see :meth:`get_estimation_clock` and
        :meth:`get_displacement_on_source_clock`.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``MotionEstimate`` row.

        Returns
        -------
        spikeinterface.core.motion.Motion
        """
        from spyglass.spikesorting.v2._motion.estimation import (
            motion_from_storage_dict,
        )

        return motion_from_storage_dict((self & key).fetch1("motion"))

    def get_estimation_clock(self, key: dict):
        """The time map one estimate's bins are on.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``MotionEstimate`` row.

        Returns
        -------
        _motion.EstimationClock
            The continuity spans in frames, each span's first and last
            timestamp on the source clock, its start on the estimation clock,
            and the sampling frequency.
        """
        return _estimation_clock_of(
            (self & key).proj(*_ESTIMATION_CLOCK_COLUMNS.values()).fetch1()
        )

    def get_spans_without_evidence(self, key: dict) -> list[dict]:
        """The continuity spans of one estimate that kept no peak.

        Displacement over such a span comes only from the estimator's
        temporal prior. Reported, not refused, since dropped-frame gaps can
        leave spans too short to hold a peak.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``MotionEstimate`` row.

        Returns
        -------
        list[dict]
            ``_motion.spans_without_evidence``: one ``{"span_index",
            "start_frame", "end_frame", "source_start_s", "source_end_s"}``
            per such span (times on the source's own clock, s); empty when
            every span kept a peak.
        """
        from spyglass.spikesorting.v2._motion.estimation import (
            spans_without_evidence,
        )

        return spans_without_evidence(
            (self & key).fetch1("peaks_per_continuity_span"),
            self.get_estimation_clock(key),
        )

    def get_displacement_on_source_clock(self, key: dict):
        """One estimate's displacement with its bins in source time.

        For inspection only: a bin whose center lies in a capped gap between
        two continuity spans is reported as such (NaN time, span ``-1``)
        rather than assigned to a span.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``MotionEstimate`` row.

        Returns
        -------
        _motion.SourceClockDisplacement
        """
        from spyglass.spikesorting.v2._motion.estimation import (
            displacement_on_source_clock,
        )

        return displacement_on_source_clock(
            self.get_motion(key), self.get_estimation_clock(key)
        )

    def report(
        self,
        key,
        corrected_key=None,
        trace_window_s=None,
        *,
        trace_channel_ids=None,
    ):
        """Summarize and plot one estimate before (or after) applying it.

        Spans that kept no peak (corrected from the temporal prior alone)
        are listed in the summary and marked in the figure.

        Parameters
        ----------
        key : dict, uuid.UUID or str
            One ``MotionEstimate``: a restriction, an ``estimate_motion``
            receipt (only its ``motion_estimate_id`` is used) or the id.
        corrected_key : dict, uuid.UUID or str, optional
            A ``MotionCorrectedRecording`` made from this estimate: a
            restriction, any mapping holding its
            ``motion_corrected_recording_id`` (only the id is used, e.g. a
            ``run_v2_pipeline`` receipt) or the id. Adds its
            ``border_mode`` and ``removed_channel_ids``.
        trace_window_s : tuple of float, optional
            ``(start, end)`` on the source's own clock (s, the clock of
            ``get_spans_without_evidence``): adds original vs corrected
            traces over that window. Keep it short; the traces are read
            into memory. Requires ``corrected_key``.
        trace_channel_ids : list, optional
            Channels of the corrected recording to show in the trace panel.
            Defaults to the four nearest the middle of the probe's depth.

        Returns
        -------
        matplotlib.figure.Figure
            :func:`._motion.report.plot_motion_report`.
        dict
            :func:`._motion.report.motion_report_summary` plus
            ``motion_estimate_id`` and ``motion_corrected_recording_id``
            (``None`` without ``corrected_key``).

        Raises
        ------
        ValueError
            If ``trace_window_s`` is given without ``corrected_key`` or
            selects no frame, if the corrected recording was made from
            another estimate, or if a trace channel is not in it.

        Notes
        -----
        Times inside a continuity span are approximate up to timestamp
        jitter; span starts and ends are exact.

        Each call opens a new figure; close it in loops.
        """
        return _motion_report_inputs.motion_estimate_report(
            self,
            key,
            corrected_key,
            trace_window_s,
            trace_channel_ids=trace_channel_ids,
        )


#: ``MotionEstimate`` columns the corrected recording is computed from.
_ESTIMATE_APPLICATION_FIELDS = (
    "motion",
    "n_samples",
    *_ESTIMATION_CLOCK_COLUMNS.values(),
    "statistics_spans",
    "channel_ids",
    "channel_locations",
)


def _live_source_row(
    lineage: SourceLineage, motion_estimate_id, selected_hash: str
) -> tuple:
    """The source table and live row of an estimate's resolved lineage.

    Raises
    ------
    ValueError
        If the source artifact's ``content_hash`` differs from
        ``selected_hash``.
    """
    table = _SOURCE_TABLES[lineage.kind]
    source_row = (table & lineage.key).fetch1()
    if source_row["content_hash"] != selected_hash:
        raise ValueError(
            f"{table.__name__} {lineage.key} changed since motion estimate "
            f"{motion_estimate_id} was selected (content_hash "
            f"{source_row['content_hash']} != {selected_hash}); the saved "
            "motion no longer describes these traces. Select and populate a "
            "new estimate."
        )
    return table, source_row


def _estimate_source(motion_estimate_id) -> tuple:
    """The source table, lineage and live row of a saved estimate.

    Raises
    ------
    ValueError
        If the source artifact's ``content_hash`` changed since the estimate
        was selected (see :func:`_live_source_row`).
    """
    estimate_key = {"motion_estimate_id": motion_estimate_id}
    lineage = MotionEstimateSelection.resolve_source(estimate_key)
    table, source_row = _live_source_row(
        lineage,
        motion_estimate_id,
        (MotionEstimateSelection & estimate_key).fetch1("source_content_hash"),
    )
    return table, lineage, source_row


@schema
class MotionCorrectedRecordingSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One corrected recording to compute: a saved estimate and a recipe.

    ``motion_corrected_recording_id`` hashes the ``motion_estimate_id``,
    the interpolation recipe and its resolved configuration, and the
    SpikeInterface and algorithm versions. Create rows with
    :meth:`insert_selection`.
    """

    definition = """
    motion_corrected_recording_id: uuid
    ---
    -> MotionEstimate
    -> MotionInterpolationParameters
    resolved_params_hash: char(64)                # SHA-256 of the resolved interpolation configuration
    spikeinterface_version: varchar(32)           # SpikeInterface the corrected recording is computed with
    motion_interpolation_algorithm_version: int   # application algorithm version at selection
    """

    _INPUT_FIELDS = frozenset(
        {
            "motion_estimate_id",
            "motion_interpolation_params_name",
            "motion_corrected_recording_id",
        }
    )

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert (or find) the selection for a saved estimate and a recipe.

        Parameters
        ----------
        key : dict
            ``motion_estimate_id`` (a populated ``MotionEstimate``) and
            ``motion_interpolation_params_name``. An explicit
            ``motion_corrected_recording_id`` is cross-checked against the
            derived id.

        Returns
        -------
        dict
            ``{"motion_corrected_recording_id": ...}`` of the
            inserted-or-existing row.

        Raises
        ------
        ValueError
            On unknown or missing fields, a missing recipe, an unpopulated
            estimate, a source whose content changed since the estimate was
            selected, or a mismatched explicit id.
        DuplicateSelectionError
            If a matching row has a non-deterministic id.
        """
        return _motion_selection_insert.insert_corrected_selection(cls, key)

    @classmethod
    def resolve_source(cls, key: dict) -> SourceLineage:
        """Return the source and artifact mask the corrected recording is of.

        Parameters
        ----------
        key : dict
            Restriction selecting one selection row.

        Returns
        -------
        SourceLineage
        """
        motion_estimate_id = (cls & key).fetch1("motion_estimate_id")
        return MotionEstimateSelection.resolve_source(
            {"motion_estimate_id": motion_estimate_id}
        )

    @classmethod
    def _find_existing_pk(
        cls, master_row: dict, deterministic_id
    ) -> dict | None:
        """Return the canonical PK for this selection, or ``None``.

        Raises
        ------
        DuplicateSelectionError
            If a row with the same identity has a non-deterministic id (a raw
            insert bypassing :meth:`insert_selection`).
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            existing_selection_pk,
        )

        return existing_selection_pk(
            list((cls & master_row).fetch("motion_corrected_recording_id")),
            deterministic_id,
            pk_field="motion_corrected_recording_id",
            bypass_message=lambda bypassed: (
                "MotionCorrectedRecordingSelection has rows for "
                f"{master_row} whose motion_corrected_recording_id is not the "
                f"deterministic id {deterministic_id}: {bypassed}. Drop them "
                "and re-insert via insert_selection."
            ),
        )


class MotionCorrectedFetched(NamedTuple):
    """DB inputs of :meth:`MotionCorrectedRecording.make_compute`.

    Attributes
    ----------
    selection : dict
        The ``MotionCorrectedRecordingSelection`` row.
    interpolation_params : dict
        The interpolation recipe's ``params`` blob.
    estimate : dict
        The saved estimate's ``_ESTIMATE_APPLICATION_FIELDS``.
    source_content_hash : str
        ``content_hash`` of the source trace artifact (unchanged since the
        estimate was selected).
    source_path : str
        Absolute path of the source's analysis NWB (rebuilt if missing).
    source_electrical_series_path : str
        The source's stored ``electrical_series_path``.
    nwb_file_name : str
        The parent NWB the source artifact belongs to.
    source_kind : str
        ``"recording"`` or ``"concatenated_recording"``.
    source_key : dict
        The source row's primary key.
    """

    selection: dict
    interpolation_params: dict
    estimate: dict
    source_content_hash: str
    source_path: str
    source_electrical_series_path: str
    nwb_file_name: str
    source_kind: str
    source_key: dict


class MotionCorrectedComputed(NamedTuple):
    """``MotionCorrectedRecording`` row fields plus the parent NWB name."""

    analysis_file_name: str
    object_id: str
    content_hash: str
    source_content_hash: str
    n_samples: int
    n_channels: int
    sampling_frequency: float
    channel_ids: list
    removed_channel_ids: list
    channel_locations: np.ndarray
    statistics_spans: np.ndarray
    continuity_spans: np.ndarray
    nwb_file_name: str

    def staged_outputs(self) -> StagedOutputs:
        """The staged analysis file ``make_insert`` registers."""
        return StagedOutputs(analysis_file_names=(self.analysis_file_name,))


def _series_filtering(abs_path: str, electrical_series_path: str) -> str:
    """The ``filtering`` attribute of a persisted ``ElectricalSeries``."""
    import h5py

    with h5py.File(abs_path, "r") as handle:
        value = handle[electrical_series_path].attrs.get("filtering", "")
    return value.decode() if isinstance(value, bytes) else str(value)


@schema
class MotionCorrectedRecording(
    StagedOutputCleanupMixin, SpyglassMixin, dj.Computed
):
    """A saved motion estimate applied to the traces it was estimated from.

    The source is silenced outside the statistics spans, interpolated on
    the estimation clock, and silenced again. The corrected, masked,
    unwhitened traces keep the source's timestamps, sample count and rate.

    ``channel_locations`` are the unmoved source positions of the kept
    channels. :meth:`get_recording` rebuilds a missing file from the saved
    motion (never re-estimating) and installs it only if its content hash
    matches.
    """

    _populate_key_lock = True

    definition = """
    -> MotionCorrectedRecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255)
    object_id: varchar(72)
    content_hash: char(64)             # content fingerprint of the corrected traces, timestamps, geometry and scaling
    source_content_hash: char(64)      # content_hash of the source trace artifact the estimate was computed from
    n_samples: bigint                  # frames; equal to the source's
    n_channels: int                    # output channels
    sampling_frequency: double         # Hz; equal to the source's
    channel_ids: longblob              # output channel ids in order; each is the electrode_id the series references
    removed_channel_ids: longblob      # source channel ids border_mode remove_channels dropped, in source order; empty otherwise
    channel_locations: longblob        # (n_channels, 2) float64 unmoved source contact positions in um
    statistics_spans: longblob         # (n, 2) int64 copy of the estimate's statistics spans; frames outside them are zero
    continuity_spans: longblob         # (n, 2) int64 copy of the estimate's continuity spans
    """

    _parallel_make = True

    def make_fetch(self, key) -> MotionCorrectedFetched:
        """Resolve the selection, recipe, saved estimate and source artifact.

        Rebuilds a missing source NWB.

        Raises
        ------
        ValueError
            If the source's ``content_hash`` changed since the estimate was
            selected.
        """
        from spyglass.spikesorting.v2._storage.rebuilds import (
            ensure_artifact_file,
        )

        selection = (MotionCorrectedRecordingSelection & key).fetch1()
        interpolation_params = (
            MotionInterpolationParameters
            & {
                "motion_interpolation_params_name": selection[
                    "motion_interpolation_params_name"
                ]
            }
        ).fetch1("params")
        estimate_key = {"motion_estimate_id": selection["motion_estimate_id"]}
        estimate = (
            (MotionEstimate & estimate_key)
            .proj(*_ESTIMATE_APPLICATION_FIELDS)
            .fetch1()
        )
        estimate.pop("motion_estimate_id")
        table, lineage, source_row = _estimate_source(
            selection["motion_estimate_id"]
        )
        source_path = ensure_artifact_file(
            table, lineage.key, source_row["analysis_file_name"]
        )
        nwb_file_name = (
            AnalysisNwbfile
            & {"analysis_file_name": source_row["analysis_file_name"]}
        ).fetch1("nwb_file_name")
        return MotionCorrectedFetched(
            selection=selection,
            interpolation_params=interpolation_params,
            estimate=estimate,
            source_content_hash=source_row["content_hash"],
            source_path=source_path,
            source_electrical_series_path=source_row["electrical_series_path"],
            nwb_file_name=nwb_file_name,
            source_kind=lineage.kind,
            source_key=dict(lineage.key),
        )

    def make_compute(
        self,
        key,
        selection,
        interpolation_params,
        estimate,
        source_content_hash,
        source_path,
        source_electrical_series_path,
        nwb_file_name,
        source_kind,
        source_key,
        *,
        allow_spikeinterface_version_change: bool = False,
    ) -> MotionCorrectedComputed:
        """Apply the saved motion and write the staged artifact.

        Uses only ``make_fetch`` inputs; the one DB access is staging the
        output file.

        Parameters
        ----------
        allow_spikeinterface_version_change : bool, optional
            Skip the SpikeInterface-version staleness check. Used by
            :meth:`_rebuild_nwb_artifact`, which verifies the content hash.

        Raises
        ------
        ValueError
            On a stale selection (interpolation recipe resolution,
            SpikeInterface version or application algorithm changed), a
            source whose frames, rate or channels differ from the estimate's,
            or an application failure (see
            ``_motion.apply_motion_on_estimation_clock``).
        """
        from spyglass.spikesorting.v2._motion import estimation as _motion
        from spyglass.spikesorting.v2._recording.geometry import (
            flatten_planar_geometry,
        )
        from spyglass.spikesorting.v2._storage.nwb import (
            read_recording_nwb,
            write_nwb_artifact,
        )
        from spyglass.spikesorting.v2._recording.restriction import (
            _LazyRecordingTimestamps,
        )

        resolved = _motion.resolve_interpolation_params(interpolation_params)
        stale = _motion_compute.stale_corrected_selection_fields(
            selection,
            resolved,
            check_spikeinterface_version=(
                not allow_spikeinterface_version_change
            ),
        )
        if stale:
            raise ValueError(
                f"MotionCorrectedRecording {key}: the selection is stale "
                f"({'; '.join(stale)}). Recreate it with "
                "MotionCorrectedRecordingSelection.insert_selection."
            )

        source = read_recording_nwb(
            source_path, electrical_series_path=source_electrical_series_path
        )
        source.annotate(is_filtered=True)
        n_samples = int(source.get_num_samples())
        sampling_frequency = float(source.get_sampling_frequency())
        _motion_compute.assert_source_matches_estimate(
            key, source, estimate, n_samples, sampling_frequency
        )
        timestamps = _LazyRecordingTimestamps(source, 0, n_samples)
        source_filtering = _series_filtering(
            source_path, source_electrical_series_path
        )

        # The estimate stored the flattened positions it was computed on.
        flatten_planar_geometry(source)
        _motion_compute.assert_geometry_matches_estimate(key, source, estimate)

        clock = _estimation_clock_of(estimate)
        statistics = np.asarray(
            estimate["statistics_spans"], dtype=np.int64
        ).reshape(-1, 2)
        applied = _motion.apply_motion_on_estimation_clock(
            source,
            _motion.motion_from_storage_dict(estimate["motion"]),
            clock=clock,
            statistics_spans=statistics,
            resolved_interpolation=resolved,
        )
        corrected = applied.recording
        if int(corrected.get_num_samples()) != n_samples:
            raise ValueError(
                f"MotionCorrectedRecording {key}: interpolation changed the "
                f"sample count ({corrected.get_num_samples()} != {n_samples})."
            )
        channel_ids = corrected.channel_ids.tolist()
        interpolation_text = ", ".join(
            f"{name}={resolved[name]}" for name in sorted(resolved)
        )
        provenance_tables = _motion_compute.motion_correction_provenance_tables(
            key,
            selection,
            resolved,
            source_content_hash,
            applied,
            source_kind,
            source_key,
            statistics,
            clock,
            source_path,
        )
        analysis_file_name, object_id, content_hash = write_nwb_artifact(
            corrected,
            nwb_file_name,
            timestamps_override=timestamps,
            filtering_description=(
                f"{source_filtering}; motion corrected with saved estimate "
                f"{selection['motion_estimate_id']} ({interpolation_text}); "
                "frames outside the statistics spans silenced; unwhitened"
            ),
            description=(
                "Motion-corrected preprocessed recording from "
                f"{nwb_file_name} for spike sorting"
            ),
            provenance_tables=provenance_tables,
        )
        return MotionCorrectedComputed(
            analysis_file_name=analysis_file_name,
            object_id=object_id,
            content_hash=content_hash,
            source_content_hash=str(source_content_hash),
            n_samples=n_samples,
            n_channels=len(channel_ids),
            sampling_frequency=sampling_frequency,
            channel_ids=channel_ids,
            removed_channel_ids=applied.removed_channel_ids,
            channel_locations=np.asarray(
                corrected.get_channel_locations(), dtype=np.float64
            ),
            statistics_spans=statistics,
            continuity_spans=np.asarray(clock.spans, dtype=np.int64),
            nwb_file_name=nwb_file_name,
        )

    def make_insert(self, key, *computed) -> None:
        """Register the staged artifact and insert the row atomically.

        A direct call (outside ``populate``) must remove a failed attempt's
        staged file itself.
        """
        from spyglass.spikesorting.v2.recording import _ELECTRICAL_SERIES_PATH

        row = MotionCorrectedComputed(*computed)._asdict()
        nwb_file_name = row.pop("nwb_file_name")
        with self._safe_context():
            AnalysisNwbfile().add(nwb_file_name, row["analysis_file_name"])
            self.insert1(
                {
                    **key,
                    **row,
                    "electrical_series_path": _ELECTRICAL_SERIES_PATH,
                }
            )

    def get_recording(self, key: dict):
        """Return the corrected recording, rebuilding a missing file first.

        As ``Recording.get_recording``: never deletes the row; a rebuild is
        installed only if its content hash matches.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``MotionCorrectedRecording`` row.

        Returns
        -------
        si.BaseRecording
            The corrected, masked, unwhitened recording, annotated
            ``is_filtered=True``.
        """
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces
        from spyglass.spikesorting.v2._storage.rebuilds import stored_traces

        return read_stored_traces(
            stored_traces(type(self), key, (self & key).fetch1())
        )

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild a missing corrected artifact from the saved motion.

        Kept on the class because ``ensure_artifact_file`` calls it on every
        trace-artifact table.
        """
        from spyglass.spikesorting.v2._storage import (
            rebuilds as _recording_rebuilds,
        )

        return _recording_rebuilds.rebuild_motion_corrected_artifact(self, key)
