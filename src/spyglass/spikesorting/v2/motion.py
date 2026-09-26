"""Motion estimation for spike-sorting v2, independent of concatenation.

Tables:
    MotionEstimationParameters -- Named, validated SpikeInterface motion recipes.
    MotionEstimateSelection    -- One source (a Recording or an uncorrected
                                  ConcatenatedRecording), an optional artifact
                                  mask, and a recipe, content-addressed.
        .RecordingSource             -- single-recording source.
        .ConcatenatedRecordingSource -- concatenated-recording source.
        .ArtifactDetectionSource     -- optional mask (single recording only).
    MotionEstimate             -- The saved SpikeInterface ``Motion`` (on the
                                  source's estimation clock) with that clock,
                                  its resolved configuration and diagnostics.
    MotionInterpolationParameters -- Named interpolation recipes, every
                                  ``interpolate_motion`` argument explicit.
    MotionCorrectionParameters -- Public recipe: one estimation recipe and one
                                  interpolation recipe.
    MotionCorrectedRecordingSelection -- One saved estimate and an
                                  interpolation recipe, content-addressed.
    MotionCorrectedRecording   -- The corrected, masked, unwhitened traces
                                  written with the source's own timestamps.

No existing table populates or reads these rows yet. The DB-free computation
(parameter resolution, the estimation adapter, the ``Motion`` serialization,
applying a saved estimate) lives in ``_motion``.
"""

from __future__ import annotations

import uuid
from typing import NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._params.motion_estimation import (
    MOTION_ESTIMATION_SCHEMA_VERSION,
    MotionEstimationParamsSchema,
)
from spyglass.spikesorting.v2._params.motion_interpolation import (
    MOTION_INTERPOLATION_SCHEMA_VERSION,
    MotionInterpolationParamsSchema,
)
from spyglass.spikesorting.v2._recipe_catalog import (
    motion_correction_default_contents,
    motion_estimation_default_contents,
    motion_interpolation_default_contents,
)
from spyglass.spikesorting.v2._source_resolution import SourceLineage
from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,
    ConcatenatedRecordingSelection,
)
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    reject_duplicate_parameter_content,
    transaction_or_noop,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("spikesorting_v2_motion")


@schema
class MotionEstimationParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Named motion-estimation recipes: a SpikeInterface preset plus overrides.

    The ``params`` blob is validated by :class:`MotionEstimationParamsSchema`
    and every row must resolve
    (``_motion.resolve_estimation_params``): an override key that is not a
    parameter of the selected SpikeInterface method is rejected at insert.
    ``insert_default`` ships ``dredge_v1`` and ``dredge_fast_v1``; the
    ``rigid_fast`` preset is allowed but ships no row. No recipe here is
    validated for a particular probe.

    ``job_kwargs`` is the optional per-row SpikeInterface job-kwargs blob
    (``n_jobs``, ``chunk_duration``, ...) for the detect-and-localize pass. It
    is not part of the estimate's identity. The noise-estimate seed is the
    identity-bearing ``noise_levels_seed`` params field, so a ``random_seed``
    job kwarg is rejected.
    """

    definition = f"""
    motion_estimation_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={MOTION_ESTIMATION_SCHEMA_VERSION}: int
    job_kwargs=null: blob  # SI job kwargs for detection and localization
    """

    _DEFAULT_CONTENTS: tuple = motion_estimation_default_contents()

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Insert one validated motion-estimation parameter row."""
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert motion-estimation parameter rows after validation.

        Each row's blob is validated, resolved against the installed
        SpikeInterface (unknown override keys raise), and checked by the
        duplicate-content guard. ``allow_duplicate_params=True`` opts out of
        that guard; see ``reject_duplicate_parameter_content``.
        """
        from spyglass.spikesorting.v2._motion import resolve_estimation_params

        def _resolve_and_check_job_kwargs(row, _schema_cls):
            resolve_estimation_params(row["params"])
            if "random_seed" in (row.get("job_kwargs") or {}):
                raise ValueError(
                    "MotionEstimationParameters.job_kwargs must not contain "
                    "'random_seed': the noise-estimate seed is the identity-"
                    "bearing params field 'noise_levels_seed'."
                )

        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda _row: MotionEstimationParamsSchema,
            table_name="MotionEstimationParameters",
            per_row_hook=_resolve_and_check_job_kwargs,
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="MotionEstimationParameters",
            name_attr="motion_estimation_params_name",
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert the shipped motion-estimation recipes if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class MotionInterpolationParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Named motion-interpolation recipes for applying a saved estimate.

    The ``params`` blob is validated by
    :class:`MotionInterpolationParamsSchema`: ``border_mode``
    (``remove_channels`` or ``force_extrapolate``),
    ``spatial_interpolation_method``, ``sigma_um``, ``p`` and ``num_closest``
    are all required, so no SpikeInterface default is ever relied on.
    ``insert_default`` ships ``kriging_force_extrapolate_v1`` (the
    interpolation of the ``dredge`` / ``dredge_fast`` presets) and
    ``kriging_remove_channels_v1``.
    """

    definition = f"""
    motion_interpolation_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={MOTION_INTERPOLATION_SCHEMA_VERSION}: int
    """

    _DEFAULT_CONTENTS: tuple = motion_interpolation_default_contents()

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Insert one validated motion-interpolation parameter row."""
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert motion-interpolation parameter rows after validation.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard; see ``reject_duplicate_parameter_content``.
        """
        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda _row: MotionInterpolationParamsSchema,
            table_name="MotionInterpolationParameters",
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="MotionInterpolationParameters",
            name_attr="motion_interpolation_params_name",
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert the shipped motion-interpolation recipes if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class MotionCorrectionParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Named public motion-correction recipes: estimation plus interpolation.

    Each row composes one ``MotionEstimationParameters`` row and one
    ``MotionInterpolationParameters`` row. The two stages keep their own
    identities: two recipes that share an estimation row reuse the same saved
    ``MotionEstimate`` and differ only in the corrected recording.
    ``insert_default`` ships ``dredge_v1`` and ``dredge_fast_v1``, each with
    its preset's interpolation. Neither is validated for a probe.
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


def _assert_concat_tables_current() -> None:
    """Refuse concat tables whose live heading predates motion's removal."""
    from spyglass.spikesorting.v2._motion import assert_concat_schema_current

    assert_concat_schema_current(
        ConcatenatedRecording.heading.names,
        ConcatenatedRecordingSelection.heading.names,
    )


@schema
class MotionEstimateSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One motion estimate to compute: a source, an optional mask, a recipe.

    Exactly one of ``RecordingSource`` / ``ConcatenatedRecordingSource``
    exists per row. An ``ArtifactDetectionSource`` part pins the artifact mask
    of a single-recording source; a concatenated recording carries its own
    member masks, so it never has one. ``motion_estimate_id`` is derived from
    the source kind and id, the source artifact's ``content_hash``, the
    artifact detection (when present), the recipe name, the resolved
    configuration's hash, the SpikeInterface version and the estimation
    algorithm version, so a change to any of them selects a new estimate.
    Create rows with :meth:`insert_selection`.
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
            a concat source, a missing recipe or unpopulated source, an
            artifact detection that does not belong to the recording, concat
            tables that predate motion's removal from concatenation, or a
            mismatched explicit ``motion_estimate_id``.
        DuplicateSelectionError
            If a matching row has a non-deterministic ``motion_estimate_id``.
        SchemaBypassError
            If the deterministic master exists with other source parts.
        """
        import spikeinterface

        from spyglass.spikesorting.v2._motion import (
            MOTION_ALGORITHM_VERSION,
            motion_estimate_identity_payload,
            resolve_estimation_params,
            resolved_params_hash,
        )
        from spyglass.spikesorting.v2._selection_identity import (
            deterministic_id,
        )
        from spyglass.spikesorting.v2.artifact import (
            assert_artifact_detection_covers_recording,
        )
        from spyglass.spikesorting.v2.utils import _ensure_lookup_row_exists

        caller = "MotionEstimateSelection.insert_selection"
        extra = sorted(set(key) - cls._INPUT_FIELDS)
        if extra:
            raise ValueError(
                f"{caller} received unknown field(s) {extra}; pass only "
                f"{sorted(cls._INPUT_FIELDS)}."
            )
        recording_id = key.get("recording_id")
        concat_recording_id = key.get("concat_recording_id")
        if (recording_id is None) == (concat_recording_id is None):
            raise ValueError(
                f"{caller}: pass exactly one of recording_id or "
                "concat_recording_id."
            )
        artifact_detection_id = key.get("artifact_detection_id")
        if artifact_detection_id is not None:
            artifact_detection_id = uuid.UUID(str(artifact_detection_id))
            if concat_recording_id is not None:
                raise ValueError(
                    f"{caller}: a concatenated recording carries its own "
                    "member artifact masks; artifact_detection_id applies to "
                    "a single recording only."
                )
        params_name = key.get("motion_estimation_params_name")
        if params_name is None:
            raise ValueError(
                f"{caller}: motion_estimation_params_name is required."
            )

        if concat_recording_id is not None:
            _assert_concat_tables_current()
            source_kind = "concatenated_recording"
            source_key = {
                "concat_recording_id": uuid.UUID(str(concat_recording_id))
            }
            source_part = cls.ConcatenatedRecordingSource
        else:
            source_kind = "recording"
            source_key = {"recording_id": uuid.UUID(str(recording_id))}
            source_part = cls.RecordingSource
        source_table = _SOURCE_TABLES[source_kind]

        params_key = {"motion_estimation_params_name": params_name}
        _ensure_lookup_row_exists(
            MotionEstimationParameters,
            params_key,
            helper_name=caller,
            insert_default_path="MotionEstimationParameters.insert_default()",
        )
        resolved_hash = resolved_params_hash(
            resolve_estimation_params(
                (MotionEstimationParameters & params_key).fetch1("params")
            )
        )
        content_hashes = (source_table & source_key).fetch("content_hash")
        if len(content_hashes) == 0:
            raise ValueError(
                f"{caller}: {source_key} is not in {source_table.__name__}. "
                "Populate it before selecting a motion estimate on it."
            )
        if source_kind == "recording":
            assert_artifact_detection_covers_recording(
                recording_id=source_key["recording_id"],
                artifact_detection_id=artifact_detection_id,
                caller=caller,
            )

        master_row = {
            "motion_estimation_params_name": params_name,
            "resolved_params_hash": resolved_hash,
            "spikeinterface_version": spikeinterface.__version__,
            "motion_algorithm_version": MOTION_ALGORITHM_VERSION,
            "source_content_hash": str(content_hashes[0]),
        }
        motion_estimate_id = deterministic_id(
            "motion_estimate",
            motion_estimate_identity_payload(
                source_kind=source_kind,
                source_id=next(iter(source_key.values())),
                source_content_hash=master_row["source_content_hash"],
                artifact_detection_id=artifact_detection_id,
                motion_estimation_params_name=params_name,
                resolved_params_hash=resolved_hash,
                spikeinterface_version=master_row["spikeinterface_version"],
                motion_algorithm_version=MOTION_ALGORITHM_VERSION,
            ),
        )
        explicit = key.get("motion_estimate_id")
        if explicit is not None and uuid.UUID(str(explicit)) != (
            motion_estimate_id
        ):
            raise ValueError(
                f"{caller}: motion_estimate_id {explicit} does not match the "
                f"id derived from this selection ({motion_estimate_id})."
            )

        def _existing():
            return cls._find_existing_pk(
                master_row,
                source_part,
                source_key,
                artifact_detection_id,
                motion_estimate_id,
            )

        existing = _existing()
        if existing is not None:
            return existing
        return cls._insert_rows(
            {"motion_estimate_id": motion_estimate_id, **master_row},
            source_part,
            {"motion_estimate_id": motion_estimate_id, **source_key},
            artifact_detection_id,
            refetch=_existing,
        )

    @classmethod
    def _insert_rows(
        cls,
        master_row,
        source_part,
        source_row,
        artifact_detection_id,
        *,
        refetch,
    ) -> dict:
        """Insert the master and its parts atomically, locked against deletion.

        The same protocol as ``SortingSelection.insert_selection``: an
        artifact-bound selection must own its transaction (the advisory lock
        that serializes it against the detection's deletion is released when
        this returns), the artifact's merge id is resolved before the
        transaction, and a duplicate-key race refetches the winner.
        """
        from contextlib import ExitStack

        from spyglass.spikesorting.v2._db_locking import required_advisory_lock
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        art_merge_id = None
        if artifact_detection_id is not None:
            if cls.connection.in_transaction:
                raise ValueError(
                    "MotionEstimateSelection.insert_selection: refusing to link "
                    "an artifact detection while a caller-owned transaction is "
                    "open; the lock that serializes it against the detection's "
                    "deletion is released before your transaction commits. "
                    "Call insert_selection outside the transaction."
                )
            art_key = {"artifact_detection_id": artifact_detection_id}
            try:
                art_merge_id = ArtifactDetectionOutput.get_merge_id(art_key)
            except KeyError:
                ArtifactDetectionOutput.insert_detection(art_key)
                art_merge_id = ArtifactDetectionOutput.get_merge_id(art_key)

        with ExitStack() as art_lock:
            if art_merge_id is not None:
                art_lock.enter_context(
                    required_advisory_lock(
                        ArtifactDetectionOutput,
                        {"artifact_detection_id": artifact_detection_id},
                    )
                )
            try:
                with transaction_or_noop(cls.connection):
                    cls.insert1(master_row, allow_direct_insert=True)
                    source_part.insert1(source_row)
                    if art_merge_id is not None:
                        cls.ArtifactDetectionSource.insert1(
                            {
                                "motion_estimate_id": master_row[
                                    "motion_estimate_id"
                                ],
                                "artifact_detection_merge_id": art_merge_id,
                            }
                        )
            except dj.errors.DuplicateError as exc:
                existing = refetch()
                if existing is not None:
                    return existing
                raise SchemaBypassError(
                    "MotionEstimateSelection master "
                    f"{master_row['motion_estimate_id']} exists but its "
                    "source/artifact parts do not match this selection (a "
                    "raw-insert orphan). Drop the orphan master and use "
                    "insert_selection()."
                ) from exc
        return {"motion_estimate_id": master_row["motion_estimate_id"]}

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
        bypassed or legacy row.

        Raises
        ------
        DuplicateSelectionError
            If a match has a non-deterministic ``motion_estimate_id``.
        """
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
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
        bypassed = [mid for mid in matching if mid != deterministic_id]
        if bypassed:
            raise DuplicateSelectionError(
                "MotionEstimateSelection has rows for "
                f"{source_key} / {master_row['motion_estimation_params_name']} "
                "whose motion_estimate_id is not the deterministic id "
                f"{deterministic_id}: {bypassed}. Drop them and re-insert via "
                "insert_selection."
            )
        return {"motion_estimate_id": deterministic_id} if matching else None

    @classmethod
    def resolve_artifact_detection(cls, key: dict):
        """Return the pinned ``artifact_detection_id``, or ``None``.

        Raises
        ------
        SchemaBypassError
            If more than one ``ArtifactDetectionSource`` row exists.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        rows = (cls.ArtifactDetectionSource & master_key).fetch(
            "artifact_detection_merge_id"
        )
        if len(rows) > 1:
            raise SchemaBypassError(
                f"MotionEstimateSelection {master_key} has {len(rows)} "
                "ArtifactDetectionSource rows; expected zero or one."
            )
        if len(rows) == 0:
            return None
        return ArtifactDetectionOutput.resolve_artifact_detection_id(rows[0])

    @classmethod
    def resolve_source(cls, key: dict) -> SourceLineage:
        """Return the selection's source and pinned artifact detection.

        Raises
        ------
        SchemaBypassError
            If the row has zero or two source parts, or a concat source
            carries an artifact detection (a raw insert bypassing
            :meth:`insert_selection`).
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        recording_rows = (cls.RecordingSource & master_key).fetch(
            "recording_id"
        )
        concat_rows = (cls.ConcatenatedRecordingSource & master_key).fetch(
            "concat_recording_id"
        )
        if len(recording_rows) + len(concat_rows) != 1:
            raise SchemaBypassError(
                f"MotionEstimateSelection {master_key} has "
                f"{len(recording_rows) + len(concat_rows)} source part rows; "
                "expected exactly one. Use insert_selection()."
            )
        artifact_detection_id = cls.resolve_artifact_detection(master_key)
        if len(recording_rows):
            return SourceLineage(
                kind="recording",
                key={"recording_id": recording_rows[0]},
                artifact_detection_id=artifact_detection_id,
            )
        if artifact_detection_id is not None:
            raise SchemaBypassError(
                f"MotionEstimateSelection {master_key} pairs a concatenated "
                "recording with an ArtifactDetectionSource; a concat source "
                "owns its member masks. Drop the stray part row."
            )
        return SourceLineage(
            kind="concatenated_recording",
            key={"concat_recording_id": concat_rows[0]},
            artifact_detection_id=None,
        )


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
    """The ``MotionEstimate`` row fields :meth:`MotionEstimate.make_compute`
    returns (see the table definition for each column)."""

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
    from spyglass.spikesorting.v2._motion import estimation_clock_from_blob

    return estimation_clock_from_blob(
        {
            field: row[column]
            for field, column in _ESTIMATION_CLOCK_COLUMNS.items()
        }
    )


@schema
class MotionEstimate(SpyglassMixin, dj.Computed):
    """A saved motion estimate with its resolved configuration and evidence.

    ``motion`` is the SpikeInterface ``Motion`` as a blob dict
    (``_motion.motion_to_storage_dict``); :meth:`get_motion` rebuilds it.
    ``resolved_params`` is exactly the configuration passed to
    SpikeInterface. The spans record which frames were one uninterrupted
    acquisition and which samples were valid evidence; the diagnostics are
    counts, never peak arrays.

    Every source is estimated once, on its *estimation clock*: within each
    continuity span time advances by ``1 / fs`` per frame from the span's
    ``estimation_start_s``, and the real gap between two spans (an
    acquisition gap or a concatenation member join) is kept up to the
    recipe's ``max_gap_s`` (``_motion.build_estimation_clock``); a gap runs
    from one sample after a span's last timestamp to the next span's first
    timestamp, both on the source's own clock. All spans
    therefore share one reference frame. The ``motion`` bins are on that
    clock; :meth:`get_estimation_clock` returns the time map and
    :meth:`get_displacement_on_source_clock` maps the bins back to source
    time for inspection.

    Populated explicitly (``MotionEstimate.populate(selection_key)``); no
    other table populates or reads it. Estimation runs outside the DB
    transaction (``make_fetch`` / ``make_compute`` / ``make_insert``), and
    ``make_compute`` reads only the files ``make_fetch`` resolved. A recipe
    whose resolution, SpikeInterface version or algorithm version changed
    since selection raises: recreate the selection.
    """

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
    n_peaks_kept: int                 # peaks whose localization window lies in one statistics span
    peaks_per_temporal_bin: longblob  # (n_temporal_bins,) int64 kept peaks per temporal bin
    peaks_per_continuity_span: longblob  # (n_spans,) int64 kept peaks per continuity span; 0 marks a span with no evidence
    noise_levels: longblob            # (n_channels,) float64 detection noise in recording units
    input_fingerprint: char(64)       # SHA-256 of source content, spans, channels and configuration
    """

    def make_fetch(self, key) -> MotionEstimateFetched:
        """Resolve the selection, recipe, source artifact and mask.

        Rebuilds a missing source NWB through the owning table's own verified
        self-heal. Refuses a source whose ``content_hash`` changed since
        selection, and concat tables that predate motion's removal from
        concatenation. A concat source's continuity spans and their start
        times come from its row.
        """
        from spyglass.spikesorting.v2._artifact_intervals import (
            read_artifact_removed_intervals,
        )
        from spyglass.spikesorting.v2._recording_nwb import (
            ensure_artifact_file,
        )
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        lineage = MotionEstimateSelection.resolve_source(key)
        selection = (MotionEstimateSelection & key).fetch1()
        params, job_kwargs = (
            MotionEstimationParameters
            & {
                "motion_estimation_params_name": selection[
                    "motion_estimation_params_name"
                ]
            }
        ).fetch1("params", "job_kwargs")
        if lineage.kind == "concatenated_recording":
            _assert_concat_tables_current()
        table = _SOURCE_TABLES[lineage.kind]
        source_row = (table & lineage.key).fetch1()
        if source_row["content_hash"] != selection["source_content_hash"]:
            raise ValueError(
                f"MotionEstimate: {table.__name__} {lineage.key} changed "
                "since this estimate was selected (content_hash "
                f"{source_row['content_hash']} != "
                f"{selection['source_content_hash']}). Select a new estimate "
                "with MotionEstimateSelection.insert_selection."
            )
        source_path = ensure_artifact_file(
            table, lineage.key, source_row["analysis_file_name"]
        )

        artifact_valid_times = None
        if lineage.artifact_detection_id is not None:
            nwb_file_name = (RecordingSelection & lineage.key).fetch1(
                "nwb_file_name"
            )
            by_nwb = read_artifact_removed_intervals(
                {"artifact_detection_id": lineage.artifact_detection_id},
                as_dict=True,
            )
            if nwb_file_name not in by_nwb:
                raise ValueError(
                    "MotionEstimate: artifact-removed intervals for "
                    f"{nwb_file_name!r} not found for artifact_detection_id="
                    f"{lineage.artifact_detection_id}; the detection may be "
                    "partially deleted."
                )
            artifact_valid_times = by_nwb[nwb_file_name]

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

        Re-resolves the recipe and requires its hash, the SpikeInterface
        version and the algorithm version to equal the selection's. Reads the
        source traces. For a single recording, reads the continuity spans and
        each span's first and last timestamp from the persisted timestamps,
        silences the artifact ranges and computes the statistics spans as the
        sort stage does; for a concat, reads the same spans, timestamps and
        statistics spans from its row. Builds the estimation clock with the
        recipe's ``max_gap_s`` and runs ``_motion.estimate_motion_in_spans``
        once.

        Raises
        ------
        ValueError
            On a stale selection, spans out of acquisition order, or any
            estimation failure (see ``estimate_motion_in_spans``).
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2 import _motion
        from spyglass.spikesorting.v2._recording_nwb import read_recording_nwb
        from spyglass.spikesorting.v2._sorting_artifact_mask import (
            artifact_frame_ranges,
            continuity_from_timestamps,
            silence_frame_ranges,
            statistics_spans,
        )

        resolved = _motion.resolve_estimation_params(params)
        resolved_hash = _motion.resolved_params_hash(resolved)
        stale = [
            f"{name} {now!r} != selected {then!r}"
            for name, now, then in (
                (
                    "resolved configuration hash",
                    resolved_hash,
                    selection["resolved_params_hash"],
                ),
                (
                    "SpikeInterface version",
                    si.__version__,
                    selection["spikeinterface_version"],
                ),
                (
                    "motion algorithm version",
                    _motion.MOTION_ALGORITHM_VERSION,
                    selection["motion_algorithm_version"],
                ),
            )
            if now != then
        ]
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
        if lineage.kind == "recording":
            continuity, continuity_start_s, continuity_end_s = (
                continuity_from_timestamps(recording)
            )
            excluded = []
            if artifact_valid_times is not None:
                excluded = artifact_frame_ranges(
                    recording,
                    artifact_valid_times,
                    artifact_detection_id=lineage.artifact_detection_id,
                    recording_id=lineage.key["recording_id"],
                )
                if excluded:
                    recording = silence_frame_ranges(recording, excluded)
            statistics = statistics_spans(n_samples, excluded, continuity)
        else:
            continuity = _motion.normalize_spans(source_row["continuity_spans"])
            continuity_start_s = source_row["continuity_start_s"]
            continuity_end_s = source_row["continuity_end_s"]
            statistics = _motion.normalize_spans(source_row["statistics_spans"])

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
        from spyglass.spikesorting.v2._motion import motion_from_storage_dict

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
        from spyglass.spikesorting.v2._motion import (
            displacement_on_source_clock,
        )

        return displacement_on_source_clock(
            self.get_motion(key), self.get_estimation_clock(key)
        )


#: ``MotionEstimate`` columns the corrected recording is computed from.
_ESTIMATE_APPLICATION_FIELDS = (
    "motion",
    "n_samples",
    "sampling_frequency",
    "continuity_spans",
    "continuity_start_s",
    "continuity_end_s",
    "estimation_start_s",
    "statistics_spans",
    "channel_ids",
    "channel_locations",
)


def _estimate_source(motion_estimate_id) -> tuple:
    """The source table, lineage and live row of a saved estimate.

    Raises
    ------
    ValueError
        If the source artifact's ``content_hash`` changed since the estimate
        was selected: the saved motion no longer describes those traces.
    """
    estimate_key = {"motion_estimate_id": motion_estimate_id}
    lineage = MotionEstimateSelection.resolve_source(estimate_key)
    if lineage.kind == "concatenated_recording":
        _assert_concat_tables_current()
    table = _SOURCE_TABLES[lineage.kind]
    source_row = (table & lineage.key).fetch1()
    selected_hash = (MotionEstimateSelection & estimate_key).fetch1(
        "source_content_hash"
    )
    if source_row["content_hash"] != selected_hash:
        raise ValueError(
            f"{table.__name__} {lineage.key} changed since motion estimate "
            f"{motion_estimate_id} was selected (content_hash "
            f"{source_row['content_hash']} != {selected_hash}); the saved "
            "motion no longer describes these traces. Select and populate a "
            "new estimate."
        )
    return table, lineage, source_row


@schema
class MotionCorrectedRecordingSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One corrected recording to compute: a saved estimate and a recipe.

    ``motion_corrected_recording_id`` is derived from the
    ``motion_estimate_id`` (which already carries the source, its content,
    the mask and the estimation recipe), the interpolation recipe name, its
    resolved configuration's hash, the SpikeInterface version that applies
    it and the application algorithm version. Changing only the interpolation recipe
    therefore selects a new corrected recording on the same estimate. Create
    rows with :meth:`insert_selection`.
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
        import spikeinterface

        from spyglass.spikesorting.v2._motion import (
            MOTION_INTERPOLATION_ALGORITHM_VERSION,
            motion_corrected_identity_payload,
            resolve_interpolation_params,
            resolved_params_hash,
        )
        from spyglass.spikesorting.v2._selection_identity import (
            deterministic_id,
        )
        from spyglass.spikesorting.v2.utils import _ensure_lookup_row_exists

        caller = "MotionCorrectedRecordingSelection.insert_selection"
        extra = sorted(set(key) - cls._INPUT_FIELDS)
        if extra:
            raise ValueError(
                f"{caller} received unknown field(s) {extra}; pass only "
                f"{sorted(cls._INPUT_FIELDS)}."
            )
        missing = [
            name
            for name in (
                "motion_estimate_id",
                "motion_interpolation_params_name",
            )
            if key.get(name) is None
        ]
        if missing:
            raise ValueError(f"{caller}: {missing} are required.")
        motion_estimate_id = uuid.UUID(str(key["motion_estimate_id"]))
        params_name = key["motion_interpolation_params_name"]
        params_key = {"motion_interpolation_params_name": params_name}
        _ensure_lookup_row_exists(
            MotionInterpolationParameters,
            params_key,
            helper_name=caller,
            insert_default_path=(
                "MotionInterpolationParameters.insert_default()"
            ),
        )
        if not (MotionEstimate & {"motion_estimate_id": motion_estimate_id}):
            raise ValueError(
                f"{caller}: motion estimate {motion_estimate_id} is not "
                "populated. Populate MotionEstimate before selecting a "
                "corrected recording on it."
            )
        _estimate_source(motion_estimate_id)
        resolved_hash = resolved_params_hash(
            resolve_interpolation_params(
                (MotionInterpolationParameters & params_key).fetch1("params")
            )
        )
        master_row = {
            "motion_estimate_id": motion_estimate_id,
            "motion_interpolation_params_name": params_name,
            "resolved_params_hash": resolved_hash,
            "spikeinterface_version": spikeinterface.__version__,
            "motion_interpolation_algorithm_version": (
                MOTION_INTERPOLATION_ALGORITHM_VERSION
            ),
        }
        corrected_id = deterministic_id(
            "motion_corrected_recording",
            motion_corrected_identity_payload(**master_row),
        )
        explicit = key.get("motion_corrected_recording_id")
        if explicit is not None and uuid.UUID(str(explicit)) != corrected_id:
            raise ValueError(
                f"{caller}: motion_corrected_recording_id {explicit} does not "
                f"match the id derived from this selection ({corrected_id})."
            )
        existing = cls._find_existing_pk(master_row, corrected_id)
        if existing is not None:
            return existing
        try:
            cls.insert1(
                {"motion_corrected_recording_id": corrected_id, **master_row},
                allow_direct_insert=True,
            )
        except dj.errors.DuplicateError:
            existing = cls._find_existing_pk(master_row, corrected_id)
            if existing is None:
                raise
            return existing
        return {"motion_corrected_recording_id": corrected_id}

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
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        existing = list(
            (cls & master_row).fetch("motion_corrected_recording_id")
        )
        bypassed = [cid for cid in existing if cid != deterministic_id]
        if bypassed:
            raise DuplicateSelectionError(
                "MotionCorrectedRecordingSelection has rows for "
                f"{master_row} whose motion_corrected_recording_id is not the "
                f"deterministic id {deterministic_id}: {bypassed}. Drop them "
                "and re-insert via insert_selection."
            )
        return (
            {"motion_corrected_recording_id": deterministic_id}
            if existing
            else None
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
    """

    selection: dict
    interpolation_params: dict
    estimate: dict
    source_content_hash: str
    source_path: str
    source_electrical_series_path: str
    nwb_file_name: str


class MotionCorrectedComputed(NamedTuple):
    """The ``MotionCorrectedRecording`` row fields
    :meth:`MotionCorrectedRecording.make_compute` returns (plus the parent
    NWB name the artifact is registered under)."""

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


def _series_filtering(abs_path: str, electrical_series_path: str) -> str:
    """The ``filtering`` attribute of a persisted ``ElectricalSeries``."""
    import h5py

    with h5py.File(abs_path, "r") as handle:
        value = handle[electrical_series_path].attrs.get("filtering", "")
    return value.decode() if isinstance(value, bytes) else str(value)


@schema
class MotionCorrectedRecording(SpyglassMixin, dj.Computed):
    """A saved motion estimate applied to the traces it was estimated from.

    The source artifact (a ``Recording`` or ``ConcatenatedRecording``) is
    silenced outside the estimate's statistics spans, presented on the
    estimate's estimation clock so every frame looks up the displacement at
    the time it had during estimation, interpolated with the recipe's
    explicit ``interpolate_motion`` arguments and silenced again
    (``_motion.apply_motion_on_estimation_clock``). The corrected, masked,
    unwhitened traces are written with the source artifact's own timestamps:
    a single recording's acquisition timestamps, a concatenation's own
    clock. Motion correction changes positions, not sample times: sample
    count, order and rate are the source's.

    ``channel_ids`` are the output channels in order; each is the
    ``electrode_id`` the persisted series references. ``remove_channels``
    records the dropped source channels in ``removed_channel_ids``.
    ``channel_locations`` are the UNMOVED source positions of the kept
    channels (SpikeInterface copies the parent's metadata). The spans are
    copies of the estimate's: statistics come from the same valid samples.

    Tri-part: ``make_fetch`` resolves every row and the source path (its
    self-heal may rebuild the source), ``make_compute`` reads only files and
    writes the staged artifact outside the DB transaction, ``make_insert``
    registers it. :meth:`get_recording` rebuilds a missing file by
    reapplying the SAVED motion (it never estimates again) and installs it
    only when its content hash matches.
    """

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

    def make_fetch(self, key) -> MotionCorrectedFetched:
        """Resolve the selection, recipe, saved estimate and source artifact.

        Rebuilds a missing source NWB through its own verified self-heal and
        refuses a source whose ``content_hash`` changed since the estimate
        was selected.
        """
        from spyglass.spikesorting.v2._recording_nwb import (
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
    ) -> MotionCorrectedComputed:
        """Apply the saved motion and write the staged artifact; no DB reads.

        The artifact file is created through ``write_nwb_artifact`` (as every
        v2 trace writer does) and registered only by :meth:`make_insert`.

        Raises
        ------
        ValueError
            On a stale selection (interpolation recipe resolution,
            SpikeInterface version or application algorithm changed), a source whose frames, rate or
            channels differ from the estimate's, or an application failure
            (see ``_motion.apply_motion_on_estimation_clock``).
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2 import _motion
        from spyglass.spikesorting.v2._nwb_provenance import (
            MOTION_CORRECTION_PROVENANCE,
            build_provenance_table,
        )
        from spyglass.spikesorting.v2._recording_geometry import (
            flatten_planar_geometry,
        )
        from spyglass.spikesorting.v2._recording_nwb import (
            read_recording_nwb,
            write_nwb_artifact,
        )
        from spyglass.spikesorting.v2._recording_restriction import (
            _LazyRecordingTimestamps,
        )

        resolved = _motion.resolve_interpolation_params(interpolation_params)
        stale = [
            f"{name} {now!r} != selected {then!r}"
            for name, now, then in (
                (
                    "resolved interpolation hash",
                    _motion.resolved_params_hash(resolved),
                    selection["resolved_params_hash"],
                ),
                (
                    "SpikeInterface version",
                    si.__version__,
                    selection["spikeinterface_version"],
                ),
                (
                    "motion interpolation algorithm version",
                    _motion.MOTION_INTERPOLATION_ALGORITHM_VERSION,
                    selection["motion_interpolation_algorithm_version"],
                ),
            )
            if now != then
        ]
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
        mismatched = [
            name
            for name, now, then in (
                ("n_samples", n_samples, int(estimate["n_samples"])),
                (
                    "sampling_frequency",
                    sampling_frequency,
                    float(estimate["sampling_frequency"]),
                ),
                (
                    "channel_ids",
                    source.channel_ids.tolist(),
                    np.asarray(estimate["channel_ids"]).tolist(),
                ),
            )
            if now != then
        ]
        if mismatched:
            raise ValueError(
                f"MotionCorrectedRecording {key}: the source's {mismatched} "
                "differ from the saved estimate's; the estimate does not "
                "describe these traces."
            )
        timestamps = _LazyRecordingTimestamps(source, 0, n_samples)
        source_filtering = _series_filtering(
            source_path, source_electrical_series_path
        )

        # The estimate stored the flattened positions it was computed on.
        flatten_planar_geometry(source)
        source_locations = np.asarray(
            source.get_channel_locations(), dtype=np.float64
        )
        estimate_locations = np.asarray(
            estimate["channel_locations"], dtype=np.float64
        )
        if source_locations.shape != estimate_locations.shape or not (
            np.array_equal(source_locations, estimate_locations)
        ):
            raise ValueError(
                f"MotionCorrectedRecording {key}: the source's contact "
                f"positions {source_locations.tolist()} differ from the saved "
                f"estimate's {estimate_locations.tolist()}; the estimate does "
                "not describe this geometry."
            )

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
        provenance = build_provenance_table(
            MOTION_CORRECTION_PROVENANCE,
            {
                "motion_corrected_recording_id": str(
                    key["motion_corrected_recording_id"]
                ),
                "motion_estimate_id": str(selection["motion_estimate_id"]),
                "motion_interpolation_params_name": selection[
                    "motion_interpolation_params_name"
                ],
                "interpolation": resolved,
                "motion_interpolation_algorithm_version": int(
                    selection["motion_interpolation_algorithm_version"]
                ),
                "source_content_hash": str(source_content_hash),
                "removed_channel_ids": applied.removed_channel_ids,
                "spikeinterface_version": si.__version__,
            },
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
            provenance_tables=[provenance],
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

        On any failure the staged file is removed before re-raising, so no
        unregistered artifact outlives a failed populate.
        """
        from spyglass.spikesorting.v2.recording import (
            _ELECTRICAL_SERIES_PATH,
            _unlink_staged_analysis_file,
        )

        row = MotionCorrectedComputed(*computed)._asdict()
        nwb_file_name = row.pop("nwb_file_name")
        try:
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(nwb_file_name, row["analysis_file_name"])
                self.insert1(
                    {
                        **key,
                        **row,
                        "electrical_series_path": _ELECTRICAL_SERIES_PATH,
                    }
                )
        except Exception:
            _unlink_staged_analysis_file(
                row["analysis_file_name"],
                context="MotionCorrectedRecording.make_insert",
            )
            raise

    def get_recording(self, key: dict):
        """Return the corrected recording, rebuilding a missing file first.

        The same self-heal contract as ``Recording.get_recording``: the row
        is never deleted, and a rebuild is installed only when its content
        hash matches the stored one.

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
        from spyglass.spikesorting.v2._recording_nwb import (
            ensure_artifact_file,
            read_recording_nwb,
        )

        row = (self & key).fetch1()
        abs_path = ensure_artifact_file(
            type(self), key, row["analysis_file_name"]
        )
        recording = read_recording_nwb(
            abs_path, electrical_series_path=row["electrical_series_path"]
        )
        recording.annotate(is_filtered=True)
        return recording

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild a missing corrected artifact from the SAVED motion.

        Locked on the corrected recording, double-checked under the lock,
        then ``make_fetch`` / ``make_compute`` write a fresh temp artifact:
        the saved estimate is reapplied, never estimated again. Only a temp
        whose ``content_hash`` equals the stored one is installed
        (``install_rebuilt_recording``); otherwise it is removed,
        ``RecordingContentDriftError`` is raised and the canonical slot is
        left untouched.
        """
        from pathlib import Path

        from spyglass.spikesorting.v2._motion import (
            motion_corrected_recording_artifact_lock,
        )
        from spyglass.spikesorting.v2._recording_nwb import (
            install_rebuilt_recording,
        )
        from spyglass.spikesorting.v2.exceptions import (
            RecordingContentDriftError,
        )
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )
        from spyglass.utils import logger

        row = (self & key).fetch1()
        analysis_file_name = row["analysis_file_name"]
        canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with motion_corrected_recording_artifact_lock(
            row["motion_corrected_recording_id"]
        ):
            if Path(canonical_abs).exists():
                return
            logger.info(
                "MotionCorrectedRecording.get_recording: cache miss for "
                f"{analysis_file_name!r}; reapplying the saved motion..."
            )
            master_key = {
                "motion_corrected_recording_id": row[
                    "motion_corrected_recording_id"
                ]
            }
            computed = self.make_compute(
                master_key, *self.make_fetch(master_key)
            )
            if computed.content_hash != row["content_hash"]:
                _unlink_staged_analysis_file(
                    computed.analysis_file_name,
                    context="MotionCorrectedRecording._rebuild_nwb_artifact",
                )
                raise RecordingContentDriftError(
                    "MotionCorrectedRecording._rebuild_nwb_artifact: rebuilt "
                    f"content_hash {computed.content_hash} does not match the "
                    f"stored content_hash {row['content_hash']} for "
                    f"{analysis_file_name!r}. The current environment no "
                    "longer reproduces this corrected recording (e.g. a "
                    "SpikeInterface/BLAS upgrade). The canonical artifact was "
                    "NOT modified. Recover by restoring a backup or deleting "
                    "and repopulating the MotionCorrectedRecording row."
                )
            install_rebuilt_recording(
                AnalysisNwbfile.get_abs_path(computed.analysis_file_name),
                canonical_abs,
                analysis_file_name,
            )
