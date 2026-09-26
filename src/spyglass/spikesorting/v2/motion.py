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

Estimating motion changes nothing downstream: no existing table populates or
reads these rows. The DB-free computation (parameter resolution, the
estimation adapter, the ``Motion`` serialization) lives in ``_motion``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._params.motion_estimation import (
    MOTION_ESTIMATION_SCHEMA_VERSION,
    MotionEstimationParamsSchema,
)
from spyglass.spikesorting.v2._recipe_catalog import (
    motion_estimation_default_contents,
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


def _artifact_path(table, key: dict, row: dict) -> str:
    """Absolute path of a cached trace artifact, rebuilt first if missing."""
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    if not Path(abs_path).exists():
        table()._rebuild_nwb_artifact(key)
    return abs_path


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
    recipe's ``max_gap_s`` (``_motion.build_estimation_clock``). All spans
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
        source_path = _artifact_path(table, lineage.key, source_row)

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
        source traces; for a single recording derives the continuity spans
        and their start times from its timestamps, silences the artifact
        ranges and computes the statistics spans as the sort stage does; for a
        concat reads its persisted continuity spans, start times and
        statistics spans. Builds the estimation clock with the recipe's
        ``max_gap_s`` and runs ``_motion.estimate_motion_in_spans`` once.

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
            boundary_spans_from_timestamps,
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
            continuity = boundary_spans_from_timestamps(recording)
            continuity_start_s = [
                float(recording.sample_index_to_time(start))
                for start, _ in continuity
            ]
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
            statistics = _motion.normalize_spans(source_row["statistics_spans"])

        sampling_frequency = float(recording.get_sampling_frequency())
        clock = _motion.build_estimation_clock(
            continuity,
            continuity_start_s,
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
            The continuity spans in frames, each span's start on the source
            clock and on the estimation clock, and the sampling frequency.
        """
        from spyglass.spikesorting.v2._motion import estimation_clock_from_blob

        spans, source_start, estimation_start, fs = (self & key).fetch1(
            "continuity_spans",
            "continuity_start_s",
            "estimation_start_s",
            "sampling_frequency",
        )
        return estimation_clock_from_blob(
            {
                "spans": spans,
                "source_start_s": source_start,
                "estimation_start_s": estimation_start,
                "sampling_frequency": fs,
            }
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
