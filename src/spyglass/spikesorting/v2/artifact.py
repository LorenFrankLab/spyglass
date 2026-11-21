"""Artifact detection over a preprocessed Recording.

Tables (one result table per source kind, unified by the
``ArtifactDetectionOutput`` merge in ``artifact_output.py``):

- ``ArtifactDetectionParameters``: threshold detection parameters
- ``SharedArtifactGroup`` (+ ``Member``): opt-in cross-recording detection
- ``RecordingArtifactSelection``: single-recording request
- ``SharedArtifactSelection`` (+ ``Member``): cross-recording request with
  its frozen member set
- ``RecordingArtifactDetection`` / ``SharedArtifactDetection``: results;
  each ``RemovedInterval`` part owns the ``IntervalList`` rows it writes

Each selection master has a required source FK, so a selection has exactly
one source. Artifact-removed valid times are stored in ``IntervalList``; the
``RemovedInterval`` parts keep generic ``IntervalList`` cleanup from treating
them as orphans. Shared machinery lives on ``_ArtifactDetectionMixin``.
"""

from __future__ import annotations

from typing import NamedTuple

import datajoint as dj
import numpy as np
import spikeinterface as si

from spyglass.common import IntervalList, Session  # noqa: F401

from spyglass.spikesorting.v2._artifacts.intervals import (
    build_artifact_interval_part_rows,
    build_artifact_interval_rows,
    detect_artifacts,
)
from spyglass.spikesorting.v2._artifacts.readers import (
    collect_artifact_interval_rows_to_remove,
    insert_artifact_interval_rows,
    read_owned_artifact_intervals,
    refuse_used_artifact_intervals,
    remove_artifact_interval_rows,
)
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    artifact_default_contents,
)
from spyglass.spikesorting.v2._storage.nwb import StoredTraces
from spyglass.spikesorting.v2._core.signal_math import timestamp_fingerprint
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _insert_parameter_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.mixins.immutable import FactoryOnlyMaster
from spyglass.spikesorting.v2._artifacts.manual import (
    apply_manual_exclusions,
    manual_exclusion_problem,
)
from spyglass.spikesorting.v2._core.signal_math import (
    normalize_manual_exclusions,
)
from spyglass.spikesorting.v2._artifacts.shared_group import (
    assert_shared_group_recordings_aggregatable,
    validate_shared_artifact_group_members,
)
from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs
from spyglass.spikesorting.v2._core.lookup_validation import (
    _ensure_lookup_row_exists,
)
from spyglass.spikesorting.v2._core.selection_identity import (
    artifact_detection_identity_payload,
    assert_supplied_id_matches,
    deterministic_id,
)
from spyglass.spikesorting.v2.exceptions import (
    IneffectiveManualExclusionError,
    SchemaBypassError,
)
from spyglass.spikesorting.v2.recording import RecordingSelection

schema = dj.schema("spikesorting_v2_artifact")


class ArtifactComputed(NamedTuple):
    """Outputs of the artifact-detection tables' ``make_compute``.

    ``per_member_nwb_files`` lists the distinct ``nwb_file_name`` values that
    ``make_insert`` writes one ``IntervalList`` row for.
    """

    valid_times: np.ndarray
    nwb_file_name: str
    per_member_nwb_files: tuple = ()


@schema
class ArtifactDetectionParameters(
    SpyglassMixin, ImmutableParamsLookup, dj.Lookup
):
    """Validated artifact-detection parameter blob.

    ``artifact_detection_params`` is validated by
    :class:`ArtifactDetectionParamsSchema`.
    ``insert_default`` ships:

    - ``"none"``: no scan
    - ``"default"``: 500 uV amplitude threshold
    - ``"franklab_100uv_p07_2026_06"``: 100 uV, proportion 0.7
    - ``"franklab_50uv_p07_2026_06"``: 50 uV, proportion 0.7

    For a concat run, detection runs on each member before concatenation.

    ``job_kwargs`` (optional) sets SpikeInterface job kwargs for the chunked
    scan, merged over the SI-global and
    ``dj.config['custom']['spikesorting_v2_job_kwargs']`` defaults. The chunk
    size (``chunk_duration``, default ``"1s"``; ``chunk_size``; or
    ``chunk_memory``) bounds peak memory at about
    ``4 × chunk_frames × n_channels × 4 bytes``. ``n_jobs`` defaults to 1.
    """

    definition = """
    artifact_detection_params_name: varchar(200)
    ---
    artifact_detection_params: blob
    params_schema_version=2: int
    job_kwargs=null: blob  # SI job-kwargs for chunked scan; see docstring
    """

    _DEFAULT_CONTENTS: tuple = artifact_default_contents()

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert artifact-detection parameter rows after validation.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob); see
        ``reject_duplicate_parameter_content``.
        """
        _insert_parameter_rows(
            self,
            rows,
            insert_rows=super().insert,
            schema_for=lambda _row: ArtifactDetectionParamsSchema,
            table_name="ArtifactDetectionParameters",
            name_attr="artifact_detection_params_name",
            allow_duplicate_params=allow_duplicate_params,
            **kwargs,
        )

    @classmethod
    def insert_default(cls):
        """Insert v2 default artifact-detection presets if missing."""
        cls()._insert_missing_defaults(cls._DEFAULT_CONTENTS)


@schema
class SharedArtifactGroup(SpyglassMixin, FactoryOnlyMaster, dj.Manual):
    """Named bundle of one session's Recording rows that share a detection.

    For behavioral artifacts seen on every probe (chewing, licking, head
    bumps), which per-sort-group detection can miss. One detection pass over
    the union of the members' channels yields valid times shared by every
    member.

    Create groups with :meth:`create_group`. Membership is fixed at creation:
    direct ``insert`` and ``update1`` of the master and ``Member`` are
    blocked, so a different member set is a new group.
    """

    #: Named in ``FactoryOnlyMaster``'s reject messages.
    _factory_create_call = "SharedArtifactGroup.create_group()"

    definition = """
    -> Session
    shared_artifact_group_name: varchar(64)
    """

    class Member(FactoryOnlyMaster, SpyglassMixinPart):
        """One member Recording of a shared artifact-detection group."""

        _factory_create_call = "SharedArtifactGroup.create_group()"

        definition = """
        -> master
        -> Recording
        """

    @classmethod
    def create_group(
        cls, nwb_file_name: str, group_name: str, members: list[dict]
    ) -> None:
        """Insert master + Member rows after checking members share a time axis.

        Re-creating an existing group with the same members does nothing.

        Parameters
        ----------
        nwb_file_name : str
            The session every member belongs to.
        group_name : str
            Group name, unique within the session.
        members : list of dict
            Member recordings; each must contain ``recording_id``. Other
            fields are ignored.

        Raises
        ------
        ValueError
            If ``members`` is empty, a ``recording_id`` is not a populated
            ``Recording``, members differ in session (or are not in
            ``nwb_file_name``), sampling frequency, dtype or timestamps, or
            the group exists with other members.
        RuntimeError
            If a member recording cannot be loaded.
        """
        if not members:
            raise ValueError(
                "SharedArtifactGroup.create_group: members list is empty. "
                "Pass at least one recording_id dict."
            )

        member_recording_ids = []
        for m in members:
            if "recording_id" not in m:
                raise ValueError(
                    "SharedArtifactGroup.create_group: every member dict "
                    f"must include 'recording_id'. Got: {m!r}."
                )
            member_recording_ids.append(m["recording_id"])

        group_key = {
            "nwb_file_name": nwb_file_name,
            "shared_artifact_group_name": group_name,
        }
        if cls & group_key:
            existing = {
                str(rid)
                for rid in (cls.Member & group_key).fetch("recording_id")
            }
            if existing == {str(rid) for rid in member_recording_ids}:
                return
            raise ValueError(
                f"SharedArtifactGroup.create_group: group {group_key} exists "
                "with other members. A group's members are fixed; use a new "
                "group name, or delete this group (and the detections that "
                "use it) first."
            )

        # ``si.aggregate_channels`` needs equal n_samples, fs and dtype; check
        # here for a clear error instead of an opaque one at populate.
        per_member_recording_rows = (
            Recording & [{"recording_id": rid} for rid in member_recording_ids]
        ).fetch("recording_id", "sampling_frequency", as_dict=True)
        per_member_selection_rows = (
            RecordingSelection
            & [{"recording_id": rid} for rid in member_recording_ids]
        ).fetch(
            "recording_id",
            "nwb_file_name",
            "interval_list_name",
            as_dict=True,
        )
        rec_by_id = {
            str(r["recording_id"]): r for r in per_member_recording_rows
        }
        missing = [
            rid for rid in member_recording_ids if str(rid) not in rec_by_id
        ]
        if missing:
            raise ValueError(
                "SharedArtifactGroup.create_group: recording_id(s) "
                f"{missing} are not in Recording. Populate Recording for "
                "those selections first."
            )
        sel_by_id = {
            str(r["recording_id"]): r for r in per_member_selection_rows
        }

        # Cheap session/fs checks before loading multi-GB recordings.
        member_session = validate_shared_artifact_group_members(
            [
                {
                    "nwb_file_name": sel_by_id[str(rid)]["nwb_file_name"],
                    "sampling_frequency": rec_by_id[str(rid)][
                        "sampling_frequency"
                    ],
                }
                for rid in member_recording_ids
            ]
        )
        if member_session != nwb_file_name:
            raise ValueError(
                "SharedArtifactGroup.create_group: the members belong to "
                f"session {member_session!r}, not {nwb_file_name!r}."
            )

        # ``aggregate_channels`` stacks by frame index, so timestamps must
        # match exactly; equal lengths can still start at different times.
        member_dtypes: set[str] = set()
        reference_fingerprint = None
        reference_rid = None
        for rid in member_recording_ids:
            try:
                rec_obj = Recording().get_recording({"recording_id": rid})
            except Exception as exc:
                raise RuntimeError(
                    "SharedArtifactGroup.create_group: failed to load "
                    f"preprocessed recording for recording_id={rid!r} "
                    f"({type(exc).__name__}: {exc}). The strict "
                    "n_samples / dtype check at insert time requires "
                    "every member recording to be readable."
                ) from exc
            n_samples = int(rec_obj.get_num_samples())
            # len() of the lazy h5py time vector reads no data.
            explicit_time_vector = rec_obj.get_time_info(segment_index=0).get(
                "time_vector"
            )
            if (
                explicit_time_vector is not None
                and len(explicit_time_vector) != n_samples
            ):
                raise ValueError(
                    "SharedArtifactGroup.create_group: recording_id="
                    f"{rid!r} reports get_num_samples()={n_samples}, "
                    f"but its timestamp vector has length "
                    f"{len(explicit_time_vector)}. The preprocessed recording "
                    "is internally inconsistent and cannot be used for a "
                    "shared artifact group."
                )
            # Streamed fingerprint (includes n_samples) avoids holding two
            # full float64 timestamp vectors in memory.
            fingerprint = timestamp_fingerprint(rec_obj)
            if reference_fingerprint is None:
                reference_fingerprint = fingerprint
                reference_rid = rid
            elif fingerprint != reference_fingerprint:
                raise ValueError(
                    "SharedArtifactGroup.create_group: members have "
                    "differing exact timestamps (recording_id="
                    f"{reference_rid!r} vs {rid!r}). Use recordings from "
                    "the same interval, or run separate artifact detections."
                )
            member_dtypes.add(str(rec_obj.get_dtype()))
        if len(member_dtypes) != 1:
            raise ValueError(
                "SharedArtifactGroup.create_group: members have "
                f"differing dtypes {sorted(member_dtypes)}; "
                "``si.aggregate_channels`` requires identical dtype."
            )

        member_rows = [
            {**group_key, "recording_id": rid} for rid in member_recording_ids
        ]
        with cls._safe_context():
            # Rows were validated above.
            cls.insert1(group_key, allow_direct_insert=True)
            cls.Member.insert(member_rows, allow_direct_insert=True)


def _assert_manual_exclusions_effective(
    exclusions, caller: str, recording_ids
) -> None:
    """Raise unless every exclusion overlaps time the source recording keeps.

    The valid time is the source recording's stored kept time
    (``Recording.get_valid_interval``), the same rule preflight checks. A
    shared group's members are timestamp-identical, so the first member's
    time is used.

    Parameters
    ----------
    exclusions : list
        Normalized ``(n, 2)`` ``[start, stop)`` session-second intervals.
    caller : str
        The selection table name, prefixed to the message.
    recording_ids : sequence
        The source's recording ids: one, or a shared group's members.

    Raises
    ------
    IneffectiveManualExclusionError
        If an exclusion overlaps none of that time.
    ValueError
        If the source recording has no ``Recording.ValidInterval`` row.
    """
    requests = (
        RecordingSelection
        & [{"recording_id": rid} for rid in recording_ids]
        & Recording
    ).fetch(
        "recording_id",
        "nwb_file_name",
        "interval_list_name",
        as_dict=True,
        order_by="recording_id",
    )
    if not requests:
        return  # The FK insert reports a missing source.
    valid_times = (
        Recording()
        .get_valid_interval({"recording_id": requests[0]["recording_id"]})
        .times
    )
    problem = manual_exclusion_problem(
        exclusions,
        valid_times,
        nwb_file_name=requests[0]["nwb_file_name"],
        interval_list_name=requests[0]["interval_list_name"],
        prefix=f"{caller}.insert_selection: ",
    )
    if problem is not None:
        raise IneffectiveManualExclusionError(problem)


def _insert_artifact_selection(
    selection_cls,
    *,
    artifact_detection_params_name,
    source: dict,
    recording_ids,
    supplied_id=None,
    manual_excluded_times=None,
    member_part=None,
) -> dict:
    """Idempotently insert one artifact-detection selection row.

    Shared body of both ``insert_selection`` methods. The
    ``artifact_detection_id`` is content-addressed over params, source and
    manual exclusions. Validates any caller-supplied id, ensures the params
    row exists, and handles a concurrent duplicate-id race.

    Parameters
    ----------
    selection_cls : dj.Manual
        The source-specific selection table to insert into.
    artifact_detection_params_name : str
        The ``ArtifactDetectionParameters`` row name.
    source : dict
        The source identity passed to ``artifact_detection_identity_payload``
        (``recording_id``, or the shared group's ``nwb_file_name``,
        ``shared_artifact_group_name`` and ``member_recording_ids``).
    recording_ids : sequence
        The recordings the detection scans; one row each in ``member_part``.
    supplied_id : optional
        A caller-supplied ``artifact_detection_id`` to validate against the
        derived one, or ``None``.
    manual_excluded_times : array-like, optional
        ``[start, stop)`` absolute session-second intervals to exclude; part
        of the identity. Each must overlap the source's sorted time.
    member_part : dj.Part, optional
        The part that freezes ``recording_ids`` with the selection, written in
        the same transaction as the master.

    Returns
    -------
    dict
        ``{"artifact_detection_id": ...}``, the content-addressed PK.

    Raises
    ------
    IneffectiveManualExclusionError
        If an exclusion overlaps none of the source's sorted time, checked
        before an existing row is returned.
    """
    exclusions = normalize_manual_exclusions(manual_excluded_times)
    if exclusions:
        _assert_manual_exclusions_effective(
            exclusions, selection_cls.__name__, recording_ids
        )
    payload = artifact_detection_identity_payload(
        artifact_detection_params_name=artifact_detection_params_name,
        manual_excluded_times=exclusions,
        **source,
    )
    artifact_detection_id = deterministic_id("artifact_detection", payload)
    assert_supplied_id_matches(
        supplied_id, artifact_detection_id, field="artifact_detection_id"
    )
    pk = {"artifact_detection_id": artifact_detection_id}
    if selection_cls & pk:
        return pk

    _ensure_lookup_row_exists(
        ArtifactDetectionParameters,
        {"artifact_detection_params_name": artifact_detection_params_name},
        helper_name=f"{selection_cls.__name__}.insert_selection",
        insert_default_path="ArtifactDetectionParameters.insert_default()",
    )
    row = {
        **pk,
        "artifact_detection_params_name": artifact_detection_params_name,
        "manual_excluded_times": exclusions,
        **{k: v for k, v in source.items() if k != "member_recording_ids"},
    }
    try:
        with selection_cls._safe_context():
            # allow_direct_insert: insert_selection IS the validation boundary.
            selection_cls.insert1(row, allow_direct_insert=True)
            if member_part is not None:
                member_part.insert(
                    [{**pk, "recording_id": rid} for rid in recording_ids]
                )
    except dj.errors.DuplicateError:
        # A concurrent caller inserted the same content-addressed row.
        logger.debug(
            "%s.insert_selection: lost deterministic-id race on %s; "
            "returning the existing row.",
            selection_cls.__name__,
            artifact_detection_id,
        )
    return pk


@schema
class RecordingArtifactSelection(
    SpyglassMixin, SelectionMasterInsertGuard, dj.Manual
):
    """Single-recording artifact-detection request."""

    definition = """
    artifact_detection_id: uuid
    ---
    -> ArtifactDetectionParameters
    -> Recording
    manual_excluded_times=null: longblob # immutable [start, stop) session seconds
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Idempotently register a single-recording artifact selection.

        Parameters
        ----------
        key : dict
            Must carry ``artifact_detection_params_name`` and
            ``recording_id``. An explicit ``artifact_detection_id`` is
            validated against the derived deterministic id.

        Returns
        -------
        dict
            ``{"artifact_detection_id": ...}``, the content-addressed PK.
        """
        recording_id = key["recording_id"]
        return _insert_artifact_selection(
            cls,
            artifact_detection_params_name=key[
                "artifact_detection_params_name"
            ],
            source={"recording_id": recording_id},
            recording_ids=[recording_id],
            supplied_id=key.get("artifact_detection_id"),
            manual_excluded_times=key.get("manual_excluded_times"),
        )


@schema
class SharedArtifactSelection(
    SpyglassMixin, SelectionMasterInsertGuard, dj.Manual
):
    """Cross-recording (shared-group) artifact-detection request.

    The identity includes the session and the group's member set, and the
    ``Member`` part freezes that set: it is the one record of which
    recordings the detection scans and may mask.
    """

    definition = """
    artifact_detection_id: uuid
    ---
    -> ArtifactDetectionParameters
    -> SharedArtifactGroup
    manual_excluded_times=null: longblob # immutable [start, stop) session seconds
    """

    class Member(SpyglassMixinPart):
        """One recording the detection scans, frozen with the selection."""

        definition = """
        -> master
        -> Recording
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Idempotently register a shared-group artifact selection.

        Freezes the group's members in the ``Member`` part.

        Parameters
        ----------
        key : dict
            Must carry ``artifact_detection_params_name``, ``nwb_file_name``
            and ``shared_artifact_group_name``. An explicit
            ``artifact_detection_id`` is validated against the derived id.

        Returns
        -------
        dict
            ``{"artifact_detection_id": ...}``, the content-addressed PK.

        Raises
        ------
        ValueError
            If the ``SharedArtifactGroup`` does not exist.
        """
        group_key = {
            "nwb_file_name": key["nwb_file_name"],
            "shared_artifact_group_name": key["shared_artifact_group_name"],
        }
        member_recording_ids = sorted(
            str(rid)
            for rid in (SharedArtifactGroup.Member & group_key).fetch(
                "recording_id"
            )
        )
        if not member_recording_ids:
            raise ValueError(
                f"SharedArtifactSelection.insert_selection: {group_key} is "
                "not a SharedArtifactGroup. Create it with "
                "SharedArtifactGroup.create_group() first."
            )
        return _insert_artifact_selection(
            cls,
            artifact_detection_params_name=key[
                "artifact_detection_params_name"
            ],
            source={**group_key, "member_recording_ids": member_recording_ids},
            recording_ids=member_recording_ids,
            supplied_id=key.get("artifact_detection_id"),
            manual_excluded_times=key.get("manual_excluded_times"),
            member_part=cls.Member,
        )


class RecordingArtifactFetched(NamedTuple):
    """DB-side inputs for :meth:`RecordingArtifactDetection.make_fetch`.

    ``traces`` is the recording's cached artifact, rebuilt if missing.
    """

    validated: ArtifactDetectionParamsSchema
    recording_id: object
    nwb_file_name: str
    artifact_job_kwargs: dict | None
    traces: StoredTraces
    manual_excluded_times: object = None


class SharedArtifactFetched(NamedTuple):
    """DB-side inputs for :meth:`SharedArtifactDetection.make_fetch`.

    Per-member fields are tuples of length n_members. ``member_traces``
    holds each member's cached artifact, rebuilt if missing.
    ``nwb_file_name`` is the common session.
    """

    validated: ArtifactDetectionParamsSchema
    shared_artifact_group_name: str
    member_recording_ids: tuple
    member_nwb_file_names: tuple
    nwb_file_name: str
    artifact_job_kwargs: dict | None
    member_traces: tuple[StoredTraces, ...]
    manual_excluded_times: object = None


class _ArtifactDetectionMixin:
    """Shared source-agnostic detection machinery for both result tables.

    The tables differ only in how they load their source. The scan,
    ``IntervalList`` write, read-back and delete cleanup live here.
    ``_single_source`` means one owned ``IntervalList`` row, so
    :meth:`get_artifact_removed_intervals` returns a bare array.
    """

    # populate(processes>1) needs the non-daemon pool when SpikeInterface
    # starts worker processes.
    _parallel_make = True
    _single_source: bool = True

    def _run_artifact_scan(
        self,
        recording,
        validated,
        artifact_job_kwargs,
        context,
        manual_excluded_times=None,
    ):
        """Resolve job kwargs and scan a loaded recording for artifacts.

        Merges the per-row ``job_kwargs`` over the defaults, runs the chunked
        threshold scan, then applies manual exclusions.

        Parameters
        ----------
        recording : si.BaseRecording
            The loaded single or channel-unioned recording to scan.
        validated : ArtifactDetectionParamsSchema
            Validated detection parameters.
        artifact_job_kwargs : dict or None
            Per-row SI job-kwargs blob.
        context : str
            Diagnostic suffix identifying the selection for empty-scan logs.
        manual_excluded_times : array-like, optional
            ``[start, stop)`` session-second intervals to remove from the
            result.

        Returns
        -------
        np.ndarray
            Artifact-removed ``valid_times``, shape ``(n_intervals, 2)``.
        """
        resolved_job_kwargs = _resolved_job_kwargs(artifact_job_kwargs)
        valid_times = detect_artifacts(
            recording,
            validated,
            context=context,
            job_kwargs=resolved_job_kwargs,
        )

        return apply_manual_exclusions(
            valid_times,
            manual_excluded_times,
            validated.min_length_s,
        )

    def make_insert(
        self, key, valid_times, nwb_file_name, per_member_nwb_files=()
    ):
        """Write the artifact ``IntervalList`` + master + part rows atomically.

        Writes one ``IntervalList`` row per distinct member
        ``nwb_file_name``, the master row, the ``RemovedInterval`` ownership
        rows, and the ``ArtifactDetectionOutput`` registration. A stale
        interval row left by an upstream cascade is adopted or replaced
        (``insert_artifact_interval_rows``).

        Parameters
        ----------
        key : dict
            Primary-key restriction for the row being inserted.
        valid_times : np.ndarray
            Artifact-removed valid times, shape ``(n_intervals, 2)``.
        nwb_file_name : str
            Fallback ``IntervalList`` target when ``per_member_nwb_files`` is
            empty.
        per_member_nwb_files : tuple, optional
            Distinct member ``nwb_file_name`` s to write one row each.
        """
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )

        interval_rows = build_artifact_interval_rows(
            key, valid_times, nwb_file_name, per_member_nwb_files
        )
        part_rows = build_artifact_interval_part_rows(key, interval_rows)
        detection_key = {"artifact_detection_id": key["artifact_detection_id"]}
        with self._safe_context():
            insert_artifact_interval_rows(interval_rows, self.RemovedInterval)
            self.insert1(key)
            self.RemovedInterval.insert(part_rows)
            # Registered iff materialized, so sorting never inserts the merge.
            ArtifactDetectionOutput.insert_detection(detection_key)

    def get_artifact_removed_intervals(self, key, as_dict=False):
        """Return the artifact-removed ``valid_times`` for ``key``.

        A single-recording source returns an ``(n_intervals, 2)`` array; a
        shared group returns ``{nwb_file_name: array}`` with equal values.

        Parameters
        ----------
        key : dict
            Restriction selecting one detection row; must include
            ``artifact_detection_id``.
        as_dict : bool, optional
            Force the dict shape even for a single-recording source.

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            The ``valid_times`` array (single-recording, ``as_dict=False``)
            or the per-member dict otherwise.
        """
        result = read_owned_artifact_intervals(type(self), key)
        if self._single_source:
            if len(result) != 1:
                raise ValueError(
                    f"{type(self).__name__}.get_artifact_removed_intervals: "
                    f"recording-backed {key!r} has {len(result)} "
                    "RemovedInterval part rows; expected exactly one."
                )
            if not as_dict:
                return next(iter(result.values()))
        return result

    def delete(self, *restrictions, safemode=None, **kwargs):
        """Delete detections, their dependents, and owned intervals.

        The cascade removes the merge registration and any sorting,
        motion-estimate or concat selection that uses a detection; the
        confirmation prompt lists them. Owned ``IntervalList`` rows are
        removed for detections actually deleted, since only this table's
        ``RemovedInterval`` parts record which intervals it owns.
        Positional arguments are restrictions, applied before rows are
        fetched.

        Raises
        ------
        ValueError
            Before deleting anything, if another table (a sort, LFP or
            decoding selection) uses an owned interval.
        """
        if restrictions:
            return self._restrict_delete_args(restrictions).delete(
                safemode=safemode, **kwargs
            )

        detection_cls = type(self)
        rows = self.fetch(as_dict=True)
        delete_targets = [
            (
                {k: row[k] for k in self.primary_key},
                collect_artifact_interval_rows_to_remove(
                    [row], detection_cls=detection_cls
                ),
            )
            for row in rows
        ]
        refuse_used_artifact_intervals(
            [r for _, interval_rows in delete_targets for r in interval_rows],
            self.RemovedInterval,
            context=f"{detection_cls.__name__}.delete",
        )

        # force_masters cascade removes the merge registration too.
        if safemode is None:
            super().delete(**kwargs)
        else:
            super().delete(safemode=safemode, **kwargs)

        remove_artifact_interval_rows(
            [
                restriction
                for master_key, interval_rows_to_remove in delete_targets
                if not (detection_cls & master_key)
                for restriction in interval_rows_to_remove
            ]
        )


@schema
class RecordingArtifactDetection(
    _ArtifactDetectionMixin, SpyglassMixin, dj.Computed
):
    """Artifact-removed valid times for a single ``Recording``.

    Writes one ``IntervalList`` row. Always reads the uncorrected
    ``Recording``; motion correction is downstream of the mask.

    Steps
    -----
    1. Resolve the parameters and the recording, rebuilding a
       missing file.
    2. Open the recording.
    3. Scan it in chunks for threshold-crossing artifacts.
    4. Remove the manual exclusions.
    5. Insert the ``IntervalList``, master, part and merge rows.
    """

    _populate_key_lock = True

    definition = """
    -> RecordingArtifactSelection
    """

    _single_source = True

    class RemovedInterval(SpyglassMixinPart):
        """Generated artifact-removed ``IntervalList`` row (relational owner).

        Short name keeps the sibling's identifiers under MySQL's 64 chars.
        """

        definition = """
        -> master
        -> IntervalList
        """

    def make_fetch(self, key) -> RecordingArtifactFetched:
        """Read the params blob, resolved recording, and parent session.

        A missing cached artifact is rebuilt through ``Recording``.

        Returns
        -------
        RecordingArtifactFetched
            Validated params, the resolved ``recording_id``, its parent
            ``nwb_file_name``, the per-row job kwargs and the resolved
            artifact.
        """
        (
            params_blob,
            artifact_job_kwargs,
            recording_id,
            nwb_file_name,
            manual_excluded_times,
        ) = (
            ArtifactDetectionParameters
            * (RecordingArtifactSelection & key)
            * RecordingSelection
        ).fetch1(
            "artifact_detection_params",
            "job_kwargs",
            "recording_id",
            "nwb_file_name",
            "manual_excluded_times",
        )
        validated = ArtifactDetectionParamsSchema.model_validate(params_blob)
        return RecordingArtifactFetched(
            validated=validated,
            recording_id=recording_id,
            nwb_file_name=nwb_file_name,
            artifact_job_kwargs=artifact_job_kwargs,
            traces=Recording().resolve_stored_traces(
                {"recording_id": recording_id}
            ),
            manual_excluded_times=manual_excluded_times,
        )

    def make_compute(
        self,
        key,
        validated,
        recording_id,
        nwb_file_name,
        artifact_job_kwargs,
        traces,
        manual_excluded_times=None,
    ) -> ArtifactComputed:
        """Read the single recording and scan it for artifacts; no DB access.

        Returns
        -------
        ArtifactComputed
            The ``valid_times`` plus the one-element per-member target list.
        """
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces

        # 2. Open the recording.
        recording = read_stored_traces(traces)
        # 3-4. Scan for artifacts, then remove the manual exclusions.
        valid_times = self._run_artifact_scan(
            recording,
            validated,
            artifact_job_kwargs,
            manual_excluded_times=manual_excluded_times,
            context=(
                f" for artifact_detection_id={key['artifact_detection_id']}, "
                f"recording_id={recording_id}"
            ),
        )
        return ArtifactComputed(
            valid_times=valid_times,
            nwb_file_name=nwb_file_name,
            per_member_nwb_files=(nwb_file_name,),
        )


@schema
class SharedArtifactDetection(
    _ArtifactDetectionMixin, SpyglassMixin, dj.Computed
):
    """Artifact-removed valid times for a ``SharedArtifactGroup``.

    Steps
    -----
    1. Resolve the parameters and the members frozen in
       ``SharedArtifactSelection.Member``.
    2. Open each member's recording and check they can be aggregated.
    3. Union the members' channels and scan once for artifacts.
    4. Remove the manual exclusions.
    5. Insert one ``IntervalList`` row per member session, plus the
       master, part and merge rows.
    """

    _populate_key_lock = True

    definition = """
    -> SharedArtifactSelection
    """

    _single_source = False

    class RemovedInterval(SpyglassMixinPart):
        """Generated artifact-removed ``IntervalList`` row (relational owner).

        Short name keeps identifiers under MySQL's 64-char limit.
        """

        definition = """
        -> master
        -> IntervalList
        """

    def make_fetch(self, key) -> SharedArtifactFetched:
        """Read params and the members frozen with the selection.

        Missing cached artifacts are rebuilt through ``Recording``.

        Returns
        -------
        SharedArtifactFetched
            Validated params, the group name, the ordered member
            ``recording_id`` / ``nwb_file_name`` tuples, the canonical parent
            ``nwb_file_name``, the per-row job kwargs and the resolved member
            artifacts.
        """
        (
            group_name,
            manual_excluded_times,
            params_blob,
            artifact_job_kwargs,
        ) = (
            ArtifactDetectionParameters * (SharedArtifactSelection & key)
        ).fetch1(
            "shared_artifact_group_name",
            "manual_excluded_times",
            "artifact_detection_params",
            "job_kwargs",
        )
        validated = ArtifactDetectionParamsSchema.model_validate(params_blob)

        # Ordered so both make_fetch calls DeepHash equal.
        members = (
            SharedArtifactSelection.Member * RecordingSelection & key
        ).fetch(
            "recording_id",
            "nwb_file_name",
            as_dict=True,
            order_by="recording_id",
        )
        if not members:
            raise SchemaBypassError(
                "SharedArtifactDetection.make: SharedArtifactSelection "
                f"{key['artifact_detection_id']} has no Member rows; "
                "insert_selection writes them with the selection."
            )
        member_recording_ids = tuple(str(m["recording_id"]) for m in members)
        member_nwb_file_names = tuple(m["nwb_file_name"] for m in members)
        return SharedArtifactFetched(
            validated=validated,
            shared_artifact_group_name=group_name,
            member_recording_ids=member_recording_ids,
            member_nwb_file_names=member_nwb_file_names,
            nwb_file_name=member_nwb_file_names[0],
            artifact_job_kwargs=artifact_job_kwargs,
            member_traces=tuple(
                Recording().resolve_stored_traces({"recording_id": rid})
                for rid in member_recording_ids
            ),
            manual_excluded_times=manual_excluded_times,
        )

    def make_compute(
        self,
        key,
        validated,
        shared_artifact_group_name,
        member_recording_ids,
        member_nwb_file_names,
        nwb_file_name,
        artifact_job_kwargs,
        member_traces,
        manual_excluded_times=None,
    ) -> ArtifactComputed:
        """Union the members' channels and scan once; no DB access.

        Returns
        -------
        ArtifactComputed
            The shared ``valid_times`` plus the distinct member
            ``nwb_file_name`` target list ``make_insert`` writes one row per.
        """
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces

        if not member_recording_ids:
            raise RuntimeError(
                "SharedArtifactDetection.make_compute: shared-group "
                "source has no member_recording_ids; make_fetch contract "
                "violated."
            )
        # 2. Open each member and re-check create_group's invariants, which a
        # direct insert can bypass.
        per_member_recordings = [
            read_stored_traces(traces) for traces in member_traces
        ]

        assert_shared_group_recordings_aggregatable(
            per_member_recordings,
            member_recording_ids,
            member_nwb_file_names,
        )
        # 3-4. Union the channels, scan once, remove the manual exclusions.
        unioned = si.aggregate_channels(per_member_recordings)
        valid_times = self._run_artifact_scan(
            unioned,
            validated,
            artifact_job_kwargs,
            manual_excluded_times=manual_excluded_times,
            context=(
                f" for artifact_detection_id={key['artifact_detection_id']}, "
                f"shared_artifact_group={shared_artifact_group_name}"
            ),
        )
        per_member_nwb_files = tuple(dict.fromkeys(member_nwb_file_names))
        return ArtifactComputed(
            valid_times=valid_times,
            nwb_file_name=nwb_file_name,
            per_member_nwb_files=per_member_nwb_files,
        )


def assert_artifact_detection_covers_recording(
    *, recording_id, artifact_detection_id, caller: str
) -> None:
    """Ensure an artifact detection may mask a given ``Recording``.

    A single-recording detection may mask only its ``recording_id``; a
    shared-group detection may mask any member. The detection must be
    populated.

    Parameters
    ----------
    recording_id : uuid.UUID or str
        The recording the detection would mask.
    artifact_detection_id : uuid.UUID, str or None
        The detection; ``None`` (no artifact pass) always passes.
    caller : str
        Keyword-only. Prefix for the error messages (e.g.
        ``"SortingSelection.insert_selection"``).

    Raises
    ------
    ValueError
        If the detection belongs to another recording or group, or is not
        populated.
    """
    if artifact_detection_id is None:
        return

    artifact_detection_key = {"artifact_detection_id": artifact_detection_id}
    target_recording_id = str(recording_id)
    # Only populated detections are registered in ArtifactDetectionOutput.
    if RecordingArtifactDetection & artifact_detection_key:
        artifact_recording_id = str(
            (RecordingArtifactSelection & artifact_detection_key).fetch1(
                "recording_id"
            )
        )
        if artifact_recording_id != target_recording_id:
            raise ValueError(
                f"{caller}: artifact_detection_id "
                f"{artifact_detection_id!r} belongs to recording_id="
                f"{artifact_recording_id!r}, not the requested "
                f"recording_id={target_recording_id!r}."
            )
        return

    if SharedArtifactDetection & artifact_detection_key:
        if not (
            SharedArtifactSelection.Member
            & artifact_detection_key
            & {"recording_id": recording_id}
        ):
            group = (SharedArtifactSelection & artifact_detection_key).fetch(
                "nwb_file_name", "shared_artifact_group_name", as_dict=True
            )[0]
            raise ValueError(
                f"{caller}: artifact_detection_id "
                f"{artifact_detection_id!r} was detected over shared artifact "
                f"group {group}, whose members do not include requested "
                f"recording_id={target_recording_id!r}."
            )
        return

    raise ValueError(
        f"{caller}: artifact_detection_id "
        f"{artifact_detection_id!r} is not in RecordingArtifactDetection or "
        "SharedArtifactDetection. Populate the artifact detection "
        "before linking it."
    )


# Declare the merge table now; make_insert cannot create it mid-transaction.
# Last because artifact_output imports the tables above.
from spyglass.spikesorting.v2 import (  # noqa: E402, F401
    artifact_output as _artifact_output,
)
