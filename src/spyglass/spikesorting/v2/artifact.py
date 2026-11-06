"""Artifact detection over a preprocessed Recording.

Tables (one result table per source kind, unified by the
``ArtifactDetectionOutput`` merge in ``artifact_output.py``):

- ``ArtifactDetectionParameters``: threshold detection parameters
- ``SharedArtifactGroup`` (+ ``Member``): opt-in cross-recording detection
- ``RecordingArtifactSelection``: single-recording request
- ``SharedGroupArtifactSelection``: cross-recording request
- ``RecordingArtifactDetection`` / ``SharedGroupArtifactDetection``: results;
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

from spyglass.common import IntervalList, Session  # noqa: F401

from spyglass.spikesorting.v2._artifacts.intervals import (
    build_artifact_interval_part_rows,
    build_artifact_interval_rows,
    detect_artifacts,
)
from spyglass.spikesorting.v2._artifacts.readers import (
    collect_artifact_interval_rows_to_remove,
    read_owned_artifact_intervals,
    remove_artifact_interval_rows,
)
from spyglass.spikesorting.v2._params.artifact_detection import (
    ARTIFACT_DETECTION_SCHEMA_VERSION,
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
    restrict_by_leading_args,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

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
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Validated artifact-detection parameter blob.

    ``params`` is validated by :class:`ArtifactDetectionParamsSchema`.
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

    definition = f"""
    artifact_detection_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={ARTIFACT_DETECTION_SCHEMA_VERSION}: int
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
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class SharedArtifactGroup(SpyglassMixin, dj.Manual):
    """Named bundle of Recording rows that share an artifact-detection pass.

    For behavioral artifacts seen on every probe (chewing, licking, head
    bumps), which per-sort-group detection can miss. One detection pass over
    the union of the members' channels yields valid times shared by every
    member. All members must belong to one session.
    """

    definition = """
    shared_artifact_group_name: varchar(64)
    ---
    -> Session
    """

    class Member(SpyglassMixinPart):
        """One member Recording of a shared artifact-detection group."""

        definition = """
        -> master
        -> Recording
        """

    @classmethod
    def insert_group(cls, name: str, members: list[dict]) -> None:
        """Insert master + Member rows after checking members share a time axis.

        Parameters
        ----------
        name : str
            Group name (master PK).
        members : list of dict
            Member recordings; each must contain ``recording_id``. Other
            fields are ignored.

        Raises
        ------
        ValueError
            If ``members`` is empty, a ``recording_id`` is not a populated
            ``Recording``, or members differ in session, sampling frequency,
            dtype or timestamps.
        RuntimeError
            If a member recording cannot be loaded.
        """
        from spyglass.spikesorting.v2._artifacts.shared_group import (
            validate_shared_artifact_group_members,
        )
        from spyglass.spikesorting.v2.recording import (
            Recording,
            RecordingSelection,
        )

        if not members:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members list is empty. "
                "Pass at least one recording_id dict."
            )

        member_recording_ids = []
        for m in members:
            if "recording_id" not in m:
                raise ValueError(
                    "SharedArtifactGroup.insert_group: every member dict "
                    f"must include 'recording_id'. Got: {m!r}."
                )
            member_recording_ids.append(m["recording_id"])

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
                "SharedArtifactGroup.insert_group: recording_id(s) "
                f"{missing} are not in Recording. Populate Recording for "
                "those selections first."
            )
        sel_by_id = {
            str(r["recording_id"]): r for r in per_member_selection_rows
        }

        # Cheap session/fs checks before loading multi-GB recordings.
        nwb_file_name = validate_shared_artifact_group_members(
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
                    "SharedArtifactGroup.insert_group: failed to load "
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
                    "SharedArtifactGroup.insert_group: recording_id="
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
                    "SharedArtifactGroup.insert_group: members have "
                    "differing exact timestamps (recording_id="
                    f"{reference_rid!r} and recording_id={rid!r} have "
                    "non-identical timestamp vectors). Shared artifact "
                    "detection requires time-aligned recordings; use "
                    "recordings from the same interval identity or run "
                    "separate artifact detections."
                )
            member_dtypes.add(str(rec_obj.get_dtype()))
        if len(member_dtypes) != 1:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members have "
                f"differing dtypes {sorted(member_dtypes)}; "
                "``si.aggregate_channels`` requires identical dtype."
            )

        master_row = {
            "shared_artifact_group_name": name,
            "nwb_file_name": nwb_file_name,
        }
        member_rows = [
            {
                "shared_artifact_group_name": name,
                "recording_id": rid,
            }
            for rid in member_recording_ids
        ]

        with cls._safe_context():
            cls.insert1(master_row)
            cls.Member.insert(member_rows)


def _insert_artifact_selection(
    selection_cls,
    *,
    artifact_detection_params_name,
    recording_id=None,
    shared_artifact_group_name=None,
    supplied_id=None,
    manual_excluded_times=None,
    extra_row,
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
    recording_id : uuid or str, optional
        Single-recording source id (mutually exclusive with
        ``shared_artifact_group_name``).
    shared_artifact_group_name : str, optional
        Shared-group source name (mutually exclusive with ``recording_id``).
    supplied_id : optional
        A caller-supplied ``artifact_detection_id`` to validate against the
        derived one, or ``None``.
    manual_excluded_times : array-like, optional
        ``[start, stop)`` session-second intervals to exclude; part of the
        identity.
    extra_row : dict
        Source-specific columns to add to the inserted row (the source FK,
        plus ``member_set_hash`` for the shared-group table).

    Returns
    -------
    dict
        ``{"artifact_detection_id": ...}``, the content-addressed PK.
    """
    from spyglass.spikesorting.v2._core.selection_identity import (
        artifact_detection_identity_payload,
        assert_supplied_id_matches,
        deterministic_id,
    )
    from spyglass.spikesorting.v2._core.lookup_validation import (
        _ensure_lookup_row_exists,
    )

    from spyglass.spikesorting.v2._artifacts.manual import (
        normalize_manual_exclusions,
    )

    exclusions = normalize_manual_exclusions(manual_excluded_times)
    payload = artifact_detection_identity_payload(
        artifact_detection_params_name=artifact_detection_params_name,
        recording_id=recording_id,
        shared_artifact_group_name=shared_artifact_group_name,
        manual_excluded_times=exclusions,
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
        **extra_row,
    }
    try:
        with selection_cls._safe_context():
            # allow_direct_insert: insert_selection IS the validation boundary.
            selection_cls.insert1(row, allow_direct_insert=True)
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
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
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
            recording_id=recording_id,
            supplied_id=key.get("artifact_detection_id"),
            manual_excluded_times=key.get("manual_excluded_times"),
            extra_row={"recording_id": recording_id},
        )


@schema
class SharedGroupArtifactSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """Cross-recording (shared-group) artifact-detection request.

    ``member_set_hash`` snapshots the member ``recording_id`` set, which is
    not part of the identity. ``SharedGroupArtifactDetection.make_fetch``
    rejects a changed member set.
    """

    definition = """
    artifact_detection_id: uuid
    ---
    -> ArtifactDetectionParameters
    -> SharedArtifactGroup
    member_set_hash: char(64)   # frozen sha256 of the sorted (order-independent) member recording_id set
    manual_excluded_times=null: longblob # immutable [start, stop) session seconds
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Idempotently register a shared-group artifact selection.

        Snapshots the group's current member set onto ``member_set_hash``.

        Parameters
        ----------
        key : dict
            Must carry ``artifact_detection_params_name`` and
            ``shared_artifact_group_name``. An explicit
            ``artifact_detection_id`` is validated against the derived id.

        Returns
        -------
        dict
            ``{"artifact_detection_id": ...}``, the content-addressed PK.
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            shared_group_member_set_hash,
        )

        group_name = key["shared_artifact_group_name"]
        member_recording_ids = (
            SharedArtifactGroup.Member
            & {"shared_artifact_group_name": group_name}
        ).fetch("recording_id")
        member_set_hash = shared_group_member_set_hash(member_recording_ids)
        return _insert_artifact_selection(
            cls,
            artifact_detection_params_name=key[
                "artifact_detection_params_name"
            ],
            shared_artifact_group_name=group_name,
            supplied_id=key.get("artifact_detection_id"),
            manual_excluded_times=key.get("manual_excluded_times"),
            extra_row={
                "shared_artifact_group_name": group_name,
                "member_set_hash": member_set_hash,
            },
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


class SharedGroupArtifactFetched(NamedTuple):
    """DB-side inputs for :meth:`SharedGroupArtifactDetection.make_fetch`.

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

    # Tri-part make: the long detection scan runs outside the transaction.
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
        from spyglass.spikesorting.v2._core.job_config import (
            _resolved_job_kwargs,
        )

        resolved_job_kwargs = _resolved_job_kwargs(artifact_job_kwargs)
        valid_times = detect_artifacts(
            recording,
            validated,
            context=context,
            job_kwargs=resolved_job_kwargs,
        )
        from spyglass.spikesorting.v2._artifacts.manual import (
            apply_manual_exclusions,
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
        rows, and the ``ArtifactDetectionOutput`` registration.

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
            IntervalList.insert(interval_rows)
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

    def _merge_registration(self, row):
        """Return merge registration and count of its selection dependencies.

        Counts sorting, motion-estimate and concat selections. Concat member
        FKs can reference a detection directly even if its merge registration
        is absent.
        """
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )
        from spyglass.spikesorting.v2.motion import MotionEstimateSelection
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecordingSelection,
        )
        from spyglass.spikesorting.v2.sorting import SortingSelection

        det_key = {"artifact_detection_id": row["artifact_detection_id"]}
        n_concat = len(ConcatenatedRecordingSelection.MemberSnapshot & det_key)
        try:
            merge_id = ArtifactDetectionOutput.get_merge_id(det_key)
        except KeyError:
            return None, n_concat
        merge_key = {"artifact_detection_merge_id": merge_id}
        n = len(SortingSelection.ArtifactDetectionSource & merge_key) + len(
            MotionEstimateSelection.ArtifactDetectionSource & merge_key
        )
        return merge_id, n + n_concat

    def delete(self, *args, safemode=None, _cascade_sorts=False, **kwargs):
        """Delete detections, their merge registration and owned intervals.

        Refused when a ``SortingSelection``, ``MotionEstimateSelection`` or
        ``ConcatenatedRecordingSelection`` member references a detection;
        delete those first or use :meth:`cascade_delete`. Owned
        ``IntervalList`` rows are removed for detections actually deleted.
        A leading positional restriction is accepted.

        Raises
        ------
        ValueError
            If a detection is referenced and ``_cascade_sorts`` is False.
        AdvisoryLockError
            If a per-detection lifecycle lock cannot be taken in time.
        """
        restricted, args = restrict_by_leading_args(self, args)
        if restricted is not None:
            return restricted.delete(
                *args,
                safemode=safemode,
                _cascade_sorts=_cascade_sorts,
                **kwargs,
            )

        from spyglass.spikesorting.v2.artifact_output import (
            artifact_lifecycle_locks,
        )

        detection_cls = type(self)
        rows = self.fetch(as_dict=True)

        # Same lock as insert_selection; closes the check-then-cascade race.
        with artifact_lifecycle_locks(
            row["artifact_detection_id"] for row in rows
        ):
            if not _cascade_sorts:
                referenced = [
                    str(row["artifact_detection_id"])
                    for row in rows
                    if self._merge_registration(row)[1]
                ]
                if referenced:
                    raise ValueError(
                        f"{detection_cls.__name__}.delete: refusing to delete "
                        "artifact detection(s) referenced by a SortingSelection, "
                        "MotionEstimateSelection or ConcatenatedRecordingSelection:"
                        f" {referenced}. Delete those selections and outputs first, or call "
                        "cascade_delete() to remove them too."
                    )

            delete_targets = [
                (
                    {k: row[k] for k in self.primary_key},
                    collect_artifact_interval_rows_to_remove(
                        [row], detection_cls=detection_cls
                    ),
                )
                for row in rows
            ]

            # force_masters cascade removes the merge registration too.
            if safemode is None:
                super().delete(*args, **kwargs)
            else:
                super().delete(*args, safemode=safemode, **kwargs)

        remove_artifact_interval_rows(
            [
                restriction
                for master_key, interval_rows_to_remove in delete_targets
                if not (detection_cls & master_key)
                for restriction in interval_rows_to_remove
            ]
        )

    def cascade_delete(self, *args, safemode=None, **kwargs):
        """Delete these detections and their dependent sorts and concats.

        Removes dependent ``ConcatenatedRecordingSelection``,
        ``SortingSelection`` and ``MotionEstimateSelection`` rows, their
        downstream outputs, and the merge registration.
        """
        return self.delete(
            *args, safemode=safemode, _cascade_sorts=True, **kwargs
        )


@schema
class RecordingArtifactDetection(
    _ArtifactDetectionMixin, SpyglassMixin, dj.Computed
):
    """Artifact-removed valid times for a single ``Recording``.

    Writes one ``IntervalList`` row. Always reads the uncorrected
    ``Recording``; motion correction is downstream of the mask.
    """

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

    def make_fetch(self, key):
        """Read the params blob, resolved recording, and parent session.

        A missing cached artifact is rebuilt through ``Recording``.

        Returns
        -------
        RecordingArtifactFetched
            Validated params, the resolved ``recording_id``, its parent
            ``nwb_file_name``, the per-row job kwargs and the resolved
            artifact.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

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
            "params",
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

        recording = read_stored_traces(traces)
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
class SharedGroupArtifactDetection(
    _ArtifactDetectionMixin, SpyglassMixin, dj.Computed
):
    """Artifact-removed valid times for a ``SharedArtifactGroup``.

    ``make_fetch`` rejects a member set that differs from
    ``member_set_hash``. ``make_compute`` scans the members' unioned channels
    once; ``make_insert`` writes the result for every member session.
    """

    definition = """
    -> SharedGroupArtifactSelection
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

    def make_fetch(self, key):
        """Read params, resolve members, and reject a member-set drift.

        Missing cached artifacts are rebuilt through ``Recording``.

        Returns
        -------
        SharedGroupArtifactFetched
            Validated params, the group name, the ordered member
            ``recording_id`` / ``nwb_file_name`` tuples, the canonical parent
            ``nwb_file_name``, the per-row job kwargs and the resolved member
            artifacts.

        Raises
        ------
        SharedArtifactGroupMemberDriftError
            If the live member set differs from ``member_set_hash``.
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            shared_group_member_set_hash,
        )
        from spyglass.spikesorting.v2.exceptions import (
            SharedArtifactGroupMemberDriftError,
        )
        from spyglass.spikesorting.v2.recording import RecordingSelection

        (
            group_name,
            frozen_hash,
            manual_excluded_times,
            params_blob,
            artifact_job_kwargs,
        ) = (
            ArtifactDetectionParameters * (SharedGroupArtifactSelection & key)
        ).fetch1(
            "shared_artifact_group_name",
            "member_set_hash",
            "manual_excluded_times",
            "params",
            "job_kwargs",
        )
        validated = ArtifactDetectionParamsSchema.model_validate(params_blob)

        # Ordered so both make_fetch calls DeepHash equal.
        members = (
            SharedArtifactGroup.Member * RecordingSelection
            & {"shared_artifact_group_name": group_name}
        ).fetch(
            "recording_id",
            "nwb_file_name",
            as_dict=True,
            order_by="recording_id",
        )
        if not members:
            raise RuntimeError(
                "SharedGroupArtifactDetection.make: SharedArtifactGroup "
                f"{group_name!r} has zero members; insert_group should have "
                "rejected the empty case."
            )
        member_recording_ids = tuple(str(m["recording_id"]) for m in members)
        current_hash = shared_group_member_set_hash(member_recording_ids)
        if current_hash != frozen_hash:
            raise SharedArtifactGroupMemberDriftError(
                "SharedGroupArtifactDetection.make: SharedArtifactGroup "
                f"{group_name!r} member set changed since "
                f"artifact_detection_id={key['artifact_detection_id']} was "
                f"created (frozen member_set_hash {frozen_hash[:12]} != "
                f"current {current_hash[:12]}). The artifact_detection_id "
                "identity is params+group only, so insert_selection() returns "
                "this same id with its stale snapshot -- to scan the new "
                "membership, DELETE this selection and re-run "
                "insert_selection() (which re-snapshots the current members), "
                "or restore the group's members."
            )
        member_nwb_file_names = tuple(m["nwb_file_name"] for m in members)
        return SharedGroupArtifactFetched(
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
        import spikeinterface as si

        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces

        if not member_recording_ids:
            raise RuntimeError(
                "SharedGroupArtifactDetection.make_compute: shared-group "
                "source has no member_recording_ids; make_fetch contract "
                "violated."
            )
        per_member_recordings = [
            read_stored_traces(traces) for traces in member_traces
        ]
        # Re-check insert_group's invariants; a direct insert can bypass it.
        from spyglass.spikesorting.v2._artifacts.shared_group import (
            assert_shared_group_recordings_aggregatable,
        )

        assert_shared_group_recordings_aggregatable(
            per_member_recordings,
            member_recording_ids,
            member_nwb_file_names,
        )
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

    if SharedGroupArtifactDetection & artifact_detection_key:
        group_name = (
            SharedGroupArtifactSelection & artifact_detection_key
        ).fetch1("shared_artifact_group_name")
        if not (
            SharedArtifactGroup.Member
            & {
                "shared_artifact_group_name": group_name,
                "recording_id": recording_id,
            }
        ):
            raise ValueError(
                f"{caller}: artifact_detection_id "
                f"{artifact_detection_id!r} belongs to shared artifact group "
                f"{group_name!r}, which does not include requested "
                f"recording_id={target_recording_id!r}."
            )
        return

    raise ValueError(
        f"{caller}: artifact_detection_id "
        f"{artifact_detection_id!r} is not in RecordingArtifactDetection or "
        "SharedGroupArtifactDetection. Populate the artifact detection "
        "before linking it."
    )


# Declare the merge table now; make_insert cannot create it mid-transaction.
# Last because artifact_output imports the tables above.
from spyglass.spikesorting.v2 import (  # noqa: E402, F401
    artifact_output as _artifact_output,
)
