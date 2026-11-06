"""Grouping of sessions for chronic and concatenated-recording workflows.

``SessionGroup`` names a bundle of sorting members. ``ConcatenatedRecording``
builds one masked, unwhitened concatenation from each member's populated
``Recording``, which ``SortingSelection.ConcatenatedRecordingSource`` sorts as
one piece.

Tables:

- ``SessionGroup`` (+ ``Member``): Manual; user-facing grouping.
- ``ConcatenatedRecordingSelection``: Manual; UUID PK.
- ``ConcatenatedRecording`` (+ ``MemberBoundary``): Computed cache.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj

from spyglass.common import IntervalList, LabTeam, Session  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._recording import (
    concat_fetch as _concat_recording_fetch,
    session_group as _session_group_insert,
)
from spyglass.spikesorting.v2._storage import rebuilds as _recording_rebuilds
from spyglass.spikesorting.v2._storage.nwb import StoredTraces
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
    StagedOutputs,
)
from spyglass.spikesorting.v2.artifact import (
    RecordingArtifactDetection,  # noqa: F401
)
from spyglass.spikesorting.v2.recording import (
    PreprocessingParameters,  # noqa: F401
    SortGroupV2,  # noqa: F401
)
from spyglass.spikesorting.v2._core.table_integrity import (
    FactoryOnlyMaster,
    SelectionMasterInsertGuard,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

if TYPE_CHECKING:
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_session_group")


def _member_electrode_signature(member: dict) -> tuple:
    """Return a member sort group's electrode/region signature.

    Electrodes are keyed by ``(electrode_group_name, electrode_id)`` because
    ids can repeat across groups. ``electrode_id`` is stable across an
    animal's sessions, so members of the same implant share a signature. An
    electrode without a region maps to ``None``.
    """
    from spyglass.spikesorting.v2._recording.concat import (
        electrode_signature_from_rows,
    )
    from spyglass.spikesorting.v2._recording.unit_metadata import (
        sort_group_electrode_regions,
    )

    electrode_rows = sort_group_electrode_regions(
        {
            "nwb_file_name": member["nwb_file_name"],
            "sort_group_id": member["sort_group_id"],
        }
    ).fetch("electrode_group_name", "electrode_id", "region_name", as_dict=True)
    region_by_key = {
        (str(row["electrode_group_name"]), int(row["electrode_id"])): str(
            row["region_name"]
        )
        for row in electrode_rows
    }
    return electrode_signature_from_rows(electrode_rows, region_by_key)


def distinct_recording_dates(session_start_times) -> list:
    """Distinct calendar dates of session start times, ascending.

    Two or more dates means the recordings span multiple days.

    Parameters
    ----------
    session_start_times : iterable of datetime.datetime
        ``Session.session_start_time`` values.

    Returns
    -------
    list of datetime.date
    """
    return sorted({start_time.date() for start_time in session_start_times})


def assert_members_share_electrode_space(members: list[dict]) -> None:
    """Reject a SessionGroup whose members map to different electrode spaces.

    Members can share a channel layout yet reference different physical
    electrodes or regions. The concatenation is read in the anchor (lowest
    ``member_index``) member's electrode frame, so each member must match it.

    Parameters
    ----------
    members : list of dict
        ``SessionGroup.Member`` rows (each carrying ``nwb_file_name``,
        ``sort_group_id``, ``member_index``).

    Raises
    ------
    ValueError
        If any member's electrode/region signature differs from the anchor's.
    """
    if len(members) < 2:
        return
    ordered = sorted(members, key=lambda m: m["member_index"])
    anchor = ordered[0]
    anchor_signature = _member_electrode_signature(anchor)
    for member in ordered[1:]:
        signature = _member_electrode_signature(member)
        if signature != anchor_signature:
            raise ValueError(
                "ConcatenatedRecordingSelection.insert_selection: member "
                f"(index {member['member_index']}, "
                f"sort_group_id={member['sort_group_id']}, "
                f"nwb_file_name={member['nwb_file_name']!r}) maps to a "
                "different electrode space than the anchor member (index "
                f"{anchor['member_index']}): its sort group's electrode "
                "ids/brain regions differ. Concatenated members must share the "
                "same physical electrodes so the result reads correctly in the "
                "anchor frame -- check that every member uses the same sort "
                "group / probe layout."
            )


@schema
class SessionGroup(FactoryOnlyMaster, SpyglassMixin, dj.Manual):
    """A named bundle of sorting members to analyze together.

    A member is a ``(nwb_file_name, sort_group_id, interval_list_name,
    team_name)`` tuple, not necessarily a whole NWB file.
    ``session_group_owner`` is a projected ``LabTeam.team_name`` so teams can
    reuse group names. Same-day groups are the default; multi-day requires
    ``allow_multi_day=True``.

    Create groups with :meth:`create_group`; direct ``insert`` and
    ``update1`` are blocked.
    """

    #: Named in ``FactoryOnlyMaster``'s reject messages.
    _factory_create_call = "SessionGroup.create_group()"

    definition = """
    -> LabTeam.proj(session_group_owner='team_name')
    session_group_name: varchar(64)
    ---
    description: varchar(255)
    """

    class Member(SpyglassMixinPart):
        """One sorting member belonging to a ``SessionGroup``."""

        definition = """
        -> master
        member_index: int
        ---
        -> Session
        -> SortGroupV2
        -> IntervalList
        -> LabTeam
        """

    @classmethod
    def create_group(
        cls,
        session_group_owner: str,
        session_group_name: str,
        members: list[dict],
        description: str = "",
        allow_multi_day: bool = False,
    ) -> None:
        """Atomically insert the master + Member rows for a sorting group.

        Each member dict carries ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name`` and an optional ``team_name`` (the team that
        owns that member's ``RecordingSelection``; defaults to
        ``session_group_owner``). One NWB may contribute several members.

        Recording dates come from ``Session.session_start_time``. For sessions
        days apart, sort each and match units instead of concatenating.

        Parameters
        ----------
        session_group_owner : str
            ``LabTeam.team_name`` that owns (namespaces) the group.
        session_group_name : str
            Group name, unique within ``session_group_owner``.
        members : list of dict
            Member dicts (see above). Order sets ``member_index``.
        description : str, optional
            Free-text group description. Default ``""``.
        allow_multi_day : bool, optional
            Opt in to multi-date members. Default ``False``.

        Raises
        ------
        SessionGroupInputError
            If ``members`` is empty, a member dict is missing a required key
            (``nwb_file_name`` / ``sort_group_id`` / ``interval_list_name``),
            references a non-ingested session, duplicates another member, or
            references a ``SortGroupV2`` / ``IntervalList`` / ``LabTeam`` that
            does not exist.
        SessionGroupDateError
            If a member dict carries ``recording_date`` (dates are derived),
            or if members span multiple dates without ``allow_multi_day=True``.
        """
        from spyglass.spikesorting.v2.exceptions import SessionGroupInputError

        rows = _session_group_insert.group_member_rows(
            session_group_owner, session_group_name, members, allow_multi_day
        )

        try:
            with cls.connection.transaction:
                # Rows were validated above.
                cls.insert1(
                    {
                        "session_group_owner": session_group_owner,
                        "session_group_name": session_group_name,
                        "description": description,
                    },
                    allow_direct_insert=True,
                )
                cls.Member.insert(rows)
        except dj.errors.IntegrityError as exc:
            raise SessionGroupInputError(
                "SessionGroup.create_group: a member references a SortGroupV2, "
                "IntervalList, or LabTeam row that does not exist. Verify each "
                "member's sort_group_id / interval_list_name / team_name "
                f"(create the sort groups / team first). ({exc})"
            ) from exc


@schema
class ConcatenatedRecordingSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One selection per group, recipes, and frozen member/artifact choices.

    UUID-keyed so downstream FKs are single-column. ``insert_selection``
    requires every member to have a populated ``Recording`` under the
    requested preprocessing parameters.
    """

    definition = """
    concat_recording_id: uuid
    ---
    -> SessionGroup
    -> PreprocessingParameters
    member_set_hash: char(64)   # sha256 of the ordered frozen member set; also folded into concat_recording_id
    """

    class MemberSnapshot(SpyglassMixinPart):
        """Frozen per-member identity captured when the concat id was minted.

        Holds each member's identity and resolved ``Recording``
        (``recording_id``, ``recording_content_hash``); the ordered set is
        hashed into ``concat_recording_id``. Concat reads use this, never the
        live ``SessionGroup.Member`` rows, so a group edit does not affect an
        existing concat row. ``recording_content_hash`` detects member drift
        (``ConcatMemberDriftError``). The artifact detection is a foreign key
        so it cannot disappear.
        """

        definition = """
        -> master
        member_index: int
        ---
        nwb_file_name: varchar(64)
        sort_group_id: int
        interval_list_name: varchar(170)
        team_name: varchar(80)
        recording_id: uuid
        recording_content_hash: char(64)
        -> [nullable] RecordingArtifactDetection
        """

    #: Identity fields besides ``member_set_hash``, which covers ordered
    #: members and their artifact choices.
    _IDENTITY_FIELDS = (
        "session_group_owner",
        "session_group_name",
        "preprocessing_params_name",
    )

    @classmethod
    def insert_selection(
        cls,
        key: dict,
        *,
        artifact_detection_ids: Mapping[int, uuid.UUID | str | None],
    ) -> dict:
        """Find-existing-or-insert a concat selection; return a PK-only dict.

        Every member must already have a populated ``Recording`` under
        ``preprocessing_params_name``, so ``ConcatenatedRecording.make`` never
        populates inline. Idempotent for the same identity and member set.

        Duplicate-key recovery has no savepoint; it relies on the master
        insert being the transaction's first statement.

        Parameters
        ----------
        key : dict
            Must carry ``session_group_owner``, ``session_group_name``, and
            ``preprocessing_params_name``, and nothing else.
            A caller-supplied ``concat_recording_id`` is ignored; the id is
            minted/found here.
        artifact_detection_ids : mapping
            Exact populated detection ID for every member index. An explicit
            ``None`` disables masking for that member. Each detection must
            belong to that member's recording; populate detections first.

        Returns
        -------
        dict
            A PK-only dict ``{"concat_recording_id": ...}`` for the
            existing-or-inserted selection row.

        Raises
        ------
        ValueError
            If a required identity field is missing, or ``key`` carries any
            other field.
        MissingRecordingForConcatError
            If any member has no populated ``Recording`` under
            ``preprocessing_params_name``.
        DuplicateSelectionError
            If more than one selection row already matches the identity (a raw
            insert bypassed this helper).
        """
        identity, set_hash, concat_recording_id, snapshot_rows, artifacts = (
            _session_group_insert.plan_concat_selection(
                cls, key, artifact_detection_ids
            )
        )

        existing = cls._find_existing_pk(
            identity, set_hash, concat_recording_id
        )
        if existing is not None:
            return existing
        snapshot_inserts = [
            {"concat_recording_id": concat_recording_id, **row}
            for row in snapshot_rows
        ]
        from spyglass.spikesorting.v2.artifact_output import (
            artifact_lifecycle_locks,
        )

        detection_ids = [
            value for value in artifacts.values() if value is not None
        ]
        if detection_ids and cls.connection.in_transaction:
            raise ValueError(
                "Create an artifact-backed concat selection outside a caller-owned "
                "transaction so its artifact lifecycle locks cover the commit."
            )
        try:
            # Validated above; master and snapshot commit together.
            with artifact_lifecycle_locks(detection_ids):
                with cls._safe_context():
                    cls.insert1(
                        {
                            **identity,
                            "member_set_hash": set_hash,
                            "concat_recording_id": concat_recording_id,
                        },
                        allow_direct_insert=True,
                    )
                    cls.MemberSnapshot.insert(snapshot_inserts)
        except dj.errors.DuplicateError:
            existing = cls._find_existing_pk(
                identity, set_hash, concat_recording_id
            )
            if existing is not None:
                return existing
            raise
        return {"concat_recording_id": concat_recording_id}

    @classmethod
    def _find_existing_pk(
        cls, identity: dict, member_set_hash, deterministic_concat_recording_id
    ) -> dict | None:
        """Return the canonical PK for this selection identity, or None.

        Matches on identity plus ``member_set_hash``, so selections over
        different member sets do not collide. A match whose id is not the
        deterministic one came from a raw insert and is rejected.
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            existing_selection_pk,
        )

        lookup = {**identity, "member_set_hash": member_set_hash}
        master_ids = {
            row["concat_recording_id"]
            for row in (cls & lookup).fetch("KEY", as_dict=True)
        }
        return existing_selection_pk(
            master_ids,
            deterministic_concat_recording_id,
            pk_field="concat_recording_id",
            bypass_message=lambda bypassed: (
                f"ConcatenatedRecordingSelection has {len(master_ids)} master "
                f"row(s) for identity {identity} (member_set_hash "
                f"{member_set_hash}) whose concat_recording_id is not the "
                f"deterministic id {deterministic_concat_recording_id}: "
                f"{bypassed}. This is a non-deterministic selection row (a raw "
                "insert bypassing insert_selection); drop it and re-insert "
                "via insert_selection."
            ),
        )


class ConcatRecordingFetched(NamedTuple):
    """DB-side inputs for ``ConcatenatedRecording.make_compute``.

    ``member_plan`` is the ``member_index``-ordered plan from
    :func:`._recording.concat_fetch.resolve_snapshot_recordings`,
    existence-checked at fetch time.
    ``member_traces`` holds each member's file, aligned with ``member_plan``
    and rebuilt at fetch time if missing.
    """

    member_plan: list[dict]
    member_traces: tuple[StoredTraces, ...]
    preprocessing_params_name: str
    anchor_nwb_file_name: str


class ConcatRecordingComputed(NamedTuple):
    """Compute -> insert carrier for ``ConcatenatedRecording``.

    ``member_boundaries`` holds per-member ``{"member_index" (int),
    "end_sample" (int)}`` rows; ``make_insert`` adds the master PK.
    """

    analysis_file_name: str
    object_id: str
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    n_samples: int
    content_hash: str
    anchor_nwb_file_name: str
    member_boundaries: list[dict]
    obs_intervals: object
    # (n, 2) int64 half-open artifact-free frame ranges within one member's
    # continuous acquisition; estimators sample only inside them.
    statistics_spans: object
    # (n, 2) int64 frame ranges of uninterrupted acquisition, and (n,) float64
    # first/last times on the member's own clock.
    continuity_spans: object
    continuity_start_s: object
    continuity_end_s: object

    def staged_outputs(self) -> StagedOutputs:
        """The staged analysis file ``make_insert`` registers."""
        return StagedOutputs(analysis_file_names=(self.analysis_file_name,))


@schema
class ConcatenatedRecording(
    StagedOutputCleanupMixin, SpyglassMixin, dj.Computed
):
    """Materialized cross-session concatenated recording cache.

    Writes one masked, unwhitened ``ElectricalSeries`` over the ordered
    members (no motion correction), plus cumulative member sample boundaries
    in ``MemberBoundary``.
    """

    definition = """
    -> ConcatenatedRecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255)
    object_id: varchar(72)
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    n_samples: bigint            # concat sample count; exact integer basis for the MemberBoundary back-mapping
    content_hash: char(64)
    obs_intervals: longblob     # kept intervals on the synthetic concat timeline, in seconds
    statistics_spans: longblob  # (n, 2) int64 half-open concat frame ranges: artifact-free and never crossing a member join or a member-internal timestamp gap
    continuity_spans: longblob  # (n, 2) int64 half-open concat frame ranges of uninterrupted acquisition, split at every member join and member-internal timestamp gap
    continuity_start_s: longblob  # (n,) float64 first timestamp of each continuity span on its member's own acquisition clock, in seconds
    continuity_end_s: longblob  # (n,) float64 last timestamp of each continuity span on its member's own acquisition clock, in seconds
    """

    class MemberBoundary(SpyglassMixinPart):
        """Per-member end-sample boundary in the concatenated recording."""

        definition = """
        -> master
        member_index: int
        ---
        end_sample: bigint
        member_valid_times: longblob  # kept intervals on the original member timeline
        """

    @staticmethod
    def _load_member_recordings(member_plan, member_traces):
        """Load each member's cached ``Recording`` from the resolved plan (SI I/O).

        No DB access; files were resolved at fetch time.

        Parameters
        ----------
        member_plan : list[dict]
            Resolved per-member plan dicts from
            :func:`._recording.concat_fetch.resolve_snapshot_recordings`.
        member_traces : tuple[StoredTraces, ...]
            Each member's resolved file, aligned with ``member_plan``.

        Returns
        -------
        tuple[list, list[int], list[int]]
            ``(recordings, member_sample_counts, member_indices)`` in
            ``member_index`` order.
        """
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces

        recordings = []
        member_sample_counts = []
        member_indices = []
        for plan, traces in zip(member_plan, member_traces, strict=True):
            recording = read_stored_traces(traces)
            recordings.append(recording)
            member_sample_counts.append(int(recording.get_num_samples()))
            member_indices.append(int(plan["member_index"]))
        return recordings, member_sample_counts, member_indices

    # Tri-part make keeps the concat and NWB write outside the transaction.
    _parallel_make = True

    def make_fetch(self, key) -> ConcatRecordingFetched:
        """Read every DB input the materialization needs.

        Reads the frozen member snapshot and each member's ``Recording``,
        rebuilding a missing member file. No other SI or NWB I/O.

        Parameters
        ----------
        key : dict
            The ``ConcatenatedRecordingSelection`` primary key
            (``{"concat_recording_id": ...}``) being populated.

        Returns
        -------
        ConcatRecordingFetched

        Raises
        ------
        SchemaBypassError
            If the selection has no frozen ``MemberSnapshot`` (a raw insert that
            bypassed ``insert_selection``).
        MissingRecordingForConcatError
            If any frozen member's ``Recording`` row is gone.
        ConcatMemberDriftError
            If a frozen member's recording content drifted from the snapshot.
        """
        return _concat_recording_fetch.fetch_concat_inputs(key)

    def make_compute(
        self,
        key,
        member_plan,
        member_traces,
        preprocessing_params_name,
        anchor_nwb_file_name,
    ) -> ConcatRecordingComputed:
        """Materialize the concat cache outside any DB transaction.

        Zeros each member's artifact intervals, stitches the members into one
        segment, and writes one ``ElectricalSeries``. No motion correction or
        whitening. Masked samples are 0 uV; members with a nonzero channel
        offset are stored as float32 microvolts, others keep their stored
        samples and calibration (``mask_member_recordings``).

        Returns
        -------
        ConcatRecordingComputed

        Raises
        ------
        RuntimeError
            If the stitched recording's sample count differs from the summed
            member sample counts (which would misalign the ``MemberBoundary``
            back-mapping).
        """
        from spyglass.spikesorting.v2._recording.concat import (
            build_concatenated_recording,
            concat_provenance_tables,
            concat_span_arrays,
            cumulative_member_boundaries,
            mask_member_recordings,
            observation_intervals,
        )
        from spyglass.spikesorting.v2._storage.nwb import write_nwb_artifact
        from spyglass.spikesorting.v2._core.signal_math import (
            base_intervals_and_gaps,
        )

        recordings, member_sample_counts, member_indices = (
            self._load_member_recordings(member_plan, member_traces)
        )
        member_valid_times = [
            (
                plan["valid_times"]
                if plan["valid_times"] is not None
                else base_intervals_and_gaps(
                    recording, recording.get_sampling_frequency()
                ).base_intervals
            )
            for plan, recording in zip(member_plan, recordings, strict=True)
        ]
        masked_recordings, artifact_ranges = mask_member_recordings(
            recordings, [plan["valid_times"] for plan in member_plan]
        )
        # Use member timestamps; the concatenation drops their gaps.
        continuity_spans, continuity_start_s, continuity_end_s, statistics = (
            concat_span_arrays(
                recordings, member_sample_counts, artifact_ranges
            )
        )
        recordings = masked_recordings

        concatenated = build_concatenated_recording(recordings)

        # Compute before staging the NWB so a failure cannot orphan a file.
        boundaries = cumulative_member_boundaries(member_sample_counts)
        concat_n_samples = int(concatenated.get_num_samples())
        # A sample-count mismatch would misalign split_sorting_by_session.
        if boundaries and concat_n_samples != boundaries[-1]:
            raise RuntimeError(
                "ConcatenatedRecording.make: the concatenated recording has "
                f"{concat_n_samples} samples but the cumulative member sample "
                f"count is {boundaries[-1]}; the concatenation must preserve "
                "the sample count or the MemberBoundary back-mapping would be "
                "wrong."
            )
        sampling_frequency = float(concatenated.get_sampling_frequency())
        obs_intervals = observation_intervals(
            concat_n_samples, sampling_frequency, artifact_ranges
        )
        n_channels = int(concatenated.get_num_channels())
        total_duration_s = concat_n_samples / sampling_frequency

        # The analysis NWB's parent is the first member's session.
        provenance_tables = concat_provenance_tables(
            concat_recording_id=key["concat_recording_id"],
            preprocessing_params_name=preprocessing_params_name,
            anchor_nwb_file_name=anchor_nwb_file_name,
            member_plan=member_plan,
            member_sample_counts=member_sample_counts,
            boundaries=boundaries,
            artifact_ranges=artifact_ranges,
            obs_intervals=obs_intervals,
        )
        analysis_file_name, object_id, content_hash = write_nwb_artifact(
            concatenated,
            anchor_nwb_file_name,
            filtering_description=(
                f"Concatenated {len(recordings)} member recording(s) "
                f"(preprocessing_params={preprocessing_params_name!r}); "
                "selected member artifact masks applied; no motion "
                "correction; unwhitened"
            ),
            provenance_tables=provenance_tables,
        )
        member_boundaries = [
            {
                "member_index": member_index,
                "end_sample": int(end_sample),
                "member_valid_times": valid_times,
            }
            for member_index, end_sample, valid_times in zip(
                member_indices, boundaries, member_valid_times, strict=True
            )
        ]
        return ConcatRecordingComputed(
            analysis_file_name=analysis_file_name,
            object_id=object_id,
            n_channels=n_channels,
            sampling_frequency=sampling_frequency,
            total_duration_s=total_duration_s,
            # Exact integer; avoids a float total_duration_s * fs round trip.
            n_samples=concat_n_samples,
            content_hash=content_hash,
            anchor_nwb_file_name=anchor_nwb_file_name,
            member_boundaries=member_boundaries,
            obs_intervals=obs_intervals,
            statistics_spans=statistics,
            continuity_spans=continuity_spans,
            continuity_start_s=continuity_start_s,
            continuity_end_s=continuity_end_s,
        )

    def make_insert(
        self,
        key,
        analysis_file_name,
        object_id,
        n_channels,
        sampling_frequency,
        total_duration_s,
        n_samples,
        content_hash,
        anchor_nwb_file_name,
        member_boundaries,
        obs_intervals,
        statistics_spans,
        continuity_spans,
        continuity_start_s,
        continuity_end_s,
    ):
        """Atomically register the staged concat artifact + boundary rows.

        ``_safe_context()`` makes a direct (non-populate) call atomic. A
        direct call must remove a failed attempt's staged file itself.
        """
        from spyglass.spikesorting.v2.recording import _ELECTRICAL_SERIES_PATH

        boundary_rows = [
            {
                **key,
                "member_index": boundary["member_index"],
                "end_sample": int(boundary["end_sample"]),
                "member_valid_times": boundary["member_valid_times"],
            }
            for boundary in member_boundaries
        ]
        with self._safe_context():
            AnalysisNwbfile().add(anchor_nwb_file_name, analysis_file_name)
            self.insert1(
                {
                    **key,
                    "analysis_file_name": analysis_file_name,
                    "electrical_series_path": _ELECTRICAL_SERIES_PATH,
                    "object_id": object_id,
                    "n_channels": n_channels,
                    "sampling_frequency": sampling_frequency,
                    "total_duration_s": total_duration_s,
                    "n_samples": n_samples,
                    "content_hash": content_hash,
                    "obs_intervals": obs_intervals,
                    "statistics_spans": statistics_spans,
                    "continuity_spans": continuity_spans,
                    "continuity_start_s": continuity_start_s,
                    "continuity_end_s": continuity_end_s,
                }
            )
            self.MemberBoundary.insert(boundary_rows)

    def get_recording(self, key) -> "si.BaseRecording":  # noqa: F821
        """Return the cached concatenated SpikeInterface recording.

        Rebuilds a missing file and verifies it against ``content_hash``.
        Annotated ``is_filtered=True`` so sorters do not re-filter.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``ConcatenatedRecording`` row.

        Returns
        -------
        si.BaseRecording
            The concatenated, masked, unwhitened recording.
        """
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces
        from spyglass.spikesorting.v2._storage.rebuilds import stored_traces

        return read_stored_traces(
            stored_traces(type(self), key, (self & key).fetch1())
        )

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild a missing concat artifact (locked, atomic, verified).

        :func:`._storage.rebuilds.ensure_artifact_file` calls this on every
        trace-artifact table.
        """
        return _recording_rebuilds.rebuild_concat_nwb_artifact(self, key)

    def split_sorting_by_session(self, sorting, key) -> dict:
        """Back-map a concat-frame sorting into per-member local sortings.

        Uses the ``MemberBoundary`` rows. Unit ids are preserved across
        members.

        Parameters
        ----------
        sorting : si.BaseSorting
            The sorting in the CONCATENATED recording's sample frame (e.g.
            ``Sorting.get_analyzer(sorting_key).sorting`` or
            ``Sorting.get_sorting(sorting_key)``).
        key : dict
            Restriction selecting a single ``ConcatenatedRecording`` row.

        Returns
        -------
        dict[tuple[str, int, str, str], si.BaseSorting]
            One ``NumpySorting`` per member, keyed by the hashable member
            identity ``(nwb_file_name, sort_group_id, interval_list_name,
            team_name)``, with spike times in that member's local frames.
            See :func:`._recording.concat.member_split_key`.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._recording.concat import (
            member_split_key,
            split_unit_spike_trains,
        )
        from spyglass.spikesorting.v2.exceptions import ConcatSplitError

        # Use the frozen members so a SessionGroup.Member edit cannot misalign.
        members = (ConcatenatedRecordingSelection.MemberSnapshot & key).fetch(
            as_dict=True, order_by="member_index"
        )
        indices, ends = (self.MemberBoundary & key).fetch(
            "member_index", "end_sample"
        )
        end_by_index = {int(i): int(e) for i, e in zip(indices, ends)}
        member_indices = {int(member["member_index"]) for member in members}
        if set(end_by_index) != member_indices:
            raise ConcatSplitError(
                "ConcatenatedRecording.split_sorting_by_session: the "
                "MemberBoundary set "
                f"{sorted(end_by_index)} does not match exactly one boundary "
                f"per frozen member {sorted(member_indices)}. The boundaries "
                "and the frozen member snapshot are inconsistent; the concat "
                "cache may be partially written -- repopulate it."
            )
        boundaries = [
            end_by_index[int(member["member_index"])] for member in members
        ]

        unit_trains = {
            int(unit_id): sorting.get_unit_spike_train(unit_id=unit_id)
            for unit_id in sorting.unit_ids
        }
        # total_n_samples catches a final boundary that is too long, which
        # in-range spikes alone cannot detect.
        n_samples = int((self & key).fetch1("n_samples"))
        per_member = split_unit_spike_trains(
            unit_trains, boundaries, total_n_samples=n_samples
        )
        fs = float(sorting.get_sampling_frequency())
        return {
            member_split_key(member): si.NumpySorting.from_unit_dict(
                [local_trains], sampling_frequency=fs
            )
            for member, local_trains in zip(members, per_member)
        }
