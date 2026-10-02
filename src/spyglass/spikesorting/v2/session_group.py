"""Grouping of sessions for chronic and concatenated-recording workflows.

Implements same-day chronic concatenate-and-sort: ``SessionGroup`` names a
bundle of sorting members, and ``ConcatenatedRecording`` materializes one
masked, unwhitened concatenated recording cache from each member's
already-populated ``Recording`` artifact. A ``SortingSelection`` then FKs
``ConcatenatedRecording`` (via its ``ConcatenatedRecordingSource`` part) and
sorts the concatenation as one piece.

Tables:
    SessionGroup (+ Member)                  -- Manual; user-facing grouping.
    ConcatenatedRecordingSelection           -- Manual; UUID PK.
    ConcatenatedRecording (+ MemberBoundary) -- Computed; materialized cache.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj

from spyglass.common import IntervalList, LabTeam, Session  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2 import (
    _concat_recording_fetch,
    _recording_nwb,
    _session_group_insert,
)
from spyglass.spikesorting.v2._recording_nwb import StoredTraces
from spyglass.spikesorting.v2._staged_outputs import (
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
from spyglass.spikesorting.v2.utils import (
    FactoryOnlyMaster,
    SelectionMasterInsertGuard,
    _validate_params,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

if TYPE_CHECKING:
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_session_group")


def _member_electrode_signature(member: dict) -> tuple:
    """Return a member sort group's electrode/region signature.

    Fetches the sort group's electrodes (each with its owning
    ``electrode_group_name``) and per-electrode brain region, then delegates the
    signature shape to
    :func:`._concat_recording.electrode_signature_from_rows`. Each electrode is
    keyed by ``(electrode_group_name, electrode_id)`` because the ``Electrode``
    primary key carries the group and ids can repeat across groups -- so two
    physically distinct probes with reused ids and matching regions never
    collapse into one electrode space. ``electrode_id`` is per-NWB but stable
    across an animal's sessions, so two members of the same implant share a
    signature while members on different sort groups, probes, or regions
    diverge. An electrode without a region maps to ``None``.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2._concat_recording import (
        electrode_signature_from_rows,
    )

    restriction = {
        "nwb_file_name": member["nwb_file_name"],
        "sort_group_id": member["sort_group_id"],
    }
    electrode_rows = (SortGroupV2.SortGroupElectrode & restriction).fetch(
        "electrode_group_name", "electrode_id", as_dict=True
    )
    region_by_key = {
        (
            str(row["electrode_group_name"]),
            int(row["electrode_id"]),
        ): str(row["region_name"])
        for row in (
            (SortGroupV2.SortGroupElectrode & restriction)
            * Electrode
            * BrainRegion
        ).fetch(
            "electrode_group_name", "electrode_id", "region_name", as_dict=True
        )
    }
    return electrode_signature_from_rows(electrode_rows, region_by_key)


def distinct_recording_dates(session_start_times) -> list:
    """Distinct calendar dates of session start times, ascending.

    The multi-day criterion: a set of recordings spans more than one day when
    this returns two or more dates. ``SessionGroup.create_group`` (without
    ``allow_multi_day``), ``SessionGroup.is_multi_day`` and the UnitMatch
    same-day check on a concatenation input all apply it.

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

    SI's channel-id check (and the geometry/scaling checks in
    ``assert_concat_compatible``) compare the per-member SI recordings, but two
    members can share a local channel layout while their sort groups reference
    DIFFERENT physical electrodes or brain regions. The concatenated recording
    is read in the ANCHOR member's electrode frame, so a divergent member would
    be silently mis-attributed. Compare each member's electrode/region signature
    against the anchor (lowest ``member_index``) and raise on a mismatch.

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
    team_name)`` tuple, not necessarily a whole NWB file. The master PK is
    ``(session_group_owner, session_group_name)``; ``session_group_owner``
    is a projected ``LabTeam.team_name`` so two teams may both create a
    group named ``"day1"`` without collision.

    Same-day groups are the default; multi-day requires
    ``allow_multi_day=True`` (see ``create_group``).

    A direct ``insert`` / ``insert1`` and an in-place ``update1`` are blocked
    (``FactoryOnlyMaster``): a group is a provenance root that
    ``ConcatenatedRecordingSelection`` / ``UnitMatchSelection`` reference, so it
    must be written through :meth:`create_group` (which inserts the master + its
    ``Member`` rows atomically after validating member dates).
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

        ``session_group_owner`` namespaces user-facing group names in shared
        databases: two teams may both create ``"day1"`` because the master PK
        is ``(session_group_owner, session_group_name)``.

        Each member dict carries ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name`` and an optional ``team_name`` (the data-owner
        team used to resolve that member's ``RecordingSelection``; missing
        ``team_name`` defaults to ``session_group_owner`` for single-team
        groups, and mixed-team collaborations override it per member). A
        member is a sorting member tuple, not a whole-day abstraction: one
        NWB/day may contribute several members through different intervals or
        sort groups.

        Recording dates are DERIVED from each member's
        ``Session.session_start_time``, never stored on Member rows and never
        caller-supplied. Same-day groups are the default; members spanning two
        or more dates require ``allow_multi_day=True``. For days/weeks-apart
        sessions the recommended path is sort-then-match, not concatenation.

        Parameters
        ----------
        session_group_owner : str
            ``LabTeam.team_name`` that owns (namespaces) the group.
        session_group_name : str
            Group name, unique within ``session_group_owner``.
        members : list of dict
            Member tuples (see above). Order is preserved as ``member_index``.
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
                # create_group IS the validation boundary (it derived + checked
                # the member dates and shaped every Member row), so it bypasses
                # the FactoryOnlyMaster insert guard for its validated master.
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
            # A member's SortGroupV2 / IntervalList / LabTeam foreign key does
            # not exist -- translate the raw integrity error into a typed,
            # actionable one instead of leaking the DataJoint message.
            raise SessionGroupInputError(
                "SessionGroup.create_group: a member references a SortGroupV2, "
                "IntervalList, or LabTeam row that does not exist. Verify each "
                "member's sort_group_id / interval_list_name / team_name "
                f"(create the sort groups / team first). ({exc})"
            ) from exc

    @classmethod
    def is_multi_day(cls, key: dict) -> bool:
        """Report whether the group's members span two or more dates.

        Dates are derived from each member's ``Session.session_start_time``
        (the same source ``create_group`` validates against), never from a
        stored Member column.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``SessionGroup`` (e.g.
            ``{"session_group_owner": ..., "session_group_name": ...}``).

        Returns
        -------
        bool
            ``True`` iff the group's members span two or more session dates.
        """
        nwb_file_names = [
            {"nwb_file_name": n}
            for n in set((cls.Member & key).fetch("nwb_file_name"))
        ]
        if not nwb_file_names:
            return False
        # One batched Session query for all member sessions, not one fetch1
        # per member.
        start_times = (Session & nwb_file_names).fetch("session_start_time")
        return len(distinct_recording_dates(start_times)) > 1


@schema
class ConcatenatedRecordingSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One selection per group, recipes, and frozen member/artifact choices.

    UUID-keyed so downstream FKs are single-column (mirrors the
    single-session ``RecordingSelection`` / ``Recording`` shape). The
    ``insert_selection`` helper enforces that every member has a populated
    ``Recording`` row matching the requested preprocessing parameters before
    minting the ``concat_recording_id``.
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

        ``insert_selection`` freezes each ``SessionGroup.Member``'s ordered
        logical identity AND its resolved ``Recording`` (``recording_id`` +
        ``recording_content_hash``) here, and folds the ordered LOGICAL set into
        ``concat_recording_id`` via :func:`._concat_recording.member_set_hash`.
        Member identity columns are snapshots; the selected artifact detection
        has a foreign key so it cannot silently disappear. The identity stays
        frozen: a later edit to ``SessionGroup.Member`` or the member's
        ``RecordingSelection`` cannot cascade into it. Concat read /
        materialization paths read THIS, never the live ``SessionGroup.Member``
        set, so an old concat row stays valid after a group edit (the edit mints
        a new ``concat_recording_id`` on re-selection); ``recording_content_hash``
        lets materialize / rebuild detect that a frozen member's underlying
        recording content drifted (``ConcatMemberDriftError``).
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

    #: The logical-identity fields (everything but the minted PK). A concat
    #: selection combines group and recipe names with ``member_set_hash``.
    #: The hash includes ordered members and their artifact choices, so changing
    #: either creates a distinct selection under the same group name.
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

        Enforces, BEFORE inserting, that every ``SessionGroup.Member`` has a
        populated per-member ``Recording`` row under the requested
        ``preprocessing_params_name``. This selection-time precondition is the
        load-bearing layer that lets ``ConcatenatedRecording.make`` consume
        cached ``Recording`` artifacts and never call ``Recording.populate``
        inline (a DataJoint anti-pattern): a missing member surfaces here as
        ``MissingRecordingForConcatError`` listing the offending member keys,
        not as a confusing nested-populate failure later.

        Idempotent: a repeat request for the same (group, preprocessing)
        identity over the same frozen member set returns the existing
        ``concat_recording_id`` rather than minting a second one.

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
            other field -- including ``motion_correction_params_name``
            (motion correction is a separate stage, never part of a concat).
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
        from contextlib import ExitStack

        from spyglass.spikesorting.v2._db_locking import required_advisory_lock
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        detection_ids = sorted(
            {value for value in artifacts.values() if value is not None}
        )
        if detection_ids and cls.connection.in_transaction:
            raise ValueError(
                "Create an artifact-backed concat selection outside a caller-owned "
                "transaction so its artifact lifecycle locks cover the commit."
            )
        try:
            # allow_direct_insert: this helper IS the validation boundary (it
            # has already checked every member's Recording exists, frozen the
            # snapshot, and minted the deterministic id), so it bypasses the
            # master insert guard. Master + snapshot land in one transaction so
            # a concat id never exists without its frozen member set.
            with ExitStack() as locks:
                for detection_id in detection_ids:
                    locks.enter_context(
                        required_advisory_lock(
                            ArtifactDetectionOutput,
                            {"artifact_detection_id": detection_id},
                        )
                    )
                with transaction_or_noop(cls.connection):
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

        The full logical identity (group + preprocessing params) plus
        the derived ``member_set_hash`` lives in the master's own columns, so it
        is checked against the master alone. Restricting on ``member_set_hash``
        too means two selections sharing a group name but over DIFFERENT frozen
        member sets do not collide -- each resolves to its own deterministic id.
        Any master matching this (identity, member set) whose
        ``concat_recording_id`` is NOT the deterministic id is a raw-insert /
        legacy bypass of the content-addressed invariant, and is
        rejected rather than silently returned. Used by ``insert_selection`` for
        both the pre-insert lookup and the post-duplicate-key refetch.
        """
        from spyglass.spikesorting.v2._selection_identity import (
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
                "insert or legacy non-content-addressed row); drop it and "
                "re-insert via insert_selection."
            ),
        )


class ConcatRecordingFetched(NamedTuple):
    """DB-side inputs for ``ConcatenatedRecording.make_compute``.

    Gathered with no SI/NWB I/O except the self-heal rebuild of a missing
    member file.

    ``member_plan`` is the member_index-ordered list of DeepHash-stable dicts
    ``{"member_index" (int), "nwb_file_name" (str), "recording_pk" (dict whose
    ``recording_id`` is the str UUID)}`` -- each member's cached ``Recording`` PK
    resolved and existence-checked at fetch time, so a member's cache going
    missing between stages fails in fetch rather than mid-compute.
    ``member_traces`` holds each member's cached file, aligned with
    ``member_plan`` and resolved (rebuilt if missing) at fetch time too.
    """

    member_plan: list[dict]
    member_traces: tuple[StoredTraces, ...]
    preprocessing_params_name: str
    anchor_nwb_file_name: str


class ConcatRecordingComputed(NamedTuple):
    """Compute -> insert carrier for ``ConcatenatedRecording``.

    DeepHash-stable scalars plus ``member_boundaries`` -- the per-member
    ``{"member_index" (int), "end_sample" (int)}`` rows derived from the
    per-member sample counts (``make_insert`` adds the master PK).
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
    # ``(n, 2)`` int64 half-open concat frame ranges that are artifact-free and
    # never cross a member join or a member-internal timestamp gap; every
    # estimator over the concat samples statistics only inside them.
    statistics_spans: object
    # ``(n, 2)`` int64 half-open concat frame ranges of uninterrupted
    # acquisition (split at every member join and member-internal gap) and
    # ``(n,)`` float64 first / last timestamp of each on its member's own
    # clock; the synthetic concat timeline cannot recover them.
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

    Tri-part ``make`` writes a single masked, unwhitened ``ElectricalSeries``
    spanning the ordered member recordings (no motion correction), plus the
    cumulative per-member integer sample boundaries on the ``MemberBoundary``
    part (consumed by ``split_sorting_by_session``). Downstream
    ``SortingSelection`` FKs this table via its ``ConcatenatedRecordingSource``
    part.
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
    def _resolve_snapshot_recordings(snapshot_rows):
        """Verify each frozen member's ``Recording`` and build the load plan.

        The verification is
        :func:`._concat_recording_fetch.resolve_snapshot_recordings` (see it
        for the plan's fields and the errors). ``make_fetch``, member curation
        and UnitMatch call this method.
        """
        return _concat_recording_fetch.resolve_snapshot_recordings(
            snapshot_rows
        )

    @staticmethod
    def _load_member_recordings(member_plan, member_traces):
        """Load each member's cached ``Recording`` from the resolved plan (SI I/O).

        The compute-side half: given the fetch-resolved ``member_plan`` (see
        :meth:`_resolve_snapshot_recordings`) and member files, read each
        cached ``Recording`` and collect its sample count (the basis for the
        ``MemberBoundary`` back-mapping). Aligned element-wise in
        ``member_index`` order. No DB access -- the files were resolved at
        fetch time.

        Parameters
        ----------
        member_plan : list[dict]
            Resolved per-member plan dicts from
            :meth:`_resolve_snapshot_recordings`.
        member_traces : tuple[StoredTraces, ...]
            Each member's resolved file, aligned with ``member_plan``.

        Returns
        -------
        tuple[list, list[int], list[int]]
            ``(recordings, member_sample_counts, member_indices)`` -- aligned
            element-wise and in ``member_index`` order.
        """
        from spyglass.spikesorting.v2._recording_nwb import read_stored_traces

        recordings = []
        member_sample_counts = []
        member_indices = []
        for plan, traces in zip(member_plan, member_traces, strict=True):
            recording = read_stored_traces(traces)
            recordings.append(recording)
            member_sample_counts.append(int(recording.get_num_samples()))
            member_indices.append(int(plan["member_index"]))
        return recordings, member_sample_counts, member_indices

    # ``_parallel_make = True`` + the tri-part ``make_fetch`` / ``make_compute``
    # / ``make_insert`` keep the long SpikeInterface concat + NWB write
    # OUTSIDE the framework's commit transaction (mirroring
    # ``Recording`` / ``Sorting``), so no lock is held for the whole
    # materialization.
    _parallel_make = True

    def make_fetch(self, key) -> ConcatRecordingFetched:
        """Read every DB input the materialization needs.

        No SI / NWB I/O except the self-heal rebuild of a missing member file.

        Resolves the selection row, the FROZEN member snapshot (never the live
        ``SessionGroup.Member`` set) and each member's still-current ``Recording``
        (raising if any is missing or its content drifted from the snapshot),
        then each member's cached file, rebuilding a missing one through
        ``Recording``'s own verified self-heal. Returns a DeepHash-stable
        carrier so the framework's two-fetch integrity check does not trip.

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
        return _concat_recording_fetch.fetch_concat_inputs(self, key)

    def make_compute(
        self,
        key,
        member_plan,
        member_traces,
        preprocessing_params_name,
        anchor_nwb_file_name,
    ) -> ConcatRecordingComputed:
        """Materialize the concat cache outside any DB transaction.

        Reuses each member's already-populated, cached ``Recording`` artifact
        (NEVER calls ``Recording.populate`` -- the files were resolved in
        ``make_fetch``; the only DB access here stages the output file, see
        :mod:`._recording_nwb`), zeros each member's selected artifact intervals,
        stitches the members into one mono-segment recording, and writes a
        single ``ElectricalSeries`` into a fresh ``AnalysisNwbfile``. No motion
        correction and no whitening are applied here -- whitening stays a
        sorter/analyzer concern, so the persisted concat recording is the
        masked, unwhitened concatenation of the member traces. Masked samples
        are 0 uV: members that keep a nonzero channel offset (unfiltered,
        unreferenced sources) are concatenated and persisted as float32
        microvolts with a unit calibration; every other member keeps its
        stored samples and calibration (``mask_member_recordings``). The
        cumulative per-member sample boundaries are carried to
        ``make_insert``.

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
        from spyglass.spikesorting.v2._concat_recording import (
            build_concatenated_recording,
            concat_provenance_tables,
            concat_span_arrays,
            cumulative_member_boundaries,
            mask_member_recordings,
            observation_intervals,
        )
        from spyglass.spikesorting.v2._recording_nwb import write_nwb_artifact
        from spyglass.spikesorting.v2._units_nwb import (
            _base_intervals_from_recording,
        )

        recordings, member_sample_counts, member_indices = (
            self._load_member_recordings(member_plan, member_traces)
        )
        member_valid_times = [
            (
                plan["valid_times"]
                if plan["valid_times"] is not None
                else _base_intervals_from_recording(
                    recording, recording.get_sampling_frequency()
                )
            )
            for plan, recording in zip(member_plan, recordings, strict=True)
        ]
        masked_recordings, artifact_ranges = mask_member_recordings(
            recordings, [plan["valid_times"] for plan in member_plan]
        )
        # Continuity spans (and so statistics spans) come from the members as
        # loaded, whose persisted timestamps still carry each member's own
        # gaps; the concatenation below replaces them with one synthetic
        # continuous timeline.
        continuity_spans, continuity_start_s, continuity_end_s, statistics = (
            concat_span_arrays(
                recordings, member_sample_counts, artifact_ranges
            )
        )
        recordings = masked_recordings

        concatenated = build_concatenated_recording(recordings)

        # Compute the member boundaries + recording metadata BEFORE staging the
        # NWB, so a failure in this pure arithmetic cannot orphan a staged file.
        boundaries = cumulative_member_boundaries(member_sample_counts)
        concat_n_samples = int(concatenated.get_num_samples())
        # Member boundaries come from the per-member sample counts. Guard that
        # the stitched recording has exactly their sum -- a mismatch (e.g. an SI
        # concatenation change) would silently misalign the boundaries and
        # corrupt split_sorting_by_session's back-mapping.
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

        # Anchor the analysis NWB to the FIRST member's session (deterministic
        # parent, resolved in make_fetch); full multi-session provenance stays
        # queryable through ConcatenatedRecordingSelection -> SessionGroup.Member.
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
            # Exact integer concat sample count (== boundaries[-1], asserted
            # above) -- the authoritative basis split-back checks the cumulative
            # MemberBoundary set against, avoiding a float total_duration_s*fs
            # round-trip.
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

        DataJoint's tri-part dispatch already opens the master transaction
        around this method, so ``transaction_or_noop`` is a no-op here; it is
        kept so a direct (non-populate) call still commits atomically.
        Removing a failed attempt's staged ``ElectricalSeries`` is
        ``StagedOutputCleanupMixin``'s job during ``populate()``; a direct
        call leaves that to its caller.
        """
        from spyglass.spikesorting.v2.recording import _ELECTRICAL_SERIES_PATH
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        boundary_rows = [
            {
                **key,
                "member_index": boundary["member_index"],
                "end_sample": int(boundary["end_sample"]),
                "member_valid_times": boundary["member_valid_times"],
            }
            for boundary in member_boundaries
        ]
        with transaction_or_noop(self.connection):
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

        Rebuilds the concat NWB artifact on demand if the file is missing
        (mirroring ``Recording.get_recording``); the DataJoint row is never
        deleted by this path -- the stored ``content_hash`` is the source of
        truth for re-verification. Reads the persisted masked, unwhitened
        ``ElectricalSeries`` through the stored
        ``electrical_series_path`` (authoritative, not an auto-detect hint) and
        annotates ``is_filtered=True`` so a downstream sorter does not re-filter
        the already-filtered cache.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``ConcatenatedRecording`` row.

        Returns
        -------
        si.BaseRecording
            The concatenated, masked, unwhitened recording.
        """
        from spyglass.spikesorting.v2._recording_nwb import (
            read_stored_traces,
            stored_traces,
        )

        return read_stored_traces(
            stored_traces(type(self), key, (self & key).fetch1())
        )

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild a missing concat artifact -- locked, atomic, content-verified.

        The rebuild is :func:`._recording_nwb.rebuild_concat_nwb_artifact`; it
        calls ``make_fetch`` and ``make_compute`` on this instance.
        :func:`._recording_nwb.ensure_artifact_file` calls this method on every
        trace-artifact table, so it stays on the class.
        """
        return _recording_nwb.rebuild_concat_nwb_artifact(self, key)

    def split_sorting_by_session(self, sorting, key) -> dict:
        """Back-map a concat-frame sorting into per-member local sortings.

        Slices each unit's concat-frame spike train into each member's LOCAL
        sample frame using the persisted ``MemberBoundary`` cumulative sample
        counts, so a sort run over the concatenated recording can be split back
        into per-session sortings. Unit ids are preserved across members.

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
            team_name)``, with spike times in that member's local sample frame.
            The ``sort_group_id`` and ``team_name`` are in the key (not just
            ``(nwb_file_name, interval_list_name)``) so two members sharing an
            NWB/interval but on distinct sort groups -- or, for a mixed-team
            group, distinct teams -- do not collide. See
            :func:`._concat_recording.member_split_key`.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._concat_recording import (
            member_split_key,
            split_unit_spike_trains,
        )
        from spyglass.spikesorting.v2.exceptions import ConcatSplitError

        # Split over the FROZEN member set (the same authority make_fetch used to
        # materialize), so a later SessionGroup.Member edit cannot misalign the
        # back-mapping; the per-member end boundaries are the frozen
        # MemberBoundary rows.
        members = (ConcatenatedRecordingSelection.MemberSnapshot & key).fetch(
            as_dict=True, order_by="member_index"
        )
        # MemberBoundary is keyed by member_index; align to the member order and
        # require exactly one boundary per frozen member (no missing/extra).
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
        # split_unit_spike_trains enforces strictly-increasing boundaries and
        # per-spike conservation (every concat-frame spike lands in exactly one
        # member). Pass the stored exact concat sample count so the final
        # boundary is verified to cover the whole recording -- catching a
        # truncated or over-long MemberBoundary set that the per-spike range
        # check alone could miss (a too-long final boundary is never exceeded by
        # an in-range spike). ``n_samples`` is the exact integer written by
        # make_insert, not a float total_duration_s * fs round-trip.
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
