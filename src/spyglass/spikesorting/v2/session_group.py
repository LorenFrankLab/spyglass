"""Grouping of sessions for chronic and concatenated-recording workflows.

Implements same-day chronic concatenate-and-sort: ``SessionGroup`` names a
bundle of sorting members, and ``ConcatenatedRecording`` materializes one
motion-corrected, unwhitened concatenated recording cache from each member's
already-populated ``Recording`` artifact. A ``SortingSelection`` then FKs
``ConcatenatedRecording`` (via its ``ConcatenatedRecordingSource`` part) and
sorts the concatenation as one piece.

Tables:
    SessionGroup (+ Member)                  -- Manual; user-facing grouping.
    MotionCorrectionParameters               -- Lookup; Pydantic-validated.
    ConcatenatedRecordingSelection           -- Manual; UUID PK.
    ConcatenatedRecording (+ MemberBoundary) -- Computed; materialized cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj

from spyglass.common import IntervalList, LabTeam, Session  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._params.motion_correction import (
    MOTION_CORRECTION_SCHEMA_VERSION,
    MotionCorrectionParamsSchema,
)
from spyglass.spikesorting.v2.recording import (
    PreprocessingParameters,  # noqa: F401
    SortGroupV2,  # noqa: F401
)
from spyglass.spikesorting.v2.utils import (
    FactoryOnlyMaster,
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _assert_v2_db_safe,
    _validate_params,
    reject_duplicate_parameter_content,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import spikeinterface as si

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_session_group")


def _member_electrode_signature(member: dict) -> tuple:
    """Return a member sort group's ordered ``(electrode_id, region)`` signature.

    The signature identifies the physical electrode space a member contributes:
    its sort group's electrode ids (in id order) paired with each electrode's
    brain region. ``electrode_id`` is per-NWB but stable across an animal's
    sessions, and the region pins the physical target -- so two members of the
    same implant share a signature while members from different sort groups,
    probes, or regions diverge. An electrode without a region maps to ``None``.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion

    restriction = {
        "nwb_file_name": member["nwb_file_name"],
        "sort_group_id": member["sort_group_id"],
    }
    electrode_ids = sorted(
        int(eid)
        for eid in (
            SortGroupV2.SortGroupElectrode & restriction
        ).fetch("electrode_id")
    )
    region_by_electrode = {
        int(row["electrode_id"]): str(row["region_name"])
        for row in (
            (SortGroupV2.SortGroupElectrode & restriction)
            * Electrode
            * BrainRegion
        ).fetch("electrode_id", "region_name", as_dict=True)
    }
    return tuple(
        (eid, region_by_electrode.get(eid)) for eid in electrode_ids
    )


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
    ``allow_multi_day=True`` AND forces an explicit
    ``MotionCorrectionParameters`` row (see ``create_group``).

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
        or more dates require ``allow_multi_day=True``. (Multi-day groups also
        require an explicit, non-``auto`` motion-correction preset on the
        downstream ``ConcatenatedRecording``; that is enforced in
        ``ConcatenatedRecording.make``, not here.) For days/weeks-apart
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
        SessionGroupDateError
            If a member dict carries ``recording_date`` (dates are derived),
            or if members span multiple dates without ``allow_multi_day=True``.
        """
        from spyglass.spikesorting.v2.exceptions import SessionGroupDateError

        if not members:
            raise ValueError(
                "SessionGroup.create_group: members is empty; a group needs "
                "at least one sorting member (nwb_file_name, sort_group_id, "
                "interval_list_name)."
            )

        rows: list[dict] = []
        dates: list = []
        for i, member in enumerate(members):
            if "recording_date" in member:
                raise SessionGroupDateError(
                    "SessionGroup.create_group: recording_date is derived "
                    "from Session.session_start_time and must not be supplied "
                    "in member dictionaries; remove it."
                )
            derived_date = (
                (Session & {"nwb_file_name": member["nwb_file_name"]})
                .fetch1("session_start_time")
                .date()
            )
            dates.append(derived_date)
            rows.append(
                {
                    **member,
                    "team_name": member.get("team_name", session_group_owner),
                    "session_group_owner": session_group_owner,
                    "session_group_name": session_group_name,
                    "member_index": i,
                }
            )

        # Reject duplicate logical members. The Member PK is member_index only
        # (order is load-bearing), so two rows with the same
        # (nwb_file_name, sort_group_id, interval_list_name, team_name) would
        # insert without a schema error and silently concatenate the same
        # recording twice -- scientifically suspect. Catch it at create time.
        identities = [
            (
                row["nwb_file_name"],
                row["sort_group_id"],
                row["interval_list_name"],
                row["team_name"],
            )
            for row in rows
        ]
        if len(set(identities)) != len(identities):
            duplicates = sorted(
                {ident for ident in identities if identities.count(ident) > 1}
            )
            raise ValueError(
                "SessionGroup.create_group: duplicate logical member(s) "
                f"{duplicates} (same nwb_file_name / sort_group_id / "
                "interval_list_name / team_name). Each member must be distinct; "
                "concatenating the same recording twice is not supported."
            )

        unique_dates = sorted(set(dates))
        if len(unique_dates) > 1 and not allow_multi_day:
            raise SessionGroupDateError(
                "SessionGroup.create_group: members span "
                f"{len(unique_dates)} distinct recording dates "
                f"({unique_dates}); multi-day groups require "
                "allow_multi_day=True. The recommended path for cross-day "
                "analyses is sort-then-match across independent sortings, "
                "not concatenation."
            )

        with cls.connection.transaction:
            # create_group IS the validation boundary (it derived + checked the
            # member dates and shaped every Member row), so it bypasses the
            # FactoryOnlyMaster insert guard for its already-validated master.
            cls.insert1(
                {
                    "session_group_owner": session_group_owner,
                    "session_group_name": session_group_name,
                    "description": description,
                },
                allow_direct_insert=True,
            )
            cls.Member.insert(rows)

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
        dates = {start_time.date() for start_time in start_times}
        return len(dates) > 1


@schema
class MotionCorrectionParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Validated motion-correction parameter blob.

    ``insert1`` Pydantic-validates the ``params`` blob (preset +
    ``preset_kwargs``). ``ConcatenatedRecording.make()`` reads the chosen row
    and resolves the Spyglass ``"auto"`` alias (``rigid_fast`` for same-day,
    rejected for multi-day) before calling SpikeInterface.
    """

    definition = f"""
    motion_correction_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={MOTION_CORRECTION_SCHEMA_VERSION}: int
    job_kwargs=null: blob
    """

    _DEFAULT_CONTENTS: tuple = (
        (
            "none",
            MotionCorrectionParamsSchema().model_dump(),
            MOTION_CORRECTION_SCHEMA_VERSION,
            None,
        ),
        (
            "auto_default",
            MotionCorrectionParamsSchema(preset="auto").model_dump(),
            MOTION_CORRECTION_SCHEMA_VERSION,
            None,
        ),
        (
            "rigid_fast_default",
            MotionCorrectionParamsSchema(preset="rigid_fast").model_dump(),
            MOTION_CORRECTION_SCHEMA_VERSION,
            None,
        ),
    )

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Validate and insert a single motion-correction parameter row."""
        # Delegate to ``insert`` so one validated path serves both.
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Validate and insert motion-correction parameter rows.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob); see
        ``reject_duplicate_parameter_content``.
        """
        # Validate every row (incl. ``insert_default``'s positional
        # ``_DEFAULT_CONTENTS``) so a bulk insert can't bypass schema
        # validation or the outer-vs-inner params_schema_version drift
        # check.
        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda _row: MotionCorrectionParamsSchema,
            table_name="MotionCorrectionParameters",
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="MotionCorrectionParameters",
            name_attr="motion_correction_params_name",
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert v2 default motion-correction presets if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class ConcatenatedRecordingSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One row per (SessionGroup, PreprocessingParameters, motion params).

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
    -> MotionCorrectionParameters
    member_set_hash: char(64)   # sha256 of the ordered frozen member set; also folded into concat_recording_id
    """

    class MemberSnapshot(SpyglassMixinPart):
        """Frozen per-member identity captured when the concat id was minted.

        ``insert_selection`` freezes each ``SessionGroup.Member``'s ordered
        logical identity AND its resolved ``Recording`` (``recording_id`` +
        ``recording_content_hash``) here, and folds the ordered LOGICAL set into
        ``concat_recording_id`` via :func:`._concat_recording.member_set_hash`.
        Stored as PLAIN columns (no foreign keys) so the snapshot is genuinely
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
        """

    #: The logical-identity fields (everything but the minted PK). A concat
    #: selection is one (SessionGroup, PreprocessingParameters,
    #: MotionCorrectionParameters) tuple; find-existing keys on all four plus the
    #: derived ``member_set_hash`` (so the same group name over a different frozen
    #: member set is a distinct selection, not a collision).
    _IDENTITY_FIELDS = (
        "session_group_owner",
        "session_group_name",
        "preprocessing_params_name",
        "motion_correction_params_name",
    )

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Find-existing-or-insert a concat selection; return a PK-only dict.

        Enforces, BEFORE inserting, that every ``SessionGroup.Member`` has a
        populated per-member ``Recording`` row under the requested
        ``preprocessing_params_name``. This selection-time precondition is the
        load-bearing layer that lets ``ConcatenatedRecording.make`` consume
        cached ``Recording`` artifacts and never call ``Recording.populate``
        inline (a DataJoint anti-pattern): a missing member surfaces here as
        ``MissingRecordingForConcatError`` listing the offending member keys,
        not as a confusing nested-populate failure later.

        Idempotent: a repeat request for the same (group, preprocessing,
        motion) identity returns the existing ``concat_recording_id`` rather
        than minting a second one.

        Parameters
        ----------
        key : dict
            Must carry ``session_group_owner``, ``session_group_name``,
            ``preprocessing_params_name``, ``motion_correction_params_name``.
            A caller-supplied ``concat_recording_id`` is ignored; the id is
            minted/found here.

        Returns
        -------
        dict
            A PK-only dict ``{"concat_recording_id": ...}`` for the
            existing-or-inserted selection row.

        Raises
        ------
        ValueError
            If a required identity field is missing.
        MissingRecordingForConcatError
            If any member has no populated ``Recording`` under
            ``preprocessing_params_name``.
        DuplicateSelectionError
            If more than one selection row already matches the identity (a raw
            insert bypassed this helper).
        """
        from spyglass.spikesorting.v2._concat_recording import (
            member_recording_selection_key,
            member_set_hash,
        )
        from spyglass.spikesorting.v2._selection_identity import (
            deterministic_id,
        )
        from spyglass.spikesorting.v2.exceptions import (
            MissingRecordingForConcatError,
        )
        from spyglass.spikesorting.v2.recording import (
            Recording,
            RecordingSelection,
        )
        from spyglass.spikesorting.v2.utils import _is_duplicate_key_error

        missing_fields = [f for f in cls._IDENTITY_FIELDS if f not in key]
        if missing_fields:
            raise ValueError(
                "ConcatenatedRecordingSelection.insert_selection requires "
                f"field(s) {missing_fields}. Required identity fields are "
                f"{list(cls._IDENTITY_FIELDS)}."
            )
        identity = {f: key[f] for f in cls._IDENTITY_FIELDS}
        group_key = {
            "session_group_owner": identity["session_group_owner"],
            "session_group_name": identity["session_group_name"],
        }
        preprocessing_params_name = identity["preprocessing_params_name"]

        # Precondition: the group must be non-empty, and every member needs a
        # populated Recording for the shared preprocessing recipe. A Recording
        # exists iff its RecordingSelection row exists AND the Computed Recording
        # populated. The empty-group check is explicit: without it the loop
        # passes vacuously and make_fetch later fails indexing members[0].
        # Members are read in member_index order so the frozen snapshot (and its
        # folded hash) records the concatenation order.
        members = (SessionGroup.Member & group_key).fetch(
            as_dict=True, order_by="member_index"
        )
        if not members:
            raise ValueError(
                "ConcatenatedRecordingSelection.insert_selection: SessionGroup "
                f"{group_key} has no members. Create it via "
                "SessionGroup.create_group() with at least one member first."
            )
        # Reject members in different physical electrode spaces (different sort
        # group electrodes / brain regions). The concat result is read in the
        # anchor member's electrode frame, so this must hold regardless of
        # whether the per-member Recording caches are populated yet.
        assert_members_share_electrode_space(members)
        # Freeze each member's logical identity + its resolved Recording
        # (recording_id + content_hash) into an ordered snapshot. The snapshot is
        # the authority every downstream concat path reads (never the live
        # SessionGroup.Member set), and its LOGICAL set hash is folded into the
        # concat id so a different ordered member set is a different concat.
        snapshot_rows: list[dict] = []
        missing: list[dict] = []
        for member in members:
            rec_sel_key = member_recording_selection_key(
                member, preprocessing_params_name
            )
            rec_sel = RecordingSelection & rec_sel_key
            rec_pk = rec_sel.fetch1("KEY") if rec_sel else None
            content_hashes = (
                (Recording & rec_pk).fetch("content_hash")
                if rec_pk is not None
                else []
            )
            if rec_pk is None or len(content_hashes) == 0:
                missing.append(rec_sel_key)
                continue
            snapshot_rows.append(
                {
                    "member_index": int(member["member_index"]),
                    "nwb_file_name": member["nwb_file_name"],
                    "sort_group_id": int(member["sort_group_id"]),
                    "interval_list_name": member["interval_list_name"],
                    "team_name": member["team_name"],
                    "recording_id": str(rec_pk["recording_id"]),
                    "recording_content_hash": str(content_hashes[0]),
                }
            )
        if missing:
            raise MissingRecordingForConcatError(
                "ConcatenatedRecordingSelection.insert_selection requires "
                "every member's Recording to be populated under "
                f"preprocessing_params_name={preprocessing_params_name!r} "
                f"first. Missing {len(missing)} member(s): {missing}. Run "
                "Recording.populate(...) for each missing key, then retry."
            )

        # Content-address the concat_recording_id from the logical identity AND
        # the ordered member-set hash (mirrors RecordingSelection /
        # SortingSelection): two callers that request the same (group,
        # preprocessing, motion) over the same ordered member set compute the
        # same id, so the PK-uniqueness constraint -- not a check-then-insert
        # dedup race -- is the concurrency guard. A different member set folds to
        # a different id rather than silently reusing this concat.
        set_hash = member_set_hash(snapshot_rows)
        concat_recording_id = deterministic_id(
            "concat_recording", {**identity, "member_set_hash": set_hash}
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
        try:
            # allow_direct_insert: this helper IS the validation boundary (it
            # has already checked every member's Recording exists, frozen the
            # snapshot, and minted the deterministic id), so it bypasses the
            # master insert guard. Master + snapshot land in one transaction so
            # a concat id never exists without its frozen member set.
            with cls.connection.transaction:
                cls.insert1(
                    {
                        **identity,
                        "member_set_hash": set_hash,
                        "concat_recording_id": concat_recording_id,
                    },
                    allow_direct_insert=True,
                )
                cls.MemberSnapshot.insert(snapshot_inserts)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless dup-PK
            if not _is_duplicate_key_error(exc):
                raise
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

        The full logical identity (group + preprocessing + motion params) plus
        the derived ``member_set_hash`` lives in the master's own columns, so it
        is checked against the master alone. Restricting on ``member_set_hash``
        too means two selections sharing a group name but over DIFFERENT frozen
        member sets do not collide -- each resolves to its own deterministic id.
        Any master matching this (identity, member set) whose
        ``concat_recording_id`` is NOT the deterministic id is a raw-insert /
        pre-determinism legacy bypass of the content-addressed invariant, and is
        rejected rather than silently returned. Used by ``insert_selection`` for
        both the pre-insert lookup and the post-duplicate-key refetch.
        """
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        lookup = {**identity, "member_set_hash": member_set_hash}
        master_ids = {
            row["concat_recording_id"]
            for row in (cls & lookup).fetch("KEY", as_dict=True)
        }
        bypassed = [
            cid
            for cid in master_ids
            if cid != deterministic_concat_recording_id
        ]
        if bypassed:
            raise DuplicateSelectionError(
                f"ConcatenatedRecordingSelection has {len(master_ids)} master "
                f"row(s) for identity {identity} (member_set_hash "
                f"{member_set_hash}) whose concat_recording_id is not the "
                f"deterministic id {deterministic_concat_recording_id}: "
                f"{bypassed}. This is a non-deterministic selection row (a raw "
                "insert or pre-determinism legacy row); drop it and re-insert "
                "via insert_selection."
            )
        return (
            {"concat_recording_id": deterministic_concat_recording_id}
            if master_ids
            else None
        )


class ConcatRecordingFetched(NamedTuple):
    """DB-side inputs for ``ConcatenatedRecording.make_compute`` (no SI/NWB I/O).

    ``member_plan`` is the member_index-ordered list of DeepHash-stable dicts
    ``{"member_index" (int), "nwb_file_name" (str), "recording_pk" (dict whose
    ``recording_id`` is the str UUID)}`` -- each member's cached ``Recording`` PK
    resolved and existence-checked at fetch time, so a member's cache going
    missing between stages fails in fetch rather than mid-compute.
    """

    member_plan: list[dict]
    preprocessing_params_name: str
    motion_params: dict
    motion_job_kwargs: dict | None
    preprocessing_job_kwargs: dict | None
    is_multi_day: bool
    anchor_nwb_file_name: str


class ConcatRecordingComputed(NamedTuple):
    """Compute -> insert carrier for ``ConcatenatedRecording``.

    DeepHash-stable scalars plus ``member_boundaries`` -- the per-member
    ``{"member_index" (int), "end_sample" (int)}`` rows derived from the
    PRE-motion sample counts (``make_insert`` adds the master PK).
    """

    analysis_file_name: str
    object_id: str
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    content_hash: str
    anchor_nwb_file_name: str
    member_boundaries: list[dict]
    # The RESOLVED motion-correction preset string ("rigid_fast" for an "auto"
    # same-day request, the explicit preset otherwise, "none" when skipped) --
    # resolved in make_compute and persisted so the row records what actually
    # ran, not the unresolved alias.
    motion_preset: str


@schema
class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    """Materialized cross-session concatenated recording cache.

    Tri-part ``make`` writes a single motion-corrected, unwhitened
    ``ElectricalSeries`` spanning the ordered member recordings, plus the
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
    content_hash: char(64)
    motion_preset: varchar(64)   # RESOLVED motion preset ('rigid_fast' for 'auto' same-day, the explicit preset, or 'none')
    """

    class MemberBoundary(SpyglassMixinPart):
        """Per-member end-sample boundary in the concatenated recording."""

        definition = """
        -> master
        member_index: int
        ---
        end_sample: bigint
        """

    @staticmethod
    def _resolve_snapshot_recordings(snapshot_rows):
        """Verify each FROZEN member's ``Recording`` and build the load plan.

        The fetch-side half of the materialization contract, driven off the
        frozen ``ConcatenatedRecordingSelection.MemberSnapshot`` (never the live
        ``SessionGroup.Member`` set). For each member, in ``member_index`` order,
        confirm its frozen ``recording_id`` still resolves to a populated
        ``Recording`` AND that the recording's current ``content_hash`` still
        matches the one frozen at selection. No SI/NWB I/O -- the heavy load
        lives in :meth:`_load_member_recordings`, which ``make_compute`` drives
        off the returned plan.

        Never calls ``Recording.populate``: the selection-time precondition in
        ``insert_selection`` guarantees every member was cached when the concat
        id was minted. This re-checks (a row could be deleted later) and, by
        comparing ``content_hash``, refuses to materialize / rebuild from member
        data that drifted out from under the frozen id.

        Parameters
        ----------
        snapshot_rows : list[dict]
            ``ConcatenatedRecordingSelection.MemberSnapshot`` rows, ordered by
            ``member_index`` (each carrying ``recording_id`` +
            ``recording_content_hash`` and the member's logical identity).

        Returns
        -------
        list[dict]
            member_index-ordered plan dicts ``{"member_index" (int),
            "nwb_file_name" (str), "interval_list_name" (str), "recording_pk"
            (dict whose ``recording_id`` is the str UUID -- DeepHash-stable for
            the tri-part carrier)}``.

        Raises
        ------
        MissingRecordingForConcatError
            If any frozen member's ``Recording`` row is gone.
        ConcatMemberDriftError
            If a frozen member's current ``Recording.content_hash`` diverges
            from the one captured in the snapshot.
        """
        from spyglass.spikesorting.v2.exceptions import (
            ConcatMemberDriftError,
            MissingRecordingForConcatError,
        )
        from spyglass.spikesorting.v2.recording import Recording

        member_plan = []
        missing: list[dict] = []
        drifted: list[dict] = []
        for row in snapshot_rows:
            recording_id = row["recording_id"]
            content_hashes = (
                Recording & {"recording_id": recording_id}
            ).fetch("content_hash")
            if len(content_hashes) == 0:
                missing.append({"recording_id": str(recording_id)})
                continue
            current_hash = str(content_hashes[0])
            if current_hash != str(row["recording_content_hash"]):
                drifted.append(
                    {
                        "member_index": int(row["member_index"]),
                        "recording_id": str(recording_id),
                        "snapshot_content_hash": str(
                            row["recording_content_hash"]
                        ),
                        "current_content_hash": current_hash,
                    }
                )
                continue
            member_plan.append(
                {
                    "member_index": int(row["member_index"]),
                    "nwb_file_name": row["nwb_file_name"],
                    "interval_list_name": row["interval_list_name"],
                    "recording_pk": {"recording_id": str(recording_id)},
                }
            )
        if missing:
            raise MissingRecordingForConcatError(
                "ConcatenatedRecording.make: "
                f"{len(missing)} frozen member Recording row(s) are gone: "
                f"{missing}. The member set was frozen at insert_selection; "
                "restore the missing Recording(s), or DELETE this concat and "
                "re-run ConcatenatedRecordingSelection.insert_selection (which "
                "re-snapshots and mints a new concat_recording_id). Run "
                "Recording.populate(...) for each missing key first."
            )
        if drifted:
            raise ConcatMemberDriftError(
                "ConcatenatedRecording.make: "
                f"{len(drifted)} frozen member(s) have a Recording whose current "
                f"content_hash no longer matches the snapshot: {drifted}. The "
                "concatenation would be built from different underlying data than "
                "concat_recording_id was minted for. Restore the original member "
                "recording content, or DELETE this concat and re-run "
                "insert_selection to mint a new concat for the changed inputs."
            )
        return member_plan

    @staticmethod
    def _snapshot_is_multi_day(snapshot_rows) -> bool:
        """Report whether the frozen member set spans two or more dates.

        Derived from the snapshot's distinct ``nwb_file_name``s (the frozen
        member set), not the live ``SessionGroup.Member`` rows, so a concat's
        multi-day status is fixed once its id is minted. ``Session`` start times
        are themselves immutable, so reading them live is safe.
        """
        nwb_keys = [
            {"nwb_file_name": name}
            for name in {row["nwb_file_name"] for row in snapshot_rows}
        ]
        if not nwb_keys:
            return False
        start_times = (Session & nwb_keys).fetch("session_start_time")
        return len({start_time.date() for start_time in start_times}) > 1

    @staticmethod
    def _load_member_recordings(member_plan):
        """Load each member's cached ``Recording`` from the resolved plan (SI I/O).

        The compute-side half: given the fetch-resolved ``member_plan`` (see
        :meth:`_resolve_snapshot_recordings`), load each cached ``Recording``
        and collect the pre-motion sample count (the basis for the
        ``MemberBoundary`` back-mapping). Aligned element-wise in
        ``member_index`` order. No DB resolution happens here -- the PKs were
        pinned at fetch time.

        Parameters
        ----------
        member_plan : list[dict]
            Resolved per-member plan dicts from
            :meth:`_resolve_snapshot_recordings`.

        Returns
        -------
        tuple[list, list[int], list[int]]
            ``(recordings, member_sample_counts, member_indices)`` -- aligned
            element-wise and in ``member_index`` order.
        """
        from spyglass.spikesorting.v2.recording import Recording

        recordings = []
        member_sample_counts = []
        member_indices = []
        for plan in member_plan:
            recording = Recording().get_recording(plan["recording_pk"])
            recordings.append(recording)
            member_sample_counts.append(int(recording.get_num_samples()))
            member_indices.append(int(plan["member_index"]))
        return recordings, member_sample_counts, member_indices

    # ``_parallel_make = True`` + the tri-part ``make_fetch`` / ``make_compute``
    # / ``make_insert`` keep the long SpikeInterface concat + motion correction
    # + NWB write OUTSIDE the framework's commit transaction (mirroring
    # ``Recording`` / ``Sorting``); a single ``make()`` held the lock for the
    # whole materialization.
    _parallel_make = True

    def make_fetch(self, key) -> ConcatRecordingFetched:
        """Read every DB input the materialization needs (no SI / NWB I/O).

        Resolves the selection row, the FROZEN member snapshot (never the live
        ``SessionGroup.Member`` set) and each member's still-current ``Recording``
        (raising if any is missing or its content drifted from the snapshot), the
        motion-correction and preprocessing parameter blobs, and the group's
        multi-day status. Returns a DeepHash-stable carrier so the framework's
        two-fetch integrity check does not trip.

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
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        # The populate key carries only concat_recording_id; every member and
        # parameter query restricts with the fetched selection row, not the
        # UUID-only key, so independent concat selections never cross-restrict.
        sel = (ConcatenatedRecordingSelection & key).fetch1()
        preprocessing_params_name = sel["preprocessing_params_name"]
        # The frozen snapshot -- not the live SessionGroup.Member set -- is the
        # authority: a later member edit mints a NEW concat id on re-selection
        # and never silently changes what THIS concat materializes.
        snapshot = (
            ConcatenatedRecordingSelection.MemberSnapshot & key
        ).fetch(as_dict=True, order_by="member_index")
        if not snapshot:
            raise SchemaBypassError(
                "ConcatenatedRecording.make_fetch: selection "
                f"{dict(key)} has no MemberSnapshot rows. The frozen member set "
                "is written by ConcatenatedRecordingSelection.insert_selection; "
                "this selection was inserted by a raw bypass. Drop it and "
                "re-insert via insert_selection."
            )
        # Re-assert electrode-space compatibility at the compute boundary against
        # the frozen snapshot, defending a raw ``allow_direct_insert`` selection
        # whose members span different physical electrode spaces (same SI channel
        # ids/geometry, different DB electrodes/regions).
        assert_members_share_electrode_space(snapshot)
        member_plan = self._resolve_snapshot_recordings(snapshot)
        motion_row = (
            MotionCorrectionParameters
            & {
                "motion_correction_params_name": (
                    sel["motion_correction_params_name"]
                )
            }
        ).fetch1()
        preprocessing_job_kwargs = (
            PreprocessingParameters
            & {"preprocessing_params_name": preprocessing_params_name}
        ).fetch1("job_kwargs")
        return ConcatRecordingFetched(
            member_plan=member_plan,
            preprocessing_params_name=preprocessing_params_name,
            motion_params=motion_row["params"],
            motion_job_kwargs=motion_row["job_kwargs"],
            preprocessing_job_kwargs=preprocessing_job_kwargs,
            # Multi-day status of the FROZEN member set (DB-derived); the 'auto'
            # motion alias is resolved against it in compute.
            is_multi_day=self._snapshot_is_multi_day(snapshot),
            anchor_nwb_file_name=snapshot[0]["nwb_file_name"],
        )

    def make_compute(
        self,
        key,
        member_plan,
        preprocessing_params_name,
        motion_params,
        motion_job_kwargs,
        preprocessing_job_kwargs,
        is_multi_day,
        anchor_nwb_file_name,
    ) -> ConcatRecordingComputed:
        """Materialize the concat cache outside any DB transaction.

        Reuses each member's already-populated, cached ``Recording`` artifact
        (NEVER calls ``Recording.populate`` -- the PKs were pinned in
        ``make_fetch``), stitches them into one mono-segment recording, applies
        motion correction (resolving the Spyglass ``"auto"`` alias to a same-day
        preset and rejecting it on multi-day groups), and writes a single
        ``ElectricalSeries`` into a fresh ``AnalysisNwbfile``. Whitening is
        deliberately NOT applied here -- it stays a sorter/analyzer concern, so
        the persisted concat recording is motion-corrected but unwhitened. The
        cumulative per-member sample boundaries are carried to ``make_insert``.

        Returns
        -------
        ConcatRecordingComputed

        Raises
        ------
        ValueError
            If ``preset="auto"`` on a multi-day group.
        RuntimeError
            If motion correction did not preserve the total sample count (which
            would misalign the ``MemberBoundary`` back-mapping).
        """
        from spyglass.spikesorting.v2._concat_recording import (
            build_concatenated_recording,
            cumulative_member_boundaries,
            resolve_motion_correction,
        )
        from spyglass.spikesorting.v2._recording_nwb import write_nwb_artifact
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        recordings, member_sample_counts, member_indices = (
            self._load_member_recordings(member_plan)
        )

        # Resolve the Spyglass 'auto' alias against the group's (fetch-time)
        # multi-day status before any SpikeInterface call.
        motion_preset, preset_kwargs = resolve_motion_correction(
            motion_params, is_multi_day=is_multi_day
        )
        # Job kwargs: preprocessing first, motion last (motion wins on
        # conflict), per the job-kwargs resolution contract.
        job_kwargs = _resolved_job_kwargs(
            preprocessing_job_kwargs, motion_job_kwargs
        )

        corrected = build_concatenated_recording(
            recordings,
            motion_preset=motion_preset,
            preset_kwargs=preset_kwargs,
            job_kwargs=job_kwargs,
        )

        # Compute the member boundaries + recording metadata BEFORE staging the
        # NWB, so a failure in this pure arithmetic cannot orphan a staged file.
        boundaries = cumulative_member_boundaries(member_sample_counts)
        corrected_n_samples = int(corrected.get_num_samples())
        # Member boundaries come from the PRE-motion per-member sample counts;
        # motion correction is interpolation that preserves sample count. Guard
        # that invariant explicitly -- a mismatch (an SI change or an unexpected
        # preset that resampled) would silently misalign the boundaries and
        # corrupt split_sorting_by_session's back-mapping.
        if boundaries and corrected_n_samples != boundaries[-1]:
            raise RuntimeError(
                "ConcatenatedRecording.make: the motion-corrected recording "
                f"has {corrected_n_samples} samples but the cumulative member "
                f"sample count is {boundaries[-1]}; motion correction must "
                "preserve the sample count or the MemberBoundary back-mapping "
                "would be wrong."
            )
        sampling_frequency = float(corrected.get_sampling_frequency())
        n_channels = int(corrected.get_num_channels())
        total_duration_s = corrected_n_samples / sampling_frequency

        # Anchor the analysis NWB to the FIRST member's session (deterministic
        # parent, resolved in make_fetch); full multi-session provenance stays
        # queryable through ConcatenatedRecordingSelection -> SessionGroup.Member.
        preset_label = motion_preset or "none"
        # Self-describing provenance: a header (the resolved motion preset +
        # KWARGS -- the producing params, NOT the displacement field, which stays
        # derivable) and the ordered member map with per-member frame boundaries,
        # so split_sorting_by_session is reconstructable from the file alone.
        from spyglass.spikesorting.v2._nwb_provenance import (
            CONCAT_MEMBERS,
            CONCAT_PROVENANCE,
            build_long_provenance_table,
            build_provenance_table,
        )

        member_rows = []
        concat_start = 0
        for plan, n_samples, cum_end in zip(
            member_plan, member_sample_counts, boundaries
        ):
            member_rows.append(
                {
                    "member_index": int(plan["member_index"]),
                    "recording_id": str(plan["recording_pk"]["recording_id"]),
                    "nwb_file_name": plan["nwb_file_name"],
                    "interval_list_name": plan["interval_list_name"],
                    "start_sample": 0,
                    "end_sample": int(n_samples),
                    "concat_start_sample": int(concat_start),
                    "concat_end_sample": int(cum_end),
                }
            )
            concat_start = int(cum_end)
        provenance_tables = [
            build_provenance_table(
                CONCAT_PROVENANCE,
                {
                    "concat_recording_id": str(key["concat_recording_id"]),
                    "preprocessing_params_name": preprocessing_params_name,
                    "motion_preset": preset_label,
                    "motion_kwargs": preset_kwargs,
                    "anchor_nwb_file_name": anchor_nwb_file_name,
                    "n_members": len(member_plan),
                },
            ),
            build_long_provenance_table(
                CONCAT_MEMBERS,
                member_rows,
                [
                    ("member_index", int),
                    ("recording_id", str),
                    ("nwb_file_name", str),
                    ("interval_list_name", str),
                    ("start_sample", int),
                    ("end_sample", int),
                    ("concat_start_sample", int),
                    ("concat_end_sample", int),
                ],
            ),
        ]
        analysis_file_name, object_id, content_hash = write_nwb_artifact(
            corrected,
            anchor_nwb_file_name,
            filtering_description=(
                f"Concatenated {len(recordings)} member recording(s) "
                f"(preprocessing_params={preprocessing_params_name!r}); "
                f"motion correction preset={preset_label!r}; unwhitened"
            ),
            provenance_tables=provenance_tables,
        )
        member_boundaries = [
            {"member_index": member_index, "end_sample": int(end_sample)}
            for member_index, end_sample in zip(member_indices, boundaries)
        ]
        return ConcatRecordingComputed(
            analysis_file_name=analysis_file_name,
            object_id=object_id,
            n_channels=n_channels,
            sampling_frequency=sampling_frequency,
            total_duration_s=total_duration_s,
            content_hash=content_hash,
            anchor_nwb_file_name=anchor_nwb_file_name,
            member_boundaries=member_boundaries,
            # Persist the RESOLVED preset ("rigid_fast" for "auto" same-day),
            # not the alias; ``preset_label`` already maps a None (skip) to "none".
            motion_preset=preset_label,
        )

    def make_insert(
        self,
        key,
        analysis_file_name,
        object_id,
        n_channels,
        sampling_frequency,
        total_duration_s,
        content_hash,
        anchor_nwb_file_name,
        member_boundaries,
        motion_preset,
    ):
        """Atomically register the staged concat artifact + boundary rows.

        DataJoint's tri-part dispatch already opens the master transaction
        around this method, so ``transaction_or_noop`` is a no-op here; it is
        kept so a direct (non-populate) call still commits atomically. On any
        registration failure the staged ``ElectricalSeries`` is unlinked before
        re-raising so it does not orphan.
        """
        from spyglass.spikesorting.v2.recording import (
            _ELECTRICAL_SERIES_PATH,
            _unlink_staged_analysis_file,
        )
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        boundary_rows = [
            {
                **key,
                "member_index": boundary["member_index"],
                "end_sample": int(boundary["end_sample"]),
            }
            for boundary in member_boundaries
        ]
        try:
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(
                    anchor_nwb_file_name, analysis_file_name
                )
                self.insert1(
                    {
                        **key,
                        "analysis_file_name": analysis_file_name,
                        "electrical_series_path": _ELECTRICAL_SERIES_PATH,
                        "object_id": object_id,
                        "n_channels": n_channels,
                        "sampling_frequency": sampling_frequency,
                        "total_duration_s": total_duration_s,
                        "content_hash": content_hash,
                        "motion_preset": motion_preset,
                    }
                )
                self.MemberBoundary.insert(boundary_rows)
        except Exception:
            # ``write_nwb_artifact`` already wrote the concat ElectricalSeries
            # to disk; if registration (AnalysisNwbfile.add + the inserts) then
            # fails, that file would orphan, so unlink it before re-raising --
            # matching Recording.make_insert's staged-file cleanup.
            _unlink_staged_analysis_file(
                analysis_file_name,
                context="ConcatenatedRecording.make_insert",
            )
            raise

    def get_recording(self, key) -> "si.BaseRecording":  # noqa: F821
        """Return the cached concatenated SpikeInterface recording.

        Rebuilds the concat NWB artifact on demand if the file is missing
        (mirroring ``Recording.get_recording``); the DataJoint row is never
        deleted by this path -- the stored ``content_hash`` is the source of
        truth for re-verification. Reads the persisted motion-corrected,
        unwhitened ``ElectricalSeries`` through the stored
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
            The concatenated, motion-corrected, unwhitened recording.
        """
        from pathlib import Path

        import spikeinterface.extractors as se

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        if not Path(abs_path).exists():
            self._rebuild_nwb_artifact(key)
            abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        rec = se.read_nwb_recording(
            abs_path,
            electrical_series_path=row["electrical_series_path"],
            load_time_vector=True,
        )
        rec.annotate(is_filtered=True)
        return rec

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild a missing concat artifact -- locked, atomic, content-verified.

        The concat analog of ``Recording._rebuild_nwb_artifact``. Acquire
        ``concat_recording_artifact_lock(concat_recording_id)``, double-check the
        file is still missing under the lock (a peer may have rebuilt while we
        waited), then re-run the materialization (``make_fetch`` -> ``make_compute``
        -- which also re-verifies the frozen member set against the live
        recordings) to a FRESH temp analysis file, fingerprint it, and only on a
        ``content_hash`` match ``os.replace`` it into the canonical slot and
        refresh the DataJoint ``~external`` byte checksum. A rebuild whose
        fingerprint diverges from the stored ``content_hash`` raises
        ``RecordingContentDriftError`` and never touches the canonical slot --
        drifted bytes are never served.

        Cleanup contract (all-or-nothing): the temp is unlinked on any failure;
        if ``os.replace`` ran but the checksum refresh then failed, the canonical
        is unlinked to return the slot to the missing state for the next (locked)
        ``get_recording``. Ordering is load-bearing -- the atomic ``os.replace``
        precedes ``_resolve_external``.

        Note: a motion-corrected concat is only byte-reproducible insofar as
        ``correct_motion`` is deterministic; an irreproducible rebuild surfaces
        loudly as ``RecordingContentDriftError`` rather than silently serving
        different bytes (the fingerprint's trace rounding absorbs sub-µV noise).
        Safe to call directly (it takes the lock itself) and from
        ``get_recording`` (which does not hold the lock).
        """
        import os
        from pathlib import Path

        from spyglass.spikesorting.v2._concat_recording import (
            concat_recording_artifact_lock,
        )
        from spyglass.spikesorting.v2.exceptions import (
            RecordingContentDriftError,
        )
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        row = (self & key).fetch1()
        concat_recording_id = row["concat_recording_id"]
        analysis_file_name = row["analysis_file_name"]
        canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)

        with concat_recording_artifact_lock(concat_recording_id):
            # Double-checked: a peer rebuilt (or the file was never gone) while
            # we waited for the lock -- nothing to do.
            if Path(canonical_abs).exists():
                return

            logger.info(
                "ConcatenatedRecording.get_recording: cache miss for "
                f"{analysis_file_name!r} (reason=missing cache); rebuilding "
                "the concatenated artifact..."
            )
            fetched = self.make_fetch(key)
            # make_compute writes a FRESH (unregistered) temp analysis file and
            # returns its readback content fingerprint as ``content_hash``.
            computed = self.make_compute(key, *fetched)
            temp_abs = AnalysisNwbfile.get_abs_path(computed.analysis_file_name)

            if computed.content_hash != row["content_hash"]:
                _unlink_staged_analysis_file(
                    computed.analysis_file_name,
                    context="ConcatenatedRecording._rebuild_nwb_artifact",
                )
                raise RecordingContentDriftError(
                    "ConcatenatedRecording._rebuild_nwb_artifact: rebuilt "
                    f"content_hash {computed.content_hash} does not match the "
                    f"stored content_hash {row['content_hash']} for "
                    f"{analysis_file_name!r}. The current environment no longer "
                    "reproduces this concatenated recording (e.g. a "
                    "SpikeInterface/BLAS upgrade, a changed member recording, or "
                    "non-deterministic motion correction). The canonical artifact "
                    "was NOT modified. Recover by restoring a backup, rerunning "
                    "under the original environment, or deleting and repopulating "
                    "the ConcatenatedRecording row (and its downstream)."
                )

            # Verified content match: atomically install the temp into the
            # canonical slot, then reconcile the byte checksum so the next
            # checksum-validated get_abs_path read succeeds.
            try:
                os.replace(temp_abs, canonical_abs)
            except Exception:
                Path(temp_abs).unlink(missing_ok=True)
                raise
            try:
                AnalysisNwbfile()._resolve_external(analysis_file_name)
            except Exception:
                # os.replace already ran; return the slot to MISSING so the next
                # (locked) get_recording retries cleanly rather than leaving
                # byte-different content under a stale checksum.
                Path(canonical_abs).unlink(missing_ok=True)
                raise
            # Confirm the reconciled slot resolves through the checksum-
            # validating path. get_recording re-reads after this, but a DIRECT
            # caller has no follow-on read, so a botched reconcile would
            # otherwise stay silent until the next reader. Unlink + raise if so.
            try:
                AnalysisNwbfile.get_abs_path(analysis_file_name)
            except Exception:
                Path(canonical_abs).unlink(missing_ok=True)
                raise

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
        members = (
            ConcatenatedRecordingSelection.MemberSnapshot & key
        ).fetch(as_dict=True, order_by="member_index")
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
        # member). The final boundary equals the concat sample count by the
        # make_compute write-time invariant (corrected_n_samples == boundaries[-1]),
        # so it is not re-derived from a heavy recording load here.
        per_member = split_unit_spike_trains(unit_trains, boundaries)
        fs = float(sorting.get_sampling_frequency())
        return {
            member_split_key(member): si.NumpySorting.from_unit_dict(
                [local_trains], sampling_frequency=fs
            )
            for member, local_trains in zip(members, per_member)
        }
