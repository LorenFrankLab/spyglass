"""Grouping of sessions for chronic and concatenated-recording workflows.

These tables are declared in their final-shape so ``SortingSelection``
can FK ``ConcatenatedRecording`` from day one (zero-migration policy).
The schema is frozen; the populate body of ``ConcatenatedRecording`` is
gated behind ``NotImplementedError`` until the concat materializer
lands -- only the consumer logic changes, never the row shape.

Tables (all final-shape):
    SessionGroup (+ Member)                  -- Manual; user-facing grouping.
    MotionCorrectionParameters               -- Lookup; Pydantic-validated.
    ConcatenatedRecordingSelection           -- Manual; UUID PK.
    ConcatenatedRecording (+ MemberBoundary) -- Computed; make() is gated.
"""

from __future__ import annotations

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
    _assert_v2_db_safe,
    _validate_params,
    reject_duplicate_parameter_content,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_session_group")


@schema
class SessionGroup(SpyglassMixin, dj.Manual):
    """A named bundle of sorting members to analyze together.

    A member is a ``(nwb_file_name, sort_group_id, interval_list_name,
    team_name)`` tuple, not necessarily a whole NWB file. The master PK is
    ``(session_group_owner, session_group_name)``; ``session_group_owner``
    is a projected ``LabTeam.team_name`` so two teams may both create a
    group named ``"day1"`` without collision.

    Same-day groups are the default; multi-day requires
    ``allow_multi_day=True`` AND forces an explicit
    ``MotionCorrectionParameters`` row (see ``create_group``).
    """

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
            cls.insert1(
                {
                    "session_group_owner": session_group_owner,
                    "session_group_name": session_group_name,
                    "description": description,
                }
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
        dates = {
            (Session & {"nwb_file_name": nwb_file_name})
            .fetch1("session_start_time")
            .date()
            for nwb_file_name in (cls.Member & key).fetch("nwb_file_name")
        }
        return len(dates) > 1


@schema
class MotionCorrectionParameters(SpyglassMixin, dj.Lookup):
    """Validated motion-correction parameter blob.

    Ships before the consumer (``ConcatenatedRecording.make()``) lands
    so this Lookup's ``insert1`` can Pydantic-validate its ``params``
    blob at insert time. The validation surface is final-shape today;
    the consumer activates without a migration once the concat
    materializer is implemented.
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
class ConcatenatedRecordingSelection(SpyglassMixin, dj.Manual):
    """One row per (SessionGroup, PreprocessingParameters, motion params).

    UUID-keyed so downstream FKs are single-column (mirrors the
    single-session ``RecordingSelection`` / ``Recording`` shape). Schema
    is final-shape; the ``insert_selection`` helper, when implemented,
    enforces that every member has a populated ``Recording`` row
    matching the requested preprocessing parameters.
    """

    definition = """
    concat_recording_id: uuid
    ---
    -> SessionGroup
    -> PreprocessingParameters
    -> MotionCorrectionParameters
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Find-existing-or-insert; return a single PK-only dict.

        Gated until the concat materializer lands (raises
        ``NotImplementedError`` today). Forward-declared; the schema is in
        place so single-session tables can FK ``concat_recording_id`` from
        day one.

        Returns
        -------
        dict
            A PK-only dict ``{"concat_recording_id": ...}`` for the
            existing-or-inserted selection row.

        Raises
        ------
        NotImplementedError
            Always, until this helper is implemented.
        """
        raise NotImplementedError(
            "ConcatenatedRecordingSelection.insert_selection is not yet "
            "implemented"
        )


@schema
class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    """Materialized cross-session concatenated recording cache.

    When implemented, ``make()`` writes a single motion-corrected,
    unwhitened ``ElectricalSeries`` covering the union of member intervals,
    plus integer sample boundaries on the ``MemberBoundary`` part. The schema
    is in place from day one as the forward-compat FK target for
    ``SortingSelection``.
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
    cache_hash: char(64)
    """

    class MemberBoundary(SpyglassMixinPart):
        """Per-member end-sample boundary in the concatenated recording."""

        definition = """
        -> master
        member_index: int
        ---
        end_sample: bigint
        """

    def make(self, key):
        """Materialize the concatenated recording cache.

        Gated until the concat materializer lands (raises
        ``NotImplementedError`` today).
        Forward-declared. ``SortingSelection.insert_selection`` also
        rejects ``ConcatenatedRecordingSource`` requests with a matching
        error until the consumer lands; the schemas are in their final
        shape so both helpers can be relaxed without a migration.

        Parameters
        ----------
        key : dict
            The ``ConcatenatedRecordingSelection`` primary key being
            populated.

        Raises
        ------
        NotImplementedError
            Always, until the concat materializer is implemented.
        """
        raise NotImplementedError(
            "ConcatenatedRecording.make() is not implemented yet"
        )
