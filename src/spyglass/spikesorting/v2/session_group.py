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

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import spikeinterface as si

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
class ConcatenatedRecordingSelection(SpyglassMixin, dj.Manual):
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
    """

    #: The logical-identity fields (everything but the minted PK). A concat
    #: selection is one (SessionGroup, PreprocessingParameters,
    #: MotionCorrectionParameters) tuple; find-existing keys on all four.
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
        import uuid

        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
            MissingRecordingForConcatError,
        )
        from spyglass.spikesorting.v2.recording import (
            Recording,
            RecordingSelection,
        )

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

        # Precondition: every member needs a populated Recording for the
        # shared preprocessing recipe. A Recording exists iff its
        # RecordingSelection row exists AND the Computed Recording populated.
        missing: list[dict] = []
        for member in (SessionGroup.Member & group_key).fetch(as_dict=True):
            rec_sel_key = {
                "nwb_file_name": member["nwb_file_name"],
                "sort_group_id": member["sort_group_id"],
                "interval_list_name": member["interval_list_name"],
                "preprocessing_params_name": preprocessing_params_name,
                "team_name": member["team_name"],
            }
            rec_sel = RecordingSelection & rec_sel_key
            if not rec_sel or not (Recording & rec_sel.fetch1("KEY")):
                missing.append(rec_sel_key)
        if missing:
            raise MissingRecordingForConcatError(
                "ConcatenatedRecordingSelection.insert_selection requires "
                "every member's Recording to be populated under "
                f"preprocessing_params_name={preprocessing_params_name!r} "
                f"first. Missing {len(missing)} member(s): {missing}. Run "
                "Recording.populate(...) for each missing key, then retry."
            )

        existing = (cls & identity).fetch("KEY", as_dict=True)
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise DuplicateSelectionError(
                "ConcatenatedRecordingSelection has "
                f"{len(existing)} duplicate selection rows for identity "
                f"{identity}; a raw insert bypassed insert_selection. Drop the "
                "duplicates and re-insert via insert_selection."
            )
        concat_recording_id = uuid.uuid4()
        cls.insert1({**identity, "concat_recording_id": concat_recording_id})
        return {"concat_recording_id": concat_recording_id}


@schema
class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    """Materialized cross-session concatenated recording cache.

    ``make()`` writes a single motion-corrected, unwhitened
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
        """Materialize the motion-corrected, unwhitened concat cache.

        Reuses each member's already-populated, cached ``Recording`` artifact
        (NEVER calls ``Recording.populate`` -- a DataJoint anti-pattern; the
        selection-time precondition guarantees the rows exist, and this method
        defensively re-checks), stitches them into one mono-segment recording,
        applies motion correction (resolving the Spyglass ``"auto"`` alias to
        a same-day preset and rejecting it on multi-day groups), and writes a
        single ``ElectricalSeries`` into an ``AnalysisNwbfile``. Whitening is
        deliberately NOT applied here -- it stays a sorter/analyzer concern, so
        the persisted concat recording is motion-corrected but unwhitened.
        Cumulative per-member sample boundaries are persisted on
        ``MemberBoundary`` for spike-time back-mapping.

        Parameters
        ----------
        key : dict
            The ``ConcatenatedRecordingSelection`` primary key
            (``{"concat_recording_id": ...}``) being populated.

        Raises
        ------
        MissingRecordingForConcatError
            If any member's ``Recording`` row is missing (the selection-time
            precondition was bypassed).
        ValueError
            If ``preset="auto"`` on a multi-day group.
        """
        import datetime as dt

        from spyglass.spikesorting.v2._concat_recording import (
            build_concatenated_recording,
            cumulative_member_boundaries,
            resolve_motion_correction,
        )
        from spyglass.spikesorting.v2._recording_nwb import write_nwb_artifact
        from spyglass.spikesorting.v2.exceptions import (
            MissingRecordingForConcatError,
        )
        from spyglass.spikesorting.v2.recording import (
            Recording,
            RecordingSelection,
            _ELECTRICAL_SERIES_PATH,
            _unlink_staged_analysis_file,
        )
        from spyglass.spikesorting.v2.utils import (
            _resolved_job_kwargs,
            transaction_or_noop,
        )

        # The populate key carries only concat_recording_id; every member and
        # parameter query restricts with the fetched selection row, not the
        # UUID-only key, so independent concat selections never cross-restrict.
        sel = (ConcatenatedRecordingSelection & key).fetch1()
        group_key = {
            "session_group_owner": sel["session_group_owner"],
            "session_group_name": sel["session_group_name"],
        }
        preprocessing_params_name = sel["preprocessing_params_name"]
        members = (SessionGroup.Member & group_key).fetch(
            as_dict=True, order_by="member_index"
        )

        recordings = []
        member_sample_counts = []
        member_indices = []
        missing: list[dict] = []
        for member in members:
            rec_sel_key = {
                "nwb_file_name": member["nwb_file_name"],
                "sort_group_id": member["sort_group_id"],
                "interval_list_name": member["interval_list_name"],
                "preprocessing_params_name": preprocessing_params_name,
                "team_name": member["team_name"],
            }
            rec_sel = RecordingSelection & rec_sel_key
            if not rec_sel or not (Recording & rec_sel.fetch1("KEY")):
                missing.append(rec_sel_key)
                continue
            recording = Recording().get_recording(rec_sel.fetch1("KEY"))
            recordings.append(recording)
            member_sample_counts.append(int(recording.get_num_samples()))
            member_indices.append(int(member["member_index"]))
        if missing:
            raise MissingRecordingForConcatError(
                "ConcatenatedRecording.make: "
                f"{len(missing)} member Recording row(s) are not populated "
                f"under preprocessing_params_name={preprocessing_params_name!r}: "
                f"{missing}. Populate Recording for each member first "
                "(ConcatenatedRecordingSelection.insert_selection enforces "
                "this; the selection was likely inserted by a raw bypass)."
            )

        # Motion correction: resolve the Spyglass 'auto' alias against the
        # group's multi-day status before any SpikeInterface call.
        motion_row = (
            MotionCorrectionParameters
            & {
                "motion_correction_params_name": (
                    sel["motion_correction_params_name"]
                )
            }
        ).fetch1()
        motion_preset, preset_kwargs = resolve_motion_correction(
            motion_row["params"],
            is_multi_day=SessionGroup.is_multi_day(group_key),
        )
        # Job kwargs: preprocessing first, motion last (motion wins on
        # conflict), per the job-kwargs resolution contract.
        preprocessing_job_kwargs = (
            PreprocessingParameters
            & {"preprocessing_params_name": preprocessing_params_name}
        ).fetch1("job_kwargs")
        job_kwargs = _resolved_job_kwargs(
            preprocessing_job_kwargs, motion_row["job_kwargs"]
        )

        corrected = build_concatenated_recording(
            recordings,
            motion_preset=motion_preset,
            preset_kwargs=preset_kwargs,
            job_kwargs=job_kwargs,
        )

        # Anchor the analysis NWB to the FIRST member's session (deterministic
        # parent); full multi-session provenance stays queryable through
        # ConcatenatedRecordingSelection -> SessionGroup.Member.
        anchor_nwb_file_name = members[0]["nwb_file_name"]
        preset_label = motion_preset or "none"
        analysis_file_name, object_id, cache_hash = write_nwb_artifact(
            corrected,
            anchor_nwb_file_name,
            filtering_description=(
                f"Concatenated {len(recordings)} member recording(s) "
                f"(preprocessing_params={preprocessing_params_name!r}); "
                f"motion correction preset={preset_label!r}; unwhitened"
            ),
        )

        boundaries = cumulative_member_boundaries(member_sample_counts)
        sampling_frequency = float(corrected.get_sampling_frequency())
        boundary_rows = [
            {
                **key,
                "member_index": member_index,
                "end_sample": int(end_sample),
            }
            for member_index, end_sample in zip(member_indices, boundaries)
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
                        "n_channels": int(corrected.get_num_channels()),
                        "sampling_frequency": sampling_frequency,
                        "total_duration_s": (
                            int(corrected.get_num_samples())
                            / sampling_frequency
                        ),
                        "cache_hash": cache_hash,
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
                context="ConcatenatedRecording.make",
            )
            raise

    def get_recording(self, key) -> "si.BaseRecording":  # noqa: F821
        """Return the cached concatenated SpikeInterface recording.

        Reads the persisted motion-corrected, unwhitened ``ElectricalSeries``
        through the stored ``electrical_series_path`` (authoritative, not an
        auto-detect hint) and annotates ``is_filtered=True`` so a downstream
        sorter does not re-filter the already-filtered cache -- the same
        contract as ``Recording.get_recording``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``ConcatenatedRecording`` row.

        Returns
        -------
        si.BaseRecording
            The concatenated, motion-corrected, unwhitened recording.
        """
        import spikeinterface.extractors as se

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        rec = se.read_nwb_recording(
            abs_path,
            electrical_series_path=row["electrical_series_path"],
            load_time_vector=True,
        )
        rec.annotate(is_filtered=True)
        return rec

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
        dict[tuple[str, str], si.BaseSorting]
            One ``NumpySorting`` per member, keyed by the hashable
            ``(nwb_file_name, interval_list_name)`` tuple, with spike times in
            that member's local sample frame.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._concat_recording import (
            split_unit_spike_trains,
        )

        sel = (ConcatenatedRecordingSelection & key).fetch1()
        group_key = {
            "session_group_owner": sel["session_group_owner"],
            "session_group_name": sel["session_group_name"],
        }
        members = (SessionGroup.Member & group_key).fetch(
            as_dict=True, order_by="member_index"
        )
        # MemberBoundary is keyed by member_index; align to the member order.
        indices, ends = (self.MemberBoundary & key).fetch(
            "member_index", "end_sample"
        )
        end_by_index = {int(i): int(e) for i, e in zip(indices, ends)}
        boundaries = [
            end_by_index[int(member["member_index"])] for member in members
        ]

        unit_trains = {
            int(unit_id): sorting.get_unit_spike_train(unit_id=unit_id)
            for unit_id in sorting.unit_ids
        }
        per_member = split_unit_spike_trains(unit_trains, boundaries)
        fs = float(sorting.get_sampling_frequency())
        return {
            (member["nwb_file_name"], member["interval_list_name"]): (
                si.NumpySorting.from_unit_dict(
                    [local_trains], sampling_frequency=fs
                )
            )
            for member, local_trains in zip(members, per_member)
        }
