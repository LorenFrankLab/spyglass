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
    MotionCorrectionParamsSchema,
)
from spyglass.spikesorting.v2.recording import (
    PreprocessingParameters,  # noqa: F401
    SortGroupV2,  # noqa: F401
)
from spyglass.spikesorting.v2.utils import (
    _assert_schema_version_matches,
    _assert_v2_db_safe,
    _validate_params,
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
        """Insert master + Member rows; derive dates from each Session.

        The schema is final-shape so users can start declaring groups
        for the concat workflow as soon as the Recording chain
        materializes; this helper is forward-declared.
        """
        raise NotImplementedError(
            "SessionGroup.create_group is not yet implemented"
        )

    @classmethod
    def is_multi_day(cls, key: dict) -> bool:
        """True if the group's members span two or more session dates.

        Implemented in a follow-up change.
        """
        raise NotImplementedError(
            "SessionGroup.is_multi_day is not yet implemented"
        )


@schema
class MotionCorrectionParameters(SpyglassMixin, dj.Lookup):
    """Validated motion-correction parameter blob.

    Ships before the consumer (``ConcatenatedRecording.make()``) lands
    so this Lookup's ``insert1`` can Pydantic-validate its ``params``
    blob at insert time. The validation surface is final-shape today;
    the consumer activates without a migration once the concat
    materializer is implemented.
    """

    definition = """
    motion_correction_params_name: varchar(64)
    ---
    params: blob
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    _DEFAULT_CONTENTS: tuple = (
        (
            "none",
            MotionCorrectionParamsSchema().model_dump(),
            1,
            None,
        ),
        (
            "auto_default",
            MotionCorrectionParamsSchema(preset="auto").model_dump(),
            1,
            None,
        ),
        (
            "rigid_fast_default",
            MotionCorrectionParamsSchema(preset="rigid_fast").model_dump(),
            1,
            None,
        ),
    )

    def insert1(self, row, **kwargs):
        row = dict(row)
        row["params"] = _validate_params(
            MotionCorrectionParamsSchema, row["params"]
        )
        # Catch outer-column vs inner-blob schema-version drift, mirroring
        # SorterParameters.insert1: a row whose params_schema_version
        # disagrees with the validated blob's schema_version would route
        # downstream version-branching code to the wrong path.
        _assert_schema_version_matches(
            row,
            MotionCorrectionParamsSchema,
            table_name="MotionCorrectionParameters",
        )
        super().insert1(row, **kwargs)

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
        """Find-existing-or-insert; returns a single PK-only dict.

        Forward-declared; the schema is in place so single-session
        tables can FK ``concat_recording_id`` from day one.
        """
        raise NotImplementedError(
            "ConcatenatedRecordingSelection.insert_selection is not yet "
            "implemented"
        )


@schema
class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    """Materialized cross-session concatenated recording cache.

    When implemented, ``make()`` writes a single sorter-ready
    ``ElectricalSeries`` covering the union of member intervals (with
    motion correction if requested), plus integer sample boundaries on
    the ``MemberBoundary`` part. The schema is in place from day one as
    the forward-compat FK target for ``SortingSelection``.
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
        definition = """
        -> master
        member_index: int
        ---
        end_sample: bigint
        """

    def make(self, key):
        """Materialize the concatenated recording cache.

        Forward-declared. ``SortingSelection.insert_selection`` also
        rejects ``ConcatenatedRecordingSource`` requests with a matching
        error until the consumer lands; the schemas are in their final
        shape so both helpers can be relaxed without a migration.
        """
        raise NotImplementedError(
            "ConcatenatedRecording.make() is not implemented yet"
        )
