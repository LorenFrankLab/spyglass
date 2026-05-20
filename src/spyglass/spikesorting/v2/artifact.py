"""Artifact detection over a preprocessed Recording.

Slice 1a lands the final-shape table declarations; slice 1b fills in the
classmethod / make() bodies that scan the recording for noisy intervals
and write the artifact-removed valid times into ``common.IntervalList``
under name ``f"artifact_{artifact_id}"``.

Tables (all final-shape under the zero-migration policy):
    ArtifactDetectionParameters       -- threshold detection parameters.
    SharedArtifactGroup (+ Member)    -- opt-in cross-recording detection.
    ArtifactDetectionSelection        -- source parts encode input shape.
        .RecordingSource              -- single-recording path (default).
        .SharedArtifactGroupSource    -- cross-recording path (#928).
    ArtifactDetection                 -- writes IntervalList rows; no part.

Artifact-removed valid times live in ``common.IntervalList`` rather than a
dedicated part table -- the UUID-suffixed name prevents collision with
human-authored session intervals while letting downstream
IntervalList-querying code consume them through the standard interface.
"""

from __future__ import annotations

import datajoint as dj

from spyglass.common import IntervalList, Session  # noqa: F401
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.utils import _validate_params
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("spikesorting_v2_artifact")


@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    """Validated artifact-detection parameter blob.

    The ``params`` blob is validated by
    :class:`ArtifactDetectionParamsSchema`. ``insert_default`` ships two
    presets: ``"none"`` (detect=False, skip artifact scanning) and
    ``"default"`` (amplitude threshold + proportion-above-thresh).
    """

    definition = """
    artifact_params_name: varchar(64)
    ---
    params: blob
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    _DEFAULT_CONTENTS: tuple = (
        (
            "none",
            ArtifactDetectionParamsSchema(
                detect=False, amplitude_thresh_uV=None
            ).model_dump(),
            1,
            None,
        ),
        (
            "default",
            ArtifactDetectionParamsSchema().model_dump(),
            1,
            None,
        ),
    )

    def insert1(self, row, **kwargs):
        row = dict(row)
        row["params"] = _validate_params(
            ArtifactDetectionParamsSchema, row["params"]
        )
        super().insert1(row, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert v2 default artifact-detection presets if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class SharedArtifactGroup(SpyglassMixin, dj.Manual):
    """Named bundle of Recording rows that share an artifact-detection pass.

    Addresses Spyglass issue #928 (behavioral artifacts visible on every
    probe -- chewing, licking, head-bumps). Per-recording artifact
    detection misses these because each sort group is processed
    independently. ``SharedArtifactGroup`` lets users declare a set of
    Recording rows from the same session whose artifact intervals should
    be unioned; one detection pass over the union of channels produces a
    shared set of valid times applied to every member.

    All members must belong to one session (enforced by the master row's
    Session FK and re-checked by ``insert_group`` in slice 1b).
    """

    definition = """
    shared_artifact_group_name: varchar(64)
    ---
    -> Session
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        -> Recording
        """

    @classmethod
    def insert_group(cls, name: str, members: list[dict]) -> None:
        """Insert master + Member rows; validate session consistency.

        Implemented in slice 1b.
        """
        raise NotImplementedError(
            "SharedArtifactGroup.insert_group lands in slice 1b"
        )


@schema
class ArtifactDetectionSelection(SpyglassMixin, dj.Manual):
    """One row per (parameters, source) artifact detection request.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` (single-recording, default) or
    ``SharedArtifactGroupSource`` (cross-recording, opt-in) must exist for
    each selection row. Enforced by ``insert_selection`` and re-checked at
    the start of ``ArtifactDetection.make()`` per the shared-contracts
    Source Part Pattern.
    """

    definition = """
    artifact_id: uuid
    ---
    -> ArtifactDetectionParameters
    """

    class RecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> Recording
        """

    class SharedArtifactGroupSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> SharedArtifactGroup
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert master + exactly one source part; return PK-only dict.

        Implemented in slice 1b.
        """
        raise NotImplementedError(
            "ArtifactDetectionSelection.insert_selection lands in slice 1b"
        )

    @classmethod
    def resolve_source(cls, key: dict):
        """Return the source part class + row for an artifact selection.

        Used by ``ArtifactDetection.make()`` to dispatch on source shape
        and by the integrity tests to assert exactly-one-source.

        Implemented in slice 1b.
        """
        raise NotImplementedError(
            "ArtifactDetectionSelection.resolve_source lands in slice 1b"
        )


@schema
class ArtifactDetection(SpyglassMixin, dj.Computed):
    """Artifact-removed valid times for a Recording or SharedArtifactGroup.

    The valid times are written to ``common.IntervalList`` under name
    ``f"artifact_{artifact_id}"`` at the end of ``make()`` -- one row per
    affected session. Single-recording detections write exactly one row;
    cross-recording detections write one row per distinct member
    ``nwb_file_name``.
    """

    definition = """
    -> ArtifactDetectionSelection
    """

    def make(self, key):
        """Detect artifacts and write IntervalList rows.

        Implemented in slice 1b. ``make()`` MUST re-check that the
        upstream selection has exactly one source part row at entry, per
        the shared-contracts Source Part Pattern.
        """
        raise NotImplementedError(
            "ArtifactDetection.make lands in slice 1b (recording chain)"
        )

    def get_artifact_removed_intervals(self, key):
        """Thin ``IntervalList.fetch1('valid_times')`` wrapper.

        Implemented in slice 1b.
        """
        raise NotImplementedError(
            "ArtifactDetection.get_artifact_removed_intervals lands in slice 1b"
        )

    def delete(self, *args, **kwargs):
        """Override that also removes the matching IntervalList rows.

        DataJoint does not cascade through ``interval_list_name``-keyed
        dependencies, so the cleanup is explicit. Implemented in slice 1b.
        """
        raise NotImplementedError(
            "ArtifactDetection.delete override lands in slice 1b"
        )
