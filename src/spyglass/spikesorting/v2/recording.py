"""Preprocessed recording materialization for spike sorting.

Slice 1a lands the final-shape table declarations; slice 1b fills in the
classmethod / make() bodies that materialize the preprocessed
``ElectricalSeries`` into an ``AnalysisNwbfile``.

Tables (all final-shape under the zero-migration policy):
    SortGroupV2          -- per-session electrode grouping.
    PreprocessingParameters -- Pydantic-validated preprocessing blob.
    RecordingSelection   -- one row per (raw, sort group, interval, params).
    Recording            -- materialized preprocessed recording (NWB-resident).
"""

from __future__ import annotations

import datajoint as dj

from spyglass.common import IntervalList, LabTeam, Session  # noqa: F401
from spyglass.common.common_ephys import Electrode, Raw  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2.utils import _validate_params
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("spikesorting_v2_recording")


@schema
class SortGroupV2(SpyglassMixin, dj.Manual):
    """Electrode groupings used for spike sorting.

    A sort group is one tetrode, one shank, or any other contiguous subset
    of channels processed together. The master row stores the group's
    Session FK and reference electrode; the part table enumerates the
    Electrode FKs that make up the group.

    Constructors landing in slice 1b:
        ``set_group_by_shank``
        ``set_group_by_electrode_table_column``
    """

    definition = """
    -> Session
    sort_group_id: int
    ---
    sort_reference_electrode_id = -1: int
    """

    class SortGroupElectrode(SpyglassMixinPart):
        definition = """
        -> master
        -> Electrode
        """


@schema
class PreprocessingParameters(SpyglassMixin, dj.Lookup):
    """Bandpass + reference + optional whitening parameters.

    The ``params`` blob is validated by :class:`PreprocessingParamsSchema`.
    ``insert_default`` bulk-inserts the v2 default presets.
    """

    definition = """
    preproc_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    _DEFAULT_CONTENTS: tuple = (
        (
            "default_franklab",
            PreprocessingParamsSchema().model_dump(),
            1,
            None,
        ),
        (
            "default_neuropixels",
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
                    "whiten": None,
                }
            ).model_dump(),
            1,
            None,
        ),
        (
            "no_filter",
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {
                        "freq_min": 1.0,
                        "freq_max": 14999.0,
                    },
                    "whiten": None,
                }
            ).model_dump(),
            1,
            None,
        ),
    )

    def insert1(self, row, **kwargs):
        row = dict(row)
        row["params"] = _validate_params(
            PreprocessingParamsSchema, row["params"]
        )
        super().insert1(row, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert v2 default preprocessing presets if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class RecordingSelection(SpyglassMixin, dj.Manual):
    """One row per (raw, sort group, interval, preproc params, team).

    UUID-keyed so downstream FKs (``Recording``, ``SortingSelection``) are
    single-column. ``insert_selection`` follows the shared-contracts
    ``insert_selection() Return-Value Normalization`` convention.
    """

    definition = """
    recording_id: uuid
    ---
    -> Raw
    -> SortGroupV2
    -> IntervalList
    -> PreprocessingParameters
    -> LabTeam
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Find-existing-or-insert; returns a single PK-only dict.

        Implemented in slice 1b.
        """
        raise NotImplementedError(
            "RecordingSelection.insert_selection lands in slice 1b"
        )


@schema
class Recording(SpyglassMixin, dj.Computed):
    """Preprocessed recording materialized NWB-resident in AnalysisNwbfile.

    The preprocessed ``ElectricalSeries`` lives inside an
    ``AnalysisNwbfile`` (the canonical artifact). No binary sidecar in
    Phase 1; see shared-contracts ``Recording Cache Format``.
    """

    definition = """
    -> RecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255)
    object_id: varchar(72)
    n_channels: int
    sampling_frequency: float
    duration_s: float
    cache_hash: char(64)
    """

    def make(self, key):
        """Materialize the preprocessed recording. Implemented in slice 1b."""
        raise NotImplementedError(
            "Recording.make lands in slice 1b (recording chain)"
        )

    def get_recording(self, key):
        """Return the preprocessed SpikeInterface recording.

        Rebuilds the NWB artifact on demand if missing, without deleting
        the DataJoint row. Implemented in slice 1b.
        """
        raise NotImplementedError(
            "Recording.get_recording lands in slice 1b"
        )
