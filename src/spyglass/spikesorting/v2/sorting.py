"""Spike sorting and per-unit brain-region metadata.

Slice 1a lands the final-shape table declarations and the Pydantic
validation of ``SorterParameters``; slice 1c fills in the classmethod /
make() bodies that run SpikeInterface sorters, build SortingAnalyzers,
and populate ``Sorting.Unit`` with peak-channel metadata.

Tables (all final-shape under the zero-migration policy):
    SorterParameters          -- Per-sorter Pydantic-validated params.
    SortingSelection          -- Source-polymorphic sorting request.
        .RecordingSource          -- single-session source (default).
        .ConcatenatedRecordingSource -- concat source; rejected until Phase 3.
    Sorting (+ Unit)          -- Sorted units NWB + SortingAnalyzer folder.
"""

from __future__ import annotations

import datajoint as dj

from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema
from spyglass.spikesorting.v2.artifact import ArtifactDetection  # noqa: F401
from spyglass.spikesorting.v2.recording import Recording  # noqa: F401
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,  # noqa: F401
)
from spyglass.spikesorting.v2.utils import _validate_params
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("spikesorting_v2_sorting")


@schema
class SorterParameters(SpyglassMixin, dj.Lookup):
    """Per-sorter Pydantic-validated parameter blob.

    The ``params`` blob is validated by the per-sorter schema returned by
    ``_get_sorter_schema(sorter)``. Phase 1 ships explicit default rows
    for MS4, MS5, KS4, SC2, TDC2, and ``clusterless_thresholder``. Users
    can insert additional rows for any installed SI sorter; the generic
    ``extra="allow"`` schema is the fallback dispatch for non-default
    sorters, preserving v1's "try any installed sorter" escape hatch.
    """

    definition = """
    sorter: varchar(64)
    sorter_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    def insert1(self, row, **kwargs):
        row = dict(row)
        schema_cls = _get_sorter_schema(row["sorter"])
        row["params"] = _validate_params(schema_cls, row["params"])
        super().insert1(row, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert v2 default sorter rows if missing.

        Implemented in slice 1c (alongside the populate() body that
        actually consumes them). The default-content list is documented
        in ``designs.md § SorterParameters + SortingSelection + Sorting``
        and includes MS4, MS5, KS4, SC2, TDC2, and clusterless_thresholder.
        """
        raise NotImplementedError(
            "SorterParameters.insert_default lands in slice 1c"
        )


@schema
class SortingSelection(SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` or ``ConcatenatedRecordingSource`` exists for
    each selection row. Phase 1 ``insert_selection`` rejects the
    concat path with a clear "not implemented yet" error until Phase 3.
    ``artifact_id`` is a real DataJoint nullable FK to
    ``ArtifactDetection`` (not a loose UUID column) so DataJoint
    enforces referential integrity.
    """

    definition = """
    sorting_id: uuid
    ---
    -> SorterParameters
    -> [nullable] ArtifactDetection
    """

    class RecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> Recording
        """

    class ConcatenatedRecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> ConcatenatedRecording
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert master + exactly one source part; return PK-only dict.

        Phase 1 rejects ``ConcatenatedRecordingSource`` requests with
        ``NotImplementedError("Concatenated recording sorting is not "
        "implemented yet")``; the validator gate is what changes in
        Phase 3, not the schema. Implemented in slice 1c.
        """
        raise NotImplementedError(
            "SortingSelection.insert_selection lands in slice 1c"
        )

    @classmethod
    def resolve_source(cls, key: dict):
        """Return the source part class + row for a sorting selection.

        Implemented in slice 1c.
        """
        raise NotImplementedError(
            "SortingSelection.resolve_source lands in slice 1c"
        )


@schema
class Sorting(SpyglassMixin, dj.Computed):
    """Sorted units NWB + SortingAnalyzer folder.

    Slice 1c materializes the sort: applies post-motion preprocessing,
    runs the sorter, removes excess spikes, builds a
    ``SortingAnalyzer(format="binary_folder", sparse=True)``, computes
    the base extensions (``random_spikes``, ``noise_levels``,
    ``templates``, ``waveforms``), writes a fresh/whitelisted
    ``AnalysisNwbfile`` containing only the v2 sorting Units (NOT the
    parent NWB's units), and populates ``Sorting.Unit``.
    """

    definition = """
    -> SortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(72)
    analyzer_folder: varchar(255)
    n_units: int
    time_of_sort: datetime
    """

    class Unit(SpyglassMixinPart):
        """Per-unit metadata persisted at sort time.

        Brain region is reached through ``Sorting.Unit * Electrode *
        BrainRegion`` (NON-NULL on Spyglass's ``Electrode``); see
        shared-contracts ``Unit-Level Brain Region Tracing``. For concat
        sorts the Electrode FK is anchored to the FIRST member's row.
        """

        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uV: float
        n_spikes: int
        """

    def make(self, key):
        """Sort, build analyzer, populate Unit part. Implemented in slice 1c.

        Re-checks ``SortingSelection.resolve_source(key)`` at entry per
        the shared-contracts Source Part Pattern.
        """
        raise NotImplementedError(
            "Sorting.make lands in slice 1c (sorting + analyzer chain)"
        )

    def get_sorting(self, key):
        """Return the SpikeInterface BaseSorting. Implemented in slice 1c."""
        raise NotImplementedError("Sorting.get_sorting lands in slice 1c")

    def get_analyzer(self, key):
        """Return the SortingAnalyzer; rebuild on missing folder.

        Implemented in slice 1c. Recompute is in-place; the DataJoint row
        is not deleted on a missing analyzer folder.
        """
        raise NotImplementedError("Sorting.get_analyzer lands in slice 1c")

    def get_unit_brain_regions(
        self, key, *, allow_anchor_member: bool = False
    ):
        """Per-unit brain regions via Sorting.Unit * Electrode * BrainRegion.

        Single-session sorts return ``region_resolution='single_session'``.
        Concat sorts raise ``ConcatBrainRegionAmbiguousError`` unless
        ``allow_anchor_member=True``; with the flag the returned rows are
        labeled ``region_resolution='anchor_member'``. Implemented in
        slice 1c.
        """
        raise NotImplementedError(
            "Sorting.get_unit_brain_regions lands in slice 1c"
        )
