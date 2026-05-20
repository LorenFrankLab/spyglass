"""Spike sorting and per-unit brain-region metadata.

Tables (all final-shape under the zero-migration policy):
    SorterParameters          -- Per-sorter Pydantic-validated params.
    SortingSelection          -- Source-polymorphic sorting request.
        .RecordingSource          -- single-session source (default).
        .ConcatenatedRecordingSource -- concat source; the runtime helper
                                       rejects this source today, but the
                                       FK is real so the schema is stable.
    Sorting (+ Unit)          -- Sorted units NWB + SortingAnalyzer folder.

``insert1`` on ``SorterParameters`` is live and dispatches to the
per-sorter Pydantic schema via ``_get_sorter_schema``; ``make`` /
``insert_selection`` / accessor methods are forward-declared stubs that
raise ``NotImplementedError`` until the matching runtime change lands.
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
    ``_get_sorter_schema(sorter)``. ``insert_default`` ships explicit
    default rows for MS4, MS5, KS4, SC2, TDC2, and
    ``clusterless_thresholder``. Users can insert additional rows for any
    installed SI sorter; the generic ``extra="allow"`` schema is the
    fallback dispatch for non-default sorters, preserving v1's "try any
    installed sorter" escape hatch.
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

    _DEFAULT_CONTENTS: tuple = (
        (
            "mountainsort4",
            "franklab_tetrode_hippocampus_30kHz_ms4",
            _validate_params(
                _get_sorter_schema("mountainsort4"),
                {"freq_min": 600.0, "freq_max": 6000.0},
            ),
            1,
            None,
        ),
        (
            "mountainsort5",
            "franklab_tetrode_hippocampus_30kHz_ms5",
            _validate_params(_get_sorter_schema("mountainsort5"), {}),
            1,
            None,
        ),
        (
            "kilosort4",
            "franklab_neuropixels_default",
            _validate_params(_get_sorter_schema("kilosort4"), {}),
            1,
            None,
        ),
        (
            "spykingcircus2",
            "default",
            _validate_params(_get_sorter_schema("spykingcircus2"), {}),
            1,
            None,
        ),
        (
            "tridesclous2",
            "default",
            _validate_params(_get_sorter_schema("tridesclous2"), {}),
            1,
            None,
        ),
        (
            "clusterless_thresholder",
            "default",
            _validate_params(
                _get_sorter_schema("clusterless_thresholder"), {}
            ),
            1,
            None,
        ),
    )

    @classmethod
    def insert_default(cls):
        """Insert v2 default sorter rows if missing.

        The default-content list mirrors the designs.md
        ``SorterParameters`` section and includes MS4, MS5, KS4, SC2,
        TDC2, and clusterless_thresholder. MS4 + KS4 are not
        deterministic and ``clusterless_thresholder`` is a Spyglass
        peak-detection special case, not an SI registered sorter; see
        the per-sorter Pydantic schemas in
        ``spyglass.spikesorting.v2._params.sorter`` for the validated
        field surface.
        """
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class SortingSelection(SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` or ``ConcatenatedRecordingSource`` exists for
    each selection row. The runtime helper today rejects the concat
    path with a clear "not implemented yet" error; the schema is final
    so the validator can be relaxed without a migration once the concat
    materializer lands. ``artifact_id`` is a real DataJoint nullable FK
    to ``ArtifactDetection`` (not a loose UUID column) so DataJoint
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

        Today the helper rejects ``ConcatenatedRecordingSource`` requests
        with a "concatenated recording sorting is not implemented yet"
        error; only the runtime validator changes when the concat
        materializer lands, not the schema.
        """
        raise NotImplementedError(
            "SortingSelection.insert_selection is not yet implemented"
        )

    @classmethod
    def resolve_source(cls, key: dict):
        """Return the source part class + row for a sorting selection.

        Implemented in a follow-up change.
        """
        raise NotImplementedError(
            "SortingSelection.resolve_source is not yet implemented"
        )


@schema
class Sorting(SpyglassMixin, dj.Computed):
    """Sorted units NWB + SortingAnalyzer folder.

    ``make()`` applies post-motion preprocessing, runs the sorter,
    removes excess spikes, builds a
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
        peak_amplitude_uv: float    # peak template amplitude in microvolts
        n_spikes: int
        """

    def make(self, key):
        """Sort, build analyzer, populate Unit part.

        Re-checks ``SortingSelection.resolve_source(key)`` at entry per
        the shared-contracts Source Part Pattern.
        """
        raise NotImplementedError(
            "Sorting.make is not yet implemented"
        )

    def get_sorting(self, key):
        """Return the SpikeInterface BaseSorting. Implemented in a follow-up change."""
        raise NotImplementedError("Sorting.get_sorting is not yet implemented")

    def get_analyzer(self, key):
        """Return the SortingAnalyzer; rebuild on missing folder.

        Implemented in a follow-up change. Recompute is in-place; the DataJoint row
        is not deleted on a missing analyzer folder.
        """
        raise NotImplementedError("Sorting.get_analyzer is not yet implemented")

    def get_unit_brain_regions(
        self, key, *, allow_anchor_member: bool = False
    ):
        """Per-unit brain regions via Sorting.Unit * Electrode * BrainRegion.

        Single-session sorts return ``region_resolution='single_session'``.
        Concat sorts raise ``ConcatBrainRegionAmbiguousError`` unless
        ``allow_anchor_member=True``; with the flag the returned rows are
        labeled ``region_resolution='anchor_member'``.
        """
        raise NotImplementedError(
            "Sorting.get_unit_brain_regions is not yet implemented"
        )
