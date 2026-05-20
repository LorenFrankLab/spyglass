"""Curation of sorted units.

Slice 1a lands the final-shape table declarations; slice 1d fills in the
classmethod / make() bodies that record manual labels and merges and
register into ``SpikeSortingOutput.CurationV2``.

Tables (all final-shape under the zero-migration policy):
    CurationV2 (+ Unit + UnitLabel) -- Manual; lineage via parent_curation_id.

``metrics_source`` is restricted to true CurationV2 provenance values
(``manual``, ``analyzer_curation``, ``figpack``). External or
ground-truth NWB Units continue to use ``ImportedSpikeSorting``; Phase 1
does NOT duplicate them into ``CurationV2``.
"""

from __future__ import annotations

import datajoint as dj

from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2.sorting import Sorting  # noqa: F401
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("spikesorting_v2_curation")


@schema
class CurationV2(SpyglassMixin, dj.Manual):
    """Manual curation labels + merge groups for a sorted Sorting.

    Multiple curations per sort are allowed via ``curation_id``; each
    can record a ``parent_curation_id`` for lineage (validation-only --
    the schema does not self-FK because DataJoint cannot resolve a
    nullable self-FK across renamed columns cleanly).

    ``object_id`` (NOT ``units_object_id``) matches the convention
    ``SpikeSortingOutput.get_spike_times()`` dispatches against; see
    shared-contracts ``NWB Column-Name Convention for SpikeSortingOutput
    Routing``.
    """

    definition = """
    -> Sorting
    curation_id=0: int
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)
    merges_applied=0: bool
    metrics_source = 'manual': enum('manual', 'analyzer_curation', 'figpack')
    description: varchar(255)
    """

    class Unit(SpyglassMixinPart):
        """Per-curated-unit peak channel.

        Populated by ``insert_curation`` from the upstream
        ``Sorting.Unit`` rows after applying ``merge_groups``. A merged
        unit inherits the peak channel of the highest-amplitude
        contributing unit; see shared-contracts ``Unit-Level Brain
        Region Tracing``.
        """

        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uV: float
        n_spikes: int
        """

    class UnitLabel(SpyglassMixinPart):
        """Labels on curated units; one row per (unit, label).

        A unit may carry multiple labels (e.g. ``mua`` + ``burst_parent``);
        unlabeled units have zero ``UnitLabel`` rows. The NWB units table
        still gets a ``curation_label`` indexed column so v1-style
        consumers see empty lists for unlabeled units. Labels are
        validated against the ``CurationLabel`` enum at insert time
        (see shared-contracts ``Curation Label Enum``).
        """

        definition = """
        -> CurationV2.Unit
        curation_label: varchar(32)
        """

    @classmethod
    def insert_curation(
        cls,
        sorting_key: dict,
        labels: dict,
        parent_curation_id: int = -1,
        merge_groups=None,
        apply_merges: bool = False,
        description: str = "",
    ) -> dict:
        """Insert master + Unit + UnitLabel rows + merge-table registration.

        All four inserts run inside one ``cls.connection.transaction``
        block; the curated-units NWB is staged separately and is
        deleted on any later failure because DataJoint cannot roll back
        filesystem side effects. Implemented in slice 1d.

        Parameters
        ----------
        sorting_key
            ``{sorting_id}`` of the upstream Sorting row.
        labels
            Dict ``unit_id -> [label, ...]`` validated against
            ``CurationLabel``. Use ``{}`` for no labels; ``None`` is
            invalid.
        parent_curation_id
            ``-1`` for a root curation; otherwise must reference an
            existing CurationV2 row for the same sorting.
        merge_groups
            Optional list of merge-group dicts (slice 1d).
        apply_merges
            If True, write the merged-units NWB; otherwise the
            curated-units NWB matches the source Sorting units 1:1.
        description
            Free-text curation description.
        """
        raise NotImplementedError(
            "CurationV2.insert_curation lands in slice 1d"
        )

    def get_sorting(self, key, as_dataframe: bool = False):
        """Return the curated SpikeInterface BaseSorting (or DataFrame)."""
        raise NotImplementedError("CurationV2.get_sorting lands in slice 1d")

    def get_merged_sorting(self, key):
        """Return the curated BaseSorting with merge groups applied."""
        raise NotImplementedError(
            "CurationV2.get_merged_sorting lands in slice 1d"
        )

    def get_unit_brain_regions(
        self,
        key,
        *,
        include_labels=None,
        allow_anchor_member: bool = False,
    ):
        """Per-unit brain regions via CurationV2.Unit * Electrode * BrainRegion.

        If ``include_labels`` is provided, restricts through
        ``CurationV2.UnitLabel`` and returns units with any requested
        label. Same concat-sort guard semantics as
        ``Sorting.get_unit_brain_regions``. Implemented in slice 1d.
        """
        raise NotImplementedError(
            "CurationV2.get_unit_brain_regions lands in slice 1d"
        )

    def get_matchable_unit_ids(
        self,
        key,
        exclude_labels=frozenset({"reject", "noise", "artifact"}),
    ):
        """Curated unit IDs with no excluded labels. Implemented in slice 1d."""
        raise NotImplementedError(
            "CurationV2.get_matchable_unit_ids lands in slice 1d"
        )

    def get_sort_group_info(self, key):
        """Return ALL electrodes in the sort group joined to BrainRegion.

        Fix for the v1 ``fetch(limit=1)`` multi-region under-reporting
        bug. Returns a DataJoint relation (not a DataFrame) so callers
        can chain restrictions. Implemented in slice 1d.
        """
        raise NotImplementedError(
            "CurationV2.get_sort_group_info lands in slice 1d"
        )
