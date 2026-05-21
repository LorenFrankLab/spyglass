"""Curation of sorted units.

Tables (all final-shape under the zero-migration policy):
    CurationV2 (+ Unit + UnitLabel) -- Manual; lineage via parent_curation_id.

``metrics_source`` is restricted to true CurationV2 provenance values
(``manual``, ``analyzer_curation``, ``figpack``). External or
ground-truth NWB Units continue to use ``ImportedSpikeSorting``; v2 does
NOT duplicate them into ``CurationV2``.

``insert_curation`` registers each curation atomically across the
master, ``Unit``, and ``UnitLabel`` parts plus the
``SpikeSortingOutput.CurationV2`` merge-table row. The curated-units
NWB is staged first; the AnalysisNwbfile DB-row registration moves
inside the transaction so a later failure rolls the row back and the
staged file is cleaned up in the except path.
"""

from __future__ import annotations

import datajoint as dj

from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
from spyglass.spikesorting.v2.utils import (
    CurationLabel,
    MetricsSource,
    _assert_v2_db_safe,
    transaction_or_noop,
    unit_brain_region_df,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

_assert_v2_db_safe()
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
        peak_amplitude_uv: float    # peak template amplitude in microvolts
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
        merge_groups: list[list[int]] | None = None,
        apply_merges: bool = False,
        description: str = "",
        metrics_source: str | MetricsSource = "manual",
    ) -> dict:
        """Insert master + Unit + UnitLabel rows; stage curated-units NWB.

        Atomically inserts the CurationV2 master, ``Unit`` and
        ``UnitLabel`` part rows, and the ``SpikeSortingOutput.CurationV2``
        merge-table registration in one transaction. The curated-units
        NWB is staged separately and deleted on any later failure
        (DataJoint cannot roll back filesystem side effects).

        Parameters
        ----------
        sorting_key
            ``{sorting_id}`` of the upstream Sorting row.
        labels
            Dict ``unit_id -> [label, ...]``. Each label is validated
            against the ``CurationLabel`` enum. Use ``{}`` for an
            unlabeled curation; ``None`` is rejected (callers must be
            explicit about "no labels").
        parent_curation_id
            ``-1`` for a root curation; otherwise must reference an
            existing CurationV2 row for the same sorting.
        merge_groups
            Optional list of merge groups, each a list of ``unit_id``
            ints. The merged unit inherits the peak channel + amplitude
            of the highest-amplitude contributor; its ``unit_id`` is
            the first id in the group. Non-listed units pass through
            1:1.
        apply_merges
            If True, the curated-units NWB stores merged spike trains
            (union of contributors); if False (default), it stores the
            original Sorting spike trains 1:1 so merge edits can be
            reviewed before committing. CurationV2.Unit always reflects
            the post-merge unit set regardless.
        description
            Free-text curation description.
        metrics_source
            Provenance for any attached metrics blob. Must be one of
            'manual' (default), 'analyzer_curation', or 'figpack'.

        Returns
        -------
        dict
            ``{"sorting_id": ..., "curation_id": ...}`` PK-only dict.
        """
        import pathlib as _pathlib

        sorting_id = sorting_key["sorting_id"]
        # Fetch the upstream Sorting.Unit rows once -- the parent
        # existence check and _build_curated_unit_rows would otherwise
        # query the same restriction twice.
        sorting_units = (
            Sorting.Unit & {"sorting_id": sorting_id}
        ).fetch(as_dict=True)
        if not sorting_units and not (Sorting & {"sorting_id": sorting_id}):
            raise ValueError(
                f"CurationV2.insert_curation: sorting_id {sorting_id} "
                "not in Sorting. Populate Sorting first."
            )
        if labels is None:
            raise ValueError(
                "CurationV2.insert_curation: labels=None is invalid. "
                "Use labels={} for an unlabeled curation."
            )
        # Normalize label keys to int once so the rest of the helper
        # can do straight ``labels.get(int_uid, [])`` lookups.
        labels = {int(uid): list(lbls) for uid, lbls in labels.items()}

        cls._validate_labels(labels)

        if parent_curation_id != -1:
            if not (
                cls
                & {
                    "sorting_id": sorting_id,
                    "curation_id": parent_curation_id,
                }
            ):
                raise ValueError(
                    f"CurationV2.insert_curation: parent_curation_id="
                    f"{parent_curation_id} does not exist for sorting_id="
                    f"{sorting_id}. Pass parent_curation_id=-1 for a "
                    "root curation."
                )

        # Coerce metrics_source through the enum so a typo raises here
        # rather than at the DataJoint enum-mismatch layer.
        try:
            metrics_source = MetricsSource(metrics_source).value
        except ValueError:
            raise ValueError(
                f"CurationV2.insert_curation: metrics_source="
                f"{metrics_source!r} is not a MetricsSource value. "
                f"Valid: {[m.value for m in MetricsSource]}."
            )

        # Resolve which curation_id to use (auto-increment within sort).
        existing_ids = (
            cls & {"sorting_id": sorting_id}
        ).fetch("curation_id")
        curation_id = int(max(existing_ids)) + 1 if len(existing_ids) else 0

        # Resolve the post-merge unit set. For every "kept" unit, we
        # need its peak Electrode FK + amplitude (inherited from the
        # highest-amplitude contributor if it was the head of a merge
        # group).
        unit_rows, kept_unit_to_contributors = cls._build_curated_unit_rows(
            sorting_id=sorting_id,
            sorting_units=sorting_units,
            merge_groups=merge_groups or [],
            curation_id=curation_id,
        )

        # Stage the curated-units NWB (filesystem side-effect; the DB
        # row registration happens inside the transaction block below
        # so it rolls back atomically with the other inserts).
        (
            analysis_file_name,
            units_object_id,
            staged_parent_nwb,
        ) = cls._stage_curated_units_nwb(
            sorting_id=sorting_id,
            kept_unit_to_contributors=kept_unit_to_contributors,
            apply_merges=apply_merges,
            labels=labels,
        )

        try:
            master_row = {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "parent_curation_id": parent_curation_id,
                "analysis_file_name": analysis_file_name,
                "object_id": units_object_id,
                "merges_applied": bool(apply_merges and merge_groups),
                "metrics_source": metrics_source,
                "description": description,
            }
            unit_label_rows = [
                {
                    "sorting_id": sorting_id,
                    "curation_id": curation_id,
                    "unit_id": unit_id,
                    "curation_label": (
                        label.value
                        if isinstance(label, CurationLabel)
                        else str(label)
                    ),
                }
                for unit_id, lbls in labels.items()
                if unit_id in kept_unit_to_contributors
                for label in lbls
            ]
            # Auto-register into SpikeSortingOutput.CurationV2 so
            # downstream consumers (SortedSpikesGroup, decoding, etc.)
            # see the curation without the user having to call the
            # merge insert manually. Imported lazily to keep curation.py
            # free of the merge-table dependency at module load. The
            # merge insert MUST go through ``_merge_insert``, not direct
            # ``insert1`` on the master + part -- the master row's
            # ``merge_id`` is generated by ``_merge_insert`` from the
            # part-row identity hash, and a hand-rolled UUID would fail
            # the master FK referenced by the part.
            from spyglass.spikesorting.spikesorting_merge import (
                SpikeSortingOutput,
            )

            with transaction_or_noop(cls.connection):
                # Register the analysis file row FIRST inside the
                # transaction so it rolls back atomically with the
                # CurationV2 rows on any later failure.
                AnalysisNwbfile().add(staged_parent_nwb, analysis_file_name)
                cls.insert1(master_row)
                cls.Unit.insert(unit_rows)
                if unit_label_rows:
                    cls.UnitLabel.insert(unit_label_rows)
                SpikeSortingOutput._merge_insert(
                    [
                        {
                            "sorting_id": sorting_id,
                            "curation_id": curation_id,
                        }
                    ],
                    part_name="CurationV2",
                    skip_duplicates=True,
                )
        except Exception:
            # The transaction rolled back the AnalysisNwbfile row and
            # the CurationV2 rows together; only the file on disk is
            # left to clean up.
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                if _pathlib.Path(abs_path).exists():
                    _pathlib.Path(abs_path).unlink()
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "CurationV2.insert_curation: failed to clean up "
                    f"staged analysis file {analysis_file_name!r}: "
                    f"{cleanup_exc!r}"
                )
            raise

        return {"sorting_id": sorting_id, "curation_id": curation_id}

    # ---- Validation helpers ---------------------------------------------

    @staticmethod
    def _validate_labels(labels: dict) -> None:
        """Validate that every value in ``labels`` is a recognized enum.

        Accepts ``CurationLabel`` instances OR their string values; any
        other label raises ``ValueError`` listing the offending entry
        and the valid enum names.
        """
        valid = {member.value for member in CurationLabel}
        for unit_id, lbls in labels.items():
            if not isinstance(lbls, (list, tuple)):
                raise ValueError(
                    "CurationV2.insert_curation: labels[unit_id] must be "
                    f"a list of labels; got {type(lbls).__name__} for "
                    f"unit_id={unit_id}."
                )
            for lbl in lbls:
                label_value = (
                    lbl.value if isinstance(lbl, CurationLabel) else lbl
                )
                if label_value not in valid:
                    raise ValueError(
                        f"CurationV2.insert_curation: label {lbl!r} for "
                        f"unit_id={unit_id} is not in CurationLabel. "
                        f"Valid labels: {sorted(valid)}."
                    )

    # ---- Curated-unit construction --------------------------------------

    @classmethod
    def _build_curated_unit_rows(
        cls,
        sorting_id,
        sorting_units: list[dict],
        merge_groups: list[list[int]],
        curation_id: int,
    ) -> tuple[list[dict], dict[int, list[int]]]:
        """Resolve the post-merge ``CurationV2.Unit`` rows.

        Returns ``(unit_rows, kept_unit_to_contributors)``. Each kept
        unit's Electrode FK + peak amplitude come from the contributor
        with the largest ``peak_amplitude_uv``; merged-unit ``unit_id``
        is the first id in the group, matching the v1 convention.

        ``sorting_units`` is the pre-fetched ``Sorting.Unit`` rows for
        ``sorting_id``; the caller fetches once and threads through to
        avoid re-querying.
        """
        if not sorting_units:
            return [], {}
        by_id = {int(row["unit_id"]): row for row in sorting_units}

        # Build mapping from contributor -> kept unit (head of group).
        # A unit may only appear in ONE merge group: overlap across
        # groups would silently double-count spikes when apply_merges
        # is True and ambiguates the "kept unit" choice.
        merged_ids: set[int] = set()
        kept_to_contributors: dict[int, list[int]] = {}
        for group in merge_groups:
            if not group:
                continue
            for uid in group:
                if uid not in by_id:
                    raise ValueError(
                        f"CurationV2.insert_curation: merge_groups "
                        f"references unit_id={uid} that is not in "
                        f"Sorting.Unit for sorting_id={sorting_id}."
                    )
            int_group = [int(u) for u in group]
            overlap = merged_ids & set(int_group)
            if overlap:
                raise ValueError(
                    f"CurationV2.insert_curation: merge_groups overlap "
                    f"on unit_ids {sorted(overlap)}. A unit can belong "
                    "to at most one merge group."
                )
            head = int_group[0]
            if head in kept_to_contributors:
                raise ValueError(
                    f"CurationV2.insert_curation: merge_groups reuses "
                    f"unit_id={head} as the head of two groups. Pick "
                    "distinct heads."
                )
            kept_to_contributors[head] = int_group
            merged_ids.update(int_group)

        # Non-merged units pass through 1:1.
        for uid in by_id:
            if uid not in merged_ids:
                kept_to_contributors[uid] = [uid]

        unit_rows: list[dict] = []
        for kept_uid, contribs in kept_to_contributors.items():
            # Inherit peak channel from the highest-amplitude contributor.
            anchor = max(contribs, key=lambda u: by_id[u]["peak_amplitude_uv"])
            anchor_row = by_id[anchor]
            n_spikes = sum(by_id[u]["n_spikes"] for u in contribs)
            unit_rows.append(
                {
                    "sorting_id": sorting_id,
                    "curation_id": curation_id,
                    "unit_id": int(kept_uid),
                    "nwb_file_name": anchor_row["nwb_file_name"],
                    "electrode_group_name": anchor_row["electrode_group_name"],
                    "electrode_id": int(anchor_row["electrode_id"]),
                    "peak_amplitude_uv": float(anchor_row["peak_amplitude_uv"]),
                    "n_spikes": int(n_spikes),
                }
            )
        return unit_rows, kept_to_contributors

    @classmethod
    def _stage_curated_units_nwb(
        cls,
        sorting_id,
        kept_unit_to_contributors: dict,
        apply_merges: bool,
        labels: dict,
    ) -> tuple[str, str]:
        """Write the curated-units NWB. Returns (analysis_file_name, units_object_id).

        With ``apply_merges=True`` the kept unit's spike train is the
        union of its contributors' spike trains. With
        ``apply_merges=False`` only the non-merged units (and the
        first/head id of each merge group, taken from the source) are
        written 1:1 -- merge edits can still be reviewed before
        committing. CurationV2.Unit reflects the post-merge set
        regardless.
        """
        import numpy as _np
        import pynwb

        from spyglass.spikesorting.v2.recording import RecordingSelection

        rec_source = (
            SortingSelection.RecordingSource & {"sorting_id": sorting_id}
        )
        if not rec_source:
            raise NotImplementedError(
                "CurationV2.insert_curation: only RecordingSource sorts "
                "are supported today; concat-source sorts are not yet "
                "implemented."
            )
        recording_id = rec_source.fetch1("recording_id")
        nwb_file_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name")

        sorting = Sorting().get_sorting({"sorting_id": sorting_id})

        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name=nwb_file_name
        )
        analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            nwbf.add_unit_column(
                name="curation_label",
                description=(
                    "Curation label(s) from CurationV2.insert_curation; "
                    "comma-separated when multi-labeled, empty if "
                    "unlabeled."
                ),
            )
            for kept_uid, contribs in kept_unit_to_contributors.items():
                if apply_merges and len(contribs) > 1:
                    spike_times = _np.concatenate(
                        [
                            sorting.get_unit_spike_train(
                                unit_id=u, return_times=True
                            )
                            for u in contribs
                        ]
                    )
                    spike_times.sort()
                else:
                    spike_times = sorting.get_unit_spike_train(
                        unit_id=kept_uid, return_times=True
                    )
                lbl_list = labels.get(int(kept_uid), [])
                label_str = ",".join(
                    lbl.value if isinstance(lbl, CurationLabel) else str(lbl)
                    for lbl in lbl_list
                )
                nwbf.add_unit(
                    spike_times=_np.asarray(spike_times, dtype=_np.float64),
                    id=int(kept_uid),
                    curation_label=label_str,
                )
            units_object_id = nwbf.units.object_id
            io.write(nwbf)

        # The AnalysisNwbfile DB-row registration (.add) is deliberately
        # NOT done here -- the caller does it inside its transaction
        # block so the row rolls back atomically if any of the
        # CurationV2 / Unit / UnitLabel / merge-insert steps fail.
        # The file on disk is the only side effect left to clean up on
        # rollback.
        return analysis_file_name, units_object_id, nwb_file_name

    # ---- Accessors -------------------------------------------------------

    def get_sorting(self, key, as_dataframe: bool = False):
        """Return the curated SpikeInterface BaseSorting (or DataFrame).

        With ``as_dataframe=True`` returns a pandas DataFrame mirroring
        v1's ``Curation.fetch_nwb`` convenience -- one row per unit
        with the spike-times list, useful for ad-hoc inspection that
        does not need a full SI sorting object.
        """
        from spikeinterface.extractors import NwbSortingExtractor

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

        # CurationV2's units NWB does not carry an ElectricalSeries, so
        # we must supply ``sampling_frequency`` and ``t_start`` directly
        # (mirrors ``Sorting.get_sorting``). t_start is the recording's
        # first wall-clock timestamp.
        recording_row = self._upstream_recording_row(key)
        fs = float(recording_row["sampling_frequency"])
        t_start = Sorting._recording_t_start(recording_row)

        si_sorting = NwbSortingExtractor(
            file_path=abs_path, sampling_frequency=fs, t_start=t_start
        )
        if not as_dataframe:
            return si_sorting

        import pandas as pd

        return pd.DataFrame(
            {
                "unit_id": [int(uid) for uid in si_sorting.unit_ids],
                "spike_times": [
                    si_sorting.get_unit_spike_train(
                        unit_id=uid, return_times=True
                    )
                    for uid in si_sorting.unit_ids
                ],
            }
        )

    def get_merged_sorting(self, key):
        """Return the curated BaseSorting with merge groups applied.

        ``insert_curation(apply_merges=True)`` already writes the
        merged spike trains into the NWB, so this method is a
        convenience alias for ``get_sorting(key)`` on a merges-applied
        row. For curations where ``merges_applied=False`` the returned
        sorting is identical to the source Sorting's spike trains.
        """
        return self.get_sorting(key)

    def get_unit_brain_regions(
        self,
        key,
        *,
        include_labels=None,
        allow_anchor_member: bool = False,
    ):
        """Per-unit brain regions via CurationV2.Unit * Electrode * BrainRegion.

        If ``include_labels`` is provided (iterable of strings or
        ``CurationLabel``), restricts to units carrying at least one
        of those labels. Otherwise returns all CurationV2.Unit rows
        for the key. Same concat-sort guard semantics as
        ``Sorting.get_unit_brain_regions``: raises
        ``ConcatBrainRegionAmbiguousError`` for concat-backed
        sortings unless ``allow_anchor_member=True``.
        """
        from spyglass.spikesorting.v2.exceptions import (
            ConcatBrainRegionAmbiguousError,
        )

        source = SortingSelection.resolve_source(
            {"sorting_id": key["sorting_id"]}
        )
        if source.kind == "concatenated_recording":
            if not allow_anchor_member:
                raise ConcatBrainRegionAmbiguousError(
                    f"CurationV2.get_unit_brain_regions: sorting_id "
                    f"{key['sorting_id']} is concat-backed; pass "
                    "allow_anchor_member=True for anchor-only regions or "
                    "use TrackedUnit.get_unit_brain_regions for per-"
                    "session regions."
                )
            resolution = "anchor_member"
        else:
            resolution = "single_session"

        unit_restriction = self.Unit & key
        if include_labels is not None:
            include_values = {
                lbl.value if isinstance(lbl, CurationLabel) else str(lbl)
                for lbl in include_labels
            }
            labeled = (
                self.UnitLabel & key & [{"curation_label": v} for v in include_values]
            ).fetch("unit_id")
            unit_restriction = unit_restriction & [
                {"unit_id": int(uid)} for uid in labeled
            ]
        return unit_brain_region_df(unit_restriction, resolution)

    def get_matchable_unit_ids(
        self,
        key,
        exclude_labels=frozenset({"reject", "noise", "artifact"}),
    ):
        """Curated unit IDs with no excluded labels.

        Unlabeled units AND units labeled only ``accept`` / ``mua`` are
        included. A unit with ANY excluded label is excluded even if it
        also carries an included label (e.g., a ``mua`` + ``artifact``
        unit is excluded).
        """
        import numpy as np

        exclude_values = {
            lbl.value if isinstance(lbl, CurationLabel) else str(lbl)
            for lbl in exclude_labels
        }
        all_units = (self.Unit & key).fetch("unit_id")
        if exclude_values:
            excluded = (
                self.UnitLabel
                & key
                & [{"curation_label": v} for v in exclude_values]
            ).fetch("unit_id")
            excluded_set = {int(u) for u in excluded}
            kept = [int(u) for u in all_units if int(u) not in excluded_set]
        else:
            kept = [int(u) for u in all_units]
        return np.asarray(sorted(kept), dtype=int)

    def get_sort_group_info(self, key):
        """Return ALL electrodes in the sort group joined to BrainRegion.

        Fix for the v1 ``fetch(limit=1)`` multi-region under-reporting
        bug: this returns a DataJoint relation (not a DataFrame, not
        single-row) covering EVERY electrode in the sort group so a
        multi-region probe surfaces every represented region. Callers
        can chain restrictions / fetches on the returned relation.
        """
        from spyglass.common.common_ephys import Electrode as _Electrode
        from spyglass.common.common_region import BrainRegion
        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
            SortGroupV2,
        )

        source = SortingSelection.resolve_source(
            {"sorting_id": key["sorting_id"]}
        )
        if source.kind != "recording":
            raise NotImplementedError(
                "CurationV2.get_sort_group_info: concat-source sorts are "
                "not yet supported."
            )
        recording_key = source.key
        # ``RecordingSelection.fetch1("KEY")`` returns only the UUID PK;
        # the upstream nwb_file_name + sort_group_id are non-PK columns
        # that we have to fetch explicitly.
        nwb_file_name, sort_group_id = (
            RecordingSelection & recording_key
        ).fetch1("nwb_file_name", "sort_group_id")
        sg_restriction = {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
        }
        return (
            SortGroupV2.SortGroupElectrode & sg_restriction
        ) * _Electrode * BrainRegion

    def _upstream_recording_row(self, key) -> dict:
        """Fetch the upstream Recording row for a CurationV2 key.

        Used by ``get_sorting`` to recover the recording's
        sampling-frequency and first-timestamp metadata (matching the
        ``Sorting.get_sorting`` round-trip convention).
        """
        from spyglass.spikesorting.v2.recording import Recording

        source = SortingSelection.resolve_source(
            {"sorting_id": key["sorting_id"]}
        )
        return (Recording & source.key).fetch1()
