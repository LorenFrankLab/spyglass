"""Curation of sorted units.

Tables:
    CurationV2 (+ Unit, UnitLabel, MergeGroup, ParentMergeGroup): Manual;
    lineage via ``parent_curation_id``.

``curation_source`` values written by v2: ``manual``, ``figpack``, and
``curation_evaluation`` (a child accepted from a ``CurationEvaluation``).
``analyzer_curation`` is a valid enum value that no v2 workflow writes.
External NWB Units stay in ``ImportedSpikeSorting``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import datajoint as dj

from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._curation import (
    insert as _curation_insert,
    readers as _curation_readers,
    restriction as _curation_restriction,
)
from spyglass.spikesorting.v2._curation.transforms import (
    ManualMergeAction,
    build_merge_provenance_rows,
    group_contributor_rows,
    is_merge_preview,
    key_manual_labels_to_merged_units,
    manual_curation_applies_merges,
    normalize_manual_curation,
    validate_curation_label_rows,
)
from spyglass.spikesorting.v2._storage.units_nwb import write_curated_units_nwb
from spyglass.spikesorting.v2._storage.staged_outputs import (
    unlink_staged_analysis_file as _unlink_staged_analysis_file,
)
from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
from spyglass.spikesorting.v2._core.enums import CurationLabel, CurationSource
from spyglass.spikesorting.v2._core.table_integrity import FactoryOnlyMaster
from spyglass.spikesorting.v2._recording.unit_metadata import (
    unit_brain_region_df,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_curation")

#: Retries when a concurrent insert claims the allocated ``curation_id``
#: (allocated outside the transaction).
_CURATION_ID_RACE_RETRIES = 5

CONCAT_MERGE_GATE_MESSAGE = (
    "CurationV2 for concat-backed sorting {sorting_id} was NOT registered in "
    "SpikeSortingOutput: its spike times are on the concatenated recording's "
    "synthetic 0-based timeline (member wall-clock gaps are dropped), so a "
    "downstream consumer keyed by nwb_file_name would misalign them. Populate "
    "ConcatMemberCuration for one wall-clock-aligned SpikeSortingOutput row "
    "per frozen member session."
)


@schema
class CurationV2(FactoryOnlyMaster, SpyglassMixin, dj.Manual):
    """Manual curation labels + merge groups for a sorted Sorting.

    - Several curations per sort, keyed by ``curation_id``.
    - ``parent_curation_id`` records lineage. It is not a foreign key because
      DataJoint cannot declare a nullable self-FK; :meth:`delete` guards it.
    - ``object_id`` is the column name ``SpikeSortingOutput`` reads in
      ``get_spike_times()``.
    - ``curation_uuid`` identifies one row generation; ``curation_id`` can be
      reused after deletion.
    - Write only through :meth:`insert_curation`; direct ``insert`` and
      ``update1`` are blocked.
    """

    #: Named in ``FactoryOnlyMaster``'s reject messages.
    _factory_create_call = "CurationV2.insert_curation()"

    definition = """
    -> Sorting
    curation_id: int
    ---
    curation_uuid: uuid
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)
    merges_applied=0: bool
    curation_source = 'manual': enum('manual', 'analyzer_curation', 'figpack', 'curation_evaluation')
    description: varchar(255)
    created_at=CURRENT_TIMESTAMP: timestamp
    created_by='': varchar(128)
    UNIQUE INDEX (curation_uuid)
    """

    class Unit(SpyglassMixinPart):
        """Per-curated-unit peak channel.

        A merged unit inherits the peak channel of its highest-amplitude
        contributor.
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

        - A unit may carry several labels; an unlabeled unit has no rows.
        - ``curation_label`` is a ``varchar`` validated against
          ``CurationLabel`` on every insert path, so labs can add labels
          without altering the column. Pass ``allow_custom_labels=True`` to
          insert a label outside the canonical set.
        """

        definition = """
        -> CurationV2.Unit
        curation_label: varchar(32)
        """

        def insert(self, rows, *, allow_custom_labels: bool = False, **kwargs):
            """Validate ``curation_label`` values, then insert the rows."""
            rows = list(rows)
            validate_curation_label_rows(
                rows, allow_custom_labels=allow_custom_labels
            )
            super().insert(rows, **kwargs)

    class MergeGroup(SpyglassMixinPart):
        """Per-merge-group provenance: kept unit <- contributor units.

        - One row per ``(kept unit_id, contributor_unit_id)``.
        - An unmerged unit is its own contributor, so every unit has a row.
        - ``contributor_unit_id`` is a foreign key to ``Sorting.Unit`` that
          shares ``sorting_id``, so a contributor must be a unit of this sort.
        - A kept ``unit_id`` may be a fresh merged id absent from
          ``Sorting.Unit``; contributors are always raw units.
        """

        definition = """
        -> CurationV2.Unit
        -> Sorting.Unit.proj(contributor_unit_id='unit_id')
        ---
        """

    class ParentMergeGroup(SpyglassMixinPart):
        """Per-merge-group provenance in the IMMEDIATE PARENT's namespace.

        Lists the parent ``CurationV2.Unit`` ids merged into each child
        unit. ``MergeGroup`` lists raw ``Sorting.Unit`` contributors instead.

        - Only child curations have rows; a root curation has none.
        - A pass-through child unit has a self-entry, so every child unit
          has a row (``get_unit_contributor_groups`` relies on this).
        - ``parent_unit_id`` is a plain int because a merged parent id is
          absent from ``Sorting.Unit``; :meth:`insert` checks it instead.
        """

        definition = """
        -> CurationV2.Unit
        parent_unit_id: int
        ---
        """

        def insert(self, rows, **kwargs):
            """Insert parent-namespace merge rows, enforcing provenance.

            Raises if a row belongs to a root curation (row presence marks a
            child) or names a ``parent_unit_id`` absent from the immediate
            parent curation.
            """
            rows = [dict(row) for row in rows]
            if rows:
                self._validate_parent_provenance(rows)
            super().insert(rows, **kwargs)

        @staticmethod
        def _validate_parent_provenance(rows):
            """Raise if any row violates the parent-namespace invariants."""
            from collections import defaultdict

            parents_by_curation: dict[tuple, set] = defaultdict(set)
            for row in rows:
                parents_by_curation[
                    (row["sorting_id"], int(row["curation_id"]))
                ].add(int(row["parent_unit_id"]))
            for (
                sorting_id,
                curation_id,
            ), parent_uids in parents_by_curation.items():
                curation_key = {
                    "sorting_id": sorting_id,
                    "curation_id": curation_id,
                }
                parent_curation_ids = (CurationV2 & curation_key).fetch(
                    "parent_curation_id"
                )
                if len(parent_curation_ids) != 1:
                    raise ValueError(
                        "ParentMergeGroup.insert: no CurationV2 row for "
                        f"{curation_key}; insert the curation row first."
                    )
                parent_curation_id = int(parent_curation_ids[0])
                if parent_curation_id == -1:
                    raise ValueError(
                        f"ParentMergeGroup.insert: curation {curation_key} is "
                        "a ROOT (parent_curation_id=-1); a root composes from "
                        "the raw sort and must have no ParentMergeGroup rows "
                        "(their presence is the child discriminator in "
                        "get_unit_contributor_groups)."
                    )
                parent_unit_set = {
                    int(u)
                    for u in (
                        CurationV2.Unit
                        & {
                            "sorting_id": sorting_id,
                            "curation_id": parent_curation_id,
                        }
                    ).fetch("unit_id")
                }
                stray = sorted(parent_uids - parent_unit_set)
                if stray:
                    raise ValueError(
                        f"ParentMergeGroup.insert: parent_unit_id(s) {stray} "
                        "are not units in the parent curation "
                        f"(sorting_id={sorting_id}, "
                        f"curation_id={parent_curation_id}); a row must "
                        "reference a unit in the immediate parent's namespace."
                    )

    def delete(self, *args, safemode=None, **kwargs):
        """Delete curations, refusing to orphan a child's lineage pointer.

        Refuses when:

        - a descendant curation (``parent_curation_id`` equal to a deleted
          ``curation_id``) is not also being deleted. Delete leaf-up.
        - another table uses the curation's ``SpikeSortingOutput`` entry
          (``SortedSpikesGroup``, a unit annotation, waveform features).

        ``super_delete``, ``delete_quick``, and upstream cascades skip this
        guard; :meth:`audit_orphaned_lineage` finds the orphans they leave.
        Accepts a leading positional restriction.
        """
        from spyglass.spikesorting.v2._core.table_integrity import (
            restrict_by_leading_args,
        )

        restricted, args = restrict_by_leading_args(self, args)
        if restricted is not None:
            return restricted.delete(*args, safemode=safemode, **kwargs)

        cls = type(self)
        pk = self.primary_key
        rows = self.fetch("KEY")
        # Snapshot concat members: the cascade cannot remove their upstream
        # AnalysisNwbfile, so orphans are reclaimed after the delete.
        from spyglass.spikesorting.v2.concat_member_curation import (
            ConcatMemberCuration,
        )

        concat_member_rows = (
            (ConcatMemberCuration & list(rows)).fetch(as_dict=True)
            if len(rows)
            else []
        )
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        merge_ids = list(
            (SpikeSortingOutput.CurationV2 & list(rows)).fetch("merge_id")
        ) + ConcatMemberCuration._merge_ids(concat_member_rows)
        in_delete_set = {tuple(sorted(row.items())) for row in rows}
        sorting_fields = [k for k in pk if k != "curation_id"]
        children = (
            cls
            & [
                {
                    **{k: row[k] for k in sorting_fields},
                    "parent_curation_id": row["curation_id"],
                }
                for row in rows
            ]
        ).fetch(
            *pk,
            "parent_curation_id",
            as_dict=True,
            order_by=(*sorting_fields, "parent_curation_id", "curation_id"),
        )
        orphaned: list[tuple[int, int]] = [
            (int(child["parent_curation_id"]), int(child["curation_id"]))
            for child in children
            if tuple(sorted((k, child[k]) for k in pk)) not in in_delete_set
        ]
        if orphaned:
            raise ValueError(
                f"{cls.__name__}.delete: refusing to delete curation(s) that "
                "still have descendant curations -- removing a parent would "
                "orphan the child's parent_curation_id lineage. Offending "
                f"(parent_curation_id, child_curation_id) pairs: {orphaned}. "
                "Delete the descendant curations first (leaf-up)."
            )
        if safemode is not None:
            kwargs["safemode"] = safemode
        return ConcatMemberCuration._delete_and_reclaim(
            super().delete,
            args,
            kwargs,
            member_rows=concat_member_rows,
            merge_ids=merge_ids,
            context=f"{cls.__name__}.delete",
        )

    @classmethod
    def audit_orphaned_lineage(cls, *, sorting_id=None) -> list[dict]:
        """Return child curations whose ``parent_curation_id`` names a missing parent.

        Finds orphans left by deletes that bypass :meth:`delete`.

        Parameters
        ----------
        sorting_id : optional
            Audit only this sorting. Default audits all.

        Returns
        -------
        list of dict
            One ``{sorting_id, curation_id, parent_curation_id}`` per orphaned
            child (its ``parent_curation_id`` has no matching curation).
        """
        curations = cls & (
            {} if sorting_id is None else {"sorting_id": sorting_id}
        )
        children = curations & "parent_curation_id != -1"
        existing_parents = curations.proj(parent_curation_id="curation_id")
        orphaned = children - existing_parents
        return orphaned.fetch(
            "sorting_id", "curation_id", "parent_curation_id", as_dict=True
        )

    @classmethod
    def insert_curation(
        cls,
        sorting_key: dict,
        labels: dict | None = None,
        parent_curation_id: int = -1,
        merge_groups: list[list[int]] | None = None,
        apply_merge: bool = False,
        description: str = "",
        curation_source: str | CurationSource = "manual",
        reuse_existing: bool = False,
        allow_unknown_unit_ids: bool = False,
        allow_custom_labels: bool = False,
        label_policy: str = "inherit",
    ) -> dict:
        """Insert master + Unit + UnitLabel rows; stage curated-units NWB.

        Atomically inserts the CurationV2 master, ``Unit`` and ``UnitLabel``
        part rows and, for a single-recording sort, the
        ``SpikeSortingOutput.CurationV2`` merge-table registration in one
        transaction. A concat-backed curation is not registered because its
        synthetic timeline cannot safely serve a session-scoped consumer. The
        curated-units NWB is staged separately and deleted on any later failure
        (DataJoint cannot roll back filesystem side effects).

        Parameters
        ----------
        sorting_key
            ``{sorting_id}`` of the upstream Sorting row.
        labels
            Dict ``unit_id -> [label, ...]`` with integer unit IDs. Use
            ``FigPackCuration.save_curation_from_uri`` to import JSON annotations.
            Each label is validated against the ``CurationLabel`` enum.
            ``None`` (the default) and ``{}`` are equivalent and produce a
            curation with no ``UnitLabel`` rows.
        parent_curation_id
            ``-1`` for a root curation; otherwise must reference an
            existing CurationV2 row for the same sorting.
        merge_groups
            Optional list of merge groups, each a list of at least 2
            ``unit_id`` ints. The merged unit inherits the peak channel and
            amplitude of the highest-amplitude contributor.

            - ``apply_merge=True``: merged ids start at
              ``max(source unit_ids) + 1``, assigned in ascending
              min-contributor order. ``get_merged_sorting`` on a preview
              assigns the same ids.
            - ``apply_merge=False``: every unit keeps its id; the merge is
              recorded in ``MergeGroup`` with ``min(group)`` as the kept unit.
        apply_merge
            If True, the NWB and ``CurationV2.Unit`` store the merged units
            (spike trains unioned, contributors absorbed). If False (default),
            units, trains, and labels pass through 1:1 and merges are applied
            lazily by ``get_merged_sorting()``.
        description
            Free-text curation description.
        curation_source
            One of 'manual' (default), 'figpack', 'curation_evaluation', or
            'analyzer_curation'.
        reuse_existing : bool, optional
            Root: if a root exists and non-default arguments were passed,
            False (default) raises ``ValueError`` and True returns the
            existing root. Child: True returns an existing child whose
            parent, labels, merge groups, ``apply_merge``, description, and
            source all match.
        allow_unknown_unit_ids
            For label keys absent from both ``Sorting.Unit`` and the curated
            units: False (default) raises ``ValueError``, True warns and drops
            them. Labels on contributors absorbed by ``apply_merge=True`` are
            always dropped with a warning.
        allow_custom_labels
            If False (default), a label outside ``CurationLabel`` raises
            ``ValueError``. True accepts custom labels.
        label_policy
            How a child curation combines its labels with the parent's.
            ``"inherit"`` (default) overlays ``labels`` per unit on the
            parent's labels; a committed merge gets the union of its
            contributors' labels. ``"replace"`` uses only ``labels``. For a
            root curation both mean ``labels``.

        Returns
        -------
        dict
            ``{"sorting_id": ..., "curation_id": ...}`` PK-only dict.

        Raises
        ------
        ValueError
            If a label value is a scalar/non-list (it must be a list or
            tuple of labels); if ``curation_source`` is not a valid
            ``CurationSource`` value; if ``parent_curation_id`` does not
            reference an existing curation for the sorting; if a root
            curation already exists and non-default parameters were passed
            without ``reuse_existing=True``; or if ``labels`` reference
            truly-stray unit_id(s) and ``allow_unknown_unit_ids`` is False.
        """
        sorting_id = sorting_key["sorting_id"]
        # This order sets the NWB unit write order; DataJoint has no default.
        sorting_units = (Sorting.Unit & {"sorting_id": sorting_id}).fetch(
            as_dict=True, order_by="unit_id"
        )
        if not sorting_units and not (Sorting & {"sorting_id": sorting_id}):
            raise ValueError(
                f"CurationV2.insert_curation: sorting_id {sorting_id} "
                "not in Sorting. Populate Sorting first."
            )

        labels, curation_source = _curation_insert.normalize_curation_inputs(
            labels, curation_source, allow_custom_labels
        )

        existing_root = _curation_insert.validate_parent_or_reuse_root(
            cls,
            sorting_id=sorting_id,
            parent_curation_id=parent_curation_id,
            labels=labels,
            merge_groups=merge_groups,
            description=description,
            apply_merge=apply_merge,
            curation_source=curation_source,
            reuse_existing=reuse_existing,
        )
        if existing_root is not None:
            return existing_root

        # A root composes from the raw sort, a child from its parent's units.
        (
            source_units,
            source_units_abs_path,
            parent_raw_contributors,
            parent_labels,
        ) = _curation_insert.resolve_curation_source(
            cls,
            sorting_id=sorting_id,
            parent_curation_id=parent_curation_id,
            raw_sorting_units=sorting_units,
        )

        from spyglass.spikesorting.v2._curation.plan import (
            build_curation_header,
            build_curation_insert_plan,
        )

        curation_id, unit_rows, kept_unit_to_contributors, labels = (
            build_curation_insert_plan(
                sorting_id=sorting_id,
                sorting_units=source_units,
                merge_groups=merge_groups,
                apply_merge=apply_merge,
                labels=labels,
                allow_unknown_unit_ids=allow_unknown_unit_ids,
                curation_id=cls._next_curation_id(sorting_id),
                parent_labels=parent_labels,
                label_policy=label_policy,
            )
        )

        def _find_matching_child():
            # Reads ``unit_rows`` at call time: a curation_id retry re-stamps it.
            return _curation_insert.find_matching_child_curation(
                cls,
                sorting_id=sorting_id,
                parent_curation_id=parent_curation_id,
                labels=labels,
                unit_rows=unit_rows,
                kept_unit_to_contributors=kept_unit_to_contributors,
                apply_merge=apply_merge,
                description=description,
                curation_source=curation_source,
            )

        if parent_curation_id != -1 and reuse_existing:
            existing_child = _find_matching_child()
            if existing_child is not None:
                logger.warning(
                    "CurationV2.insert_curation: matching child curation "
                    f"already exists for sorting_id={sorting_id}, "
                    f"parent_curation_id={parent_curation_id}; returning "
                    "existing key without staging a new NWB."
                )
                return existing_child

        # Concurrent children can allocate the same curation_id; retry on
        # DuplicateError with a fresh id.
        for attempt in range(_CURATION_ID_RACE_RETRIES + 1):
            # Random, not content-derived, so a recreated curation_id never
            # makes stale caches look current.
            curation_uuid = uuid.uuid4()
            # Shared by the NWB lineage and the DB insert so both agree.
            merge_group_rows, parent_merge_group_rows = (
                build_merge_provenance_rows(
                    sorting_id=sorting_id,
                    curation_id=curation_id,
                    kept_unit_to_contributors=kept_unit_to_contributors,
                    parent_raw_contributors=parent_raw_contributors,
                )
            )

            # Stage outside the transaction to keep it short.
            from spyglass.spikesorting.v2._core.runtime import (
                runtime_environment_provenance,
            )

            analysis_file_name, units_object_id, staged_parent_nwb = (
                cls._stage_curation_artifact(
                    sorting_id=sorting_id,
                    kept_unit_to_contributors=kept_unit_to_contributors,
                    apply_merge=apply_merge,
                    labels=labels,
                    unit_rows=unit_rows,
                    source_units_abs_path=source_units_abs_path,
                    merge_group_rows=merge_group_rows,
                    curation_header={
                        **build_curation_header(
                            sorting_id=sorting_id,
                            curation_id=curation_id,
                            curation_uuid=curation_uuid,
                            parent_curation_id=parent_curation_id,
                            curation_source=curation_source,
                            merges_applied=apply_merge,
                            description=description,
                        ),
                        **runtime_environment_provenance(
                            execution_params={"stage": "curation"}
                        ),
                    },
                )
            )

            try:
                cls._insert_curation_rows_transaction(
                    sorting_id=sorting_id,
                    curation_id=curation_id,
                    curation_uuid=curation_uuid,
                    parent_curation_id=parent_curation_id,
                    analysis_file_name=analysis_file_name,
                    units_object_id=units_object_id,
                    staged_parent_nwb=staged_parent_nwb,
                    apply_merge=apply_merge,
                    curation_source=curation_source,
                    description=description,
                    unit_rows=unit_rows,
                    labels=labels,
                    merge_group_rows=merge_group_rows,
                    parent_merge_group_rows=parent_merge_group_rows,
                    allow_custom_labels=allow_custom_labels,
                )
            except Exception as exc:
                # DB rows rolled back; remove the staged file.
                _unlink_staged_analysis_file(
                    analysis_file_name, context="CurationV2.insert_curation"
                )
                # Children only; root reuse is handled above.
                if (
                    parent_curation_id != -1
                    and isinstance(exc, dj.errors.DuplicateError)
                    and attempt < _CURATION_ID_RACE_RETRIES
                ):
                    # The concurrent insert may be this same child.
                    if reuse_existing:
                        existing_child = _find_matching_child()
                        if existing_child is not None:
                            logger.warning(
                                "CurationV2.insert_curation: a concurrent "
                                "insert created the matching child for "
                                f"sorting_id={sorting_id}, parent_curation_id="
                                f"{parent_curation_id}; returning it instead of "
                                "staging a duplicate."
                            )
                            return existing_child
                    new_id = cls._next_curation_id(sorting_id)
                    logger.warning(
                        "CurationV2.insert_curation: curation_id "
                        f"{curation_id} for sorting_id={sorting_id} was claimed "
                        "by a concurrent insert; retrying with curation_id "
                        f"{new_id} (attempt {attempt + 1})."
                    )
                    curation_id = new_id
                    # Other rows are rebuilt from curation_id next iteration.
                    unit_rows = [
                        {**row, "curation_id": curation_id} for row in unit_rows
                    ]
                    continue
                raise
            return {"sorting_id": sorting_id, "curation_id": curation_id}
        raise RuntimeError(  # pragma: no cover
            "CurationV2.insert_curation: exhausted curation_id retries without "
            "inserting or raising."
        )

    # ---- insert_curation steps ---------------------------------------------

    @staticmethod
    def _normalized_real_merge_groups(
        merge_groups,
    ) -> tuple[tuple[int, ...], ...]:
        """Normalize merge groups, dropping 1-unit self provenance entries."""
        if isinstance(merge_groups, dict):
            groups = merge_groups.values()
        else:
            groups = merge_groups or []
        return tuple(
            sorted(
                tuple(sorted(int(unit_id) for unit_id in group))
                for group in groups
                if len(group) >= 2
            )
        )

    @classmethod
    def _next_curation_id(cls, sorting_id) -> int:
        """Return the next auto-increment ``curation_id`` for a sorting.

        ``max(existing) + 1``, or 0. Not serialized against concurrent
        inserts; ``insert_curation`` retries on collision.
        """
        existing_ids = (cls & {"sorting_id": sorting_id}).fetch("curation_id")
        return int(max(existing_ids)) + 1 if len(existing_ids) else 0

    @classmethod
    def _stage_curation_artifact(
        cls,
        sorting_id,
        kept_unit_to_contributors: dict,
        apply_merge: bool,
        labels: dict,
        unit_rows: list[dict],
        source_units_abs_path: str | None = None,
        *,
        curation_header: dict,
        merge_group_rows: list[dict] | None = None,
    ) -> tuple[str, str, str]:
        """Stage the curated-units NWB and reconcile per-unit ``n_spikes``.

        Mutates ``unit_rows`` in place, setting ``n_spikes`` to the stored
        (post-dedup) train length. ``source_units_abs_path`` is ``None`` for a
        root (raw Sorting NWB) or the parent curation's NWB for a child. Run
        outside the DB transaction; the caller removes the file on failure.

        Returns
        -------
        tuple of str
            ``(analysis_file_name, units_object_id, staged_parent_nwb)``.
        """
        (
            analysis_file_name,
            units_object_id,
            staged_parent_nwb,
            n_spikes_by_uid,
        ) = write_curated_units_nwb(
            sorting_id=sorting_id,
            kept_unit_to_contributors=kept_unit_to_contributors,
            apply_merge=apply_merge,
            labels=labels,
            source_units_abs_path=source_units_abs_path,
            curation_header=curation_header,
            merge_group_rows=merge_group_rows,
        )
        # Merge dedup can shrink trains below the contributor-sum estimate.
        for row in unit_rows:
            if row["unit_id"] in n_spikes_by_uid:
                row["n_spikes"] = n_spikes_by_uid[row["unit_id"]]
        return analysis_file_name, units_object_id, staged_parent_nwb

    @classmethod
    def _insert_curation_rows_transaction(
        cls,
        sorting_id,
        curation_id: int,
        curation_uuid: uuid.UUID,
        parent_curation_id: int,
        analysis_file_name: str,
        units_object_id: str,
        staged_parent_nwb: str,
        apply_merge: bool,
        curation_source: str,
        description: str,
        unit_rows: list[dict],
        labels: dict,
        merge_group_rows: list[dict],
        parent_merge_group_rows: list[dict],
        allow_custom_labels: bool,
    ) -> None:
        """Insert the master/Unit/UnitLabel/MergeGroup rows atomically.

        Registers the staged AnalysisNwbfile, inserts the master and part
        rows, and, for a single-recording sort, the
        ``SpikeSortingOutput.CurationV2`` row. Concat-backed curations are
        not registered: their synthetic timeline is not session-safe.
        ``merge_group_rows`` and ``parent_merge_group_rows`` are inserted
        verbatim.
        """
        master_row = {
            "sorting_id": sorting_id,
            "curation_id": curation_id,
            "curation_uuid": curation_uuid,
            "parent_curation_id": parent_curation_id,
            "analysis_file_name": analysis_file_name,
            "object_id": units_object_id,
            # User intent: True even when merge_groups is empty.
            "merges_applied": bool(apply_merge),
            "curation_source": curation_source,
            "description": description,
            "created_by": str(dj.config["database.user"]),
        }
        # Labels on absorbed contributors are dropped.
        written_unit_ids = {row["unit_id"] for row in unit_rows}
        # Dedupe: a merge union can repeat a label.
        unit_label_rows = []
        seen_unit_labels: set[tuple[int, str]] = set()
        for unit_id, lbls in labels.items():
            if unit_id not in written_unit_ids:
                continue
            for label in lbls:
                normalized = CurationLabel.normalize(label)
                if (unit_id, normalized) in seen_unit_labels:
                    continue
                seen_unit_labels.add((unit_id, normalized))
                unit_label_rows.append(
                    {
                        "sorting_id": sorting_id,
                        "curation_id": curation_id,
                        "unit_id": unit_id,
                        "curation_label": normalized,
                    }
                )
        # Use ``_merge_insert``: it derives merge_id from the part-row hash.
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        register_merge = source.kind == "recording"

        with cls._safe_context():
            AnalysisNwbfile().add(staged_parent_nwb, analysis_file_name)
            # insert_curation already validated every row.
            cls.insert1(master_row, allow_direct_insert=True)
            cls.Unit.insert(unit_rows)
            if unit_label_rows:
                cls.UnitLabel.insert(
                    unit_label_rows,
                    allow_custom_labels=allow_custom_labels,
                )
            if merge_group_rows:
                cls.MergeGroup.insert(merge_group_rows)
            if parent_merge_group_rows:
                cls.ParentMergeGroup.insert(parent_merge_group_rows)
            if register_merge:
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
            else:
                logger.warning(
                    CONCAT_MERGE_GATE_MESSAGE.format(sorting_id=sorting_id)
                )

    # ---- Wrappers over insert_curation -----------------------------------

    @classmethod
    def create_initial_curation(
        cls,
        sorting_key: dict,
        labels: dict | None = None,
        description: str = "",
        *,
        allow_custom_labels: bool = False,
    ) -> dict:
        """Create the initial curation (no merges) over a sort.

        Calls :meth:`insert_curation` with ``parent_curation_id=-1`` and
        ``apply_merge=False``.

        Parameters
        ----------
        sorting_key
            ``{sorting_id}`` of the upstream Sorting row.
        labels
            Optional ``unit_id -> [label, ...]`` dict (validated by
            ``insert_curation``).
        description
            Free-text description.
        allow_custom_labels
            Forwarded to ``insert_curation`` to accept labels outside the
            canonical ``CurationLabel`` set.

        Returns
        -------
        dict
            ``{"sorting_id", "curation_id"}`` of the curation.

        See Also
        --------
        insert_curation : the full expert API this wraps.
        """
        return cls.insert_curation(
            sorting_key=sorting_key,
            labels=labels,
            parent_curation_id=-1,
            description=description,
            allow_custom_labels=allow_custom_labels,
        )

    @classmethod
    def save_manual_curation(
        cls,
        sorting_key: dict,
        *,
        parent_curation_id: int = -1,
        labels: dict | None = None,
        merge_groups: list[list[int]] | None = None,
        merge_action: ManualMergeAction = "preview",
        curation_source: str | CurationSource = "manual",
        description: str = "manual curation",
        reuse_existing: bool = False,
        allow_unknown_unit_ids: bool = False,
        allow_custom_labels: bool = False,
        label_policy: str = "inherit",
    ) -> dict:
        """Save native manual edits as the next curation.

        Pass ``labels={unit_id: [label, ...]}`` and
        ``merge_groups=[[unit_id, ...], ...]`` with integer unit IDs keyed in
        the parent's namespace. ``merge_action``:

        - ``"preview"`` stores merge groups as unapplied proposals.
        - ``"commit"`` applies non-empty merge groups; contributors' labels
          are unioned onto the merged unit.
        - With no merge groups, either value saves a label-edit child.

        Labels inherit from the parent by default; pass
        ``label_policy="replace"`` when ``labels`` is the full label state.
        Other arguments are forwarded to :meth:`insert_curation`.
        """
        labels, merge_groups = normalize_manual_curation(
            labels=labels, merge_groups=merge_groups
        )
        apply_merge = manual_curation_applies_merges(merge_action, merge_groups)
        if apply_merge:
            if parent_curation_id == -1:
                source, parent_labels = Sorting.Unit & sorting_key, {}
            else:
                parent_key = {
                    "sorting_id": sorting_key["sorting_id"],
                    "curation_id": parent_curation_id,
                }
                source = cls.Unit & parent_key
                parent_labels = cls._labels_by_unit(parent_key)
            labels = key_manual_labels_to_merged_units(
                labels,
                merge_groups,
                source_unit_ids=[int(u) for u in source.fetch("unit_id")],
                parent_labels=parent_labels,
                label_policy=label_policy,
            )

        _curation_insert.assert_child_reuse_has_parent(
            parent_curation_id=parent_curation_id,
            reuse_existing=reuse_existing,
            caller="save_manual_curation",
        )

        return cls.insert_curation(
            sorting_key=sorting_key,
            labels=labels or None,
            merge_groups=merge_groups or None,
            apply_merge=apply_merge,
            parent_curation_id=parent_curation_id,
            description=description,
            curation_source=curation_source,
            reuse_existing=reuse_existing,
            allow_unknown_unit_ids=allow_unknown_unit_ids,
            allow_custom_labels=allow_custom_labels,
            label_policy=label_policy,
        )

    @classmethod
    def label_options(cls) -> list[str]:
        """Return the canonical curation labels, in display order.

        These are the labels ``insert_curation`` accepts without
        ``allow_custom_labels=True``, in ``CurationLabel`` order.
        """
        return [label.value for label in CurationLabel]

    @classmethod
    def _labels_by_unit(cls, key) -> dict[int, list[str]]:
        """Group ``UnitLabel`` rows under ``key`` into ``{unit_id: [label]}``."""
        labels: dict[int, list[str]] = {}
        for lr in (cls.UnitLabel & key).fetch(
            "unit_id", "curation_label", as_dict=True
        ):
            labels.setdefault(int(lr["unit_id"]), []).append(
                str(lr["curation_label"])
            )
        return labels

    @classmethod
    def _unit_ids_with_labels(cls, key, values: set) -> set[int]:
        """``unit_id`` set under ``key`` carrying at least one of ``values``.

        ``values`` holds canonical label strings. Empty ``values`` returns
        an empty set.
        """
        if not values:
            return set()
        matched = (
            cls.UnitLabel & key & [{"curation_label": v} for v in values]
        ).fetch("unit_id")
        return {int(u) for u in matched}

    @classmethod
    def summarize_curation(
        cls,
        curation_key: dict,
        *,
        evaluation=None,
        annotation_sets=None,
    ) -> dict:
        """Return a notebook-printable summary of one curation.

        Parameters
        ----------
        curation_key
            Carries ``sorting_id`` and ``curation_id``; extra keys are
            ignored. From a ``run_v2_pipeline`` summary ``s``, use
            ``{"sorting_id": s["sorting_id"],
            "curation_id": s["root_curation_id"]}`` (or
            ``auto_labeled_curation_id``).
        evaluation : EvaluationResult, optional
            Evaluation whose metrics fill ``unit_properties``. Requires
            ``annotation_sets``.
        annotation_sets : sequence, optional
            Annotation sets to add to ``unit_properties``; ``[]`` for none.

        Returns
        -------
        dict
            - ``sorting_id``, ``curation_id``, ``n_units``
            - ``labels``: ``unit_id -> [label, ...]``
            - ``merge_groups``: groups with more than one contributor
            - ``unit_contributor_groups``: includes a self-entry per unit
            - ``merges_applied``
            - ``is_merge_preview``: not applied and has a real merge group
            - ``merge_id``: ``None`` if not in ``SpikeSortingOutput``
            - ``description``, ``created_at``, ``created_by``
            - ``unit_properties``: only when ``annotation_sets`` is passed

        See Also
        --------
        insert_curation : the expert write API.
        get_unit_contributor_groups : the merge-group provenance this summarizes.
        """
        # Validate before importing SpikeSortingOutput, which activates a schema.
        if (
            "sorting_id" not in curation_key
            or "curation_id" not in curation_key
        ):
            raise ValueError(
                "summarize_curation: key must contain 'sorting_id' and "
                "'curation_id' (curation_id is only unique within a sort). "
                f"Got keys: {sorted(curation_key)}."
            )

        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
        from spyglass.spikesorting.v2._curation.plan import (
            build_curation_summary,
        )

        pk = {
            "sorting_id": curation_key["sorting_id"],
            "curation_id": curation_key["curation_id"],
        }

        # fetch1 doubles as the "exactly one curation" check.
        merges_applied, description, created_at, created_by = (cls & pk).fetch1(
            "merges_applied", "description", "created_at", "created_by"
        )

        labels = cls._labels_by_unit(pk)

        # Concat-backed curations are not registered.
        merge_ids = (SpikeSortingOutput.CurationV2 & pk).fetch("merge_id")
        merge_id = merge_ids[0] if len(merge_ids) else None

        summary = build_curation_summary(
            sorting_id=pk["sorting_id"],
            curation_id=pk["curation_id"],
            merges_applied=merges_applied,
            description=description,
            labels=labels,
            merge_id=merge_id,
            unit_contributor_groups=cls.get_unit_contributor_groups(pk),
            n_units=len(cls.Unit & pk),
        )
        summary["created_at"] = created_at
        summary["created_by"] = str(created_by)
        if annotation_sets is not None:
            from spyglass.spikesorting.v2.curation_api import CurationRef
            from spyglass.spikesorting.v2.unit_annotation import (
                read_unit_properties,
            )

            summary["unit_properties"] = read_unit_properties(
                CurationRef.from_key(pk),
                evaluation=evaluation,
                annotation_sets=annotation_sets,
            )
        elif evaluation is not None:
            raise ValueError(
                "summarize_curation requires annotation_sets to be supplied "
                "explicitly whenever evaluation is selected; pass [] for no "
                "custom annotation sets."
            )
        return summary

    # ---- Accessors -------------------------------------------------------

    @classmethod
    def get_recording(cls, key: dict) -> "si.BaseRecording":
        """Return the sort's effective traces as persisted.

        What this returns depends on the sort; each case is an alias of one
        of the two accessors that have a single meaning:

        =================  =============  ======  =========  ================
        Sort               Alias of       Masked  Corrected  Clock
        =================  =============  ======  =========  ================
        single recording   source         no      no         acquisition
        single, corrected  sorting input  yes     yes        acquisition
        concatenation      sorting input  yes     no         synthetic concat
        concat, corrected  sorting input  yes     yes        synthetic concat
        =================  =============  ======  =========  ================

        "Source" is :meth:`get_source_recording` and "sorting input" is
        :meth:`get_sorting_input_recording`; call those for one meaning
        across sorts. "Masked" means silenced over the sort's artifact
        exclusions. A classmethod because ``SpikeSortingOutput`` dispatches
        to it on the part class.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row (dict or the
            merge dispatcher's list-of-dict form).

        Returns
        -------
        si.BaseRecording
            The cached preprocessed recording, annotated
            ``is_filtered=True``.
        """
        sorting_id = (cls & key).fetch1("sorting_id")
        traces = SortingSelection.resolve_effective_source(
            {"sorting_id": sorting_id}
        ).traces
        # A corrected recording was stored masked; its
        # apply_artifact_mask=False means "do not mask again".
        return SortingSelection.load_stored_traces(traces)

    @classmethod
    def get_source_recording(cls, key: dict) -> "si.BaseRecording":
        """Return the sort's original source recording.

        The preprocessed ``Recording`` on its acquisition clock, never
        artifact-masked or motion-corrected. For a concat-backed curation use
        ``ConcatMemberCuration.get_source_recording`` per member.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row (the merge
            dispatcher's list-of-dict form is accepted).

        Returns
        -------
        si.BaseRecording
            The source ``Recording``, annotated ``is_filtered=True``.

        Raises
        ------
        ValueError
            If the curation's sort reads a concatenated recording.
        """
        from spyglass.spikesorting.v2.recording import Recording

        sorting_id = (cls & key).fetch1("sorting_id")
        lineage = SortingSelection.resolve_effective_source(
            {"sorting_id": sorting_id}
        ).lineage
        if lineage.kind != "recording":
            raise ValueError(
                f"CurationV2.get_source_recording: sorting_id {sorting_id} "
                f"sorts concatenated recording "
                f"{lineage.key['concat_recording_id']}, whose original source "
                "is one recording per member. Call "
                "ConcatMemberCuration.get_source_recording with each member's "
                "key (sorting_id, curation_id, member_index)."
            )
        return Recording().get_recording(lineage.key)

    @classmethod
    def get_sorting_input_recording(cls, key: dict) -> "si.BaseRecording":
        """Return the traces the sorter read.

        Unwhitened, masked over the pinned artifact exclusions (member masks
        for a concat), and motion-corrected when the sort selected it. On the
        sort's clock: acquisition for a single recording, synthetic for a
        concat (see ``ConcatMemberCuration.get_sorting_input_recording``).
        Analyzer rebuilds and UnitMatch extraction start from this recording.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row (the merge
            dispatcher's list-of-dict form is accepted).

        Returns
        -------
        si.BaseRecording
            The sorting input, annotated ``is_filtered=True``.
        """
        from spyglass.spikesorting.v2._sorting.analyzer import (
            read_canonical_recording,
            resolve_canonical_recording,
        )

        sorting_id = (cls & key).fetch1("sorting_id")
        return read_canonical_recording(
            resolve_canonical_recording({"sorting_id": sorting_id})
        )

    @classmethod
    def get_sorting(
        cls, key: dict, as_dataframe: bool = False
    ) -> "si.BaseSorting | pd.DataFrame":
        """Return the curated SpikeInterface BaseSorting (or DataFrame).

        - The SI path reads stored ``spike_sample_index`` frames, not SI's
          affine ``NwbSortingExtractor``, so disjoint-interval sorts keep
          their original frames.
        - The DataFrame has one row per unit: absolute spike times in
          seconds and ``curation_label`` lists.
        - An ``apply_merge=False`` curation returns unmerged units and warns;
          use ``get_merged_sorting`` to apply proposed merges.
        - A zero-unit curation returns an empty sorting and warns.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        as_dataframe : bool, optional
            If ``True``, return a per-unit DataFrame instead of an SI
            sorting object. Defaults to ``False``.

        Returns
        -------
        si.BaseSorting or pd.DataFrame
            The curated sorting (a ``NumpySorting``) when
            ``as_dataframe`` is ``False``; otherwise a DataFrame indexed
            by ``unit_id`` with ``spike_times`` and ``curation_label``
            columns.
        """

        return _curation_readers.get_sorting(
            cls, key, as_dataframe=as_dataframe
        )

    @classmethod
    def has_unapplied_proposed_merges(cls, key, *, merges_applied=None) -> bool:
        """Return whether a curation has a proposed but unapplied merge.

        A real merge is a group with more than one contributor, so a curation
        with only self-entries returns ``False``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        merges_applied : bool, optional
            The row's ``merges_applied`` if already held; skips a fetch.
            ``None`` (default) fetches it.

        Returns
        -------
        bool
            ``True`` if the curation was not applied and has at least one
            >1-contributor merge group; ``False`` otherwise.
        """
        if merges_applied is None:
            merges_applied = (cls & key).fetch1("merges_applied")
        if bool(merges_applied):
            return False
        return is_merge_preview(
            merges_applied, cls.get_unit_contributor_groups(key)
        )

    @classmethod
    def is_committed_curation(cls, key, *, merges_applied=None) -> bool:
        """Return whether a curation is a committed (final) curation state.

        Committed: a root, a label-only child, or an applied-merge child.
        A preview (``apply_merge=False`` with a real proposed merge) is not.
        The negation of :meth:`has_unapplied_proposed_merges`.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        merges_applied : bool, optional
            The row's ``merges_applied`` if already held; skips a fetch.

        Returns
        -------
        bool
            ``True`` for root / label-only / applied-merge rows; ``False`` for
            preview rows.
        """
        return not cls.has_unapplied_proposed_merges(
            key, merges_applied=merges_applied
        )

    @classmethod
    def matches_raw_namespace(cls, key) -> bool:
        """Return whether a curation's unit set IS the raw sort's unit set.

        When ``True``, the raw-sort analyzer serves this curation.
        ``CurationEvaluation.make_fetch`` and the curation analyzer resolver
        both use this check. A label-only child of a merged parent is
        committed but returns ``False``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.

        Returns
        -------
        bool
            ``True`` iff the curation unit set equals the raw sort's unit set.
        """
        sorting_id = (cls & key).fetch1("sorting_id")
        curation_units = {int(u) for u in (cls.Unit & key).fetch("unit_id")}
        raw_units = {
            int(u)
            for u in (Sorting.Unit & {"sorting_id": sorting_id}).fetch(
                "unit_id"
            )
        }
        return curation_units == raw_units

    @classmethod
    def assert_committed_curation(
        cls, key, *, context: str = "", merges_applied=None
    ) -> None:
        """Raise if ``key`` is a preview curation; no-op for committed states.

        Scoring a preview would attach data to its unmerged units.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        context : str, optional
            Caller label woven into the error message (e.g.
            ``"CurationEvaluation"``).
        merges_applied : bool, optional
            The row's ``merges_applied`` value when already held by the caller.

        Raises
        ------
        ValueError
            If the curation is a preview (has unapplied proposed merges).
        """
        if cls.is_committed_curation(key, merges_applied=merges_applied):
            return
        where = f"{context}: " if context else ""
        raise ValueError(
            f"{where}curation {dict(key)} is a preview/draft curation "
            "(apply_merge=False with a proposed merge group that has not been "
            "applied), not a committed curation state. Evaluating it would "
            "score the UNMERGED preview units instead of the final merged unit "
            "set. Commit the merge first (commit_merges / "
            "insert_curation(apply_merge=True)), then evaluate that curation."
        )

    @classmethod
    def resolve_restriction(
        cls,
        key: dict,
        *,
        restrict_by_artifact: bool = True,
        strict: bool = True,
    ):
        """Resolve an interpretable restriction to the matching CurationV2 rows.

        Accepted keys:

        - recording: ``nwb_file_name``, ``team_name``, ``sort_group_id``,
          ``interval_list_name``, ``recording_id``
        - concat: ``concat_recording_id``, ``session_group_owner``,
          ``session_group_name``
        - either source: ``preprocessing_params_name``
        - shared: ``sorter``, ``sorter_params_name``, ``sorting_id``,
          ``artifact_detection_id``, ``motion_corrected_recording_id``,
          ``curation_id``

        With no source key, both sources match. Recording and concat keys
        cannot be mixed. For ``artifact_detection_id`` and
        ``motion_corrected_recording_id``, ``None`` matches only sorts
        without one; only an absent key is a wildcard. A detection id
        matches a concat if any member uses it.
        ``SpikeSortingOutput._get_restricted_merge_ids_v2`` delegates here.

        Parameters
        ----------
        key : dict
            Interpretable restriction over the v2 part-table convention
            keys (e.g. ``nwb_file_name``, ``sorting_id``, ``curation_id``,
            ``artifact_detection_id``).
        restrict_by_artifact : bool, optional
            If ``True`` (default), map an ``artifact_detection_{uuid}``
            ``interval_list_name`` back to ``artifact_detection_id`` so the
            join restricts by the artifact-removed valid_times row.
        strict : bool, optional
            If ``True`` (default), an unknown restriction key raises
            ``ValueError``. If ``False``, it returns ``None`` (for
            multi-pipeline dispatch).

        Returns
        -------
        datajoint.expression.QueryExpression or None
            A ``CurationV2`` query selecting the matching rows, or
            ``None`` in lenient mode (``strict=False``) when the key names
            no v2 column.

        Raises
        ------
        ValueError
            If ``strict`` is True and ``key`` contains restriction keys
            that are not v2 columns, or if ``key`` mixes recording and
            concat source keys.
        """

        return _curation_restriction.resolve_restriction(
            cls, key, restrict_by_artifact=restrict_by_artifact, strict=strict
        )

    @classmethod
    def get_sort_metadata(cls, key) -> tuple:
        """Return ``(sorter, nwb_file_name)`` for a curation's underlying sort.

        For a concat-backed sort, ``nwb_file_name`` is the first frozen
        member's session (the anchor of the sort's analysis NWB).

        Parameters
        ----------
        key : dict
            Restriction carrying ``sorting_id``.

        Returns
        -------
        tuple of (str, str)
            ``(sorter, nwb_file_name)`` for the underlying sort (anchor-member
            nwb for concat sorts).
        """
        from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

        sorting_id = key["sorting_id"]
        sorter = (SortingSelection & {"sorting_id": sorting_id}).fetch1(
            "sorter"
        )
        nwb_file_name = Sorting.resolve_anchor_nwb_file_name(
            {"sorting_id": sorting_id}
        )
        return sorter, nwb_file_name

    @classmethod
    def get_unit_semantics(cls, key) -> str:
        """Return the unit semantics of a sort: sorted units vs threshold crossings.

        A clusterless sort's single "unit" is a threshold-crossing stream,
        not a neuron. Derived from the sort's ``sorter``.

        Parameters
        ----------
        key : dict
            Restriction carrying ``sorting_id``.

        Returns
        -------
        str
            ``"clusterless_threshold_crossings"`` or ``"sorted_units"``.
        """
        from spyglass.spikesorting.v2._sorting.dispatch import (
            unit_semantics_for_sorter,
        )

        sorter, _ = cls.get_sort_metadata(key)
        return unit_semantics_for_sorter(sorter)

    @classmethod
    def get_unit_contributor_groups(cls, key) -> dict[int, list[int]]:
        """Return kept-unit contributor groups in this curation's namespace.

        Reads ``MergeGroup`` for a root and ``ParentMergeGroup`` for a child.
        A child's ``MergeGroup`` holds inherited raw merges, not its own.
        Every unit has an entry; ``len(groups[X]) > 1`` means X leads a
        proposed or applied merge. For raw ``Sorting.Unit`` contributors use
        ``_raw_contributor_groups``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.

        Returns
        -------
        dict[int, list[int]]
            ``{kept_unit_id: [contributor_unit_id, ...]}`` with sorted
            contributor lists, in the curation's own unit namespace.
        """
        # Only a child has ParentMergeGroup rows.
        return cls._fetch_contributor_groups(
            cls.ParentMergeGroup & key, "parent_unit_id"
        ) or cls._raw_contributor_groups(key)

    @classmethod
    def _raw_contributor_groups(cls, key) -> dict[int, list[int]]:
        """Return each unit's raw ``MergeGroup`` contributors.

        ``{unit_id: [contributor_unit_id, ...]}`` in ascending order.
        """
        return cls._fetch_contributor_groups(
            cls.MergeGroup & key, "contributor_unit_id"
        )

    @staticmethod
    def _fetch_contributor_groups(
        relation, contributor_field: str
    ) -> dict[int, list[int]]:
        """Fetch merge-provenance rows grouped by kept unit, in id order.

        Order matters: SI's MergeUnitsSorting assigns merged ids by dict
        insertion order.
        """
        rows = relation.fetch(
            "unit_id",
            contributor_field,
            as_dict=True,
            order_by=("unit_id", contributor_field),
        )
        return group_contributor_rows(rows, contributor_field)

    @classmethod
    def get_merged_sorting(cls, key: dict) -> "si.BaseSorting":
        """Return the curated BaseSorting with merge groups applied.

        Applies a preview's proposed merges lazily, deduplicating in
        absolute spike time so wall-clock gaps are respected. Returns the
        base sorting unchanged when merges were already applied or no group
        has more than one contributor.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.

        Returns
        -------
        si.BaseSorting
            The curated sorting with proposed merge groups applied
            lazily (a ``NumpySorting``), or the unmodified base sorting
            when merges were already applied or no group has more than
            one contributor.
        """

        return _curation_readers.get_merged_sorting(cls, key)

    def get_unit_brain_regions(
        self,
        key: dict,
        *,
        include_labels: "Iterable | None" = None,
        allow_anchor_member: bool = False,
    ) -> "pd.DataFrame":
        """Per-unit brain regions via CurationV2.Unit * Electrode * BrainRegion.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        include_labels : iterable of str or CurationLabel, optional
            If a non-empty iterable, restrict to units carrying at least
            one of these labels. ``None`` (the default) and an empty
            iterable both mean no filter (all units).
        allow_anchor_member : bool, optional
            If ``True``, return anchor-member regions for concat-backed
            sortings instead of raising. Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            One row per (unit, electrode) with the brain-region columns
            and a ``region_resolution`` label.

        Raises
        ------
        ConcatBrainRegionAmbiguousError
            For a concat-backed sorting unless ``allow_anchor_member=True``.
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
                    "allow_anchor_member=True for anchor-only regions, or match "
                    "this curation with UnitMatch and use "
                    "TrackedUnit.get_unit_brain_regions, which resolves each "
                    "unit's region in every member recording from that "
                    "member's own session."
                )
            resolution = "anchor_member"
        else:
            resolution = "single_session"

        unit_restriction = self.Unit & key
        if include_labels is not None:
            include_values = {
                CurationLabel.normalize(lbl) for lbl in include_labels
            }
            # ``& []`` would match nothing; empty means no filter.
            if include_values:
                labeled = self._unit_ids_with_labels(key, include_values)
                unit_restriction = unit_restriction & [
                    {"unit_id": uid} for uid in labeled
                ]
        return unit_brain_region_df(unit_restriction, resolution)

    @classmethod
    def get_sort_group_info(cls, key: dict) -> "dj.Table":
        """Return ALL electrodes in the sort group joined to BrainRegion.

        For a concat-backed sort, uses the first frozen member's sort group;
        later members' regions may differ. A classmethod because
        ``SpikeSortingOutput`` dispatches to it on the part class.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.

        Returns
        -------
        dj.Table
            A DataJoint relation (``SortGroupElectrode * Electrode *
            BrainRegion``) covering every electrode in the sort group (the
            anchor member's sort group for concat sorts).
        """
        from spyglass.spikesorting.v2._sorting import fetch as _sorting_fetch
        from spyglass.spikesorting.v2._recording.unit_metadata import (
            sort_group_electrode_regions,
        )

        sorting_id = (cls & key).fetch1("sorting_id")
        anchor = _sorting_fetch.resolve_sort_anchor(
            SortingSelection.resolve_source({"sorting_id": sorting_id})
        )
        return sort_group_electrode_regions(
            {
                "nwb_file_name": anchor.nwb_file_name,
                "sort_group_id": anchor.sort_group_id,
            }
        )
