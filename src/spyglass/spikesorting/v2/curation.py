"""Curation of sorted units.

Tables:
    CurationV2 (+ Unit + UnitLabel) -- Manual; lineage via parent_curation_id.

``curation_source`` is restricted to true CurationV2 provenance values
(``manual``, ``figpack``, ``curation_evaluation`` -- a child accepted from a
``CurationEvaluation`` output; ``analyzer_curation`` remains a valid enum
value, but no v2 workflow writes it). External or ground-truth NWB Units
continue to use ``ImportedSpikeSorting``; v2 does NOT duplicate them into ``CurationV2``.

``insert_curation`` writes each curation atomically across the master,
``Unit``, and ``UnitLabel`` parts and, for single-recording sorts, the
``SpikeSortingOutput.CurationV2`` merge-table row. Concat-backed curations are
not registered because their synthetic timeline is unsafe for session-scoped
consumers. The curated-units NWB is staged first; the AnalysisNwbfile DB-row
registration moves inside the transaction so a later failure rolls the row
back and the staged file is cleaned up in the except path.
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

    import numpy as np
    import pandas as pd
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_curation")

#: How many times ``insert_curation`` recomputes a child ``curation_id`` and
#: retries after a concurrent insert claimed the id it allocated (the id is
#: chosen outside the insert transaction, so two concurrent children of one
#: sorting can collide). Small: the contention window is brief and each retry
#: just re-reads ``max(curation_id) + 1``.
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

    Multiple curations per sort are allowed via ``curation_id``; each
    can record a ``parent_curation_id`` for lineage (validation-only --
    the schema does not self-FK because DataJoint cannot resolve a
    nullable self-FK across renamed columns cleanly).

    ``object_id`` (NOT ``units_object_id``) is the column name the
    ``SpikeSortingOutput`` merge-table router expects when dispatching
    ``get_spike_times()``; using the wrong name would break that routing.

    ``curation_uuid`` is the immutable identity of one row generation. The
    numeric ``curation_id`` remains part of the ergonomic DataJoint primary key
    but may be reused after deletion; every fresh insert therefore receives a
    random, database-unique UUID while an idempotent reuse keeps the stored one.

    A direct ``insert`` / ``insert1`` and an in-place ``update1`` are blocked
    (``FactoryOnlyMaster``): a curation is an identity/provenance root that
    ``SpikeSortingOutput`` and child curations reference, so it must be written
    through :meth:`insert_curation` (which builds the master, the analysis-file
    row, and the ``Unit`` / ``UnitLabel`` / ``MergeGroup`` parts atomically).
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

        Populated by ``insert_curation`` from the upstream
        ``Sorting.Unit`` rows after applying ``merge_groups``. A merged
        unit inherits the peak channel (and thus the Electrode /
        brain-region trace) of its highest-amplitude contributing unit.
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

        A unit may carry multiple labels (e.g. ``mua`` + ``artifact``);
        unlabeled units have zero ``UnitLabel`` rows. The NWB units table
        still gets a ``curation_label`` indexed column so consumers
        reading the NWB see empty lists for unlabeled units.

        ``curation_label`` is a ``varchar(32)`` validated against the
        canonical ``CurationLabel`` set at insert time on EVERY insert
        path -- including a direct ``UnitLabel.insert1`` / ``insert`` (the
        overrides below), not only via ``CurationV2.insert_curation``.
        DataJoint *can* declare an enum column (``curation_source`` on the
        master is one); v2 deliberately uses ``varchar`` + Python-side
        validation because the label set is open-ended -- a lab adding a
        custom label later extends the Python vocabulary without changing
        the column definition of a populated table. Pass
        ``allow_custom_labels=True`` on either path to insert a label
        outside the canonical set.
        """

        definition = """
        -> CurationV2.Unit
        curation_label: varchar(32)
        """

        def insert1(self, row, *, allow_custom_labels: bool = False, **kwargs):
            """Validate and insert a single ``UnitLabel`` row."""
            # Delegate to ``insert`` (as DataJoint's own ``insert1`` does:
            # ``self.insert((row,))``) so the single validation happens in
            # one place AND ``allow_custom_labels`` survives the dispatch.
            # Validating here then calling ``super().insert1`` would lose
            # the flag, because DataJoint's internal ``insert1 -> insert``
            # hop re-enters this override with the default False.
            self.insert(
                [row], allow_custom_labels=allow_custom_labels, **kwargs
            )

        def insert(self, rows, *, allow_custom_labels: bool = False, **kwargs):
            """Validate ``curation_label`` values, then insert the rows."""
            rows = list(rows)
            validate_curation_label_rows(
                rows, allow_custom_labels=allow_custom_labels
            )
            super().insert(rows, **kwargs)

    class MergeGroup(SpyglassMixinPart):
        """Per-merge-group provenance: kept unit <- contributor units.

        Merge groups are stored here as queryable part rows (one row
        per kept-unit/contributor pair) so bulk-audit queries ("show me
        every (kept_unit, contributor) pair across sortings") and
        provenance retrieval ("for this paper, list every merge
        decision") are easy.

        ``contributor_unit_id`` is a DataJoint FK to ``Sorting.Unit``
        (declared ``-> Sorting.Unit.proj(contributor_unit_id='unit_id')``).
        The part already inherits ``sorting_id`` from ``-> CurationV2.Unit``,
        so DataJoint UNIFIES the two ``sorting_id`` references and the FK
        enforces BOTH that the contributor is a real unit AND that it
        belongs to THIS sort -- including on a direct ``MergeGroup.insert``
        that bypasses ``insert_curation`` (a bogus contributor raises
        ``IntegrityError`` instead of silently corrupting provenance).
        ``insert_curation`` keeps its Python-side contributor check too, for
        a friendlier error than the raw FK violation. The kept ``unit_id``
        may be a fresh ``max+1`` merge id (apply_merge=True) that is NOT in
        ``Sorting.Unit``; that id only ever appears as the kept ``unit_id``
        (FK'd to ``CurationV2.Unit``), never as a ``contributor_unit_id`` --
        contributors are always original source units, so the FK is
        satisfiable in every mode.

        One row per ``(kept_unit_id, contributor_unit_id)``. The
        kept unit appears as its own contributor for unmerged
        units (one row, ``contributor_unit_id == unit_id``); this
        makes "list every unit's merge provenance" a single
        restriction without special-casing the no-merge units.
        """

        definition = """
        -> CurationV2.Unit
        -> Sorting.Unit.proj(contributor_unit_id='unit_id')
        ---
        """

    class ParentMergeGroup(SpyglassMixinPart):
        """Per-merge-group provenance in the IMMEDIATE PARENT's namespace.

        Records the parent-curation operation that produced each child unit:
        which PARENT ``CurationV2.Unit`` ids were composed into this child
        unit. Distinct from ``MergeGroup`` (raw ``Sorting.Unit`` contributors)
        so neither structure has to mean both -- ``MergeGroup`` answers "which
        original raw units contributed?" and ``ParentMergeGroup`` answers
        "which parent units did this child operation merge?".

        ``parent_unit_id`` is a plain int, NOT a DataJoint FK: a child of a
        merged parent may reference a fresh ``max+1`` merged id that exists
        only in the parent ``CurationV2.Unit`` set and not in ``Sorting.Unit``,
        so it cannot be FK'd to the raw sort. Because the FK cannot do it,
        membership in the parent's unit set is enforced on EVERY insert path by
        the ``insert`` override below (not only by construction): a root
        curation gets no rows and every ``parent_unit_id`` must be a unit in the
        immediate parent. ``insert_curation`` builds these rows from
        ``kept_unit_to_contributors``, whose contributors come from the parent
        ``Unit`` rows themselves (``build_curated_unit_rows`` rejects any
        merge-group member not in that set). Only child curations
        (``parent_curation_id != -1``)
        get these rows; a root curation composes from the raw sort and has
        none. A 1-element self-entry is recorded per pass-through child unit so
        every child ``Unit`` row has at least one ``ParentMergeGroup`` row
        keyed by its own ``unit_id`` -- the own-namespace merge view
        (``get_unit_contributor_groups``) reads this for a child.
        """

        definition = """
        -> CurationV2.Unit
        parent_unit_id: int
        ---
        """

        def insert1(self, row, **kwargs):
            """Validate, then insert a single ``ParentMergeGroup`` row."""
            self.insert([row], **kwargs)

        def insert(self, rows, **kwargs):
            """Insert parent-namespace merge rows, enforcing provenance.

            ``parent_unit_id`` is deliberately not a DataJoint FK (a merged
            parent id is absent from ``Sorting.Unit``), so the two invariants
            that ``get_unit_contributor_groups`` relies on are enforced here -- on every
            insert path, including a direct insert that bypasses
            ``insert_curation``: a ROOT curation gets NO ``ParentMergeGroup``
            rows (their mere presence is the child discriminator), and every
            ``parent_unit_id`` must be a unit in the IMMEDIATE PARENT curation.
            Without this a forged row could misclassify a committed curation as
            a preview or break merged-sorting reconstruction.
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

        LINEAGE BOUNDARY. A child curation records its parent via
        ``parent_curation_id`` -- a validation-only int. DataJoint cannot express
        the nullable self-referential foreign key that would enforce it (a true
        self-FK is not declarable; an equivalent two-FK edge table breaks
        DataJoint's delete cascade), so lineage integrity is a bounded
        APPLICATION-level invariant, guaranteed only through the supported
        workflows: ``insert_curation`` (the sole writer) and this
        ``CurationV2.delete``, which REFUSES to remove a row whose descendant
        (same sorting, ``parent_curation_id`` == its ``curation_id``) is not
        itself in the delete set. Deleting a parent leaf-up, or deleting the whole
        ``Sorting`` lineage, is always safe.

        ADMINISTRATIVE BYPASSES that may violate lineage (leaving a child pointing
        at a missing parent): the raw ``super_delete`` / ``delete_quick`` paths,
        and a targeted upstream cascade (e.g. deleting a parent curation's
        ``AnalysisNwbfile``, which cascades to the curation through DataJoint's
        generic cascade without invoking this guard). These are intentionally
        unaffected; :meth:`audit_orphaned_lineage` detects any orphan they leave.

        A leading positional restriction is accepted for the
        ``Table().delete(restriction)`` form.
        """
        from spyglass.spikesorting.v2._core.table_integrity import (
            split_leading_restrictions,
        )

        restriction_args, args = split_leading_restrictions(args)
        if restriction_args:
            target = self
            for restriction in restriction_args:
                target = target & restriction
            return target.delete(*args, safemode=safemode, **kwargs)

        cls = type(self)
        pk = self.primary_key
        rows = self.fetch("KEY")
        # A concat member's AnalysisNwbfile is an UPSTREAM registry row, so the
        # normal DataJoint cascade can remove the member and merge-part rows but
        # cannot remove that registry row or external file. Snapshot those
        # dependents before the cascade; after a successful (non-cancelled)
        # delete the owning table reclaims only registry rows that became true
        # orphans. The local import avoids the module's intentional CurationV2
        # foreign-key import cycle at declaration time.
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
        ) + list(
            (
                SpikeSortingOutput.ConcatMemberCuration
                & [
                    {
                        name: row[name]
                        for name in ConcatMemberCuration.primary_key
                    }
                    for row in concat_member_rows
                ]
            ).fetch("merge_id")
        )
        in_delete_set = {tuple(sorted(row.items())) for row in rows}
        orphaned: list[tuple[int, int]] = []
        for row in rows:
            sorting_key = {k: row[k] for k in pk if k != "curation_id"}
            descendants = (
                cls & {**sorting_key, "parent_curation_id": row["curation_id"]}
            ).fetch("KEY")
            for child in descendants:
                if tuple(sorted(child.items())) not in in_delete_set:
                    orphaned.append(
                        (int(row["curation_id"]), int(child["curation_id"]))
                    )
        if orphaned:
            raise ValueError(
                f"{cls.__name__}.delete: refusing to delete curation(s) that "
                "still have descendant curations -- removing a parent would "
                "orphan the child's parent_curation_id lineage. Offending "
                f"(parent_curation_id, child_curation_id) pairs: {orphaned}. "
                "Delete the descendant curations first (leaf-up)."
            )
        dry_run = bool(kwargs.get("dry_run", False))
        if dry_run:
            ConcatMemberCuration._delete_inventory(
                concat_member_rows, context=f"{cls.__name__}.delete"
            )
        # ``force_masters=True`` is normally how cautious deletion removes a
        # merge master through its source part. CurationV2 has nested parts,
        # though, and DataJoint 0.14 can then revisit/delete CurationV2 through
        # a grandchild before the outer cascade reaches it. The outer delete
        # reports zero rows and rolls the whole transaction back. Opt out for
        # this cascade and remove only proven-orphan merge masters afterwards.
        kwargs["force_masters"] = False
        kwargs["force_parts"] = True
        if safemode is None:
            result = super().delete(*args, **kwargs)
        else:
            result = super().delete(*args, safemode=safemode, **kwargs)
        if not dry_run:
            ConcatMemberCuration._cleanup_orphaned_merge_masters(merge_ids)
            ConcatMemberCuration._cleanup_deleted_analysis_rows(
                concat_member_rows
            )
        return result

    @classmethod
    def audit_orphaned_lineage(cls, *, sorting_id=None) -> list[dict]:
        """Return child curations whose ``parent_curation_id`` names a missing parent.

        The lineage-integrity audit for the boundary documented on :meth:`delete`:
        because ``parent_curation_id`` is a validation-only int the database
        cannot enforce, an administrative bypass (``super_delete`` /
        ``delete_quick``, or a targeted upstream cascade such as deleting a parent
        curation's ``AnalysisNwbfile``) can remove a parent while leaving a child
        pointing at it. This flags exactly those orphans: non-root rows
        (``parent_curation_id != -1``) whose ``(sorting_id, parent_curation_id)``
        is not itself a ``CurationV2`` row. An empty list means every recorded
        lineage edge resolves -- run it as a strict/maintenance gate.
        Pass ``sorting_id`` to audit only that sorting's lineage.

        Returns
        -------
        list of dict
            One ``{sorting_id, curation_id, parent_curation_id}`` per orphaned
            child (its ``parent_curation_id`` has no matching curation).
        """
        # Antijoin: children whose (sorting_id, parent_curation_id) has no
        # matching (sorting_id, curation_id) among existing curations.
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
    def audit_concat_merge_rows(cls) -> list[dict]:
        """Return merge rows whose CurationV2 sorting is concat-backed."""
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        concat_sortings = SortingSelection.ConcatenatedRecordingSource.fetch(
            "sorting_id"
        )
        if not len(concat_sortings):
            return []
        return (
            SpikeSortingOutput.CurationV2
            & [{"sorting_id": sorting_id} for sorting_id in concat_sortings]
        ).fetch("KEY")

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
            Optional list of merge groups, each a list of ``unit_id``
            ints. Each group must have at least 2 members (a single-unit
            group is rejected as a likely typo). The merged unit inherits
            the peak channel + amplitude of the highest-amplitude
            contributor. For ``apply_merge=True`` each merged unit gets a
            fresh id from ``max(source unit_ids) + 1`` upward, assigned in
            ascending min-contributor order regardless of the order the
            groups are listed; ``get_merged_sorting`` on a preview numbers
            merges the same way, so both paths give a group the same id.
            For ``apply_merge=False`` (preview) every original unit --
            contributors included -- keeps its own id in
            ``CurationV2.Unit``; the proposed merge is recorded in
            ``CurationV2.MergeGroup`` with ``min(group)`` as the
            kept-unit leader. Non-listed units pass through 1:1.
        apply_merge
            If True, both the curated-units NWB and ``CurationV2.Unit``
            store the MERGED unit set: each merged unit's spike train is
            the union of its contributors, and the contributors are
            absorbed. If False (default), every original unit passes
            through 1:1 -- units, spike trains, AND labels are all
            preserved -- and the proposed merges are recorded in
            ``CurationV2.MergeGroup`` for lazy application via
            ``get_merged_sorting()``.
        description
            Free-text curation description.
        curation_source
            Provenance for how this curation row was created. Must be one of
            'manual' (default), 'figpack', or 'curation_evaluation' (a
            child accepted from a CurationEvaluation). 'analyzer_curation' is
            also a valid enum value, though no v2 workflow writes it.
        reuse_existing : bool, optional
            When a root curation already exists for the sorting and the
            caller passes non-default parameters (labels / merge_groups /
            description / apply_merge / a non-'manual' curation_source),
            those would be silently ignored by returning the existing
            row. If False (default), that situation raises ``ValueError``;
            pass True to opt into reusing the existing root and return its
            key instead. For child curations (``parent_curation_id != -1``),
            True reuses an existing child only when the parent, labels, real
            merge groups, ``apply_merge`` state, description, and curation
            source match the curation this call would insert.
        allow_unknown_unit_ids
            Controls ONLY truly-stray label keys (ids that are neither
            in ``Sorting.Unit`` nor in the curated unit set -- usually a
            typo). If False (default), a truly-stray key raises
            ``ValueError``; pass True for best-effort labeling that
            warns and drops them instead. Labels on absorbed
            contributors (present in the source sorting but merged away
            for ``apply_merge=True``) are ALWAYS silently dropped with a
            warning; this flag does not affect that path.
        allow_custom_labels
            If False (default), every label value must be in the
            canonical ``CurationLabel`` set or ``ValueError`` is raised
            (a typo guard). Pass True to accept labels outside the set
            (labs tagging units with custom semantics); the flag is
            forwarded to both ``validate_labels`` and the
            ``UnitLabel.insert`` part-table validation so a custom label
            survives the whole path. Distinct from ``allow_unknown_unit_ids``,
            which governs stray unit-id keys, not label values.
        label_policy
            How a CHILD curation (``parent_curation_id != -1``) composes its
            labels with the parent's. ``"inherit"`` (default) starts from the
            parent's labels and overlays the supplied ``labels`` per unit (a
            committed merge inherits the UNION of its contributors' labels, so
            labels on absorbed contributors do not vanish). ``"replace"`` makes
            the supplied ``labels`` the entire child state. A root curation has
            no parent, so ``"inherit"`` reduces to the supplied labels (the
            full label state).

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
        # Fetch the upstream Sorting.Unit rows once -- the parent
        # existence check and build_curated_unit_rows would otherwise
        # query the same restriction twice.
        # ``order_by="unit_id"`` makes the row order explicit -- DataJoint
        # gives no order guarantee without it. build_curated_unit_rows
        # treats this iteration as the canonical "source order" that
        # drives the NWB write order (surviving units first), so an
        # unordered fetch would silently leak DB row-order quirks.
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

        # Resolve the composition SOURCE: a root composes from the raw sort; a
        # child composes from its parent curation's committed state (units,
        # trains, label/merge namespace), so a merged-parent id is a valid
        # input and the absorbed raw contributors are not resurrected.
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

        curation_id, unit_rows, kept_unit_to_contributors, labels = (
            _curation_insert.plan_curation_insert(
                cls,
                sorting_id=sorting_id,
                sorting_units=source_units,
                merge_groups=merge_groups,
                apply_merge=apply_merge,
                labels=labels,
                allow_unknown_unit_ids=allow_unknown_unit_ids,
                parent_labels=parent_labels,
                label_policy=label_policy,
            )
        )
        if parent_curation_id != -1 and reuse_existing:
            existing_child = cls._find_matching_child_curation(
                sorting_id=sorting_id,
                parent_curation_id=parent_curation_id,
                labels=labels,
                unit_rows=unit_rows,
                kept_unit_to_contributors=kept_unit_to_contributors,
                apply_merge=apply_merge,
                description=description,
                curation_source=curation_source,
            )
            if existing_child is not None:
                logger.warning(
                    "CurationV2.insert_curation: matching child curation "
                    f"already exists for sorting_id={sorting_id}, "
                    f"parent_curation_id={parent_curation_id}; returning "
                    "existing key without staging a new NWB."
                )
                return existing_child

        # Build the id-stamped rows, stage the NWB, and insert -- retrying on a
        # concurrent curation_id collision. A child's curation_id is allocated
        # as max(existing)+1 OUTSIDE the insert transaction and the NWB is
        # staged before it, so two concurrent children of one sorting can pick
        # the same id; the loser would otherwise fail with a duplicate-key error
        # after the filesystem write. On that collision recompute the id,
        # rebuild the id-stamped rows + NWB, and retry (lock-free).
        for attempt in range(_CURATION_ID_RACE_RETRIES + 1):
            # This is the durable identity of one committed curation
            # generation. It is intentionally random rather than derived from
            # curation content: deleting and recreating the same numeric
            # curation_id must not make stale figures or caches look current.
            # Mint inside the retry loop so even a lost curation_id race gets a
            # fresh token for the row that is ultimately committed.
            curation_uuid = uuid.uuid4()
            # Build the raw MergeGroup + parent-operation rows ONCE (pure, no DB
            # I/O) and reuse them for BOTH the NWB merge-lineage scratch and the
            # CurationV2.MergeGroup insert, so the file's lineage matches the DB
            # exactly -- including for a child of a merged parent, whose raw
            # contributors are expanded here (the source-namespace
            # ``kept_unit_to_contributors`` would carry only inherited
            # singletons).
            merge_group_rows, parent_merge_group_rows = (
                build_merge_provenance_rows(
                    sorting_id=sorting_id,
                    curation_id=curation_id,
                    kept_unit_to_contributors=kept_unit_to_contributors,
                    parent_raw_contributors=parent_raw_contributors,
                )
            )

            # Stage the curated-units NWB (filesystem side-effect; the DB row
            # registration happens inside the transaction below so it rolls back
            # atomically). Staging MUST run OUTSIDE the transaction to keep the
            # inner transaction short -- pinned by
            # ``test_v1_parity.test_curation_v2_nwb_write_outside_transaction``.
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
                        "sorting_id": str(sorting_id),
                        "curation_id": int(curation_id),
                        "curation_uuid": str(curation_uuid),
                        "parent_curation_id": int(parent_curation_id),
                        # Canonical string for the enum option on the row.
                        "curation_source": getattr(
                            curation_source, "value", curation_source
                        ),
                        "merges_applied": bool(apply_merge),
                        "description": description,
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
                # The transaction rolled back the AnalysisNwbfile row and the
                # CurationV2 rows together; only the file on disk is left to
                # clean up.
                _unlink_staged_analysis_file(
                    analysis_file_name, context="CurationV2.insert_curation"
                )
                # A concurrent insert claimed this curation_id between our
                # allocation and the transaction: recompute and retry. Children
                # only -- a root's id is 0 and root idempotency is handled
                # above; any non-duplicate error propagates.
                if (
                    parent_curation_id != -1
                    and isinstance(exc, dj.errors.DuplicateError)
                    and attempt < _CURATION_ID_RACE_RETRIES
                ):
                    # If reuse is on and a concurrent insert created the SAME
                    # logical child (the likely cause of the collision), return
                    # it rather than staging a duplicate under a fresh id.
                    if reuse_existing:
                        existing_child = cls._find_matching_child_curation(
                            sorting_id=sorting_id,
                            parent_curation_id=parent_curation_id,
                            labels=labels,
                            unit_rows=unit_rows,
                            kept_unit_to_contributors=(
                                kept_unit_to_contributors
                            ),
                            apply_merge=apply_merge,
                            description=description,
                            curation_source=curation_source,
                        )
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
                    # Re-stamp the curation_id-bearing Unit rows; the master,
                    # label, merge, and NWB rows are rebuilt from curation_id at
                    # the top of the next iteration / inside the transaction.
                    unit_rows = [
                        {**row, "curation_id": curation_id} for row in unit_rows
                    ]
                    continue
                raise
            return {"sorting_id": sorting_id, "curation_id": curation_id}
        # Unreachable: the final attempt either returns or re-raises above.
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
    def _find_matching_child_curation(
        cls,
        *,
        sorting_id,
        parent_curation_id: int,
        labels: dict,
        unit_rows: list[dict],
        kept_unit_to_contributors: dict[int, list[int]],
        apply_merge: bool,
        description: str,
        curation_source: str,
    ) -> dict | None:
        """Return an existing child with the same committed curation content.

        See :func:`._curation_insert.find_matching_child_curation`. Tests patch
        it, so it stays on the class and ``insert_curation`` calls it through
        the class.
        """
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

    @classmethod
    def _next_curation_id(cls, sorting_id) -> int:
        """Return the next auto-increment ``curation_id`` for a sorting.

        ``max(existing) + 1`` (0 for the first curation). This read is NOT
        serialized against concurrent inserts, so two child curations of one
        sorting can resolve the same id; ``insert_curation`` retries on the
        resulting duplicate-key error rather than locking.
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

        Delegates the NWB write to
        :func:`._units_nwb.write_curated_units_nwb`, then
        overwrites each ``unit_rows`` entry's ``n_spikes`` (mutated in
        place) with the staged train's actual post-dedup length so the
        ``Unit.n_spikes == len(get_sorting train)`` invariant holds for
        cross-unit duplicate removal. ``source_units_abs_path`` (``None`` for a
        root -> raw Sorting NWB; the parent curation's NWB for a child) selects
        which units NWB the source spike trains come from, so a child composes
        from the parent state. Returns ``(analysis_file_name,
        units_object_id, staged_parent_nwb)``. This MUST run OUTSIDE the
        DB transaction -- it is heavy filesystem IO and the caller cleans
        up the staged file if the later insert fails.
        """
        # Stage the curated-units NWB (filesystem side-effect; the DB
        # row registration happens inside the transaction block below
        # so it rolls back atomically with the other inserts).
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
        # ``n_spikes`` must equal the length of the STORED (post-dedup)
        # train. ``build_curated_unit_rows`` estimated it from the raw
        # contributor sum; override with the staged train's actual length
        # so cross-unit duplicate removal (apply_merge=True) keeps the
        # ``Unit.n_spikes == len(get_sorting train)`` invariant.
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

        Runs the ``_safe_context()`` block: registers the already
        staged AnalysisNwbfile row, inserts the CurationV2 master + part
        rows (including raw ``MergeGroup`` and, for a child,
        ``ParentMergeGroup``), and registers the
        ``SpikeSortingOutput.CurationV2`` merge row for a single-recording
        sort. A concat-backed curation deliberately omits that row. The
        curated-units NWB was staged OUTSIDE this transaction (see
        :meth:`_stage_curation_artifact`) so the transaction stays short; on
        failure the caller removes the staged file (the DB rows roll back here).

        ``merge_group_rows`` (raw ``MergeGroup`` rows) and
        ``parent_merge_group_rows`` (the immediate parent operation in the
        parent namespace) are prebuilt by the caller and inserted verbatim --
        the same rows the curated-units NWB embeds as merge lineage, so the file
        and the DB agree.
        """
        master_row = {
            "sorting_id": sorting_id,
            "curation_id": curation_id,
            "curation_uuid": curation_uuid,
            "parent_curation_id": parent_curation_id,
            "analysis_file_name": analysis_file_name,
            "object_id": units_object_id,
            # Record user intent verbatim. ``apply_merge=True``
            # with empty ``merge_groups`` stores True (intent was
            # to merge, nothing to merge) rather than collapsing
            # to the effective state, avoiding silent semantic
            # divergence.
            "merges_applied": bool(apply_merge),
            "curation_source": curation_source,
            "description": description,
            "created_by": str(dj.config["database.user"]),
        }
        # Labels attach to the units actually written. For
        # apply_merge=False that is every original unit; for
        # apply_merge=True it is the kept set, so a label on an
        # absorbed contributor is dropped with the unit.
        written_unit_ids = {row["unit_id"] for row in unit_rows}
        # Dedupe (unit_id, label): a label repeated in the payload (or unioned
        # in from a merge) would otherwise emit a duplicate UnitLabel row and
        # fail the part's (unit_id, curation_label) primary key.
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
        # Auto-register single-recording curations so downstream consumers see
        # them without a manual merge insert. Concat curations are gated below
        # because their synthetic timeline is not session-safe. The lazy import
        # keeps curation.py free of the merge-table dependency at module load.
        # Registration MUST go through ``_merge_insert``: it derives the master
        # merge_id from the part-row identity hash.
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        register_merge = source.kind == "recording"

        # ``merge_group_rows`` / ``parent_merge_group_rows`` are built once by
        # the caller (before staging) and reused here, so the NWB merge lineage
        # and the DB ``MergeGroup`` rows are the same data. ``merge_group_rows``
        # carries a 1-element self-entry for every ``CurationV2.Unit`` that is
        # not a merge target, so ``Unit * MergeGroup`` on unit_id preserves every
        # unit; a child's rows are raw-expanded and the immediate parent
        # operation is in ``parent_merge_group_rows``.
        with cls._safe_context():
            # Register the analysis file row FIRST inside the
            # transaction so it rolls back atomically with the
            # CurationV2 rows on any later failure.
            AnalysisNwbfile().add(staged_parent_nwb, analysis_file_name)
            # insert_curation IS the validation boundary (it staged the NWB,
            # validated labels/merges, and shaped every part row), so it bypasses
            # the FactoryOnlyMaster insert guard for its already-validated master.
            cls.insert1(master_row, allow_direct_insert=True)
            cls.Unit.insert(unit_rows)
            if unit_label_rows:
                # Labels were already validated above via
                # ``validate_labels``; forward ``allow_custom_labels``
                # so the part-table override does not re-reject the
                # custom labels the caller opted into.
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

    # ---- Friendly wrappers (intent-first sugar over insert_curation) -----

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

        Intent-first sugar over the expert :meth:`insert_curation`: pre-fills
        ``parent_curation_id=-1`` (a root/initial curation) and
        ``apply_merge=False``. Use this as the first curation of a sort; later
        merge curations can branch off it via ``parent_curation_id``.

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
    def propose_merge_curation(
        cls,
        sorting_key: dict,
        merge_groups: list[list[int]],
        labels: dict | None = None,
        parent_curation_id: int = -1,
        description: str = "",
        reuse_existing: bool = False,
        label_policy: str = "inherit",
        allow_custom_labels: bool = False,
    ) -> dict:
        """Record proposed merges WITHOUT applying them (reviewable).

        Intent-first sugar over :meth:`insert_curation` that pre-fills
        ``apply_merge=False`` with ``merge_groups``. Every original unit keeps
        its id; the proposed merges live in ``CurationV2.MergeGroup`` and are
        applied lazily by ``get_merged_sorting``. ``parent_curation_id``
        (default ``-1``) lets the proposal branch off an existing initial
        curation rather than always rooting a new one. The ≥2-member-per-group
        rule is enforced by ``insert_curation`` (not re-implemented here).

        Parameters
        ----------
        sorting_key
            ``{sorting_id}`` of the upstream Sorting row.
        merge_groups
            List of merge groups, each a list of ``unit_id`` ints (≥2 each).
        labels
            Optional ``unit_id -> [label, ...]`` dict.
        parent_curation_id
            ``-1`` to root a new curation, or an existing ``curation_id`` of
            the same sort to branch off it.
        description
            Free-text description.
        reuse_existing
            If True, reuse an existing child with the same parent, labels,
            proposed merge groups, description, and provenance. Requires an
            explicit ``parent_curation_id``; root reuse would return the
            existing root and ignore the proposed merge.

        Returns
        -------
        dict
            ``{"sorting_id", "curation_id"}`` of the curation.

        See Also
        --------
        insert_curation : the full expert API this wraps.
        create_merged_curation : commit the merges instead of proposing them.
        """
        _curation_insert.assert_child_reuse_for_merge_wrapper(
            parent_curation_id=parent_curation_id,
            reuse_existing=reuse_existing,
            wrapper_name="propose_merge_curation",
        )
        return cls.insert_curation(
            sorting_key=sorting_key,
            labels=labels,
            merge_groups=merge_groups,
            apply_merge=False,
            parent_curation_id=parent_curation_id,
            description=description,
            reuse_existing=reuse_existing,
            label_policy=label_policy,
            allow_custom_labels=allow_custom_labels,
        )

    @classmethod
    def create_merged_curation(
        cls,
        sorting_key: dict,
        merge_groups: list[list[int]],
        labels: dict | None = None,
        parent_curation_id: int = -1,
        description: str = "",
        reuse_existing: bool = False,
        label_policy: str = "inherit",
        allow_custom_labels: bool = False,
    ) -> dict:
        """Create a new curation with merges applied (committed unit set).

        Intent-first sugar over :meth:`insert_curation` that pre-fills
        ``apply_merge=True`` with ``merge_groups``: each merged unit's spike
        train is the union of its contributors and the contributors are
        absorbed (the curated unit set shrinks). ``parent_curation_id``
        (default ``-1``) lets the merged curation branch off an existing
        initial curation. The ≥2-member-per-group rule is enforced by
        ``insert_curation`` (not re-implemented here).

        Parameters
        ----------
        sorting_key
            ``{sorting_id}`` of the upstream Sorting row.
        merge_groups
            List of merge groups, each a list of ``unit_id`` ints (≥2 each).
        labels
            Optional ``unit_id -> [label, ...]`` dict.
        parent_curation_id
            ``-1`` to root a new curation, or an existing ``curation_id`` of
            the same sort to branch off it.
        description
            Free-text description.
        reuse_existing
            If True, reuse an existing child with the same parent, labels,
            applied merge groups, description, and provenance. Requires an
            explicit ``parent_curation_id``; root reuse would return the
            existing root and ignore the requested merge.

        Returns
        -------
        dict
            ``{"sorting_id", "curation_id"}`` of the curation.

        See Also
        --------
        insert_curation : the full expert API this wraps.
        propose_merge_curation : record the merges without applying them.
        """
        _curation_insert.assert_child_reuse_for_merge_wrapper(
            parent_curation_id=parent_curation_id,
            reuse_existing=reuse_existing,
            wrapper_name="create_merged_curation",
        )
        return cls.insert_curation(
            sorting_key=sorting_key,
            labels=labels,
            merge_groups=merge_groups,
            apply_merge=True,
            parent_curation_id=parent_curation_id,
            description=description,
            reuse_existing=reuse_existing,
            label_policy=label_policy,
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
        ``merge_groups=[[unit_id, ...], ...]`` with integer unit IDs.
        FigPack edits are decoded by ``FigPackCuration.save_curation_from_uri``
        before reaching this API. ``merge_action`` makes the review/commit
        choice explicit:

        * ``"preview"`` stores merge groups as
          unapplied proposals for review.
        * ``"commit"`` applies non-empty merge groups into the
          child curation's unit set.
        * with no merge groups, the result is a normal committed label-edit
          child regardless of ``merge_action``.

        Manual UI edits inherit parent labels by default because the user is
        editing the visible parent curation state; pass ``label_policy="replace"``
        when ``labels`` describes the full label state.
        """
        labels, merge_groups = normalize_manual_curation(
            labels=labels, merge_groups=merge_groups
        )
        apply_merge = manual_curation_applies_merges(merge_action, merge_groups)

        if reuse_existing and parent_curation_id == -1:
            raise ValueError(
                "CurationV2.save_manual_curation(reuse_existing=True) requires "
                "an explicit parent_curation_id; root reuse would return the "
                "existing root row and ignore the manual edits."
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

        The recognition-over-recall companion to
        ``insert_curation(labels=...)``: the exact strings accepted (validated)
        for a unit label without ``allow_custom_labels=True``. Order matches the
        ``CurationLabel`` enum definition. Labels are lowercase (e.g. ``"mua"``);
        ``CurationLabel.mua`` is the typed equivalent for code that prefers an
        enum member over a string literal.
        """
        return [label.value for label in CurationLabel]

    @classmethod
    def _labels_by_unit(cls, key) -> dict[int, list[str]]:
        """Group ``UnitLabel`` rows under ``key`` into ``{unit_id: [label]}``.

        Single source for the per-unit label grouping both
        ``summarize_curation`` and ``get_sorting`` (DataFrame form) return.
        """
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

        ``values`` is a set of canonical label strings (see
        :meth:`CurationLabel.normalize`). An empty ``values`` returns an
        empty set; the caller decides whether "no labels requested" means
        all units (include filter) or none (exclude filter). Shared by
        ``get_unit_brain_regions`` (intersect) and ``get_matchable_unit_ids``
        (subtract).
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

        Pure read accessor: it reads only existing master fields and part
        rows, computing nothing new. Pass a curation key carrying
        ``sorting_id`` + ``curation_id``; any extra keys are ignored
        (normalized to the ``(sorting_id, curation_id)`` PK before any
        restriction). A ``run_v2_pipeline`` summary names its curations
        ``root_curation_id`` / ``auto_labeled_curation_id`` (it has no bare
        ``curation_id``, since one run can produce two curations), so build the
        key from the one you mean, e.g.
        ``{"sorting_id": s["sorting_id"], "curation_id": s["root_curation_id"]}``.

        Parameters
        ----------
        curation_key
            A curation key carrying ``sorting_id`` and ``curation_id``
            (``curation_id`` is only unique within a sort). From a
            ``run_v2_pipeline`` summary, build it from ``root_curation_id`` or
            ``auto_labeled_curation_id``.

        Returns
        -------
        dict
            ``sorting_id`` (UUID), ``curation_id`` (int), ``n_units`` (count of
            ``CurationV2.Unit`` rows), ``labels`` (``unit_id -> [label, ...]``
            from ``UnitLabel``), ``merge_groups`` (only real >1-contributor
            merge groups), ``unit_contributor_groups`` (from
            ``get_unit_contributor_groups`` -- includes a 1-element self-entry
            per unit), ``merges_applied`` (the stored field),
            ``is_merge_preview`` (True iff not applied AND at least one
            >1-contributor merge group -- the same condition as
            ``has_unapplied_proposed_merges``), ``merge_id`` (from
            ``SpikeSortingOutput.CurationV2``, ``None`` if unregistered), and
            ``description``, ``created_at``, and ``created_by``. When
            ``annotation_sets`` is explicitly supplied, ``unit_properties``
            contains the explicitly selected evaluation/custom properties;
            no latest evaluation or annotation set is inferred.

        See Also
        --------
        insert_curation : the expert write API.
        get_unit_contributor_groups : the merge-group provenance this summarizes.
        """
        # The full-PK guard MUST run before any DB/schema access: importing
        # SpikeSortingOutput activates its dj.schema, so it is imported only
        # after the guard (build_curation_summary is DB-free but kept here too
        # so the guard is unambiguously the first statement).
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

        # Defensive: a curation may not be merge-registered in edge cases;
        # return None rather than letting fetch1 raise on zero rows.
        merge_ids = (SpikeSortingOutput.CurationV2 & pk).fetch("merge_id")
        merge_id = merge_ids[0] if len(merge_ids) else None

        # The pure formatter (DB-free) assembles the return dict and derives
        # ``is_merge_preview`` from the values in hand rather than re-fetching
        # via ``has_unapplied_proposed_merges``.
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

        "source" is :meth:`get_source_recording` and "sorting input" is
        :meth:`get_sorting_input_recording`. "Masked" means silenced over the
        sort's artifact exclusions: a single-recording sort that pins an
        artifact detection still comes back unmasked here.

        A single-recording sort returns its reusable preprocessed
        ``Recording``, without the sort's pinned artifact mask. A concat
        sort returns the materialized ``ConcatenatedRecording``, which
        includes the member masks. A sort of a motion-corrected recording
        returns that ``MotionCorrectedRecording``: the traces the sorter
        read, with the sort's mask already applied and only the channels the
        correction kept. Call :meth:`get_source_recording` or
        :meth:`get_sorting_input_recording` to get one meaning for every
        sort.

        ``@classmethod`` so the merge-table dispatcher's
        ``source_table.get_recording(merge_key)`` call (which binds
        the part class, not an instance) resolves correctly for v2
        ``merge_id``s. Without this method, the merge dispatcher
        ``SpikeSortingOutput.get_recording`` raises ``AttributeError``
        on every v2 ``merge_id``.

        ``key`` is normalized through a DataJoint restriction so it
        accepts both the single-dict form (``{"sorting_id": ...}``)
        and the list-of-dict form the merge dispatcher passes
        (``query.fetch("KEY")``). Both forms are valid DataJoint
        restrictions; ``fetch1`` raises if more than one row matches.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.

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
        # The curated spike times live in the sort's effective-traces timeline:
        # a single-recording sort reads its Recording cache; a concat sort reads
        # the materialized ConcatenatedRecording cache; a corrected sort reads
        # its MotionCorrectedRecording. The traces load as persisted, so a
        # single recording comes back WITHOUT the sort's artifact mask (the
        # reusable preprocessed traces), while a corrected recording comes back
        # already masked: it was written with the sort's mask applied, and its
        # ``apply_artifact_mask=False`` means "do not mask again", not
        # "unmasked".
        return SortingSelection.load_stored_traces(traces)

    @classmethod
    def get_source_recording(cls, key: dict) -> "si.BaseRecording":
        """Return the sort's original source recording.

        The preprocessed ``Recording`` cache the sort's source was built
        from, on its acquisition clock: never artifact-masked and never
        motion-corrected, whatever the sort selected. A concatenation has
        several original recordings, so a concat-backed curation raises;
        read each member's with ``ConcatMemberCuration.get_source_recording``.

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

        The sort's effective traces, unwhitened: silenced over the pinned
        artifact detection's excluded periods when the sort pins one (a
        concatenation carries its member masks), and the
        ``MotionCorrectedRecording`` (only its kept channels) when the sort
        selected a motion correction. The clock is the sort's: acquisition
        time for a single recording, the synthetic concatenation clock for a
        concat sort (see ``ConcatMemberCuration.get_sorting_input_recording``
        for a member on its own clock). This is the recording every analyzer
        rebuild and UnitMatch bundle extraction start from
        (``_sorting_analyzer.resolve_canonical_recording`` opened by
        ``read_canonical_recording``).

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

        With ``as_dataframe=True`` returns a pandas DataFrame with one
        row per unit and the spike-times list, useful for ad-hoc
        inspection that does not need a full SI sorting object.

        ``@classmethod`` so the merge-table dispatcher binds
        correctly when called as
        ``source_table.get_sorting(merge_key)``.

        Like ``Sorting.get_sorting``, the SI-object path maps the stored
        ABSOLUTE spike times back to recording frames via
        ``np.searchsorted`` rather than SI's affine
        ``NwbSortingExtractor``, so disjoint-interval sorts recover the
        original frames. ``as_dataframe=True`` returns the absolute
        seconds read straight from the curated units NWB plus the
        ``curation_label`` lists joined from ``UnitLabel``.

        A curation created with ``apply_merge=False`` returns its UNMERGED
        preview units here -- the proposed merges live in ``MergeGroup`` and
        are applied only by ``get_merged_sorting``. Because consumers
        (SortedSpikesGroup / decoding) read through this method, a warning is
        emitted in that case so the proposed merges are not silently ignored.

        A zero-unit curation returns an empty sorting (with a warning);
        ``Sorting.get_analyzer`` raises ``ZeroUnitAnalyzerError`` instead.

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

        A curation created with ``apply_merge=False`` records proposed merges in
        ``MergeGroup`` without applying them. A real merge is a group with more
        than one contributor; every unit also carries a 1-element self-entry, so
        a plain root curation (all self-entries) returns ``False``. Short-
        circuits on ``merges_applied`` so the ``MergeGroup`` fetch is skipped on
        the common already-applied / root path.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        merges_applied : bool, optional
            The row's ``merges_applied`` value when the caller already holds
            it; passing it skips the redundant scalar fetch. ``None`` (default)
            fetches it from ``key``.

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

        Committed states -- the root, a label-only child, and an applied-merge
        child -- are valid downstream curations: their ``CurationV2.Unit`` set is
        the unit namespace a consumer (decoding, evaluation) reads. A PREVIEW
        curation (``apply_merge=False`` with a real >=2-unit proposed merge
        group) is a draft: it still carries every original unit and only records
        the proposed merge in ``MergeGroup``, so its unit set is not the final
        merged set. ``is_committed_curation`` is the exact negation of
        :meth:`has_unapplied_proposed_merges`; named affirmatively because the
        evaluation boundary asks "may I score this curation?".

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        merges_applied : bool, optional
            The row's ``merges_applied`` value when the caller already holds it;
            passing it skips the redundant scalar fetch (and the ``MergeGroup``
            fetch on the already-applied / root path).

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

        ``True`` for a root, or a label-only child of a non-merged ancestor:
        the curation carries exactly the raw ``Sorting.Unit`` ids, so the cached
        raw-sort display analyzer already holds its namespace. ``False`` for a
        merged curation -- or a label-only child of a merged parent -- whose unit
        set includes merged ids absent from the raw sort. The single owner of
        this predicate: ``CurationEvaluation.make_fetch`` and the curation
        analyzer resolver route their raw-analyzer fast paths on it, so compute
        and interactive reads cannot drift. Distinct from
        :meth:`is_committed_curation`: a committed label-only child of a merged
        parent is committed yet does NOT match the raw namespace.

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

        The guard at every evaluation/analyzer boundary: a preview curation has
        unapplied proposed merges, so scoring or plotting it would attach data
        to the UNMERGED preview units rather than the final merged unit set the
        user intends.

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
            "set. Commit the merge first (create_merged_curation / "
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

        Walks the v2 Selection tables and source parts, dispatching on the
        input source the restriction names:

        - single-recording keys route ``RecordingSelection`` ->
          ``SortingSelection.RecordingSource`` -> ``SortingSelection`` ->
          ``ArtifactDetectionSource``;
        - concat keys (``concat_recording_id`` / ``session_group_owner`` /
          ``session_group_name``) route
          ``ConcatenatedRecordingSelection`` ->
          ``SortingSelection.ConcatenatedRecordingSource`` ->
          ``SortingSelection``;
        - with no source key, both source families are matched (a broad query
          must not silently drop concat-backed curations);
        - ``preprocessing_params_name`` is a CROSS-SOURCE key -- it lives on
          both the recording and concat selections -- so it filters whichever
          family routes and, alone, matches both.

        Accepted keys: the recording keys (``nwb_file_name``, ``team_name``,
        ``sort_group_id``, ``interval_list_name``, ``recording_id``), the
        cross-source ``preprocessing_params_name``, the concat keys above, and
        the shared sort / curation keys (``sorter``, ``sorter_params_name``,
        ``sorting_id``, ``artifact_detection_id``,
        ``motion_corrected_recording_id``, ``curation_id``).
        Mixing recording and concat source keys is rejected (a sort has
        exactly one input source).

        This method is the single owner of v2's source-part join topology;
        ``SpikeSortingOutput._get_restricted_merge_ids_v2`` delegates here.

        ``artifact_detection_id=None`` means "no artifact-detection pass"
        (no standalone or concat-member detection selected), NOT "match
        anything" -- only an absent key is a wildcard. A detection ID matches
        a concat if any frozen member uses it. ``motion_corrected_recording_id``
        follows the same convention: ``None`` matches only sorts that read
        their source's own traces, an id matches only sorts of that
        ``MotionCorrectedRecording``, and an absent key matches corrected and
        uncorrected sorts of the same source alike, so pass it to tell them
        apart.

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
            ``ValueError`` (a deliberate v2 query). If ``False``, an
            unknown key instead returns ``None`` (multi-source dispatch:
            the key names another pipeline's column).

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

        Owns the v2 source-part walk so consumers -- e.g. decoding's
        ``UnitWaveformFeatures`` -- don't re-implement v2's join topology.
        ``key`` must carry ``sorting_id`` (sorter and nwb_file_name are fixed
        per sort, independent of ``curation_id``). For a concat-backed sort the
        ``nwb_file_name`` is the FIRST frozen ``MemberSnapshot`` member's session
        (the same deterministic parent anchor the sort's analysis NWB uses), so
        downstream provenance resolves to the anchor member rather than raising.

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

        ``"clusterless_threshold_crossings"`` when the underlying sort is the
        clusterless thresholder (its single "unit" is a threshold-crossing event
        stream, not a sorted neuron) and ``"sorted_units"`` otherwise. Derived
        from the sort's ``sorter`` (single source of truth) via
        ``get_sort_metadata``, so it never drifts from the sort row. Consuming
        surfaces use it to avoid treating a threshold-crossing pseudo-unit as a
        trackable neuron.

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

        ``{kept_unit_id: [contributor_unit_id, ...]}`` where the contributor
        ids are in the curation's own (composition) namespace -- the namespace
        its ``CurationV2.Unit`` rows live in. A root curation composes from the
        raw sort, so this is the raw ``MergeGroup``; a CHILD composes from its
        parent, so this is the parent-namespace ``ParentMergeGroup`` (a child
        of a merged parent inherits raw-contributor entries on ``MergeGroup``
        that are NOT proposed merges of its own, so reading ``MergeGroup`` here
        would misreport a committed child as a preview). Every
        ``CurationV2.Unit`` row has at least one own-namespace entry keyed by
        its own ``unit_id`` (a 1-element self-entry for a pass-through unit);
        ``len(groups[X]) > 1`` means unit X is the kept-unit leader of a
        proposed or applied merge.

        Used by ``get_merged_sorting`` (which filters ``len(contribs) > 1``, so
        self-entries are auto-skipped) and ``has_unapplied_proposed_merges``.
        For RAW-contributor provenance ("which original raw units contributed
        to kept unit X?") use ``_raw_contributor_groups`` -- ``MergeGroup``
        always stays in the raw ``Sorting.Unit`` namespace.

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
        # A child records its parent-namespace operation in ParentMergeGroup;
        # a root has none, so its own namespace IS the raw MergeGroup. The
        # presence of ParentMergeGroup rows is the child discriminator (a child
        # always has at least the per-unit self-entries).
        if cls.ParentMergeGroup & key:
            return cls._fetch_contributor_groups(
                cls.ParentMergeGroup & key, "parent_unit_id"
            )
        return cls._raw_contributor_groups(key)

    @classmethod
    def _raw_contributor_groups(cls, key) -> dict[int, list[int]]:
        """Return each unit's raw ``MergeGroup`` contributors.

        ``{unit_id: [contributor_unit_id, ...]}`` -- raw ``Sorting.Unit``
        provenance for every unit under ``key``, in ascending unit and
        contributor order.
        """
        return cls._fetch_contributor_groups(
            cls.MergeGroup & key, "contributor_unit_id"
        )

    @staticmethod
    def _fetch_contributor_groups(
        relation, contributor_field: str
    ) -> dict[int, list[int]]:
        """Fetch merge-provenance rows grouped by kept unit, in id order.

        ``order_by`` makes BOTH the outer dict key order and each contributor
        list (ascending) deterministic; DataJoint gives no ordering without
        it. ``units_to_merge`` -- and therefore the ids SI's
        MergeUnitsSorting assigns on the lazy merge path -- depends on the
        dict insertion order, so an unordered fetch would let DB row-order
        quirks leak into the lazy merged-unit ids.
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

        A curation built with ``apply_merge=False`` (preview) still
        carries every original unit, so the proposed merges recorded in
        ``CurationV2.MergeGroup`` are applied lazily here without
        re-running the sort. The lazy merge deduplicates in absolute
        spike time so disjoint-recording wall-clock gaps are respected, then
        reuses the required stored ``spike_sample_index`` frames.

        When the curation was created with ``apply_merge=True`` the base
        sorting is ALREADY merged (contributors absorbed at insert), so
        the MergeGroup contributors are no longer present in it; the base
        is returned verbatim rather than re-applying merges over missing
        units. Likewise returns the base unchanged when no merge group
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

        If ``include_labels`` is provided (iterable of strings or
        ``CurationLabel``), restricts to units carrying at least one
        of those labels. Otherwise returns all CurationV2.Unit rows
        for the key. Same concat-sort guard semantics as
        ``Sorting.get_unit_brain_regions``: raises
        ``ConcatBrainRegionAmbiguousError`` for concat-backed
        sortings unless ``allow_anchor_member=True``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        include_labels : iterable of str or CurationLabel, optional
            If a non-empty iterable, restrict to units carrying at least
            one of these labels. ``None`` (the default) and an empty
            iterable both mean no filter (all units), matching
            ``get_matchable_unit_ids``.
        allow_anchor_member : bool, optional
            If ``True``, return anchor-member regions for concat-backed
            sortings instead of raising. Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            One row per (unit, electrode) with the brain-region columns
            and a ``region_resolution`` label.
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
            # An empty label set means "no filter" (all units), NOT "match
            # nothing": restricting with ``& []`` (an empty DataJoint OrList)
            # would silently return zero rows. Guard on the materialized set's
            # truthiness, matching ``get_matchable_unit_ids``.
            if include_values:
                labeled = self._unit_ids_with_labels(key, include_values)
                unit_restriction = unit_restriction & [
                    {"unit_id": uid} for uid in labeled
                ]
        return unit_brain_region_df(unit_restriction, resolution)

    def get_matchable_unit_ids(
        self,
        key: dict,
        exclude_labels: "Iterable" = frozenset({"reject", "noise", "artifact"}),
    ) -> "np.ndarray":
        """Curated unit IDs with no excluded labels.

        Unlabeled units AND units labeled only ``accept`` / ``mua`` are
        included. A unit with ANY excluded label is excluded even if it
        also carries an included label (e.g., a ``mua`` + ``artifact``
        unit is excluded).

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.
        exclude_labels : iterable of str or CurationLabel, optional
            Labels that disqualify a unit. Defaults to
            ``{"reject", "noise", "artifact"}``.

        Returns
        -------
        np.ndarray
            Sorted 1-D integer array ``(n_units,)`` of unit IDs carrying
            no excluded label.
        """
        import numpy as np

        exclude_values = {
            CurationLabel.normalize(lbl) for lbl in exclude_labels
        }
        all_units = (self.Unit & key).fetch("unit_id")
        excluded_set = self._unit_ids_with_labels(key, exclude_values)
        kept = [int(u) for u in all_units if int(u) not in excluded_set]
        return np.asarray(sorted(kept), dtype=int)

    @classmethod
    def get_sort_group_info(cls, key: dict) -> "dj.Table":
        """Return ALL electrodes in the sort group joined to BrainRegion.

        Returns a DataJoint relation (not a DataFrame, not single-row)
        covering EVERY electrode in the sort group so a multi-region
        probe surfaces every represented region. Callers can chain
        restrictions / fetches on the returned relation.

        For a concat-backed sort the electrodes come from the FIRST frozen
        ``MemberSnapshot`` member's sort group (the same deterministic parent
        anchor the per-unit Electrode FK uses), so the merge dispatcher and
        downstream sort-group queries resolve rather than raising. As with the
        concat brain-region anchor, the anchor member's regions may differ from
        later members if the probe re-anatomized across sessions.

        ``@classmethod`` so the merge-table dispatcher
        ``SpikeSortingOutput.get_sort_group_info`` (which calls
        ``source_table.get_sort_group_info(merge_key)`` with the
        bound part *class*, not an instance) does not raise
        ``TypeError`` on v2 ``merge_id``s.

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
        from spyglass.spikesorting.v2.recording import RecordingSelection

        sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        if source.kind == "recording":
            recording_key = source.key
        else:  # concatenated_recording -> anchor member's sort group
            anchor_recording_id, _nwb, _preproc = (
                _sorting_fetch.resolve_concat_anchor(source.key)
            )
            recording_key = {"recording_id": anchor_recording_id}
        # ``RecordingSelection.fetch1("KEY")`` returns only the UUID PK;
        # the upstream nwb_file_name + sort_group_id are non-PK columns
        # that we have to fetch explicitly.
        nwb_file_name, sort_group_id = (
            RecordingSelection & recording_key
        ).fetch1("nwb_file_name", "sort_group_id")
        return sort_group_electrode_regions(
            {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
        )
