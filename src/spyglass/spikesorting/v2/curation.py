"""Curation of sorted units.

Tables (all final-shape under the zero-migration policy):
    CurationV2 (+ Unit + UnitLabel) -- Manual; lineage via parent_curation_id.

``curation_source`` is restricted to true CurationV2 provenance values
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

from typing import TYPE_CHECKING

import datajoint as dj

from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._curation_transforms import (
    validate_curation_label_rows,
    validate_labels,
)
from spyglass.spikesorting.v2._signal_math import _MERGE_DEDUP_DELTA_MS
from spyglass.spikesorting.v2._units_nwb import (
    build_lazy_merged_sorting,
    numpysorting_from_abs_times,
    read_units_abs_spike_times,
    recording_timestamps,
    write_curated_units_nwb,
)
from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
from spyglass.spikesorting.v2.utils import (
    CurationLabel,
    CurationSource,
    _assert_v2_db_safe,
    transaction_or_noop,
    unit_brain_region_df,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import pandas as pd
    import spikeinterface as si

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_curation")


@schema
class CurationV2(SpyglassMixin, dj.Manual):
    """Manual curation labels + merge groups for a sorted Sorting.

    Multiple curations per sort are allowed via ``curation_id``; each
    can record a ``parent_curation_id`` for lineage (validation-only --
    the schema does not self-FK because DataJoint cannot resolve a
    nullable self-FK across renamed columns cleanly).

    ``object_id`` (NOT ``units_object_id``) is the column name the
    ``SpikeSortingOutput`` merge-table router expects when dispatching
    ``get_spike_times()``; using the wrong name would break that routing.
    """

    definition = """
    -> Sorting
    curation_id: int
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)
    merges_applied=0: bool
    curation_source = 'manual': enum('manual', 'analyzer_curation', 'figpack')
    description: varchar(255)
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
        custom label later would otherwise need a forbidden ``ALTER
        TABLE`` under the zero-migration policy. Pass
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
        permissive_labels: bool = False,
        allow_custom_labels: bool = False,
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
            against the ``CurationLabel`` enum. ``None`` (the default)
            and ``{}`` are equivalent and produce a curation with no
            ``UnitLabel`` rows.
        parent_curation_id
            ``-1`` for a root curation; otherwise must reference an
            existing CurationV2 row for the same sorting.
        merge_groups
            Optional list of merge groups, each a list of ``unit_id``
            ints. Each group must have at least 2 members (a single-unit
            group is rejected as a likely typo). The merged unit inherits
            the peak channel + amplitude of the highest-amplitude
            contributor. For ``apply_merge=True`` the merged unit gets a
            fresh id ``max(source unit_ids) + 1``, assigned in ASCENDING
            MIN-CONTRIBUTOR order (``sorted(groups, key=min)``), INDEPENDENT
            of the order the caller lists the groups. The lazy preview path
            (``get_merged_sorting`` on an apply_merge=False curation) numbers
            merges the same way, so the applied and lazy paths assign the
            SAME fresh id to the SAME content group (guarded by
            ``test_lazy_vs_applied_merge_frames_equal`` and
            ``test_curation_two_merge_groups_assign_ids_in_canonical_min_order``).
            Merged-unit ids are arbitrary labels: only which group receives
            ``max+1`` changes with input order -- spike content and unit
            count are identical.
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
            ``get_merged_sorting()`` (so a preview can be reviewed before
            committing). Empty and singleton merge groups are rejected
            (rather than treated as no-ops/renames) to surface likely
            typos.
        description
            Free-text curation description.
        curation_source
            Provenance for how this curation row was created. Must be one of
            'manual' (default), 'analyzer_curation', or 'figpack'.
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
        permissive_labels
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
            survives the whole path. Distinct from ``permissive_labels``,
            which governs stray unit-id keys, not label values.

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
            truly-stray unit_id(s) and ``permissive_labels`` is False.
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

        labels, curation_source = cls._normalize_curation_inputs(
            labels, curation_source, allow_custom_labels
        )

        existing_root = cls._validate_parent_or_reuse_root(
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

        curation_id, unit_rows, kept_unit_to_contributors = (
            cls._build_curation_insert_plan(
                sorting_id=sorting_id,
                sorting_units=sorting_units,
                merge_groups=merge_groups,
                apply_merge=apply_merge,
                labels=labels,
                permissive_labels=permissive_labels,
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

        # Stage the curated-units NWB (filesystem side-effect; the DB
        # row registration happens inside the transaction below so it
        # rolls back atomically). Staging MUST run OUTSIDE the
        # transaction to keep the inner transaction short -- pinned by
        # ``test_v1_parity.test_curation_v2_nwb_write_outside_transaction``.
        analysis_file_name, units_object_id, staged_parent_nwb = (
            cls._stage_curation_artifact(
                sorting_id=sorting_id,
                kept_unit_to_contributors=kept_unit_to_contributors,
                apply_merge=apply_merge,
                labels=labels,
                unit_rows=unit_rows,
            )
        )

        try:
            cls._insert_curation_rows_transaction(
                sorting_id=sorting_id,
                curation_id=curation_id,
                parent_curation_id=parent_curation_id,
                analysis_file_name=analysis_file_name,
                units_object_id=units_object_id,
                staged_parent_nwb=staged_parent_nwb,
                apply_merge=apply_merge,
                curation_source=curation_source,
                description=description,
                unit_rows=unit_rows,
                labels=labels,
                kept_unit_to_contributors=kept_unit_to_contributors,
                allow_custom_labels=allow_custom_labels,
            )
        except Exception:
            # The transaction rolled back the AnalysisNwbfile row and
            # the CurationV2 rows together; only the file on disk is
            # left to clean up.
            cls._cleanup_staged_curation_file(analysis_file_name)
            raise

        return {"sorting_id": sorting_id, "curation_id": curation_id}

    # ---- insert_curation steps (extracted helpers) -----------------------

    @staticmethod
    def _normalized_labels(labels: dict | None) -> dict[int, tuple[str, ...]]:
        """Normalize ``{unit_id: labels}`` for semantic equality checks."""
        return {
            int(unit_id): tuple(
                sorted(CurationLabel.normalize(label) for label in unit_labels)
            )
            for unit_id, unit_labels in (labels or {}).items()
            if unit_labels
        }

    @staticmethod
    def _normalized_real_merge_groups(merge_groups) -> tuple[tuple[int, ...], ...]:
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
        """Return an existing child with the same committed curation content."""
        written_unit_ids = {int(row["unit_id"]) for row in unit_rows}
        effective_labels = {
            int(unit_id): unit_labels
            for unit_id, unit_labels in labels.items()
            if int(unit_id) in written_unit_ids
        }
        target_labels = cls._normalized_labels(effective_labels)
        target_merges = cls._normalized_real_merge_groups(
            kept_unit_to_contributors
        )
        candidates = (
            cls
            & {
                "sorting_id": sorting_id,
                "parent_curation_id": parent_curation_id,
                "merges_applied": bool(apply_merge),
                "curation_source": curation_source,
                "description": description,
            }
        ).fetch("KEY", as_dict=True, order_by="curation_id")
        for candidate in candidates:
            existing_labels = cls._normalized_labels(
                cls._labels_by_unit(candidate)
            )
            if existing_labels != target_labels:
                continue
            if (
                cls._normalized_real_merge_groups(cls.get_merge_groups(candidate))
                != target_merges
            ):
                continue
            return candidate
        return None

    @classmethod
    def _normalize_curation_inputs(
        cls,
        labels: dict | None,
        curation_source: str | CurationSource,
        allow_custom_labels: bool,
    ) -> tuple[dict, str]:
        """Normalize + validate ``labels`` and ``curation_source``.

        Returns ``(labels, curation_source)`` with ``labels`` coerced to a
        ``{int unit_id: [label, ...]}`` dict (``None`` -> ``{}``) and
        ``curation_source`` coerced to its canonical ``CurationSource``
        value. Raises ``ValueError`` on a scalar label value, a label that
        fails ``validate_labels``, or an invalid ``curation_source``.
        """
        if labels is None:
            # ``None`` is semantically equivalent to "no labels".
            # Normalize to ``{}`` so the rest of the helper does not
            # need an extra None check.
            labels = {}
        # Reject a scalar string label value BEFORE coercing to list:
        # ``list("custom_tag")`` would silently split into per-character
        # labels (and ``allow_custom_labels=True`` would then write them).
        # A label value must be an explicit list/tuple of labels.
        for uid, lbls in labels.items():
            if isinstance(lbls, str) or not isinstance(lbls, (list, tuple)):
                raise ValueError(
                    "CurationV2.insert_curation: labels[unit_id] must be a "
                    f"list of labels; got {type(lbls).__name__} for "
                    f"unit_id={uid}. Wrap a single label as a one-element "
                    'list, e.g. {unit_id: ["mua"]}.'
                )
        # Normalize label keys to int once so the rest of the helper
        # can do straight ``labels.get(int_uid, [])`` lookups.
        labels = {int(uid): list(lbls) for uid, lbls in labels.items()}

        validate_labels(labels, allow_custom_labels=allow_custom_labels)

        # Coerce curation_source through the enum up front so a typo raises a
        # friendly error on EVERY path -- including the idempotent
        # existing-root early return below, which would otherwise swallow an
        # invalid value and return the existing root instead of rejecting it.
        try:
            curation_source = CurationSource(curation_source).value
        except ValueError as exc:
            raise ValueError(
                f"CurationV2.insert_curation: curation_source="
                f"{curation_source!r} is not a CurationSource value. "
                f"Valid: {[m.value for m in CurationSource]}."
            ) from exc

        return labels, curation_source

    @classmethod
    def _validate_parent_or_reuse_root(
        cls,
        sorting_id,
        parent_curation_id: int,
        labels: dict,
        merge_groups,
        description: str,
        apply_merge: bool,
        curation_source: str,
        reuse_existing: bool,
    ) -> dict | None:
        """Validate the parent curation, or return an existing root to reuse.

        For a child curation (``parent_curation_id != -1``) verifies the
        parent row exists. For a root curation (``parent_curation_id ==
        -1``) returns the canonical existing-root key when one already
        exists (idempotent re-insert), or ``None`` to proceed with a fresh
        insert. Raises ``ValueError`` if the named parent is missing, or if
        a root already exists and the caller passed non-default parameters
        without ``reuse_existing=True``.
        """
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
        else:
            # Idempotency: if a root curation already exists for this
            # sorting_id, return its key without staging a new NWB or
            # creating a duplicate row. Without this, every repeat
            # call grows another row + analysis file + merge-table
            # entry.
            # ``order_by="curation_id"`` makes the reused root deterministic:
            # if more than one root somehow exists (a raw/manual insert that
            # bypassed this helper), the canonical lowest-curation_id root is
            # returned every time rather than a DB-row-order coin flip.
            existing_root = (
                cls
                & {
                    "sorting_id": sorting_id,
                    "parent_curation_id": -1,
                }
            ).fetch("KEY", as_dict=True, order_by="curation_id")
            if existing_root:
                # A root curation already exists. If the caller passed ANY
                # non-default parameter -- labels / merge_groups / description
                # / apply_merge / a non-"manual" curation_source -- it would be
                # SILENTLY ignored by returning the existing row, a
                # parameter-change-with-no-effect footgun. The apply_merge /
                # curation_source cases matter because ``merges_applied`` and
                # the curation provenance record user intent: a second
                # ``apply_merge=True`` insert must NOT quietly return a
                # ``merges_applied=False`` root. Raise unless the caller opts
                # into reuse. curation_source is already coerced to its enum
                # value above, so a typo has already raised before reaching
                # this point (it is never silently treated as "unchanged").
                curation_source_changed = curation_source != "manual"
                if (
                    bool(labels)
                    or bool(merge_groups)
                    or bool(description)
                    or apply_merge
                    or curation_source_changed
                ) and not reuse_existing:
                    raise ValueError(
                        "CurationV2.insert_curation: a root curation "
                        f"already exists for sorting_id={sorting_id}, but "
                        "you passed labels / merge_groups / description / "
                        "apply_merge / curation_source that would be silently "
                        "ignored. Pass reuse_existing=True to reuse the "
                        "existing root, or curate as a child with "
                        "parent_curation_id=<existing root curation_id>."
                    )
                logger.warning(
                    "CurationV2.insert_curation: root curation already "
                    f"exists for sorting_id={sorting_id}; returning "
                    "existing key without staging a new NWB."
                )
                return existing_root[0]
        return None

    @classmethod
    def _build_curation_insert_plan(
        cls,
        sorting_id,
        sorting_units: list[dict],
        merge_groups,
        apply_merge: bool,
        labels: dict,
        permissive_labels: bool,
    ) -> tuple[int, list[dict], dict[int, list[int]]]:
        """Resolve the curation_id and build the curated ``Unit`` rows.

        Resolves the auto-increment ``curation_id`` (the one DB-coupled
        step) and delegates row shaping + label-key validation to the
        DB-free :func:`._curation_plan.build_curation_insert_plan`. Returns
        ``(curation_id, unit_rows, kept_unit_to_contributors)`` -- the
        :class:`._curation_plan.CurationInsertPlan` namedtuple, which
        unpacks positionally.
        """
        from spyglass.spikesorting.v2._curation_plan import (
            build_curation_insert_plan,
        )

        # Resolve which curation_id to use (auto-increment within sort) --
        # the one DB-coupled step; the rest of the plan is pure.
        existing_ids = (cls & {"sorting_id": sorting_id}).fetch("curation_id")
        curation_id = int(max(existing_ids)) + 1 if len(existing_ids) else 0

        return build_curation_insert_plan(
            sorting_id=sorting_id,
            sorting_units=sorting_units,
            merge_groups=merge_groups,
            apply_merge=apply_merge,
            labels=labels,
            permissive_labels=permissive_labels,
            curation_id=curation_id,
        )

    @classmethod
    def _stage_curation_artifact(
        cls,
        sorting_id,
        kept_unit_to_contributors: dict,
        apply_merge: bool,
        labels: dict,
        unit_rows: list[dict],
    ) -> tuple[str, str, str]:
        """Stage the curated-units NWB and reconcile per-unit ``n_spikes``.

        Delegates the NWB write to
        :func:`._units_nwb.write_curated_units_nwb`, then
        overwrites each ``unit_rows`` entry's ``n_spikes`` (mutated in
        place) with the staged train's actual post-dedup length so the
        ``Unit.n_spikes == len(get_sorting train)`` invariant holds for
        cross-unit duplicate removal. Returns ``(analysis_file_name,
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
        parent_curation_id: int,
        analysis_file_name: str,
        units_object_id: str,
        staged_parent_nwb: str,
        apply_merge: bool,
        curation_source: str,
        description: str,
        unit_rows: list[dict],
        labels: dict,
        kept_unit_to_contributors: dict,
        allow_custom_labels: bool,
    ) -> None:
        """Insert the master/Unit/UnitLabel/MergeGroup rows atomically.

        Runs the ``transaction_or_noop`` block: registers the already
        staged AnalysisNwbfile row, inserts the CurationV2 master + part
        rows, and registers the ``SpikeSortingOutput.CurationV2`` merge
        row. The curated-units NWB was staged OUTSIDE this transaction
        (see :meth:`_stage_curation_artifact`) so the transaction stays
        short; on failure the caller removes the staged file (the DB rows
        roll back here).
        """
        master_row = {
            "sorting_id": sorting_id,
            "curation_id": curation_id,
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
        }
        # Labels attach to the units actually written. For
        # apply_merge=False that is every original unit; for
        # apply_merge=True it is the kept set, so a label on an
        # absorbed contributor is dropped with the unit.
        written_unit_ids = {row["unit_id"] for row in unit_rows}
        unit_label_rows = [
            {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "unit_id": unit_id,
                "curation_label": CurationLabel.normalize(label),
            }
            for unit_id, lbls in labels.items()
            if unit_id in written_unit_ids
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

        # Persist per-unit merge provenance for queryable
        # bulk-audit. ``kept_unit_to_contributors`` carries 1-element
        # self-entries for every ``CurationV2.Unit`` row that is not
        # itself a merge target (unmerged units always, and -- for
        # apply_merge=False preview -- the absorbed contributors that
        # remain in Unit), so every Unit row has at least one
        # MergeGroup row keyed by its own ``unit_id``. Joining
        # ``Unit * MergeGroup`` on unit_id therefore preserves every
        # unit in both modes.
        merge_group_rows = [
            {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "unit_id": kept_uid,
                "contributor_unit_id": int(contributor_uid),
            }
            for kept_uid, contributors in kept_unit_to_contributors.items()
            for contributor_uid in contributors
        ]

        with transaction_or_noop(cls.connection):
            # Register the analysis file row FIRST inside the
            # transaction so it rolls back atomically with the
            # CurationV2 rows on any later failure.
            AnalysisNwbfile().add(staged_parent_nwb, analysis_file_name)
            cls.insert1(master_row)
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

    @classmethod
    def _cleanup_staged_curation_file(cls, analysis_file_name: str) -> None:
        """Delete a staged curated-units NWB after a failed insert.

        The DB transaction already rolled back the ``AnalysisNwbfile`` row
        and the ``CurationV2`` rows together; only the file on disk is
        left to clean up (DataJoint cannot roll back filesystem side
        effects). A failure to unlink is logged, not raised -- the
        original insert error is what the caller re-raises.
        """
        import pathlib

        try:
            abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            pathlib.Path(abs_path).unlink(missing_ok=True)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "CurationV2.insert_curation: failed to clean up "
                f"staged analysis file {analysis_file_name!r}: "
                f"{cleanup_exc!r}"
            )

    # ---- Friendly wrappers (intent-first sugar over insert_curation) -----

    @classmethod
    def create_initial_curation(
        cls,
        sorting_key: dict,
        labels: dict | None = None,
        description: str = "",
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
        )

    @staticmethod
    def _assert_child_reuse_for_merge_wrapper(
        *,
        parent_curation_id: int,
        reuse_existing: bool,
        wrapper_name: str,
    ) -> None:
        """Prevent merge wrappers from reusing an unrelated root curation."""
        if not reuse_existing or parent_curation_id != -1:
            return
        raise ValueError(
            f"CurationV2.{wrapper_name}(reuse_existing=True) requires an "
            "explicit parent_curation_id. Root curation reuse returns the "
            "existing root row and would ignore the requested merge/proposal; "
            "branch from an existing curation instead."
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
        cls._assert_child_reuse_for_merge_wrapper(
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
        cls._assert_child_reuse_for_merge_wrapper(
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
        from spyglass.spikesorting.v2._enums import CurationLabel

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
    def summarize_curation(cls, curation_key: dict) -> dict:
        """Return a notebook-printable summary of one curation.

        Pure read accessor: it reads only existing master fields and part
        rows, computing nothing new. Accepts either a minimal curation key
        (``{"sorting_id", "curation_id"}``) or a full ``run_v2_pipeline``
        run summary -- the run summary carries keys that are NOT part of the
        curation primary key (``pipeline_preset`` / ``recording_id`` /
        ``artifact_detection_id`` / ``n_units`` / ...), so it is normalized
        to the ``(sorting_id, curation_id)`` PK before any restriction.

        Parameters
        ----------
        curation_key
            A curation key or a ``run_v2_pipeline`` run summary. Must carry both
            ``sorting_id`` and ``curation_id`` (``curation_id`` is only unique
            within a sort).

        Returns
        -------
        dict
            ``sorting_id`` (UUID), ``curation_id`` (int), ``n_units`` (count of
            ``CurationV2.Unit`` rows), ``labels`` (``unit_id -> [label, ...]``
            from ``UnitLabel``), ``merge_groups`` (from ``get_merge_groups`` --
            note this includes a 1-element self-entry per unit; real merges
            have >1 contributor), ``merges_applied`` (the stored field),
            ``is_merge_preview`` (True iff not applied AND at least one
            >1-contributor merge group -- the same condition as
            ``has_unapplied_proposed_merges``), ``merge_id`` (from
            ``SpikeSortingOutput.CurationV2``, ``None`` if unregistered), and
            ``description``.

        See Also
        --------
        insert_curation : the expert write API.
        get_merge_groups : the merge-group provenance this summarizes.
        """
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        if (
            "sorting_id" not in curation_key
            or "curation_id" not in curation_key
        ):
            raise ValueError(
                "summarize_curation: key must contain 'sorting_id' and "
                "'curation_id' (curation_id is only unique within a sort). "
                f"Got keys: {sorted(curation_key)}."
            )
        pk = {
            "sorting_id": curation_key["sorting_id"],
            "curation_id": curation_key["curation_id"],
        }

        # fetch1 doubles as the "exactly one curation" check.
        merges_applied, description = (cls & pk).fetch1(
            "merges_applied", "description"
        )

        labels = cls._labels_by_unit(pk)

        # Defensive: a curation may not be merge-registered in edge cases;
        # return None rather than letting fetch1 raise on zero rows.
        merge_ids = (SpikeSortingOutput.CurationV2 & pk).fetch("merge_id")
        merge_id = merge_ids[0] if len(merge_ids) else None

        # Derive ``is_merge_preview`` from the values already in hand rather
        # than calling ``has_unapplied_proposed_merges`` (which would re-fetch
        # ``merges_applied`` and rebuild the merge groups). This is its exact
        # definition: not applied AND at least one >1-contributor group.
        merge_groups = cls.get_merge_groups(pk)
        is_merge_preview = not bool(merges_applied) and any(
            len(contribs) > 1 for contribs in merge_groups.values()
        )

        return {
            "sorting_id": pk["sorting_id"],
            "curation_id": int(pk["curation_id"]),
            "n_units": len(cls.Unit & pk),
            "labels": labels,
            "merge_groups": merge_groups,
            "merges_applied": bool(merges_applied),
            "is_merge_preview": is_merge_preview,
            "merge_id": merge_id,
            "description": str(description),
        }

    # ---- Accessors -------------------------------------------------------

    @classmethod
    def get_recording(cls, key: dict) -> "si.BaseRecording":
        """Return the cached preprocessed recording for a CurationV2 row.

        Resolves the upstream ``Recording`` via
        ``SortingSelection.resolve_source`` and delegates to
        ``Recording().get_recording`` (which already applies the
        ``is_filtered=True`` annotation). Repeating the annotation
        here is harmless.

        ``@classmethod`` so the merge-table dispatcher's
        ``source_table.get_recording(merge_key)`` call (which binds
        the part class, not an instance) resolves correctly for v2
        ``merge_id``s. Without this method, the dispatcher at
        ``spikesorting_merge.py:317`` raises ``AttributeError`` on
        every v2 ``merge_id``.

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
        from spyglass.spikesorting.v2.recording import Recording
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        # The curated spike times live in the sort's input-recording timeline:
        # a single-recording sort reads its Recording cache; a concat sort reads
        # the materialized ConcatenatedRecording cache.
        if source.kind == "recording":
            recording = Recording().get_recording(source.key)
        else:  # concatenated_recording
            recording = ConcatenatedRecording().get_recording(source.key)
        recording.annotate(is_filtered=True)
        return recording

    @classmethod
    def _load_curation_recording_meta(cls, key):
        """Fetch the master row + upstream recording metadata for a curation.

        Returns ``(row, recording_row, fs, abs_path)``: the ``CurationV2``
        master row, the upstream ``Recording`` row, its sampling frequency,
        and the curated-units NWB path. Shared by ``get_sorting`` and
        ``get_merged_sorting`` so the master row is fetched ONCE and its
        ``sorting_id`` threaded into ``_upstream_recording_row`` (skipping a
        redundant lookup). The curated-units NWB itself is NOT read here --
        callers read it lazily so ``get_sorting`` can short-circuit a
        zero-unit curation without touching the filesystem.
        """
        row = (cls & key).fetch1()
        recording_row = cls._upstream_recording_row(
            key, sorting_id=row["sorting_id"]
        )
        fs = float(recording_row["sampling_frequency"])
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return row, recording_row, fs, abs_path

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
        import spikeinterface as si

        row, recording_row, fs, abs_path = cls._load_curation_recording_meta(
            key
        )

        # A curation created with apply_merge=False records PROPOSED merges in
        # MergeGroup but does NOT apply them: get_sorting (what consumers such
        # as SortedSpikesGroup / decoding read via SpikeSortingOutput) returns
        # the UNMERGED preview units. Warn here so ad-hoc inspection is not
        # silently misled; the decoding CONSUMERS additionally RAISE via
        # SpikeSortingOutput.assert_decoding_merge_ids_ok so a preview curation
        # never reaches a decode.
        if cls.has_unapplied_proposed_merges(
            key, merges_applied=row["merges_applied"]
        ):
            logger.warning(
                "CurationV2.get_sorting: curation "
                f"(sorting_id={row['sorting_id']}, "
                f"curation_id={row['curation_id']}) has proposed merges that "
                "are NOT applied (apply_merge=False); this returns the "
                "UNMERGED units. Use get_merged_sorting to apply the proposal, "
                "or re-curate with apply_merge=True to commit it."
            )

        if len(cls.Unit & key) == 0:
            # Zero-unit curations are valid (a user may curate a
            # zero-unit sort; the Empty/Boundary invariant allows an
            # empty ``CurationV2.Unit``).
            logger.warning(
                "CurationV2.get_sorting: curation "
                f"(sorting_id={row['sorting_id']}, "
                f"curation_id={row['curation_id']}) has zero units; "
                "returning an empty sorting."
            )
            if not as_dataframe:
                return si.NumpySorting.from_unit_dict({}, sampling_frequency=fs)
            import pandas as pd

            return pd.DataFrame(
                {"spike_times": [], "curation_label": []},
                index=pd.Index([], name="unit_id", dtype=int),
            )

        abs_times = read_units_abs_spike_times(abs_path)
        if not as_dataframe:
            return numpysorting_from_abs_times(abs_times, recording_row, fs)

        import pandas as pd

        # Join the ``curation_label`` lists from ``UnitLabel`` so the
        # returned DataFrame carries a ``curation_label`` column.
        # External notebook code reading ``df["curation_label"]`` works
        # without poking at the part table directly.
        unit_ids = list(abs_times)
        labels_by_unit = cls._labels_by_unit(key)
        # Index by ``unit_id`` (the same indexing
        # Sorting.get_sorting(as_dataframe=True) uses); see that
        # method's docstring for the rationale.
        return pd.DataFrame(
            {
                "spike_times": [abs_times[u] for u in unit_ids],
                "curation_label": [labels_by_unit.get(u, []) for u in unit_ids],
            },
            index=pd.Index(unit_ids, name="unit_id"),
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
        return any(
            len(contribs) > 1 for contribs in cls.get_merge_groups(key).values()
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
          ``session_group_name`` / ``motion_correction_params_name``) route
          ``ConcatenatedRecordingSelection`` ->
          ``SortingSelection.ConcatenatedRecordingSource`` ->
          ``SortingSelection``;
        - with no source key, both source families are matched (a broad query
          must not silently drop concat-backed curations).

        So a cross-table restriction over the v2 part-table convention keys --
        the recording keys (``nwb_file_name``, ``team_name``, ``sort_group_id``,
        ``interval_list_name``, ``preprocessing_params_name``, ``recording_id``,
        ``artifact_detection_id``), the concat keys above, and the shared sort /
        curation keys (``sorter``, ``sorter_params_name``, ``sorting_id``,
        ``curation_id``) -- resolves to the ``CurationV2`` rows it selects.
        Mixing recording and concat source keys is rejected (a sort has exactly
        one input source).

        This method is the SINGLE owner of v2's source-part join topology.
        ``SpikeSortingOutput._get_restricted_merge_ids_v2`` delegates here
        instead of re-implementing v2 schema knowledge in the merge master,
        so a new v2 source part (e.g. concat) is taught to exactly one
        place. Returns a ``CurationV2`` query (callers map it to merge ids
        via ``SpikeSortingOutput.CurationV2``), or ``None`` in lenient mode
        (``strict=False``) when the key names no v2 column.

        Unknown restriction keys raise ``ValueError`` when ``strict`` (the
        default -- a deliberate query, where an unknown key is a typo;
        silently dropping it would return wrong-but-non-empty results);
        when ``strict=False`` (the multi-source ``get_restricted_merge_ids``
        dispatch) an unknown key instead returns ``None`` (the caller then
        contributes no v2 rows), since it names another pipeline's column and
        this is not a v2 query.
        ``restrict_by_artifact=True`` honors
        the v2 IntervalList convention where the artifact-removed
        valid_times row is named
        ``f"artifact_detection_{artifact_detection_id}"``; callers can
        supply either the bare ``artifact_detection_id`` or the
        artifact-detection IntervalList and both resolve.
        ``artifact_detection_id=None`` means "no artifact-detection pass"
        (anti-join to sorts with no ``ArtifactDetectionSource`` row), NOT
        "match anything" -- only an absent key is a wildcard.

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
            that are not v2 columns.
        """
        import uuid

        from spyglass.spikesorting.v2.recording import RecordingSelection
        from spyglass.spikesorting.v2.sorting import SortingSelection
        from spyglass.spikesorting.v2.utils import (
            parse_artifact_detection_interval_list_name,
        )
        from spyglass.utils import logger

        rec_keys = [
            "nwb_file_name",
            "team_name",
            "sort_group_id",
            "interval_list_name",
            "preprocessing_params_name",
            "recording_id",
        ]
        # Concat-source restriction keys route through
        # ``SortingSelection.ConcatenatedRecordingSource`` ->
        # ``ConcatenatedRecordingSelection`` -> ``SessionGroup`` instead of the
        # single-recording chain. ``motion_correction_params_name`` is included
        # so a concat restriction can pin the motion recipe too.
        concat_keys = [
            "concat_recording_id",
            "session_group_owner",
            "session_group_name",
            "motion_correction_params_name",
        ]
        sort_keys = [
            "sorter",
            "sorter_params_name",
            "sorting_id",
            "artifact_detection_id",
        ]
        curation_keys = ["curation_id"]
        allowed = (
            set(rec_keys)
            | set(concat_keys)
            | set(sort_keys)
            | set(curation_keys)
        )
        unknown = set(key) - allowed
        if unknown:
            if not strict:
                # Lenient multi-source-dispatch path: an unknown key names a
                # column from another pipeline, so this is not a v2 query.
                # Return ``None`` so the caller contributes no v2 rows. The
                # strict raise is reserved for a deliberate v2 query
                # (sources=['v2'] or a direct resolve_restriction), where an
                # unknown key is a typo.
                return None
            raise ValueError(
                "CurationV2.resolve_restriction: "
                f"unknown restriction keys {sorted(unknown)}. Allowed: "
                f"{sorted(allowed)}."
            )

        key = key.copy()
        # ``restrict_by_artifact`` maps the artifact-detection IntervalList
        # convention (``f"artifact_detection_{artifact_detection_id}"``) back
        # to the ``artifact_detection_id`` master-side column so the v2 join
        # chain downstream resolves correctly.
        if restrict_by_artifact and "interval_list_name" in key:
            artifact_detection_id = parse_artifact_detection_interval_list_name(
                key["interval_list_name"]
            )
            if artifact_detection_id is not None:
                key["artifact_detection_id"] = artifact_detection_id
                key.pop("interval_list_name", None)
            elif "artifact_detection_id" not in key:
                # The caller asked to restrict by artifact, but the
                # interval name is not the
                # ``artifact_detection_{uuid}`` form and no
                # artifact_detection_id was supplied, so there is nothing to
                # map. Warn instead of silently returning unrestricted ids.
                logger.warning(
                    "CurationV2.resolve_restriction: "
                    "restrict_by_artifact=True but interval_list_name "
                    f"{key['interval_list_name']!r} is not an "
                    "'artifact_detection_{uuid}' interval and no "
                    "artifact_detection_id was given; v2 results will NOT be "
                    "artifact-restricted. Pass artifact_detection_id=... or "
                    "use the artifact-detection interval to restrict."
                )

        # ``parse_artifact_detection_interval_list_name`` returns a str and a caller may
        # pass either a str or a UUID, but ``artifact_detection_id`` is a uuid column.
        # Normalize to a ``uuid.UUID`` so the ArtifactDetectionSource intersection below
        # is unambiguous and a malformed id fails fast here rather than
        # silently matching nothing.
        if key.get("artifact_detection_id") is not None:
            key["artifact_detection_id"] = uuid.UUID(
                str(key["artifact_detection_id"])
            )

        # Route through the input source the restriction names. A concat
        # restriction (concat_recording_id / session-group / motion recipe)
        # joins SortingSelection.ConcatenatedRecordingSource ->
        # ConcatenatedRecordingSelection; otherwise the single-recording chain
        # joins SortingSelection.RecordingSource -> RecordingSelection. The two
        # sets are mutually exclusive (a sort has exactly one input source), so
        # mixing them is a contradictory restriction and is rejected.
        concat_restriction = {k: key[k] for k in concat_keys if k in key}
        rec_restriction = {k: key[k] for k in rec_keys if k in key}
        if concat_restriction and rec_restriction:
            raise ValueError(
                "CurationV2.resolve_restriction: cannot combine concat-source "
                f"keys {sorted(concat_restriction)} with single-recording "
                f"keys {sorted(rec_restriction)}; a sort has exactly one input "
                "source. Restrict by one source family."
            )
        if concat_restriction:
            from spyglass.spikesorting.v2.session_group import (
                ConcatenatedRecordingSelection,
            )

            concat_sel = ConcatenatedRecordingSelection & concat_restriction
            sort_concat_source = (
                SortingSelection.ConcatenatedRecordingSource
                * concat_sel.proj()
            )
            sort_master = SortingSelection * sort_concat_source.proj()
        elif rec_restriction:
            rec_table = RecordingSelection & rec_restriction
            sort_rec_source = (
                SortingSelection.RecordingSource * rec_table.proj()
            )
            sort_master = SortingSelection * sort_rec_source.proj()
        else:
            # No source-specific key (an unrestricted v2 query or a
            # sort-/curation-only restriction): match BOTH source families.
            # Every SortingSelection has exactly one input source, so the
            # master itself IS the recording-source union concat-source set --
            # restricting through one source part here would silently drop the
            # other (the bug that omitted concat-backed curations from broad v2
            # merge queries).
            sort_master = SortingSelection
        # ``artifact_detection_id`` lives on the optional ``SortingSelection.
        # ArtifactDetectionSource`` part, NOT on ``SortingSelection`` -- it is absent
        # from ``sort_master``'s heading, so a dict restriction with it is
        # silently dropped (verified: the compiled SQL is identical with and
        # without the key). Apply the non-artifact sort keys directly, then
        # resolve ``artifact_detection_id`` through the part. In the v2 design
        # the presence/absence of an ``ArtifactDetectionSource`` row IS the
        # artifact-detection state, so ``artifact_detection_id=None`` means
        # "no artifact-detection pass" (anti-join to sorts with no part row),
        # NOT "match anything" -- only an absent key is a wildcard.
        sort_restriction = {
            k: key[k]
            for k in sort_keys
            if k in key and k != "artifact_detection_id"
        }
        sort_master = sort_master & sort_restriction
        if "artifact_detection_id" in key:
            if key["artifact_detection_id"] is None:
                sort_master = (
                    sort_master
                    - SortingSelection.ArtifactDetectionSource.proj()
                )
            else:
                artifact_detection_source = (
                    SortingSelection.ArtifactDetectionSource
                    & {"artifact_detection_id": key["artifact_detection_id"]}
                )
                sort_master = sort_master & artifact_detection_source.proj()

        curation_restriction = {k: key[k] for k in curation_keys if k in key}
        return (cls * sort_master.proj("sorting_id")) & curation_restriction

    @classmethod
    def get_sort_metadata(cls, key) -> tuple:
        """Return ``(sorter, nwb_file_name)`` for a curation's underlying sort.

        Owns the v2 source-part walk so consumers -- e.g. decoding's
        ``UnitWaveformFeatures`` -- don't re-implement v2's join topology.
        ``key`` must carry ``sorting_id`` (sorter and nwb_file_name are fixed
        per sort, independent of ``curation_id``). For a concat-backed sort the
        ``nwb_file_name`` is the FIRST ``SessionGroup.Member``'s session (the
        same deterministic parent anchor the sort's analysis NWB uses), so
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
    def get_merge_groups(cls, key) -> dict[int, list[int]]:
        """Return ``{kept_unit_id: [contributor_unit_id, ...]}`` for a curation.

        Reads the ``CurationV2.MergeGroup`` part rows and groups them by
        kept unit. Every ``CurationV2.Unit`` row has at least one
        ``MergeGroup`` entry keyed by its own ``unit_id`` -- unmerged
        units and (in apply_merge=False preview) absorbed contributors
        carry a 1-element self-entry. ``len(merge_groups[X]) > 1`` means
        unit X is the kept-unit leader of a (proposed or applied) merge;
        to find merges X is a contributor TO (preview mode), query
        ``CurationV2.MergeGroup`` directly by ``contributor_unit_id``.

        Used by ``get_merged_sorting`` (which filters
        ``len(contribs) > 1``, so self-entries are auto-skipped) and
        exposed publicly so users can do bulk-audit queries directly.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationV2`` row.

        Returns
        -------
        dict[int, list[int]]
            ``{kept_unit_id: [contributor_unit_id, ...]}`` with sorted
            contributor lists.
        """
        # ``order_by`` makes BOTH the outer dict key order and the
        # contributor list order deterministic (DataJoint gives no
        # ordering without it). ``units_to_merge`` -- and therefore the
        # ids SI's MergeUnitsSorting assigns on the lazy merge path --
        # depends on the dict insertion order; an unordered fetch would
        # let DB row-order quirks leak into the lazy merged-unit ids.
        rows = (cls.MergeGroup & key).fetch(
            "unit_id",
            "contributor_unit_id",
            as_dict=True,
            order_by=("unit_id", "contributor_unit_id"),
        )
        groups: dict[int, list[int]] = {}
        for row in rows:
            uid = int(row["unit_id"])
            groups.setdefault(uid, []).append(int(row["contributor_unit_id"]))
        # Sort contributor lists deterministically so callers can
        # rely on stable ordering when comparing across runs.
        for uid in groups:
            groups[uid].sort()
        return groups

    @classmethod
    def get_merged_sorting(cls, key: dict) -> "si.BaseSorting":
        """Return the curated BaseSorting with merge groups applied.

        A curation built with ``apply_merge=False`` (preview) still
        carries every original unit, so the proposed merges recorded in
        ``CurationV2.MergeGroup`` are applied lazily here without
        re-running the sort. The lazy merge is rebuilt from absolute
        spike times so disjoint-recording wall-clock gaps are respected.

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
        # merges_applied OR no multi-contributor group -> the base sorting
        # already IS the result; delegate to get_sorting (its read is
        # unavoidable in those cases).
        if bool((cls & key).fetch1("merges_applied")):
            return cls.get_sorting(key)
        merge_groups = cls.get_merge_groups(key)
        units_to_merge = [
            contribs
            for kept_uid, contribs in merge_groups.items()
            if len(contribs) > 1
        ]
        if not units_to_merge:
            return cls.get_sorting(key)

        # Read the curated units NWB + the recording timeline ONCE, then
        # rebuild the merged sorting via the pure compute core
        # (``build_lazy_merged_sorting``). Calling get_sorting here would
        # re-open the units NWB, re-fetch the recording row, and re-read the
        # timeline only to discard the intermediates (and would emit a
        # spurious "merges NOT applied" warning -- we ARE applying them). The
        # merge is applied in ABSOLUTE time (gap-correct on disjoint
        # recordings) inside the compute core; see its docstring.
        _row, recording_row, fs, abs_path = cls._load_curation_recording_meta(
            key
        )
        abs_times = read_units_abs_spike_times(abs_path)
        timestamps = recording_timestamps(recording_row)
        return build_lazy_merged_sorting(
            abs_times,
            units_to_merge,
            timestamps,
            fs,
            delta_s=_MERGE_DEDUP_DELTA_MS / 1000.0,
        )

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
                    "allow_anchor_member=True for anchor-only regions. "
                    "Per-session regions require cross-session unit "
                    "matching, which is not available in this build."
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

        For a concat-backed sort the electrodes come from the FIRST
        ``SessionGroup.Member``'s sort group (the same deterministic parent
        anchor the per-unit Electrode FK uses), so the merge dispatcher and
        downstream sort-group queries resolve rather than raising. As with the
        concat brain-region anchor, the anchor member's regions may differ from
        later members if the probe re-anatomized across sessions.

        ``@classmethod`` so the merge-table dispatcher at
        ``spikesorting_merge.py:346`` (which calls
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
        from spyglass.common.common_ephys import Electrode as _Electrode
        from spyglass.common.common_region import BrainRegion
        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
            SortGroupV2,
        )
        from spyglass.spikesorting.v2.sorting import Sorting

        sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        if source.kind == "recording":
            recording_key = source.key
        else:  # concatenated_recording -> anchor member's sort group
            anchor_recording_id, _nwb, _preproc = (
                Sorting._resolve_concat_anchor(source.key)
            )
            recording_key = {"recording_id": anchor_recording_id}
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
            (SortGroupV2.SortGroupElectrode & sg_restriction)
            * _Electrode
            * BrainRegion
        )

    @classmethod
    def _upstream_recording_row(cls, key, *, sorting_id=None) -> dict:
        """Fetch the upstream Recording row for a CurationV2 key.

        Used by ``get_sorting`` to recover the recording's
        sampling-frequency and timestamp metadata (matching the
        ``Sorting.get_sorting`` round-trip convention). For a concat-backed
        sort this is the ``ConcatenatedRecording`` row (the timeline the curated
        spike times were written against), NOT a per-member ``Recording``.
        ``@classmethod`` so it can be invoked from the other classmethod
        accessors.

        ``key`` may be a single dict or the list-of-dict form the
        merge dispatcher passes; the restriction-based fetch
        normalizes both. Pass ``sorting_id`` when the caller already
        holds the master row to skip the redundant ``sorting_id`` lookup.
        """
        from spyglass.spikesorting.v2.recording import Recording
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        if sorting_id is None:
            sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        if source.kind == "recording":
            return (Recording & source.key).fetch1()
        return (ConcatenatedRecording & source.key).fetch1()
