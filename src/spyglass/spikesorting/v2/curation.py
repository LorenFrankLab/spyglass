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

from typing import TYPE_CHECKING

import datajoint as dj

from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._curation_transforms import (
    build_curated_unit_rows,
    validate_curation_label_rows,
    validate_labels,
)
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
    MetricsSource,
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

# Coincidence window (ms) for cross-unit duplicate-spike removal when
# merging units. Matches SpikeInterface's ``MergeUnitsSorting`` default
# (the value v1's lazy ``get_merged_sorting`` used). A neuron's
# refractory period (~1-2 ms) means a genuine spike train never has a
# sub-0.4 ms pair, so this only removes double-detections of one physical
# event shared across merged contributors.
_MERGE_DEDUP_DELTA_MS = 0.4


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
        still gets a ``curation_label`` indexed column so v1-style
        consumers see empty lists for unlabeled units.

        ``curation_label`` is a ``varchar(32)`` validated against the
        canonical ``CurationLabel`` set at insert time on EVERY insert
        path -- including a direct ``UnitLabel.insert1`` / ``insert`` (the
        overrides below), not only via ``CurationV2.insert_curation``.
        DataJoint *can* declare an enum column (``metrics_source`` on the
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
            rows = list(rows)
            validate_curation_label_rows(
                rows, allow_custom_labels=allow_custom_labels
            )
            super().insert(rows, **kwargs)

    class MergeGroup(SpyglassMixinPart):
        """Per-merge-group provenance: kept unit <- contributor units.

        Restores v1's merge-group recoverability lost when v2
        replaced v1's ``merge_groups`` NWB column with a per-unit
        kept-set. v1 stored merge groups as a single list-of-lists
        column on the units NWB; v2 stores them here as queryable
        part rows so bulk-audit queries ("show me every (kept_unit,
        contributor) pair across sortings") and provenance retrieval
        ("for this paper, list every merge decision") are easy.

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
        metrics_source: str | MetricsSource = "manual",
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
            ``UnitLabel`` rows -- matches v1's permissive default at
            ``v1/curation.py:49``.
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
            of the order the caller lists the groups -- a deliberate
            departure from v1's user-iteration order (``v1/curation.py:359``;
            see feature-parity.md). The lazy preview path
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
            committing). Matches v1's ``apply_merge`` semantics at
            ``v1/curation.py:359`` for non-degenerate input. (v2 rejects
            empty and singleton merge groups, which v1 silently accepted
            as no-ops/renames -- a deliberate deviation that surfaces
            likely typos.)
        description
            Free-text curation description.
        metrics_source
            Provenance for any attached metrics blob. Must be one of
            'manual' (default), 'analyzer_curation', or 'figpack'.
        permissive_labels
            Controls ONLY truly-stray label keys (ids that are neither
            in ``Sorting.Unit`` nor in the curated unit set -- usually a
            typo). If False (default), a truly-stray key raises
            ``ValueError``; pass True for best-effort labeling that
            warns and drops them instead. Labels on absorbed
            contributors (present in the source sorting but merged away
            for ``apply_merge=True``) are ALWAYS silently dropped with a
            warning (v1 parity, ``v1/curation.py:391``); this flag does
            not affect that path.
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
        """
        import pathlib as _pathlib

        sorting_id = sorting_key["sorting_id"]
        # Fetch the upstream Sorting.Unit rows once -- the parent
        # existence check and _build_curated_unit_rows would otherwise
        # query the same restriction twice.
        # ``order_by="unit_id"`` makes the row order explicit -- DataJoint
        # gives no order guarantee without it. _build_curated_unit_rows
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
        if labels is None:
            # ``None`` is semantically equivalent to "no labels" per
            # v1's ``CurationV1.insert_curation`` signature at
            # ``v1/curation.py:49``. Normalize to ``{}`` so the rest
            # of the helper does not need an extra None check.
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

        # Coerce metrics_source through the enum up front so a typo raises a
        # friendly error on EVERY path -- including the idempotent
        # existing-root early return below, which would otherwise swallow an
        # invalid value and return the existing root instead of rejecting it.
        try:
            metrics_source = MetricsSource(metrics_source).value
        except ValueError as exc:
            raise ValueError(
                f"CurationV2.insert_curation: metrics_source="
                f"{metrics_source!r} is not a MetricsSource value. "
                f"Valid: {[m.value for m in MetricsSource]}."
            ) from exc

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
            # creating a duplicate row. Matches v1's pattern at
            # ``v1/curation.py:88-93``; without this, every repeat
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
                # / apply_merge / a non-"manual" metrics_source -- it would be
                # SILENTLY ignored by returning the existing row, a
                # parameter-change-with-no-effect footgun. The apply_merge /
                # metrics_source cases matter because ``merges_applied`` and
                # the metrics provenance record user intent: a second
                # ``apply_merge=True`` insert must NOT quietly return a
                # ``merges_applied=False`` root. Raise unless the caller opts
                # into reuse. metrics_source is already coerced to its enum
                # value above, so a typo has already raised before reaching
                # this point (it is never silently treated as "unchanged").
                metrics_source_changed = metrics_source != "manual"
                if (
                    bool(labels)
                    or bool(merge_groups)
                    or bool(description)
                    or apply_merge
                    or metrics_source_changed
                ) and not reuse_existing:
                    raise ValueError(
                        "CurationV2.insert_curation: a root curation "
                        f"already exists for sorting_id={sorting_id}, but "
                        "you passed labels / merge_groups / description / "
                        "apply_merge / metrics_source that would be silently "
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

        # Resolve which curation_id to use (auto-increment within sort).
        existing_ids = (cls & {"sorting_id": sorting_id}).fetch("curation_id")
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
            apply_merge=apply_merge,
        )

        # Validate labels against ``source_unit_ids ∪ written_unit_ids``
        # (v1 parity, ``v1/curation.py:391``: v1 writes labels by
        # iterating final unit_ids, silently dropping labels keyed on
        # absorbed contributors). Two kinds of "stray":
        #   - TRULY stray (typo): keys not in source AND not in written
        #     -- nothing they could ever have referred to. Raise by
        #     default (typo protection); ``permissive_labels=True``
        #     restores warn-and-drop.
        #   - ABSORBED-source: keys that exist in source but are
        #     contributors absorbed into a merge (so not in written).
        #     Always warn and drop -- the label cannot attach to the
        #     merged unit, but v1 callers passing original-id labels
        #     should not break.
        written_unit_ids = {row["unit_id"] for row in unit_rows}
        source_unit_ids = {int(r["unit_id"]) for r in sorting_units}
        label_keys = set(labels)
        truly_stray = label_keys - (source_unit_ids | written_unit_ids)
        absorbed_label_ids = label_keys & (source_unit_ids - written_unit_ids)
        if truly_stray:
            message = (
                "CurationV2.insert_curation: labels reference unit_id(s) "
                f"{sorted(truly_stray)} that are neither in Sorting.Unit "
                "nor in the curated unit set for "
                f"sorting_id={sorting_id} (apply_merge={apply_merge}). "
                f"Curated unit_ids: {sorted(written_unit_ids)}."
            )
            if not permissive_labels:
                raise ValueError(
                    f"{message} Pass permissive_labels=True to ignore "
                    "stray labels instead of raising."
                )
            logger.warning(f"{message} Those labels are ignored.")
        if absorbed_label_ids:
            logger.warning(
                "CurationV2.insert_curation: labels keyed on absorbed "
                f"contributor unit(s) {sorted(absorbed_label_ids)} are "
                "dropped -- they cannot attach to the merged unit (v1 "
                "parity: v1 also silently ignored these). "
                f"sorting_id={sorting_id}."
            )

        # Stage the curated-units NWB (filesystem side-effect; the DB
        # row registration happens inside the transaction block below
        # so it rolls back atomically with the other inserts).
        (
            analysis_file_name,
            units_object_id,
            staged_parent_nwb,
            n_spikes_by_uid,
        ) = cls._stage_curated_units_nwb(
            sorting_id=sorting_id,
            kept_unit_to_contributors=kept_unit_to_contributors,
            apply_merge=apply_merge,
            labels=labels,
        )
        # ``n_spikes`` must equal the length of the STORED (post-dedup)
        # train. ``_build_curated_unit_rows`` estimated it from the raw
        # contributor sum; override with the staged train's actual length
        # so cross-unit duplicate removal (apply_merge=True) keeps the
        # ``Unit.n_spikes == len(get_sorting train)`` invariant.
        for row in unit_rows:
            if row["unit_id"] in n_spikes_by_uid:
                row["n_spikes"] = n_spikes_by_uid[row["unit_id"]]

        try:
            master_row = {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "parent_curation_id": parent_curation_id,
                "analysis_file_name": analysis_file_name,
                "object_id": units_object_id,
                # Record user intent verbatim, matching v1's
                # ``v1/curation.py:123`` semantic. ``apply_merge=True``
                # with empty ``merge_groups`` stores True (intent was
                # to merge, nothing to merge) rather than collapsing
                # to the effective state -- consistent with v1 and
                # avoids silent semantic divergence.
                "merges_applied": bool(apply_merge),
                "metrics_source": metrics_source,
                "description": description,
            }
            # Labels attach to the units actually written. For
            # apply_merge=False that is every original unit (v1 preview
            # parity); for apply_merge=True it is the kept set, so a
            # label on an absorbed contributor is dropped with the unit.
            written_unit_ids = {row["unit_id"] for row in unit_rows}
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
        except Exception:
            # The transaction rolled back the AnalysisNwbfile row and
            # the CurationV2 rows together; only the file on disk is
            # left to clean up.
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                _pathlib.Path(abs_path).unlink(missing_ok=True)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "CurationV2.insert_curation: failed to clean up "
                    f"staged analysis file {analysis_file_name!r}: "
                    f"{cleanup_exc!r}"
                )
            raise

        return {"sorting_id": sorting_id, "curation_id": curation_id}

    # ---- Curated-unit construction --------------------------------------

    @classmethod
    def _build_curated_unit_rows(
        cls,
        sorting_id,
        sorting_units: list[dict],
        merge_groups: list[list[int]],
        curation_id: int,
        apply_merge: bool,
    ) -> tuple[list[dict], dict[int, list[int]]]:
        """Resolve the post-merge ``CurationV2.Unit`` rows.

        Thin orchestrator over
        :func:`._curation_transforms.build_curated_unit_rows`, which owns
        the merge-group validation, ``kept_unit_to_contributors`` mapping,
        and per-unit row construction (DB-free, parity-critical). The
        classmethod is kept as the surface ``insert_curation`` calls and
        the v2 tests call / monkeypatch; the logic lives in the service
        module.

        ``sorting_units`` is the pre-fetched ``Sorting.Unit`` rows for
        ``sorting_id``; the caller fetches once and threads through to
        avoid re-querying. Returns ``(unit_rows,
        kept_unit_to_contributors)``.
        """
        return build_curated_unit_rows(
            sorting_id=sorting_id,
            sorting_units=sorting_units,
            merge_groups=merge_groups,
            curation_id=curation_id,
            apply_merge=apply_merge,
        )

    @classmethod
    def _stage_curated_units_nwb(
        cls,
        sorting_id,
        kept_unit_to_contributors: dict,
        apply_merge: bool,
        labels: dict,
    ) -> tuple[str, str, str, dict]:
        """Write the curated-units NWB.

        Thin delegator to :func:`._units_nwb.write_curated_units_nwb`;
        kept as a ``CurationV2`` classmethod because ``insert_curation``
        calls ``cls._stage_curated_units_nwb(...)`` and the v1-parity AST
        test (``test_v1_parity.py``) asserts that call by name. The NWB
        staging IO -- resolving the source sort's units NWB, the
        ``apply_merge`` cross-unit dedup, and the ragged ``curation_label``
        column -- lives in the service module. Returns
        ``(analysis_file_name, units_object_id, nwb_file_name,
        n_spikes_by_uid)``.
        """
        return write_curated_units_nwb(
            sorting_id=sorting_id,
            kept_unit_to_contributors=kept_unit_to_contributors,
            apply_merge=apply_merge,
            labels=labels,
        )

    # ---- Accessors -------------------------------------------------------

    @classmethod
    def get_recording(cls, key: dict) -> "si.BaseRecording":
        """Return the cached preprocessed recording for a CurationV2 row.

        Mirrors ``CurationV1.get_recording`` at
        ``v1/curation.py:163-179``. Resolves the upstream
        ``Recording`` via ``SortingSelection.resolve_source`` and
        delegates to ``Recording().get_recording`` (which already
        applies the ``is_filtered=True`` annotation). Repeating
        the annotation here is harmless and matches v1's exact
        call surface.

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

        sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        if source.kind != "recording":
            raise NotImplementedError(
                "CurationV2.get_recording: concat-source sorts are not "
                "yet supported."
            )
        recording = Recording().get_recording(source.key)
        recording.annotate(is_filtered=True)
        return recording

    @classmethod
    def get_sorting(
        cls, key: dict, as_dataframe: bool = False
    ) -> "si.BaseSorting | pd.DataFrame":
        """Return the curated SpikeInterface BaseSorting (or DataFrame).

        With ``as_dataframe=True`` returns a pandas DataFrame mirroring
        v1's ``Curation.fetch_nwb`` convenience -- one row per unit
        with the spike-times list, useful for ad-hoc inspection that
        does not need a full SI sorting object.

        ``@classmethod`` so the merge-table dispatcher binds
        correctly when called as
        ``source_table.get_sorting(merge_key)``.

        Like ``Sorting.get_sorting``, the SI-object path maps the stored
        ABSOLUTE spike times back to recording frames via
        ``np.searchsorted`` (v1-parity) rather than SI's affine
        ``NwbSortingExtractor``, so disjoint-interval sorts recover the
        original frames. ``as_dataframe=True`` returns the absolute
        seconds read straight from the curated units NWB plus the
        ``curation_label`` lists joined from ``UnitLabel``.

        A curation created with ``apply_merge=False`` returns its UNMERGED
        preview units here -- the proposed merges live in ``MergeGroup`` and
        are applied only by ``get_merged_sorting``. Because consumers
        (SortedSpikesGroup / decoding) read through this method, a warning is
        emitted in that case so the proposed merges are not silently ignored.

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

        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        recording_row = cls._upstream_recording_row(key)
        fs = float(recording_row["sampling_frequency"])

        # A curation created with apply_merge=False records PROPOSED merges in
        # MergeGroup but does NOT apply them: get_sorting (what consumers such
        # as SortedSpikesGroup / decoding read via SpikeSortingOutput) returns
        # the UNMERGED preview units. Warn here so ad-hoc inspection is not
        # silently misled; the decoding CONSUMERS additionally RAISE via
        # SpikeSortingOutput.assert_decoding_merge_ids_ok so a preview curation
        # never reaches a decode.
        if cls.has_unapplied_proposed_merges(key):
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
        # returned DataFrame mirrors v1's ``Curation.fetch_nwb`` shape
        # (``v1/curation.py:197-209`` returns ``nwbf.units.to_dataframe()``
        # which carries the ``curation_label`` column). External
        # notebook code reading ``df["curation_label"]`` works on v2
        # rows without poking at the part table directly.
        unit_ids = list(abs_times)
        label_rows = (cls.UnitLabel & key).fetch(
            "unit_id", "curation_label", as_dict=True
        )
        labels_by_unit: dict[int, list[str]] = {}
        for lr in label_rows:
            labels_by_unit.setdefault(int(lr["unit_id"]), []).append(
                str(lr["curation_label"])
            )
        # Index by ``unit_id`` (matches v1's ``nwb.units.to_dataframe()``
        # shape, the same indexing Sorting.get_sorting(as_dataframe=True)
        # uses); see that method's docstring for the rationale.
        return pd.DataFrame(
            {
                "spike_times": [abs_times[u] for u in unit_ids],
                "curation_label": [labels_by_unit.get(u, []) for u in unit_ids],
            },
            index=pd.Index(unit_ids, name="unit_id"),
        )

    @classmethod
    def has_unapplied_proposed_merges(cls, key) -> bool:
        """True if a curation has a proposed merge that was NOT applied.

        A curation created with ``apply_merge=False`` records proposed merges in
        ``MergeGroup`` without applying them. A real merge is a group with more
        than one contributor; every unit also carries a 1-element self-entry, so
        a plain root curation (all self-entries) returns ``False``. Short-
        circuits on ``merges_applied`` so the ``MergeGroup`` fetch is skipped on
        the common already-applied / root path.
        """
        if bool((cls & key).fetch1("merges_applied")):
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

        Walks the v2 Selection tables and source parts
        (``RecordingSelection`` -> ``SortingSelection.RecordingSource`` ->
        ``SortingSelection`` -> ``ArtifactSource``) so a cross-table
        restriction over the v2 part-table convention keys --
        ``nwb_file_name``, ``team_name``, ``sort_group_id``,
        ``interval_list_name``, ``preproc_params_name``, ``recording_id``,
        ``artifact_id``, ``sorter``, ``sorter_params_name``, ``sorting_id``,
        ``curation_id`` -- resolves to the ``CurationV2`` rows it selects.

        This method is the SINGLE owner of v2's source-part join topology.
        ``SpikeSortingOutput._get_restricted_merge_ids_v2`` delegates here
        instead of re-implementing v2 schema knowledge in the merge master,
        so a new v2 source part (e.g. concat) is taught to exactly one
        place. Returns a ``CurationV2`` query (callers map it to merge ids
        via ``SpikeSortingOutput.CurationV2``), or ``None`` in lenient mode
        (``strict=False``) when the key names no v2 column.

        Unknown restriction keys raise ``ValueError`` when ``strict`` (the
        default -- a deliberate v2 query, where an unknown key is a typo, as
        v1 dropping bad keys quietly returned wrong-but-non-empty results);
        when ``strict=False`` (the multi-source ``get_restricted_merge_ids``
        dispatch) an unknown key instead returns ``None`` (the caller then
        contributes no v2 rows), since it names another pipeline's column and
        this is not a v2 query.
        ``restrict_by_artifact=True`` honors
        the v2 IntervalList convention where the artifact-removed
        valid_times row is named ``f"artifact_{artifact_id}"``; callers can
        supply either the bare artifact id or the artifact-named
        IntervalList and both resolve. ``artifact_id=None`` means "no
        artifact pass" (anti-join to sorts with no ``ArtifactSource`` row),
        NOT "match anything" -- only an absent key is a wildcard.
        """
        import uuid

        from spyglass.spikesorting.v2.recording import RecordingSelection
        from spyglass.spikesorting.v2.sorting import SortingSelection
        from spyglass.spikesorting.v2.utils import (
            parse_artifact_interval_list_name,
        )
        from spyglass.utils import logger

        rec_keys = [
            "nwb_file_name",
            "team_name",
            "sort_group_id",
            "interval_list_name",
            "preproc_params_name",
            "recording_id",
        ]
        sort_keys = [
            "sorter",
            "sorter_params_name",
            "sorting_id",
            "artifact_id",
        ]
        curation_keys = ["curation_id"]
        allowed = set(rec_keys) | set(sort_keys) | set(curation_keys)
        unknown = set(key) - allowed
        if unknown:
            if not strict:
                # Lenient multi-source-dispatch path: an unknown key names a
                # non-v2 (v0/v1) column, so this is not a v2 query. Return
                # ``None`` so the caller contributes no v2 rows -- mirroring
                # how the v0/v1 resolvers drop keys their tables lack. The
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
        # ``restrict_by_artifact`` maps the artifact-named IntervalList
        # convention (``f"artifact_{artifact_id}"``) back to the
        # ``artifact_id`` master-side column so the v2 join chain
        # downstream resolves correctly.
        if restrict_by_artifact and "interval_list_name" in key:
            artifact_id = parse_artifact_interval_list_name(
                key["interval_list_name"]
            )
            if artifact_id is not None:
                key["artifact_id"] = artifact_id
                key.pop("interval_list_name", None)
            elif "artifact_id" not in key:
                # The caller asked to restrict by artifact, but the
                # interval name is not the ``artifact_{uuid}`` form and no
                # artifact_id was supplied, so there is nothing to map.
                # Warn instead of silently returning unrestricted ids.
                logger.warning(
                    "CurationV2.resolve_restriction: "
                    "restrict_by_artifact=True but interval_list_name "
                    f"{key['interval_list_name']!r} is not an "
                    "'artifact_{uuid}' interval and no artifact_id was "
                    "given; v2 results will NOT be artifact-restricted. "
                    "Pass artifact_id=... or use the artifact-named "
                    "interval to restrict."
                )

        # ``parse_artifact_interval_list_name`` returns a str and a caller may
        # pass either a str or a UUID, but ``artifact_id`` is a uuid column.
        # Normalize to a ``uuid.UUID`` so the ArtifactSource intersection below
        # is unambiguous and a malformed id fails fast here rather than
        # silently matching nothing.
        if key.get("artifact_id") is not None:
            key["artifact_id"] = uuid.UUID(str(key["artifact_id"]))

        rec_restriction = {k: key[k] for k in rec_keys if k in key}
        rec_table = RecordingSelection & rec_restriction

        sort_rec_source = SortingSelection.RecordingSource * rec_table.proj()
        sort_master = SortingSelection * sort_rec_source.proj()
        # ``artifact_id`` lives on the optional ``SortingSelection.
        # ArtifactSource`` part, NOT on ``SortingSelection`` -- it is absent
        # from ``sort_master``'s heading, so a dict restriction with it is
        # silently dropped (verified: the compiled SQL is identical with and
        # without the key). Apply the non-artifact sort keys directly, then
        # resolve ``artifact_id`` through the part. In the v2 design the
        # presence/absence of an ``ArtifactSource`` row IS the artifact state,
        # so ``artifact_id=None`` means "no artifact pass" (anti-join to sorts
        # with no part row), NOT "match anything" -- only an absent key is a
        # wildcard.
        sort_restriction = {
            k: key[k] for k in sort_keys if k in key and k != "artifact_id"
        }
        sort_master = sort_master & sort_restriction
        if "artifact_id" in key:
            if key["artifact_id"] is None:
                sort_master = (
                    sort_master - SortingSelection.ArtifactSource.proj()
                )
            else:
                artifact_source = SortingSelection.ArtifactSource & {
                    "artifact_id": key["artifact_id"]
                }
                sort_master = sort_master & artifact_source.proj()

        curation_restriction = {k: key[k] for k in curation_keys if k in key}
        return (cls * sort_master.proj("sorting_id")) & curation_restriction

    @classmethod
    def get_sort_metadata(cls, key) -> tuple:
        """Return ``(sorter, nwb_file_name)`` for a curation's underlying sort.

        Owns the v2 source-part walk (``SortingSelection`` ->
        ``SortingSelection.RecordingSource`` -> ``RecordingSelection``) so
        consumers -- e.g. decoding's ``UnitWaveformFeatures`` -- don't
        re-implement v2's join topology. ``key`` must carry ``sorting_id``
        (sorter and nwb_file_name are fixed per sort, independent of
        ``curation_id``).
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection
        from spyglass.spikesorting.v2.sorting import SortingSelection

        joined = (
            SortingSelection
            * SortingSelection.RecordingSource
            * RecordingSelection
        ) & {"sorting_id": key["sorting_id"]}
        return joined.fetch1("sorter", "nwb_file_name")

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

        Matches v1's semantic at ``v1/curation.py:228-266``: a curation
        built with ``apply_merge=False`` (preview) still carries every
        original unit, so the proposed merges recorded in
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
        row = (cls & key).fetch1()
        recording_row = cls._upstream_recording_row(key)
        fs = float(recording_row["sampling_frequency"])
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
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
            If provided, restrict to units carrying at least one of
            these labels. Defaults to ``None`` (all units).
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
                lbl.value if isinstance(lbl, CurationLabel) else str(lbl)
                for lbl in include_labels
            }
            labeled = (
                self.UnitLabel
                & key
                & [{"curation_label": v} for v in include_values]
            ).fetch("unit_id")
            unit_restriction = unit_restriction & [
                {"unit_id": int(uid)} for uid in labeled
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

    @classmethod
    def get_sort_group_info(cls, key: dict) -> "dj.Table":
        """Return ALL electrodes in the sort group joined to BrainRegion.

        Fix for the v1 ``fetch(limit=1)`` multi-region under-reporting
        bug: this returns a DataJoint relation (not a DataFrame, not
        single-row) covering EVERY electrode in the sort group so a
        multi-region probe surfaces every represented region. Callers
        can chain restrictions / fetches on the returned relation.

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
            BrainRegion``) covering every electrode in the sort group.
        """
        from spyglass.common.common_ephys import Electrode as _Electrode
        from spyglass.common.common_region import BrainRegion
        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
            SortGroupV2,
        )

        sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
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
            (SortGroupV2.SortGroupElectrode & sg_restriction)
            * _Electrode
            * BrainRegion
        )

    @classmethod
    def _upstream_recording_row(cls, key) -> dict:
        """Fetch the upstream Recording row for a CurationV2 key.

        Used by ``get_sorting`` to recover the recording's
        sampling-frequency and first-timestamp metadata (matching
        the ``Sorting.get_sorting`` round-trip convention).
        ``@classmethod`` so it can be invoked from the other
        classmethod accessors.

        ``key`` may be a single dict or the list-of-dict form the
        merge dispatcher passes; the restriction-based fetch
        normalizes both.
        """
        from spyglass.spikesorting.v2.recording import Recording

        sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        return (Recording & source.key).fetch1()
