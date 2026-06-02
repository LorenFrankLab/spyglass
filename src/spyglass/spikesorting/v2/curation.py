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

from collections.abc import Mapping

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


def _validate_curation_label_rows(
    rows: list, allow_custom_labels: bool = False
) -> None:
    """Reject ``UnitLabel`` rows whose ``curation_label`` is not canonical.

    Validates the ``curation_label`` on each row dict against the
    ``CurationLabel`` set so a direct ``CurationV2.UnitLabel.insert1`` /
    ``insert`` cannot slip a typo'd label past the Python-side guard the
    way a bare ``varchar`` column would (DataJoint does not enforce the
    open-ended label set at the DB; see ``CurationLabel``).
    ``allow_custom_labels=True`` accepts labels outside the canonical set
    (labs using custom semantics). Rows without a ``curation_label`` key
    are left alone -- DataJoint raises its own missing-attribute error
    for those.
    """
    valid = {member.value for member in CurationLabel}
    for row in rows:
        # DataJoint accepts ordered (tuple/list) rows too, but this guard
        # reads ``curation_label`` by key -- a membership test that is
        # False for an ordered row, which would silently skip validation
        # and let a bogus varchar label through. Require a mapping so the
        # label is always inspected (validate on every insert path).
        if not isinstance(row, Mapping):
            raise ValueError(
                "CurationV2.UnitLabel insert requires mapping (dict) rows "
                "so curation_label can be validated; got an ordered "
                f"{type(row).__name__}. Pass a dict like "
                "{'sorting_id': ..., 'curation_id': ..., 'unit_id': ..., "
                "'curation_label': ...}."
            )
        if allow_custom_labels:
            continue
        if "curation_label" not in row:
            continue
        label = row["curation_label"]
        label_value = (
            label.value if isinstance(label, CurationLabel) else label
        )
        if label_value not in valid:
            raise ValueError(
                f"CurationV2.UnitLabel: curation_label {label!r} is not in "
                f"CurationLabel. Valid labels: {sorted(valid)}. Pass "
                "allow_custom_labels=True to accept labels outside the "
                "canonical set."
            )


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

        def insert1(
            self, row, *, allow_custom_labels: bool = False, **kwargs
        ):
            # Delegate to ``insert`` (as DataJoint's own ``insert1`` does:
            # ``self.insert((row,))``) so the single validation happens in
            # one place AND ``allow_custom_labels`` survives the dispatch.
            # Validating here then calling ``super().insert1`` would lose
            # the flag, because DataJoint's internal ``insert1 -> insert``
            # hop re-enters this override with the default False.
            self.insert(
                [row], allow_custom_labels=allow_custom_labels, **kwargs
            )

        def insert(
            self, rows, *, allow_custom_labels: bool = False, **kwargs
        ):
            rows = list(rows)
            _validate_curation_label_rows(
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

        ``contributor_unit_id`` is a plain int, NOT a DataJoint FK:
        the schema cannot express a nullable self-FK across the
        renamed unit columns cleanly. ``insert_curation`` validates
        every contributor id against the sorting's units, but a
        direct part-table insert is NOT FK-checked -- treat this as
        helper-maintained provenance, not a schema-enforced relation.

        One row per ``(kept_unit_id, contributor_unit_id)``. The
        kept unit appears as its own contributor for unmerged
        units (one row, ``contributor_unit_id == unit_id``); this
        makes "list every unit's merge provenance" a single
        restriction without special-casing the no-merge units.
        """

        definition = """
        -> CurationV2.Unit
        contributor_unit_id: int     # source unit before the merge
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
            fresh id, ``max(source unit_ids) + 1``, sequentially
            incremented per group in USER-PROVIDED order -- v1 parity
            (``v1/curation.py:359``). NOTE: the lazy merge path
            (``get_merged_sorting`` on an apply_merge=False preview) reads
            MergeGroup ordered by ``unit_id`` and so iterates by
            kept-uid-ascending; when input order != kept-uid order,
            applied and lazy paths assign the same fresh ids to different
            content groups.
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
            forwarded to both ``_validate_labels`` and the
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
        sorting_units = (
            Sorting.Unit & {"sorting_id": sorting_id}
        ).fetch(as_dict=True, order_by="unit_id")
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

        cls._validate_labels(labels, allow_custom_labels=allow_custom_labels)

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
            existing_root = (
                cls
                & {
                    "sorting_id": sorting_id,
                    "parent_curation_id": -1,
                }
            ).fetch("KEY", as_dict=True)
            if existing_root:
                # A root curation already exists. If the caller passed
                # non-default labels / merge_groups / description, those
                # would be SILENTLY ignored by returning the existing
                # row -- a parameter-change-with-no-effect footgun. Raise
                # unless the caller explicitly opts into reuse.
                if (
                    bool(labels) or bool(merge_groups) or bool(description)
                ) and not reuse_existing:
                    raise ValueError(
                        "CurationV2.insert_curation: a root curation "
                        f"already exists for sorting_id={sorting_id}, but "
                        "you passed labels/merge_groups/description that "
                        "would be silently ignored. Pass "
                        "reuse_existing=True to reuse the existing root, "
                        "or curate as a child with "
                        "parent_curation_id=<existing root curation_id>."
                    )
                logger.warning(
                    "CurationV2.insert_curation: root curation already "
                    f"exists for sorting_id={sorting_id}; returning "
                    "existing key without staging a new NWB."
                )
                return existing_root[0]

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
        ) = cls._stage_curated_units_nwb(
            sorting_id=sorting_id,
            kept_unit_to_contributors=kept_unit_to_contributors,
            apply_merge=apply_merge,
            labels=labels,
        )

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
                    # ``_validate_labels``; forward ``allow_custom_labels``
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

    # ---- Validation helpers ---------------------------------------------

    @staticmethod
    def _validate_labels(
        labels: dict, allow_custom_labels: bool = False
    ) -> None:
        """Validate that every value in ``labels`` is a recognized label.

        Accepts ``CurationLabel`` instances OR their string values; any
        other label raises ``ValueError`` listing the offending entry
        and the valid label names. Pass ``allow_custom_labels=True`` to
        accept labels outside the canonical ``CurationLabel`` set (labs
        tagging units with custom semantics); the list-shape check on
        each ``labels[unit_id]`` value still applies either way.
        """
        valid = {member.value for member in CurationLabel}
        for unit_id, lbls in labels.items():
            # Defense-in-depth: ``insert_curation`` already rejects a
            # non-list/tuple label value before coercing to list, so this
            # branch is normally unreachable from that path -- it guards
            # any future direct caller of ``_validate_labels``.
            if not isinstance(lbls, (list, tuple)):
                raise ValueError(
                    "CurationV2.insert_curation: labels[unit_id] must be "
                    f"a list of labels; got {type(lbls).__name__} for "
                    f"unit_id={unit_id}."
                )
            if allow_custom_labels:
                continue
            for lbl in lbls:
                label_value = (
                    lbl.value if isinstance(lbl, CurationLabel) else lbl
                )
                if label_value not in valid:
                    raise ValueError(
                        f"CurationV2.insert_curation: label {lbl!r} for "
                        f"unit_id={unit_id} is not in CurationLabel. "
                        f"Valid labels: {sorted(valid)}. Pass "
                        "allow_custom_labels=True to accept labels outside "
                        "the canonical set."
                    )

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

        Returns ``(unit_rows, kept_unit_to_contributors)``. Each kept
        unit's Electrode FK + peak amplitude come from the contributor
        with the largest ``peak_amplitude_uv``. The merged-unit
        ``unit_id`` depends on ``apply_merge``: a fresh
        ``max(source unit_ids) + 1`` for ``apply_merge=True`` (v1 parity,
        ``v1/curation.py:361``); ``min(group)`` for ``apply_merge=False``
        (the proposed merge leader, kept as a regular Unit row alongside
        the absorbed contributors so the preview retains every original
        unit). Each merge group must have at least 2 members; empty or
        singleton groups raise ``ValueError``.

        ``sorting_units`` is the pre-fetched ``Sorting.Unit`` rows for
        ``sorting_id``; the caller fetches once and threads through to
        avoid re-querying.

        ``n_spikes`` is computed to match what ``_stage_curated_units_nwb``
        writes for the SAME ``apply_merge``: the merged sum only when the
        merged spike train is actually staged (``apply_merge and
        len(contribs) > 1``), otherwise the kept (head) unit's own count.
        This keeps the invariant ``CurationV2.Unit.n_spikes ==
        len(get_sorting().get_unit_spike_train(kept_uid))`` for BOTH
        ``apply_merge`` values -- otherwise an ``apply_merge=False``
        preview would store the merged sum against a head-only train.
        """
        by_id = {int(row["unit_id"]): row for row in sorting_units}

        # Build mapping ``kept_unit_id -> contributors``:
        #   apply_merge=True multi-contributor group: kept id is a fresh
        #     ``max(source unit_ids) + 1`` (sequentially incremented per
        #     multi-merge group), matching v1's
        #     ``np.max(units_dict.keys()) + 1`` pattern; assignment is
        #     in USER-PROVIDED group order (v1's iteration order). The
        #     user's first element is one of the contributors, NOT the
        #     kept id.
        #   apply_merge=False: kept id is ``min(group)`` -- the proposed
        #     merge leader recorded in MergeGroup until the merge is
        #     applied via get_merged_sorting.
        # A unit may appear in at most one merge group: the overlap
        # check below would otherwise silently double-count spikes when
        # apply_merge=True and ambiguate the kept-unit choice.
        # Validate merge groups and stage their (key, contributors) pairs
        # WITHOUT inserting them yet -- we want kept_to_contributors's
        # insertion order to be "surviving source units first (in
        # original source order), then merged kept ids appended in
        # merge-group order." That matches v1's pop-then-append pattern
        # (``v1/curation.py:359``) and SI's lazy MergeUnitsSorting
        # (originals retained + merged appended) at the SHAPE level
        # (survivor-then-appended). The content-to-id mapping can still
        # diverge between applied and lazy when input order differs from
        # kept-uid order -- see the NOTE in the next block.
        # Iterate merge groups in USER-PROVIDED order so new-id
        # assignment matches v1 (``v1/curation.py:359`` iterates
        # ``for merge_group in merge_groups:`` and assigns the next
        # ``max(unit_ids) + 1`` per group). NOTE: the lazy merge path
        # (``get_merged_sorting`` on an apply_merge=False preview) reads
        # MergeGroup ordered by ``unit_id`` and so iterates by
        # kept-uid-ascending; when the user-given group order differs
        # from kept-uid-ascending, applied (insert-time) and lazy paths
        # produce different new ids for the same content groups. v1
        # parity for the applied path is the explicit goal.
        normalized_groups: list[list[int]] = [
            [int(u) for u in g] for g in merge_groups
        ]
        # Validate ALL merge groups eagerly -- BEFORE any early return
        # below -- so a zero-unit sort with non-empty merge_groups raises
        # rather than silently no-op. Checks: shape (>=2 members),
        # intra-group uniqueness (no double-counting), id membership in
        # by_id, and across-group overlap. ``merged_ids`` is populated
        # here and reused by the non-merged-units loop later.
        merged_ids: set[int] = set()
        for int_group in normalized_groups:
            if len(int_group) < 2:
                # Empty or singleton "merge groups" aren't merges. v1
                # would either no-op (empty) or silently rename the
                # singleton to max+1; surface the likely typo instead.
                raise ValueError(
                    f"CurationV2.insert_curation: merge_groups contains "
                    f"a group with fewer than 2 members ({int_group}); "
                    "each merge group must have at least 2 units."
                )
            if len(set(int_group)) != len(int_group):
                # ``len`` is list-length, not set-size: a group like
                # [0, 0] passes the >=2 check but would double-count
                # contributor 0 during staging.
                raise ValueError(
                    f"CurationV2.insert_curation: merge_groups contains "
                    f"a group with duplicate members ({int_group}); "
                    "each unit can appear at most once per merge group."
                )
            for uid in int_group:
                if uid not in by_id:
                    raise ValueError(
                        f"CurationV2.insert_curation: merge_groups "
                        f"references unit_id={uid} that is not in "
                        f"Sorting.Unit for sorting_id={sorting_id}."
                    )
            overlap = merged_ids & set(int_group)
            if overlap:
                raise ValueError(
                    f"CurationV2.insert_curation: merge_groups overlap "
                    f"on unit_ids {sorted(overlap)}. A unit can belong "
                    "to at most one merge group."
                )
            merged_ids.update(int_group)

        if not by_id:
            # Zero-unit sort with no merge groups -> empty curation.
            # (Non-empty merge groups already raised above.)
            return [], {}

        # All groups validated; assign keys.
        merge_specs: list[tuple[int, list[int]]] = []
        next_merged_id = max(by_id) + 1
        for int_group in normalized_groups:
            if apply_merge and len(int_group) > 1:
                key = next_merged_id
                next_merged_id += 1
            else:
                key = min(int_group)
            merge_specs.append((key, int_group))

        kept_to_contributors: dict[int, list[int]] = {}
        # Non-merged units FIRST, in original source order.
        for uid in by_id:
            if uid not in merged_ids:
                kept_to_contributors[uid] = [uid]
        # Then merge groups, appended in input order.
        for key, int_group in merge_specs:
            kept_to_contributors[key] = int_group

        # Symmetric provenance for apply_merge=False: absorbed
        # contributors still get a CurationV2.Unit row (preview parity),
        # so give each a 1-element self-entry so every Unit row has at
        # least one MergeGroup row with its own ``unit_id`` -- a
        # ``Unit * MergeGroup`` join on unit_id no longer drops them.
        # get_merged_sorting filters ``len(contribs) > 1`` so the
        # self-entries are auto-skipped when reconstructing merges.
        if not apply_merge:
            for uid in by_id:
                if uid not in kept_to_contributors:
                    kept_to_contributors[uid] = [uid]

        # The written unit set depends on apply_merge:
        #   apply_merge=True  -> kept units only; a merged head absorbs
        #     its contributors (peak channel inherited from the
        #     highest-amplitude contributor, n_spikes = summed train).
        #   apply_merge=False -> every original unit passes through 1:1
        #     (v1 preview parity, ``v1/curation.py:359``); the proposed
        #     merges live in MergeGroup for lazy application via
        #     get_merged_sorting. n_spikes is each unit's own count,
        #     matching the train _stage_curated_units_nwb writes.
        if apply_merge:
            specs = []
            for kept_uid, contribs in kept_to_contributors.items():
                anchor = max(
                    contribs, key=lambda u: by_id[u]["peak_amplitude_uv"]
                )
                n_spikes = (
                    sum(by_id[u]["n_spikes"] for u in contribs)
                    if len(contribs) > 1
                    else by_id[int(kept_uid)]["n_spikes"]
                )
                specs.append((int(kept_uid), by_id[anchor], int(n_spikes)))
        else:
            specs = [
                (int(uid), row, int(row["n_spikes"]))
                for uid, row in by_id.items()
            ]
        unit_rows = [
            {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "unit_id": unit_id,
                "nwb_file_name": src["nwb_file_name"],
                "electrode_group_name": src["electrode_group_name"],
                "electrode_id": int(src["electrode_id"]),
                "peak_amplitude_uv": float(src["peak_amplitude_uv"]),
                "n_spikes": n_spikes,
            }
            for unit_id, src, n_spikes in specs
        ]
        return unit_rows, kept_to_contributors

    @classmethod
    def _stage_curated_units_nwb(
        cls,
        sorting_id,
        kept_unit_to_contributors: dict,
        apply_merge: bool,
        labels: dict,
    ) -> tuple[str, str]:
        """Write the curated-units NWB. Returns (analysis_file_name, units_object_id).

        With ``apply_merge=True`` the kept unit's spike train is the
        sorted union of its contributors' spike trains and its id is a
        fresh ``max(source unit_ids) + 1`` assigned in USER-PROVIDED
        group order (v1 parity); the absorbed contributors are dropped
        from both the NWB and ``CurationV2.Unit``. Surviving source units
        are written first (in source order) and merged ids are appended,
        matching v1's pop-then-append per-unit layout.

        With ``apply_merge=False`` (preview) every original unit is
        written 1:1 -- contributors included -- so the proposed merge
        can be reviewed before committing; the merge structure lives in
        ``CurationV2.MergeGroup`` and is reconstructed by
        ``get_merged_sorting`` on demand.
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

        # Source the pre-curation units' ABSOLUTE spike times straight
        # from the Sorting units NWB. Reading absolute seconds (not a
        # frame-based SI object) keeps the curated NWB's stored times
        # exact and gap-correct for disjoint recordings -- a frame->time
        # round-trip through a NumpySorting (t_start=0) would drop the
        # absolute offset and the wall-clock gaps.
        src_abs_path = AnalysisNwbfile.get_abs_path(
            (Sorting & {"sorting_id": sorting_id}).fetch1(
                "analysis_file_name"
            )
        )
        abs_times_by_uid = Sorting._read_units_abs_spike_times(src_abs_path)

        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name=nwb_file_name
        )
        analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # Add the ``curation_label`` column ONLY when at least one
            # unit will be written. Adding a column ahead of any
            # ``add_unit`` call would create an empty Units table whose
            # ``curation_label`` column has no rows; pynwb's writer
            # then fails dtype inference at ``io.write`` with
            # "Cannot infer dtype of empty list or tuple". For the
            # empty-curation case (zero kept units after filtering, or
            # the contrived all-merged-away case) we initialize a
            # bare ``pynwb.misc.Units`` without the column so the
            # write succeeds.
            # Resolve which units get written + their spike trains:
            #   apply_merge=True  -> kept units; a merged head gets the
            #     concatenated contributor trains.
            #   apply_merge=False -> every original unit 1:1 (v1 preview
            #     parity, ``v1/curation.py:359``); proposed merges stay in
            #     MergeGroup for lazy application via get_merged_sorting.
            if apply_merge:
                write_specs = []
                for kept_uid, contribs in kept_unit_to_contributors.items():
                    if len(contribs) > 1:
                        spike_times = _np.concatenate(
                            [abs_times_by_uid[int(u)] for u in contribs]
                        )
                        spike_times.sort()
                    else:
                        spike_times = abs_times_by_uid[int(kept_uid)]
                    write_specs.append((int(kept_uid), spike_times))
            else:
                write_specs = [
                    (int(uid), abs_times_by_uid[int(uid)])
                    for uid in sorted(abs_times_by_uid)
                ]

            if write_specs:
                # ``curation_label`` is written as an ``index=True``
                # (ragged) column with a per-unit list of label
                # strings, matching v1's ``v1/curation.py:398-403``
                # shape. External readers (e.g.
                # ``v1/figurl_curation.py:83-101`` which does
                # ``list(nwb_sorting.get('curation_label', []))``)
                # expect a list per unit and would misparse a
                # comma-separated string by splitting on every
                # character. The ``CurationV2.UnitLabel`` docstring
                # already described this shape.
                #
                # v1's pattern is to call ``add_unit(...)`` first,
                # then add the column with ``data=label_values``
                # AFTER -- this gives pynwb a full per-unit
                # list-of-lists to infer dtype from. If we
                # pre-declare the column and pass labels per
                # ``add_unit``, pynwb fails dtype inference when
                # all labels happen to be empty (the no-labels
                # case).
                all_labels: list[list[str]] = []
                for unit_id, spike_times in write_specs:
                    lbl_list = labels.get(int(unit_id), [])
                    label_list = [
                        lbl.value
                        if isinstance(lbl, CurationLabel)
                        else str(lbl)
                        for lbl in lbl_list
                    ]
                    all_labels.append(label_list)
                    nwbf.add_unit(
                        spike_times=_np.asarray(
                            spike_times, dtype=_np.float64
                        ),
                        id=int(unit_id),
                    )
                # Only add the column when at least one unit
                # carries a non-empty label list. pynwb's dtype
                # inference fails on an all-empty list-of-lists
                # ("Cannot infer dtype of empty list"); the
                # column-missing case is handled by downstream
                # readers via ``nwb_sorting.get('curation_label',
                # [])``.
                if any(all_labels):
                    nwbf.add_unit_column(
                        name="curation_label",
                        description=(
                            "Curation label list from "
                            "CurationV2.insert_curation; one entry "
                            "per label, empty list if unlabeled. "
                            "Indexed (ragged) column matching v1's "
                            "shape at v1/curation.py:398-403."
                        ),
                        data=all_labels,
                        index=True,
                    )
            else:
                # Empty curation: initialize an empty Units table so
                # ``.object_id`` is defined and ``io.write`` does not
                # try to infer a dtype for any column.
                nwbf.units = pynwb.misc.Units(
                    name="units",
                    description=(
                        "Empty units table (curation kept zero units)."
                    ),
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

    @classmethod
    def get_recording(cls, key):
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
        """
        from spyglass.spikesorting.v2.recording import Recording

        sorting_id = (cls & key).fetch1("sorting_id")
        source = SortingSelection.resolve_source(
            {"sorting_id": sorting_id}
        )
        if source.kind != "recording":
            raise NotImplementedError(
                "CurationV2.get_recording: concat-source sorts are not "
                "yet supported."
            )
        recording = Recording().get_recording(source.key)
        recording.annotate(is_filtered=True)
        return recording

    @classmethod
    def get_sorting(cls, key, as_dataframe: bool = False):
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
        """
        import spikeinterface as si

        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        recording_row = cls._upstream_recording_row(key)
        fs = float(recording_row["sampling_frequency"])

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
                return si.NumpySorting.from_unit_dict(
                    {}, sampling_frequency=fs
                )
            import pandas as pd

            return pd.DataFrame(
                {"spike_times": [], "curation_label": []},
                index=pd.Index([], name="unit_id", dtype=int),
            )

        abs_times = Sorting._read_units_abs_spike_times(abs_path)
        if not as_dataframe:
            return Sorting._numpysorting_from_abs_times(
                abs_times, recording_row, fs
            )

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
                "curation_label": [
                    labels_by_unit.get(u, []) for u in unit_ids
                ],
            },
            index=pd.Index(unit_ids, name="unit_id"),
        )

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
    def get_merged_sorting(cls, key):
        """Return the curated BaseSorting with merge groups applied.

        Matches v1's semantic at ``v1/curation.py:228-266``: a curation
        built with ``apply_merge=False`` (preview) still carries every
        original unit, so the proposed merges recorded in
        ``CurationV2.MergeGroup`` are applied lazily here via
        ``spikeinterface.curation.MergeUnitsSorting`` -- the caller gets
        the merged trains without re-running the sort.

        When the curation was created with ``apply_merge=True`` the base
        sorting is ALREADY merged (contributors absorbed at insert), so
        the MergeGroup contributors are no longer present in it; the base
        is returned verbatim rather than re-applying merges over missing
        units. Likewise returns the base unchanged when no merge group
        has more than one contributor.
        """
        import spikeinterface.curation as sc

        base = cls.get_sorting(key)
        if bool((cls & key).fetch1("merges_applied")):
            return base
        merge_groups = cls.get_merge_groups(key)
        units_to_merge = [
            contribs
            for kept_uid, contribs in merge_groups.items()
            if len(contribs) > 1
        ]
        if not units_to_merge:
            return base
        # SI 0.104: MergeUnitsSorting(sorting, units_to_merge, ...) --
        # ``sorting`` is positional (no ``parent_sorting`` kwarg).
        # ``delta_time_ms=None`` is the ONLY value that disables SI's
        # duplicate-spike check entirely (``delta_time_ms=0`` still drops
        # exact same-sample duplicates -- see
        # ``spikeinterface/curation/mergeunitssorting.py`` ``if
        # delta_time_ms is None: do_not_check`` semantics). The merged
        # train is the full concatenation of its contributors, matching
        # v1's ``np.concatenate`` and v2's own apply_merge=True staging.
        return sc.MergeUnitsSorting(
            base, units_to_merge=units_to_merge, delta_time_ms=None
        )

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

    @classmethod
    def get_sort_group_info(cls, key):
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
            SortGroupV2.SortGroupElectrode & sg_restriction
        ) * _Electrode * BrainRegion

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
