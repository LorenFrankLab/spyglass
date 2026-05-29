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

        A unit may carry multiple labels (e.g. ``mua`` + ``artifact``);
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
            ints. The merged unit inherits the peak channel + amplitude
            of the highest-amplitude contributor; its ``unit_id`` is
            the first id in the group. Non-listed units pass through
            1:1.
        apply_merge
            If True, the curated-units NWB stores merged spike trains
            (union of contributors); if False (default), it stores the
            original Sorting spike trains 1:1 so merge edits can be
            reviewed before committing. CurationV2.Unit always reflects
            the post-merge unit set regardless. v1 spelling matches
            ``CurationV1.insert_curation`` at ``v1/curation.py:50``
            so existing v1 caller code keeps working unchanged.
        description
            Free-text curation description.
        metrics_source
            Provenance for any attached metrics blob. Must be one of
            'manual' (default), 'analyzer_curation', or 'figpack'.
        permissive_labels
            If False (default), labels keyed by a unit_id not present in
            the sorting raise ``ValueError`` -- a stray key (usually a
            typo) would otherwise be dropped silently from the curated
            result. Pass True for best-effort labeling that warns and
            drops the stray keys instead.

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
            # ``None`` is semantically equivalent to "no labels" per
            # v1's ``CurationV1.insert_curation`` signature at
            # ``v1/curation.py:49``. Normalize to ``{}`` so the rest
            # of the helper does not need an extra None check.
            labels = {}
        # Normalize label keys to int once so the rest of the helper
        # can do straight ``labels.get(int_uid, [])`` lookups.
        labels = {int(uid): list(lbls) for uid, lbls in labels.items()}

        cls._validate_labels(labels)

        # Labels keyed by unit_ids not in the sorting are usually a typo.
        # ``_stage_curated_units_nwb`` only writes labels for kept units
        # (``labels.get(kept_uid, [])``), so a stray key would silently
        # vanish from the curated result. Raise by default; callers who
        # genuinely want best-effort labeling pass permissive_labels=True
        # to restore the warn-and-drop behavior.
        valid_unit_ids = {int(r["unit_id"]) for r in sorting_units}
        stray_label_ids = set(labels) - valid_unit_ids
        if stray_label_ids:
            message = (
                "CurationV2.insert_curation: labels reference unit_id(s) "
                f"{sorted(stray_label_ids)} not in Sorting.Unit for "
                f"sorting_id={sorting_id}. Valid unit_ids: "
                f"{sorted(valid_unit_ids)}."
            )
            if not permissive_labels:
                raise ValueError(
                    f"{message} Pass permissive_labels=True to ignore "
                    "stray labels instead of raising."
                )
            logger.warning(f"{message} Those labels are ignored.")

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

            # Persist per-unit merge provenance for queryable
            # bulk-audit. ``kept_unit_to_contributors`` already includes
            # 1-unit entries for unmerged units (built in
            # ``_build_curated_unit_rows``), so every ``CurationV2.Unit``
            # row has at least one matching ``MergeGroup`` row.
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
                    cls.UnitLabel.insert(unit_label_rows)
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
        apply_merge: bool,
    ) -> tuple[list[dict], dict[int, list[int]]]:
        """Resolve the post-merge ``CurationV2.Unit`` rows.

        Returns ``(unit_rows, kept_unit_to_contributors)``. Each kept
        unit's Electrode FK + peak amplitude come from the contributor
        with the largest ``peak_amplitude_uv``; merged-unit ``unit_id``
        is the first id in the group, matching the v1 convention.

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
        if not sorting_units:
            return [], {}
        by_id = {int(row["unit_id"]): row for row in sorting_units}

        # Build mapping from contributor -> kept unit (head of group).
        # A unit may only appear in ONE merge group: overlap across
        # groups would silently double-count spikes when apply_merge
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
            # Merged sum only when the merged train is actually staged;
            # otherwise the head unit's own count (matches the NWB write).
            if apply_merge and len(contribs) > 1:
                n_spikes = sum(by_id[u]["n_spikes"] for u in contribs)
            else:
                n_spikes = by_id[int(kept_uid)]["n_spikes"]
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
        apply_merge: bool,
        labels: dict,
    ) -> tuple[str, str]:
        """Write the curated-units NWB. Returns (analysis_file_name, units_object_id).

        With ``apply_merge=True`` the kept unit's spike train is the
        union of its contributors' spike trains. With
        ``apply_merge=False`` only the non-merged units (and the
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
            if kept_unit_to_contributors:
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
                for kept_uid, contribs in kept_unit_to_contributors.items():
                    if apply_merge and len(contribs) > 1:
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
                        id=int(kept_uid),
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
        """
        from spikeinterface.extractors import NwbSortingExtractor

        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

        # CurationV2's units NWB does not carry an ElectricalSeries, so
        # we must supply ``sampling_frequency`` and ``t_start`` directly
        # (mirrors ``Sorting.get_sorting``). t_start is the recording's
        # first wall-clock timestamp.
        recording_row = cls._upstream_recording_row(key)
        fs = float(recording_row["sampling_frequency"])
        t_start = Sorting._recording_t_start(recording_row)

        if len(cls.Unit & key) == 0:
            # Zero-unit curations are valid (a user may curate a
            # zero-unit sort; the Empty/Boundary invariant allows an
            # empty ``CurationV2.Unit``). ``NwbSortingExtractor`` cannot
            # open the empty curated-units NWB, so return an empty
            # ``NumpySorting`` -- mirrors ``Sorting.get_sorting``. This
            # accessor builds its OWN extractor (it does NOT delegate to
            # ``Sorting.get_sorting``), so it needs the same guard.
            import spikeinterface as si

            logger.warning(
                "CurationV2.get_sorting: curation "
                f"(sorting_id={row['sorting_id']}, "
                f"curation_id={row['curation_id']}) has zero units; "
                "returning an empty sorting."
            )
            si_sorting = si.NumpySorting.from_unit_dict(
                {}, sampling_frequency=fs
            )
        else:
            si_sorting = NwbSortingExtractor(
                file_path=abs_path, sampling_frequency=fs, t_start=t_start
            )
        if not as_dataframe:
            return si_sorting

        import pandas as pd

        # Join the ``curation_label`` lists from ``UnitLabel`` so the
        # returned DataFrame mirrors v1's ``Curation.fetch_nwb`` shape
        # (``v1/curation.py:197-209`` returns ``nwbf.units.to_dataframe()``
        # which carries the ``curation_label`` column). External
        # notebook code reading ``df["curation_label"]`` works on v2
        # rows without poking at the part table directly.
        unit_ids = [int(uid) for uid in si_sorting.unit_ids]
        label_rows = (cls.UnitLabel & key).fetch(
            "unit_id", "curation_label", as_dict=True
        )
        labels_by_unit: dict[int, list[str]] = {}
        for lr in label_rows:
            labels_by_unit.setdefault(int(lr["unit_id"]), []).append(
                str(lr["curation_label"])
            )
        spike_times = [
            si_sorting.get_unit_spike_train(
                unit_id=uid, return_times=True
            )
            for uid in si_sorting.unit_ids
        ]
        # Index by ``unit_id`` (matches v1's ``nwb.units.to_dataframe()``
        # shape, the same indexing Sorting.get_sorting(as_dataframe=True)
        # uses); see that method's docstring for the rationale.
        return pd.DataFrame(
            {
                "spike_times": spike_times,
                "curation_label": [
                    labels_by_unit.get(uid, []) for uid in unit_ids
                ],
            },
            index=pd.Index(unit_ids, name="unit_id"),
        )

    @classmethod
    def get_merge_groups(cls, key) -> dict[int, list[int]]:
        """Return ``{kept_unit_id: [contributor_unit_id, ...]}`` for a curation.

        Reads the ``CurationV2.MergeGroup`` part rows and groups
        them by kept unit. Every ``CurationV2.Unit`` row has at
        least one ``MergeGroup`` entry -- unmerged units carry
        themselves as their sole contributor. This makes "did unit
        X involve a merge?" a simple ``len(merge_groups[X]) > 1``
        check.

        Used by ``get_merged_sorting`` and exposed publicly so
        users can do bulk-audit queries directly.
        """
        rows = (cls.MergeGroup & key).fetch(
            "unit_id", "contributor_unit_id", as_dict=True
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

        Matches v1's semantic at ``v1/curation.py:228-266``: merges
        are queried from ``CurationV2.MergeGroup`` and applied
        lazily via ``spikeinterface.curation.MergeUnitsSorting``,
        regardless of the ``merges_applied`` flag on the master
        row. Code that built a curation with ``apply_merge=False``
        (to preview) and then asked for the merged sorting now
        actually gets the merged trains.

        If no merge group has more than one contributor (i.e. no
        merge was requested), returns the base sorting verbatim
        without invoking ``MergeUnitsSorting``.
        """
        import spikeinterface.curation as sc

        base = cls.get_sorting(key)
        merge_groups = cls.get_merge_groups(key)
        units_to_merge = [
            contribs
            for kept_uid, contribs in merge_groups.items()
            if len(contribs) > 1
        ]
        if not units_to_merge:
            return base
        return sc.MergeUnitsSorting(
            parent_sorting=base, units_to_merge=units_to_merge
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
