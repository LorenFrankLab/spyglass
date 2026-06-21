"""Pure curation transforms behind ``CurationV2``.

These three functions are the dependency-light core of v2 curation --
label-value validation, ``UnitLabel``-row validation, and the post-merge
``CurationV2.Unit`` row construction (merge-group validation,
``kept_unit_to_contributors`` mapping, and the per-unit row build). They
are pure Python: given the already-fetched ``Sorting.Unit`` rows and the
caller's label / merge-group payloads, they decide WHAT to insert without
touching the database.

Why this lives in its own module rather than in ``curation.py``:
``curation.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and pulls the merge-table / NWB dependencies. The
transforms here need none of that, so ``CurationV2`` becomes a thin
orchestrator (fetch ``Sorting.Unit`` -> call these -> insert) while the
parity-critical merge/label logic lives in one hermetically-testable
place. This is the same "thin DataJoint shell over pure services"
direction as ``_artifact_compute`` / ``_selection_identity`` /
``_analyzer_cache``.

DEPENDENCY-LIGHT BY CONTRACT. This module opens no database connection
and activates no ``dj.schema`` at import. Its own imports are limited to
the standard library plus the ``CurationLabel`` enum, taken from the
stdlib-only ``_enums`` module rather than through ``utils`` (which imports
DataJoint / SpikeInterface at load). So the transform layer itself does
not depend on the heavy table-support module. (Importing it as a
``spyglass`` submodule still triggers ``spyglass``'s package ``__init__``,
which loads DataJoint -- that is package-wide and unavoidable; the point
here is that these transforms add no DB/SpikeInterface dependency of their
own.)
"""

from __future__ import annotations

from collections.abc import Mapping

from spyglass.spikesorting.v2._enums import CurationLabel


def validate_curation_label_rows(
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
        label_value = CurationLabel.normalize(label)
        if label_value not in valid:
            raise ValueError(
                f"CurationV2.UnitLabel: curation_label {label!r} is not in "
                f"CurationLabel. Valid labels: {sorted(valid)}. Pass "
                "allow_custom_labels=True to accept labels outside the "
                "canonical set."
            )


def validate_labels(labels: dict, allow_custom_labels: bool = False) -> None:
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
        # any future direct caller of ``validate_labels``.
        if not isinstance(lbls, (list, tuple)):
            raise ValueError(
                "CurationV2.insert_curation: labels[unit_id] must be "
                f"a list of labels; got {type(lbls).__name__} for "
                f"unit_id={unit_id}."
            )
        if allow_custom_labels:
            continue
        for lbl in lbls:
            label_value = CurationLabel.normalize(lbl)
            if label_value not in valid:
                raise ValueError(
                    f"CurationV2.insert_curation: label {lbl!r} for "
                    f"unit_id={unit_id} is not in CurationLabel. "
                    f"Valid labels: {sorted(valid)}. Pass "
                    "allow_custom_labels=True to accept labels outside "
                    "the canonical set."
                )


def build_curated_unit_rows(
    sorting_id,
    sorting_units: list[dict],
    merge_groups: list[list[int]],
    curation_id: int,
    apply_merge: bool,
) -> tuple[list[dict], dict[int, list[int]]]:
    """Resolve the post-merge ``CurationV2.Unit`` rows.

    Each kept unit's Electrode FK + peak amplitude come from the
    contributor with the largest ``peak_amplitude_uv``. The merged-unit
    ``unit_id`` depends on ``apply_merge``: a fresh
    ``max(source unit_ids) + 1`` for ``apply_merge=True`` (v1 parity,
    ``v1/curation.py:361``); ``min(group)`` for ``apply_merge=False``
    (the proposed merge leader, kept as a regular Unit row alongside
    the absorbed contributors so the preview retains every original
    unit). Each merge group must have at least 2 members; empty or
    singleton groups raise ``ValueError``.

    ``n_spikes`` is computed to match what ``write_curated_units_nwb``
    writes for the SAME ``apply_merge``: the merged sum only when the
    merged spike train is actually staged (``apply_merge and
    len(contribs) > 1``), otherwise the kept (head) unit's own count.
    This keeps the invariant ``CurationV2.Unit.n_spikes ==
    len(get_sorting().get_unit_spike_train(kept_uid))`` for BOTH
    ``apply_merge`` values -- otherwise an ``apply_merge=False``
    preview would store the merged sum against a head-only train.

    Parameters
    ----------
    sorting_id
        ``sorting_id`` of the upstream Sorting row.
    sorting_units : list of dict
        Pre-fetched ``Sorting.Unit`` rows for ``sorting_id``; the
        caller fetches once and threads through to avoid re-querying.
    merge_groups : list of list of int
        Merge groups, each a list of ``unit_id`` ints (>=2 each).
    curation_id : int
        ``curation_id`` to stamp on the resulting rows.
    apply_merge : bool
        If True, build the merged (committed) unit set; if False, keep
        every original unit and record the proposed merges.

    Returns
    -------
    unit_rows : list of dict
        One ``CurationV2.Unit`` row per kept unit.
    kept_unit_to_contributors : dict[int, list[int]]
        ``{kept_unit_id: [contributor_unit_id, ...]}`` mapping.

    Raises
    ------
    ValueError
        If a merge group has fewer than 2 members, contains duplicate
        members, references a unit_id not in ``sorting_units``, or
        overlaps another merge group.
    """
    by_id = {int(row["unit_id"]): row for row in sorting_units}

    # Build mapping ``kept_unit_id -> contributors``:
    #   apply_merge=True multi-contributor group: kept id is a fresh
    #     ``max(source unit_ids) + 1`` (sequentially incremented per
    #     multi-merge group), matching v1's
    #     ``np.max(units_dict.keys()) + 1`` pattern. The user's first
    #     element is one of the contributors, NOT the kept id.
    #   apply_merge=False: kept id is ``min(group)`` -- the proposed
    #     merge leader recorded in MergeGroup until the merge is
    #     applied via get_merged_sorting.
    # A unit may appear in at most one merge group: the overlap
    # check below would otherwise silently double-count spikes when
    # apply_merge=True and ambiguate the kept-unit choice.
    #
    # CANONICAL-ORDER NOTE (preview==apply contract). The fresh
    # ``max+1`` ids are assigned in ascending MIN-CONTRIBUTOR order,
    # NOT user-provided order. This is deliberate: the lazy merge path
    # (``get_merged_sorting`` on an apply_merge=False preview) reads
    # MergeGroup ordered by ``unit_id`` and so numbers merges by
    # ascending kept-uid (== ascending min(group), since the preview
    # stores ``min(group)`` as each kept id). Numbering the applied
    # path the same way guarantees apply_merge=True and the lazy
    # preview assign the SAME fresh id to the SAME content group, even
    # when the user lists groups out of min order. v2 departs here from
    # v1's user-iteration-order labels (``v1/curation.py:359``): the
    # merged-unit integer id is an arbitrary fresh label, so spike
    # content and unit count are unchanged -- only which group gets
    # ``max+1`` for reordered input differs, and matching the two
    # paths is the more important contract.
    # Validate merge groups and stage their (key, contributors) pairs
    # WITHOUT inserting them yet -- kept_to_contributors's insertion
    # order is "surviving source units first (in original source
    # order), then merged kept ids appended in ascending-min order,"
    # matching SI's lazy MergeUnitsSorting (originals retained + merged
    # appended) at the SHAPE level (survivor-then-appended).
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

    # All groups validated; assign keys. Iterate in ascending
    # min-contributor order (NOT user-provided order) so the fresh
    # ``max+1`` ids match the lazy ``get_merged_sorting`` preview path
    # -- see the canonical-order NOTE above.
    merge_specs: list[tuple[int, list[int]]] = []
    next_merged_id = max(by_id) + 1
    for int_group in sorted(normalized_groups, key=min):
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
    #     matching the train write_curated_units_nwb writes.
    if apply_merge:
        specs = []
        for kept_uid, contribs in kept_to_contributors.items():
            anchor = max(contribs, key=lambda u: by_id[u]["peak_amplitude_uv"])
            n_spikes = (
                sum(by_id[u]["n_spikes"] for u in contribs)
                if len(contribs) > 1
                else by_id[int(kept_uid)]["n_spikes"]
            )
            specs.append((int(kept_uid), by_id[anchor], int(n_spikes)))
    else:
        specs = [
            (int(uid), row, int(row["n_spikes"])) for uid, row in by_id.items()
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
