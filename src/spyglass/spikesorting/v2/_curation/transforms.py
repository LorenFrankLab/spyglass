"""Pure curation transforms behind ``CurationV2``.

Label validation and composition, manual-input normalization, post-merge
``CurationV2.Unit`` rows, and ``MergeGroup`` / ``ParentMergeGroup``
provenance rows. Given pre-fetched ``Unit`` rows, they decide what to insert.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal

from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.spikesorting.v2._core.lookup_validation import lossless_int

ManualMergeAction = Literal["preview", "commit"]


def validate_curation_label_rows(
    rows: list, allow_custom_labels: bool = False
) -> None:
    """Reject ``UnitLabel`` rows whose ``curation_label`` is not canonical.

    The column is a bare ``varchar``, so direct inserts need this guard.
    ``allow_custom_labels=True`` accepts non-canonical labels. Rows without
    ``curation_label`` are left to DataJoint's missing-attribute error.
    """
    valid = {member.value for member in CurationLabel}
    for row in rows:
        # An ordered row would silently skip the by-key label check.
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

    Accepts ``CurationLabel`` instances or their string values; others raise
    ``ValueError`` unless ``allow_custom_labels=True``. Each
    ``labels[unit_id]`` must already be a list.
    """
    if allow_custom_labels:
        return
    valid = {member.value for member in CurationLabel}
    for unit_id, lbls in labels.items():
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


def _normalize_manual_labels(labels) -> dict[int, list[str]]:
    """Normalize native labels to ``{int unit_id: [label, ...]}``."""
    if labels is None:
        return {}
    if not isinstance(labels, Mapping):
        raise ValueError(
            "Manual curation labels must be a mapping of unit_id to a list of "
            f"labels; got {type(labels).__name__}."
        )

    normalized: dict[int, list[str]] = {}
    for unit_id, unit_labels in labels.items():
        unit_id = lossless_int(unit_id, "unit_id")
        if unit_labels is None:
            normalized[unit_id] = []
            continue
        if isinstance(unit_labels, str) or not isinstance(
            unit_labels, (list, tuple)
        ):
            raise ValueError(
                "Manual curation labels[unit_id] must be a list of labels; "
                f"got {type(unit_labels).__name__} for unit_id={unit_id}."
            )
        normalized[unit_id] = [
            CurationLabel.normalize(label) for label in unit_labels
        ]
    return normalized


def _normalize_manual_merge_groups(merge_groups) -> list[list[int]]:
    """Normalize list groups, preserving empty/singleton typo guards."""
    if merge_groups is None:
        return []
    if not isinstance(merge_groups, list):
        raise ValueError(
            "Manual curation merge_groups must be a list of lists of unit ids; got "
            f"{type(merge_groups).__name__}."
        )
    normalized = []
    for group in merge_groups:
        if not isinstance(group, list):
            raise ValueError(
                "Manual curation merge_groups must be a list of lists of unit ids; "
                f"got a {type(group).__name__} group."
            )
        normalized.append(
            [lossless_int(unit_id, "unit_id") for unit_id in group]
        )
    return normalized


def normalize_manual_curation(
    *,
    labels: dict[int, list[str]] | None = None,
    merge_groups: list[list[int]] | None = None,
) -> tuple[dict[int, list[str]], list[list[int]]]:
    """Normalize native ``labels=`` and ``merge_groups=`` inputs.

    Unit IDs must be integers (NumPy included); floats, booleans and string
    IDs are rejected. Labels are coerced through :class:`CurationLabel`.
    Merge-group validation is left to ``CurationV2.insert_curation``.
    """
    return (
        _normalize_manual_labels(labels),
        _normalize_manual_merge_groups(merge_groups),
    )


def manual_curation_applies_merges(
    merge_action: ManualMergeAction, merge_groups: list[list[int]]
) -> bool:
    """Validate the manual action and decide whether its groups are applied."""
    if merge_action not in ("preview", "commit"):
        raise ValueError(
            "CurationV2.save_manual_curation: merge_action must be "
            f"'preview' or 'commit'; got {merge_action!r}."
        )
    return merge_action == "commit" and bool(merge_groups)


def normalize_label_state(
    labels: Mapping[int, Iterable] | None,
) -> dict[int, tuple[str, ...]]:
    """Normalize ``{unit_id: labels}`` for semantic equality checks.

    Unit ids become ``int``; each unit's labels are coerced through
    :meth:`CurationLabel.normalize` and sorted into a tuple; units with no
    labels are dropped. ``None`` is treated as empty.
    """
    return {
        int(unit_id): tuple(
            sorted(CurationLabel.normalize(label) for label in unit_labels)
        )
        for unit_id, unit_labels in (labels or {}).items()
        if unit_labels
    }


def inherit_parent_labels(
    parent_labels: Mapping[int, Sequence[str]],
    kept_unit_to_contributors: Mapping[int, Sequence[int]],
    *,
    apply_merge: bool,
) -> dict[int, list[str]]:
    """Return the labels each kept unit inherits from its parent curation.

    With ``apply_merge=True`` a kept unit inherits the sorted union of its
    contributors' parent labels. With ``apply_merge=False`` (preview) every
    parent unit passes through 1:1, so a kept unit inherits only its own
    parent labels. Units that inherit no labels are omitted.

    Parameters
    ----------
    parent_labels : Mapping[int, Sequence[str]]
        ``{parent_unit_id: [label, ...]}``.
    kept_unit_to_contributors : Mapping[int, Sequence[int]]
        ``{kept_unit_id: [contributor unit ids]}`` in the parent namespace.
    apply_merge : bool
        Whether the child commits the merges.

    Returns
    -------
    dict[int, list[str]]
        ``{kept_unit_id: [label, ...]}``.
    """
    inherited: dict[int, list[str]] = {}
    for kept_uid, contributors in kept_unit_to_contributors.items():
        kept_uid = int(kept_uid)
        if apply_merge:
            union: set[str] = set()
            for contributor in contributors:
                union.update(parent_labels.get(int(contributor), []))
            if union:
                inherited[kept_uid] = sorted(union)
        else:
            own = parent_labels.get(kept_uid, [])
            if own:
                inherited[kept_uid] = list(own)
    return inherited


def is_merge_preview(
    merges_applied, unit_contributor_groups: Mapping[int, Sequence[int]]
) -> bool:
    """Whether a curation proposes merges it has not applied.

    True when ``merges_applied`` is false and at least one kept unit has more
    than one contributor (every unit also carries a 1-element self-entry).
    """
    return not bool(merges_applied) and any(
        len(contributors) > 1
        for contributors in unit_contributor_groups.values()
    )


def group_contributor_rows(
    rows: Iterable[Mapping], contributor_field: str
) -> dict[int, list[int]]:
    """Group merge-provenance rows into ``{unit_id: [contributor, ...]}``.

    ``rows`` are ``MergeGroup`` / ``ParentMergeGroup`` row dicts carrying
    ``unit_id`` and ``contributor_field``; contributors keep row order.
    """
    groups: dict[int, list[int]] = {}
    for row in rows:
        groups.setdefault(int(row["unit_id"]), []).append(
            int(row[contributor_field])
        )
    return groups


def compose_curation_labels(
    *,
    parent_labels: dict[int, list[str]],
    supplied_labels: dict[int, list[str]],
    kept_unit_to_contributors: dict[int, list[int]],
    label_policy: str,
    apply_merge: bool,
) -> dict[int, list[str]]:
    """Resolve a child curation's effective label state.

    - ``"inherit"``: parent labels (see :func:`inherit_parent_labels`), then
      supplied labels override per unit; an empty list clears.
    - ``"replace"``: supplied labels are the whole state.

    Units with no labels are dropped.

    Parameters
    ----------
    parent_labels : dict[int, list[str]]
        ``{parent_unit_id: [label, ...]}`` from the parent curation's
        ``UnitLabel`` rows. Empty for a root curation.
    supplied_labels : dict[int, list[str]]
        The caller's normalized ``{unit_id: [label, ...]}`` payload (keyed in
        the child/parent unit namespace).
    kept_unit_to_contributors : dict[int, list[int]]
        ``{kept_unit_id: [source contributor ids]}`` for the child, in the
        source (parent) namespace.
    label_policy : str
        ``"inherit"`` (default) or ``"replace"``.
    apply_merge : bool
        Whether the child commits the merges (union on the merged unit) or
        previews them (per-unit identity inheritance).

    Returns
    -------
    dict[int, list[str]]
        The effective ``{unit_id: [label, ...]}`` for the child, empty-valued
        units omitted.
    """
    if label_policy not in ("inherit", "replace"):
        raise ValueError(
            "CurationV2.insert_curation: label_policy must be 'inherit' or "
            f"'replace'; got {label_policy!r}."
        )
    if label_policy == "replace":
        return {int(k): list(v) for k, v in supplied_labels.items() if v}

    composed = inherit_parent_labels(
        parent_labels, kept_unit_to_contributors, apply_merge=apply_merge
    )
    for unit_id, labels in supplied_labels.items():
        composed[int(unit_id)] = list(labels)
    return {k: v for k, v in composed.items() if v}


def validate_merge_groups(
    merge_groups: Iterable[Sequence[int]],
    *,
    unit_ids: Collection[int] | None = None,
    unit_namespace: str = "",
    prefix: str = "",
) -> None:
    """Reject merge groups that are not disjoint sets of at least 2 units.

    Groups are checked in order. For each group: it has at least 2 members,
    no member repeats, every member is in ``unit_ids`` (when given), and no
    member belongs to an earlier group. The first violation raises.

    Parameters
    ----------
    merge_groups : iterable of sequence of int
        Merge groups whose members are already integer unit ids.
    unit_ids : collection of int, optional
        The unit ids a member must belong to. ``None`` skips the membership
        check (the caller checks membership itself).
    unit_namespace : str, optional
        Names the ``unit_ids`` source in the membership error.
    prefix : str, optional
        Prepended to every error message (e.g. the calling method's name).

    Raises
    ------
    ValueError
        On the first group that is too small, repeats a member, references
        a unit outside ``unit_ids``, or overlaps an earlier group.
    """
    seen: set[int] = set()
    for group in merge_groups:
        if len(group) < 2:
            # Likely a typo; do not silently no-op.
            raise ValueError(
                f"{prefix}merge_groups contains a group with fewer than 2 "
                f"members ({group}); each merge group must contain at least "
                "two unit ids (at least 2 units)."
            )
        if len(set(group)) != len(group):
            raise ValueError(
                f"{prefix}merge_groups contains a group with duplicate "
                f"members ({group}); each unit can appear at most once per "
                "merge group."
            )
        if unit_ids is not None:
            for unit_id in group:
                if unit_id not in unit_ids:
                    raise ValueError(
                        f"{prefix}merge_groups references unit_id={unit_id} "
                        f"that is not in {unit_namespace}."
                    )
        overlap = seen & set(group)
        if overlap:
            raise ValueError(
                f"{prefix}merge_groups overlap on unit_ids {sorted(overlap)}; "
                "merge groups must be disjoint (a unit can belong to at most "
                "one merge group)."
            )
        seen.update(group)


def allocate_merged_unit_ids(
    source_unit_ids: Iterable[int],
    merge_groups: Iterable[Sequence[int]],
) -> dict[int, list[int]]:
    """Map fresh IDs to validated merge groups in min-contributor order.

    Matches the preview's merge-leader order so preview, committed rows, and
    review labels agree on IDs. Contributor order is preserved for
    tie-breaking. Validate groups first (:func:`validate_merge_groups`).
    """
    next_id = max(source_unit_ids, default=-1) + 1
    return {
        unit_id: list(group)
        for unit_id, group in enumerate(
            sorted(merge_groups, key=min), start=next_id
        )
    }


def key_manual_labels_to_merged_units(
    labels: dict[int, list[str]],
    merge_groups: list[list[int]],
    *,
    source_unit_ids: Iterable[int],
    parent_labels: Mapping[int, Sequence[str]],
    label_policy: str,
) -> dict[int, list[str]]:
    """Move a committed merge's contributor labels onto its merged unit.

    Manual edits are keyed by parent unit ids, so contributor labels would
    otherwise be dropped as absorbed. Each merged unit gets the union of its
    contributors' supplied labels (parent labels for unsupplied contributors
    under ``"inherit"``). Other labels are unchanged.
    """
    absorbed = {unit_id for group in merge_groups for unit_id in group}
    if not absorbed & labels.keys():
        return labels
    keyed = {u: v for u, v in labels.items() if u not in absorbed}
    for merged_id, group in allocate_merged_unit_ids(
        source_unit_ids, merge_groups
    ).items():
        union: set[str] = set()
        for unit_id in group:
            if unit_id in labels:
                union.update(labels[unit_id])
            elif label_policy == "inherit":
                union.update(parent_labels.get(unit_id, []))
        keyed[merged_id] = sorted(union)
    return keyed


def build_curated_unit_rows(
    sorting_id,
    sorting_units: list[dict],
    merge_groups: list[list[int]],
    curation_id: int,
    apply_merge: bool,
) -> tuple[list[dict], dict[int, list[int]]]:
    """Resolve the post-merge ``CurationV2.Unit`` rows.

    - Electrode and peak amplitude come from the contributor with the
      largest ``peak_amplitude_uv``.
    - Merged ``unit_id``: fresh ``max(source unit_ids) + 1`` when
      ``apply_merge``; else ``min(group)``, with every original unit kept.
    - ``n_spikes`` matches the train ``write_curated_units_nwb`` writes: the
      merged sum only when ``apply_merge``, else each unit's own count.

    Parameters
    ----------
    sorting_id
        ``sorting_id`` of the upstream Sorting row.
    sorting_units : list of dict
        Pre-fetched ``Sorting.Unit`` rows for ``sorting_id``.
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

    normalized_groups: list[list[int]] = [
        [lossless_int(u, "merge group unit_id") for u in g]
        for g in merge_groups
    ]
    # Before the zero-unit return, so merge groups on an empty sort raise.
    validate_merge_groups(
        normalized_groups,
        unit_ids=by_id,
        unit_namespace=f"Sorting.Unit for sorting_id={sorting_id}",
        prefix="CurationV2.insert_curation: ",
    )
    merged_ids = {uid for group in normalized_groups for uid in group}

    if not by_id:
        return [], {}

    # Committed groups get fresh IDs; previews keep min(group) as the
    # proposed merge leader while retaining every source unit.
    merge_specs = (
        allocate_merged_unit_ids(by_id, normalized_groups)
        if apply_merge
        else {min(g): g for g in sorted(normalized_groups, key=min)}
    )

    kept_to_contributors: dict[int, list[int]] = {}
    # Non-merged units FIRST, in original source order.
    for uid in by_id:
        if uid not in merged_ids:
            kept_to_contributors[uid] = [uid]
    # Then merged units, in canonical min-contributor order (SI parity).
    kept_to_contributors.update(merge_specs)

    # Previews keep absorbed units, so give each a self-entry; a
    # Unit * MergeGroup join then keeps every row.
    if not apply_merge:
        for uid in by_id:
            if uid not in kept_to_contributors:
                kept_to_contributors[uid] = [uid]

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


def build_merge_provenance_rows(
    *,
    sorting_id,
    curation_id: int,
    kept_unit_to_contributors: dict[int, list[int]],
    parent_raw_contributors: dict[int, list[int]] | None,
) -> tuple[list[dict], list[dict]]:
    """Build the raw ``MergeGroup`` + parent ``ParentMergeGroup`` rows.

    Returns ``(merge_group_rows, parent_merge_group_rows)``.

    - Root (``parent_raw_contributors is None``): contributors are raw
      ``Sorting.Unit`` ids, written verbatim; no parent rows.
    - Child: contributors are parent unit ids, expanded to raw ids for
      ``MergeGroup`` (FK-safe) and recorded verbatim in ``ParentMergeGroup``.
    """
    if parent_raw_contributors is None:
        merge_group_rows = [
            {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "unit_id": int(kept_uid),
                "contributor_unit_id": int(contributor_uid),
            }
            for kept_uid, contributors in kept_unit_to_contributors.items()
            for contributor_uid in contributors
        ]
        return merge_group_rows, []

    merge_group_rows = []
    parent_merge_group_rows = []
    for kept_uid, parent_contributors in kept_unit_to_contributors.items():
        kept_uid = int(kept_uid)
        # A hand-inserted parent may lack raw provenance; name it rather
        # than raise a bare KeyError mid-transaction.
        missing = [
            int(p)
            for p in parent_contributors
            if int(p) not in parent_raw_contributors
        ]
        if missing:
            raise ValueError(
                "CurationV2.insert_curation: parent unit(s) "
                f"{sorted(missing)} have no CurationV2.MergeGroup raw "
                "provenance, so a child's raw contributors cannot be "
                "resolved. The parent curation must have been created via "
                "insert_curation (every unit gets a MergeGroup self-entry)."
            )
        raw_ids = sorted(
            {
                int(raw)
                for parent_uid in parent_contributors
                for raw in parent_raw_contributors[int(parent_uid)]
            }
        )
        merge_group_rows.extend(
            {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "unit_id": kept_uid,
                "contributor_unit_id": raw,
            }
            for raw in raw_ids
        )
        parent_merge_group_rows.extend(
            {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "unit_id": kept_uid,
                "parent_unit_id": int(parent_uid),
            }
            for parent_uid in sorted(int(u) for u in parent_contributors)
        )
    return merge_group_rows, parent_merge_group_rows
