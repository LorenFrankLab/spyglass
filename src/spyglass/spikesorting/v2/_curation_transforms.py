"""Pure curation transforms behind ``CurationV2``.

These functions are the dependency-light core of v2 curation --
label-value validation, ``UnitLabel``-row validation, manual/FigURL payload
normalization (the v1/v2 spelling shim feeding ``save_manual_curation``),
parent/supplied label composition (inherit vs. replace; union on a committed
merge), and the post-merge ``CurationV2.Unit`` row construction (merge-group
validation, ``kept_unit_to_contributors`` mapping, and the per-unit row build).
They are pure Python: given the already-fetched source ``Unit`` rows (raw
``Sorting.Unit`` for a root, the parent ``CurationV2.Unit`` for a child) and
the caller's label / merge-group payloads, they decide WHAT to insert without
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
the standard library and dependency-light enum / validation helpers,
rather than through ``utils`` (which imports
DataJoint / SpikeInterface at load). So the transform layer itself does
not depend on the heavy table-support module. (Importing it as a
``spyglass`` submodule still triggers ``spyglass``'s package ``__init__``,
which loads DataJoint -- that is package-wide and unavoidable; the point
here is that these transforms add no DB/SpikeInterface dependency of their
own.)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from spyglass.spikesorting.v2._enums import CurationLabel
from spyglass.spikesorting.v2._lookup_validation import lossless_int


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


def _payload_value(payload: Mapping, field_names: tuple[str, ...]):
    """Return one payload value, rejecting contradictory aliases."""
    present = [
        name
        for name in field_names
        if name in payload and payload[name] is not None
    ]
    if not present:
        return None
    value = payload[present[0]]
    for name in present[1:]:
        if payload[name] != value:
            raise ValueError(
                "Curation payload contains multiple values for the same field "
                f"({', '.join(present)}); pass only one spelling."
            )
    return value


def parse_curation_unit_id(value) -> int:
    """Parse a transport unit ID, allowing integer strings used by JSON.

    Numeric values must already be integers; floats and booleans must not
    silently become another unit's ID.
    """
    if isinstance(value, str):
        return int(value)
    return lossless_int(value, "unit_id")


def _normalize_payload_labels(labels) -> dict[int, list[str]]:
    """Normalize transport labels to ``{int unit_id: [label, ...]}``."""
    if labels is None:
        return {}
    if not isinstance(labels, Mapping):
        raise ValueError(
            "Curation payload labels must be a mapping of unit_id to a list of "
            f"labels; got {type(labels).__name__}."
        )

    normalized: dict[int, list[str]] = {}
    for unit_id, unit_labels in labels.items():
        unit_id = parse_curation_unit_id(unit_id)
        if unit_labels is None:
            normalized[unit_id] = []
            continue
        if isinstance(unit_labels, str) or not isinstance(
            unit_labels, (list, tuple)
        ):
            raise ValueError(
                "Curation payload labels[unit_id] must be a list of labels; "
                f"got {type(unit_labels).__name__} for unit_id={unit_id}."
            )
        normalized[unit_id] = [
            CurationLabel.normalize(label) for label in unit_labels
        ]
    return normalized


def _iter_payload_merge_group(group) -> list[int]:
    """Normalize one merge group, preserving singleton/empty typo guards."""
    if isinstance(group, str) or not isinstance(group, Iterable):
        raise ValueError(
            "Curation payload merge groups must be lists of unit ids; got "
            f"{type(group).__name__}."
        )
    return [parse_curation_unit_id(unit_id) for unit_id in group]


def _union_intersecting(groups: list[set[int]]) -> list[set[int]]:
    """Union member sets that share any unit (connected components).

    Matches v1's ``_union_intersecting_lists`` so a transitive association
    chain ``{1: [2], 2: [3]}`` collapses to one group ``{1, 2, 3}`` instead of
    overlapping ``{1, 2}`` / ``{2, 3}`` that would later double-assign a unit.
    """
    remaining = [set(group) for group in groups]
    result: list[set[int]] = []
    while remaining:
        first, *rest = remaining
        merged = True
        while merged:
            merged = False
            for idx, other in enumerate(rest):
                if first & other:
                    first |= other
                    del rest[idx]
                    merged = True
                    break
        result.append(first)
        remaining = rest
    return result


def _normalize_payload_merge_groups(merge_groups) -> list[list[int]]:
    """Normalize transport merge groups to ``list[list[int]]``.

    Mapping input is the v1/FigURL per-unit association shape
    ``{unit_id: [other_unit_ids...]}``; it is converted to full groups with
    INTERSECTING associations unioned (v1 ``_merge_dict_to_list`` parity --
    ``{1: [2], 2: [3]}`` becomes ``[[1, 2, 3]]``), dropping resulting singletons.
    List input is preserved verbatim except for integer coercion so singleton or
    empty groups still reach ``insert_curation``'s typo guard.
    """
    if merge_groups is None:
        return []
    if isinstance(merge_groups, Mapping):
        association_sets: list[set[int]] = []
        for unit_id, associated in merge_groups.items():
            members = {parse_curation_unit_id(unit_id)}
            if associated is not None and not (
                isinstance(associated, str) and associated == ""
            ):
                if isinstance(associated, str) or not isinstance(
                    associated, Iterable
                ):
                    associated = [associated]
                for member in associated:
                    if member is None or (
                        isinstance(member, str) and member == ""
                    ):
                        continue
                    members.add(parse_curation_unit_id(member))
            association_sets.append(members)
        return [
            sorted(group)
            for group in _union_intersecting(association_sets)
            if len(group) >= 2
        ]

    if isinstance(merge_groups, str) or not isinstance(merge_groups, Iterable):
        raise ValueError(
            "Curation payload merge_groups must be a list of merge groups; got "
            f"{type(merge_groups).__name__}."
        )
    return [_iter_payload_merge_group(group) for group in merge_groups]


def normalize_curation_payload(
    payload: Mapping | None = None,
    *,
    labels=None,
    merge_groups=None,
) -> tuple[dict[int, list[str]], list[list[int]]]:
    """Normalize a manual/FigURL/FigPack curation payload.

    Accepts the v1/FigURL JSON spellings (``labelsByUnit`` / ``mergeGroups``)
    and the v2/Python spellings (``labels_by_unit`` / ``merge_groups``), plus
    explicit ``labels=`` / ``merge_groups=`` kwargs for already-unpacked
    payloads. Unit ids must be integers or integer strings; floats and booleans
    are rejected. Label values are coerced through :class:`CurationLabel`;
    merge-group shape validation is left to
    ``CurationV2.insert_curation`` so typos still produce the same errors there.
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Curation payload must be a mapping with labels/merge fields; got "
            f"{type(payload).__name__}."
        )

    payload_labels = _payload_value(payload, ("labelsByUnit", "labels_by_unit"))
    payload_merges = _payload_value(payload, ("mergeGroups", "merge_groups"))
    if labels is not None and payload_labels is not None:
        raise ValueError(
            "Curation payload labels were provided both inside payload and via "
            "labels=; pass only one source."
        )
    if merge_groups is not None and payload_merges is not None:
        raise ValueError(
            "Curation payload merge groups were provided both inside payload and "
            "via merge_groups=; pass only one source."
        )

    return (
        _normalize_payload_labels(
            labels if labels is not None else payload_labels
        ),
        _normalize_payload_merge_groups(
            merge_groups if merge_groups is not None else payload_merges
        ),
    )


def compose_curation_labels(
    *,
    parent_labels: dict[int, list[str]],
    supplied_labels: dict[int, list[str]],
    kept_unit_to_contributors: dict[int, list[int]],
    label_policy: str,
    apply_merge: bool,
) -> dict[int, list[str]]:
    """Resolve a child curation's effective label state.

    A child curation edits its parent's committed state, so by default it
    inherits the parent's labels (``label_policy="inherit"``); a caller wanting
    the supplied labels to be the whole child state passes
    ``label_policy="replace"``. Root curations have an empty ``parent_labels``,
    so ``"inherit"`` reduces to "supplied labels only" -- the historical
    full-state insert.

    Inheritance is identity-preserving except across a committed merge:

    * ``apply_merge=True`` -- a kept unit (possibly a fresh merged id) inherits
      the UNION of its contributors' parent labels, so labels on absorbed
      contributors do not disappear when the merge is committed.
    * ``apply_merge=False`` (preview) -- every parent unit passes through 1:1,
      so each written unit inherits only its OWN parent labels (the proposed
      merge does not yet combine label state).

    Supplied labels are then overlaid per unit (an explicit override wins over
    the inherited value, and an explicit empty list clears it). Units that end
    up with no labels are dropped from the result.

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

    composed: dict[int, list[str]] = {}
    for kept_uid, contributors in kept_unit_to_contributors.items():
        kept_uid = int(kept_uid)
        if apply_merge:
            union: set[str] = set()
            for contributor in contributors:
                union.update(parent_labels.get(int(contributor), []))
            if union:
                composed[kept_uid] = sorted(union)
        else:
            own = parent_labels.get(kept_uid, [])
            if own:
                composed[kept_uid] = list(own)

    for unit_id, labels in supplied_labels.items():
        # An explicit supplied entry overrides the inherited value for that
        # unit (an empty list clears it; the trailing filter drops it).
        composed[int(unit_id)] = list(labels)
    return {k: v for k, v in composed.items() if v}


def allocate_merged_unit_ids(
    source_unit_ids: Iterable[int],
    merge_groups: Iterable[Sequence[int]],
) -> dict[int, list[int]]:
    """Map fresh IDs to validated merge groups in min-contributor order.

    This order matches the stored preview's merge leaders, so preview,
    committed rows, and review labels agree on each merged unit's ID.
    Contributor order within a group is preserved for metadata tie-breaking.
    Callers validate group membership, size, uniqueness, and disjointness.
    """
    next_id = max(source_unit_ids, default=-1) + 1
    return {
        unit_id: list(group)
        for unit_id, group in enumerate(
            sorted(merge_groups, key=min), start=next_id
        )
    }


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
    ``max(source unit_ids) + 1`` for ``apply_merge=True``;
    ``min(group)`` for ``apply_merge=False`` (the proposed merge
    leader, kept as a regular Unit row alongside the absorbed
    contributors so the preview retains every original unit). Each
    merge group must have at least 2 members; empty or singleton
    groups raise ``ValueError``.

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

    normalized_groups: list[list[int]] = [
        [lossless_int(u, "merge group unit_id") for u in g]
        for g in merge_groups
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
            # Empty or singleton "merge groups" aren't merges.
            # Silently no-oping (empty) or renaming the singleton to
            # max+1 would hide a likely typo; raise instead.
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
    #     (preview); the proposed merges live in MergeGroup for lazy
    #     application via get_merged_sorting. n_spikes is each unit's
    #     own count, matching the train write_curated_units_nwb writes.
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
