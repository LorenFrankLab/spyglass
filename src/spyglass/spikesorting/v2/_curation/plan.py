"""Pure halves of ``CurationV2.insert_curation`` and ``summarize_curation``.

The table methods keep the DB queries, NWB staging, and transaction.
"""

from __future__ import annotations

from typing import NamedTuple

from spyglass.spikesorting.v2._curation.transforms import (
    build_curated_unit_rows,
    compose_curation_labels,
    is_merge_preview,
)
from spyglass.utils import logger


class CurationInsertPlan(NamedTuple):
    """The curated rows a ``CurationV2`` insert needs.

    - ``unit_rows``: post-merge ``CurationV2.Unit`` rows.
    - ``kept_unit_to_contributors``: kept unit_id to absorbed source
      unit_ids (self-entry for pass-through), in the source namespace.
    - ``labels``: effective ``{unit_id: [label, ...]}`` after composition.
    - ``curation_id``: the id the caller allocated.
    """

    curation_id: int
    unit_rows: list[dict]
    kept_unit_to_contributors: dict[int, list[int]]
    labels: dict[int, list[str]]


def build_curation_header(
    *,
    sorting_id,
    curation_id: int,
    curation_uuid,
    parent_curation_id: int,
    curation_source,
    merges_applied: bool,
    description: str,
) -> dict:
    """Return the ``CurationV2`` identity fields of a curated-units NWB header.

    Parameters
    ----------
    sorting_id, curation_uuid : uuid.UUID or str
        Written as strings.
    curation_id, parent_curation_id : int
    curation_source : CurationSource or str
        Written as the enum option's canonical string.
    merges_applied : bool
    description : str

    Returns
    -------
    dict
        JSON-native header fields; callers add stage-specific provenance.
    """
    return {
        "sorting_id": str(sorting_id),
        "curation_id": int(curation_id),
        "curation_uuid": str(curation_uuid),
        "parent_curation_id": int(parent_curation_id),
        "curation_source": getattr(curation_source, "value", curation_source),
        "merges_applied": bool(merges_applied),
        "description": description,
    }


def validate_label_unit_ids(
    labels: dict,
    *,
    source_unit_ids: set[int],
    written_unit_ids: set[int],
    sorting_id,
    apply_merge: bool,
    allow_unknown_unit_ids: bool,
) -> None:
    """Validate label keys against the source-union-written unit sets.

    Warns or raises only; the caller drops unwritten keys.

    - Stray keys (in neither set): raise ``ValueError``, or warn when
      ``allow_unknown_unit_ids=True``.
    - Absorbed merge contributors: always warn.
    """
    label_keys = set(labels)
    truly_stray = label_keys - (source_unit_ids | written_unit_ids)
    absorbed_label_ids = label_keys & (source_unit_ids - written_unit_ids)
    if truly_stray:
        message = (
            "CurationV2.insert_curation: labels reference unit_id(s) "
            f"{sorted(truly_stray)} that are neither in the source unit set "
            "(the raw sort for a root curation, or the parent curation for a "
            "child) nor in the curated unit set for "
            f"sorting_id={sorting_id} (apply_merge={apply_merge}). "
            f"Curated unit_ids: {sorted(written_unit_ids)}."
        )
        if not allow_unknown_unit_ids:
            raise ValueError(
                f"{message} Pass allow_unknown_unit_ids=True to ignore "
                "stray labels instead of raising."
            )
        logger.warning(f"{message} Those labels are ignored.")
    if absorbed_label_ids:
        logger.warning(
            "CurationV2.insert_curation: labels keyed on absorbed "
            f"contributor unit(s) {sorted(absorbed_label_ids)} are "
            "dropped -- they cannot attach to the merged unit. "
            f"sorting_id={sorting_id}."
        )


def build_curation_insert_plan(
    *,
    sorting_id,
    sorting_units: list[dict],
    merge_groups,
    apply_merge: bool,
    labels: dict,
    allow_unknown_unit_ids: bool,
    curation_id: int,
    parent_labels: dict | None = None,
    label_policy: str = "inherit",
) -> CurationInsertPlan:
    """Build the curated ``Unit`` rows, compose labels, and validate them.

    ``sorting_units`` are the source units: raw ``Sorting.Unit`` rows for a
    root, the parent's ``CurationV2.Unit`` rows for a child.
    ``parent_labels`` (empty for a root) and ``label_policy`` drive
    :func:`._curation.transforms.compose_curation_labels`.

    Raises
    ------
    ValueError
        From ``build_curated_unit_rows`` (empty/singleton/overlapping merge
        groups, or a member not in ``sorting_units``), from
        ``compose_curation_labels`` (bad ``label_policy``), or from
        ``validate_label_unit_ids`` (truly-stray label keys without
        ``allow_unknown_unit_ids``).
    """
    unit_rows, kept_unit_to_contributors = build_curated_unit_rows(
        sorting_id=sorting_id,
        sorting_units=sorting_units,
        merge_groups=merge_groups or [],
        curation_id=curation_id,
        apply_merge=apply_merge,
    )
    effective_labels = compose_curation_labels(
        parent_labels=parent_labels or {},
        supplied_labels=labels,
        kept_unit_to_contributors=kept_unit_to_contributors,
        label_policy=label_policy,
        apply_merge=apply_merge,
    )
    validate_label_unit_ids(
        effective_labels,
        source_unit_ids={int(r["unit_id"]) for r in sorting_units},
        written_unit_ids={row["unit_id"] for row in unit_rows},
        sorting_id=sorting_id,
        apply_merge=apply_merge,
        allow_unknown_unit_ids=allow_unknown_unit_ids,
    )
    return CurationInsertPlan(
        curation_id=curation_id,
        unit_rows=unit_rows,
        kept_unit_to_contributors=kept_unit_to_contributors,
        labels=effective_labels,
    )


def build_curation_summary(
    *,
    sorting_id,
    curation_id,
    merges_applied,
    description,
    labels: dict,
    merge_id,
    unit_contributor_groups: dict,
    n_units: int,
) -> dict:
    """Assemble the ``summarize_curation`` return dict.

    Pure formatter over already-fetched fields.

    Parameters
    ----------
    sorting_id
        The sort's UUID (returned unchanged).
    curation_id
        The curation id within the sort; coerced to ``int`` in the result.
    merges_applied
        The stored ``merges_applied`` field; coerced to ``bool``.
    description
        The stored description; coerced to ``str``.
    labels : dict
        ``unit_id -> [label, ...]`` from ``UnitLabel``.
    merge_id
        The ``SpikeSortingOutput.CurationV2`` merge id, or ``None`` if the
        curation is not merge-registered.
    unit_contributor_groups : dict
        ``{kept_unit_id: [contributor_unit_id, ...]}`` from
        ``get_unit_contributor_groups`` (includes a 1-element self-entry per
        unit; real merges have >1 contributor).
    n_units : int
        Count of ``CurationV2.Unit`` rows.

    Returns
    -------
    dict
        The ``summarize_curation`` summary.
    """
    merge_groups = {
        kept_unit_id: contributors
        for kept_unit_id, contributors in unit_contributor_groups.items()
        if len(contributors) > 1
    }
    return {
        "sorting_id": sorting_id,
        "curation_id": int(curation_id),
        "n_units": n_units,
        "labels": labels,
        "merge_groups": merge_groups,
        "unit_contributor_groups": unit_contributor_groups,
        "merges_applied": bool(merges_applied),
        "is_merge_preview": is_merge_preview(
            merges_applied, unit_contributor_groups
        ),
        "merge_id": merge_id,
        "description": str(description),
    }
