"""Pure builders for the v2 curation insert plan and read summary.

``CurationV2.insert_curation`` is split into a pure planning half (here)
and a DB half (the table method). :func:`build_curation_insert_plan`
resolves the post-merge curated ``Unit`` rows (via
:func:`._curation_transforms.build_curated_unit_rows`) and validates the
supplied labels against the source/written unit sets. The table method
resolves the auto-increment ``curation_id`` (a DB query), stages the
curated-units NWB, and runs the atomic insert transaction on the plan.

``CurationV2.summarize_curation`` is split the same way:
:func:`build_curation_summary` assembles its return dict (including the
``is_merge_preview`` derivation) from the already-fetched fields, while
the table method keeps the PK guard and the DB fetches.

Isolating the pure halves makes the row-shaping + stray-label decision
matrix (truly-stray raise vs warn-and-drop, absorbed-contributor drop) and
the summary formatting unit-testable without a DataJoint transaction.

Dependency-light: imports only :mod:`_curation_transforms` (DB-free) and the
shared spyglass logger.

Naming note: "plan" here is a domain noun -- the computed set of rows/labels to
insert, like a query plan -- NOT a project-phase or milestone reference.
"""

from __future__ import annotations

from typing import NamedTuple

from spyglass.spikesorting.v2._curation_transforms import (
    build_curated_unit_rows,
    compose_curation_labels,
)
from spyglass.utils import logger


class CurationInsertPlan(NamedTuple):
    """The curated rows a ``CurationV2`` insert needs.

    ``unit_rows`` are the post-merge ``CurationV2.Unit`` rows (each with its
    peak Electrode FK + amplitude); ``kept_unit_to_contributors`` maps each
    kept unit_id to the source unit_ids it absorbed (a 1-element self-entry
    for a pass-through unit), in the SOURCE namespace (raw sort for a root,
    the parent curation for a child). ``labels`` is the effective
    ``{unit_id: [label, ...]}`` after inheritance/override composition (so a
    child inherits its parent's labels by default). ``curation_id`` is the
    auto-increment id the caller resolved, threaded through for row keying.
    """

    curation_id: int
    unit_rows: list[dict]
    kept_unit_to_contributors: dict[int, list[int]]
    labels: dict[int, list[str]]


def validate_label_unit_ids(
    labels: dict,
    *,
    source_unit_ids: set[int],
    written_unit_ids: set[int],
    sorting_id,
    apply_merge: bool,
    permissive_labels: bool,
) -> None:
    """Validate label keys against the source-union-written unit sets.

    Labels are written by iterating the final (written) unit_ids, so a key
    that is not written is dropped. Two kinds of unwritten key are handled
    differently:

    * TRULY stray (typo): keys in neither the source sorting nor the
      written curated set -- nothing they could ever refer to. Raises
      ``ValueError`` by default (typo protection); ``permissive_labels=True``
      restores warn-and-drop.
    * ABSORBED-source: keys that exist in the source sorting but are
      contributors absorbed into a merge (so not written). Always
      warn-and-drop -- the label cannot attach to the merged unit, but a
      caller passing original-id labels should not break.

    Side-effect only (warns / raises); the caller drops the unwritten keys
    by iterating ``written_unit_ids`` when inserting ``UnitLabel`` rows.
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
    permissive_labels: bool,
    curation_id: int,
    parent_labels: dict | None = None,
    label_policy: str = "inherit",
) -> CurationInsertPlan:
    """Build the curated ``Unit`` rows, compose labels, and validate them.

    ``curation_id`` is resolved by the caller (an auto-increment DB query)
    and threaded in. ``sorting_units`` are the SOURCE units the curation
    composes from -- the raw ``Sorting.Unit`` rows for a root curation, or the
    parent curation's ``CurationV2.Unit`` rows for a child. Row shaping is
    delegated to :func:`._curation_transforms.build_curated_unit_rows`, label
    inheritance/override to
    :func:`._curation_transforms.compose_curation_labels`, and label-key
    validation to :func:`validate_label_unit_ids`.

    ``parent_labels`` (``{parent_unit_id: [label, ...]}``, empty/None for a
    root) and ``label_policy`` (``"inherit"`` default / ``"replace"``) drive
    label composition: a child inherits its parent's labels by default, with a
    committed merge inheriting the union of its contributors' labels. The
    returned plan carries the EFFECTIVE (composed) labels.

    Raises
    ------
    ValueError
        From ``build_curated_unit_rows`` (empty/singleton/overlapping merge
        groups, or a member not in ``sorting_units``), from
        ``compose_curation_labels`` (bad ``label_policy``), or from
        ``validate_label_unit_ids`` (truly-stray label keys without
        ``permissive_labels``).
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
        permissive_labels=permissive_labels,
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
    merge_groups: dict,
    n_units: int,
) -> dict:
    """Assemble the ``summarize_curation`` return dict.

    Pure formatter: given the already-fetched curation fields, build the
    notebook-printable summary. ``is_merge_preview`` is derived here from the
    values in hand (not applied AND at least one >1-contributor merge group --
    the same condition as ``has_unapplied_proposed_merges``) rather than
    re-fetching. No database access.

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
    merge_groups : dict
        ``{kept_unit_id: [contributor_unit_id, ...]}`` from
        ``get_merge_groups`` (includes a 1-element self-entry per unit; real
        merges have >1 contributor).
    n_units : int
        Count of ``CurationV2.Unit`` rows.

    Returns
    -------
    dict
        The ``summarize_curation`` summary.
    """
    is_merge_preview = not bool(merges_applied) and any(
        len(contribs) > 1 for contribs in merge_groups.values()
    )
    return {
        "sorting_id": sorting_id,
        "curation_id": int(curation_id),
        "n_units": n_units,
        "labels": labels,
        "merge_groups": merge_groups,
        "merges_applied": bool(merges_applied),
        "is_merge_preview": is_merge_preview,
        "merge_id": merge_id,
        "description": str(description),
    }
