"""Pure builder for the v2 curation insert plan.

``CurationV2.insert_curation`` is split into a pure planning half (here)
and a DB half (the table method). :func:`build_curation_insert_plan`
resolves the post-merge curated ``Unit`` rows (via
:func:`._curation_transforms.build_curated_unit_rows`) and validates the
supplied labels against the source/written unit sets. The table method
resolves the auto-increment ``curation_id`` (a DB query), stages the
curated-units NWB, and runs the atomic insert transaction on the plan.

Isolating the pure half makes the row-shaping + stray-label decision
matrix (truly-stray raise vs warn-and-drop, absorbed-contributor drop)
unit-testable without a DataJoint transaction.

Dependency-light: imports only :mod:`_curation_transforms` (DB-free) and
the spyglass logger.
"""

from __future__ import annotations

from typing import NamedTuple

from spyglass.spikesorting.v2._curation_transforms import (
    build_curated_unit_rows,
)
from spyglass.utils import logger


class CurationInsertPlan(NamedTuple):
    """The curated rows a ``CurationV2`` insert needs.

    ``unit_rows`` are the post-merge ``CurationV2.Unit`` rows (each with its
    peak Electrode FK + amplitude); ``kept_unit_to_contributors`` maps each
    kept unit_id to the source unit_ids it absorbed (a 1-element self-entry
    for a pass-through unit). ``curation_id`` is the auto-increment id the
    caller resolved, threaded through for row keying.
    """

    curation_id: int
    unit_rows: list[dict]
    kept_unit_to_contributors: dict[int, list[int]]


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
) -> CurationInsertPlan:
    """Build the curated ``Unit`` rows + validate labels for a curation.

    ``curation_id`` is resolved by the caller (an auto-increment DB query)
    and threaded in. Row shaping is delegated to
    :func:`._curation_transforms.build_curated_unit_rows` and label-key
    validation to :func:`validate_label_unit_ids`.

    Raises
    ------
    ValueError
        From ``build_curated_unit_rows`` (empty/singleton/overlapping merge
        groups, or a member not in ``sorting_units``) or from
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
    validate_label_unit_ids(
        labels,
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
    )
