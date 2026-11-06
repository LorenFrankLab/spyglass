"""Pre-staging steps of ``CurationV2.insert_curation``.

``table_cls`` is ``CurationV2``. DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._curation.transforms import (
    normalize_label_state,
    validate_labels,
)
from spyglass.spikesorting.v2._core.lookup_validation import lossless_int
from spyglass.spikesorting.v2._core.enums import CurationSource


def find_matching_child_curation(
    table_cls,
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
    target_labels = normalize_label_state(effective_labels)
    target_merges = table_cls._normalized_real_merge_groups(
        kept_unit_to_contributors
    )
    candidates = (
        table_cls
        & {
            "sorting_id": sorting_id,
            "parent_curation_id": parent_curation_id,
            "merges_applied": bool(apply_merge),
            "curation_source": curation_source,
            "description": description,
        }
    ).fetch("KEY", as_dict=True, order_by="curation_id")
    for candidate in candidates:
        existing_labels = normalize_label_state(
            table_cls._labels_by_unit(candidate)
        )
        if existing_labels != target_labels:
            continue
        if (
            table_cls._normalized_real_merge_groups(
                table_cls.get_unit_contributor_groups(candidate)
            )
            != target_merges
        ):
            continue
        return candidate
    return None


def normalize_curation_inputs(
    labels: dict | None,
    curation_source: str | CurationSource,
    allow_custom_labels: bool,
) -> tuple[dict, str]:
    """Normalize + validate ``labels`` and ``curation_source``.

    Returns ``(labels, curation_source)`` with integer label keys,
    list values (``None`` becomes ``{}``), and the canonical
    ``CurationSource`` value. Raises ``ValueError`` on a scalar label
    value, an unrecognized label, a non-integer ID, or an invalid source.
    """
    if labels is None:
        labels = {}
    # ``list("mua")`` would split a scalar into per-character labels.
    for uid, lbls in labels.items():
        if isinstance(lbls, str) or not isinstance(lbls, (list, tuple)):
            raise ValueError(
                "CurationV2.insert_curation: labels[unit_id] must be a "
                f"list of labels; got {type(lbls).__name__} for "
                f"unit_id={uid}. Wrap a single label as a one-element "
                'list, e.g. {unit_id: ["mua"]}.'
            )
    labels = {
        lossless_int(uid, "label unit_id"): list(lbls)
        for uid, lbls in labels.items()
    }

    validate_labels(labels, allow_custom_labels=allow_custom_labels)

    # Coerce first so the existing-root reuse path cannot swallow a typo.
    try:
        curation_source = CurationSource(curation_source).value
    except ValueError as exc:
        raise ValueError(
            f"CurationV2.insert_curation: curation_source="
            f"{curation_source!r} is not a CurationSource value. "
            f"Valid: {[m.value for m in CurationSource]}."
        ) from exc

    return labels, curation_source


def validate_parent_or_reuse_root(
    table_cls,
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

    - Child: verifies the parent exists and is committed.
    - Root (``parent_curation_id == -1``): returns the existing root's key,
      or ``None`` for a fresh insert.

    Raises ``ValueError`` if the parent is missing or a preview, or if a
    root exists and non-default parameters were passed without
    ``reuse_existing=True``.
    """
    from spyglass.utils import logger

    if parent_curation_id != -1:
        parent_key = {
            "sorting_id": sorting_id,
            "curation_id": parent_curation_id,
        }
        if not (table_cls & parent_key):
            raise ValueError(
                f"CurationV2.insert_curation: parent_curation_id="
                f"{parent_curation_id} does not exist for sorting_id="
                f"{sorting_id}. Pass parent_curation_id=-1 for a "
                "root curation."
            )
        # A child of a preview would read as committed and inherit its
        # unapplied merges as applied.
        if not table_cls.is_committed_curation(parent_key):
            raise ValueError(
                f"CurationV2.insert_curation: parent_curation_id="
                f"{parent_curation_id} (sorting_id={sorting_id}) is a "
                "preview/draft curation (apply_merge=False with an "
                "unapplied proposed merge); a child cannot branch from a "
                "preview. Commit the proposed merge first "
                "(commit_merges / insert_curation(apply_merge=True)) or "
                "discard the preview, then branch from the "
                "committed curation."
            )
    else:
        # Reuse an existing root; order_by picks the lowest id if several.
        existing_root = (
            table_cls
            & {
                "sorting_id": sorting_id,
                "parent_curation_id": -1,
            }
        ).fetch("KEY", as_dict=True, order_by="curation_id")
        if existing_root:
            # Returning the root would silently ignore non-default params.
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


def resolve_curation_source(
    table_cls,
    *,
    sorting_id,
    parent_curation_id: int,
    raw_sorting_units: list[dict],
) -> tuple[list[dict], str | None, dict[int, list[int]] | None, dict]:
    """Resolve where a curation composes its units / trains / labels from.

    A root (``parent_curation_id == -1``) composes from the raw sort; a
    child from its parent ``CurationV2`` row. Returns:

    - ``source_units``: raw ``Sorting.Unit`` or parent ``CurationV2.Unit``
      rows.
    - ``source_units_abs_path``: the parent's units NWB (``None`` for root:
      the raw Sorting NWB).
    - ``parent_raw_contributors``: ``{parent_unit_id: [raw ids]}`` from the
      parent's ``MergeGroup`` (``None`` for root).
    - ``parent_labels``: ``{unit_id: [label, ...]}`` (``{}`` for root).
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    if parent_curation_id == -1:
        return raw_sorting_units, None, None, {}

    parent_key = {
        "sorting_id": sorting_id,
        "curation_id": parent_curation_id,
    }
    source_units = (table_cls.Unit & parent_key).fetch(
        as_dict=True, order_by="unit_id"
    )
    source_units_abs_path = AnalysisNwbfile.get_abs_path(
        (table_cls & parent_key).fetch1("analysis_file_name")
    )
    parent_raw_contributors = table_cls._raw_contributor_groups(parent_key)
    parent_labels = table_cls._labels_by_unit(parent_key)
    return (
        source_units,
        source_units_abs_path,
        parent_raw_contributors,
        parent_labels,
    )


def assert_child_reuse_has_parent(
    *,
    parent_curation_id: int,
    reuse_existing: bool,
    caller: str,
) -> None:
    """Refuse ``reuse_existing`` without a parent; it would reuse the root."""
    if not reuse_existing or parent_curation_id != -1:
        return
    raise ValueError(
        f"CurationV2.{caller}(reuse_existing=True) requires an "
        "explicit parent_curation_id. Root curation reuse returns the "
        "existing root row and would ignore the requested edits; "
        "branch from an existing curation instead."
    )
