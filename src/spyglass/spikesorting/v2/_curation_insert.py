"""Pre-staging steps of ``CurationV2.insert_curation``.

``insert_curation`` stays on ``CurationV2`` as the orchestrator: it stages the
curated-units NWB and then commits the rows in one transaction, retrying on a
concurrent ``curation_id`` collision. The steps before staging live here:
:func:`normalize_curation_inputs` validates the labels and source,
:func:`validate_parent_or_reuse_root` checks the parent (or returns an existing
root to reuse), :func:`resolve_curation_source` reads what the new curation
composes from, :func:`plan_curation_insert` allocates the ``curation_id`` and
builds the unit rows, and :func:`find_matching_child_curation` finds an
existing child with the same content (the body behind the patched
``CurationV2._find_matching_child_curation``, which ``insert_curation`` calls
through the class). ``table_cls`` is ``CurationV2``; ``plan_curation_insert``
calls its ``_next_curation_id`` through it.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._curation_transforms import (
    normalize_label_state,
    validate_labels,
)
from spyglass.spikesorting.v2._lookup_validation import lossless_int
from spyglass.spikesorting.v2.utils import CurationSource

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._curation_plan import CurationInsertPlan


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
        # ``None`` is semantically equivalent to "no labels".
        # Normalize to ``{}`` so the rest of the helper does not
        # need an extra None check.
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
    labels = {
        lossless_int(uid, "label unit_id"): list(lbls)
        for uid, lbls in labels.items()
    }

    validate_labels(labels, allow_custom_labels=allow_custom_labels)

    # Coerce curation_source through the enum up front so a typo raises a
    # friendly error on EVERY path -- including the idempotent
    # existing-root early return below, which would otherwise swallow an
    # invalid value and return the existing root instead of rejecting it.
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

    For a child curation (``parent_curation_id != -1``) verifies the
    parent row exists. For a root curation (``parent_curation_id ==
    -1``) returns the canonical existing-root key when one already
    exists (idempotent re-insert), or ``None`` to proceed with a fresh
    insert. Raises ``ValueError`` if the named parent is missing, or if
    a root already exists and the caller passed non-default parameters
    without ``reuse_existing=True``.
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
        # Reject building on a PREVIEW/draft parent. A preview
        # (apply_merge=False with an unapplied proposed merge) is not a
        # committed state: a child of it would get self-only
        # ParentMergeGroup rows -- so get_unit_contributor_groups /
        # has_unapplied_proposed_merges would read the child as committed --
        # AND its raw MergeGroup would inherit the parent's UNAPPLIED
        # proposed merge as though it had been applied. Either way the
        # draft is silently laundered into a committed-looking curation.
        # Commit the draft (create_merged_curation /
        # insert_curation(apply_merge=True) on the proposed groups) or
        # discard it, then branch from the committed curation.
        if not table_cls.is_committed_curation(parent_key):
            raise ValueError(
                f"CurationV2.insert_curation: parent_curation_id="
                f"{parent_curation_id} (sorting_id={sorting_id}) is a "
                "preview/draft curation (apply_merge=False with an "
                "unapplied proposed merge); a child cannot branch from a "
                "preview. Commit the proposed merge first "
                "(create_merged_curation / insert_curation(apply_merge="
                "True)) or discard the preview, then branch from the "
                "committed curation."
            )
    else:
        # Idempotency: if a root curation already exists for this
        # sorting_id, return its key without staging a new NWB or
        # creating a duplicate row. Without this, every repeat
        # call grows another row + analysis file + merge-table
        # entry.
        # ``order_by="curation_id"`` makes the reused root deterministic:
        # if more than one root somehow exists (a raw/manual insert that
        # bypassed this helper), the canonical lowest-curation_id root is
        # returned every time rather than a DB-row-order coin flip.
        existing_root = (
            table_cls
            & {
                "sorting_id": sorting_id,
                "parent_curation_id": -1,
            }
        ).fetch("KEY", as_dict=True, order_by="curation_id")
        if existing_root:
            # A root curation already exists. If the caller passed ANY
            # non-default parameter -- labels / merge_groups / description
            # / apply_merge / a non-"manual" curation_source -- it would be
            # SILENTLY ignored by returning the existing row, a
            # parameter-change-with-no-effect footgun. The apply_merge /
            # curation_source cases matter because ``merges_applied`` and
            # the curation provenance record user intent: a second
            # ``apply_merge=True`` insert must NOT quietly return a
            # ``merges_applied=False`` root. Raise unless the caller opts
            # into reuse. curation_source is already coerced to its enum
            # value above, so a typo has already raised before reaching
            # this point (it is never silently treated as "unchanged").
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

    A root curation (``parent_curation_id == -1``) composes from the raw
    sort; a child composes from its PARENT ``CurationV2`` row. Returns
    ``(source_units, source_units_abs_path, parent_raw_contributors,
    parent_labels)``:

    * ``source_units`` -- the source ``Unit`` rows the merge/label plan is
      built over (raw ``Sorting.Unit`` for a root, parent
      ``CurationV2.Unit`` for a child). Both shapes carry ``unit_id`` + the
      peak ``Electrode`` FK + ``peak_amplitude_uv`` + ``n_spikes``.
    * ``source_units_abs_path`` -- the units NWB to read source spike
      trains from (``None`` for a root -> the raw Sorting NWB; the parent
      curation's NWB for a child).
    * ``parent_raw_contributors`` -- ``{parent_unit_id: [raw contributor
      ids]}`` from the PARENT's ``MergeGroup`` (always raw by invariant),
      so a child's parent-namespace contributors can be expanded back to
      raw provenance. ``None`` for a root (its contributors are already
      raw).
    * ``parent_labels`` -- the parent's ``{unit_id: [label, ...]}`` for
      label inheritance (``{}`` for a root).
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


def plan_curation_insert(
    table_cls,
    sorting_id,
    sorting_units: list[dict],
    merge_groups,
    apply_merge: bool,
    labels: dict,
    allow_unknown_unit_ids: bool,
    parent_labels: dict | None = None,
    label_policy: str = "inherit",
) -> CurationInsertPlan:
    """Resolve the curation_id and build the curated ``Unit`` rows.

    Resolves the auto-increment ``curation_id`` (the one DB-coupled
    step) and delegates row shaping, label composition, and label-key
    validation to the DB-free
    :func:`._curation_plan.build_curation_insert_plan`. ``sorting_units``
    are the SOURCE units (raw for a root, parent ``CurationV2.Unit`` for a
    child); ``parent_labels`` + ``label_policy`` drive label inheritance.
    Returns the :class:`._curation_plan.CurationInsertPlan` namedtuple
    ``(curation_id, unit_rows, kept_unit_to_contributors, labels)``, which
    unpacks positionally.
    """
    from spyglass.spikesorting.v2._curation_plan import (
        build_curation_insert_plan,
    )

    # Resolve which curation_id to use (auto-increment within sort) --
    # the one DB-coupled step; the rest of the plan is pure.
    curation_id = table_cls._next_curation_id(sorting_id)

    return build_curation_insert_plan(
        sorting_id=sorting_id,
        sorting_units=sorting_units,
        merge_groups=merge_groups,
        apply_merge=apply_merge,
        labels=labels,
        allow_unknown_unit_ids=allow_unknown_unit_ids,
        curation_id=curation_id,
        parent_labels=parent_labels,
        label_policy=label_policy,
    )


def assert_child_reuse_for_merge_wrapper(
    *,
    parent_curation_id: int,
    reuse_existing: bool,
    wrapper_name: str,
) -> None:
    """Prevent merge wrappers from reusing an unrelated root curation."""
    if not reuse_existing or parent_curation_id != -1:
        return
    raise ValueError(
        f"CurationV2.{wrapper_name}(reuse_existing=True) requires an "
        "explicit parent_curation_id. Root curation reuse returns the "
        "existing root row and would ignore the requested merge/proposal; "
        "branch from an existing curation instead."
    )
