"""Acceptance workflow behind ``CurationEvaluation``'s acceptance verbs.

Turns an evaluation's outputs into a child ``CurationV2`` row. ``table`` is
the ``CurationEvaluation`` instance. DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._core.lookup_validation import lossless_int


def evaluated_curation_key(key) -> dict:
    """Resolve a CurationEvaluation key to its evaluated curation key.

    Requires a populated evaluation, even with explicit labels or merges,
    since children are tagged ``curation_source='curation_evaluation'``.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    if not (CurationEvaluation & key):
        raise ValueError(
            "CurationEvaluation acceptance requires a POPULATED evaluation "
            f"(curation_source='curation_evaluation'); no CurationEvaluation "
            f"row for {dict(key)}. Call CurationEvaluation.populate(key) "
            "before accepting (accept_evaluation_outputs / accept_merges / "
            "preview_merges / use_evaluation_labels / ...)."
        )
    sel = (CurationEvaluationSelection & key).fetch1()
    return {
        "sorting_id": sel["sorting_id"],
        "curation_id": int(sel["curation_id"]),
    }


def resolve_accepted_merges(
    table, key, merge_groups, use_all_suggested_merges
) -> list[list[int]]:
    """Resolve the merge groups to accept (explicit, all-suggested, none).

    Suggestions apply only with ``use_all_suggested_merges=True`` and are
    filtered to >=2-member groups. Caller groups are kept as-is so
    ``insert_curation``'s typo guard sees singletons.
    """
    if merge_groups is not None and use_all_suggested_merges:
        raise ValueError(
            "CurationEvaluation acceptance: pass either merge_groups or "
            "use_all_suggested_merges=True, not both."
        )
    if use_all_suggested_merges:
        return [
            [int(u) for u in group]
            for group in table.get_suggested_merge_groups(key)
            if len(group) >= 2
        ]
    if merge_groups is not None:
        return [
            [lossless_int(u, "merge group unit_id") for u in group]
            for group in merge_groups
        ]
    return []


def accept_evaluation_outputs(
    table,
    key,
    *,
    merge_groups=None,
    use_all_suggested_merges: bool = False,
    labels: dict | None = None,
    label_policy: str = "replace",
    description: str = "accepted from curation evaluation",
    allow_custom_labels: bool = False,
    reuse_existing: bool = True,
) -> dict:
    """Accept selected evaluation outputs into a COMMITTED child curation.

    The body of ``CurationEvaluation.accept_evaluation_outputs`` (see its
    docstring for the arguments).
    """
    curation_key = evaluated_curation_key(key)
    accepted = resolve_accepted_merges(
        table, key, merge_groups, use_all_suggested_merges
    )
    return _insert_evaluation_child(
        table,
        key,
        curation_key,
        accepted,
        labels,
        apply_merge=bool(accepted),
        label_policy=label_policy,
        description=description,
        allow_custom_labels=allow_custom_labels,
        reuse_existing=reuse_existing,
    )


def require_merge_acceptance(
    table,
    key,
    merge_groups,
    use_all_suggested_merges: bool,
    *,
    action_name: str,
) -> list[list[int]]:
    """Resolve merge groups for an action method and require a real merge.

    Checks population first so an unpopulated selection gets a clear error.
    """
    evaluated_curation_key(key)
    accepted = resolve_accepted_merges(
        table, key, merge_groups, use_all_suggested_merges
    )
    if not accepted:
        raise ValueError(
            f"CurationEvaluation.{action_name} needs at least one merge "
            "group. Pass merge_groups=[[...]] or use a selection with "
            "persisted merge suggestions."
        )
    return accepted


def create_preview_curation(
    table,
    key,
    *,
    merge_groups=None,
    use_all_suggested_merges: bool = False,
    labels: dict | None = None,
    label_policy: str = "replace",
    description: str = "draft from curation evaluation",
    allow_custom_labels: bool = False,
    reuse_existing: bool = True,
) -> dict:
    """Create a DRAFT (preview) child from the evaluation's outputs.

    Records the merges unapplied (``apply_merge=False``); downstream
    consumers reject the child until committed. Raises ``ValueError`` if no
    merge resolves. Returns the child's ``{"sorting_id", "curation_id"}``.
    """
    curation_key = evaluated_curation_key(key)
    accepted = resolve_accepted_merges(
        table, key, merge_groups, use_all_suggested_merges
    )
    if not accepted:
        raise ValueError(
            "preview_merges drafts an UNAPPLIED merge for "
            "review, so it needs at least one merge: pass "
            "merge_groups=[[...]] or use_all_suggested_merges=True (with "
            "proposed merges present). For a committed labels-only child, "
            "call use_evaluation_labels() or overlay_evaluation_labels() "
            "instead."
        )
    return _insert_evaluation_child(
        table,
        key,
        curation_key,
        accepted,
        labels,
        apply_merge=False,
        label_policy=label_policy,
        description=description,
        allow_custom_labels=allow_custom_labels,
        reuse_existing=reuse_existing,
    )


def _insert_evaluation_child(
    table,
    key,
    curation_key: dict,
    accepted: list[list[int]],
    labels: dict | None,
    *,
    apply_merge: bool,
    label_policy: str,
    description: str,
    allow_custom_labels: bool,
    reuse_existing: bool,
) -> dict:
    """Insert an evaluation-sourced child of the evaluated curation.

    ``labels=None`` takes the evaluation's proposed labels
    (``table.get_labels``). Returns the child's
    ``{"sorting_id", "curation_id"}``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    effective_labels = table.get_labels(key) if labels is None else labels
    return CurationV2.insert_curation(
        {"sorting_id": curation_key["sorting_id"]},
        labels=effective_labels or None,
        merge_groups=accepted or None,
        apply_merge=apply_merge,
        parent_curation_id=curation_key["curation_id"],
        description=description,
        curation_source="curation_evaluation",
        label_policy=label_policy,
        allow_custom_labels=allow_custom_labels,
        reuse_existing=reuse_existing,
    )
