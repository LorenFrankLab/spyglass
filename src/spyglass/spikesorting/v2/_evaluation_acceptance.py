"""Acceptance workflow behind ``CurationEvaluation``'s acceptance verbs.

The public verbs (``accept_evaluation_outputs``, ``preview_merges``,
``accept_merges``, ``accept_all_suggested_merges``, ``use_evaluation_labels``,
``overlay_evaluation_labels``) stay on ``CurationEvaluation``; this module turns
an evaluation's outputs into a child ``CurationV2`` row. It requires a
populated evaluation (:func:`evaluated_curation_key`), resolves which merges
were accepted (:func:`resolve_accepted_merges`), and inserts a committed child
(:func:`accept_evaluation_outputs`) or a draft with unapplied merges
(:func:`create_preview_curation`). ``table`` is the ``CurationEvaluation``
instance whose ``get_labels`` / ``get_suggested_merge_groups`` supply the
proposals.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations


def evaluated_curation_key(key) -> dict:
    """Resolve a CurationEvaluation key to its evaluated curation key.

    Requires the ``CurationEvaluation`` row to be POPULATED, not merely
    selected. Acceptance writes ``curation_source='curation_evaluation'``
    children, so the workflow contract is "evaluate, THEN accept": minting
    an evaluation-sourced child from a bare selection (no computed
    metrics/proposals) would be a provenance lie. The check holds even when
    labels / merge groups are supplied explicitly -- the provenance tag
    claims an evaluation backs the child regardless of how the merges/labels
    were chosen.
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

    Never applies all suggested merges implicitly: the caller must pass an
    explicit ``merge_groups`` OR ``use_all_suggested_merges=True``.

    CALLER-SUPPLIED ``merge_groups`` are returned VERBATIM (only coerced to
    ints) -- they are NOT silently filtered, so a singleton/empty group
    reaches ``CurationV2.insert_curation``'s >=2-member typo guard and
    raises instead of degrading into a labels-only child. Only the PERSISTED
    suggestions (``use_all_suggested_merges``) are filtered to the real
    (>=2-member) groups, since the stored suggestion set is not a caller
    typo. All ids are in the evaluated curation's own unit namespace.
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
        return [[int(u) for u in group] for group in merge_groups]
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

    Enforces the populated-evaluation contract BEFORE resolving suggestions:
    the ``use_all_suggested_merges`` path reads the evaluation NWB via
    ``get_suggested_merge_groups``, which on an unpopulated selection would fail with
    an opaque fetch error instead of the friendly "populate first" message.
    """
    evaluated_curation_key(key)  # populated-evaluation guard
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

    The explicit opt-in for drafting a merge for review before committing:
    the proposed merges are recorded in ``CurationV2.MergeGroup`` WITHOUT
    being applied (``apply_merge=False``), so the child is a preview --
    ``has_unapplied_proposed_merges`` is True and downstream consumers
    reject it until it is committed. Distinct from
    ``CurationEvaluation.accept_evaluation_outputs``, which only ever
    produces committed children. A preview is, by
    definition, an UNAPPLIED merge for review, so it must actually draft a
    merge: pass ``merge_groups`` or ``use_all_suggested_merges=True`` (and
    the latter must resolve at least one merge). With no merge this would
    otherwise produce a normal committed labels-only child, contradicting
    the "preview/draft" contract -- so it raises instead. For a committed
    labels-only child use ``CurationEvaluation.use_evaluation_labels`` /
    ``CurationEvaluation.overlay_evaluation_labels``.

    Returns the child's ``{"sorting_id", "curation_id"}``.
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
