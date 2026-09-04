"""Browser-first, identity-verified curation review facade.

The durable state is deliberately limited to existing immutable rows: a review
profile, evaluation, content-addressed FigPack selection, and figure bundle.
These value objects reconstruct progress from those rows and never add mutable
workflow-status schema.
"""

from __future__ import annotations

import uuid
import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from spyglass.spikesorting.v2._figpack_curation import (
    annotations_payload_hash,
    curation_annotations_to_labels_and_merges,
    unpack_display_config,
)
from spyglass.spikesorting.v2.curation_api import (
    CurationRef,
    EvaluationResult,
    EvaluationSpec,
)
from spyglass.spikesorting.v2.exceptions import (
    FigPackIdentityError,
    ReviewChangedSincePreviewError,
    UnresolvedMergeLabelConflictError,
)

ReviewStageState = Literal["computed", "reused", "complete"]


def _uuid(value) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _label_snapshot(
    labels: Mapping[int, Sequence[str]],
) -> Mapping[int, tuple[str, ...]]:
    """Return a stable, read-only labels mapping with empty values omitted."""
    return MappingProxyType(
        {
            int(unit_id): tuple(sorted(map(str, unit_labels)))
            for unit_id, unit_labels in sorted(labels.items())
            if unit_labels
        }
    )


def _profile_snapshot(ref: "ReviewProfileRef") -> dict:
    return {
        "review_profile_name": ref.review_profile_name,
        "profile_hash": ref.profile_hash,
        "evaluation_spec": {
            "metric_params_name": ref.evaluation_spec.metric_params_name,
            "auto_curation_rules_name": (
                ref.evaluation_spec.auto_curation_rules_name
            ),
        },
        "displayed_unit_properties": list(ref.displayed_unit_properties),
        "label_options": list(ref.label_options),
        "label_import_mode": ref.label_import_mode,
    }


@dataclass(frozen=True)
class ReviewProfileRef:
    """Resolved immutable browser-review profile."""

    review_profile_name: str
    profile_hash: str
    evaluation_spec: EvaluationSpec
    displayed_unit_properties: tuple[str, ...]
    label_options: tuple[str, ...]
    label_import_mode: Literal["replace", "overlay"]

    @classmethod
    def resolve(cls, profile: str | "ReviewProfileRef") -> "ReviewProfileRef":
        """Resolve and recheck a persisted profile by name."""
        from spyglass.spikesorting.v2.review_profile import (
            CurationReviewProfile,
        )

        name = (
            profile.review_profile_name
            if isinstance(profile, cls)
            else str(profile)
        )
        rows = (CurationReviewProfile & {"review_profile_name": name}).fetch(
            as_dict=True
        )
        if len(rows) != 1:
            raise ValueError(
                f"Review profile {name!r} does not resolve to exactly one "
                "CurationReviewProfile row."
            )
        row = rows[0]
        resolved = cls(
            review_profile_name=str(row["review_profile_name"]),
            profile_hash=str(row["profile_hash"]),
            evaluation_spec=EvaluationSpec(
                metric_params_name=str(row["metric_params_name"]),
                auto_curation_rules_name=str(row["auto_curation_rules_name"]),
            ),
            displayed_unit_properties=tuple(
                map(str, row["displayed_unit_properties"])
            ),
            label_options=tuple(map(str, row["label_options"])),
            label_import_mode=str(row["label_import_mode"]),
        )
        if isinstance(profile, cls) and resolved != profile:
            raise ValueError(
                f"Review profile {name!r} no longer matches the pinned "
                f"profile_hash={profile.profile_hash}."
            )
        return resolved


@dataclass(frozen=True)
class ReviewStageStatus:
    """Named status derived from durable review rows."""

    name: str
    status: ReviewStageState


@dataclass(frozen=True)
class MergeLabelConflict:
    """Incompatible contributor labels for one predicted merged unit."""

    merged_unit_id: int
    contributor_unit_ids: tuple[int, ...]
    contributor_labels: Mapping[int, tuple[str, ...]]

    def __post_init__(self):
        object.__setattr__(
            self,
            "contributor_labels",
            MappingProxyType(dict(self.contributor_labels)),
        )


def _unknown_conflict_resolution_labels(
    values: Sequence[str],
    valid_labels: set[str],
    conflict: MergeLabelConflict,
) -> list[str]:
    """Return labels newly introduced by a merge-conflict resolution.

    A profile palette controls labels the reviewer may add. It does not erase
    valid labels inherited from the merged contributors, even when an older or
    site-specific label is absent from the active profile.
    """
    inherited_labels = {
        label
        for labels in conflict.contributor_labels.values()
        for label in labels
    }
    return sorted(set(map(str, values)) - valid_labels - inherited_labels)


@dataclass(frozen=True)
class FigPackReview:
    """Resumable handle for one profile-backed FigPack review."""

    review_id: uuid.UUID
    parent: CurationRef
    profile: ReviewProfileRef
    evaluation: EvaluationResult
    uri: str
    upload: bool
    ephemeral: bool
    annotation_sets: tuple[Any, ...] = field(default_factory=tuple)
    stages: tuple[ReviewStageStatus, ...] = field(default_factory=tuple)
    _started_sibling_uuids: tuple[uuid.UUID, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )

    def open(self) -> None:
        """Open the hosted URL or local bundle in the default browser."""
        target = self.uri
        if not target.startswith(("http://", "https://", "file://")):
            target = Path(target).resolve().as_uri()
        webbrowser.open(target)

    def preview_import(self) -> "CurationChangeSet":
        """Read, verify, and diff browser edits without mutating state."""
        return _preview_review(self)

    @classmethod
    def resume(cls, review_id) -> "FigPackReview":
        """Reconstruct a review from its selection, bundle, and profile."""
        from spyglass.spikesorting.v2.figpack_curation import (
            FigPackCuration,
            FigPackCurationSelection,
            _assert_figure_identity,
            _assert_selection_identity,
            _require_figpack,
        )

        _require_figpack()
        key = {"figpack_curation_id": _uuid(review_id)}
        selections = (FigPackCurationSelection & key).fetch(as_dict=True)
        if len(selections) != 1 or not (FigPackCuration & key):
            raise LookupError(
                f"FigPackReview.resume({review_id}) requires one built "
                "FigPackCuration selection."
            )
        selection = selections[0]
        _assert_selection_identity(selection, key)
        properties, config = unpack_display_config(
            selection["displayed_unit_properties"]
        )
        if config is None:
            raise FigPackIdentityError(
                f"FigPack selection {review_id} is an expert view, not a "
                "profile-backed resumable review."
            )
        parent = CurationRef.from_key(selection)
        uri = str((FigPackCuration & key).fetch1("figpack_uri"))
        figure_config = _assert_figure_identity(
            uri,
            parent.as_key(),
            expected_config_hash=str(selection["figpack_config_hash"]),
        )
        if figure_config.get("review") != config:
            raise FigPackIdentityError(
                "FigPack figure review snapshot differs from its persisted "
                "selection configuration."
            )
        profile = ReviewProfileRef.resolve(config["review_profile_name"])
        expected_profile = _profile_snapshot(profile)
        try:
            embedded_profile = {key: config[key] for key in expected_profile}
        except KeyError as exc:
            raise FigPackIdentityError(
                "Persisted FigPack review configuration is missing a profile "
                f"snapshot field: {exc.args[0]}."
            ) from exc
        if embedded_profile != expected_profile:
            raise FigPackIdentityError(
                "Persisted FigPack review configuration no longer matches its "
                "immutable CurationReviewProfile snapshot."
            )
        if list(properties or []) != list(profile.displayed_unit_properties):
            raise FigPackIdentityError(
                "FigPack selection display properties differ from its review "
                "profile snapshot."
            )
        evaluation = EvaluationResult.from_key(
            {"curation_evaluation_id": config["curation_evaluation_id"]}
        )
        if evaluation.curation != parent or evaluation.spec != (
            profile.evaluation_spec
        ):
            raise FigPackIdentityError(
                "FigPack review evaluation does not match its pinned parent "
                "and profile recipe."
            )
        delivery = config["delivery"]
        from spyglass.spikesorting.v2.unit_annotation import AnnotationSetRef

        annotation_sets = tuple(
            AnnotationSetRef.from_snapshot(snapshot)
            for snapshot in config.get("annotation_sets", [])
        )
        return cls(
            review_id=_uuid(review_id),
            parent=parent,
            profile=profile,
            evaluation=evaluation,
            uri=uri,
            upload=bool(delivery["upload"]),
            ephemeral=bool(delivery["ephemeral"]),
            annotation_sets=annotation_sets,
            stages=(
                ReviewStageStatus("identity_verified", "complete"),
                ReviewStageStatus("evaluation_populated", "reused"),
                ReviewStageStatus("verification_view_ready", "reused"),
            ),
            _started_sibling_uuids=tuple(
                _uuid(value)
                for value in config.get("sibling_uuids_at_start", [])
            ),
        )


@dataclass(frozen=True)
class CurationChangeSet:
    """Pure, identity-pinned preview of annotations waiting to import."""

    review: FigPackReview
    annotations_hash: str
    labels_before: Mapping[int, tuple[str, ...]]
    labels_after: Mapping[int, tuple[str, ...]]
    merge_groups: tuple[tuple[int, ...], ...]
    unit_count_before: int
    unit_count_after: int
    label_conflicts: tuple[MergeLabelConflict, ...]
    newer_sibling_curations: tuple[CurationRef, ...]
    reviewed_parent_created_at: Any
    reviewed_parent_created_by: str

    def commit(
        self,
        *,
        conflict_resolutions: Mapping[int, tuple[str, ...]] | None = None,
        confirm_no_changes: bool = False,
    ) -> "ReviewImportReceipt":
        """Commit exactly this preview after a UUID + annotation-hash recheck."""
        return _commit_change_set(
            self,
            conflict_resolutions=conflict_resolutions,
            confirm_no_changes=confirm_no_changes,
        )


@dataclass(frozen=True)
class ReviewImportReceipt:
    """Committed child, optional post-merge evaluation, and restart stages."""

    curation: CurationRef
    evaluation: EvaluationResult | None
    changes: CurationChangeSet
    warnings: tuple[str, ...]
    stages: tuple[ReviewStageStatus, ...]
    needs_merge_verification: bool

    @property
    def created_at(self):
        """Creation timestamp of the committed/reused child."""
        return self.curation.created_at

    @property
    def created_by(self) -> str:
        """Database user recorded for the committed/reused child."""
        return self.curation.created_by

    def continue_review(self) -> FigPackReview:
        """Open/reuse a verification review over the actual child.

        Parent annotation sets are intentionally not carried across the new
        curation identity. Compute/select child-scoped sets explicitly after a
        merge or label commit.
        """
        return start_review(
            self.curation,
            self.changes.review.profile,
            upload=self.changes.review.upload,
            ephemeral=self.changes.review.ephemeral,
            evaluation=self.evaluation,
        )


def _review_config(
    parent: CurationRef,
    profile: ReviewProfileRef,
    evaluation: EvaluationResult,
    *,
    upload: bool,
    ephemeral: bool,
    annotation_sets: Sequence[Any],
) -> dict:
    config = _profile_snapshot(profile)
    config.update(
        {
            "curation_evaluation_id": str(evaluation.evaluation_id),
            "delivery": {
                "upload": bool(upload),
                "ephemeral": bool(ephemeral if upload else False),
            },
            "sibling_uuids_at_start": [
                str(child.curation_uuid) for child in parent.children
            ],
        }
    )
    # Preserve byte-for-byte Phase-3 identity when no custom set is selected.
    if annotation_sets:
        config["annotation_sets"] = [ref.snapshot() for ref in annotation_sets]
    return config


def _resolve_annotation_sets(
    parent: CurationRef, annotation_sets: Sequence[Any]
) -> tuple[Any, ...]:
    """Resolve exact set refs and reject cross-curation selections."""
    from spyglass.spikesorting.v2.unit_annotation import AnnotationSetRef

    resolved = tuple(
        AnnotationSetRef.from_key(value) for value in annotation_sets
    )
    for ref in resolved:
        if ref.curation != parent:
            raise ValueError(
                f"Annotation set {ref.set_hash} belongs to a different "
                "curation than the review parent."
            )
    if len({ref.as_key()["set_hash"] for ref in resolved}) != len(resolved):
        raise ValueError("annotation_sets contains a duplicate set reference.")
    return resolved


def start_review(
    parent: CurationRef,
    profile: str | ReviewProfileRef,
    *,
    upload: bool = False,
    ephemeral: bool = False,
    evaluation: EvaluationResult | None = None,
    annotation_sets: Sequence[Any] = (),
) -> FigPackReview:
    """Evaluate one pinned curation and build/reuse its seeded review view."""
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCuration,
        FigPackCurationSelection,
        _require_figpack,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    _require_figpack()
    if not isinstance(parent, CurationRef):
        raise TypeError(
            "start_review requires parent=CurationRef; resolve a key with "
            "CurationRef.from_key first."
        )
    parent.as_key()
    resolved_annotation_sets = _resolve_annotation_sets(parent, annotation_sets)
    resolved_profile = ReviewProfileRef.resolve(profile)
    spec = resolved_profile.evaluation_spec
    evaluation_identity = {
        **parent.as_key(),
        "metric_params_name": spec.metric_params_name,
        "auto_curation_rules_name": spec.auto_curation_rules_name,
    }
    existing_evaluation_ids = (
        CurationEvaluationSelection & evaluation_identity
    ).fetch("curation_evaluation_id")
    evaluation_existed = any(
        CurationEvaluation & {"curation_evaluation_id": evaluation_id}
        for evaluation_id in existing_evaluation_ids
    )
    if evaluation is None:
        evaluation = parent.evaluate(
            metric_params_name=spec.metric_params_name,
            auto_curation_rules_name=spec.auto_curation_rules_name,
        )
    elif evaluation.curation != parent or evaluation.spec != spec:
        raise ValueError(
            "EvaluationResult.start_review requires an evaluation over this "
            "same curation using the chosen profile's exact recipe names."
        )
    evaluation_populated = bool(
        CurationEvaluation
        & {"curation_evaluation_id": evaluation.evaluation_id}
    )
    if not evaluation_populated:
        raise LookupError(
            "start_review requires a populated CurationEvaluation."
        )

    config = _review_config(
        parent,
        resolved_profile,
        evaluation,
        upload=upload,
        ephemeral=ephemeral,
        annotation_sets=resolved_annotation_sets,
    )
    selection = FigPackCurationSelection.insert_selection(
        parent.as_key(),
        label_options=list(resolved_profile.label_options),
        displayed_unit_properties=list(
            resolved_profile.displayed_unit_properties
        ),
        upload=upload,
        ephemeral=ephemeral,
        review_config=config,
    )
    view_result = FigPackCuration.build_curation_view_result(
        parent.as_key(),
        label_options=list(resolved_profile.label_options),
        displayed_unit_properties=list(
            resolved_profile.displayed_unit_properties
        ),
        upload=upload,
        ephemeral=ephemeral,
        review_config=config,
    )
    review = FigPackReview.resume(selection["figpack_curation_id"])
    return replace(
        review,
        uri=view_result.uri,
        stages=(
            ReviewStageStatus("identity_verified", "complete"),
            ReviewStageStatus(
                "evaluation_populated",
                "reused" if evaluation_existed else "computed",
            ),
            ReviewStageStatus(
                "verification_view_ready",
                "reused" if view_result.reused else "computed",
            ),
        ),
    )


def _normalize_review_edits(
    review: FigPackReview, labels: dict, merge_groups: list
) -> tuple[
    Mapping[int, tuple[str, ...]],
    tuple[tuple[int, ...], ...],
    tuple[MergeLabelConflict, ...],
    int,
]:
    """Validate edits and compute effective parent labels + merge conflicts."""
    from spyglass.spikesorting.v2.curation import CurationV2

    parent_key = review.parent.as_key()
    unit_ids = {
        int(value) for value in (CurationV2.Unit & parent_key).fetch("unit_id")
    }
    before = {
        int(unit_id): tuple(sorted(map(str, values)))
        for unit_id, values in CurationV2._labels_by_unit(parent_key).items()
        if values
    }
    edited: dict[int, tuple[str, ...]] = {}
    valid_labels = set(review.profile.label_options)
    for unit_id, unit_labels in labels.items():
        uid = int(unit_id)
        values = tuple(sorted(map(str, unit_labels)))
        if len(values) != len(set(values)):
            raise ValueError(
                f"FigPack annotations for unit {uid} contain duplicate labels."
            )
        # FigPack seeds every existing parent label into the annotations even
        # when the active review profile does not offer that label as a new
        # choice. Preserve those inherited labels on their original unit while
        # still rejecting an out-of-palette label newly added to another unit.
        inherited_labels = set(before.get(uid, ()))
        unknown_labels = sorted(set(values) - valid_labels - inherited_labels)
        if unknown_labels:
            raise ValueError(
                f"FigPack annotations for unit {uid} contain new labels "
                "outside the review profile palette: "
                f"{unknown_labels}."
            )
        edited[uid] = values

    normalized_groups = []
    seen: set[int] = set()
    for group in merge_groups:
        members = tuple(sorted(map(int, group)))
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(
                "FigPack merge groups must contain at least two distinct unit "
                f"ids; got {list(group)}."
            )
        overlap = seen & set(members)
        if overlap:
            raise ValueError(
                "FigPack merge groups must be disjoint; repeated unit ids: "
                f"{sorted(overlap)}."
            )
        seen.update(members)
        normalized_groups.append(members)
    normalized_groups.sort(key=min)
    referenced = set(edited) | seen
    unknown_units = sorted(referenced - unit_ids)
    if unknown_units:
        raise ValueError(
            "FigPack annotations reference units absent from the pinned "
            f"curation: {unknown_units}. Available units: {sorted(unit_ids)}."
        )

    if review.profile.label_import_mode == "replace":
        after = {uid: values for uid, values in edited.items() if values}
    else:
        after = dict(before)
        for uid, values in edited.items():
            if values:
                after[uid] = values
            else:
                after.pop(uid, None)

    conflicts = []
    next_id = max(unit_ids) + 1 if unit_ids else 0
    for offset, group in enumerate(normalized_groups):
        contributor_labels = {uid: after.get(uid, ()) for uid in group}
        if len(set(contributor_labels.values())) > 1:
            conflicts.append(
                MergeLabelConflict(
                    merged_unit_id=next_id + offset,
                    contributor_unit_ids=group,
                    contributor_labels=contributor_labels,
                )
            )
    unit_count_after = len(unit_ids) - sum(
        len(group) - 1 for group in normalized_groups
    )
    return (
        MappingProxyType(dict(sorted(after.items()))),
        tuple(normalized_groups),
        tuple(conflicts),
        unit_count_after,
    )


def _preview_review(review: FigPackReview) -> CurationChangeSet:
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCurationSelection,
        _assert_figure_identity,
        _load_annotations_json,
    )

    parent_key = review.parent.as_key()
    selection = (
        FigPackCurationSelection & {"figpack_curation_id": review.review_id}
    ).fetch1()
    _assert_figure_identity(
        review.uri,
        parent_key,
        expected_config_hash=str(selection["figpack_config_hash"]),
    )
    annotations = _load_annotations_json(review.uri)
    labels, groups = curation_annotations_to_labels_and_merges(annotations)
    labels_after, groups, conflicts, count_after = _normalize_review_edits(
        review, labels, groups
    )
    before = _label_snapshot(CurationV2._labels_by_unit(parent_key))
    current_siblings = {
        child.curation_uuid: child for child in review.parent.children
    }
    newer = tuple(
        current_siblings[value]
        for value in sorted(current_siblings, key=str)
        if value not in set(review._started_sibling_uuids)
    )
    return CurationChangeSet(
        review=review,
        annotations_hash=annotations_payload_hash(annotations),
        labels_before=before,
        labels_after=labels_after,
        merge_groups=groups,
        unit_count_before=len((CurationV2.Unit & parent_key).fetch("unit_id")),
        unit_count_after=count_after,
        label_conflicts=conflicts,
        newer_sibling_curations=newer,
        reviewed_parent_created_at=review.parent.created_at,
        reviewed_parent_created_by=review.parent.created_by,
    )


def _child_labels(
    changes: CurationChangeSet,
    resolutions: Mapping[int, tuple[str, ...]],
) -> dict[int, list[str]]:
    """Translate effective parent labels into the committed child namespace."""
    from spyglass.spikesorting.v2.curation import CurationV2

    parent_units = {
        int(value)
        for value in (CurationV2.Unit & changes.review.parent.as_key()).fetch(
            "unit_id"
        )
    }
    absorbed = {unit_id for group in changes.merge_groups for unit_id in group}
    labels = {
        unit_id: list(values)
        for unit_id, values in changes.labels_after.items()
        if unit_id not in absorbed and values
    }
    next_id = max(parent_units) + 1 if parent_units else 0
    conflicts = {
        conflict.merged_unit_id: conflict
        for conflict in changes.label_conflicts
    }
    for offset, group in enumerate(changes.merge_groups):
        merged_id = next_id + offset
        if merged_id in conflicts:
            values = tuple(resolutions[merged_id])
        else:
            values = changes.labels_after.get(group[0], ())
        if values:
            labels[merged_id] = list(values)
    return labels


def _commit_change_set(
    changes: CurationChangeSet,
    *,
    conflict_resolutions: Mapping[int, tuple[str, ...]] | None,
    confirm_no_changes: bool,
) -> ReviewImportReceipt:
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCurationSelection,
        _assert_figure_identity,
        _load_annotations_json,
    )

    parent_key = changes.review.parent.as_key()
    if CurationV2.connection.in_transaction:
        raise RuntimeError(
            "CurationChangeSet.commit must be called outside an open DataJoint "
            "transaction; curation and evaluation manage their own writes."
        )
    selection = (
        FigPackCurationSelection
        & {"figpack_curation_id": changes.review.review_id}
    ).fetch1()
    _assert_figure_identity(
        changes.review.uri,
        parent_key,
        expected_config_hash=str(selection["figpack_config_hash"]),
    )
    annotations = _load_annotations_json(changes.review.uri)
    current_hash = annotations_payload_hash(annotations)
    if current_hash != changes.annotations_hash:
        raise ReviewChangedSincePreviewError(
            "FigPack annotations changed after preview for review_id="
            f"{changes.review.review_id}. Run preview_import() again before "
            "committing."
        )
    labels, groups = curation_annotations_to_labels_and_merges(annotations)
    current_after, current_groups, current_conflicts, _ = (
        _normalize_review_edits(changes.review, labels, groups)
    )
    if (
        dict(current_after) != dict(changes.labels_after)
        or current_groups != changes.merge_groups
        or current_conflicts != changes.label_conflicts
    ):
        raise ReviewChangedSincePreviewError(
            "FigPack review semantics changed after preview. Preview the "
            "annotations again before committing."
        )
    no_changes = (
        dict(changes.labels_before) == dict(changes.labels_after)
        and not changes.merge_groups
    )
    if no_changes and not confirm_no_changes:
        raise ValueError(
            "This review contains no label or merge changes. Pass "
            "confirm_no_changes=True to record an explicit reviewed/no-change "
            "child."
        )

    provided = {
        int(unit_id): tuple(map(str, values))
        for unit_id, values in (conflict_resolutions or {}).items()
    }
    required = {conflict.merged_unit_id for conflict in changes.label_conflicts}
    if set(provided) != required:
        missing = sorted(required - set(provided))
        unexpected = sorted(set(provided) - required)
        raise UnresolvedMergeLabelConflictError(
            "Merge contributors have incompatible labels. Provide exactly one "
            "conflict_resolutions entry per predicted merged unit; "
            f"missing={missing}, unexpected={unexpected}."
        )
    valid_labels = set(changes.review.profile.label_options)
    conflicts_by_unit = {
        conflict.merged_unit_id: conflict
        for conflict in changes.label_conflicts
    }
    for unit_id, values in provided.items():
        unknown = _unknown_conflict_resolution_labels(
            values, valid_labels, conflicts_by_unit[unit_id]
        )
        if unknown:
            raise ValueError(
                f"Conflict resolution for merged unit {unit_id} contains "
                "new labels outside the review profile palette: "
                f"{unknown}."
            )

    existing_children = {
        int(value)
        for value in (
            CurationV2
            & {
                "sorting_id": changes.review.parent.sorting_id,
                "parent_curation_id": changes.review.parent.curation_id,
            }
        ).fetch("curation_id")
    }
    child_labels = _child_labels(changes, provided)
    child_key = CurationV2.save_manual_curation(
        {"sorting_id": changes.review.parent.sorting_id},
        parent_curation_id=changes.review.parent.curation_id,
        labels=child_labels,
        merge_groups=[list(group) for group in changes.merge_groups],
        merge_action="commit",
        curation_source="figpack",
        description=(
            "FigPack reviewed: no changes"
            if no_changes
            else "curated in FigPack review"
        ),
        reuse_existing=True,
        label_policy="replace",
    )
    child = CurationRef.from_key(child_key)
    child_status: ReviewStageState = (
        "reused" if child.curation_id in existing_children else "computed"
    )

    # No-op for ordinary sorts; concat-backed curations materialize one
    # session-scoped downstream merge row per frozen member.
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )

    ConcatMemberCuration.populate(child.as_key(), reserve_jobs=False)

    evaluation = None
    stages = [
        ReviewStageStatus("identity_verified", "complete"),
        ReviewStageStatus("edits_loaded", "complete"),
        ReviewStageStatus("child_committed", child_status),
    ]
    if changes.merge_groups:
        evaluation = child.evaluate(
            metric_params_name=(
                changes.review.profile.evaluation_spec.metric_params_name
            ),
            auto_curation_rules_name=(
                changes.review.profile.evaluation_spec.auto_curation_rules_name
            ),
        )
        stages.extend(
            (
                ReviewStageStatus("merged_analyzer_built", "complete"),
                ReviewStageStatus("evaluation_populated", "complete"),
            )
        )
    return ReviewImportReceipt(
        curation=child,
        evaluation=evaluation,
        changes=changes,
        warnings=(),
        stages=tuple(stages),
        needs_merge_verification=bool(changes.merge_groups),
    )


__all__ = [
    "CurationChangeSet",
    "FigPackReview",
    "MergeLabelConflict",
    "ReviewImportReceipt",
    "ReviewProfileRef",
    "ReviewStageStatus",
    "start_review",
]
