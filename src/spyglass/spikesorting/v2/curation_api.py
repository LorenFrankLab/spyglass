"""Public, identity-safe facade for Spike Sorting V2 curation.

The table classes remain the expert layer.  This module provides the smaller
scripted surface used by notebooks and automation: a generation-pinned
``CurationRef``, evaluation snapshots, merge/evaluate receipts, and lifecycle
inspection.  Browser review entry points are added by the FigPack review layer;
this module deliberately has no optional FigPack imports.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

StageStatus = Literal["computed", "reused"]
CommitStatus = Literal["preview", "committed"]


def _uuid(value) -> uuid.UUID:
    """Normalize UUID-like values returned by DataJoint drivers."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _require_outside_merge_evaluation_transaction() -> None:
    """Reject orchestration before it can perform any nested write."""
    from spyglass.spikesorting.v2.curation import CurationV2

    if CurationV2.connection.in_transaction:
        raise RuntimeError(
            "merge_and_evaluate must be called outside any open DataJoint "
            "transaction; populate manages its own transaction."
        )


@dataclass(frozen=True)
class CurationOperation(Mapping[str, str]):
    """Schema-free provenance summary for one curation operation."""

    producer: str
    change_kind: str

    def __getitem__(self, key: str) -> str:
        if key not in ("producer", "change_kind"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("producer", "change_kind"))

    def __len__(self) -> int:
        return 2


@dataclass(frozen=True)
class CurationDeletePreview:
    """Leaf-first deletion inventory for a curation subtree."""

    root: "CurationRef"
    leaf_first: tuple["CurationRef", ...]

    @property
    def count(self) -> int:
        return len(self.leaf_first)


@dataclass(frozen=True)
class CurationDeleteReceipt:
    """Identity snapshots of the rows removed by a subtree deletion."""

    deleted: tuple["CurationRef", ...]

    @property
    def count(self) -> int:
        return len(self.deleted)


@dataclass(frozen=True)
class CurationRef:
    """A curation handle pinned to one immutable row generation.

    The numeric ``curation_id`` is ergonomic but reusable after deletion.
    Every database-consuming operation therefore verifies ``curation_uuid``
    again and raises ``CurationNotFoundError`` if the row is gone or replaced.
    """

    sorting_id: uuid.UUID
    curation_id: int
    curation_uuid: uuid.UUID

    @classmethod
    def from_key(cls, key: Mapping[str, Any] | "CurationRef") -> "CurationRef":
        """Validate a curation key and pin its current generation UUID."""
        if isinstance(key, cls):
            key._current_row()
            return key
        missing = {"sorting_id", "curation_id"} - set(key)
        if missing:
            raise ValueError(
                "CurationRef.from_key requires sorting_id and curation_id; "
                f"missing {sorted(missing)}."
            )
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.exceptions import CurationNotFoundError

        dj_key = {
            "sorting_id": key["sorting_id"],
            "curation_id": int(key["curation_id"]),
        }
        rows = (CurationV2 & dj_key).fetch("curation_uuid")
        if len(rows) != 1:
            raise CurationNotFoundError(
                "CurationRef.from_key: no current CurationV2 row for "
                f"sorting_id={dj_key['sorting_id']}, "
                f"curation_id={dj_key['curation_id']}."
            )
        return cls(
            sorting_id=_uuid(dj_key["sorting_id"]),
            curation_id=dj_key["curation_id"],
            curation_uuid=_uuid(rows[0]),
        )

    def _unchecked_key(self) -> dict[str, Any]:
        return {
            "sorting_id": self.sorting_id,
            "curation_id": self.curation_id,
        }

    def _current_row(self) -> dict[str, Any]:
        """Fetch this exact generation or raise the typed stale-ref error."""
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.exceptions import CurationNotFoundError

        rows = (CurationV2 & self._unchecked_key()).fetch(as_dict=True)
        current_uuid = (
            _uuid(rows[0]["curation_uuid"]) if len(rows) == 1 else None
        )
        if current_uuid != self.curation_uuid:
            detail = "was deleted" if current_uuid is None else "was replaced"
            raise CurationNotFoundError(
                "CurationRef no longer identifies its CurationV2 generation: "
                f"sorting_id={self.sorting_id}, curation_id={self.curation_id} "
                f"{detail} (expected curation_uuid={self.curation_uuid}, "
                f"current={current_uuid}). Resolve a fresh CurationRef from "
                "the intended curation."
            )
        return rows[0]

    def as_key(self) -> dict[str, Any]:
        """Return the expert-layer key after re-validating row identity."""
        self._current_row()
        return self._unchecked_key()

    @property
    def merge_id(self) -> uuid.UUID | None:
        """Return this exact curation's downstream merge id, if registered."""
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        key = self.as_key()
        merge_ids = (SpikeSortingOutput.CurationV2 & key).fetch("merge_id")
        if len(merge_ids) > 1:
            raise RuntimeError(
                "CurationRef.merge_id: expected at most one curated merge row "
                f"for {key}; found {len(merge_ids)}."
            )
        return _uuid(merge_ids[0]) if len(merge_ids) else None

    @property
    def member_merge_ids(self) -> Mapping[int, uuid.UUID]:
        """Return concat member ``member_index -> merge_id`` outputs."""
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
        from spyglass.spikesorting.v2.concat_member_curation import (
            ConcatMemberCuration,
        )

        rows = (
            ConcatMemberCuration * SpikeSortingOutput.ConcatMemberCuration
            & self.as_key()
        ).fetch("member_index", "merge_id", as_dict=True)
        return MappingProxyType(
            {int(row["member_index"]): _uuid(row["merge_id"]) for row in rows}
        )

    @property
    def parent(self) -> "CurationRef | None":
        row = self._current_row()
        parent_id = int(row["parent_curation_id"])
        if parent_id == -1:
            return None
        return type(self).from_key(
            {"sorting_id": self.sorting_id, "curation_id": parent_id}
        )

    @property
    def children(self) -> tuple["CurationRef", ...]:
        from spyglass.spikesorting.v2.curation import CurationV2

        self._current_row()
        rows = (
            CurationV2
            & {
                "sorting_id": self.sorting_id,
                "parent_curation_id": self.curation_id,
            }
        ).fetch("KEY", order_by="curation_id")
        return tuple(type(self).from_key(row) for row in rows)

    @property
    def commit_status(self) -> CommitStatus:
        from spyglass.spikesorting.v2.curation import CurationV2

        row = self._current_row()
        committed = CurationV2.is_committed_curation(
            self._unchecked_key(), merges_applied=row["merges_applied"]
        )
        return "committed" if committed else "preview"

    @property
    def is_root(self) -> bool:
        return int(self._current_row()["parent_curation_id"]) == -1

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def created_at(self):
        """Return the database timestamp recorded for this curation."""
        return self._current_row()["created_at"]

    @property
    def created_by(self) -> str:
        """Return the database user recorded for this curation."""
        return str(self._current_row()["created_by"])

    @property
    def has_committed_children(self) -> bool:
        return any(
            child.commit_status == "committed" for child in self.children
        )

    @property
    def operation_type(self) -> CurationOperation:
        """Return producer plus the change kind derived from actual part rows."""
        from spyglass.spikesorting.v2.curation import CurationV2

        row = self._current_row()
        if int(row["parent_curation_id"]) == -1:
            return CurationOperation(
                producer=str(row["curation_source"]), change_kind="initial"
            )

        key = self._unchecked_key()
        groups = CurationV2.get_unit_contributor_groups(key)
        has_merge = any(len(group) > 1 for group in groups.values())
        current_labels = _normalized_labels(CurationV2._labels_by_unit(key))
        parent_key = {
            "sorting_id": self.sorting_id,
            "curation_id": int(row["parent_curation_id"]),
        }
        parent_labels = _normalized_labels(
            CurationV2._labels_by_unit(parent_key)
        )
        expected_labels = _inherited_labels_after_operation(
            parent_labels,
            groups,
            merges_applied=bool(row["merges_applied"]),
        )
        has_label_delta = current_labels != expected_labels
        if has_merge and has_label_delta:
            change_kind = "merge+label"
        elif has_merge:
            change_kind = "merge"
        elif has_label_delta:
            change_kind = "label"
        else:
            change_kind = "no_change"
        return CurationOperation(
            producer=str(row["curation_source"]), change_kind=change_kind
        )

    def evaluate(
        self,
        *,
        metric_params_name: str,
        auto_curation_rules_name: str,
    ) -> "EvaluationResult":
        """Create/reuse and populate an evaluation over this curation."""
        self.as_key()
        spec = EvaluationSpec(
            metric_params_name=metric_params_name,
            auto_curation_rules_name=auto_curation_rules_name,
        )
        return _evaluate_curation(self, spec)

    def start_review(
        self,
        profile,
        *,
        upload: bool = False,
        ephemeral: bool = False,
        annotation_sets=(),
        display_options=None,
    ):
        """Start/reuse a seeded browser review over this exact generation.

        ``display_options`` (``ReviewDisplayOptions`` / mapping / ``None``)
        bounds the browser payload and is persisted with the review.
        """
        from spyglass.spikesorting.v2.review_api import start_review

        return start_review(
            self,
            profile,
            upload=upload,
            ephemeral=ephemeral,
            annotation_sets=annotation_sets,
            display_options=display_options,
        )

    def preview_merges(
        self, groups: Sequence[Sequence[int]], **kwargs
    ) -> "CurationRef":
        """Create/reuse a manual preview child from this typed parent."""
        return preview_merges(parent_curation=self, groups=groups, **kwargs)

    def commit_merges(
        self, groups: Sequence[Sequence[int]], **kwargs
    ) -> "CurationRef":
        """Create/reuse a committed manual merge child without evaluation."""
        return commit_merges(parent_curation=self, groups=groups, **kwargs)

    def lineage(self) -> tuple["CurationRef", ...]:
        """Return ancestors from root through this curation."""
        lineage: list[CurationRef] = []
        current: CurationRef | None = self
        seen: set[tuple[uuid.UUID, int]] = set()
        while current is not None:
            identity = (current.sorting_id, current.curation_id)
            if identity in seen:
                raise RuntimeError(
                    "Curation lineage contains a cycle at "
                    f"sorting_id={current.sorting_id}, "
                    f"curation_id={current.curation_id}."
                )
            seen.add(identity)
            lineage.append(current)
            current = current.parent
        return tuple(reversed(lineage))

    def visualize_lineage(self) -> str:
        """Return a compact text tree for every curation in this sorting."""
        from spyglass.spikesorting.v2.curation import CurationV2

        self._current_row()
        rows = (CurationV2 & {"sorting_id": self.sorting_id}).fetch(
            as_dict=True, order_by="curation_id"
        )
        by_parent: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_parent.setdefault(int(row["parent_curation_id"]), []).append(row)
        lines: list[str] = []

        def visit(row, prefix: str) -> None:
            ref = type(self).from_key(row)
            marker = "*" if ref.curation_uuid == self.curation_uuid else "-"
            operation = ref.operation_type
            lines.append(
                f"{prefix}{marker} curation {ref.curation_id} "
                f"[{ref.commit_status}; {operation.producer}/"
                f"{operation.change_kind}]"
            )
            for child in by_parent.get(ref.curation_id, []):
                visit(child, prefix + "  ")

        for root in by_parent.get(-1, []):
            visit(root, "")
        return "\n".join(lines)

    def preview_curation_delete(self) -> CurationDeletePreview:
        """Inventory this subtree in the leaf-first order deletion will use."""
        self._current_row()
        leaf_first: list[CurationRef] = []

        def visit(ref: CurationRef) -> None:
            for child in ref.children:
                visit(child)
            leaf_first.append(ref)

        visit(self)
        return CurationDeletePreview(root=self, leaf_first=tuple(leaf_first))

    def delete_subtree(self, *, safemode: bool = True) -> CurationDeleteReceipt:
        """Delete this curation and descendants leaf-up without orphaning lineage."""
        from spyglass.spikesorting.v2.curation import CurationV2

        preview = self.preview_curation_delete()
        for ref in preview.leaf_first:
            (CurationV2 & ref._unchecked_key()).delete(safemode=safemode)
        return CurationDeleteReceipt(deleted=preview.leaf_first)

    def health_report(self) -> Mapping[str, Any]:
        """Compose lineage and analyzer-cache health for this sorting."""
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.sorting import Sorting

        self._current_row()
        lineage = [
            row
            for row in CurationV2.audit_orphaned_lineage()
            if _uuid(row["sorting_id"]) == self.sorting_id
        ]
        analyzer = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        return MappingProxyType(
            {"orphaned_lineage": tuple(lineage), "analyzer_cache": analyzer}
        )

    def summarize(
        self,
        *,
        evaluation: "EvaluationResult | None" = None,
        annotation_sets=None,
    ) -> dict:
        """Summarize this curation and explicitly selected unit properties."""
        from spyglass.spikesorting.v2.curation import CurationV2

        return CurationV2.summarize_curation(
            self.as_key(),
            evaluation=evaluation,
            annotation_sets=annotation_sets,
        )


def _normalized_labels(
    labels: Mapping[int, Sequence[str]],
) -> dict[int, tuple[str, ...]]:
    return {
        int(unit_id): tuple(sorted(map(str, unit_labels)))
        for unit_id, unit_labels in labels.items()
        if unit_labels
    }


def _inherited_labels_after_operation(
    parent_labels: Mapping[int, tuple[str, ...]],
    groups: Mapping[int, Sequence[int]],
    *,
    merges_applied: bool,
) -> dict[int, tuple[str, ...]]:
    """Predict label inheritance so a pure merge is not called a label edit."""
    if not merges_applied:
        return dict(parent_labels)
    inherited: dict[int, tuple[str, ...]] = {}
    for child_id, parent_ids in groups.items():
        labels = {
            label
            for parent_id in parent_ids
            for label in parent_labels.get(int(parent_id), ())
        }
        if labels:
            inherited[int(child_id)] = tuple(sorted(labels))
    return inherited


@dataclass(frozen=True)
class EvaluationSpec:
    """The exact metric and rule recipes used for an evaluation."""

    metric_params_name: str
    auto_curation_rules_name: str


class EvaluationPlots:
    """Controlled plotting facade for one persisted evaluation."""

    def __init__(self, result: "EvaluationResult"):
        self._result = result

    def _call(self, method: str, *args, **kwargs):
        from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

        key = self._result._current_evaluation_key()
        return getattr(CurationEvaluation(), method)(key, *args, **kwargs)

    def metrics(self, **kwargs):
        return self._call("plot_metrics", **kwargs)

    def units_qc(self, **kwargs):
        return self._call("plot_units_qc", **kwargs)

    def correlograms(self, **kwargs):
        return self._call("plot_correlograms", **kwargs)

    def suggested_merges(self, **kwargs):
        return self._call("plot_suggested_merges", **kwargs)

    def si_quality_metrics(self, **kwargs):
        return self._call("plot_si_quality_metrics", **kwargs)

    def si_template_metrics(self, **kwargs):
        return self._call("plot_si_template_metrics", **kwargs)

    def pair_correlograms(self, pairs, **kwargs):
        return self._call("investigate_pair_xcorrel", pairs, **kwargs)

    def pair_peaks(self, pairs, **kwargs):
        return self._call("investigate_pair_peaks", pairs, **kwargs)

    def peak_over_time(self, pairs, **kwargs):
        return self._call("plot_peak_over_time", pairs, **kwargs)

    def burst_pair_metrics(self, pairs=None, **kwargs):
        return self._call("plot_burst_pair_metrics", pairs=pairs, **kwargs)


@dataclass(frozen=True, init=False)
class EvaluationResult:
    """Point-in-time evaluation snapshot with defensive mutable-value copies."""

    curation: CurationRef
    spec: EvaluationSpec
    evaluation_id: uuid.UUID
    warnings: tuple[str, ...]
    _metrics: pd.DataFrame = field(repr=False, compare=False)
    _suggested_merges: tuple[tuple[int, ...], ...] = field(
        repr=False, compare=False
    )
    _proposed_labels: tuple[tuple[int, tuple[str, ...]], ...] = field(
        repr=False, compare=False
    )

    def __init__(
        self,
        *,
        curation: CurationRef,
        spec: EvaluationSpec,
        evaluation_id,
        metrics: pd.DataFrame,
        suggested_merges: Sequence[Sequence[int]],
        proposed_labels: Mapping[int, Sequence[str]],
        warnings: Sequence[str] = (),
    ):
        object.__setattr__(self, "curation", curation)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "evaluation_id", _uuid(evaluation_id))
        object.__setattr__(self, "warnings", tuple(map(str, warnings)))
        object.__setattr__(self, "_metrics", metrics.copy(deep=True))
        object.__setattr__(
            self,
            "_suggested_merges",
            tuple(tuple(map(int, group)) for group in suggested_merges),
        )
        object.__setattr__(
            self,
            "_proposed_labels",
            tuple(
                (int(unit_id), tuple(map(str, labels)))
                for unit_id, labels in sorted(proposed_labels.items())
            ),
        )

    @property
    def metrics(self) -> pd.DataFrame:
        return self._metrics.copy(deep=True)

    @property
    def suggested_merges(self) -> list[list[int]]:
        return [list(group) for group in self._suggested_merges]

    @property
    def proposed_labels(self) -> dict[int, list[str]]:
        return {
            unit_id: list(labels) for unit_id, labels in self._proposed_labels
        }

    @property
    def plots(self) -> EvaluationPlots:
        return EvaluationPlots(self)

    def burst_pair_metrics(self, pairs=None, **kwargs) -> pd.DataFrame:
        """Burst-merge diagnostics per ordered unit pair, as a DataFrame.

        The data twin of ``plots.burst_pair_metrics()``. A method rather than
        a property like ``metrics`` because it loads the display analyzer on
        demand instead of being part of the frozen snapshot. See
        ``CurationEvaluation.get_burst_pair_metrics`` for the columns and
        keyword arguments.
        """
        from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

        key = self._current_evaluation_key()
        return CurationEvaluation().get_burst_pair_metrics(
            key, pairs=pairs, **kwargs
        )

    @classmethod
    def from_key(cls, key: Mapping[str, Any]) -> "EvaluationResult":
        """Load one populated evaluation and freeze a defensive snapshot."""
        from spyglass.spikesorting.v2.metric_curation import (
            CurationEvaluation,
            CurationEvaluationSelection,
        )

        selection = (CurationEvaluationSelection & key).fetch1()
        evaluation_key = {
            "curation_evaluation_id": selection["curation_evaluation_id"]
        }
        if not (CurationEvaluation & evaluation_key):
            raise LookupError(
                "EvaluationResult.from_key requires a populated "
                f"CurationEvaluation; populate {evaluation_key} first."
            )
        curation = CurationRef.from_key(selection)
        return cls(
            curation=curation,
            spec=EvaluationSpec(
                metric_params_name=str(selection["metric_params_name"]),
                auto_curation_rules_name=str(
                    selection["auto_curation_rules_name"]
                ),
            ),
            evaluation_id=selection["curation_evaluation_id"],
            metrics=CurationEvaluation.get_metrics(evaluation_key),
            suggested_merges=CurationEvaluation.get_suggested_merge_groups(
                evaluation_key
            ),
            proposed_labels=CurationEvaluation.get_labels(evaluation_key),
        )

    def _current_evaluation_key(self) -> dict[str, uuid.UUID]:
        from spyglass.spikesorting.v2.metric_curation import (
            CurationEvaluation,
            CurationEvaluationSelection,
        )

        self.curation.as_key()
        key = {"curation_evaluation_id": self.evaluation_id}
        selections = (CurationEvaluationSelection & key).fetch(as_dict=True)
        if len(selections) != 1 or not (CurationEvaluation & key):
            raise LookupError(
                "EvaluationResult no longer has its populated evaluation row: "
                f"curation_evaluation_id={self.evaluation_id}."
            )
        selected = selections[0]
        if (
            _uuid(selected["sorting_id"]) != self.curation.sorting_id
            or int(selected["curation_id"]) != self.curation.curation_id
        ):
            raise LookupError(
                "EvaluationResult selection identity no longer matches its "
                f"curation: curation_evaluation_id={self.evaluation_id}."
            )
        return key

    def preview_merges(self, groups: Sequence[Sequence[int]]) -> CurationRef:
        from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

        key = self._current_evaluation_key()
        child = CurationEvaluation().preview_merges(
            key, merge_groups=_validate_merge_groups(self.curation, groups)
        )
        return CurationRef.from_key(child)

    def commit_merges(self, groups: Sequence[Sequence[int]]) -> CurationRef:
        """Expert scripted merge commit without re-evaluating the child."""
        from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

        key = self._current_evaluation_key()
        child = CurationEvaluation().accept_merges(
            key, merge_groups=_validate_merge_groups(self.curation, groups)
        )
        return CurationRef.from_key(child)

    def merge_and_evaluate(
        self, groups: Sequence[Sequence[int]]
    ) -> "MergeEvaluateReceipt":
        """Commit merges and evaluate the child with this exact recipe spec."""
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.metric_curation import (
            CurationEvaluation,
            CurationEvaluationSelection,
        )

        _require_outside_merge_evaluation_transaction()
        evaluation_key = self._current_evaluation_key()
        normalized = _validate_merge_groups(self.curation, groups)
        existing_children = {
            int(value)
            for value in (
                CurationV2
                & {
                    "sorting_id": self.curation.sorting_id,
                    "parent_curation_id": self.curation.curation_id,
                }
            ).fetch("curation_id")
        }
        child_key = CurationEvaluation().accept_merges(
            evaluation_key,
            merge_groups=normalized,
            reuse_existing=True,
        )
        curation_status: StageStatus = (
            "reused"
            if int(child_key["curation_id"]) in existing_children
            else "computed"
        )
        child = CurationRef.from_key(child_key)
        selection_identity = {
            **child._unchecked_key(),
            "metric_params_name": self.spec.metric_params_name,
            "auto_curation_rules_name": self.spec.auto_curation_rules_name,
        }
        selection_existed = bool(
            CurationEvaluationSelection & selection_identity
        )
        selection_key = CurationEvaluationSelection.insert_by_curation_id(
            child.sorting_id,
            child.curation_id,
            self.spec.metric_params_name,
            self.spec.auto_curation_rules_name,
        )
        evaluation_existed = bool(CurationEvaluation & selection_key)
        CurationEvaluation.populate(selection_key, reserve_jobs=False)
        post_merge = EvaluationResult.from_key(selection_key)
        return MergeEvaluateReceipt(
            child=child,
            evaluation=post_merge,
            merge_groups=tuple(tuple(group) for group in normalized),
            warnings=(),
            curation_status=curation_status,
            evaluation_selection_status=(
                "reused" if selection_existed else "computed"
            ),
            evaluation_status="reused" if evaluation_existed else "computed",
        )

    def accept_labels(
        self, mode: Literal["replace", "overlay"] = "replace"
    ) -> CurationRef:
        """Accept proposed labels with explicit replace or overlay semantics."""
        from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

        key = self._current_evaluation_key()
        if mode == "replace":
            child = CurationEvaluation().use_evaluation_labels(key)
        elif mode == "overlay":
            child = CurationEvaluation().overlay_evaluation_labels(key)
        else:
            raise ValueError(
                "EvaluationResult.accept_labels mode must be 'replace' or "
                f"'overlay'; got {mode!r}."
            )
        return CurationRef.from_key(child)

    def start_review(
        self,
        profile,
        *,
        upload: bool = False,
        ephemeral: bool = False,
        annotation_sets=(),
        display_options=None,
    ):
        """Start a review after verifying this evaluation matches its profile."""
        from spyglass.spikesorting.v2.review_api import start_review

        return start_review(
            self.curation,
            profile,
            upload=upload,
            ephemeral=ephemeral,
            evaluation=self,
            annotation_sets=annotation_sets,
            display_options=display_options,
        )


@dataclass(frozen=True)
class MergedCuration:
    """A merge child plus the canonical groups and create/reuse status."""

    curation: CurationRef
    merge_groups: tuple[tuple[int, ...], ...]
    status: StageStatus


@dataclass(frozen=True)
class MergeEvaluateReceipt:
    """Resumable per-stage receipt for a merge followed by evaluation."""

    child: CurationRef
    evaluation: EvaluationResult
    merge_groups: tuple[tuple[int, ...], ...]
    warnings: tuple[str, ...]
    curation_status: StageStatus
    evaluation_selection_status: StageStatus
    evaluation_status: StageStatus

    @property
    def merged(self) -> MergedCuration:
        return MergedCuration(
            curation=self.child,
            merge_groups=self.merge_groups,
            status=self.curation_status,
        )

    @property
    def stage_statuses(self) -> Mapping[str, StageStatus]:
        return MappingProxyType(
            {
                "curation": self.curation_status,
                "evaluation_selection": self.evaluation_selection_status,
                "evaluation": self.evaluation_status,
            }
        )


class RunResult(dict):
    """Mapping run receipt with generation-pinned curation accessors.

    ``root_curation`` / ``auto_labeled_curation`` are built from the
    ``*_curation_uuid`` keys the run recorded, so a receipt kept across a
    delete-and-recreate of the same numeric ``curation_id`` raises
    ``CurationNotFoundError`` instead of resolving the replacement row.
    """

    @property
    def sorting_id(self) -> uuid.UUID:
        return _uuid(self["sorting_id"])

    def _pinned_ref(self, id_key: str, uuid_key: str) -> CurationRef:
        ref = CurationRef(
            sorting_id=self.sorting_id,
            curation_id=int(self[id_key]),
            curation_uuid=_uuid(self[uuid_key]),
        )
        ref._current_row()
        return ref

    @property
    def root_curation(self) -> CurationRef:
        return self._pinned_ref("root_curation_id", "root_curation_uuid")

    @property
    def auto_labeled_curation(self) -> CurationRef | None:
        """The auto-labeled child, or ``None`` for a root-only run.

        Automatic labels are not approval and the row still holds every unit;
        pass it to ``select_units_for_analysis`` to choose the analysis set.
        """
        if self.get("auto_labeled_curation_id") is None:
            return None
        return self._pinned_ref(
            "auto_labeled_curation_id", "auto_labeled_curation_uuid"
        )

    def start_review(
        self,
        profile,
        *,
        source: Literal["auto_labeled", "root"] = "auto_labeled",
        upload: bool = False,
        ephemeral: bool = False,
        annotation_sets=(),
        display_options=None,
    ):
        """Start the canonical review without silently changing its source."""
        if source == "auto_labeled":
            curation = self.auto_labeled_curation
            if curation is None:
                raise ValueError(
                    "RunResult.start_review(source='auto_labeled') requires an "
                    "auto-labeled child, but this run did not auto-curate. "
                    "Pass source='root' explicitly to review the root curation."
                )
        elif source == "root":
            curation = self.root_curation
        else:
            raise ValueError(
                "RunResult.start_review source must be 'auto_labeled' or 'root'; "
                f"got {source!r}."
            )
        return curation.start_review(
            profile,
            upload=upload,
            ephemeral=ephemeral,
            annotation_sets=annotation_sets,
            display_options=display_options,
        )


def _validate_merge_groups(
    parent_curation: CurationRef,
    groups: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Validate merge shape and membership before any facade write."""
    from spyglass.spikesorting.v2.curation import CurationV2

    key = parent_curation.as_key()
    try:
        normalized = [list(map(int, group)) for group in groups]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "merge groups must be a sequence of unit-id sequences."
        ) from exc
    if not normalized:
        raise ValueError("merge groups must contain at least one group.")
    for group in normalized:
        if len(group) < 2:
            raise ValueError(
                "each merge group must contain at least two unit ids; "
                f"got {group}."
            )
        if len(set(group)) != len(group):
            raise ValueError(
                f"merge group contains duplicate unit ids: {group}."
            )
    flattened = [unit_id for group in normalized for unit_id in group]
    repeated = sorted(
        {unit_id for unit_id in flattened if flattened.count(unit_id) > 1}
    )
    if repeated:
        raise ValueError(
            "merge groups must be disjoint; repeated unit ids: " f"{repeated}."
        )
    available = {
        int(value) for value in (CurationV2.Unit & key).fetch("unit_id")
    }
    unknown = sorted(set(flattened) - available)
    if unknown:
        raise ValueError(
            "merge groups reference unit ids absent from parent curation "
            f"{parent_curation.curation_id}: {unknown}. Available unit ids: "
            f"{sorted(available)}."
        )
    return normalized


def _evaluate_curation(
    curation: CurationRef, spec: EvaluationSpec
) -> EvaluationResult:
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    key = curation.as_key()
    selection = CurationEvaluationSelection.insert_by_curation_id(
        key["sorting_id"],
        key["curation_id"],
        spec.metric_params_name,
        spec.auto_curation_rules_name,
    )
    CurationEvaluation.populate(selection, reserve_jobs=False)
    return EvaluationResult.from_key(selection)


def merge_and_evaluate(
    *,
    parent_curation: CurationRef,
    spec: EvaluationSpec,
    groups: Sequence[Sequence[int]],
) -> MergeEvaluateReceipt:
    """Lower-level orchestration form with an explicit parent and spec."""
    parent = _require_parent_ref(parent_curation, caller="merge_and_evaluate")
    if not isinstance(spec, EvaluationSpec):
        raise TypeError(
            "merge_and_evaluate requires spec=EvaluationSpec; pass the exact "
            "metric and auto-curation rule recipe names."
        )
    _require_outside_merge_evaluation_transaction()
    return _evaluate_curation(parent, spec).merge_and_evaluate(groups)


def create_initial_curation(
    sorting_key: Mapping[str, Any], **kwargs
) -> CurationRef:
    """Create the only facade operation that does not require a typed parent."""
    from spyglass.spikesorting.v2.curation import CurationV2

    return CurationRef.from_key(
        CurationV2.create_initial_curation(dict(sorting_key), **kwargs)
    )


def preview_merges(
    *,
    parent_curation: CurationRef,
    groups: Sequence[Sequence[int]],
    **kwargs,
) -> CurationRef:
    """Create a preview child; a typed, current parent is mandatory."""
    parent = _require_parent_ref(parent_curation, caller="preview_merges")
    from spyglass.spikesorting.v2.curation import CurationV2

    normalized = _validate_merge_groups(parent, groups)
    kwargs.setdefault("reuse_existing", True)
    child = CurationV2.propose_merge_curation(
        {"sorting_id": parent.sorting_id},
        merge_groups=normalized,
        parent_curation_id=parent.curation_id,
        **kwargs,
    )
    return CurationRef.from_key(child)


def commit_merges(
    *,
    parent_curation: CurationRef,
    groups: Sequence[Sequence[int]],
    **kwargs,
) -> CurationRef:
    """Commit a merge child without evaluation; a typed parent is mandatory."""
    parent = _require_parent_ref(parent_curation, caller="commit_merges")
    from spyglass.spikesorting.v2.curation import CurationV2

    normalized = _validate_merge_groups(parent, groups)
    kwargs.setdefault("reuse_existing", True)
    child = CurationV2.create_merged_curation(
        {"sorting_id": parent.sorting_id},
        merge_groups=normalized,
        parent_curation_id=parent.curation_id,
        **kwargs,
    )
    return CurationRef.from_key(child)


def save_manual_curation(
    *, parent_curation: CurationRef, **kwargs
) -> CurationRef:
    """Save a manual child from a typed parent; root sentinels are not accepted."""
    parent = _require_parent_ref(parent_curation, caller="save_manual_curation")
    from spyglass.spikesorting.v2.curation import CurationV2

    child = CurationV2.save_manual_curation(
        {"sorting_id": parent.sorting_id},
        parent_curation_id=parent.curation_id,
        **kwargs,
    )
    return CurationRef.from_key(child)


def _require_parent_ref(parent_curation, *, caller: str) -> CurationRef:
    """Enforce the facade's typed-parent boundary for every child operation."""
    if not isinstance(parent_curation, CurationRef):
        raise TypeError(
            f"{caller} requires parent_curation=CurationRef; raw key dicts "
            "and parent_curation_id sentinels are only supported by the expert "
            "table layer. Use CurationRef.from_key(...) first."
        )
    return CurationRef.from_key(parent_curation)


__all__ = [
    "CurationDeletePreview",
    "CurationDeleteReceipt",
    "CurationOperation",
    "CurationRef",
    "EvaluationPlots",
    "EvaluationResult",
    "EvaluationSpec",
    "MergedCuration",
    "MergeEvaluateReceipt",
    "RunResult",
    "commit_merges",
    "create_initial_curation",
    "merge_and_evaluate",
    "preview_merges",
    "save_manual_curation",
]
