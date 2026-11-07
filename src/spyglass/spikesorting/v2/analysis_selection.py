"""Explicit unit selection: the handoff from a curation to downstream analysis.

A curation holds every unit with its labels; its ``SpikeSortingOutput`` merge
id is not a filtered population. :func:`select_units_for_analysis` applies a
named ``UnitSelectionParams`` label policy and optional evaluation criteria,
freezes membership, builds the ``SortedSpikesGroup`` downstream analyses read,
and returns a receipt listing every included and excluded unit with its
reason.

Shipped v2 policies (``V2_UNIT_SELECTION_POLICIES``), added alongside the
existing ``all_units`` / ``exclude_noise`` / ``default_exclusion`` rows:

* ``v2_accepted_single_units`` (default): require ``accept``; deny ``mua``,
  ``noise``, ``reject``, ``artifact``. Unlabeled units are excluded.
* ``v2_accepted_neural_units``: require ``accept`` or ``mua``; deny
  ``noise``, ``reject``, ``artifact``. Unlabeled units are excluded.
* ``v2_unflagged_units``: deny ``noise``, ``reject``, ``artifact``; keep
  everything else, including MUA and unlabeled units. For auto-label-only
  workflows, since the shipped rules never write ``accept``.
* ``v2_unflagged_single_units``: as above, also denying ``mua``.
  Cross-session matching requires a policy that denies ``mua``.

``all_units`` remains an explicit expert choice; the receipt reports which
included units were unlabeled.

A concat sort has one output per member and ``SortedSpikesGroup`` is per
session, so the handoff builds one group per member.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from spyglass.utils.identity import sha256_json
from spyglass.spikesorting.v2.curation_api import CurationRef, EvaluationResult
from spyglass.spikesorting.v2._curation.unit_selection import (
    UnitSelectionSnapshot,
    unit_selection_hash,
)

#: Read-only ``include_labels`` / ``exclude_labels`` tuples behind a mapping
#: proxy.
LabelPolicy = Mapping[str, tuple[str, ...]]


def _label_policy(include, exclude) -> LabelPolicy:
    return MappingProxyType(
        {
            "include_labels": tuple(str(v) for v in include),
            "exclude_labels": tuple(str(v) for v in exclude),
        }
    )


# Inserted by ``ensure_v2_unit_selection_policies``.
V2_UNIT_SELECTION_POLICIES: Mapping[str, LabelPolicy] = MappingProxyType(
    {
        "v2_accepted_single_units": _label_policy(
            ["accept"], ["mua", "noise", "reject", "artifact"]
        ),
        "v2_accepted_neural_units": _label_policy(
            ["accept", "mua"], ["noise", "reject", "artifact"]
        ),
        "v2_unflagged_units": _label_policy(
            [], ["noise", "reject", "artifact"]
        ),
        "v2_unflagged_single_units": _label_policy(
            [], ["mua", "noise", "reject", "artifact"]
        ),
    }
)

DEFAULT_UNIT_SELECTION_POLICY = "v2_accepted_single_units"


@dataclass(frozen=True)
class SelectedGroup:
    """One ``SortedSpikesGroup`` the handoff created or reused."""

    nwb_file_name: str
    group_key: Mapping[str, Any]
    merge_id: uuid.UUID
    member_index: int | None
    status: str  # "created" | "reused"

    @property
    def observation(self):
        """Common observed intervals in this member's original session time."""
        from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

        return SortedSpikesGroup.get_observation_intervals(dict(self.group_key))

    def fetch_spike_data(self, **kwargs):
        """Spike times for the selected units via the downstream group API."""
        from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

        return SortedSpikesGroup.fetch_spike_data(
            dict(self.group_key), **kwargs
        )


@dataclass(frozen=True)
class UnitSelectionReceipt:
    """What was handed to analysis: exact source, policy, and unit verdicts."""

    curation: CurationRef
    policy_name: str
    policy: LabelPolicy
    included_unit_ids: tuple[int, ...]
    excluded_units: Mapping[int, str]
    unlabeled_unit_ids: tuple[int, ...]
    groups: tuple[SelectedGroup, ...]
    selection_provenance: Mapping[str, Any] = field(default_factory=dict)
    selection_snapshot: UnitSelectionSnapshot | None = None

    @property
    def included_unlabeled_unit_ids(self) -> tuple[int, ...]:
        """Included units that carry no label (exclude-only policies only)."""
        return tuple(
            u for u in self.included_unit_ids if u in self.unlabeled_unit_ids
        )

    @property
    def observation(self):
        """Availability of a single-session population (members expose their own)."""
        self.group_key  # enforce the same single-session contract as spike data
        return self.groups[0].observation

    @property
    def group_key(self) -> Mapping[str, Any]:
        """The single group's key (single-session sorts)."""
        if len(self.groups) != 1:
            raise ValueError(
                "UnitSelectionReceipt.group_key is only defined for a single "
                f"group; this receipt has {len(self.groups)} (a concat sort "
                "has one group per member -- use .groups)."
            )
        return self.groups[0].group_key

    def fetch_spike_data(self, **kwargs):
        """Spike times of the selected units (single-session sorts)."""
        if len(self.groups) != 1:
            raise ValueError(
                "This receipt covers several per-member groups; call "
                "receipt.groups[i].fetch_spike_data(...) for the member you "
                "need."
            )
        return self.groups[0].fetch_spike_data(**kwargs)

    def summary(self) -> str:
        """Compact summary of this handoff.

        Reports the source generation, policy, included/excluded counts, MUA
        and unlabeled counts, and, for an empty selection, why and what to
        do next.
        """
        labels = (
            self.selection_snapshot.labels_by_unit
            if self.selection_snapshot is not None
            else _labels_by_unit(self.curation)
        )
        n_total = len(self.included_unit_ids) + len(self.excluded_units)
        included_mua = sum(
            1 for u in self.included_unit_ids if "mua" in labels.get(u, [])
        )
        include = list(self.policy["include_labels"])
        exclude = list(self.policy["exclude_labels"])
        policy_text = (
            f"require one of {include}; " if include else "no required label; "
        ) + (f"deny {exclude}" if exclude else "deny nothing")
        lines = [
            f"curation: sorting_id={self.curation.sorting_id} "
            f"curation_id={self.curation.curation_id} "
            f"(generation {self.curation.curation_uuid})",
            f"policy: {self.policy_name} -- {policy_text}",
            f"units: {n_total} total, {len(self.included_unit_ids)} selected, "
            f"{len(self.excluded_units)} excluded "
            f"({len(self.unlabeled_unit_ids)} unlabeled overall)",
            f"selected: {included_mua} labeled mua, "
            f"{len(self.included_unlabeled_unit_ids)} unlabeled",
            "groups: "
            + ", ".join(
                f"{g.group_key['sorted_spikes_group_name']} "
                f"[{g.nwb_file_name}, {g.status}]"
                for g in self.groups
            ),
        ]
        if self.selection_provenance.get("unit_criteria"):
            lines.append(
                "metric criteria: "
                f"{self.selection_provenance['unit_criteria']}; "
                f"evaluation: {self.selection_provenance.get('evaluation_id')}"
            )
        if not self.included_unit_ids:
            if n_total == 0:
                lines.append(
                    "empty selection: the curation holds no units (zero-unit "
                    "sort); there is nothing to select."
                )
            elif include and not any(
                set(labels.get(u, [])) & set(include)
                for u in self.excluded_units
            ):
                lines.append(
                    "empty selection: no unit carries a required label "
                    f"(one of {include}). Apply one of those labels in a "
                    "review and hand over that child, or choose an "
                    "exclude-only policy such as 'v2_unflagged_units' to "
                    "keep everything the rules did not flag."
                )
            elif self.selection_provenance.get("unit_criteria"):
                lines.append(
                    "empty selection: no units satisfy both the label policy "
                    "and metric criteria. See describe() for failed predicates "
                    "and unavailable values; review the criteria and evaluation."
                )
            else:
                lines.append(
                    "empty selection: every unit was excluded by the policy "
                    "(see describe() for each unit's reason); review the "
                    "labels or choose a different policy."
                )
        return "\n".join(lines)

    def describe(self):
        """One row per unit: verdict, labels, reason."""
        import pandas as pd

        labels = (
            self.selection_snapshot.labels_by_unit
            if self.selection_snapshot is not None
            else _labels_by_unit(self.curation)
        )
        rows = []
        for unit_id in sorted(
            set(self.included_unit_ids) | set(self.excluded_units)
        ):
            rows.append(
                {
                    "unit_id": unit_id,
                    "included": unit_id in self.included_unit_ids,
                    "labels": list(labels.get(unit_id, [])),
                    "reason": self.excluded_units.get(unit_id, "selected"),
                }
            )
        return pd.DataFrame(
            rows, columns=["unit_id", "included", "labels", "reason"]
        ).set_index("unit_id")


def ensure_v2_unit_selection_policies() -> None:
    """Insert the v2 ``UnitSelectionParams`` rows if absent.

    Also inserts the table's own defaults so ``all_units`` resolves on a
    fresh database.
    """
    from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams

    UnitSelectionParams().insert_default()
    rows = [
        {
            "unit_filter_params_name": name,
            "include_labels": list(policy["include_labels"]),
            "exclude_labels": list(policy["exclude_labels"]),
        }
        for name, policy in V2_UNIT_SELECTION_POLICIES.items()
    ]
    UnitSelectionParams().insert(rows, skip_duplicates=True)


def _labels_by_unit(curation: CurationRef) -> dict[int, list[str]]:
    from spyglass.spikesorting.v2.curation import CurationV2

    return CurationV2._labels_by_unit(curation.as_key())


def _resolve_policy(policy_name: str) -> tuple[LabelPolicy, dict]:
    """Return the stored include/exclude labels of a ``UnitSelectionParams`` row."""
    from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams

    rows = (
        UnitSelectionParams & {"unit_filter_params_name": policy_name}
    ).fetch(as_dict=True)
    if len(rows) != 1:
        raise ValueError(
            f"UnitSelectionParams row {policy_name!r} does not exist. Shipped "
            f"v2 policies: {sorted(V2_UNIT_SELECTION_POLICIES)}; production "
            "rows: all_units / exclude_noise / default_exclusion. "
            "Run initialize_v2_defaults() before resolving a population."
        )
    row = rows[0]
    return (
        _label_policy(row["include_labels"] or [], row["exclude_labels"] or []),
        row.get("unit_criteria") or {},
    )


def apply_unit_selection_policy(
    labels_by_unit: Mapping[int, list[str]],
    unit_ids,
    policy: Mapping[str, Sequence[str]],
) -> tuple[tuple[int, ...], dict[int, str]]:
    """Apply a label policy to units; return (included, {excluded: reason}).

    Uses ``filter_units_by_labels``, the function
    ``SortedSpikesGroup.fetch_spike_data`` applies, so the receipt matches
    what analysis reads. No database access.
    """
    from spyglass.spikesorting.analysis.v1._unit_filter import (
        filter_units_by_labels,
    )

    unit_ids = [int(u) for u in unit_ids]
    labels = [list(labels_by_unit.get(u, [])) for u in unit_ids]
    include = list(policy["include_labels"])
    exclude = list(policy["exclude_labels"])
    mask = filter_units_by_labels(labels, include, exclude)
    included = tuple(u for u, keep in zip(unit_ids, mask) if keep)
    excluded: dict[int, str] = {}
    for unit_id, unit_labels, keep in zip(unit_ids, labels, mask):
        if keep:
            continue
        denied = sorted(set(unit_labels) & set(exclude))
        if denied:
            excluded[unit_id] = f"denied label(s): {', '.join(denied)}"
        elif not unit_labels:
            excluded[unit_id] = (
                f"unlabeled; policy requires one of: {', '.join(include)}"
            )
        else:
            excluded[unit_id] = (
                f"labels {unit_labels} lack a required label "
                f"(one of: {', '.join(include)})"
            )
    return included, excluded


def resolve_unit_selection(
    curation,
    *,
    policy: str = DEFAULT_UNIT_SELECTION_POLICY,
    evaluation: EvaluationResult | None = None,
    unit_criteria: Mapping | None = None,
    annotation_sets: Sequence = (),
) -> UnitSelectionSnapshot:
    """Read and freeze a label/metric population without writing anything.

    Takes the same selection arguments as ``select_units_for_analysis``;
    defaults must already be initialized. The curation generation, policy,
    criteria, evaluation and annotation pins, labels, unit ids and exclusion
    reasons enter the snapshot hash. Does not require a registered output, so
    it also resolves a concat parent curation for matching.
    """
    import pandas as pd

    from spyglass.spikesorting.analysis.v1._unit_filter import (
        filter_units_by_criteria,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    ref = CurationRef.from_key(curation)
    key = ref.as_key()
    row = (CurationV2 & key).fetch1()
    CurationV2.assert_committed_curation(
        key,
        context="resolve_unit_selection",
        merges_applied=row["merges_applied"],
    )
    resolved_policy, stored_criteria = _resolve_policy(policy)
    from spyglass.spikesorting.v2._core.lookup_validation import _jsonable_blob

    criteria = json.loads(
        json.dumps(
            _jsonable_blob(
                dict(
                    stored_criteria if unit_criteria is None else unit_criteria
                )
            ),
            sort_keys=True,
        )
    )
    labels_by_unit = {
        unit: sorted(labels) for unit, labels in _labels_by_unit(ref).items()
    }
    unit_ids = sorted(int(u) for u in (CurationV2.Unit & key).fetch("unit_id"))
    included, excluded = apply_unit_selection_policy(
        labels_by_unit, unit_ids, resolved_policy
    )
    from spyglass.spikesorting.v2.annotation_api import (
        AnnotationSetRef,
        read_unit_properties,
    )

    annotations = tuple(AnnotationSetRef.from_key(v) for v in annotation_sets)
    if evaluation is not None:
        if not isinstance(evaluation, EvaluationResult):
            raise TypeError(
                "evaluation must be an EvaluationResult or explicit None."
            )
        evaluation._current_evaluation_key()
    if criteria or evaluation is not None or annotations:
        properties = read_unit_properties(
            ref, evaluation=evaluation, annotation_sets=annotations
        )
    # Same operators as v1 analysis groups; columns come from the evaluation.
    for column, criterion in criteria.items():
        mask = filter_units_by_criteria(properties, {column: criterion})
        for unit_id, passed in zip(properties.index, mask):
            if not passed and unit_id not in excluded:
                value = properties.at[unit_id, column]
                missing = pd.api.types.is_scalar(value) and pd.isna(value)
                excluded[int(unit_id)] = (
                    f"unavailable metric: {column}"
                    if missing
                    else f"criterion failed: {column} {criterion} (value={value})"
                )
    included = tuple(u for u in included if u not in excluded)
    from spyglass.spikesorting.v2._core.observed_time import (
        OBSERVATION_VERSION,
    )

    provenance = {
        "observation_version": OBSERVATION_VERSION,
        "curation_uuid": str(ref.curation_uuid),
        "label_policy": {k: list(v) for k, v in resolved_policy.items()},
        "unit_criteria": criteria,
        "evaluation_id": str(evaluation.evaluation_id) if evaluation else None,
        "evaluation_recipes": (
            {
                "metric_params_name": evaluation.spec.metric_params_name,
                "auto_curation_rules_name": evaluation.spec.auto_curation_rules_name,
            }
            if evaluation
            else None
        ),
        "annotation_sets": [a.snapshot() for a in annotations],
    }
    # Canonical JSON makes non-serializable criteria fail before writes.
    provenance = _jsonable_blob(provenance)
    payload = {
        "sorting_id": str(ref.sorting_id),
        "curation_id": ref.curation_id,
        "curation_uuid": str(ref.curation_uuid),
        "policy_name": policy,
        "policy": {key: list(value) for key, value in resolved_policy.items()},
        "unit_criteria": criteria,
        "included_unit_ids": list(included),
        "excluded_units": {
            str(unit): reason for unit, reason in excluded.items()
        },
        "labels_by_unit": {
            str(unit): list(labels_by_unit.get(unit, ())) for unit in unit_ids
        },
        "selection_provenance": provenance,
    }
    payload["selection_hash"] = unit_selection_hash(payload)
    return UnitSelectionSnapshot(
        curation=ref,
        _payload_json=json.dumps(payload, sort_keys=True),
        evaluation=evaluation,
        annotation_sets=annotations,
        _criteria_from_policy=unit_criteria is None,
    )


def select_units_for_analysis(
    curation,
    *,
    policy: str = DEFAULT_UNIT_SELECTION_POLICY,
    group_name: str | None = None,
    evaluation: EvaluationResult | None = None,
    unit_criteria: Mapping | None = None,
    annotation_sets: Sequence = (),
) -> UnitSelectionReceipt:
    """Freeze a label/metric-selected population for downstream analysis.

    Parameters
    ----------
    curation : CurationRef or mapping
        The exact curation generation to select from (a ``CurationRef``, or a
        ``{sorting_id, curation_id}`` key). Must be committed (not a merge
        preview) and registered in ``SpikeSortingOutput`` (single-session) or
        have per-member outputs (concat).
    policy : str
        ``UnitSelectionParams`` row name. Default
        ``"v2_accepted_single_units"`` (reviewer-accepted units only);
        ``"v2_accepted_neural_units"`` adds ``mua``; ``"v2_unflagged_units"``
        keeps everything the rules did not flag (auto-label-only workflows);
        ``"all_units"`` is the explicit expert choice.
    group_name : str, optional
        ``SortedSpikesGroup`` name. Default
        ``"v2_{policy}_{curation_uuid_hex[:12]}"``, unique per curation
        generation so re-running is idempotent.
    evaluation : EvaluationResult, optional
        Exact evaluation of this curation supplying metric columns.
    unit_criteria : mapping, optional
        Criteria in UnitSelectionParams syntax. Defaults to that policy's
        stored criteria; an explicit mapping overrides them. Missing values
        fail a predicate. Selected membership is persisted, including empty
        selections, so every downstream consumer uses the same population.
    annotation_sets : sequence, optional
        Exact custom property sets available to the criteria.

    Returns
    -------
    UnitSelectionReceipt
        The pinned curation, the policy content, included / excluded unit ids
        with reasons, and the group(s) downstream reads.
    """
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # resolve_unit_selection is read-only, so default rows are inserted here.
    ensure_v2_unit_selection_policies()
    selection = resolve_unit_selection(
        curation,
        policy=policy,
        evaluation=evaluation,
        unit_criteria=unit_criteria,
        annotation_sets=annotation_sets,
    )
    ref = selection.curation
    key = ref.as_key()
    resolved_policy = selection.policy
    criteria = selection.unit_criteria
    included = selection.included_unit_ids
    excluded = selection.excluded_units
    unlabeled = selection.unlabeled_unit_ids
    provenance = selection.selection_provenance
    annotations = selection.annotation_sets
    from spyglass.spikesorting.v2._storage.observation_io import (
        selection_observations,
    )
    from spyglass.spikesorting._observed_time import (
        normalize_observation_provenance,
    )

    # Group names hash provenance only, not the full snapshot hash.
    selection_hash = sha256_json(provenance, separators=(", ", ": "))[:12]

    if group_name is None:
        group_name = f"v2_{policy}_{ref.curation_uuid.hex[:12]}"
        if criteria or evaluation is not None or annotations:
            group_name += f"_{selection_hash}"
    if len(group_name) > 80:
        raise ValueError("group_name must be at most 80 characters.")

    source = SortingSelection.resolve_source({"sorting_id": ref.sorting_id})
    targets: list[tuple[str, uuid.UUID, int | None]] = []
    if source.kind == "recording":
        merge_id = ref.merge_id
        if merge_id is None:
            raise ValueError(
                "select_units_for_analysis: this curation has no "
                "SpikeSortingOutput row yet; it must be registered before it "
                "can be handed to analysis."
            )
        nwb_file_name = (RecordingSelection & source.key).fetch1(
            "nwb_file_name"
        )
        targets.append((str(nwb_file_name), merge_id, None))
    else:
        from spyglass.spikesorting.v2.concat_member_curation import (
            ConcatMemberCuration,
        )

        member_merge_ids = ref.member_merge_ids
        if not member_merge_ids:
            raise ValueError(
                "select_units_for_analysis: this concat curation has no "
                "per-member SpikeSortingOutput rows (ConcatMemberCuration); "
                "populate them before handing it to analysis. Its synthetic "
                "concat timeline is never used for analysis."
            )
        member_sessions = {
            int(r["member_index"]): str(r["nwb_file_name"])
            for r in (ConcatMemberCuration & key).fetch(
                "member_index", "nwb_file_name", as_dict=True
            )
        }
        for member_index in sorted(member_merge_ids):
            targets.append(
                (
                    member_sessions[member_index],
                    member_merge_ids[member_index],
                    member_index,
                )
            )

    groups: list[SelectedGroup] = []
    for nwb_file_name, merge_id, member_index in targets:
        name = (
            group_name
            if member_index is None
            else f"{group_name}_m{member_index}"
        )
        group_key = {
            "nwb_file_name": nwb_file_name,
            "sorted_spikes_group_name": name,
            "unit_filter_params_name": policy,
        }
        member_provenance = {
            **provenance,
            "observation_intervals": selection_observations(merge_id, included),
        }
        snapshot = {
            "selected_unit_ids": list(included),
            "selection_provenance": member_provenance,
        }
        existing = SortedSpikesGroup & group_key
        if existing:
            members = {
                uuid.UUID(str(m))
                for m in (SortedSpikesGroup.Units & group_key).fetch(
                    "spikesorting_merge_id"
                )
            }
            if members != {merge_id}:
                raise ValueError(
                    f"SortedSpikesGroup {group_key} already exists but points at "
                    f"{sorted(map(str, members))}, not this curation's output "
                    f"{merge_id}. Pass a different group_name."
                )
            stored = (SortedSpikesGroup.UnitSelection & group_key).fetch(
                as_dict=True
            )
            if (
                len(stored) != 1
                or list(stored[0]["selected_unit_ids"]) != list(included)
                or normalize_observation_provenance(
                    stored[0]["selection_provenance"]
                )
                != member_provenance
            ):
                raise ValueError(
                    "Existing group has a different selection; choose a new "
                    "group_name to preserve the earlier population."
                )
            status = "reused"
        else:
            SortedSpikesGroup().create_group(
                name,
                nwb_file_name,
                unit_filter_params_name=policy,
                keys=[{"spikesorting_merge_id": merge_id}],
                unit_selections={merge_id: snapshot},
            )
            status = "created"
        groups.append(
            SelectedGroup(
                nwb_file_name=nwb_file_name,
                group_key=MappingProxyType(group_key),
                merge_id=merge_id,
                member_index=member_index,
                status=status,
            )
        )

    return UnitSelectionReceipt(
        curation=ref,
        policy_name=policy,
        policy=resolved_policy,
        included_unit_ids=included,
        excluded_units=MappingProxyType(excluded),
        unlabeled_unit_ids=unlabeled,
        groups=tuple(groups),
        selection_provenance=MappingProxyType(provenance),
        selection_snapshot=selection,
    )
