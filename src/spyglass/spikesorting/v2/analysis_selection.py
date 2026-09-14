"""Explicit unit selection: the handoff from a curation to downstream analysis.

A curation -- root, auto-labeled, or manually curated -- holds EVERY unit with
its labels; its ``SpikeSortingOutput`` merge id is a registered output, not a
filtered population. ``SpikeSortingOutput.get_spike_times`` returns all units.
The supported handoff is :func:`select_units_for_analysis`: it applies a named
``UnitSelectionParams`` policy to the curation's labels, builds the
``SortedSpikesGroup`` downstream analyses read (``fetch_spike_data`` and the
decoding / firing-rate consumers), and returns a receipt naming the exact
curation generation, the policy content, and every included / excluded unit
with its reason.

Three v2 policies are shipped (``V2_UNIT_SELECTION_POLICIES``); they are new
rows and never alter the production ``all_units`` / ``exclude_noise`` /
``default_exclusion`` rows:

* ``v2_accepted_single_units`` (default) -- require ``accept``; deny ``mua``,
  ``noise``, ``reject``, ``artifact``. Unlabeled units are excluded: a unit
  is in the analysis only because a reviewer accepted it.
* ``v2_accepted_neural_units`` -- require ``accept`` or ``mua``; deny
  ``noise``, ``reject``, ``artifact``. Unlabeled units are excluded.
* ``v2_unflagged_units`` -- deny ``noise``, ``reject``, ``artifact``; every
  other unit, MUA and unlabeled included. This is the auto-label-only
  handoff: the shipped rule sets only FLAG bad units (they never write
  ``accept``), so without a browser review the two accepted policies select
  nothing. Choosing it states explicitly that rule-passing, never-reviewed
  units count as analysis units.

``all_units`` remains available as an explicit expert choice; the receipt
always reports which included units were unlabeled so the choice is visible.

Concat sorts: the curation has one session-timeline output per frozen member
(``CurationRef.member_merge_ids``), and a ``SortedSpikesGroup`` is per session,
so the handoff builds one group per member and the receipt lists them.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from spyglass.spikesorting.v2.curation_api import CurationRef

# New v2 policies (see the module doc). Inserted into ``UnitSelectionParams``
# on first use by ``ensure_v2_unit_selection_policies`` -- additive only.
V2_UNIT_SELECTION_POLICIES: Mapping[str, Mapping[str, list[str]]] = (
    MappingProxyType(
        {
            "v2_accepted_single_units": {
                "include_labels": ["accept"],
                "exclude_labels": ["mua", "noise", "reject", "artifact"],
            },
            "v2_accepted_neural_units": {
                "include_labels": ["accept", "mua"],
                "exclude_labels": ["noise", "reject", "artifact"],
            },
            "v2_unflagged_units": {
                "include_labels": [],
                "exclude_labels": ["noise", "reject", "artifact"],
            },
        }
    )
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
    policy: Mapping[str, list[str]]
    included_unit_ids: tuple[int, ...]
    excluded_units: Mapping[int, str]
    unlabeled_unit_ids: tuple[int, ...]
    groups: tuple[SelectedGroup, ...]

    @property
    def included_unlabeled_unit_ids(self) -> tuple[int, ...]:
        """Included units that carry no label (visible only under an
        exclude-only policy such as ``v2_unflagged_units`` / ``all_units``)."""
        return tuple(
            u for u in self.included_unit_ids if u in self.unlabeled_unit_ids
        )

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

    def describe(self):
        """One row per unit: verdict, labels, reason."""
        import pandas as pd

        labels = _labels_by_unit(self.curation)
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
    """Insert the v2 ``UnitSelectionParams`` rows if absent (additive).

    Also seeds the table's own production defaults (``all_units`` /
    ``exclude_noise`` / ``default_exclusion``) through its ``insert_default``
    so the explicit ``all_units`` expert choice resolves on a fresh database.
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


def _resolve_policy(policy_name: str) -> dict[str, list[str]]:
    """Return the stored include/exclude lists of a ``UnitSelectionParams`` row."""
    from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams

    ensure_v2_unit_selection_policies()
    rows = (
        UnitSelectionParams & {"unit_filter_params_name": policy_name}
    ).fetch(as_dict=True)
    if len(rows) != 1:
        raise ValueError(
            f"UnitSelectionParams row {policy_name!r} does not exist. Shipped "
            f"v2 policies: {sorted(V2_UNIT_SELECTION_POLICIES)}; production "
            "rows: all_units / exclude_noise / default_exclusion."
        )
    row = rows[0]
    if row.get("unit_criteria"):
        raise ValueError(
            f"UnitSelectionParams row {policy_name!r} carries unit_criteria; "
            "select_units_for_analysis applies label policies only."
        )
    return {
        "include_labels": [str(v) for v in (row["include_labels"] or [])],
        "exclude_labels": [str(v) for v in (row["exclude_labels"] or [])],
    }


def apply_unit_selection_policy(
    labels_by_unit: Mapping[int, list[str]],
    unit_ids,
    policy: Mapping[str, list[str]],
) -> tuple[tuple[int, ...], dict[int, str]]:
    """Apply a label policy to units; return (included, {excluded: reason}).

    Uses ``filter_units_by_labels`` -- the exact function
    ``SortedSpikesGroup.fetch_spike_data`` applies downstream -- so the
    receipt cannot disagree with what analysis reads. Pure: no database
    access.
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


def select_units_for_analysis(
    curation,
    *,
    policy: str = DEFAULT_UNIT_SELECTION_POLICY,
    group_name: str | None = None,
) -> UnitSelectionReceipt:
    """Hand a curation to downstream analysis under an explicit label policy.

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
        ``"v2_{policy}_{curation_uuid_hex[:12]}"`` -- unique per curation
        generation, so re-running is idempotent and a recreated curation
        never reuses a stale group.

    Returns
    -------
    UnitSelectionReceipt
        The pinned curation, the policy content, included / excluded unit ids
        with reasons, and the group(s) downstream reads.
    """
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    ref = CurationRef.from_key(curation)
    key = ref.as_key()
    row = (CurationV2 & key).fetch1()
    CurationV2.assert_committed_curation(
        key,
        context="select_units_for_analysis",
        merges_applied=row["merges_applied"],
    )
    resolved_policy = _resolve_policy(policy)
    labels_by_unit = _labels_by_unit(ref)
    unit_ids = sorted(int(u) for u in (CurationV2.Unit & key).fetch("unit_id"))
    included, excluded = apply_unit_selection_policy(
        labels_by_unit, unit_ids, resolved_policy
    )
    unlabeled = tuple(u for u in unit_ids if not labels_by_unit.get(u))

    if group_name is None:
        group_name = f"v2_{policy}_{ref.curation_uuid.hex[:12]}"
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
            status = "reused"
        else:
            SortedSpikesGroup().create_group(
                name,
                nwb_file_name,
                unit_filter_params_name=policy,
                keys=[{"spikesorting_merge_id": merge_id}],
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
        policy=MappingProxyType(resolved_policy),
        included_unit_ids=included,
        excluded_units=MappingProxyType(excluded),
        unlabeled_unit_ids=unlabeled,
        groups=tuple(groups),
    )
