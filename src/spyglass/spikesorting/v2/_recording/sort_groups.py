"""Pure planning helpers for ``SortGroupV2`` electrode grouping.

No DataJoint queries or schema activation; ``SortGroupV2`` fetches the
``Electrode`` rows and runs the inserts.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from spyglass.spikesorting.v2._core.reference_resolution import (
    assert_reference_not_member,
    resolve_group_reference,
)
from spyglass.utils import logger


def _electrode_group_sort_key(name):
    """Sort key for ``electrode_group_name`` tolerant of non-numeric names.

    Numeric names sort numerically and first, then other names lexically.
    ``isdecimal`` matches exactly the strings ``int()`` accepts.
    """
    text = str(name)
    return (0, int(text)) if text.isdecimal() else (1, text)


def _reference_electrode_group(all_electrodes, reference_electrode_id):
    """Validate a ``"specific"`` reference electrode and return its group.

    Searches the unfiltered electrode set, so a reference flagged bad is
    still found. The id must exist and belong to exactly one electrode
    group, since ``electrode_id`` can repeat across groups.

    Parameters
    ----------
    all_electrodes : numpy.recarray
        Every ``Electrode`` row for the session (unfiltered), as returned by
        ``Electrode().fetch()``.
    reference_electrode_id : int
        The resolved ``"specific"`` reference electrode id.

    Returns
    -------
    str
        The single ``electrode_group_name`` containing the reference
        electrode.
    """
    match = all_electrodes["electrode_id"] == reference_electrode_id
    owners = sorted(
        {str(g) for g in all_electrodes["electrode_group_name"][match]}
    )
    if len(owners) == 0:
        raise ValueError(
            "SortGroupV2: reference electrode "
            f"{reference_electrode_id} is not in the Electrode table for this "
            "session; a 'specific' reference must name a real electrode (it "
            "is subtracted from the recording, then dropped). Check the "
            "reference electrode id."
        )
    if len(owners) > 1:
        raise ValueError(
            "SortGroupV2: reference electrode "
            f"{reference_electrode_id} appears in multiple electrode groups "
            f"{owners}; its owning electrode group is ambiguous. Disambiguate "
            "the Electrode config or choose a unique reference electrode."
        )
    return owners[0]


class _SortGroupPlan(NamedTuple):
    """One resolved sort group, ready to materialize into table rows.

    ``electrodes`` is the ordered list of
    ``(electrode_group_name, electrode_id)`` pairs that become
    ``SortGroupElectrode`` part rows; the order is the write order.
    """

    sort_group_id: int
    reference_mode: str
    reference_electrode_id: int | None
    electrodes: list[tuple]


def _plan_sort_groups_by_shank(
    electrodes,
    all_electrodes,
    nwb_file_name: str,
    *,
    omit_ref_electrode_group: bool,
    omit_unitrode: bool,
    references: dict | None,
    override_pair: tuple[str, int | None] | None,
) -> tuple[list[tuple], list[dict]]:
    """Plan sort groups by probe shank (pure; no DataJoint access).

    ``electrodes`` is bad-channel-filtered; ``all_electrodes`` is not.
    ``override_pair`` is a call-wide ``(reference_mode,
    reference_electrode_id)``, or ``None`` to resolve per group;
    ``references`` optionally maps electrode group to reference.

    Returns ``(proposed, skipped)`` where each ``proposed`` entry is
    ``(electrode_group_name, electrode_ids, reference_mode,
    reference_electrode_id)`` (sort_group_ids are assigned by the caller)
    and ``skipped`` is the list of skip records. Raises ``ValueError`` if
    no sort group survives filtering, or for the reference-resolution
    failures documented on ``set_group_by_shank``.
    """
    e_groups = sorted(
        np.unique(electrodes["electrode_group_name"]).tolist(),
        key=_electrode_group_sort_key,
    )
    # Each entry: (e_group, electrode_ids, resolved_mode, resolved_ref_id).
    proposed: list[tuple[str, np.ndarray, str, int | None]] = []
    skipped: list[dict] = []
    for e_group in e_groups:
        in_group = electrodes["electrode_group_name"] == e_group
        shanks = np.unique(electrodes["probe_shank"][in_group])
        for shank in shanks:
            mask = in_group & (electrodes["probe_shank"] == shank)
            electrode_ids = electrodes["electrode_id"][mask]
            if omit_unitrode and len(electrode_ids) == 1:
                logger.warning(
                    f"set_group_by_shank: skipping {nwb_file_name!r} "
                    f"electrode group {e_group} shank {shank} -- "
                    "single-channel unitrode."
                )
                skipped.append(
                    {
                        "electrode_group_name": str(e_group),
                        "shank": int(shank),
                        "reason": "unitrode",
                    }
                )
                continue

            if override_pair is not None:
                resolved_mode, resolved_ref_id = override_pair
            elif references is not None:
                if e_group not in references:
                    raise ValueError(
                        "set_group_by_shank: electrode group "
                        f"{e_group!r} produces a sort group but is missing "
                        "from `references`. Add a reference electrode id / "
                        "sentinel for it, or omit `references` to inherit "
                        "from the Electrode config."
                    )
                resolved_mode, resolved_ref_id = resolve_group_reference(
                    electrodes["original_reference_electrode"][mask],
                    explicit_ref=references[e_group],
                    group_label=str(e_group),
                )
            else:
                resolved_mode, resolved_ref_id = resolve_group_reference(
                    electrodes["original_reference_electrode"][mask],
                    group_label=str(e_group),
                )

            # Validated regardless of omit, so a bad id fails here.
            if resolved_mode == "specific":
                ref_group = _reference_electrode_group(
                    all_electrodes, resolved_ref_id
                )
                if omit_ref_electrode_group and str(ref_group) == str(e_group):
                    logger.warning(
                        f"set_group_by_shank: skipping "
                        f"{nwb_file_name!r} electrode group "
                        f"{e_group} -- contains the reference electrode."
                    )
                    skipped.append(
                        {
                            "electrode_group_name": str(e_group),
                            "shank": int(shank),
                            "reason": "reference_electrode_group",
                        }
                    )
                    continue

            assert_reference_not_member(
                resolved_mode, resolved_ref_id, electrode_ids
            )
            proposed.append(
                (e_group, electrode_ids, resolved_mode, resolved_ref_id)
            )

    if not proposed:
        raise ValueError(
            f"set_group_by_shank: no sort groups produced for "
            f"{nwb_file_name!r} after filtering. Loosen "
            "omit_unitrode / omit_ref_electrode_group, or use "
            "set_group_by_electrode_table_column to group by a "
            "different column."
        )
    return proposed, skipped


def _plan_sort_groups_by_column(
    electrodes,
    all_electrodes,
    nwb_file_name: str,
    column: str,
    resolved_column: str,
    *,
    groups: list[list],
    sort_group_ids: list[int],
    omit_unitrode: bool,
    override_pair: tuple[str, int | None] | None,
) -> tuple[list[tuple], list[dict]]:
    """Plan sort groups by an Electrode-table column (pure; no DataJoint).

    ``all_electrodes`` validates a ``"specific"`` reference.
    ``sort_group_ids`` aligns with ``groups``; a skipped group leaves a gap.

    Returns ``(proposed, skipped)`` where each ``proposed`` entry is
    ``(sort_group_id, group_electrodes, reference_mode,
    reference_electrode_id)``. Raises ``ValueError`` if a group matches no
    electrodes, if no group survives filtering, or for the
    reference-resolution failures documented on
    ``set_group_by_electrode_table_column``.
    """
    # Each entry: (sort_group_id, group_electrodes, resolved_mode, resolved_ref_id).
    proposed: list[tuple[int, np.ndarray, str, int | None]] = []
    skipped: list[dict] = []
    for sort_group_id, values in zip(sort_group_ids, groups):
        mask = np.isin(electrodes[resolved_column], values)
        group_electrodes = electrodes[mask]
        if len(group_electrodes) == 0:
            raise ValueError(
                f"set_group_by_electrode_table_column: sort group "
                f"{sort_group_id} (column={column!r}, values={list(values)}) "
                f"matched no electrodes for {nwb_file_name!r}."
            )
        if omit_unitrode and len(group_electrodes) == 1:
            logger.warning(
                f"set_group_by_electrode_table_column: skipping sort "
                f"group {sort_group_id} -- single-channel unitrode."
            )
            skipped.append(
                {"sort_group_id": int(sort_group_id), "reason": "unitrode"}
            )
            continue

        if override_pair is not None:
            resolved_mode, resolved_ref_id = override_pair
        else:
            resolved_mode, resolved_ref_id = resolve_group_reference(
                group_electrodes["original_reference_electrode"],
                group_label=str(sort_group_id),
            )
        if resolved_mode == "specific":
            _reference_electrode_group(all_electrodes, resolved_ref_id)
        assert_reference_not_member(
            resolved_mode,
            resolved_ref_id,
            group_electrodes["electrode_id"],
        )
        proposed.append(
            (
                sort_group_id,
                group_electrodes,
                resolved_mode,
                resolved_ref_id,
            )
        )

    if not proposed:
        raise ValueError(
            f"set_group_by_electrode_table_column: no sort groups "
            f"survived filtering for {nwb_file_name!r}. Loosen "
            "omit_unitrode or revisit the groups list."
        )
    return proposed, skipped


def _build_sort_group_rows(
    nwb_file_name: str, plans: list[_SortGroupPlan]
) -> tuple[list[dict], list[dict]]:
    """Build SortGroupV2 master + SortGroupElectrode part rows (pure).

    One master row per plan and one part row per electrode, in plan order.
    """
    master_rows, part_rows = [], []
    for plan in plans:
        master_rows.append(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": plan.sort_group_id,
                "reference_mode": plan.reference_mode,
                "reference_electrode_id": plan.reference_electrode_id,
            }
        )
        for electrode_group_name, electrode_id in plan.electrodes:
            part_rows.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": plan.sort_group_id,
                    "electrode_group_name": electrode_group_name,
                    "electrode_id": int(electrode_id),
                }
            )
    return master_rows, part_rows
