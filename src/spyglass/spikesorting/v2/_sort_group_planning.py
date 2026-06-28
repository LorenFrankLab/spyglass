"""Pure planning helpers for ``SortGroupV2`` electrode grouping.

These functions resolve electrode-grouping decisions (by probe shank or
by an arbitrary ``Electrode``-table column) and materialize the resulting
table rows. They issue no DataJoint queries and trigger no ``dj.schema``
activation -- ``SortGroupV2`` fetches the ``Electrode`` rows and runs the
inserts; everything in between lives here so it is unit-testable without a
live DB connection (the test imports THIS module, not a ``@schema``
module, so collection needs no container). "No DB" means no connection /
activation, NOT "no DataJoint installed" -- importing any spyglass module
still pulls DataJoint / SpikeInterface via the package ``__init__``.
Mirrors the ``_curation_transforms`` / ``_signal_math`` service modules.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from spyglass.spikesorting.v2.utils import (
    assert_reference_not_member,
    resolve_group_reference,
)
from spyglass.utils import logger


def _electrode_group_sort_key(name):
    """Sort key for ``electrode_group_name`` tolerant of non-numeric names.

    NWB ``electrode_group_name`` is a free-form string. A plain
    ``int(name)`` raises ``ValueError`` on names like ``"probeA"``; this
    key sorts all-numeric names numerically (and ahead of any non-numeric
    name), then non-numeric names lexically -- so purely numeric group
    names keep their natural order while arbitrary names no longer crash.
    ``isdecimal`` (not ``isdigit``) is the gate so the predicate matches
    exactly the strings ``int()`` accepts.
    """
    text = str(name)
    return (0, int(text)) if text.isdecimal() else (1, text)


def _reference_electrode_group(all_electrodes, reference_electrode_id):
    """Validate a ``"specific"`` reference electrode and return its group.

    Looks the reference electrode up in the FULL (not bad-channel-filtered)
    session electrode set -- so a configured reference flagged
    ``bad_channel='True'`` is still found -- and enforces two invariants that
    a resolved ``"specific"`` reference must satisfy before any row is
    inserted:

    * the id must exist, so a ``"specific"`` sort group never persists a
      reference the recording cannot subtract (which would otherwise pass
      group creation and only fail later inside ``Recording.populate``);
    * it must belong to exactly one electrode group. The ``Electrode`` primary
      key is ``(nwb_file_name, electrode_group_name, electrode_id)``, so the
      same ``electrode_id`` can appear under more than one
      ``electrode_group_name``; an ambiguous owner is rejected rather than
      silently resolved to the first match.

    ``set_group_by_shank`` uses the returned group to honor
    ``omit_ref_electrode_group``; ``set_group_by_electrode_table_column``
    calls it for the existence / uniqueness validation only.

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

    Given the bad-channel-filtered ``electrodes`` and the full
    ``all_electrodes`` (both as returned by ``Electrode().fetch()``),
    enumerate ``(electrode_group, shank)`` groupings and resolve each
    group's reference, applying the ``omit_unitrode`` /
    ``omit_ref_electrode_group`` skips. ``override_pair`` is the
    call-wide ``(reference_mode, reference_electrode_id)`` override (or
    ``None`` to inherit per group); ``references`` is the optional
    per-electrode-group reference mapping.

    Returns ``(proposed, skipped)`` where each ``proposed`` entry is
    ``(electrode_group_name, electrode_ids, reference_mode,
    reference_electrode_id)`` (sort_group_ids are assigned by the caller)
    and ``skipped`` is the list of skip records. Raises ``ValueError`` if
    no sort group survives filtering, or for the reference-resolution
    failures documented on ``set_group_by_shank``.
    """
    # Enumerate (electrode_group, shank) groupings.
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

            # Resolve THIS group's reference (per group, not call-wide).
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

            # A "specific" reference must name a real, unambiguous
            # electrode; validate it (and locate its owning group)
            # regardless of omit, so a bad id fails here instead of later
            # inside Recording.populate.
            if resolved_mode == "specific":
                ref_group = _reference_electrode_group(
                    all_electrodes, resolved_ref_id
                )
                # omit_ref_electrode_group: skip the electrode group that
                # CONTAINS this group's resolved specific reference.
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

            # Fail early if the resolved specific reference is itself a
            # member of the sort group we are about to insert (it would be
            # subtracted then dropped, silently shrinking the group).
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

    ``electrodes`` is the (optionally bad-channel-filtered) electrode set
    matched against; ``all_electrodes`` is the full session set used to
    validate a resolved ``"specific"`` reference. ``sort_group_ids`` is
    the per-input-group id list (already resolved by the caller, same
    length as ``groups``); a skipped group leaves a gap in the assigned
    ids, matching the historical behavior.

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

        # Resolve THIS group's reference (per group, not call-wide): the
        # call-wide override if given, else inherit from the group's
        # members' configured original_reference_electrode.
        if override_pair is not None:
            resolved_mode, resolved_ref_id = override_pair
        else:
            resolved_mode, resolved_ref_id = resolve_group_reference(
                group_electrodes["original_reference_electrode"],
                group_label=str(sort_group_id),
            )
        # A "specific" reference must name a real, unambiguous session
        # electrode -- fail here instead of later in Recording.populate.
        if resolved_mode == "specific":
            _reference_electrode_group(all_electrodes, resolved_ref_id)
        # Fail early if the resolved specific reference is itself a member
        # of the sort group (it would be subtracted then dropped).
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

    Materializes the resolved ``plans`` into the two row lists the
    transaction inserts: one master row per plan and one part row per
    ``(electrode_group_name, electrode_id)`` pair, in plan order. No
    DataJoint access -- unit-testable in isolation.
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
