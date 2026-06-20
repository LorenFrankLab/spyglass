"""Pure reference-resolution helpers for ``SortGroupV2`` grouping.

The ``reference_mode`` / ``reference_electrode_id`` validation and the
config-or-explicit reference resolution live here, free of DataJoint /
SpikeInterface, so they import without a DB connection and are
unit-testable in isolation. ``utils`` re-exports these names so existing
``from .utils import resolve_group_reference`` (etc.) call sites are
unchanged.
"""

from __future__ import annotations

from typing import Literal

# How a ``SortGroupV2`` references its channels before sorting:
#   * ``"none"`` -- no re-referencing.
#   * ``"global_median"`` -- common-median reference across the group.
#   * ``"specific"`` -- subtract a single named reference electrode
#     (``reference_electrode_id``), then drop it from the sorted channel set.
#
# Stored as a ``varchar(32)`` on ``SortGroupV2`` validated against this
# Literal at insert time (NOT a MySQL enum). The set may grow --
# SpikeInterface also supports ``global_average`` / CAR and local per-group
# referencing -- and an enum would trap a future mode behind a forbidden
# ``ALTER TABLE`` under the zero-migration policy. The Literal gives
# identical typo protection at the ``insert1`` boundary without the migration
# risk; same decision as ``CurationLabel``. It replaced a single
# ``sort_reference_electrode_id`` int whose magic sentinels (-1 none, -2
# global median, >=0 specific) conflated the mode with the channel id.
ReferenceMode = Literal["none", "global_median", "specific"]

_VALID_REFERENCE_MODES: frozenset[str] = frozenset(
    ("none", "global_median", "specific")
)


def _validate_reference_fields(row: dict) -> None:
    """Validate a ``SortGroupV2`` row's reference fields before insert.

    Enforces the two invariants the ``reference_mode`` /
    ``reference_electrode_id`` split must hold:

    1. ``reference_mode`` (defaulting to ``"none"`` when absent) is a
       member of the ``ReferenceMode`` Literal -- the varchar's typo
       guard, replacing what a MySQL enum would have done at the DB.
    2. ``reference_electrode_id`` is non-null **iff**
       ``reference_mode == 'specific'``: a specific reference needs a
       channel to subtract, and a non-specific mode must not carry a
       stray channel id the runtime would silently ignore.

    Raises
    ------
    ValueError
        If ``reference_mode`` is not a valid ``ReferenceMode``, if a
        ``'specific'`` mode omits ``reference_electrode_id``, or if a
        non-``'specific'`` mode carries a non-null
        ``reference_electrode_id``.
    """
    mode = row.get("reference_mode", "none")
    if mode not in _VALID_REFERENCE_MODES:
        raise ValueError(
            f"SortGroupV2: reference_mode {mode!r} is not a valid "
            f"ReferenceMode. Valid modes: {sorted(_VALID_REFERENCE_MODES)}."
        )
    ref_id = row.get("reference_electrode_id")
    if mode == "specific" and ref_id is None:
        raise ValueError(
            "SortGroupV2: reference_mode='specific' requires a non-null "
            "reference_electrode_id (the channel to subtract)."
        )
    if mode != "specific" and ref_id is not None:
        raise ValueError(
            f"SortGroupV2: reference_mode={mode!r} must leave "
            f"reference_electrode_id NULL; got {ref_id!r}. Only "
            "reference_mode='specific' names a reference electrode."
        )


def assert_reference_not_member(
    reference_mode, reference_electrode_id, sort_group_channel_ids
) -> None:
    """Reject a ``'specific'`` reference that is also a sort-group channel.

    The write path subtracts the specific reference from every channel and then
    REMOVES it from the recording. If the reference electrode is itself a member
    of the sort group, that silently drops a channel the user meant to sort
    (a 4-wire tetrode would sort on 3). v1 silently dropped it via
    ``setdiff1d``; v2 fails loud. No-op for non-``'specific'`` modes (which
    carry no reference electrode).

    Parameters
    ----------
    reference_mode : str
        The group's ``ReferenceMode``. Only ``'specific'`` is checked;
        any other mode is a no-op.
    reference_electrode_id : int
        The electrode id subtracted as the specific reference.
    sort_group_channel_ids : iterable of int
        The sort group's member channel ids.

    Raises
    ------
    ValueError
        If ``reference_mode == 'specific'`` and
        ``reference_electrode_id`` is itself a member of
        ``sort_group_channel_ids``.
    """
    if reference_mode != "specific":
        return
    group = {int(c) for c in sort_group_channel_ids}
    if int(reference_electrode_id) in group:
        raise ValueError(
            "SortGroupV2: reference_electrode_id "
            f"{int(reference_electrode_id)} is also a sort-group channel; a "
            "'specific' reference is subtracted from every channel and then "
            "removed, which would silently drop that channel from the sort. "
            "Choose a reference electrode outside the sort group."
        )


# Private sentinel for "derive the reference from the per-electrode config"
# in :func:`resolve_group_reference`. A distinct object (not ``None``) so an
# explicit/configured ``None`` can still mean "no reference" -- ``None`` is a
# real, v1-compatible sentinel value (``original_reference_electrode`` is
# nullable), so it cannot double as the "auto-derive" marker.
_AUTO_REFERENCE = object()


def resolve_group_reference(
    configured_ref_ids,
    *,
    explicit_ref=_AUTO_REFERENCE,
    group_label="",
) -> tuple[ReferenceMode, int | None]:
    """Resolve one electrode group's reference to ``(reference_mode, id)``.

    Maps a sort group's configured or explicitly-supplied reference to a
    validated ``(reference_mode, reference_electrode_id)`` pair that
    satisfies :func:`_validate_reference_fields`. Emits only the three real
    ``ReferenceMode`` values -- never ``"auto"`` -- so the resolved pair can
    be written straight into a ``SortGroupV2`` master row.

    Sentinels (v1-compatible): ``None`` or ``-1`` -> ``("none", None)``;
    ``-2`` -> ``("global_median", None)``; ``>= 0`` -> ``("specific", id)``.

    Parameters
    ----------
    configured_ref_ids : iterable of (int or None)
        The ``original_reference_electrode`` value of every member electrode
        in the group. Only read on the auto-derivation path
        (``explicit_ref is _AUTO_REFERENCE``).
    explicit_ref : int, None, or _AUTO_REFERENCE, optional
        A reference electrode id / sentinel from a ``references`` mapping, or
        the private ``_AUTO_REFERENCE`` sentinel (the default) to derive the
        reference from ``configured_ref_ids``. An explicit value -- including
        an explicit ``None`` ("no reference") -- always wins over config.
    group_label : str, optional
        Group name used in error messages.

    Returns
    -------
    (reference_mode, reference_electrode_id) : tuple of (str, int or None)

    Raises
    ------
    ValueError
        On the auto path, if the members carry mixed
        ``original_reference_electrode`` values (cannot pick one reference;
        v1 silently fell through here). For any path, if the resolved
        sentinel is a negative value other than ``-1`` / ``-2``.

    Notes
    -----
    Membership validation is intentionally outside this resolver: helpers that
    support ``omit_ref_electrode_group`` must honor those skips first, then
    call :func:`assert_reference_not_member` for the groups they actually
    insert.
    """
    if explicit_ref is not _AUTO_REFERENCE:
        ref = -1 if explicit_ref is None else int(explicit_ref)
    else:
        uniq = sorted({-1 if r is None else int(r) for r in configured_ref_ids})
        if len(uniq) == 0:
            raise ValueError(
                "SortGroupV2: electrode group "
                f"{group_label!r} has no member electrodes to auto-derive a "
                "reference from. Pass an explicit `references` entry, or check "
                "the group membership."
            )
        if len(uniq) != 1:
            raise ValueError(
                "SortGroupV2: electrode group "
                f"{group_label!r} has mixed original_reference_electrode "
                f"values {uniq}; cannot auto-derive a single reference. "
                "Pass an explicit `references` entry for this group, or fix "
                "the Electrode config."
            )
        ref = uniq[0]

    if ref == -1:
        return "none", None
    if ref == -2:
        return "global_median", None
    if ref < 0:
        raise ValueError(
            f"SortGroupV2: electrode group {group_label!r} has unrecognized "
            f"reference sentinel {ref}; expected -1 (none), -2 "
            "(global_median), or a non-negative electrode id."
        )
    return "specific", ref
