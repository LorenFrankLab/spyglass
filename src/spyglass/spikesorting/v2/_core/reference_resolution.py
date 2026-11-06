"""DB-free reference validation and resolution for ``SortGroupV2``."""

from __future__ import annotations

from typing import Literal, get_args

# - ``"none"``: no re-referencing.
# - ``"global_median"``: common-median reference across the group.
# - ``"specific"``: subtract ``reference_electrode_id``, then drop it.
# Stored as varchar (not a MySQL enum) so modes can be added without altering
# a populated table.
ReferenceMode = Literal["none", "global_median", "specific"]

_VALID_REFERENCE_MODES: frozenset[str] = frozenset(get_args(ReferenceMode))


def _validate_reference_fields(row: dict) -> None:
    """Validate a ``SortGroupV2`` row's reference fields before insert.

    - ``reference_mode`` (default ``"none"``) is a ``ReferenceMode``.
    - ``reference_electrode_id`` is non-null iff the mode is ``'specific'``.

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

    The specific reference is removed after subtraction, which would silently
    drop that channel from the sort.

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


# "Derive from config"; not ``None``, which means "no reference".
_AUTO_REFERENCE = object()


def resolve_group_reference(
    configured_ref_ids,
    *,
    explicit_ref=_AUTO_REFERENCE,
    group_label="",
) -> tuple[ReferenceMode, int | None]:
    """Resolve one electrode group's reference to ``(reference_mode, id)``.

    Sentinels: ``None`` or ``-1`` -> ``("none", None)``;
    ``-2`` -> ``("global_median", None)``; ``>= 0`` -> ``("specific", id)``.

    Parameters
    ----------
    configured_ref_ids : iterable of (int or None)
        The ``original_reference_electrode`` value of every member electrode
        in the group. Only read on the auto-derivation path
        (``explicit_ref is _AUTO_REFERENCE``).
    explicit_ref : int, None, or _AUTO_REFERENCE, optional
        A reference electrode id / sentinel from a ``references`` mapping, or
        ``_AUTO_REFERENCE`` (default) to derive it from ``configured_ref_ids``.
        An explicit value, including ``None``, wins over config.
    group_label : str, optional
        Group name used in error messages.

    Returns
    -------
    (reference_mode, reference_electrode_id) : tuple of (str, int or None)

    Raises
    ------
    ValueError
        On the auto path, if the group has no members or mixed
        ``original_reference_electrode`` values. On any path, if the sentinel
        is negative and not ``-1`` / ``-2``.

    Notes
    -----
    Callers check membership with :func:`assert_reference_not_member` after
    applying ``omit_ref_electrode_group`` skips.
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
