"""Pure-resolver tests for SortGroupV2 reference inheritance.

``resolve_group_reference`` maps one sort group's configured / explicit
reference to a validated ``(reference_mode, reference_electrode_id)`` pair
using the v1-compatible sentinels (``-1`` / ``None`` -> none, ``-2`` ->
global median, ``>= 0`` -> a specific electrode). The grouping helpers call
it per electrode group; these tests pin every sentinel branch and every
raise without a database (plain Python / numpy inputs, no fixture).

The membership check (``assert_reference_not_member``) lives in
``test_reference_member_guard.py``; the final test here exercises the
combined contract -- resolve to ``"specific"``, then reject a reference that
is a member of the group -- the way the helpers chain them.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2.utils import (
    assert_reference_not_member,
    resolve_group_reference,
)


def test_auto_specific_uniform_positive():
    # Uniform positive original_reference_electrode -> that specific electrode.
    assert resolve_group_reference([5, 5, 5, 5]) == ("specific", 5)


def test_auto_specific_numpy_array():
    # The helpers pass a numpy int array (Electrode column slice); numpy
    # scalars must resolve the same as Python ints, not leak through as a
    # non-int type.
    refs = np.array([7, 7, 7], dtype=np.int64)
    assert resolve_group_reference(refs) == ("specific", 7)


def test_auto_none_uniform_minus_one():
    assert resolve_group_reference([-1, -1, -1]) == ("none", None)


def test_auto_none_uniform_python_none():
    # A configured None must mean "no reference" -- NOT be confused with the
    # private _AUTO_REFERENCE sentinel that triggers config derivation.
    assert resolve_group_reference([None, None]) == ("none", None)


def test_auto_global_median_uniform_minus_two():
    assert resolve_group_reference([-2, -2, -2]) == ("global_median", None)


def test_auto_mixed_raises_naming_group_and_values():
    with pytest.raises(ValueError) as excinfo:
        resolve_group_reference([3, 7], group_label="probeA")
    msg = str(excinfo.value)
    assert "probeA" in msg
    assert "mixed" in msg
    # The offending values are surfaced (sorted) so the user can fix config.
    assert "[3, 7]" in msg


def test_explicit_overrides_mixed_config_no_raise():
    # An explicit reference wins over (and silences) mixed config: explicit
    # -1 forces "none" even though the configured values disagree.
    assert resolve_group_reference([3, 7], explicit_ref=-1) == ("none", None)


def test_explicit_none_is_no_reference_not_auto():
    # Explicit None ("no reference") must NOT fall back to config derivation,
    # even when config is mixed (which would otherwise raise).
    assert resolve_group_reference([3, 7], explicit_ref=None) == ("none", None)


def test_explicit_specific_and_global_median():
    assert resolve_group_reference([], explicit_ref=9) == ("specific", 9)
    assert resolve_group_reference([], explicit_ref=-2) == (
        "global_median",
        None,
    )


def test_bad_sentinel_raises():
    # Any negative value other than -1 / -2 is an unrecognized sentinel.
    with pytest.raises(ValueError, match="unrecognized"):
        resolve_group_reference([], explicit_ref=-3)


def test_auto_empty_members_raises_clearly():
    # Auto-deriving from an empty member set is a distinct error (no members
    # to read), not the confusing "mixed values []" message.
    with pytest.raises(ValueError, match="no member electrodes"):
        resolve_group_reference([], group_label="probeZ")


def test_resolved_specific_in_group_is_rejected_by_membership_check():
    # The helper-level contract: resolve to a specific reference, then reject
    # it when that electrode is itself a member of the sort group (it would be
    # subtracted then dropped, silently shrinking the group).
    mode, ref_id = resolve_group_reference([2, 2, 2, 2])
    assert (mode, ref_id) == ("specific", 2)
    with pytest.raises(ValueError, match="also a sort-group channel"):
        assert_reference_not_member(mode, ref_id, [0, 1, 2, 3])
