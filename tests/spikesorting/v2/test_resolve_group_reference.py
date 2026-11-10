"""Pure-resolver tests for SortGroupV2 reference inheritance.

Sentinels: ``-1`` / ``None`` -> none, ``-2`` -> global median, ``>= 0`` ->
a specific electrode. No database.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._core.reference_resolution import (
    assert_reference_not_member,
    resolve_group_reference,
)


def test_auto_specific_uniform_positive():
    # Uniform positive original_reference_electrode -> that specific electrode.
    assert resolve_group_reference([5, 5, 5, 5]) == ("specific", 5)


def test_auto_specific_numpy_array():
    # numpy scalars resolve the same as Python ints.
    refs = np.array([7, 7, 7], dtype=np.int64)
    assert resolve_group_reference(refs) == ("specific", 7)


def test_auto_none_uniform_minus_one():
    assert resolve_group_reference([-1, -1, -1]) == ("none", None)


def test_auto_none_uniform_python_none():
    # A configured None is "no reference", not _AUTO_REFERENCE.
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
    # An explicit reference wins over mixed config.
    assert resolve_group_reference([3, 7], explicit_ref=-1) == ("none", None)


def test_explicit_none_is_no_reference_not_auto():
    # Explicit None does not fall back to config derivation.
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
    # An empty member set has its own error, not "mixed values []".
    with pytest.raises(ValueError, match="no member electrodes"):
        resolve_group_reference([], group_label="probeZ")


def test_resolved_specific_in_group_is_rejected_by_membership_check():
    # A member reference would be subtracted then dropped, shrinking the group.
    mode, ref_id = resolve_group_reference([2, 2, 2, 2])
    assert (mode, ref_id) == ("specific", 2)
    with pytest.raises(ValueError, match="also a sort-group channel"):
        assert_reference_not_member(mode, ref_id, [0, 1, 2, 3])
