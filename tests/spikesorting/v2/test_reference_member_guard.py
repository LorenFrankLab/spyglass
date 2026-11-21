"""A 'specific' reference that is also a sort-group channel is rejected.

The write path removes the reference after subtracting it, which would drop
a channel the user meant to sort. No DB.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._core.reference_resolution import (
    assert_reference_not_member,
)


def test_specific_reference_in_group_raises():
    with pytest.raises(ValueError, match="also a sort-group channel"):
        assert_reference_not_member(
            reference_mode="specific",
            reference_electrode_id=2,
            sort_group_channel_ids=[0, 1, 2, 3],
        )


def test_specific_reference_outside_group_ok():
    # No raise: a reference outside the sort group is the valid case.
    assert_reference_not_member(
        reference_mode="specific",
        reference_electrode_id=99,
        sort_group_channel_ids=[0, 1, 2, 3],
    )


@pytest.mark.parametrize("mode", ["none", "global_median"])
def test_non_specific_modes_ignore_membership(mode):
    # Only 'specific' removes a named channel.
    assert_reference_not_member(
        reference_mode=mode,
        reference_electrode_id=None,
        sort_group_channel_ids=[0, 1, 2, 3],
    )
