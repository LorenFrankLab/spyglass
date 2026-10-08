"""Acceptance validates caller unit ids before creating merge decisions."""

import numpy as np
import pytest

from spyglass.spikesorting.v2._curation.evaluation_acceptance import (
    resolve_accepted_merges,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "unit_id", [1.5, 1.0, True, False, np.bool_(True), "1"]
)
def test_merge_acceptance_rejects_non_integer_unit_ids(unit_id):
    with pytest.raises(
        ValueError, match="merge group unit_id must be an integer"
    ):
        resolve_accepted_merges(None, {}, [[unit_id, 2]], False)


def test_merge_acceptance_preserves_valid_unit_ids_and_group_validation():
    assert resolve_accepted_merges(None, {}, [[np.int64(1), 2]], False) == [
        [1, 2]
    ]
    # Empty/singleton groups still reach the curation planner's size guard.
    assert resolve_accepted_merges(None, {}, [[], [1]], False) == [[], [1]]


def test_merge_acceptance_filters_persisted_suggestions_only():
    class Evaluation:
        def get_suggested_merge_groups(self, _key):
            return [[], [1], [np.int64(1), np.int64(2)]]

    assert resolve_accepted_merges(Evaluation(), {}, None, True) == [[1, 2]]
