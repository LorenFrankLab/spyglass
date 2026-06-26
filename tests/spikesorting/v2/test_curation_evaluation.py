"""Tests for committed-curation evaluation and final merged-unit metrics.

``CurationEvaluation`` scores an existing committed ``CurationV2`` row in that
curation's own unit namespace: merged units get metrics recomputed over their
merged spike trains/templates, not inherited from a contributor. Preview
(draft) curations -- ``apply_merge=False`` with a real proposed merge group --
are rejected at the evaluation boundary.

Tiers:

- ``slow`` / ``integration`` end-to-end tests populate ``CurationEvaluation``
  on the planted two-unit sort and the shared MEArec smoke sort.
- hermetic compute-level tests pin the merged-template metric contract with
  controlled planted templates (no DB populate).
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------- committed-curation predicate ------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_is_committed_curation_distinguishes_preview(planted_two_unit_sort):
    """Root, label-only, and applied-merge rows are committed; a preview is not.

    A preview (``apply_merge=False`` with a real >=2-unit proposed merge group)
    is a draft, not a final downstream curation; ``assert_committed_curation``
    raises for it and is a no-op for the committed states.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    assert len(unit_ids) >= 2, "need >=2 units to plant a merge"

    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        assert CurationV2.is_committed_curation(root) is True
        CurationV2.assert_committed_curation(root)  # no raise

        label_only = CurationV2.insert_curation(
            sorting_key,
            parent_curation_id=root["curation_id"],
            labels={unit_ids[0]: ["noise"]},
        )
        assert CurationV2.is_committed_curation(label_only) is True
        CurationV2.assert_committed_curation(label_only)

        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        assert CurationV2.is_committed_curation(merged) is True
        CurationV2.assert_committed_curation(merged)

        preview = CurationV2.propose_merge_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        assert CurationV2.is_committed_curation(preview) is False
        with pytest.raises(ValueError, match="preview"):
            CurationV2.assert_committed_curation(
                preview, context="CurationEvaluation"
            )
    finally:
        clear_curations_for(planted_two_unit_sort)
