"""Determinism check for the modern recording-cache hash helper.

``_hash_nwb_recording`` delegates to Spyglass's ``NwbfileHasher`` so the v2
recording-cache verification path uses the same hash as v1 recompute. This
test confirms repeated calls on the same ``AnalysisNwbfile`` return the same
digest, which is what the missing-artifact detection in the modern pipeline
relies on.

Database-tier (slow): requires the Docker test database and the sample NWB.
The ``analysis_nwbfile_for_hash`` fixture lives in the v2 conftest.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_hash_nwb_recording_stable(analysis_nwbfile_for_hash):
    """Two calls on the same AnalysisNwbfile return identical digests.

    Content-sensitivity is intentionally not asserted here: ``NwbfileHasher``
    currently folds in object names and attrs but discards dataset values
    (see the Phase 0a hashing decisions note in
    ``.claude/docs/plans/spikesorting-v2/SCRATCHPAD.md``). The fix for that
    bug lands in its own PR off master and will own the content-change
    regression test.
    """
    from spyglass.spikesorting.v2.utils import _hash_nwb_recording

    first = _hash_nwb_recording(analysis_nwbfile_for_hash)
    second = _hash_nwb_recording(analysis_nwbfile_for_hash)

    assert isinstance(first, str)
    assert len(first) > 0
    assert first == second
