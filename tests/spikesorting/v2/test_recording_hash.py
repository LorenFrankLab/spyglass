"""Determinism check for the modern recording-cache hash helper.

``_hash_nwb_recording`` wraps ``NwbfileHasher``; missing-artifact detection
in the modern pipeline relies on the digest being stable across calls.
Content-sensitivity is intentionally not asserted: ``NwbfileHasher``
currently folds in object names and attrs but discards dataset values
(LorenFrankLab/spyglass#1597); the content-change regression test belongs
to that fix PR.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_hash_nwb_recording_stable(analysis_nwbfile_for_hash):
    """Two calls on the same AnalysisNwbfile return identical digests."""
    from spyglass.spikesorting.v2.utils import _hash_nwb_recording

    first = _hash_nwb_recording(analysis_nwbfile_for_hash)
    second = _hash_nwb_recording(analysis_nwbfile_for_hash)

    assert isinstance(first, str)
    assert len(first) > 0
    assert first == second
