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
    """Two calls on the same AnalysisNwbfile return identical digests."""
    from spyglass.spikesorting.v2.utils import _hash_nwb_recording

    first = _hash_nwb_recording(analysis_nwbfile_for_hash)
    second = _hash_nwb_recording(analysis_nwbfile_for_hash)

    assert isinstance(first, str)
    assert len(first) > 0
    assert first == second


@pytest.mark.slow
def test_hash_nwb_recording_distinguishes_files(
    analysis_nwbfile_for_hash, dj_conn
):
    """Different AnalysisNwbfiles get different digests (sanity)."""
    from spyglass.common import Nwbfile
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    from spyglass.spikesorting.v2.utils import _hash_nwb_recording

    parent = (AnalysisNwbfile & {"analysis_file_name": analysis_nwbfile_for_hash}).fetch1(
        "nwb_file_name"
    )
    assert (Nwbfile & {"nwb_file_name": parent})
    other_name = AnalysisNwbfile().create(parent)
    AnalysisNwbfile().add(parent, other_name)

    first = _hash_nwb_recording(analysis_nwbfile_for_hash)
    other = _hash_nwb_recording(other_name)
    assert first != other, (
        "Two distinct AnalysisNwbfiles produced the same NwbfileHasher digest; "
        "the hash is not actually content-sensitive."
    )
