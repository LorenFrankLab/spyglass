"""Tests for the UnitMatch matcher backend.

The backend is the concrete MatcherProtocol implementation. Registration and
the degenerate single-session case are hermetic; the end-to-end match is an
integration test that builds two bundles from the committed polymer fixture
(pseudo-session split of the same ground-truth neurons) and checks that planted
correspondences score above random pairs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)


def test_backend_registers_unitmatch():
    import spyglass.spikesorting.v2._unitmatch_backend  # noqa: F401 (registers)
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2.matcher_protocol import (
        _get_matcher_schema,
        get_matcher,
    )

    assert get_matcher("unitmatch").name == "unitmatch"
    assert _get_matcher_schema("unitmatch") is UnitMatchParamsSchema


def test_match_single_session_returns_empty():
    from spyglass.spikesorting.v2._unitmatch_backend import UnitMatchBackend
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    one = SessionMatcherInput(
        session_key={"sorting_id": "s", "curation_id": 0},
        waveform_dir=Path("/tmp/does-not-matter"),
        channel_positions_path=Path("/tmp/does-not-matter/cp.npy"),
    )
    assert UnitMatchBackend().match([one], {}) == []


def _read_polymer_gt():
    """Return (recording, gt_spike_trains_seconds) from the polymer fixture."""
    import spikeinterface.preprocessing as spre
    from pynwb import NWBHDF5IO
    from spikeinterface.extractors import read_nwb_recording

    rec = read_nwb_recording(
        str(FIXTURE), electrical_series_path="acquisition/e-series"
    )
    rec = spre.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)
    rec = rec.set_probe(rec.get_probe().to_2d())
    with NWBHDF5IO(str(FIXTURE), "r") as io:
        gt = io.read().processing["ground_truth"].data_interfaces["units"]
        trains = {
            int(u): np.asarray(gt["spike_times"][i])
            for i, u in enumerate(gt.id[:])
        }
    return rec, trains


@pytest.fixture(scope="module")
def two_session_inputs(tmp_path_factory):
    """Build two bundles with a known overlapping unit set (8 shared)."""
    from spikeinterface.core import NumpySorting

    from spyglass.spikesorting.v2._unitmatch_backend import (
        extract_unitmatch_bundle,
    )
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    if not FIXTURE.exists():
        pytest.skip("polymer 60s fixture not present")

    rec, trains = _read_polymer_gt()
    fs = rec.get_sampling_frequency()
    n = rec.get_num_samples()
    mid = n // 2
    s1_ids, s2_ids = list(range(0, 16)), list(range(8, 24))
    shared = sorted(set(s1_ids) & set(s2_ids))

    def sorting_for(ids):
        labels = np.concatenate([np.full(len(trains[u]), u) for u in ids])
        times = np.concatenate([trains[u] for u in ids])
        order = np.argsort(times)
        return NumpySorting.from_times_and_labels(
            times[order], labels[order], sampling_frequency=fs
        )

    root = tmp_path_factory.mktemp("um_bundles")
    inputs = []
    for name, ids, (a, b) in [
        ("S1", s1_ids, (0, mid)),
        ("S2", s2_ids, (mid, n)),
    ]:
        sdir = root / name
        extract_unitmatch_bundle(
            sdir,
            rec.frame_slice(a, b),
            sorting_for(ids).frame_slice(a, b),
        )
        inputs.append(
            SessionMatcherInput(
                session_key={"sorting_id": name, "curation_id": 0},
                waveform_dir=sdir,
                channel_positions_path=sdir / "channel_positions.npy",
            )
        )
    return inputs, s1_ids, s2_ids, shared


def test_match_recovers_planted_correspondences(two_session_inputs):
    from spyglass.spikesorting.v2._unitmatch_backend import UnitMatchBackend

    inputs, _s1_ids, _s2_ids, shared = two_session_inputs
    pairs = UnitMatchBackend().match(inputs, {"match_threshold": 0.5})

    # every pair spans the two sessions and carries both side keys
    assert pairs, "expected non-empty matches on planted correspondences"
    for p in pairs:
        assert p.session_a_sorting_id != p.session_b_sorting_id
        assert 0.0 <= p.match_probability <= 1.0

    # the planted (shared GT unit) pairs should be recovered as high-prob matches
    # shared GT units keep the same unit_id in both sessions, so a planted
    # correspondence surfaces as the (u, u) pair.
    found = {
        (p.unit_a_id, p.unit_b_id)
        for p in pairs
        if p.match_probability > 0.5
    }
    recovered = sum(1 for u in shared if (u, u) in found)
    assert recovered >= len(shared) - 1, (
        f"recovered only {recovered}/{len(shared)} planted correspondences"
    )
