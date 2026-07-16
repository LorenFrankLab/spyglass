"""Tetrode-geometry gate negative cases.

``Recording._maybe_apply_tetrode_geometry`` only rewrites a sort group's probe
to the 12.5 µm tetrode square when the group is unambiguously a single
``tetrode_12.5`` with four channels in one electrode group. These pin that the
gate no-ops (and logs the failing condition) for every disqualifying setup.
"""

from __future__ import annotations

import numpy as np


def _assert_tetrode_gate_noop(
    caplog, probe_types, electrode_group_names, channel_ids, reason_substr
):
    """Call ``_maybe_apply_tetrode_geometry`` with a failing-gate setup and
    assert (a) the recording geometry is untouched and (b) an INFO log names
    the condition that failed.

    The synthetic recording carries SI's default linear probe (contacts at
    y=0,20,40,...); the tetrode patch would replace it with a 12.5 µm square,
    so an unchanged ``get_channel_locations()`` proves the gate no-opped.
    """
    import logging

    import spikeinterface as si

    from spyglass.spikesorting.v2.recording import Recording

    rec = si.generate_recording(
        num_channels=len(channel_ids),
        durations=[1.0],
        sampling_frequency=30_000.0,
    )
    before = rec.get_channel_locations().copy()
    with caplog.at_level(logging.INFO):
        result = Recording._maybe_apply_tetrode_geometry(
            rec, probe_types, electrode_group_names, channel_ids
        )
    after = result.get_channel_locations()
    assert np.array_equal(before, after), (
        "tetrode geometry was applied despite a failed gate condition; the "
        "channel locations changed."
    )
    skip_msgs = [
        r.getMessage()
        for r in caplog.records
        if "_maybe_apply_tetrode_geometry skipped" in r.getMessage()
    ]
    assert any(reason_substr in m for m in skip_msgs), (
        f"expected an INFO log naming the failed condition "
        f"({reason_substr!r}); got skip logs {skip_msgs}."
    )


def test_tetrode_geometry_gate_three_channel(dj_conn, caplog):
    """A 3-channel sort group (e.g. after a bad-channel drop) is not a
    tetrode; the gate no-ops and logs the channel-count condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=("tetrode_12.5",) * 3,
        electrode_group_names=("g",) * 3,
        channel_ids=[0, 1, 2],
        reason_substr="4 channel",
    )


def test_tetrode_geometry_gate_mixed_probe(dj_conn, caplog):
    """A sort group spanning two probe types is not a single tetrode;
    the gate no-ops and logs the multiple-probe-types condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=(
            "tetrode_12.5",
            "tetrode_12.5",
            "other_probe",
            "other_probe",
        ),
        electrode_group_names=("g",) * 4,
        channel_ids=[0, 1, 2, 3],
        reason_substr="multiple probe type",
    )


def test_tetrode_geometry_gate_renamed_probe(dj_conn, caplog):
    """A single-probe group whose probe string is not exactly
    ``tetrode_12.5`` (e.g. a renamed ``tetrode_12.5_v2``) is not patched; the
    gate no-ops and logs the probe-type-mismatch condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=("tetrode_12.5_v2",) * 4,
        electrode_group_names=("g",) * 4,
        channel_ids=[0, 1, 2, 3],
        reason_substr="not tetrode_12.5",
    )


def test_tetrode_geometry_gate_multi_group(dj_conn, caplog):
    """Four channels split across two electrode groups is not a single
    tetrode; the gate no-ops and logs the multiple-groups condition."""
    _assert_tetrode_gate_noop(
        caplog,
        probe_types=("tetrode_12.5",) * 4,
        electrode_group_names=("g1", "g1", "g2", "g2"),
        channel_ids=[0, 1, 2, 3],
        reason_substr="multiple electrode group",
    )
