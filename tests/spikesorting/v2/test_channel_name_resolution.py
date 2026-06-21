"""channel_name resolution on a real-NWB-shape fixture.

``Recording._spikeinterface_channel_ids`` resolves channel ids from the raw
NWB ``channel_name`` column when present and falls back to the integer
``electrode_id`` when absent. This exercises both branches on a fixture built
to match production Frank-lab NWB shape.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "channel_names", [None, ["ch_a", "ch_b", "ch_c", "ch_d"]]
)
def test_channel_name_resolution_path_real_nwb(
    dj_conn, tmp_path, monkeypatch, channel_names
):
    """``_spikeinterface_channel_ids`` resolves channel ids from the raw
    NWB ``channel_name`` column when present, and falls back to integer
    ``electrode_id`` when absent.

    The MEArec fixtures omit ``channel_name`` so only the integer-fallback
    branch was exercised; production Frank-lab NWBs carry the column. This
    test builds a 4-contact NWB via the fixture builder's ``channel_names``
    parameter (injecting the column) and via the default (no column), then
    asserts the resolved SpikeInterface channel ids match each branch's
    expected mapping.
    """
    from datetime import datetime, timezone

    import pynwb

    from spyglass.common import Nwbfile
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        _add_probe_and_electrodes,
        tetrode_probe_layout,
    )
    from spyglass.spikesorting.v2.recording import Recording

    nwbfile = pynwb.NWBFile(
        session_description="channel_name resolution fixture",
        identifier="a19-chan-name",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    _add_probe_and_electrodes(
        nwbfile,
        tetrode_probe_layout(),
        targeted_location="hpc",
        channel_names=channel_names,
    )
    out = tmp_path / "a19_channel_name_fixture.nwb"
    with pynwb.NWBHDF5IO(str(out), mode="w") as io:
        io.write(nwbfile)

    # _spikeinterface_channel_ids resolves the raw path via Nwbfile; redirect
    # it to our standalone fixture (no ingestion needed for the lookup).
    monkeypatch.setattr(
        Nwbfile, "get_abs_path", staticmethod(lambda *a, **k: str(out))
    )

    spyglass_ids = [0, 1, 2, 3]
    resolved = Recording._spikeinterface_channel_ids(
        "a19_channel_name_fixture.nwb", spyglass_ids
    )

    if channel_names is None:
        assert resolved == [0, 1, 2, 3], (
            "integer-fallback branch must return int electrode_ids; got "
            f"{resolved!r}"
        )
        assert all(isinstance(c, int) for c in resolved)
    else:
        assert resolved == channel_names, (
            "channel_name branch must resolve to the injected string names "
            f"in electrode order; got {resolved!r}"
        )
