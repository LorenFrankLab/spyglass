"""``electrode_table_region`` maps electrode ids to ROW INDICES.

``pynwb``'s ``create_electrode_table_region(region=...)`` interprets ``region``
as row indices into the electrodes table, not electrode ids. The recording
writer previously passed ``region=[electrode_ids]`` directly, which is correct
only when ``electrode.id == row position``. For an electrodes table whose ids
are non-contiguous / reordered (allowed by NWB; common for external/DANDI
files) that silently points the ElectricalSeries at the WRONG electrodes ->
wrong channel locations and brain-region attribution on readback.

Hermetic -- builds an in-memory NWBFile, no DB.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _nwb_with_electrode_ids(ids):
    import pynwb

    nwbfile = pynwb.NWBFile(
        session_description="t",
        identifier="t",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="probe")
    group = nwbfile.create_electrode_group(
        name="grp", description="g", location="hpc", device=device
    )
    for i, eid in enumerate(ids):
        nwbfile.add_electrode(
            id=int(eid), location="hpc", group=group, x=float(i), y=0.0, z=0.0
        )
    return nwbfile


def test_region_maps_ids_to_row_indices_not_raw_ids():
    """Non-contiguous ids [10,11,12,13]; requesting ids [12,11] must yield row
    indices [2,1] that resolve back to ids [12,11]."""
    from spyglass.spikesorting.v2.utils import electrode_table_region

    nwbfile = _nwb_with_electrode_ids([10, 11, 12, 13])
    requested = [12, 11]

    region = electrode_table_region(nwbfile, requested, "sort group")

    # The region stores ROW INDICES (2, 1), not the raw ids (12, 11) --
    # passing the ids directly would index out of a 4-row table.
    assert list(region.data) == [2, 1]
    resolved = [int(region.table.id[idx]) for idx in region.data]
    assert resolved == requested


def test_contiguous_ids_unchanged():
    """When id == row index, the mapping is the identity (no regression on
    the common Frank-lab case)."""
    from spyglass.spikesorting.v2.utils import electrode_table_region

    nwbfile = _nwb_with_electrode_ids([0, 1, 2, 3])
    region = electrode_table_region(nwbfile, [2, 0], "sort group")
    assert list(region.data) == [2, 0]
    resolved = [int(region.table.id[idx]) for idx in region.data]
    assert resolved == [2, 0]


def test_missing_electrode_id_raises():
    """An id absent from the electrodes table fails loud, not silently."""
    from spyglass.spikesorting.v2.utils import electrode_table_region

    nwbfile = _nwb_with_electrode_ids([10, 11, 12, 13])
    with pytest.raises(ValueError, match="not in"):
        electrode_table_region(nwbfile, [99], "sort group")
