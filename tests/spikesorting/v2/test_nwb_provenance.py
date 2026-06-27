"""DB-free unit tests for the NWB provenance scratch helper.

These exercise the pure (de)serialization in
``spyglass.spikesorting.v2._nwb_provenance`` -- the ``key``/``value_json``
scalar provenance container and the typed long-table container -- with no
DataJoint server and no SpikeInterface analyzer. The helper writes into and
reads back from a real (tiny) NWB file on disk, so the round-trip assertions
prove the values survive an HDF5 write, not just an in-memory build.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.unit


def _new_nwbfile():
    """A minimal, valid in-memory NWBFile (tz-aware start time)."""
    import pynwb

    return pynwb.NWBFile(
        session_description="provenance helper test",
        identifier="provenance-test",
        session_start_time=datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")),
    )


def _write(nwbfile, tmp_path):
    import pynwb

    path = str(tmp_path / "provenance.nwb")
    with pynwb.NWBHDF5IO(path=path, mode="w") as io:
        io.write(nwbfile)
    return path


def test_scalar_provenance_round_trips_mixed_types(tmp_path):
    """A scalar bundle (str/int/float/bool/None/dict/list) survives HDF5."""
    from spyglass.spikesorting.v2._nwb_provenance import (
        PROVENANCE_SCHEMA_VERSION,
        add_provenance_table,
        read_provenance_values,
    )

    values = {
        "recording_id": "abc-123",
        "sort_group_id": 7,
        "threshold_uv": 0.1,
        "merges_applied": True,
        "sorter_version": None,  # nullable provenance field
        "sorter_params": {"detect_sign": -1, "freq_min": 300.0},  # nested blob
        "members": [0, 1, 2],  # list blob
    }

    nwbfile = _new_nwbfile()
    add_provenance_table(nwbfile, "spyglass_v2_test_provenance", values)
    path = _write(nwbfile, tmp_path)

    read = read_provenance_values(path, "spyglass_v2_test_provenance")

    for key, expected in values.items():
        assert read[key] == expected, key
    # Every provenance container records its schema version.
    assert read["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION


def test_scalar_provenance_coerces_datajoint_types(tmp_path):
    """Numpy + UUID (DataJoint-deserialized types) serialize + read back.

    Provenance bundles re-emit values fetched from DataJoint (metric kwargs,
    rule thresholds, hash manifests, row ids), which deserialize as numpy
    scalars/arrays and ``uuid.UUID``; plain ``json.dumps`` cannot encode those,
    so the helper coerces them (UUIDs to their canonical string form).
    """
    import uuid

    import numpy as np

    from spyglass.spikesorting.v2._nwb_provenance import (
        add_provenance_table,
        read_provenance_values,
    )

    rec_id = uuid.uuid4()
    values = {
        "threshold": np.float64(1.5),
        "count": np.int64(7),
        "flags": np.array([1, 2, 3]),
        "ok": np.bool_(True),
        "recording_id": rec_id,
    }

    nwbfile = _new_nwbfile()
    add_provenance_table(nwbfile, "spyglass_v2_test_dj_types", values)
    path = _write(nwbfile, tmp_path)

    read = read_provenance_values(path, "spyglass_v2_test_dj_types")
    assert read["threshold"] == 1.5
    assert read["count"] == 7
    assert read["flags"] == [1, 2, 3]
    assert read["ok"] is True
    assert read["recording_id"] == str(rec_id)


def test_long_provenance_table_round_trips_typed_rows(tmp_path):
    """A typed long table (one row per member) survives HDF5 with native types."""
    from spyglass.spikesorting.v2._nwb_provenance import (
        PROVENANCE_SCHEMA_VERSION,
        add_long_provenance_table,
        read_long_provenance,
    )

    rows = [
        {"member_index": 0, "recording_id": "rec-a", "end_sample": 1000},
        {"member_index": 1, "recording_id": "rec-b", "end_sample": 2500},
    ]
    columns = [
        ("member_index", int),
        ("recording_id", str),
        ("end_sample", int),
    ]

    nwbfile = _new_nwbfile()
    add_long_provenance_table(
        nwbfile, "spyglass_v2_test_members", rows, columns
    )
    path = _write(nwbfile, tmp_path)

    read = read_long_provenance(path, "spyglass_v2_test_members")

    assert len(read) == 2
    for got, expected in zip(read, rows):
        for col, _dtype in columns:
            assert got[col] == expected[col], col
            # Native Python types, not numpy scalars, for downstream use.
            assert type(got[col]) is type(expected[col]), col
        assert got["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION


def test_long_provenance_table_handles_empty_rows(tmp_path):
    """A zero-row long table still writes and reads back as an empty list."""
    from spyglass.spikesorting.v2._nwb_provenance import (
        add_long_provenance_table,
        read_long_provenance,
    )

    columns = [("member_index", int), ("recording_id", str)]

    nwbfile = _new_nwbfile()
    add_long_provenance_table(
        nwbfile, "spyglass_v2_test_empty", [], columns
    )
    path = _write(nwbfile, tmp_path)

    read = read_long_provenance(path, "spyglass_v2_test_empty")

    assert read == []
