"""Checks of the span tables in concatenated and motion-corrected files."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import h5py
import numpy as np

STATISTICS = "spyglass_v2_statistics_spans"
CONTINUITY = "spyglass_v2_continuity_spans"
MEMBERS = "spyglass_v2_concat_member_spans"

# Runs in a fresh interpreter, so the file is read without Spyglass.
_PLAIN_READER = """
import json, sys
import pynwb
with pynwb.NWBHDF5IO(sys.argv[1], "r", load_namespaces=True) as io:
    intervals = io.read().intervals
    out = {
        name: {
            "description": intervals[name].description,
            "columns": intervals[name].to_dataframe().to_dict(orient="list"),
        }
        for name in intervals
        if name.startswith("spyglass_v2_")
    }
assert not any(m.startswith("spyglass") for m in sys.modules)
print(json.dumps(out))
"""


def read_span_tables_without_spyglass(abs_path) -> dict:
    """``{name: {"description", "columns"}}`` read by plain pynwb."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PLAIN_READER, str(abs_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def assert_span_tables_match(
    tables: dict,
    *,
    times,
    statistics_spans,
    continuity_spans,
    continuity_start_s,
    continuity_end_s,
    members=None,
) -> None:
    """Compare the file's span tables with the row's columns.

    ``times`` is the series' timestamp vector; ``members`` is a list of
    ``(member_index, nwb_file_name, start_sample, end_sample)``.
    """
    times = np.asarray(times)
    expected_names = {STATISTICS, CONTINUITY} | (
        {MEMBERS} if members else set()
    )
    assert set(tables) == expected_names
    spans_by_name = {
        STATISTICS: np.asarray(statistics_spans).reshape(-1, 2),
        CONTINUITY: np.asarray(continuity_spans).reshape(-1, 2),
    }
    if members:
        spans_by_name[MEMBERS] = np.asarray([m[2:] for m in members])
    for name, spans in spans_by_name.items():
        columns = tables[name]["columns"]
        assert "start_sample" in tables[name]["description"]
        np.testing.assert_array_equal(columns["start_sample"], spans[:, 0])
        np.testing.assert_array_equal(columns["end_sample"], spans[:, 1])
        np.testing.assert_array_equal(columns["start_time"], times[spans[:, 0]])
        np.testing.assert_array_equal(
            columns["stop_time"], times[spans[:, 1] - 1]
        )
    continuity = tables[CONTINUITY]["columns"]
    np.testing.assert_array_equal(
        continuity["acquisition_start_time"], continuity_start_s
    )
    np.testing.assert_array_equal(
        continuity["acquisition_stop_time"], continuity_end_s
    )
    if members:
        columns = tables[MEMBERS]["columns"]
        assert columns["member_index"] == [m[0] for m in members]
        assert columns["nwb_file_name"] == [m[1] for m in members]


def write_copy_without_span_tables(source, destination) -> None:
    """Copy a trace file and delete its span tables, as an older file."""
    shutil.copyfile(source, destination)
    with h5py.File(destination, "a") as handle:
        for name in (STATISTICS, CONTINUITY, MEMBERS):
            if f"intervals/{name}" in handle:
                del handle[f"intervals/{name}"]
