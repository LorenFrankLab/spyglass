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
MOTION_HEADER = "spyglass_v2_motion_correction_provenance"
MOTION_CONTINUITY = "spyglass_v2_motion_continuity_spans"

# Runs in a fresh interpreter, so the file is read without Spyglass.
_PLAIN_READER = """
import json, sys
import pynwb
with pynwb.NWBHDF5IO(sys.argv[1], "r", load_namespaces=True) as io:
    nwbfile = io.read()
    intervals = nwbfile.intervals
    out = {
        name: {
            "description": intervals[name].description,
            "columns": intervals[name].to_dataframe().to_dict(orient="list"),
        }
        for name in intervals
        if name.startswith("spyglass_v2_")
    }
    out["scratch"] = sorted(nwbfile.scratch)
assert not any(m.startswith("spyglass") for m in sys.modules)
print(json.dumps(out, default=lambda value: value.tolist()))
"""


def read_span_tables_without_spyglass(abs_path) -> dict:
    """``{name: {"description", "columns"}, "scratch": [...]}`` by plain pynwb."""
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
    estimation_start_s=None,
    members=None,
) -> None:
    """Compare the file's span tables with the row's columns.

    ``times`` is the series' timestamp vector; ``members`` is a list of
    ``(member_index, nwb_file_name, start_sample, end_sample, valid_times)``.
    """
    times = np.asarray(times)
    tables = {k: v for k, v in tables.items() if k != "scratch"}
    expected_names = {STATISTICS, CONTINUITY} | (
        {MEMBERS} if members else set()
    )
    assert set(tables) == expected_names
    spans_by_name = {
        STATISTICS: np.asarray(statistics_spans).reshape(-1, 2),
        CONTINUITY: np.asarray(continuity_spans).reshape(-1, 2),
    }
    if members:
        spans_by_name[MEMBERS] = np.asarray([m[2:4] for m in members])
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
    if estimation_start_s is None:
        assert "estimation_start_time" not in continuity
    else:
        assert "estimation_start_time" in tables[CONTINUITY]["description"]
        np.testing.assert_array_equal(
            continuity["estimation_start_time"], estimation_start_s
        )
    if members:
        columns = tables[MEMBERS]["columns"]
        assert "valid_times" in tables[MEMBERS]["description"]
        assert columns["member_index"] == [m[0] for m in members]
        assert columns["nwb_file_name"] == [m[1] for m in members]
        for stored, member in zip(columns["valid_times"], members, strict=True):
            np.testing.assert_array_equal(
                np.asarray(stored).reshape(-1, 2), member[4]
            )


def _delete_tables(path, names) -> None:
    with h5py.File(path, "a") as handle:
        for name in names:
            if f"intervals/{name}" in handle:
                del handle[f"intervals/{name}"]


def write_copy_without_span_tables(source, destination) -> None:
    """Copy a trace file and delete its span tables."""
    shutil.copyfile(source, destination)
    _delete_tables(destination, (STATISTICS, CONTINUITY, MEMBERS))


def write_concat_copy_without_valid_times(source, destination) -> None:
    """Copy a concat file and drop the member table's ``valid_times``."""
    shutil.copyfile(source, destination)
    with h5py.File(destination, "a") as handle:
        group = handle[f"intervals/{MEMBERS}"]
        del group["valid_times"], group["valid_times_index"]
        group.attrs["colnames"] = [
            name
            for name in group.attrs["colnames"]
            if name not in ("valid_times", "valid_times_index")
        ]


def add_motion_scratch_spans(
    path,
    *,
    statistics_spans,
    continuity_spans,
    continuity_start_s,
    continuity_end_s,
    estimation_start_s,
) -> None:
    """Add the scratch span copies a motion-corrected file once carried.

    The header gains ``statistics_spans``; ``MOTION_CONTINUITY`` holds each
    continuity span's frames, source-clock times and estimation-clock start.
    """
    import pynwb
    from hdmf.common import DynamicTable, VectorData

    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        header = io.read().scratch[MOTION_HEADER].to_dataframe()
    values = dict(zip(header["key"], header["value_json"], strict=True))
    values["statistics_spans"] = json.dumps(
        np.asarray(statistics_spans).tolist()
    )
    with h5py.File(path, "a") as handle:
        del handle[f"scratch/{MOTION_HEADER}"]
    spans = np.asarray(continuity_spans, dtype=np.int64).reshape(-1, 2)
    columns = {
        "span_index": np.arange(len(spans), dtype=np.int64),
        "start_sample": spans[:, 0],
        "end_sample": spans[:, 1],
        "source_start_s": np.asarray(continuity_start_s, dtype=np.float64),
        "source_end_s": np.asarray(continuity_end_s, dtype=np.float64),
        "estimation_start_s": np.asarray(estimation_start_s, dtype=np.float64),
        "provenance_schema_version": np.ones(len(spans), dtype=np.int64),
    }
    keys = sorted(values)
    with pynwb.NWBHDF5IO(path, "a", load_namespaces=True) as io:
        nwbfile = io.read()
        nwbfile.add_scratch(
            DynamicTable(
                name=MOTION_HEADER,
                description="old-style header",
                columns=[
                    VectorData(
                        name="key", description="k", data=np.array(keys, object)
                    ),
                    VectorData(
                        name="value_json",
                        description="v",
                        data=np.array([values[k] for k in keys], object),
                    ),
                ],
            )
        )
        nwbfile.add_scratch(
            DynamicTable(
                name=MOTION_CONTINUITY,
                description="old-style continuity spans",
                columns=[
                    VectorData(name=name, description=name, data=data)
                    for name, data in columns.items()
                ],
            )
        )
        io.write(nwbfile)
