"""The span facts of a trace file, read across every file generation.

Motion-corrected files exist in three generations: scratch copies only, the
scratch copies and the standard tables, and the standard tables only.
Concatenated files: no tables, tables without member valid times, and the
full tables. ``read_trace_file_spans`` must return the same facts for each.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from tests.spikesorting.v2._span_interval_checks import (
    MEMBERS,
    add_motion_scratch_spans,
    read_span_tables_without_spyglass,
    write_concat_copy_without_valid_times,
    write_copy_without_span_tables,
)

pytestmark = pytest.mark.unit

FS = 1000.0
N_SAMPLES = 100
STATISTICS = np.array([[0, 40], [45, 60], [60, 100]])
CONTINUITY = np.array([[0, 60], [60, 100]])
START_S = np.array([10.0, 20.0])
END_S = np.array([10.059, 20.039])
ESTIMATION_START_S = np.array([0.0, 0.06])
VALID_TIMES = [
    np.array([[10.0, 10.039], [10.045, 10.059]]),
    np.array([[20.0, 20.039]]),
]


def _write_trace_file(path, *, motion: bool) -> None:
    import pynwb
    import spikeinterface as si
    from pynwb.ecephys import ElectricalSeries

    from spyglass.spikesorting.v2._storage.provenance import (
        CONCAT_MEMBER_COLUMNS,
        CONCAT_MEMBERS,
        CONCAT_PROVENANCE,
        MOTION_CORRECTION_PROVENANCE,
        build_long_provenance_table,
        build_provenance_table,
    )
    from spyglass.spikesorting.v2._storage.span_intervals import (
        span_interval_tables,
    )

    nwbfile = pynwb.NWBFile(
        "s", "id", datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    )
    device = nwbfile.create_device("device")
    group = nwbfile.create_electrode_group("g", "g", "loc", device)
    for _ in range(2):
        nwbfile.add_electrode(group=group, location="loc")
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="ProcessedElectricalSeries",
            data=np.zeros((N_SAMPLES, 2)),
            electrodes=nwbfile.create_electrode_table_region([0, 1], "e"),
            rate=FS,
        )
    )
    members = [
        {
            "member_index": i,
            "nwb_file_name": f"member{i}_.nwb",
            "interval_list_name": "raw data valid times",
            "start_sample": int(start),
            "end_sample": int(end),
            "valid_times": VALID_TIMES[i],
        }
        for i, (start, end) in enumerate(CONTINUITY)
    ]
    clock = si.NumpyRecording([np.zeros((N_SAMPLES, 2), dtype=np.float32)], FS)
    for table in span_interval_tables(
        clock,
        statistics_spans=STATISTICS,
        continuity_spans=CONTINUITY,
        continuity_start_s=START_S,
        continuity_end_s=END_S,
        estimation_start_s=ESTIMATION_START_S if motion else None,
        concat_members=None if motion else members,
        is_concat=not motion,
    ):
        nwbfile.add_time_intervals(table)
    if motion:
        nwbfile.add_scratch(
            build_provenance_table(
                MOTION_CORRECTION_PROVENANCE, {"removed_channel_ids": []}
            )
        )
    else:
        nwbfile.add_scratch(
            build_provenance_table(
                CONCAT_PROVENANCE, {"artifact_frame_ranges": [[40, 45]]}
            )
        )
        nwbfile.add_scratch(
            build_long_provenance_table(
                CONCAT_MEMBERS,
                [
                    {
                        **m,
                        "recording_id": "r",
                        "artifact_detection_id": "none",
                        "start_sample": 0,
                        "end_sample": m["end_sample"] - m["start_sample"],
                        "concat_start_sample": m["start_sample"],
                        "concat_end_sample": m["end_sample"],
                    }
                    for m in members
                ],
                CONCAT_MEMBER_COLUMNS,
            )
        )
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)


@pytest.mark.parametrize("generation", ["scratch_only", "both", "standard"])
def test_motion_file_spans_read_the_same_in_every_generation(
    tmp_path, generation
):
    from spyglass.spikesorting.v2._storage.span_intervals import (
        read_trace_file_spans,
        span_interval_mismatches,
    )

    standard = tmp_path / "standard.nwb"
    _write_trace_file(standard, motion=True)
    path = tmp_path / f"{generation}.nwb"
    write = {
        "standard": lambda: path.write_bytes(standard.read_bytes()),
        "both": lambda: path.write_bytes(standard.read_bytes()),
        "scratch_only": lambda: write_copy_without_span_tables(standard, path),
    }
    write[generation]()
    if generation != "standard":
        add_motion_scratch_spans(
            path,
            statistics_spans=STATISTICS,
            continuity_spans=CONTINUITY,
            continuity_start_s=START_S,
            continuity_end_s=END_S,
            estimation_start_s=ESTIMATION_START_S,
        )

    spans = read_trace_file_spans(str(path))
    np.testing.assert_array_equal(spans.statistics_spans, STATISTICS)
    np.testing.assert_array_equal(spans.continuity_spans, CONTINUITY)
    np.testing.assert_array_equal(spans.continuity_start_s, START_S)
    np.testing.assert_array_equal(spans.continuity_end_s, END_S)
    np.testing.assert_array_equal(spans.estimation_start_s, ESTIMATION_START_S)
    check = dict(statistics_spans=STATISTICS, continuity_spans=CONTINUITY)
    assert span_interval_mismatches(str(path), **check) == []
    assert span_interval_mismatches(
        str(path), **{**check, "statistics_spans": STATISTICS[1:]}
    ) == ["statistics_spans"]


@pytest.mark.parametrize(
    "generation", ["no_tables", "no_valid_times", "standard"]
)
def test_concat_file_spans_read_the_same_in_every_generation(
    tmp_path, generation
):
    from spyglass.spikesorting.v2._storage.span_intervals import (
        read_trace_file_spans,
        span_interval_mismatches,
    )

    standard = tmp_path / "standard.nwb"
    _write_trace_file(standard, motion=False)
    path = tmp_path / f"{generation}.nwb"
    if generation == "no_tables":
        write_copy_without_span_tables(standard, path)
    elif generation == "no_valid_times":
        write_concat_copy_without_valid_times(standard, path)
    else:
        path.write_bytes(standard.read_bytes())

    spans = read_trace_file_spans(str(path))
    # Observed time is the unmasked samples, merged across the join.
    np.testing.assert_array_equal(spans.observed_spans, [[0, 40], [45, 100]])
    np.testing.assert_array_equal(spans.member_end_samples, [60, 100])
    if generation == "standard":
        for stored, expected in zip(
            spans.member_valid_times, VALID_TIMES, strict=True
        ):
            np.testing.assert_array_equal(stored, expected)
    else:
        assert spans.member_valid_times is None
    if generation == "no_tables":
        assert spans.statistics_spans is None
    check = dict(
        statistics_spans=STATISTICS,
        continuity_spans=CONTINUITY,
        continuity_start_s=START_S,
        continuity_end_s=END_S,
        member_end_samples=[60, 100],
        member_valid_times=VALID_TIMES,
    )
    assert span_interval_mismatches(str(path), **check) == []
    wrong = span_interval_mismatches(
        str(path),
        **{**check, "member_valid_times": [VALID_TIMES[0][:1], VALID_TIMES[1]]},
    )
    assert wrong == (["member_valid_times"] if generation == "standard" else [])


def test_member_valid_times_read_without_spyglass(tmp_path):
    path = tmp_path / "concat.nwb"
    _write_trace_file(path, motion=False)
    tables = read_span_tables_without_spyglass(path)
    for stored, expected in zip(
        tables[MEMBERS]["columns"]["valid_times"], VALID_TIMES, strict=True
    ):
        np.testing.assert_array_equal(stored, expected)
    assert "own clock" in tables[MEMBERS]["description"]
