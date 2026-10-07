"""Temporal preprocessing respects actual acquisition gaps and lazy clocks."""

import numpy as np
import pytest
import spikeinterface as si

from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._recording_preprocessing import (
    apply_temporal_preprocessing,
)
from spyglass.spikesorting.v2._recording_restriction import (
    restrict_recording_times,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "phase_shift,bandpass", [(False, True), (True, False), (True, True)]
)
def test_temporal_margins_do_not_cross_acquisition_gaps(phase_shift, bandpass):
    fs, n = 30_000.0, 3000
    traces = np.zeros((2 * n, 1), dtype=np.float64)
    traces[n - 1, 0] = -1000.0
    times = np.r_[np.arange(n) / fs, 5.0 + np.arange(n) / fs]
    recording = si.NumpyRecording([traces], fs)
    recording.set_times(times)
    recording.set_property("inter_sample_shift", np.array([0.25]))
    params = PreprocessingParamsSchema(
        phase_shift={} if phase_shift else None,
        bandpass_filter=(
            {"freq_min": 600.0, "freq_max": 6000.0} if bandpass else None
        ),
        min_segment_length=0.0,
    )
    filtered, _ = apply_temporal_preprocessing(recording, params)
    retained, clock, _ = restrict_recording_times(
        filtered, [[times[0], times[-1]]]
    )
    actual = retained.get_traces()
    # Filtering the independently acquired blocks is the scientific reference.
    expected = []
    for block in (traces[:n], traces[n:]):
        independent = si.NumpyRecording([block], fs)
        independent.set_property("inter_sample_shift", np.array([0.25]))
        independent, _ = apply_temporal_preprocessing(independent, params)
        expected.append(independent.get_traces())
    np.testing.assert_allclose(actual, np.concatenate(expected))
    np.testing.assert_array_equal(actual[n:], 0.0)
    np.testing.assert_array_equal(clock[:], times)

    # A user-selected edge within an acquisition span retains the full span's
    # filter context; it must not become a new filter boundary.
    selected, selected_clock, _ = restrict_recording_times(
        filtered, [[times[n - 20], times[n - 1]]]
    )
    # Phase-shift uses an FFT whose size changes with the requested chunk.
    # Allow sub-nanovolt differences while rejecting boundary transients.
    np.testing.assert_allclose(
        selected.get_traces(), expected[0][-20:], atol=1e-3
    )
    np.testing.assert_array_equal(selected_clock[:], times[n - 20 : n])


def test_acquisition_span_clock_remains_lazy_when_reconstructed():
    from spyglass.spikesorting.v2._acquisition_spans import (
        AcquisitionSpanRecording,
    )
    from spyglass.spikesorting.v2._signal_math import frames_for_times

    fs, n = 30_000.0, 90_000
    recording = si.NumpyRecording([np.zeros((n, 1), dtype=np.float32)], fs)
    times = np.arange(n) / fs
    times[n // 2 :] += 5.0

    class BoundedClock:
        def __getitem__(self, index):
            selected = times[index]
            assert np.size(selected) <= fs, "eager acquisition clock read"
            return selected

    recording._recording_segments[0].time_vector = BoundedClock()
    # Construction, filtering, binary search and restriction may scan bounded
    # chunks but must never read a whole 1.5-second span at once.
    filtered, _ = apply_temporal_preprocessing(
        recording, PreprocessingParamsSchema()
    )
    assert filtered.get_num_segments() == 2
    segment = filtered.select_segments([1])
    np.testing.assert_array_equal(
        frames_for_times(segment, [times[n // 2 + 12]]), [12]
    )
    retained, clock, _ = restrict_recording_times(
        filtered, [[times[0], times[-1]]]
    )
    np.testing.assert_array_equal(
        retained.get_traces(start_frame=n // 2, end_frame=n // 2 + 10), 0
    )
    np.testing.assert_array_equal(
        clock[n // 2 : n // 2 + 10], times[n // 2 : n // 2 + 10]
    )

    # SI reconstruction uses the span's constructor rather than persisting an
    # eagerly sliced vector in kwargs.
    span = AcquisitionSpanRecording(recording, n // 2, n)
    rebuilt = si.load(span.to_dict())
    np.testing.assert_array_equal(
        frames_for_times(rebuilt, [times[n // 2 + 12]]), [12]
    )


def test_gapped_nwb_preprocessing_streams_exact_timestamps(tmp_path):
    """Use the real lazy NWB reader and streaming writer across a gap."""
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries

    from spyglass.spikesorting.v2._nwb_iterators import (
        SpikeInterfaceRecordingDataChunkIterator,
        TimestampsDataChunkIterator,
    )
    from spyglass.spikesorting.v2._recording_nwb import read_recording_nwb

    fs, n = 30_000.0, 3000
    traces = np.zeros((2 * n, 1), dtype=np.float64)
    traces[n - 1, 0] = -1000.0
    times = np.r_[np.arange(n) / fs, 5.0 + np.arange(n) / fs]

    def write(path, data, timestamps):
        nwb = NWBFile("gap regression", path.stem, datetime.now(timezone.utc))
        device = nwb.create_device(name="probe")
        group = nwb.create_electrode_group(
            "group", "probe", "hippocampus", device
        )
        nwb.add_electrode(group=group, location="hippocampus")
        nwb.add_acquisition(
            ElectricalSeries(
                name="raw",
                data=data,
                timestamps=timestamps,
                electrodes=nwb.create_electrode_table_region([0], "channel"),
                conversion=1e-6,
            )
        )
        with NWBHDF5IO(path, "w") as io:
            io.write(nwb)

    source = tmp_path / "raw.nwb"
    write(source, traces, times)
    recording = read_recording_nwb(
        source, electrical_series_path="acquisition/raw"
    )
    try:
        filtered, _ = apply_temporal_preprocessing(
            recording, PreprocessingParamsSchema()
        )
        retained, clock, _ = restrict_recording_times(
            filtered, [[times[0], times[-1]]]
        )
        output = tmp_path / "filtered.nwb"
        write(
            output,
            SpikeInterfaceRecordingDataChunkIterator(
                retained, buffer_shape=(256, 1), chunk_shape=(128, 1)
            ),
            TimestampsDataChunkIterator(
                clock, buffer_shape=(256,), chunk_shape=(128,)
            ),
        )
        with NWBHDF5IO(output, "r") as io:
            series = io.read().acquisition["raw"]
            np.testing.assert_array_equal(series.timestamps[:], times)
            np.testing.assert_array_equal(series.data[n:], 0.0)
            assert np.max(np.abs(series.data[:n])) > 100.0
    finally:
        recording._nwbfile.get_read_io().close()
