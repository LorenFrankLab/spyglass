import numpy as np
import pytest

RIPPLE_FILTER = "Ripple 150-250 Hz"


@pytest.fixture(scope="module")
def ripple(common):
    from spyglass.ripple import v1

    return v1


@pytest.fixture(scope="module")
def ripple_band_key(
    common, lfp, lfp_band, lfp_merge_key, lfp_constants, add_interval
):
    sampling_rate = lfp.LFPOutput.merge_get_parent(lfp_merge_key).fetch1(
        "lfp_sampling_rate"
    )
    common.FirFilterParameters().add_filter(
        RIPPLE_FILTER,
        sampling_rate,
        "bandpass",
        [140, 150, 250, 260],
        "ripple filter for 1 kHz data",
    )
    nwb_file_name = lfp_constants["lfp_eg_key"]["nwb_file_name"]
    lfp_band.LFPBandSelection().set_lfp_band_electrodes(
        nwb_file_name=nwb_file_name,
        lfp_merge_id=lfp_merge_key.get("merge_id"),
        electrode_list=lfp_constants.get("lfp_band_electrode_ids"),
        filter_name=RIPPLE_FILTER,
        interval_list_name=add_interval,
        reference_electrode_list=[-1],
        lfp_band_sampling_rate=sampling_rate,
    )
    selection = lfp_band.LFPBandSelection & {
        "nwb_file_name": nwb_file_name,
        "filter_name": RIPPLE_FILTER,
    }
    key = selection.fetch1("KEY")
    lfp_band.LFPBandV1().populate(key)
    yield key

    # Not guarded by `teardown`: the LFP tests fetch1 the file's only
    # LFPBandSelection, so this second one must not outlive the module.
    selection.delete(safemode=False)


@pytest.fixture(scope="module")
def normalization_time_range(lfp_constants):
    """The first half of the detection interval, in absolute time."""
    start, end = lfp_constants["interval_key"]["valid_times"][0]
    yield (float(start), float(start + (end - start) / 2))


@pytest.fixture(scope="module")
def ripple_keys(
    ripple, ripple_band_key, pos_merge_key, normalization_time_range
):
    ripple.RippleLFPSelection.set_lfp_electrodes(
        ripple_band_key, electrode_list=[0], group_name="CA1"
    )
    params = ripple.RippleParameters()
    params.insert_default()
    default = (params & {"ripple_param_name": "default_trodes"}).fetch1(
        "ripple_param_dict"
    )
    legacy = {
        **default,
        "ripple_detection_params": {
            **default["ripple_detection_params"],
            "normalization_time_range": normalization_time_range,
        },
    }
    params.insert1(
        {"ripple_param_name": "legacy_time_range", "ripple_param_dict": legacy},
        skip_duplicates=True,
    )
    selection = (
        ripple.RippleLFPSelection & ripple_band_key & {"group_name": "CA1"}
    ).fetch1("KEY")
    keys = [
        {
            **selection,
            "ripple_param_name": name,
            "pos_merge_id": pos_merge_key["merge_id"],
        }
        for name in ("default_trodes", "legacy_time_range")
    ]
    ripple.RippleTimesV1().populate(keys)
    yield keys


def test_ripple_times_columns(ripple, ripple_keys):
    ripple_times = (ripple.RippleTimesV1 & ripple_keys[0]).fetch1_dataframe()
    for column in (
        "start_time",
        "end_time",
        "n_samples",
        "max_sustained_zscore",
        "peak_time",
        "clipped_start",
        "clipped_end",
    ):
        assert column in ripple_times.columns
    assert "max_thresh" not in ripple_times.columns


def test_legacy_time_range_row_populates(
    ripple, ripple_keys, normalization_time_range
):
    """A stored ripple_detection 1.x key is applied as normalization_mask."""
    from ripple_detection import Kay_ripple_detector

    legacy_key = ripple_keys[1]
    stored = (ripple.RippleTimesV1 & legacy_key).fetch1_dataframe()

    speed, lfps, sampling_frequency = (
        ripple.RippleTimesV1.get_ripple_lfps_and_position_info(legacy_key)
    )
    time = np.asarray(lfps.index)
    start, end = normalization_time_range
    params = (
        ripple.RippleParameters & {"ripple_param_name": "default_trodes"}
    ).fetch1("ripple_param_dict")["ripple_detection_params"]
    expected = Kay_ripple_detector(
        time,
        np.asarray(lfps),
        np.asarray(speed),
        sampling_frequency,
        normalization_mask=(time >= start) & (time <= end),
        **params,
    )
    assert len(expected) > 0, "No ripple events to compare"
    assert len(stored) == len(expected)
    np.testing.assert_allclose(
        stored[["start_time", "end_time"]].to_numpy(),
        expected[["start_time", "end_time"]].to_numpy(),
    )


def test_consensus_trace_aligned_with_lfps(ripple, ripple_keys):
    _, lfps, sampling_frequency = (
        ripple.RippleTimesV1.get_ripple_lfps_and_position_info(ripple_keys[0])
    )
    trace = ripple.RippleTimesV1.get_Kay_ripple_consensus_trace(
        lfps, sampling_frequency
    )
    assert trace.shape == (len(lfps), 1)
    assert trace.index.equals(lfps.index)
    valid = ~lfps.isna().any(axis=1).to_numpy()
    assert np.all(np.isfinite(trace.to_numpy().ravel()[valid]))
