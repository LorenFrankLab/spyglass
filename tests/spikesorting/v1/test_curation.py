import numpy as np
from spikeinterface import BaseSorting
from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor


def test_curation_rec(spike_v1, pop_curation):
    rec = spike_v1.CurationV1.get_recording(pop_curation)
    assert isinstance(
        rec, NwbRecordingExtractor
    ), "CurationV1.get_recording failed to return a RecordingExtractor"

    sample_freq = rec.get_sampling_frequency()
    assert np.isclose(
        29_959.3, sample_freq
    ), "CurqtionV1.get_sampling_frequency unexpected value"

    times = rec.get_times()
    assert np.isclose(
        1687474805.4, np.mean((times[0], times[-1]))
    ), "CurationV1.get_times unexpected value"


def test_curation_sort(spike_v1, pop_curation):
    sort = spike_v1.CurationV1.get_sorting(pop_curation)
    sort_dict = sort.to_dict()
    assert isinstance(
        sort, BaseSorting
    ), "CurationV1.get_sorting failed to return a BaseSorting"

    expected = {
        "class": "spikeinterface.core.numpyextractors.NumpySorting",
        "module": "spikeinterface",
        "relative_paths": False,
    }
    for k in expected:
        assert (
            sort_dict[k] == expected[k]
        ), f"CurationV1.get_sorting unexpected value: {k}"


def test_curation_sort_info(spike_v1, pop_curation):
    sort_info = spike_v1.CurationV1.get_sort_group_info(pop_curation).fetch1()
    exp = {
        "bad_channel": "False",
        "curation_id": 0,
        "electrode_group_name": "0",
        "electrode_id": 0,
        "filtering": "None",
        "impedance": 0.0,
        "merges_applied": 0,
        "name": "0",
        "nwb_file_name": "minirec20230622_.nwb",
        "original_reference_electrode": 0,
        "parent_curation_id": -1,
        "probe_electrode": 0,
        "probe_id": "tetrode_12.5",
        "probe_shank": 0,
        "region_id": 1,
        "sort_group_id": 0,
        "subregion_name": None,
        "subsubregion_name": None,
        "x": 0.0,
        "x_warped": 0.0,
        "y": 0.0,
        "y_warped": 0.0,
        "z": 0.0,
        "z_warped": 0.0,
    }

    for k in exp:
        assert (
            sort_info[k] == exp[k]
        ), f"CurationV1.get_sort_group_info unexpected value: {k}"


def test_curation_sort_metric(spike_v1, pop_curation, pop_curation_metric):
    sort_metric = spike_v1.CurationV1.get_sort_group_info(
        pop_curation_metric
    ).fetch1()
    expected = {
        "bad_channel": "False",
        "contacts": "",
        "description": "after metric curation",
        "electrode_group_name": "0",
        "electrode_id": 0,
        "filtering": "None",
        "impedance": 0.0,
        "merges_applied": 0,
        "name": "0",
        "nwb_file_name": "minirec20230622_.nwb",
        "original_reference_electrode": 0,
        "parent_curation_id": 0,
        "probe_electrode": 0,
        "probe_id": "tetrode_12.5",
        "probe_shank": 0,
        "region_id": 1,
        "sort_group_id": 0,
        "subregion_name": None,
        "subsubregion_name": None,
        "x": 0.0,
        "x_warped": 0.0,
        "y": 0.0,
        "y_warped": 0.0,
        "z": 0.0,
        "z_warped": 0.0,
    }
    for k in expected:
        assert (
            sort_metric[k] == expected[k]
        ), f"CurationV1.get_sort_group_info unexpected value: {k}"
