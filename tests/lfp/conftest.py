import pytest
from pynwb import NWBHDF5IO


@pytest.fixture(scope="module")
def lfp_analysis_raw(common, lfp, populate_lfp, mini_dict):
    abs_path = (common.AnalysisNwbfile * lfp.v1.LFPV1 & mini_dict).fetch(
        "analysis_file_abs_path"
    )[0]
    assert abs_path is not None, "No NWBFile found."
    with NWBHDF5IO(path=str(abs_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert nwbfile is not None, "NWBFile empty."
        yield nwbfile


@pytest.fixture(scope="session")
def lfp_band_sampling_rate(lfp, lfp_merge_key):
    yield lfp.LFPOutput.merge_get_parent(lfp_merge_key).fetch1(
        "lfp_sampling_rate"
    )


@pytest.fixture(scope="session")
def add_band_filter(common, lfp_constants, lfp_band_sampling_rate):
    filter_name = lfp_constants.get("filter2_name")
    common.FirFilterParameters().add_filter(
        filter_name,
        lfp_band_sampling_rate,
        "bandpass",
        [4, 5, 11, 12],
        "theta filter for 1 Khz data",
    )
    yield lfp_constants.get("filter2_name")


@pytest.fixture(scope="session")
def add_band_selection(
    lfp_band,
    mini_copy_name,
    mini_dict,
    lfp_merge_key,
    add_interval,
    lfp_constants,
    add_band_filter,
    add_electrode_group,
):
    lfp_band.LFPBandSelection().set_lfp_band_electrodes(
        nwb_file_name=mini_copy_name,
        lfp_merge_id=lfp_merge_key.get("merge_id"),
        electrode_list=lfp_constants.get("lfp_band_electrode_ids"),
        filter_name=add_band_filter,
        interval_list_name=add_interval,
        reference_electrode_list=[-1],
        lfp_band_sampling_rate=lfp_constants.get("lfp_band_sampling_rate"),
    )
    yield (lfp_band.LFPBandSelection & mini_dict).fetch1("KEY")


@pytest.fixture(scope="session")
def lfp_band_key(add_band_selection):
    yield add_band_selection


@pytest.fixture(scope="session")
def populate_lfp_band(lfp_band, add_band_selection):
    lfp_band.LFPBandV1().populate(add_band_selection)
    yield


@pytest.fixture(scope="module")
def lfp_band_analysis_raw(common, lfp_band, populate_lfp_band, mini_dict):
    abs_path = (common.AnalysisNwbfile * lfp_band.LFPBandV1 & mini_dict).fetch(
        "analysis_file_abs_path"
    )[0]
    assert abs_path is not None, "No NWBFile found."
    with NWBHDF5IO(path=str(abs_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert nwbfile is not None, "NWBFile empty."
        yield nwbfile


@pytest.fixture(scope="session")
def art_params(lfp):
    art_params = lfp.v1.LFPArtifactDetectionParameters()
    _ = art_params.insert_default()
    yield art_params


@pytest.fixture(scope="session")
def art_param_defaults():
    yield [
        "default_difference",
        "default_difference_ref",
        "default_mad",
        "none",
    ]


@pytest.fixture(scope="session")
def pop_art_detection(lfp, lfp_v1_key, art_param_defaults):
    lfp.v1.LFPArtifactDetectionSelection().insert(
        [
            dict(**lfp_v1_key, artifact_params_name=param)
            for param in art_param_defaults
        ]
    )
    lfp.v1.LFPArtifactDetection().populate()
    yield lfp.v1.LFPArtifactDetection().fetch("KEY", as_dict=True)
