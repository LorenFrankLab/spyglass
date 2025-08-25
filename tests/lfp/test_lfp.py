import pytest
from pandas import DataFrame, Index


@pytest.fixture(scope="module")
def lfp_raw(lfp_analysis_raw):
    lfp_raw = lfp_analysis_raw.scratch["filtered data"]
    yield lfp_raw.data, Index(lfp_raw.timestamps, name="time")


def test_lfp_fetch1_dataframe(lfp, lfp_raw):
    lfp_data, lfp_index = lfp_raw
    df_fetch = lfp.v1.LFPV1().fetch1_dataframe()
    df_raw = DataFrame(lfp_data, index=lfp_index)

    assert df_raw.equals(df_fetch), "LFP dataframe not match."


def test_lfp_dataframe(lfp, lfp_raw, lfp_merge_key):
    lfp_data, lfp_index = lfp_raw
    df_raw = DataFrame(lfp_data, index=lfp_index)
    df_fetch = (lfp.LFPOutput & lfp_merge_key).fetch1_dataframe()

    assert df_raw.equals(df_fetch), "LFP dataframe not match."


def test_lfp_band_dataframe(lfp_band_analysis_raw, lfp_band, lfp_band_key):
    lfp_band_raw = (
        lfp_band_analysis_raw.processing["ecephys"]
        .fields["data_interfaces"]["LFP"]
        .electrical_series["filtered data"]
    )
    lfp_band_index = Index(lfp_band_raw.timestamps, name="time")
    df_raw = DataFrame(lfp_band_raw.data, index=lfp_band_index)
    df_fetch = (lfp_band.LFPBandV1 & lfp_band_key).fetch1_dataframe()

    assert df_raw.equals(df_fetch), "LFPBand dataframe not match."


def test_lfp_band_compute_signal_invalid(lfp_band_v1):
    with pytest.raises(ValueError):
        lfp_band_v1.compute_analytic_signal([4])


def test_lfp_band_compute_signal(lfp_band_v1):
    signal_sum = lfp_band_v1.compute_analytic_signal([0]).iloc[:, 0].sum()
    assert (
        pytest.approx(signal_sum, 0.0001) == 189
    ), "LFPBand hilbert signal off."


def test_lfp_band_compute_phase(lfp_band_v1):
    phase_sum = lfp_band_v1.compute_signal_phase([0]).iloc[:, 0].sum()
    assert (
        pytest.approx(phase_sum, 0.0001) == 2857.9293
    ), "LFPBand phase signal off."


def test_lfp_band_compute_power(lfp_band_v1):
    power_sum = lfp_band_v1.compute_signal_power([0]).iloc[:, 0].sum()
    assert pytest.approx(power_sum) == 5_391_437, "LFPBand power signal off."


def test_invalid_band_selection(
    lfp_band,
    mini_copy_name,
    mini_dict,
    lfp_merge_key,
    add_interval,
    lfp_constants,
    add_band_filter,
    add_electrode_group,
):
    valid = dict(
        nwb_file_name=mini_copy_name,
        lfp_merge_id=lfp_merge_key.get("merge_id"),
        electrode_list=lfp_constants.get("lfp_band_electrode_ids"),
        filter_name=add_band_filter,
        interval_list_name=add_interval,
        reference_electrode_list=[-1],
        lfp_band_sampling_rate=lfp_constants.get("lfp_band_sampling_rate"),
    )
    set_elec = lfp_band.LFPBandSelection().set_lfp_band_electrodes
    with pytest.raises(ValueError):
        set_elec(**valid | {"electrode_list": [3]})
    with pytest.raises(ValueError):
        set_elec(**valid | {"filter_name": "invalid_filter"})
    with pytest.raises(ValueError):  # ref list size > electrode list size
        set_elec(**valid | {"reference_electrode_list": [1, 2]})
    with pytest.raises(ValueError):  # ref not in electrode list
        set_elec(**valid | {"reference_electrode_list": [3]})


def test_artifact_param_defaults(art_params, art_param_defaults):
    assert set(art_params.fetch("artifact_params_name")).issubset(
        set(art_param_defaults)
    ), "LFPArtifactDetectionParameters missing default item."


@pytest.mark.skip(reason="See #850")
def test_artifact_detection(lfp, pop_art_detection):
    pass


def test_pop_imported_lfp(lfp, common, mini_dict):
    # check that populated from populate_all_common
    assert len(lfp.lfp_imported.ImportedLFP()) == 1
    assert (
        len(
            lfp.lfp_imported.LFPElectrodeGroup
            & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
        )
        == 1
    )
    # check that rerunning doesn't add duplicates
    lfp.lfp_imported.ImportedLFP().make(mini_dict)
    assert len(lfp.lfp_imported.ImportedLFP()) == 1
