from pandas import DataFrame, Index


def test_lfp_dataframe(common, lfp, lfp_analysis_raw, lfp_merge_key):
    lfp_raw = lfp_analysis_raw.scratch["filtered data"]
    df_raw = DataFrame(
        lfp_raw.data, index=Index(lfp_raw.timestamps, name="time")
    )
    df_fetch = (lfp.LFPOutput & lfp_merge_key).fetch1_dataframe()

    assert df_raw.equals(df_fetch), "LFP dataframe not match."


def test_lfp_band_dataframe(lfp_band_analysis_raw, lfp_band, lfp_band_key):
    lfp_band_raw = (
        lfp_band_analysis_raw.processing["ecephys"]
        .fields["data_interfaces"]["LFP"]
        .electrical_series["filtered data"]
    )
    df_raw = DataFrame(
        lfp_band_raw.data, index=Index(lfp_band_raw.timestamps, name="time")
    )
    df_fetch = (lfp_band.LFPBandV1 & lfp_band_key).fetch1_dataframe()

    assert df_raw.equals(df_fetch), "LFPBand dataframe not match."
