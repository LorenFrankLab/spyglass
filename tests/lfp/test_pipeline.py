import numpy as np


def test_lfp_eseries(common, lfp, mini_eseries, lfp_constants, lfp_merge_key):
    lfp_elect_ids = lfp_constants.get("lfp_band_electrode_ids")

    lfp_elect_indices = common.get_electrode_indices(
        mini_eseries, lfp_elect_ids
    )
    lfp_timestamps = np.asarray(mini_eseries.timestamps)
    lfp_eseries = lfp.LFPOutput.fetch_nwb(lfp_merge_key)[0]["lfp"]
    assert False


def test_lfp_band_eseries(ccf, lfp_band, lfp_band_key, lfp_constants):
    lfp_band_elect_ids = lfp_constants.get("lfp_band_electrode_ids")
    lfp_elect_indices = common.get_electrode_indices(
        lfp_eseries, lfp_band_electrode_ids
    )
    lfp_timestamps = np.asarray(lfp_eseries.timestamps)
    lfp_band_eseries = (lfp_band.LFPBandV1 & lfp_band_key).fetch_nwb()[0][
        "lfp_band"
    ]
    lfp_band_elect_indices = common.get_electrode_indices(
        lfp_band_eseries, lfp_band_electrode_ids
    )
    lfp_band_timestamps = np.asarray(lfp_band_eseries.timestamps)
    assert False
