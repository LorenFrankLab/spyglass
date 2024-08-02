import numpy as np
import pynwb
import scipy.signal as signal

from spyglass.common.common_usage import ActivityLog


def hilbert_decomp(lfp_band_object, sampling_rate=1):
    """Generates analytical decomposition of signals in the lfp_band_object

    NOTE: This function is not currently used in the pipeline.

    Parameters
    ----------
    lfp_band_object : pynwb.ecephys.ElectricalSeries
        bandpass filtered LFP
    sampling_rate : int, optional
        bandpass filtered LFP sampling rate
        (defaults to 1; only used for instantaneous frequency)

    Returns
    -------
    envelope : pynwb.ecephys.ElectricalSeries
        envelope of the signal
    """
    ActivityLog().deprecate_log("common.signal_processing.hilbert_decomp")

    analytical_signal = signal.hilbert(lfp_band_object.data, axis=0)

    eseries_name = "envelope"
    envelope = pynwb.ecephys.ElectricalSeries(
        name=eseries_name,
        data=np.abs(analytical_signal),
        electrodes=lfp_band_object.electrodes,
        timestamps=lfp_band_object.timestamps,
    )

    eseries_name = "phase"
    instantaneous_phase = np.unwrap(np.angle(analytical_signal))
    phase = pynwb.ecephys.ElectricalSeries(
        name=eseries_name,
        data=instantaneous_phase,
        electrodes=lfp_band_object.electrodes,
        timestamps=lfp_band_object.timestamps,
    )

    eseries_name = "frequency"
    instantaneous_frequency = (
        np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_rate
    )
    frequency = pynwb.ecephys.ElectricalSeries(
        name=eseries_name,
        data=instantaneous_frequency,
        electrodes=lfp_band_object.electrodes,
        timestamps=lfp_band_object.timestamps,
    )
    return envelope, phase, frequency
