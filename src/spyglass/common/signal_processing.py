import numpy as np
import pynwb
import scipy.signal as signal


def hilbert_decomp(lfp_band_object, sampling_rate=1):
    """generates the analytical decomposition of the signals in the lfp_band_object

    :param lfp_band_object: bandpass filtered LFP
    :type lfp_band_object: pynwb electrical series
    :param sampling_rate: bandpass filtered LFP sampling rate (defaults to 1; only used for instantaneous frequency)
    :type sampling_rate: int
    :return: envelope, phase, frequency
    :rtype: pynwb electrical series objects
    """
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
