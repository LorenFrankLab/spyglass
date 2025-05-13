import numpy as np
import spikeinterface as si


def _get_peak_amplitude(
    waveform_extractor: si.WaveformExtractor,
    unit_idx: int,
    peak_sign: str = "neg",
    estimate_peak_time: bool = False,
) -> np.ndarray:
    """Returns the amplitudes of all channels at the time of the peak
    amplitude across channels.

    Parameters
    ----------
    waveform_extractor : si.WaveformExtractor
        An object that provides access to waveforms for specific units.
    unit_id : int
        The ID of the unit for which to extract waveforms.
    peak_sign : {'pos', 'neg', 'both'}, optional
        Direction of the peak in the waveform. Defaults to 'neg'.
    estimate_peak_time : bool, optional
        If True, estimates the peak times for each spike because some
        spikesorters do not align the spike time (at index n_time // 2)
        to the peak. Defaults to False.

    Returns
    -------
    peak_amplitudes : np.ndarray, shape (n_spikes, n_channels)
        The amplitudes of all channels at the time of the peak amplitude
        across channels.

    """
    waveforms = waveform_extractor.get_waveforms(unit_idx)
    if estimate_peak_time:
        if peak_sign == "neg":
            peak_inds = np.argmin(np.min(waveforms, axis=2), axis=1)
        elif peak_sign == "pos":
            peak_inds = np.argmax(np.max(waveforms, axis=2), axis=1)
        elif peak_sign == "both":
            peak_inds = np.argmax(np.max(np.abs(waveforms), axis=2), axis=1)

        # Get mode of peaks to find the peak time
        values, counts = np.unique(peak_inds, return_counts=True)
        spike_peak_ind = values[counts.argmax()]
    else:
        spike_peak_ind = waveforms.shape[1] // 2

    return waveforms[:, spike_peak_ind]
