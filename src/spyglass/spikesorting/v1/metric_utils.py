import numpy as np
import spikeinterface as si
import spikeinterface.qualitymetrics as sq


def compute_isi_violation_fractions(
    waveform_extractor: si.WaveformExtractor,
    this_unit_id: str,
    isi_threshold_ms: float = 2.0,
    min_isi_ms: float = 0.0,
):
    """Computes the fraction of interspike interval violations.

    Parameters
    ----------
    waveform_extractor: si.WaveformExtractor
        The extractor object for the recording.

    """

    # Extract the total number of spikes that violated the isi_threshold for each unit
    _, isi_violation_counts = sq.compute_isi_violations(
        waveform_extractor,
        isi_threshold_ms=isi_threshold_ms,
        min_isi_ms=min_isi_ms,
    )
    num_spikes = sq.compute_num_spikes(waveform_extractor)
    return isi_violation_counts[this_unit_id] / (num_spikes[this_unit_id] - 1)


def get_peak_offset(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the shift of the waveform peak from center of window.

    Parameters
    ----------
    waveform_extractor: si.WaveformExtractor
        The extractor object for the recording.
    peak_sign: str
        The sign of the peak to compute. ('neg', 'pos', 'both')
    """
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_offset_inds = si.get_template_extremum_channel_peak_shift(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    return {key: int(abs(val)) for key, val in peak_offset_inds.items()}


def get_peak_channel(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the electrode_id of the channel with the extremum peak for each unit."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_channel_dict = si.get_template_extremum_channel(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    return {key: int(val) for key, val in peak_channel_dict.items()}


def get_num_spikes(waveform_extractor: si.WaveformExtractor, this_unit_id: str):
    """Computes the number of spikes for each unit."""
    return sq.compute_num_spikes(waveform_extractor)[this_unit_id]
