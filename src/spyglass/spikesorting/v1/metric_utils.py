import spikeinterface as si
import spikeinterface.qualitymetrics as sq


def _compute_isi_violation_fractions(waveform_extractor, **metric_params):
    """Computes the per unit fraction of interspike interval violations to total spikes."""
    isi_threshold_ms = metric_params["isi_threshold_ms"]
    min_isi_ms = metric_params["min_isi_ms"]

    # Extract the total number of spikes that violated the isi_threshold for each unit
    isi_violation_counts = sq.compute_isi_violations(
        waveform_extractor,
        isi_threshold_ms=isi_threshold_ms,
        min_isi_ms=min_isi_ms,
    ).isi_violations_count

    # Extract the total number of spikes from each unit. The number of ISIs is one less than this
    num_spikes = sq.compute_num_spikes(waveform_extractor)

    # Calculate the fraction of ISIs that are violations
    isi_viol_frac_metric = {
        str(unit_id): isi_violation_counts[unit_id] / (num_spikes[unit_id] - 1)
        for unit_id in waveform_extractor.sorting.get_unit_ids()
    }
    return isi_viol_frac_metric


def _get_peak_offset(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the shift of the waveform peak from center of window."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_offset_inds = (
        si.postprocessing.get_template_extremum_channel_peak_shift(
            waveform_extractor=waveform_extractor,
            peak_sign=peak_sign,
            **metric_params,
        )
    )
    peak_offset = {key: int(abs(val)) for key, val in peak_offset_inds.items()}
    return peak_offset


def _get_peak_channel(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the electrode_id of the channel with the extremum peak for each unit."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_channel_dict = si.postprocessing.get_template_extremum_channel(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    peak_channel = {key: int(val) for key, val in peak_channel_dict.items()}
    return peak_channel


def _get_num_spikes(
    waveform_extractor: si.WaveformExtractor, this_unit_id: int
):
    """Computes the number of spikes for each unit."""
    all_spikes = sq.compute_num_spikes(waveform_extractor)
    cluster_spikes = all_spikes[this_unit_id]
    return cluster_spikes
