from pathlib import Path
import os
import json
import datajoint as dj
import spikeinterface.toolkit as st
import numpy as np
    
from .common_waveforms import Waveforms
from .common_nwbfile import AnalysisNwbfile

schema = dj.schema('common_metrics')

@schema
class MetricParameters(dj.Manual):
    definition = """
    # Parameters for computing quality metrics of sorted units
    metric_params_name: varchar(200)
    ---
    metric_params: blob
    """

    def get_metrics_list(self):
        "Returns a list of quality metrics available in spikeinterface"
        # not all metrics in spikeinterface are supported yet
        # metric_list = st.qualitymetrics.get_quality_metric_list()
        metric_list = ['snr','isi_violation','nn_isolation','nn_noise_overlap']
        return metric_list

    def get_metric_default_params(self, metric: str):
        "Returns default params for the given metric"
        return _get_metric_default_params(metric)

    def insert_default(self):
        list_name = 'franklab_default'
        params = {'snr': _get_metric_default_params('snr'),
                  'isi_violation': _get_metric_default_params('isi_violation'),
                  'nn_isolation': _get_metric_default_params('nn_isolation'),
                  'nn_noise_overlap': _get_metric_default_params('nn_noise_overlap')}
        self.insert1([list_name, params], skip_duplicates=True)
    
    # TODO
    def _validate_metrics_list(self, key):
        """Checks whether a row to be inserted contains only the available metrics
        """
        # get available metrics list
        # get metric list from key
        # compare
        return NotImplementedError
    
@schema
class MetricSelection(dj.Manual):
    definition="""
    -> Waveforms
    -> MetricParameters
    ---
    """        
    
@schema
class QualityMetrics(dj.Computed):
    definition="""
    -> MetricSelection    
    ---
    quality_metrics_path: varchar(500)
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    """
    def make(self, key):
        waveform_extractor = Waveforms().load_waveforms(key)
        qm = {}
        params = (MetricParameters & key).fetch1('metric_params')
        for metric_name, metric_params in params.items():
            metric = self._compute_metric(waveform_extractor, metric_name, **metric_params)
            qm[metric_name] = metric
        qm_name = self._get_quality_metrics_name(key)
        key['quality_metrics_path'] = str(Path(os.environ['NWB_DATAJOINT_WAVEFORMS_DIR']) / Path(qm_name+'.json'))
        # save metrics dict as json
        print(f'Computed all metrics: {qm}')
        self._dump_to_json(qm, key['quality_metrics_path'])

        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
        key['object_id'] = AnalysisNwbfile().add_units_metrics(key['analysis_file_name'], metrics=qm)
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])       
        
        self.insert1(key)
        
    def _get_quality_metrics_name(self, key):
        wf_name = Waveforms()._get_waveform_extractor_name(key)
        qm_name = wf_name + '_qm'
        return qm_name
    
    def _compute_metric(self, waveform_extractor, metric_name, **metric_params):
        print(f'Computing metric: {metric_name}...')
        metric_func = _metric_name_to_func[metric_name]
        if metric_name == 'snr' or metric_name == 'isi_violation':
            metric = metric_func(waveform_extractor, **metric_params)
        else:
            metric = {}
            for unit_id in waveform_extractor.sorting.get_unit_ids():
                metric[str(unit_id)] = metric_func(waveform_extractor, this_unit_id=unit_id, **metric_params)
        return metric
    
    def _dump_to_json(self, qm_dict, save_path):
        new_qm = {}
        for key, value in qm_dict.items():
            m = {}
            for unit_id, metric_val in value.items():
                m[str(unit_id)] = metric_val
            new_qm[str(key)] = m
        print(new_qm)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_qm, f, ensure_ascii=False, indent=4)

def _get_metric_default_params(metric):
    if metric == 'nn_isolation':
        return {'max_spikes_for_nn': 1000,
                'n_neighbors': 5, 
                'n_components': 7,
                'radius_um': 100,
                'seed': 0}
    elif metric == 'nn_noise_overlap':
        return {'max_spikes_for_nn': 1000,
                'n_neighbors': 5, 
                'n_components': 7,
                'radius_um': 100,
                'seed': 0}
    elif metric == 'isi_violation':
        return {'isi_threshold_ms': 1.5}
    elif metric == 'snr':
        return {'peak_sign': 'neg',
                'num_chunks_per_segment':20,
                'chunk_size':10000,
                'seed':0}
    else:
        raise NameError('That metric is not supported yet.')

def _compute_isi_violations(waveform_extractor, isi_threshold_ms=1.5):
    """Modification of spikeinterface.qualitymetrics.misc_metrics.compute_isi_violations

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The waveform extractor object
    isi_threshold_ms : float, optional, default: 1.5
        Threshold for classifying adjacent spikes as an ISI violation, in ms.
        This is the biophysical refractory period (default=1.5).

    Returns
    -------
    isi_violation_fraction : float
        Number of violations / (number of spikes-1)
    """

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()
    fs = recording.get_sampling_frequency()

    isi_threshold_s = isi_threshold_ms / 1000

    isi_threshold_samples = int(isi_threshold_s * fs)

    isi_violations_fraction = {}

    # all units converted to seconds
    for unit_id in unit_ids:
        num_violations = 0
        num_spikes = 0
        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            isis = np.diff(spike_train)
            num_spikes += len(spike_train)
            num_violations += np.sum(isis < isi_threshold_samples)

        isi_violations_fraction[unit_id] = num_violations / (num_spikes-1)

    return isi_violations_fraction

_metric_name_to_func = {
    "snr": st.qualitymetrics.compute_snrs,
    "isi_violation": _compute_isi_violations,
    'nn_isolation': st.qualitymetrics.nearest_neighbors_isolation,
    'nn_noise_overlap': st.qualitymetrics.nearest_neighbors_noise_overlap
}
