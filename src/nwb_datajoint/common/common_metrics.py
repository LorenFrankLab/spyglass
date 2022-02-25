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
    metric_default_params = {
        'snr': {'peak_sign': 'neg',
                'num_chunks_per_segment':20,
                'chunk_size':10000,
                'seed':0},
        'isi_violation': {'isi_threshold_ms': 1.5},
        'nn_isolation': {'max_spikes_for_nn': 1000,
                         'n_neighbors': 5, 
                         'n_components': 7,
                         'radius_um': 100,
                         'seed': 0},
        'nn_noise_overlap': {'max_spikes_for_nn': 1000,
                             'n_neighbors': 5, 
                             'n_components': 7,
                             'radius_um': 100,
                             'seed': 0}
        }
    available_metrics = list(metric_default_params.keys())

    def get_metric_default_params(self, metric: str):
        "Returns default params for the given metric"
        return self.metric_default_params(metric)

    def insert_default(self):
        self.insert1(['franklab_default', self.metric_default_params], skip_duplicates=True)
    
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
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_qm, f, ensure_ascii=False, indent=4)
        
def _compute_isi_violation_fractions(self, waveform_extractor, **metric_params):
    isi_threshold_ms = metric_params[isi_threshold_ms]
    min_isi_ms = metric_params[min_isi_ms]
    
    # Extract the total number of spikes that violated the isi_threshold for each unit
    isi_violation_counts = st.qualitymetrics.compute_isi_violations(waveform_extractor, isi_threshold_ms=isi_threshold_ms, min_isi_ms=min_isi_ms).isi_violations_count
    
    # Extract the total number of spikes from each unit
    num_spikes = st.qualitymetrics.compute_num_spikes(waveform_extractor)
    isi_viol_frac_metric = {}
    for unit_id in waveform_extractor.sorting.get_unit_ids():
        isi_viol_frac_metric[str(unit_id)] = isi_violation_counts[unit_id] / num_spikes[unit_id]
    return isi_viol_frac_metric

_metric_name_to_func = {
    "snr": st.qualitymetrics.compute_snrs,
    "isi_violation": _compute_isi_violation_fractions,
    'nn_isolation': st.qualitymetrics.nearest_neighbors_isolation,
    'nn_noise_overlap': st.qualitymetrics.nearest_neighbors_noise_overlap
}