import datajoint as dj
from spikeinterface.core import waveform_extractor
import spikeinterface.toolkit as st
from spikeinterface.toolkit.qualitymetrics.quality_metric_list import _metric_name_to_func
    
from .common_waveforms import Waveforms
from .common_nwbfile import AnalysisNwbfile

schema = dj.schema('common_metrics')

@schema
class MetricParameters(dj.Manual):
    definition = """
    # Parameters for computing quality metrics of sorted units
    metric_params_name: varchar(200)
    ---
    params: blob
    """

    def get_metrics_list(self):
        """Returns a list of quality metrics available in spikeinterface
        """
        metric_list = st.qualitymetrics.get_quality_metric_list()
        return metric_list

    def get_metric_params_dict(self, metric: str):
        """Returns default params for the given metric
        """
        metric_params = _get_metric_default_params(metric)
        return metric_params

    def insert_default(self):
        """Inserts default parameters
        """
        list_name = 'franklab_default'
        params = {'isi': _get_metric_default_params('isi'),
                  'snr': _get_metric_default_params('snr'),
                  'nn_isolation': _get_metric_default_params('nn_isolation'),
                  'nn_noise_overlap': _get_metric_default_params('nn_noise_overlap')}
        self.insert1([list_name, params], replace=True)
    
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
    -> AnalysisNwbfile
    """
    def make(self, key):
        waveform_extractor = Waveforms().load_waveforms(key)
        qm = {}
        for metric_name, metric_params in key['params']:
            metric_func = _metric_name_to_func[metric_name]
            metric = metric_func(waveform_extractor, **metric_params)
            qm[metric_name] = metric
        qm_name =  self._get_waveform_extractor_name(key)
        key['analysis_file_name'] = qm_name + '.nwb'
        # TODO: add metrics to analysis NWB file
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])
        
        self.insert1(key)
        
    def _get_quality_metrics_name(self, key):
        wf_name = Waveforms().get_sorting_name(key)
        qm_name = wf_name + '_qm'
        return qm_name

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
        return {'isi_threshold_ms': 1.5,
                'min_isi_ms': 0}
    elif metric == 'snr':
        return {'peak_sign': 'neg',
                'num_chunks_per_segment':20,
                'chunk_size':10000,
                'seed':0}
