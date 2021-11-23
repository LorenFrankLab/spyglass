import datajoint as dj
import spikeinterface.toolkit as st

schema = dj.schema('common_metrics')

@schema
class MetricParameters(dj.Manual):
    definition = """
    # Parameters for quality metrics of sorted units
    list_name: varchar(80)
    ---
    metric_parameters: blob  # {'metric_name': {'parameter': 'value'}}
    """

    def get_metric_list(self):
        """Create a dict where the keys are names of metrics in spikeinterface
        and values are False.
        
        Users should set the desired set of metrics to be true and insert a new
        entry for that set.
        """
        metric_list = st.qualitymetrics.get_quality_metric_list()
        return metric_list

    def get_metric_params_dict(self, metric):
        """Provides default params for the given metric

        Parameters
        ----------
        metric_dict: dict, {'metric name': bool}
        """
        metric_parmas = _get_metric_default_params(metric)
        return metric_params

    def insert_default_params(self):
        """
        Re-inserts the entry for Frank lab default parameters
        (run in case it gets accidentally deleted)
        """
        list_name = 'franklab_default'
        metric_parameters = {'isi': _get_metric_default_params('isi'),
                             'nn_isolation': _get_metric_default_params('nn_isolation'),
                             'nn_noise_overlap': _get_metric_default_params('nn_noise_overlap')}

        self.insert1([list_name, metric_parameters], replace=True)

    def validate_metrics_list(self, key):
        """Checks whether metrics_list contains only valid metric names

        :param key: key for metrics to validate
        :type key: dict
        :return: True or False
        :rtype: boolean
        """
        # TODO: get list of valid metrics from spiketoolkit when available
        valid_metrics = self.get_metric_dict()
        metric_dict = (self & key).fetch1('metric_dict')
        valid = True
        for metric in metric_dict:
            if not metric in valid_metrics.keys():
                print(
                    f'Error: {metric} not in list of valid metrics: {valid_metrics}')
                valid = False
        return valid
    
    @staticmethod
    def compute_metrics(recording, sorting):
        """
        Parameters
        ----------
        cluster_metrics_list_name: str
        recording: spikeinterface SpikeSortingRecording
        sorting: spikeinterface SortingExtractor

        Returns
        -------
        metrics: pandas.dataframe
        """
        # validate input
        m = (self & {'cluster_metrics_list_name': cluster_metrics_list_name}).fetch1()

        return st.validation.compute_quality_metrics(sorting=sorting,
                                                     recording=recording,
                                                     metric_names=self.selected_metrics_list(
                                                         m['metric_dict']),
                                                     as_dataframe=True,
                                                     **m['metric_parameter_dict'])

class MetricSelection(dj.Computed):
    defition="""
    -> SpikeSorting
    -> MetricParameters
    """
    def make(self, key):
        sorting = key['sorting']
        metric_params = key['metric_params']
        MetricParameters.compute_metrics(sorting, metric_params)
    

def _get_metric_default_params(metric):
    if metric is 'nn_isolation':
        return {'max_spikes_for_nn': 1000,
                'n_neighbors': 5, 
                'n_components': 7,
                'radius_um': 100,
                'seed': 0}