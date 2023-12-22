import os

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import temp_dir
from spyglass.spikesorting.merge import SpikeSortingOutput
from spyglass.spikesorting.v1 import CurationV1, SpikeSortingSelection
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("waveform_features")


@schema
class WaveformFeaturesParams(SpyglassMixin, dj.Lookup):
    """Defines the types of spike waveform features computed for a given spike
    time."""

    definition = """
    features_param_name : varchar(80) # a name for this set of parameters
    ---
    params : longblob # the parameters for the waveform features
    """
    _default_waveform_feature_params = {
        "amplitude": {
            "peak_sign": "neg",
            "estimate_peak_time": False,
        }
    }
    _default_waveform_extraction_params = {
        "ms_before": 0.5,
        "ms_after": 0.5,
        "max_spikes_per_unit": None,
        "n_jobs": 5,
        "total_memory": "5G",
    }
    contents = [
        [
            "amplitude",
            {
                "waveform_features_params": _default_waveform_feature_params,
                "waveform_extraction_params": _default_waveform_extraction_params,
            },
        ]
    ]

    @classmethod
    def insert_default(cls):
        cls.insert(cls.contents, skip_duplicates=True)

    @staticmethod
    def check_supported_waveform_features(waveform_features: list[str]) -> bool:
        """Checks whether the requested waveform features types are supported.
        Currently only 'amplitude" is supported.

        Parameters
        ----------
        mark_type : str

        """
        supported_features = set(WAVEFORM_FEATURE_FUNCTIONS)
        return set(waveform_features).issubset(supported_features)

    @property
    def supported_waveform_features(self) -> list[str]:
        """Returns the list of supported waveform features"""
        return list(WAVEFORM_FEATURE_FUNCTIONS)


@schema
class UnitWaveformFeaturesSelection(dj.Manual):
    definition = """
    -> SpikeSortingOutput
    -> WaveformFeaturesParams
    """


@schema
class UnitWaveformFeatures(SpyglassMixin, dj.Computed):
    """For each spike time, compute a spike waveform feature associated with that
    spike. Used for clusterless decoding.
    """

    definition = """
    -> UnitWaveformFeaturesSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40) # the NWB object that stores the waveforms
    """

    def make(self, key):
        # get the list of feature parameters
        params = (WaveformFeaturesParams & key).fetch1("params")

        # check that the feature type is supported
        if not WaveformFeaturesParams.check_supported_waveform_features(
            params["waveform_features_params"]
        ):
            raise NotImplementedError(
                f"Features {set(params['waveform_features_params'])} are not supported"
            )

        # retrieve the units from the NWB file
        merge_key = {"merge_id": key["merge_id"]}

        curation_key = (SpikeSortingOutput.CurationV1 & merge_key).fetch1()
        recording = CurationV1.get_recording(curation_key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])
        sorting = CurationV1.get_sorting(curation_key)

        waveforms_dir = temp_dir + "/" + str(curation_key["sorting_id"])
        os.makedirs(waveforms_dir, exist_ok=True)

        waveform_extractor = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveforms_dir,
            overwrite=True,
            **params["waveform_extraction_params"],
        )
        sorter, nwb_file_name = (
            SpikeSortingSelection & {"sorting_id": curation_key["sorting_id"]}
        ).fetch1("sorter", "nwb_file_name")

        waveform_features = {}

        for feature, feature_params in params[
            "waveform_features_params"
        ].items():
            waveform_features[feature] = self._compute_waveform_features(
                waveform_extractor,
                feature,
                feature_params,
                sorter,
            )

        (
            key["analysis_file_name"],
            key["object_id"],
        ) = _write_waveform_features_to_nwb(
            nwb_file_name,
            waveform_extractor,
            waveform_features,
        )

        AnalysisNwbfile().add(
            nwb_file_name,
            key["analysis_file_name"],
        )
        self.insert1(key)

    def _compute_waveform_features(
        self,
        waveform_extractor,
        feature,
        feature_params,
        sorter,
    ):
        feature_func = WAVEFORM_FEATURE_FUNCTIONS[feature]
        if sorter == "clusterless_thresholder" and feature == "amplitude":
            feature_params["estimate_peak_time"] = False

        return {
            unit_id: feature_func(
                waveform_extractor.get_waveforms(unit_id), **feature_params
            )
            for unit_id in waveform_extractor.sorting.get_unit_ids()
        }

    def fetch1_dataframe(self):
        """Convenience function for returning the marks in a readable format"""
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [self._convert_to_dataframe(data) for data in self.fetch_nwb()]

    @staticmethod
    def _convert_to_dataframe(nwb_data):
        n_marks = nwb_data["marks"].data.shape[1]
        columns = [f"amplitude_{ind:04d}" for ind in range(n_marks)]
        return pd.DataFrame(
            nwb_data["marks"].data,
            index=pd.Index(nwb_data["marks"].timestamps, name="time"),
            columns=columns,
        )


def get_peak_amplitude(
    waveforms: np.ndarray,
    peak_sign: str = "neg",
    estimate_peak_time: bool = False,
):
    """Returns the amplitudes of all channels at the time of the peak
    amplitude across channels.

    Parameters
    ----------
    waveform : array-like, shape (n_spikes, n_time, n_channels)
    peak_sign : ('pos', 'neg', 'both'), optional
        Direction of the peak in the waveform
    estimate_peak_time : bool, optional
        Find the peak times for each spike because some spikesorters do not
        align the spike time (at index n_time // 2) to the peak

    Returns
    -------
    peak_amplitudes : array-like, shape (n_spikes, n_channels)

    """

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


def get_full_waveform(waveforms, **kwargs):
    return waveforms.reshape((waveforms.shape[0], -1))


WAVEFORM_FEATURE_FUNCTIONS = {
    "amplitude": get_peak_amplitude,
    "full_waveform": get_full_waveform,
}


# spikeinterface.postprocessing.compute_spike_locations
# spikeinterface.postprocessing.compute_unit_locations
# spikeinterface.postprocessing.compute_spike_amplitudes
# before we had clusterless_thresholder just use the middle and if not estimate the peak time
# spikeinterface.postprocessing.compute_principal_components
# spikeinterface.postprocessing.compute_template_metrics


def _write_waveform_features_to_nwb(
    nwb_file_name: str,
    waveforms: si.WaveformExtractor,
    waveform_features=None,
):
    """Save waveforms, metrics, labels, and merge groups to NWB in the units table.

    Parameters
    ----------
    nwb_file_name : str
        name of the NWB file containing the spike sorting information
    waveforms : si.WaveformExtractor
        waveform extractor object containing the waveforms
    waveform_features : dict, optional
        dictionary of waveform_features to be saved in the NWB file

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the sorting and curation information
    object_id : str
        object_id of the units table in the analysis NWB file
    """

    unit_ids = [int(i) for i in waveforms.sorting.get_unit_ids()]

    # create new analysis nwb file
    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        # Write waveforms to the nwb file
        for unit_id in unit_ids:
            nwbf.add_unit(
                spike_times=waveforms.sorting.get_unit_spike_train(unit_id),
                id=unit_id,
                electrodes=waveforms.recording.get_channel_ids(),
                waveforms=waveforms.get_waveforms(unit_id),
                waveform_mean=waveforms.get_template(unit_id),
            )

        if waveform_features is not None:
            for metric, metric_dict in waveform_features.items():
                metric_values = [
                    metric_dict[unit_id] if unit_id in metric_dict else []
                    for unit_id in unit_ids
                ]
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        units_object_id = nwbf.units.object_id
        io.write(nwbf)

    return analysis_nwb_file, units_object_id
