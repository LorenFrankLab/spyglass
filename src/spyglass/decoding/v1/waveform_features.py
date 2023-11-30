import os
import shutil
import uuid
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting import (
    CuratedSpikeSortingV1,
    CurationV1,
    SpikeSortingOutput,
)
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("waveform_features")


@schema
class WaveformFeaturesParameters(dj.Manual):
    """Defines the types of spike waveform features computed for a given spike
    time."""

    definition = """
    waveform_features_param_name : varchar(80) # a name for this set of parameters
    ---
    waveform_features : BLOB  # a list of waveform features
    waveform_features_params: BLOB # dictionary of dictionary of parameters for each waveform feature
    waveform_extraction_params : BLOB # parameters for extracting waveforms
    """

    def insert_default(self):
        """Insert the default parameter set

        Examples
        --------
        waveform_features_params = {"amplitude": {'peak_sign': 'neg'}}
        corresponds to negative going waveforms of at least 100 uV size
        """
        waveform_extraction_params = {
            "ms_before": 0.5,
            "ms_after": 0.5,
            "max_spikes_per_unit": None,
            "n_jobs": 5,
            "total_memory": "5G",
        }
        waveform_feature_params = {
            "amplitude": {"peak_sign": "neg", "estimate_peak_time": False}
        }
        self.insert1(
            {
                "waveform_features_param_name": "default",
                "waveform_features": ["amplitude"],
                "waveform_features_params": waveform_feature_params,
                "waveform_extraction_params": waveform_extraction_params,
            },
            skip_duplicates=True,
        )

    @staticmethod
    def check_supported_waveform_features(waveform_features: list[str]) -> bool:
        """Checks whether the requested waveform features types are supported.
        Currently only 'amplitude" is supported.

        Parameters
        ----------
        mark_type : str

        """
        SUPPORTED_FEATURES = set(WAVEFORM_FEATURE_FUNCTIONS)
        return set(waveform_features).issubset(SUPPORTED_FEATURES)

    @property
    def supported_waveform_features(self) -> list[str]:
        """Returns the list of supported waveform features"""
        return list(WAVEFORM_FEATURE_FUNCTIONS)


@schema
class UnitWaveformFeaturesSelection(dj.Manual):
    definition = """
    -> SpikeSortingOutput
    -> WaveformFeaturesParameters
    """


@schema
class UnitWaveformFeatures(dj.Computed):
    """For each spike time, compute a spike waveform feature associated with that
    spike. Used for clusterless decoding.
    """

    definition = """
    -> UnitWaveformFeaturesSelection
    ---
    -> AnalysisNwbfile
    waveform_object_id: varchar(40) # the NWB object that stores the waveforms
    """

    def make(self, key):
        # get the list of mark parameters
        waveform_feature_params = (WaveformFeaturesParameters & key).fetch1()

        # check that the mark type is supported
        if not WaveformFeaturesParameters.check_supported_waveform_features(
            waveform_feature_params["waveform_features"]
        ):
            raise NotImplementedError(
                f"Feature {waveform_feature_params['waveform_features']} is not supported"
            )

        # retrieve the units from the NWB file
        nwb_units = (SpikeSortingOutput & key).fetch_nwb()[0]["units"]

        recording = CurationV1.get_recording(key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])
        sorting = CurationV1.get_curated_sorting(key)
        waveform_extractor_name = (
            f'{key["nwb_file_name"]}_{str(uuid.uuid4())[0:8]}_'
            f'{key["curation_id"]}_clusterless_waveforms'
        )
        waveform_extractor_path = str(
            Path(os.environ["SPYGLASS_WAVEFORMS_DIR"])
            / Path(waveform_extractor_name)
        )
        if os.path.exists(waveform_extractor_path):
            shutil.rmtree(waveform_extractor_path)

        waveform_extractor = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveform_extractor_path,
            **waveform_feature_params["waveform_extraction_params"],
        )
        sorter = (CuratedSpikeSortingV1() & key).fetch1("sorter")
        waveform_features = []

        for unit_id in nwb_units.index:
            waveforms = waveform_extractor.get_waveforms(unit_id)
            unit_waveform_features = []
            for feature, feature_params in zip(
                waveform_feature_params["waveform_features"],
                waveform_feature_params["waveform_features_params"],
            ):
                if (
                    sorter == "clusterless_thresholder"
                    and feature == "amplitude"
                ):
                    feature_params["estimate_peak_time"] = False

                unit_waveform_features.append(
                    WAVEFORM_FEATURE_FUNCTIONS[feature](
                        waveforms,
                        **feature_params,
                    )
                )
            waveform_features.append(
                np.concatenate(unit_waveform_features, axis=1)
            )

        timestamps = nwb_units["spike_times"]

        # create a new AnalysisNwbfile and a timeseries for the marks and save
        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )
        nwb_object = pynwb.TimeSeries(
            name="waveform_features",
            data=waveform_features,
            unit="uV",
            timestamps=timestamps,
            description=",".join(waveform_feature_params["waveform_features"]),
        )
        key["waveform_object_id"] = AnalysisNwbfile().add_nwb_object(
            key["analysis_file_name"], nwb_object
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

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
