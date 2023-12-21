import os

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import temp_dir
from spyglass.spikesorting.merge import SpikeSortingOutput
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("waveform_features")


@schema
class WaveformFeaturesParams(SpyglassMixin, dj.Lookup):
    """Defines the types of spike waveform features computed for a given spike
    time."""

    definition = """
    waveform_features_param_name : varchar(80) # a name for this set of parameters
    ---
    waveform_features : BLOB  # a list of waveform features
    waveform_features_params: BLOB # dictionary of dictionary of parameters for each waveform feature
    waveform_extraction_params : BLOB # parameters for extracting waveforms
    """

    @property
    def default_pk(self) -> dict:
        return {"waveform_features_param_name": "default"}

    @property
    def default_params(self) -> dict:
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
        return {
            **cls().default_pk,
            "waveform_features": ["amplitude"],
            "waveform_features_params": waveform_feature_params,
            "waveform_extraction_params": waveform_extraction_params,
        }

    @classmethod
    def insert_default(cls, **kwargs):
        """
        Insert default parameter set for position determination
        """
        cls.insert1(
            cls().default_params,
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls):
        query = cls & cls().default_pk
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)

        return query.fetch1()

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
    -> WaveformFeaturesParameters
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
    waveform_object_id: varchar(40) # the NWB object that stores the waveforms
    """

    def make(self, key):
        # get the list of feature parameters
        waveform_feature_params = (WaveformFeaturesParams & key).fetch1()

        # check that the feature type is supported
        if not WaveformFeaturesParams.check_supported_waveform_features(
            waveform_feature_params["waveform_features"]
        ):
            raise NotImplementedError(
                f"Feature {waveform_feature_params['waveform_features']} is not supported"
            )

        # retrieve the units from the NWB file
        merge_dict = {"merge_id": key["merge_id"]}
        nwb_units = SpikeSortingOutput.fetch_nwb(merge_dict)[0]["object_id"]

        curation_key = (SpikeSortingOutput.CurationV1() & merge_dict).fetch1()
        recording = sgs.CurationV1.get_recording(curation_key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])
        sorting = sgs.CurationV1.get_sorting(curation_key)

        waveforms_dir = temp_dir + "/" + str(curation_key["sorting_id"])
        os.makedirs(waveforms_dir, exist_ok=True)

        logger.info("Extracting waveforms...")
        waveform_extractor = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveforms_dir,
            overwrite=True,
            **waveform_feature_params["waveform_extraction_params"],
        )
        sorter = (
            sgs.SpikeSortingSelection()
            & {"sorting_id": curation_key["sorting_id"]}
        ).fetch1("sorter")

        waveform_features = []

        for unit_id in nwb_units.index:
            waveforms = waveform_extractor.get_waveforms(unit_id)
            unit_waveform_features = []
            for feature in waveform_feature_params["waveform_features"]:
                if (
                    sorter == "clusterless_thresholder"
                    and feature == "amplitude"
                ):
                    waveform_feature_params["waveform_features_params"][
                        feature
                    ]["estimate_peak_time"] = False

                unit_waveform_features.append(
                    WAVEFORM_FEATURE_FUNCTIONS[feature](
                        waveforms,
                        **waveform_feature_params["waveform_features_params"][
                            feature
                        ],
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
