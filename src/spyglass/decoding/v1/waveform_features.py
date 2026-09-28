import os
from itertools import chain

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import temp_dir
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v1 import SpikeSortingSelection
from spyglass.utils import SpyglassMixin
from spyglass.utils.waveforms import _get_peak_amplitude

schema = dj.schema("decoding_waveform_features")


@schema
class WaveformFeaturesParams(SpyglassMixin, dj.Lookup):
    """Defines types of waveform features computed for a given spike time.

    Attributes
    ----------
    features_param_name : str
        Name of the waveform features parameters.
    params : dict
        Dictionary of parameters for the waveform features, including...
        waveform_features_params : dict
            amplitude : dict
                peak_sign : enum ("neg", "pos")
                estimate_peak_time : bool
            spike_location : dict
        waveform_extraction_params : dict
            ms_before : float
            ms_after : float
            max_spikes_per_unit : int
            n_jobs : int
            chunk_duration : str
    """

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
    _default_waveform_extract_params = {
        "ms_before": 0.5,
        "ms_after": 0.5,
        "max_spikes_per_unit": None,
        "n_jobs": 5,
        "chunk_duration": "1000s",
    }
    contents = [
        [
            "amplitude",
            {
                "waveform_features_params": _default_waveform_feature_params,
                "waveform_extraction_params": _default_waveform_extract_params,
            },
        ],
        [
            "amplitude, spike_location",
            {
                "waveform_features_params": {
                    "amplitude": _default_waveform_feature_params["amplitude"],
                    "spike_location": {},
                },
                "waveform_extraction_params": _default_waveform_extract_params,
            },
        ],
    ]

    @classmethod
    def insert_default(cls):
        """Insert default waveform features parameters"""
        cls.insert(cls.contents, skip_duplicates=True)

    @staticmethod
    def check_supported_waveform_features(waveform_features: list[str]) -> bool:
        """Checks whether the requested waveform features types are supported

        Parameters
        ----------
        waveform_features : list
        """
        supported_features = set(WAVEFORM_FEATURE_FUNCTIONS)
        return set(waveform_features).issubset(supported_features)

    @property
    def supported_waveform_features(self) -> list[str]:
        """Returns the list of supported waveform features"""
        return list(WAVEFORM_FEATURE_FUNCTIONS)


@schema
class UnitWaveformFeaturesSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SpikeSortingOutput.proj(spikesorting_merge_id="merge_id")
    -> WaveformFeaturesParams
    """


@schema
class UnitWaveformFeatures(SpyglassMixin, dj.Computed):
    """For each spike time, compute waveform feature associated with that spike.

    Used for clusterless decoding.
    """

    definition = """
    -> UnitWaveformFeaturesSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40) # the NWB object that stores the waveforms
    """

    _parallel_make = True

    def make(self, key):
        """Populate UnitWaveformFeatures table."""
        # get the list of feature parameters
        params = (WaveformFeaturesParams & key).fetch1("params")

        # check that the feature type is supported
        if not WaveformFeaturesParams.check_supported_waveform_features(
            params["waveform_features_params"]
        ):
            raise NotImplementedError(
                f"Features {set(params['waveform_features_params'])} are "
                + "not supported"
            )

        merge_key = {"merge_id": key["spikesorting_merge_id"]}
        waveform_extractor = self._fetch_waveform(
            merge_key, params["waveform_extraction_params"]
        )

        source_key = SpikeSortingOutput().merge_get_parent(merge_key).fetch1()
        # v0 pipeline
        if "sorter" in source_key and "nwb_file_name" in source_key:
            sorter = source_key["sorter"]
            nwb_file_name = source_key["nwb_file_name"]
            analysis_nwb_key = "units"
        # v1 pipeline
        else:
            sorting_id = (SpikeSortingOutput.CurationV1 & merge_key).fetch1(
                "sorting_id"
            )
            sorter, nwb_file_name = (
                SpikeSortingSelection & {"sorting_id": sorting_id}
            ).fetch1("sorter", "nwb_file_name")
            analysis_nwb_key = "object_id"

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

        nwb = SpikeSortingOutput().fetch_nwb(merge_key)[0]
        spike_times = (
            nwb[analysis_nwb_key]["spike_times"]
            if analysis_nwb_key in nwb
            else pd.DataFrame()
        )

        (
            key["analysis_file_name"],
            key["object_id"],
        ) = _write_waveform_features_to_nwb(
            nwb_file_name,
            waveform_extractor,
            spike_times,
            waveform_features,
        )

        AnalysisNwbfile().add(
            nwb_file_name,
            key["analysis_file_name"],
        )

        self.insert1(key)

    @staticmethod
    def _fetch_waveform(
        merge_key: dict, waveform_extraction_params: dict
    ) -> si.WaveformExtractor:
        # get the recording from the parent table
        recording = SpikeSortingOutput().get_recording(merge_key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])
        # get the sorting from the parent table
        sorting = SpikeSortingOutput().get_sorting(merge_key)

        waveforms_temp_dir = temp_dir + "/" + str(merge_key["merge_id"])
        os.makedirs(waveforms_temp_dir, exist_ok=True)

        return si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveforms_temp_dir,
            overwrite=True,
            **waveform_extraction_params,
        )

    @staticmethod
    def _compute_waveform_features(
        waveform_extractor: si.WaveformExtractor,
        feature: str,
        feature_params: dict,
        sorter: str,
    ) -> dict:
        feature_func = WAVEFORM_FEATURE_FUNCTIONS[feature]
        if sorter == "clusterless_thresholder" and feature == "amplitude":
            feature_params["estimate_peak_time"] = False

        return {
            unit_id: feature_func(waveform_extractor, unit_id, **feature_params)
            for unit_id in waveform_extractor.sorting.get_unit_ids()
        }

    def fetch_data(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Fetches the spike times and features for each unit.

        Returns
        -------
        spike_times : list of np.ndarray
            List of spike times for each unit
        features : list of np.ndarray
            List of features for each unit

        """
        return tuple(
            zip(
                *list(
                    chain(
                        *[self._convert_data(data) for data in self.fetch_nwb()]
                    )
                )
            )
        )

    @staticmethod
    def _convert_data(nwb_data) -> list[tuple[np.ndarray, np.ndarray]]:
        feature_df = nwb_data["object_id"]

        feature_columns = [
            column for column in feature_df.columns if column != "spike_times"
        ]

        return [
            (
                unit.spike_times,
                np.concatenate(unit[feature_columns].to_numpy(), axis=1),
            )
            for _, unit in feature_df.iterrows()
        ]


def _get_full_waveform(
    waveform_extractor: si.WaveformExtractor, unit_id: int, **kwargs
) -> np.ndarray:
    """Returns the full waveform around each spike.

    Parameters
    ----------
    waveform_extractor : si.WaveformExtractor
    unit_id : int

    Returns
    -------
    waveforms : np.ndarray, shape (n_spikes, n_time * n_channels)
    """
    waveforms = waveform_extractor.get_waveforms(unit_id)
    return waveforms.reshape((waveforms.shape[0], -1))


def _get_spike_locations(
    waveform_extractor: si.WaveformExtractor, unit_id: int, **kwargs
) -> np.ndarray:
    """Returns the spike locations in 2D or 3D space.

    Parameters
    ----------
    waveform_extractor : si.WaveformExtractor
        _description_
    unit_id : int
        Not used but needed for the function signature

    Returns
    -------
    spike_locations: np.ndarray, shape (n_spikes, 2 or 3)
        spike locations in 2D or 3D space
    """
    spike_locations = si.postprocessing.compute_spike_locations(
        waveform_extractor
    )
    return np.array(spike_locations.tolist())


WAVEFORM_FEATURE_FUNCTIONS = {
    "amplitude": _get_peak_amplitude,
    "full_waveform": _get_full_waveform,
    "spike_location": _get_spike_locations,
}


def _write_waveform_features_to_nwb(
    nwb_file_name: str,
    waveforms: si.WaveformExtractor,
    spike_times: pd.DataFrame,
    waveform_features: dict,
) -> tuple[str, str]:
    """Save waveforms, metrics, labels, and merge groups to NWB units table.

    Parameters
    ----------
    nwb_file_name : str
        name of the NWB file containing the spike sorting information
    waveforms : si.WaveformExtractor
        waveform extractor object containing the waveforms
    spike_times : pd.DataFrame
    waveform_features : dict
        dictionary of waveform_features to be saved in the NWB file

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the sorting and curation
        information
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
                spike_times=spike_times.loc[unit_id],
                id=unit_id,
            )

        if waveform_features is not None:
            for metric, metric_dict in waveform_features.items():
                metric_values = [
                    metric_dict[unit_id] if unit_id in metric_dict else []
                    for unit_id in unit_ids
                ]
                if not metric_values:
                    metric_values = np.array([]).astype(np.float32)
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        units_object_id = nwbf.units.object_id
        io.write(nwbf)

    return analysis_nwb_file, units_object_id
