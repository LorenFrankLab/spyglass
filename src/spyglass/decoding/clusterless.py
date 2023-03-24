"""Pipeline for decoding the animal's mental position and some category of interest
from unclustered spikes and spike waveform features. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import os
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si
import xarray as xr
from replay_trajectory_classification.classifier import (
    _DEFAULT_CLUSTERLESS_MODEL_KWARGS,
    _DEFAULT_CONTINUOUS_TRANSITIONS,
    _DEFAULT_ENVIRONMENT,
)
from replay_trajectory_classification.continuous_state_transitions import (
    RandomWalk,
    Uniform,
)
from replay_trajectory_classification.discrete_state_transitions import DiagonalDiscrete
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import UniformInitialConditions
from replay_trajectory_classification.observation_model import ObservationModel
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.decoding.core import (
    convert_epoch_interval_name_to_position_interval_name,
    convert_valid_times_to_slice,
    get_valid_ephys_position_times_by_epoch,
)
from spyglass.decoding.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)
from spyglass.spikesorting.spikesorting_curation import (
    CuratedSpikeSorting,
    CuratedSpikeSortingSelection,
    Curation,
)
from spyglass.spikesorting.spikesorting_sorting import (
    SpikeSorting,
    SpikeSortingSelection,
)
from tqdm.auto import tqdm

schema = dj.schema("decoding_clusterless")


@schema
class MarkParameters(dj.Manual):
    """Defines the type of spike waveform feature computed for a given spike
    time."""

    definition = """
    mark_param_name : varchar(80) # a name for this set of parameters
    ---
    # the type of mark. Currently only 'amplitude' is supported
    mark_type = 'amplitude':  varchar(40)
    mark_param_dict:    BLOB    # dictionary of parameters for the mark extraction function
    """

    def insert_default(self):
        """Insert the default parameter set

        Examples
        --------
        {'peak_sign': 'neg', 'threshold' : 100}
        corresponds to negative going waveforms of at least 100 uV size
        """
        default_dict = {}
        self.insert1(
            {"mark_param_name": "default", "mark_param_dict": default_dict},
            skip_duplicates=True,
        )

    @staticmethod
    def supported_mark_type(mark_type):
        """checks whether the requested mark type is supported.
        Currently only 'amplitude" is supported.

        Parameters
        ----------
        mark_type : str

        """
        supported_types = ["amplitude"]
        if mark_type in supported_types:
            return True
        return False


@schema
class UnitMarkParameters(dj.Manual):
    definition = """
    -> CuratedSpikeSorting
    -> MarkParameters
    """


@schema
class UnitMarks(dj.Computed):
    """For each spike time, compute a spike waveform feature associated with that
    spike. Used for clusterless decoding.
    """

    definition = """
    -> UnitMarkParameters
    ---
    -> AnalysisNwbfile
    marks_object_id: varchar(40) # the NWB object that stores the marks
    """

    def make(self, key):
        # get the list of mark parameters
        mark_param = (MarkParameters & key).fetch1()

        # check that the mark type is supported
        if not MarkParameters().supported_mark_type(mark_param["mark_type"]):
            Warning(f'Mark type {mark_param["mark_type"]} not supported; skipping')
            return

        # retrieve the units from the NWB file
        nwb_units = (CuratedSpikeSorting() & key).fetch_nwb()[0]["units"]

        recording = Curation.get_recording(key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])
        sorting = Curation.get_curated_sorting(key)
        waveform_extractor_name = (
            f'{key["nwb_file_name"]}_{str(uuid.uuid4())[0:8]}_'
            f'{key["curation_id"]}_clusterless_waveforms'
        )
        waveform_extractor_path = str(
            Path(os.environ["SPYGLASS_WAVEFORMS_DIR"]) / Path(waveform_extractor_name)
        )
        if os.path.exists(waveform_extractor_path):
            shutil.rmtree(waveform_extractor_path)

        WAVEFORM_PARAMS = {
            "ms_before": 0.5,
            "ms_after": 0.5,
            "max_spikes_per_unit": None,
            "n_jobs": 5,
            "total_memory": "5G",
        }
        waveform_extractor = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveform_extractor_path,
            **WAVEFORM_PARAMS,
        )

        if mark_param["mark_type"] == "amplitude":
            sorter = (CuratedSpikeSorting() & key).fetch1("sorter")
            if sorter == "clusterless_thresholder":
                estimate_peak_time = False
            else:
                estimate_peak_time = True

            try:
                peak_sign = mark_param["mark_param_dict"]["peak_sign"]
            except KeyError:
                peak_sign = "neg"

            marks = np.concatenate(
                [
                    UnitMarks._get_peak_amplitude(
                        waveform=waveform_extractor.get_waveforms(unit_id),
                        peak_sign=peak_sign,
                        estimate_peak_time=estimate_peak_time,
                    )
                    for unit_id in nwb_units.index
                ],
                axis=0,
            )

            timestamps = np.concatenate(np.asarray(nwb_units["spike_times"]))
            sorted_timestamp_ind = np.argsort(timestamps)
            marks = marks[sorted_timestamp_ind]
            timestamps = timestamps[sorted_timestamp_ind]

        if "threshold" in mark_param["mark_param_dict"]:
            timestamps, marks = UnitMarks._threshold(
                timestamps, marks, mark_param["mark_param_dict"]
            )

        # create a new AnalysisNwbfile and a timeseries for the marks and save
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        nwb_object = pynwb.TimeSeries(
            name="marks",
            data=marks,
            unit="uV",
            timestamps=timestamps,
            description="spike features for clusterless decoding",
        )
        key["marks_object_id"] = AnalysisNwbfile().add_nwb_object(
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

    @staticmethod
    def _get_peak_amplitude(waveform, peak_sign="neg", estimate_peak_time=False):
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
                peak_inds = np.argmin(np.min(waveform, axis=2), axis=1)
            elif peak_sign == "pos":
                peak_inds = np.argmax(np.max(waveform, axis=2), axis=1)
            elif peak_sign == "both":
                peak_inds = np.argmax(np.max(np.abs(waveform), axis=2), axis=1)

            # Get mode of peaks to find the peak time
            values, counts = np.unique(peak_inds, return_counts=True)
            spike_peak_ind = values[counts.argmax()]
        else:
            spike_peak_ind = waveform.shape[1] // 2

        return waveform[:, spike_peak_ind]

    @staticmethod
    def _threshold(timestamps, marks, mark_param_dict):
        """Filter the marks by an amplitude threshold

        Parameters
        ----------
        timestamps : array-like, shape (n_time,)
        marks : array-like, shape (n_time, n_channels)
        mark_param_dict : dict

        Returns
        -------
        filtered_timestamps : array-like, shape (n_filtered_time,)
        filtered_marks : array-like, shape (n_filtered_time, n_channels)

        """
        if mark_param_dict["peak_sign"] == "neg":
            include = np.min(marks, axis=1) <= -1 * mark_param_dict["threshold"]
        elif mark_param_dict["peak_sign"] == "pos":
            include = np.max(marks, axis=1) >= mark_param_dict["threshold"]
        elif mark_param_dict["peak_sign"] == "both":
            include = np.max(np.abs(marks), axis=1) >= mark_param_dict["threshold"]
        return timestamps[include], marks[include]


@schema
class UnitMarksIndicatorSelection(dj.Lookup):
    """Bins the spike times and associated spike waveform features for a given
    time interval into regular time bins determined by the sampling rate."""

    definition = """
    -> UnitMarks
    -> IntervalList
    sampling_rate=500 : float
    ---
    """


@schema
class UnitMarksIndicator(dj.Computed):
    """Bins the spike times and associated spike waveform features into regular
    time bins according to the sampling rate. Features that fall into the same
    time bin are averaged.
    """

    definition = """
    -> UnitMarks
    -> UnitMarksIndicatorSelection
    ---
    -> AnalysisNwbfile
    marks_indicator_object_id: varchar(40)
    """

    def make(self, key):
        # TODO: intersection of sort interval and interval list
        interval_times = (IntervalList & key).fetch1("valid_times")

        sampling_rate = (UnitMarksIndicatorSelection & key).fetch("sampling_rate")

        marks_df = (UnitMarks & key).fetch1_dataframe()

        time = self.get_time_bins_from_interval(interval_times, sampling_rate)

        # Bin marks into time bins. No spike bins will have NaN
        marks_df = marks_df.loc[time.min() : time.max()]
        time_index = np.digitize(marks_df.index, time[1:-1])
        marks_indicator_df = (
            marks_df.groupby(time[time_index])
            .mean()
            .reindex(index=pd.Index(time, name="time"))
        )

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(key["nwb_file_name"])

        key["marks_indicator_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=marks_indicator_df.reset_index(),
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)

    @staticmethod
    def get_time_bins_from_interval(interval_times, sampling_rate):
        """Picks the superset of the interval"""
        start_time, end_time = interval_times[0][0], interval_times[-1][-1]
        n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

        return np.linspace(start_time, end_time, n_samples)

    @staticmethod
    def plot_all_marks(marks_indicators: xr.DataArray, plot_size=5, s=10):
        """Plots 2D slices of each of the spike features against each other
        for all electrodes.

        Parameters
        ----------
        marks_indicators : xr.DataArray, shape (n_time, n_electrodes, n_features)
            Spike times and associated spike waveform features binned into
        """
        for electrode_ind in marks_indicators.electrodes:
            marks = (
                marks_indicators.sel(electrodes=electrode_ind)
                .dropna("time", how="all")
                .dropna("marks")
            )
            n_features = len(marks.marks)
            fig, axes = plt.subplots(
                n_features,
                n_features,
                constrained_layout=True,
                sharex=True,
                sharey=True,
                figsize=(plot_size * n_features, plot_size * n_features),
            )
            for ax_ind1, feature1 in enumerate(marks.marks):
                for ax_ind2, feature2 in enumerate(marks.marks):
                    try:
                        axes[ax_ind1, ax_ind2].scatter(
                            marks.sel(marks=feature1), marks.sel(marks=feature2), s=s
                        )
                    except TypeError:
                        axes.scatter(
                            marks.sel(marks=feature1), marks.sel(marks=feature2), s=s
                        )

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [data["marks_indicator"].set_index("time") for data in self.fetch_nwb()]

    def fetch_xarray(self):
        # sort_group_electrodes = (
        #     SortGroup.SortGroupElectrode() &
        #     pd.DataFrame(self).to_dict('records'))
        # brain_region = (sort_group_electrodes * Electrode *
        #                 BrainRegion).fetch('region_name')

        marks_indicators = (
            xr.concat(
                [df.to_xarray().to_array("marks") for df in self.fetch_dataframe()],
                dim="electrodes",
            )
            .transpose("time", "marks", "electrodes")
            .assign_coords({"electrodes": self.fetch("sort_group_id")})
            .sortby(["electrodes", "marks"])
        )

        # hacky way to keep the marks in order
        def reformat_name(name):
            mark_type, number = name.split("_")
            return f"{mark_type}_{int(number):04d}"

        new_mark_names = [reformat_name(name) for name in marks_indicators.marks.values]

        return marks_indicators.assign_coords({"marks": new_mark_names}).sortby(
            ["electrodes", "marks"]
        )


def make_default_decoding_parameters_cpu():
    classifier_parameters = dict(
        environments=[_DEFAULT_ENVIRONMENT],
        observation_models=None,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        discrete_transition_type=DiagonalDiscrete(0.98),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        clusterless_algorithm="multiunit_likelihood_integer",
        clusterless_algorithm_params=_DEFAULT_CLUSTERLESS_MODEL_KWARGS,
    )

    predict_parameters = {
        "is_compute_acausal": True,
        "use_gpu": False,
        "state_names": ["Continuous", "Fragmented"],
    }
    fit_parameters = dict()

    return classifier_parameters, fit_parameters, predict_parameters


def make_default_decoding_parameters_gpu():
    classifier_parameters = dict(
        environments=[_DEFAULT_ENVIRONMENT],
        observation_models=None,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        discrete_transition_type=DiagonalDiscrete(0.98),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        clusterless_algorithm="multiunit_likelihood_integer_gpu",
        clusterless_algorithm_params={"mark_std": 24.0, "position_std": 6.0},
    )

    predict_parameters = {
        "is_compute_acausal": True,
        "use_gpu": True,
        "state_names": ["Continuous", "Fragmented"],
    }

    fit_parameters = dict()

    return classifier_parameters, fit_parameters, predict_parameters


@schema
class ClusterlessClassifierParameters(dj.Manual):
    """Decodes the animal's mental position and some category of interest
    from unclustered spikes and spike waveform features
    """

    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB    # initialization parameters
    fit_params :          BLOB    # fit parameters
    predict_params :      BLOB    # prediction parameters
    """

    def insert_default(self):
        (
            classifier_parameters,
            fit_parameters,
            predict_parameters,
        ) = make_default_decoding_parameters_cpu()
        self.insert1(
            {
                "classifier_param_name": "default_decoding_cpu",
                "classifier_params": classifier_parameters,
                "fit_params": fit_parameters,
                "predict_params": predict_parameters,
            },
            skip_duplicates=True,
        )

        (
            classifier_parameters,
            fit_parameters,
            predict_parameters,
        ) = make_default_decoding_parameters_gpu()
        self.insert1(
            {
                "classifier_param_name": "default_decoding_gpu",
                "classifier_params": classifier_parameters,
                "fit_params": fit_parameters,
                "predict_params": predict_parameters,
            },
            skip_duplicates=True,
        )

    def insert1(self, key, **kwargs):
        super().insert1(convert_classes_to_dict(key), **kwargs)

    def fetch1(self, *args, **kwargs):
        return restore_classes(super().fetch1(*args, **kwargs))


@schema
class MultiunitFiringRate(dj.Computed):
    """Computes the population multiunit firing rate from the spikes in
    MarksIndicator."""

    definition = """
    -> UnitMarksIndicator
    ---
    -> AnalysisNwbfile
    multiunit_firing_rate_object_id: varchar(40)
    """

    def make(self, key):
        marks = (UnitMarksIndicator & key).fetch_xarray()
        multiunit_spikes = (np.any(~np.isnan(marks.values), axis=1)).astype(float)
        multiunit_firing_rate = pd.DataFrame(
            get_multiunit_population_firing_rate(
                multiunit_spikes, key["sampling_rate"]
            ),
            index=marks.time,
            columns=["firing_rate"],
        )

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(key["nwb_file_name"])

        key["multiunit_firing_rate_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=multiunit_firing_rate.reset_index(),
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [
            data["multiunit_firing_rate"].set_index("time") for data in self.fetch_nwb()
        ]


@schema
class MultiunitHighSynchronyEventsParameters(dj.Manual):
    """Parameters for extracting times of high mulitunit activity during immobility."""

    definition = """
    param_name : varchar(80) # a name for this set of parameters
    ---
    minimum_duration = 0.015 :  float # minimum duration of event (in seconds)
    zscore_threshold = 2.0 : float    # threshold event must cross to be considered (in std. dev.)
    close_event_threshold = 0.0 :  float # events closer than this will be excluded (in seconds)
    """

    def insert_default(self):
        self.insert1(
            {
                "param_name": "default",
                "minimum_duration": 0.015,
                "zscore_threshold": 2.0,
                "close_event_threshold": 0.0,
            },
            skip_duplicates=True,
        )


@schema
class MultiunitHighSynchronyEvents(dj.Computed):
    """Finds times of high mulitunit activity during immobility."""

    definition = """
    -> MultiunitHighSynchronyEventsParameters
    -> UnitMarksIndicator
    -> IntervalPositionInfo
    ---
    -> AnalysisNwbfile
    multiunit_hse_times_object_id: varchar(40)
    """

    def make(self, key):
        marks = (UnitMarksIndicator & key).fetch_xarray()
        multiunit_spikes = (np.any(~np.isnan(marks.values), axis=1)).astype(float)
        position_info = (IntervalPositionInfo() & key).fetch1_dataframe()

        params = (MultiunitHighSynchronyEventsParameters & key).fetch1()

        multiunit_high_synchrony_times = multiunit_HSE_detector(
            marks.time.values,
            multiunit_spikes,
            position_info.head_speed.values,
            sampling_frequency=key["sampling_rate"],
            **params,
        )

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(key["nwb_file_name"])

        key["multiunit_hse_times_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=multiunit_high_synchrony_times.reset_index(),
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)


def get_decoding_data_for_epoch(
    nwb_file_name: str,
    interval_list_name: str,
    position_info_param_name="default_decoding",
    additional_mark_keys={},
):
    valid_ephys_position_times_by_epoch = get_valid_ephys_position_times_by_epoch(
        nwb_file_name
    )
    valid_ephys_position_times = valid_ephys_position_times_by_epoch[interval_list_name]
    valid_slices = convert_valid_times_to_slice(valid_ephys_position_times)
    position_interval_name = convert_epoch_interval_name_to_position_interval_name(
        interval_list_name
    )

    position_info = (
        IntervalPositionInfo()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": position_interval_name,
            "position_info_param_name": position_info_param_name,
        }
    ).fetch1_dataframe()

    position_info = pd.concat([position_info.loc[times] for times in valid_slices])

    marks = (
        (
            UnitMarksIndicator()
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": position_interval_name,
                **additional_mark_keys,
            }
        )
    ).fetch_xarray()

    marks = xr.concat([marks.sel(time=times) for times in valid_slices], dim="time")

    return position_info, marks, valid_slices


def get_data_for_multiple_epochs(
    nwb_file_name: str,
    epoch_names: list,
    position_info_param_name="default_decoding",
    additional_mark_keys={},
):
    data = []
    environment_labels = []

    for epoch in epoch_names:
        data.append(
            get_decoding_data_for_epoch(
                nwb_file_name,
                epoch,
                position_info_param_name=position_info_param_name,
                additional_mark_keys=additional_mark_keys,
            )
        )
        n_time = data[-1][0].shape[0]
        environment_labels.append([epoch] * n_time)

    environment_labels = np.concatenate(environment_labels, axis=0)
    position_info, marks, valid_slices = list(zip(*data))
    position_info = pd.concat(position_info, axis=0)
    marks = xr.concat(marks, dim="time")
    valid_slices = {
        epoch: valid_slice for epoch, valid_slice in zip(epoch_names, valid_slices)
    }

    assert position_info.shape[0] == marks.shape[0]

    return position_info, marks, valid_slices, environment_labels


def create_model_for_multiple_epochs(epoch_names: list, env_kwargs: dict):
    observation_models = []
    environments = []
    continuous_transition_types = []

    for epoch in epoch_names:
        observation_models.append(ObservationModel(epoch))
        environments.append(Environment(epoch, **env_kwargs))

    for epoch1 in epoch_names:
        continuous_transition_types.append([])
        for epoch2 in epoch_names:
            if epoch1 == epoch2:
                continuous_transition_types[-1].append(
                    RandomWalk(epoch1, use_diffusion=True)
                )
            else:
                continuous_transition_types[-1].append(Uniform(epoch1, epoch2))

    return observation_models, environments, continuous_transition_types


def populate_mark_indicators(
    spikesorting_selection_keys: list,
    mark_param_name="default",
    position_info_param_name="default_decoding",
):
    spikesorting_selection_keys = deepcopy(spikesorting_selection_keys)
    # Populate spike sorting
    SpikeSortingSelection().insert(
        spikesorting_selection_keys,
        skip_duplicates=True,
    )
    SpikeSorting.populate(spikesorting_selection_keys)

    # Skip any curation
    curation_keys = [
        Curation.insert_curation(key) for key in spikesorting_selection_keys
    ]

    CuratedSpikeSortingSelection().insert(curation_keys, skip_duplicates=True)
    CuratedSpikeSorting.populate(CuratedSpikeSortingSelection() & curation_keys)

    # Populate marks
    mark_parameters_keys = pd.DataFrame(CuratedSpikeSorting & curation_keys)
    mark_parameters_keys["mark_param_name"] = mark_param_name
    mark_parameters_keys = mark_parameters_keys.loc[
        :, UnitMarkParameters.primary_key
    ].to_dict("records")
    UnitMarkParameters().insert(mark_parameters_keys, skip_duplicates=True)
    UnitMarks.populate(UnitMarkParameters & mark_parameters_keys)

    # Compute mark indicators for each position epoch
    nwb_file_name = spikesorting_selection_keys[0]["nwb_file_name"]
    position_interval_names = (
        IntervalPositionInfo()
        & {
            "nwb_file_name": nwb_file_name,
            "position_info_param_name": position_info_param_name,
        }
    ).fetch("interval_list_name")

    for interval_name in tqdm(position_interval_names):
        position_interval = IntervalList & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }

        marks_selection = (UnitMarks & mark_parameters_keys) * position_interval
        marks_selection = (
            pd.DataFrame(marks_selection)
            .loc[:, marks_selection.primary_key]
            .to_dict("records")
        )
        UnitMarksIndicatorSelection.insert(marks_selection, skip_duplicates=True)
        UnitMarksIndicator.populate(marks_selection)
