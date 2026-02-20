"""Pipeline for decoding the animal's mental position and some category of
interest from unclustered spikes and spike waveform features. See [1] for
details.

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

from spyglass.settings import waveforms_dir
from spyglass.utils import logger

try:
    from replay_trajectory_classification.classifier import (
        _DEFAULT_CLUSTERLESS_MODEL_KWARGS,
        _DEFAULT_CONTINUOUS_TRANSITIONS,
        _DEFAULT_ENVIRONMENT,
    )
    from replay_trajectory_classification.discrete_state_transitions import (
        DiagonalDiscrete,
    )
    from replay_trajectory_classification.initial_conditions import (
        UniformInitialConditions,
    )
except (ImportError, ModuleNotFoundError) as e:
    (
        _DEFAULT_CONTINUOUS_TRANSITIONS,
        _DEFAULT_ENVIRONMENT,
        _DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
        DiagonalDiscrete,
        UniformInitialConditions,
    ) = [None] * 5
    logger.warning(e)

from tqdm.auto import tqdm

from spyglass.common.common_behav import (
    convert_epoch_interval_name_to_position_interval_name,
)
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.decoding.v0.core import (
    convert_valid_times_to_slice,
    get_valid_ephys_position_times_by_epoch,
)
from spyglass.decoding.v0.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)
from spyglass.decoding.v0.utils import (
    get_time_bins_from_interval,
    make_default_decoding_params,
)
from spyglass.spikesorting.v0.spikesorting_curation import (
    CuratedSpikeSorting,
    CuratedSpikeSortingSelection,
    Curation,
)
from spyglass.spikesorting.v0.spikesorting_sorting import (
    SpikeSorting,
    SpikeSortingSelection,
)
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.waveforms import _get_peak_amplitude

schema = dj.schema("decoding_clusterless")


@schema
class MarkParameters(SpyglassMixin, dj.Manual):
    """Defines the type of waveform feature computed for a given spike time.

    Attributes
    ----------
    mark_param_name : str
        A name for this set of parameters
    mark_type : str, optional
        The type of mark. Currently only 'amplitude' is supported.
    mark_param_dict : dict, optional
        Dictionary of parameters for the mark extraction function. Default empty
        dictionary. Options include:
            peak_sign : enum ('neg', 'pos', 'both')
                The sign of the peak to extract.
            threshold : float
                The threshold for the amplitude of the mark.
                If None, no threshold is applied.
    """

    definition = """
    mark_param_name : varchar(32) # a name for this set of parameters
    ---
    # the type of mark. Currently only 'amplitude' is supported
    mark_type = 'amplitude':  varchar(40)
    mark_param_dict: BLOB # dict of parameters for the mark extraction function
    """

    # NOTE: See #630, #664. Excessive key length.

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
        """Checks whether the requested mark type is supported.

        Currently only 'amplitude" is supported.

        Parameters
        ----------
        mark_type : str

        """
        supported_types = ["amplitude"]
        return mark_type in supported_types


@schema
class UnitMarkParameters(SpyglassMixin, dj.Manual):
    definition = """
    -> CuratedSpikeSorting
    -> MarkParameters
    """


@schema
class UnitMarks(SpyglassMixin, dj.Computed):
    """Compute spike waveform features for each spike time.

    For each spike time, compute a spike waveform feature associated with that
    spike. Used for clusterless decoding.
    """

    definition = """
    -> UnitMarkParameters
    ---
    -> AnalysisNwbfile
    marks_object_id: varchar(40) # the NWB object that stores the marks
    """

    def make(self, key):
        """Populate the UnitMarks table.

        1. Fetch parameters, units, and recording from MarkParameters,
            CuratedSpikeSorting, and CuratedRecording tables respectively.
        2. Uses spikeinterface to extract waveforms for each unit.
        3. Optionally calculates the peak amplitude of the waveform and
            thresholds the waveform.
        4. Saves the marks as a TimeSeries object in a new AnalysisNwbfile.
        """
        # create a new AnalysisNwbfile and a timeseries for the marks and save
        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )
        # get the list of mark parameters
        mark_param = (MarkParameters & key).fetch1()

        # check that the mark type is supported
        if not MarkParameters().supported_mark_type(mark_param["mark_type"]):
            Warning(
                f'Mark type {mark_param["mark_type"]} not supported; skipping'
            )
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
            Path(waveforms_dir) / Path(waveform_extractor_name)
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

            peak_sign = mark_param["mark_param_dict"].get("peak_sign")

            marks = np.concatenate(
                [
                    _get_peak_amplitude(
                        waveform_extractor=waveform_extractor,
                        peak_sign=peak_sign,
                        unit_id=unit_id,
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

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Convenience function for returning the marks in a readable format"""
        self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> list[pd.DataFrame]:
        """Fetches the marks as a list of pandas dataframes"""
        return [self._convert_to_dataframe(data) for data in self.fetch_nwb()]

    @staticmethod
    def _convert_to_dataframe(nwb_data) -> pd.DataFrame:
        """Converts the marks from an NWB object to a pandas dataframe"""
        n_marks = nwb_data["marks"].data.shape[1]
        columns = [f"amplitude_{ind:04d}" for ind in range(n_marks)]
        return pd.DataFrame(
            nwb_data["marks"].data,
            index=pd.Index(nwb_data["marks"].timestamps, name="time"),
            columns=columns,
        )

    @staticmethod
    def _threshold(
        timestamps: np.array, marks: np.array, mark_param_dict: dict
    ):
        """Filter the marks by an amplitude threshold

        Parameters
        ----------
        timestamps : np.array
            array-like, shape (n_time,)
        marks : np.array
            array-like, shape (n_time, n_channels)
        mark_param_dict : dict

        Returns
        -------
        filtered_timestamps : np.array
            array-like, shape (n_filtered_time,)
        filtered_marks : np.array
            array-like, shape (n_filtered_time, n_channels)

        """
        if mark_param_dict["peak_sign"] == "neg":
            include = np.min(marks, axis=1) <= -1 * mark_param_dict["threshold"]
        elif mark_param_dict["peak_sign"] == "pos":
            include = np.max(marks, axis=1) >= mark_param_dict["threshold"]
        elif mark_param_dict["peak_sign"] == "both":
            include = (
                np.max(np.abs(marks), axis=1) >= mark_param_dict["threshold"]
            )
        return timestamps[include], marks[include]


@schema
class UnitMarksIndicatorSelection(SpyglassMixin, dj.Lookup):
    """Pairing of a UnitMarksIndicator with a time interval and sampling rate

    Bins the spike times and associated spike waveform features for a given
    time interval into regular time bins determined by the sampling rate.
    """

    definition = """
    -> UnitMarks
    -> IntervalList
    sampling_rate=500 : float
    """


@schema
class UnitMarksIndicator(SpyglassMixin, dj.Computed):
    """Bins spike times and waveforms into regular time bins.

    Bins the spike times and associated spike waveform features into regular
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
        """Populate the UnitMarksIndicator table.

        Bin spike times and associated spike waveform features according to the
        sampling rate.
        """
        # TODO: intersection of sort interval and interval list
        interval_times = (IntervalList & key).fetch1("valid_times")

        sampling_rate = (UnitMarksIndicatorSelection & key).fetch(
            "sampling_rate"
        )

        marks_df = (UnitMarks & key).fetch1_dataframe()

        time = get_time_bins_from_interval(interval_times, sampling_rate)

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
        key["analysis_file_name"] = nwb_analysis_file.create(
            key["nwb_file_name"]
        )

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
    def plot_all_marks(
        marks_indicators: xr.DataArray,
        plot_size: int = 5,
        marker_size: int = 10,
        plot_limit: int = None,
    ):
        """Plot all marks for all electrodes.

        Plots 2D slices of each of the spike features against each other
        for all electrodes.

        Parameters
        ----------
        marks_indicators : xr.DataArray, shape (n_time, n_electrodes, n_features)
            Spike times and associated spike waveform features binned into
        plot_size : int, optional
            Default 5. Matplotlib figure size for each mark.
        marker_size : int, optional
            Default 10. Marker size
        plot_limit : int, optional
            Default None. Limits to first N electrodes.
        """
        if not plot_limit:
            plot_limit = len(marks_indicators.electrodes)

        for electrode_ind in marks_indicators.electrodes[:plot_limit]:
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
                            marks.sel(marks=feature1),
                            marks.sel(marks=feature2),
                            s=marker_size,
                        )
                    except TypeError:
                        axes.scatter(
                            marks.sel(marks=feature1),
                            marks.sel(marks=feature2),
                            s=marker_size,
                        )

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Convenience function for returning the first dataframe"""
        _ = self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> list[pd.DataFrame]:
        """Fetches the marks indicators as a list of pandas dataframes"""
        return [
            data["marks_indicator"].set_index("time")
            for data in self.fetch_nwb()
        ]

    def fetch_xarray(self):
        """Fetches the marks indicators as an xarray DataArray"""
        # sort_group_electrodes = (
        #     SortGroup.SortGroupElectrode() &
        #     pd.DataFrame(self).to_dict('records'))
        # brain_region = (sort_group_electrodes * Electrode *
        #                 BrainRegion).fetch('region_name')

        marks_indicators = (
            xr.concat(
                [
                    df.to_xarray().to_array("marks")
                    for df in self.fetch_dataframe()
                ],
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

        new_mark_names = [
            reformat_name(name) for name in marks_indicators.marks.values
        ]

        return marks_indicators.assign_coords({"marks": new_mark_names}).sortby(
            ["electrodes", "marks"]
        )


@schema
class ClusterlessClassifierParameters(SpyglassMixin, dj.Manual):
    """Decodes animal's mental position.

    Decodes the animal's mental position and some category of interest
    from unclustered spikes and spike waveform features

    Attributes
    ----------
    classifier_param_name : str
    classifier_params: dict
        Initialization parameters, including ...
            environments: list
            observation_models
            continuous_transition_types
            discrete_transition_type: DiagonalDiscrete
            initial_conditions_type: UniformInitialConditions
            infer_track_interior: bool
            clusterless_algorithm: str, optional
            clusterless_algorithm_params: dict, optional
            sorted_spikes_algorithm: str, optional
            sorted_spikes_algorithm_params: dict, optional
        For more information, see replay_trajectory_classification documentation
    fit_params: dict, optional
    predict_params: dict, optional
        Prediction parameters, including ...
            is_compute_acausal: bool
            use_gpu: bool
            state_names: List[str]
    """

    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB    # initialization parameters
    fit_params :          BLOB    # fit parameters
    predict_params :      BLOB    # prediction parameters
    """

    def insert_default(self) -> None:
        """Insert the default parameter sets"""
        self.insert(
            [
                make_default_decoding_params(clusterless=True),
                make_default_decoding_params(clusterless=True, use_gpu=True),
            ],
            skip_duplicates=True,
        )

    def insert1(self, key, **kwargs) -> None:
        """Custom insert1 to convert classes to dicts"""
        super().insert1(convert_classes_to_dict(key), **kwargs)

    def fetch1(self, *args, **kwargs) -> dict:
        """Custom fetch1 to convert dicts to classes"""
        return restore_classes(super().fetch1(*args, **kwargs))


def get_decoding_data_for_epoch(
    nwb_file_name: str,
    interval_list_name: str,
    position_info_param_name: str = "default_decoding",
    additional_mark_keys: dict = {},
) -> tuple[pd.DataFrame, xr.DataArray, list[slice]]:
    """Collects necessary data for decoding.

    Parameters
    ----------
    nwb_file_name : str
    interval_list_name : str
    position_info_param_name : str, optional
    additional_mark_keys : dict, optional

    Returns
    -------
    position_info : pd.DataFrame, shape (n_time, n_columns)
    marks : xr.DataArray, shape (n_time, n_marks, n_electrodes)
    valid_slices : list[slice]
    """

    valid_slices = convert_valid_times_to_slice(
        get_valid_ephys_position_times_by_epoch(nwb_file_name)[
            interval_list_name
        ]
    )

    # position interval
    nwb_dict = dict(nwb_file_name=nwb_file_name)
    pos_interval_dict = dict(
        nwb_dict,
        interval_list_name=convert_epoch_interval_name_to_position_interval_name(
            {
                **nwb_dict,
                "interval_list_name": interval_list_name,
            }
        ),
    )

    position_info = (
        IntervalPositionInfo()
        & {
            **pos_interval_dict,
            "position_info_param_name": position_info_param_name,
        }
    ).fetch1_dataframe()

    position_info = pd.concat(
        [position_info.loc[times] for times in valid_slices]
    )

    marks = (
        (UnitMarksIndicator() & {**pos_interval_dict, **additional_mark_keys})
    ).fetch_xarray()

    marks = xr.concat(
        [marks.sel(time=times) for times in valid_slices], dim="time"
    )

    return position_info, marks, valid_slices


def get_data_for_multiple_epochs(
    nwb_file_name: str,
    epoch_names: list[str],
    position_info_param_name="default_decoding",
    additional_mark_keys: dict = {},
) -> tuple[pd.DataFrame, xr.DataArray, dict[str, list[slice]], np.ndarray]:
    """Collects necessary data for decoding multiple environments

    Parameters
    ----------
    nwb_file_name : str
    epoch_names : list[str]
    position_info_param_name : str, optional
    additional_mark_keys : dict, optional

    Returns
    -------
    position_info : pd.DataFrame, shape (n_time, n_columns)
    marks : xr.DataArray, shape (n_time, n_marks, n_electrodes)
    valid_slices : dict[str, list[slice]]
    environment_labels : np.ndarray, shape (n_time,)
    """
    data = []
    environment_labels = []

    for epoch in epoch_names:
        logger.info(epoch)
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
        epoch: valid_slice
        for epoch, valid_slice in zip(epoch_names, valid_slices)
    }

    assert position_info.shape[0] == marks.shape[0]

    return position_info, marks, valid_slices, environment_labels


def populate_mark_indicators(
    spikesorting_selection_keys: dict,
    mark_param_name: str = "default",
    position_info_param_name: str = "default_decoding",
):
    """Populate mark indicators

    Populates for all units in a given spike sorting selection.

    This function is a way to do several pipeline steps at once. It will:
    1. Populate the SpikeSortingSelection table
    2. Populate the SpikeSorting table
    3. Populate the Curation table
    4. Populate the CuratedSpikeSortingSelection table
    5. Populate UnitMarks
    6. Compute UnitMarksIndicator for each position epoch

    Parameters
    ----------
    spikesorting_selection_keys : dict
    mark_param_name : str, optional
    position_info_param_name : str, optional
    """
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
        UnitMarksIndicatorSelection.insert(
            marks_selection, skip_duplicates=True
        )
        UnitMarksIndicator.populate(marks_selection)
