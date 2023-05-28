import uuid

import datajoint as dj
import numpy as np
import pandas as pd
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)

from spyglass.common import AnalysisNwbfile, IntervalList
from spyglass.position.position_merge import PositionOutput
from spyglass.spikesorting.spikesorting_merge import CuratedSpikeSortingOutput
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("high_multiunit_detection_v1")


def get_time_bins_from_interval(interval_times, sampling_rate):
    """Gets the superset of the interval."""
    start_time, end_time = interval_times[0][0], interval_times[-1][-1]
    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    return np.linspace(start_time, end_time, n_samples)


def convert_to_spike_indicator(spikes_times_nwb, interval_times, sampling_rate):
    time = get_time_bins_from_interval(interval_times, sampling_rate)

    # restrict to cases with units
    spikes_times_nwb = [entry for entry in spikes_times_nwb if "units" in entry]
    spike_times_list = [
        np.asarray(n_trode["units"]["spike_times"]) for n_trode in spikes_times_nwb
    ]
    if len(spike_times_list) > 0:  # if units
        spikes = np.concatenate(spike_times_list)

        # Bin spikes into time bins
        spike_indicator = []
        for spike_times in spikes:
            spike_times = spike_times[
                (spike_times > time[0]) & (spike_times <= time[-1])
            ]
            spike_indicator.append(
                np.bincount(
                    np.digitize(spike_times, time[1:-1]), minlength=time.shape[0]
                )
            )

        column_names = np.concatenate(
            [
                [
                    f'{n_trode["sort_group_id"]:04d}_{unit_number:04d}'
                    for unit_number in n_trode["units"].index
                ]
                for n_trode in spikes_times_nwb
            ]
        )
        return pd.DataFrame(
            np.stack(spike_indicator, axis=1),
            index=pd.Index(time, name="time"),
            columns=column_names,
        )


@schema
class MultiunitFiringRate(dj.Computed):
    """Computes the population multiunit firing rate."""

    definition = """
    -> IntervalList
    -> CuratedSpikeSortingOutput
    sampling_rate = 500 : int
    ---
    -> AnalysisNwbfile
    multiunit_firing_rate_object_id: varchar(40)
    """

    def make(self, key):
        interval_times = (IntervalList & key).fetch1("valid_times")
        spikes_times_nwb = (CuratedSpikeSortingOutput() & key).fetch_nwb()[0]
        spike_indicator = convert_to_spike_indicator(
            spikes_times_nwb, interval_times, key["sampling_rate"]
        )

        multiunit_firing_rate = pd.DataFrame(
            get_multiunit_population_firing_rate(spike_indicator, key["sampling_rate"]),
            index=spike_indicator.index,
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
class HighMultiunitTimesParameters(dj.Manual):
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
class HighMultiunitTimesV1(dj.Computed):
    """Finds times of high mulitunit activity during immobility."""

    definition = """
    -> HighMultiunitTimesParameters
    -> CuratedSpikeSortingOutput
    -> PositionOutput
    ---
    -> AnalysisNwbfile
    high_multiunit_times_object_id: varchar(40)
    """

    def make(self, key):
        interval_times = (IntervalList & key).fetch1("valid_times")
        spikes_times_nwb = (CuratedSpikeSortingOutput() & key).fetch_nwb()[0]
        spike_indicator = convert_to_spike_indicator(
            spikes_times_nwb, interval_times, key["sampling_rate"]
        )

        position_info = (PositionOutput() & key).fetch1_dataframe()

        params = (HighMultiunitTimesParameters & key).fetch1()

        multiunit_high_synchrony_times = multiunit_HSE_detector(
            np.asarray(spike_indicator.index),
            spike_indicator,
            position_info.head_speed.values,
            sampling_frequency=key["sampling_rate"],
            **params,
        )

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(key["nwb_file_name"])

        key["high_multiunit_times_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=multiunit_high_synchrony_times.reset_index(),
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        from spyglass.high_multiunit_detection.high_multiunit_detection_merge import (
            HighMultiunitTimesOutput,
        )

        output_key = {
            "high_multiunit_times_id": uuid.uuid1(),
            "source": "HighMultiunitTimes",
            "version": 1,
        }
        HighMultiunitTimesOutput.insert1(output_key)

        HighMultiunitTimesOutput.HighMultiunitTimesV1.insert1(key)
