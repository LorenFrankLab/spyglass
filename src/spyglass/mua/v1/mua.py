import datajoint as dj
import numpy as np
from ripple_detection import multiunit_HSE_detector

from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position import PositionOutput  # noqa: F401
from spyglass.spikesorting.analysis.v1.group import (
    SortedSpikesGroup,
)  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("mua_v1")


@schema
class MuaEventsParameters(SpyglassMixin, dj.Manual):
    """Params to extract times of high mulitunit activity during immobility."""

    definition = """
    mua_param_name : varchar(80) # a name for this set of parameters
    ----
    mua_param_dict : BLOB    # dictionary of parameters
    """
    contents = [
        {
            "mua_param_name": "default",
            "mua_param_dict": {
                "minimum_duration": 0.015,  # seconds
                "zscore_threshold": 2.0,
                "close_event_threshold": 0.0,  # seconds
                "speed_threshold": 4.0,  # cm/s
            },
        },
    ]

    @classmethod
    def insert_default(cls):
        """Insert the default parameter set"""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class MuaEventsV1(SpyglassMixin, dj.Computed):
    definition = """
    -> MuaEventsParameters
    -> SortedSpikesGroup
    -> PositionOutput.proj(pos_merge_id='merge_id')
    -> IntervalList.proj(artifact_interval_list_name='interval_list_name') # exclude artifact times
    ---
    -> AnalysisNwbfile
    mua_times_object_id : varchar(40)
    """

    def make(self, key):
        # TODO: exclude artifact times
        position_info = (
            PositionOutput & {"merge_id": key["pos_merge_id"]}
        ).fetch1_dataframe()
        speed_name = (
            "speed" if "speed" in position_info.columns else "head_speed"
        )
        speed = position_info[speed_name].to_numpy()
        time = position_info.index.to_numpy()

        spike_indicator = SortedSpikesGroup.get_spike_indicator(key, time)
        spike_indicator = spike_indicator.sum(axis=1, keepdims=True)

        sampling_frequency = 1 / np.median(np.diff(time))

        mua_params = (MuaEventsParameters & key).fetch1("mua_param_dict")

        # Exclude artifact times
        # Alternatively could set to NaN and leave them out of the firing rate calculation
        # in the multiunit_HSE_detector function
        artifact_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["artifact_interval_list_name"],
        }
        artifact_times = (IntervalList & artifact_key).fetch1("valid_times")
        mean_n_spikes = np.mean(spike_indicator)
        for artifact_time in artifact_times:
            spike_indicator[
                np.logical_and(
                    time >= artifact_time.start, time <= artifact_time.stop
                )
            ] = mean_n_spikes

        mua_times = multiunit_HSE_detector(
            time, spike_indicator, speed, sampling_frequency, **mua_params
        )
        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        nwb_file_name = (SortedSpikesGroup & key).fetch1("nwb_file_name")
        key["analysis_file_name"] = nwb_analysis_file.create(nwb_file_name)
        key["mua_times_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=mua_times,
        )
        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)

    def fetch1_dataframe(self):
        """Convenience function for returning the marks in a readable format"""
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [data["mua_times"] for data in self.fetch_nwb()]
