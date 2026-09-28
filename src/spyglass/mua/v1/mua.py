import datajoint as dj
import numpy as np
import sortingview.views as vv
from pandas import DataFrame
from ripple_detection import multiunit_HSE_detector
from scipy.stats import zscore

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
    """Params to extract times of high multiunit activity during immobility.

    Attributes
    ----------
    mua_param_name : str
        A name for this set of parameters
    mua_param_dict : dict
        Dictionary of parameters, including...
            minimum_duration : float
                Minimum duration of MUA event (seconds)
            zscore_threshold : float
                Z-score threshold for MUA detection
            close_event_threshold : float
                Minimum time between MUA events (seconds)
            speed_threshold : float
                Minimum speed for MUA detection (cm/s)
    """

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
    -> IntervalList.proj(detection_interval='interval_list_name')
    ---
    -> AnalysisNwbfile
    mua_times_object_id : varchar(40)
    """

    def make(self, key):
        """Populates the MuaEventsV1 table.

        Fetches...
            - Speed from PositionOutput
            - Spike indicator from SortedSpikesGroup
            - Valid times from IntervalList
            - Parameters from MuaEventsParameters
        Uses multiunit_HSE_detector from ripple_detection package to detect
        multiunit activity.
        """
        speed = self.get_speed(key)
        time = speed.index.to_numpy()
        speed = speed.to_numpy()

        spike_indicator = SortedSpikesGroup.get_spike_indicator(key, time)
        spike_indicator = spike_indicator.sum(axis=1, keepdims=True)

        sampling_frequency = 1 / np.median(np.diff(time))

        mua_params = (MuaEventsParameters & key).fetch1("mua_param_dict")

        valid_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["detection_interval"],
            }
        ).fetch1("valid_times")
        mask = np.zeros_like(time, dtype=bool)
        for start, end in valid_times:
            mask = mask | ((time >= start) & (time <= end))

        time = time[mask]
        speed = speed[mask]
        spike_indicator = spike_indicator[mask]

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
        _ = self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> list[DataFrame]:
        """Fetch the MUA times as a list of dataframes"""
        return [data["mua_times"] for data in self.fetch_nwb()]

    @classmethod
    def get_firing_rate(cls, key, time):
        """Get the firing rate of the multiunit activity"""
        return SortedSpikesGroup.get_firing_rate(key, time, multiunit=True)

    @staticmethod
    def get_speed(key):
        """Get the speed of the animal during the recording."""
        position_info = (
            PositionOutput & {"merge_id": key["pos_merge_id"]}
        ).fetch1_dataframe()
        speed_name = (
            "speed" if "speed" in position_info.columns else "head_speed"
        )
        return position_info[speed_name]

    def create_figurl(
        self,
        zscore_mua=True,
        mua_times_color="red",
        speed_color="black",
        mua_color="black",
        view_height=800,
    ):
        """Create a FigURL for the MUA detection."""
        key = self.fetch1("KEY")
        speed = self.get_speed(key)
        time = speed.index.to_numpy()
        multiunit_firing_rate = self.get_firing_rate(key, time)
        if zscore_mua:
            multiunit_firing_rate = zscore(multiunit_firing_rate)

        mua_times = self.fetch1_dataframe()

        multiunit_firing_rate_view = vv.TimeseriesGraph()
        multiunit_firing_rate_view.add_interval_series(
            name="MUA Events",
            t_start=mua_times.start_time.to_numpy(),
            t_end=mua_times.end_time.to_numpy(),
            color=mua_times_color,
        )
        name = "Z-Scored Multiunit Rate" if zscore_mua else "Multiunit Rate"
        multiunit_firing_rate_view.add_line_series(
            name=name,
            t=np.asarray(time),
            y=np.asarray(multiunit_firing_rate, dtype=np.float32),
            color=mua_color,
            width=1,
        )
        if zscore_mua:
            mua_params = (MuaEventsParameters & key).fetch1("mua_param_dict")
            zscore_threshold = mua_params.get("zscore_threshold")
            multiunit_firing_rate_view.add_line_series(
                name="Z-Score Threshold",
                t=np.asarray(time).squeeze(),
                y=np.ones_like(
                    multiunit_firing_rate, dtype=np.float32
                ).squeeze()
                * zscore_threshold,
                color=mua_times_color,
                width=1,
            )
        speed_view = vv.TimeseriesGraph().add_line_series(
            name="Speed [cm/s]",
            t=np.asarray(time),
            y=np.asarray(speed, dtype=np.float32),
            color=speed_color,
            width=1,
        )
        speed_view.add_interval_series(
            name="MUA Events",
            t_start=mua_times.start_time.to_numpy(),
            t_end=mua_times.end_time.to_numpy(),
            color=mua_times_color,
        )
        vertical_panel_content = [
            vv.LayoutItem(
                multiunit_firing_rate_view, stretch=2, title="Multiunit"
            ),
            vv.LayoutItem(speed_view, stretch=2, title="Speed"),
        ]

        view = vv.Box(
            direction="horizontal",
            show_titles=True,
            height=view_height,
            items=[
                vv.LayoutItem(
                    vv.Box(
                        direction="vertical",
                        show_titles=True,
                        items=vertical_panel_content,
                    )
                ),
            ],
        )

        return view.url(label="Multiunit Detection")
