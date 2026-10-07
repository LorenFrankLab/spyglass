import datajoint as dj
import numpy as np
import sortingview.views as vv
from pandas import DataFrame
from scipy.stats import zscore

from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.mua.v1._detection import (
    detect_multiunit_events_in_observed_runs,
)
from spyglass.position import PositionOutput  # noqa: F401
from spyglass.spikesorting.analysis.v1.group import (
    SortedSpikesGroup,
)  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.spikesorting import contiguous_observed_runs

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

        Events are detected within each contiguous run of observed samples,
        using the steps of ripple_detection's multiunit_HSE_detector. An
        event therefore cannot span time the group's units were not
        observed over: the detection interval and the units' observation
        intervals both bound it.
        """
        speed = self.get_speed(key)
        time = speed.index.to_numpy()
        speed = speed.to_numpy()

        spike_indicator, observed = SortedSpikesGroup.get_spike_indicator(
            key, time, return_validity=True
        )
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
        mask = mask & observed

        mua_times = detect_multiunit_events_in_observed_runs(
            time,
            spike_indicator,
            speed,
            sampling_frequency,
            mask,
            **mua_params,
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
        """Create a FigURL for the MUA detection.

        Bins the group's units were not observed over hold NaN in the firing
        rate. They are z-scored out of the statistics and left out of the
        plotted series, so the rate line breaks over unobserved time instead
        of drawing a value the data does not support.

        The rate is drawn as one line segment per contiguous observed run, so
        no segment spans unobserved time: a single series over the surviving
        samples would join across each gap. All runs share one legend entry.
        """
        key = self.fetch1("KEY")
        speed = self.get_speed(key)
        time = speed.index.to_numpy(dtype=np.float64)
        multiunit_firing_rate = np.asarray(
            self.get_firing_rate(key, time)
        ).squeeze()
        observed = np.isfinite(multiunit_firing_rate)
        if zscore_mua:
            zscored = np.full(multiunit_firing_rate.shape, np.nan)
            zscored[observed] = zscore(multiunit_firing_rate[observed])
            multiunit_firing_rate = zscored

        mua_times = self.fetch1_dataframe()

        multiunit_firing_rate_view = vv.TimeseriesGraph()
        multiunit_firing_rate_view.add_interval_series(
            name="MUA Events",
            t_start=mua_times.start_time.to_numpy(),
            t_end=mua_times.end_time.to_numpy(),
            color=mua_times_color,
        )
        name = "Z-Scored Multiunit Rate" if zscore_mua else "Multiunit Rate"
        runs = contiguous_observed_runs(
            time,
            observed,
            sampling_frequency=1 / np.median(np.diff(time)),
        )
        plotted_rate = np.asarray(multiunit_firing_rate, dtype=np.float32)
        for run_number, run in enumerate(runs, start=1):
            if run_number == 1:
                # Let SortingView establish its shared time offset before
                # downcasting timestamps, and give the rate one legend entry.
                multiunit_firing_rate_view.add_line_series(
                    name=name,
                    t=time[run],
                    y=plotted_rate[run],
                    color=mua_color,
                    width=1,
                )
                time_offset = multiunit_firing_rate_view.to_dict()["timeOffset"]
                continue

            # Dataset names must be unique; empty titles omit subsequent
            # runs from the legend without joining their line segments.
            dataset_name = f"{name} ({run_number})"
            multiunit_firing_rate_view.add_dataset(
                vv.TGDataset(
                    name=dataset_name,
                    data={
                        "t": (time[run] - time_offset).astype(np.float32),
                        "y": plotted_rate[run],
                    },
                )
            )
            multiunit_firing_rate_view.add_series(
                vv.TGSeries(
                    type="line",
                    dataset=dataset_name,
                    encoding={"t": "t", "y": "y"},
                    attributes={"color": mua_color, "width": 1},
                )
            )
        if zscore_mua:
            mua_params = (MuaEventsParameters & key).fetch1("mua_param_dict")
            zscore_threshold = mua_params.get("zscore_threshold")
            multiunit_firing_rate_view.add_line_series(
                name="Z-Score Threshold",
                t=np.asarray(time).squeeze(),
                y=np.full(
                    np.asarray(time).squeeze().shape,
                    zscore_threshold,
                    dtype=np.float32,
                ),
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
