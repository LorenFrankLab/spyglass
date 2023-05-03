import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector
from ripple_detection.core import gaussian_smooth, get_envelope
from spyglass.common import (
    Electrode,
    IntervalList,  # noqa
    IntervalPositionInfo,
    IntervalPositionInfoSelection,
    LFPBand,
    LFPBandSelection,
    Session,
)
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("common_ripple")

RIPPLE_DETECTION_ALGORITHMS = {
    "Kay_ripple_detector": Kay_ripple_detector,
    "Karlsson_ripple_detector": Karlsson_ripple_detector,
}


def interpolate_to_new_time(df, new_time, upsampling_interpolation_method="linear"):
    old_time = df.index
    new_index = pd.Index(np.unique(np.concatenate((old_time, new_time))), name="time")
    return (
        df.reindex(index=new_index)
        .interpolate(method=upsampling_interpolation_method)
        .reindex(index=new_time)
    )


@schema
class RippleLFPSelection(dj.Manual):
    definition = """
     -> LFPBand
     group_name = 'CA1' : varchar(80)
     """

    class RippleLFPElectrode(dj.Part):
        definition = """
        -> RippleLFPSelection
        -> LFPBandSelection.LFPBandElectrode
        """

    def insert1(self, key, **kwargs):
        filter_name = (LFPBand & key).fetch1("filter_name")
        if "ripple" not in filter_name.lower():
            raise UserWarning("Please use a ripple filter")
        super().insert1(key, **kwargs)

    @staticmethod
    def set_lfp_electrodes(
        key,
        electrode_list=None,
        group_name="CA1",
        **kwargs,
    ):
        """Removes all electrodes for the specified nwb file and then adds back the electrodes in the list

        Parameters
        ----------
        key : dict
            dictionary corresponding to the LFPBand entry to use for ripple detection
        electrode_list : list
            list of electrodes from LFPBandSelection.LFPBandElectrode
            to be used as the ripple LFP during detection
        group_name : str, optional
            description of the electrode group, by default "CA1"
        """

        RippleLFPSelection().insert1(
            {**key, "group_name": group_name},
            skip_duplicates=True,
            **kwargs,
        )
        if not electrode_list:
            electrode_list = (
                (LFPBandSelection.LFPBandElectrode() & key)
                .fetch("electrode_id")
                .tolist()
            )
        electrode_list.sort()
        electrode_keys = (
            pd.DataFrame(LFPBandSelection.LFPBandElectrode() & key)
            .set_index("electrode_id")
            .loc[electrode_list]
            .reset_index()
            .loc[:, LFPBandSelection.LFPBandElectrode.primary_key]
        )
        electrode_keys["group_name"] = group_name
        electrode_keys = electrode_keys.sort_values(by=["electrode_id"])
        RippleLFPSelection().RippleLFPElectrode.insert(
            electrode_keys.to_dict(orient="records"),
            replace=True,
            **kwargs,
        )


@schema
class RippleParameters(dj.Lookup):
    definition = """
    ripple_param_name : varchar(80) # a name for this set of parameters
    ----
    ripple_param_dict : BLOB    # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default parameter set"""
        default_dict = {
            "speed_name": "head_speed",
            "ripple_detection_algorithm": "Kay_ripple_detector",
            "ripple_detection_params": dict(
                speed_threshold=4.0,  # cm/s
                minimum_duration=0.015,  # sec
                zscore_threshold=2.0,  # std
                smoothing_sigma=0.004,  # sec
                close_ripple_threshold=0.0,  # sec
            ),
        }
        self.insert1(
            {"ripple_param_name": "default", "ripple_param_dict": default_dict},
            skip_duplicates=True,
        )


@schema
class RippleTimes(dj.Computed):
    definition = """
    -> RippleParameters
    -> RippleLFPSelection
    -> IntervalPositionInfo
    ---
    -> AnalysisNwbfile
    ripple_times_object_id : varchar(40)
     """

    def make(self, key):
        print(f"Computing ripple times for: {key}")
        ripple_params = (
            RippleParameters & {"ripple_param_name": key["ripple_param_name"]}
        ).fetch1("ripple_param_dict")

        ripple_detection_algorithm = ripple_params["ripple_detection_algorithm"]
        ripple_detection_params = ripple_params["ripple_detection_params"]

        (
            speed,
            interval_ripple_lfps,
            sampling_frequency,
        ) = self.get_ripple_lfps_and_position_info(key)

        ripple_times = RIPPLE_DETECTION_ALGORITHMS[ripple_detection_algorithm](
            time=np.asarray(interval_ripple_lfps.index),
            filtered_lfps=np.asarray(interval_ripple_lfps),
            speed=np.asarray(speed),
            sampling_frequency=sampling_frequency,
            **ripple_detection_params,
        )

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(key["nwb_file_name"])
        key["ripple_times_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=ripple_times,
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
        """Convenience function for returning the marks in a readable format"""
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [data["ripple_times"] for data in self.fetch_nwb()]

    @staticmethod
    def get_ripple_lfps_and_position_info(key):
        nwb_file_name = key["nwb_file_name"]
        interval_list_name = key["target_interval_list_name"]
        position_info_param_name = key["position_info_param_name"]
        ripple_params = (
            RippleParameters & {"ripple_param_name": key["ripple_param_name"]}
        ).fetch1("ripple_param_dict")

        speed_name = ripple_params["speed_name"]

        electrode_keys = (RippleLFPSelection.RippleLFPElectrode() & key).fetch(
            "electrode_id"
        )

        # warn/validate that there is only one wire per electrode
        lfp_key = key.copy()
        del lfp_key["interval_list_name"]
        ripple_lfp_nwb = (LFPBand & lfp_key).fetch_nwb()[0]
        ripple_lfp_electrodes = ripple_lfp_nwb["filtered_data"].electrodes.data[:]
        elec_mask = np.full_like(ripple_lfp_electrodes, 0, dtype=bool)
        elec_mask[
            [
                ind
                for ind, elec in enumerate(ripple_lfp_electrodes)
                if elec in electrode_keys
            ]
        ] = True
        ripple_lfp = pd.DataFrame(
            ripple_lfp_nwb["filtered_data"].data,
            index=pd.Index(ripple_lfp_nwb["filtered_data"].timestamps, name="time"),
        )
        sampling_frequency = ripple_lfp_nwb["lfp_band_sampling_rate"]

        ripple_lfp = ripple_lfp.loc[:, elec_mask]

        position_valid_times = (
            IntervalList
            & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
        ).fetch1("valid_times")

        position_info = (
            IntervalPositionInfo()
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
                "position_info_param_name": position_info_param_name,
            }
        ).fetch1_dataframe()

        position_info = pd.concat(
            [
                position_info.loc[slice(valid_time[0], valid_time[1])]
                for valid_time in position_valid_times
            ],
            axis=1,
        )
        interval_ripple_lfps = pd.concat(
            [
                ripple_lfp.loc[slice(valid_time[0], valid_time[1])]
                for valid_time in position_valid_times
            ],
            axis=1,
        )

        position_info = interpolate_to_new_time(
            valid_position_info, interval_ripple_lfps.index
        )

        return (
            position_info[speed_name],
            interval_ripple_lfps,
            sampling_frequency,
        )

    @staticmethod
    def get_Kay_ripple_consensus_trace(
        ripple_filtered_lfps, sampling_frequency, smoothing_sigma=0.004
    ):
        ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
        not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)

        ripple_consensus_trace[not_null] = get_envelope(
            np.asarray(ripple_filtered_lfps)[not_null]
        )
        ripple_consensus_trace = np.sum(ripple_consensus_trace**2, axis=1)
        ripple_consensus_trace[not_null] = gaussian_smooth(
            ripple_consensus_trace[not_null], smoothing_sigma, sampling_frequency
        )
        return pd.DataFrame(
            np.sqrt(ripple_consensus_trace), index=ripple_filtered_lfps.index
        )

    @staticmethod
    def plot_ripple_consensus_trace(
        ripple_consensus_trace,
        ripple_times,
        ripple_label=1,
        offset=0.100,
        relative=True,
        ax=None,
    ):
        ripple_start = ripple_times.loc[ripple_label].start_time
        ripple_end = ripple_times.loc[ripple_label].end_time
        time_slice = slice(ripple_start - offset, ripple_end + offset)

        start_offset = ripple_start if relative else 0
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 1))
        ax.plot(
            ripple_consensus_trace.loc[time_slice].index - start_offset,
            ripple_consensus_trace.loc[time_slice],
        )
        ax.axvspan(
            ripple_start - start_offset,
            ripple_end - start_offset,
            zorder=-1,
            alpha=0.5,
            color="lightgrey",
        )
        ax.set_xlabel("Time [s]")
        ax.set_xlim((time_slice.start - start_offset, time_slice.stop - start_offset))

    @staticmethod
    def plot_ripple(
        lfps, ripple_times, ripple_label=1, offset=0.100, relative=True, ax=None
    ):
        lfp_labels = lfps.columns
        n_lfps = len(lfp_labels)
        ripple_start = ripple_times.loc[ripple_label].start_time
        ripple_end = ripple_times.loc[ripple_label].end_time
        time_slice = slice(ripple_start - offset, ripple_end + offset)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, n_lfps * 0.20))

        start_offset = ripple_start if relative else 0

        for lfp_ind, lfp_label in enumerate(lfp_labels):
            lfp = lfps.loc[time_slice, lfp_label]
            ax.plot(
                lfp.index - start_offset,
                lfp_ind + (lfp - lfp.mean()) / (lfp.max() - lfp.min()),
                color="black",
            )

        ax.axvspan(
            ripple_start - start_offset,
            ripple_end - start_offset,
            zorder=-1,
            alpha=0.5,
            color="lightgrey",
        )
        ax.set_ylim((-1, n_lfps))
        ax.set_xlim((time_slice.start - start_offset, time_slice.stop - start_offset))
        ax.set_ylabel("LFPs")
        ax.set_xlabel("Time [s]")
