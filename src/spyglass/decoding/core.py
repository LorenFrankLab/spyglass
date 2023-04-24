import numpy as np
import pandas as pd
from spyglass.common.common_behav import RawPosition
from spyglass.common.common_interval import IntervalList, interval_list_intersect


def get_valid_ephys_position_times_from_interval(interval_list_name, nwb_file_name):
    interval_valid_times = (
        IntervalList
        & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
    ).fetch1("valid_times")

    position_interval_names = (
        RawPosition
        & {
            "nwb_file_name": nwb_file_name,
        }
    ).fetch("interval_list_name")
    position_interval_names = position_interval_names[
        np.argsort(
            [int(name.strip("pos valid time")) for name in position_interval_names]
        )
    ]
    valid_pos_times = [
        (
            IntervalList
            & {"nwb_file_name": nwb_file_name, "interval_list_name": pos_interval_name}
        ).fetch1("valid_times")
        for pos_interval_name in position_interval_names
    ]

    valid_ephys_times = (
        IntervalList
        & {"nwb_file_name": nwb_file_name, "interval_list_name": "raw data valid times"}
    ).fetch1("valid_times")

    return interval_list_intersect(
        interval_list_intersect(interval_valid_times, valid_ephys_times),
        np.concatenate(valid_pos_times),
    )


def get_epoch_interval_names(nwb_file_name):
    interval_list = pd.DataFrame(IntervalList() & {"nwb_file_name": nwb_file_name})

    interval_list = interval_list.loc[
        interval_list.interval_list_name.str.contains(
            r"^(\d+)_(\w+)$", regex=True, na=False
        )
    ]

    return interval_list.interval_list_name.tolist()


def get_valid_ephys_position_times_by_epoch(nwb_file_name):
    return {
        epoch: get_valid_ephys_position_times_from_interval(epoch, nwb_file_name)
        for epoch in get_epoch_interval_names(nwb_file_name)
    }


def convert_epoch_interval_name_to_position_interval_name(epoch_interval_name):
    pos_interval_number = int(epoch_interval_name.split("_")[0]) - 1
    return f"pos {pos_interval_number} valid times"


def convert_valid_times_to_slice(valid_times):
    return [slice(times[0], times[1]) for times in valid_times]
