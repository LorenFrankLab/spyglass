#!/usr/bin/env python3

import numpy as np
import spyglass.common as sgc

nwb_file_name = "RN2_20191110_.nwb"
interval_list_name = "01_r1"

interval_list = (
    sgc.IntervalList
    & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
).fetch1("valid_times")
sort_interval = np.copy(interval_list[0])
sort_interval_name = interval_list_name + "_first180"
sort_interval[1] = sort_interval[0] + 180

sgc.SortInterval.insert1(
    {
        "nwb_file_name": nwb_file_name,
        "sort_interval_name": sort_interval_name,
        "sort_interval": sort_interval,
    }
)

print(sgc.SortInterval & {"nwb_file_name": nwb_file_name})
