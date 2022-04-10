import numpy as np
from spyglass.common.common_behav import RawPosition
from spyglass.common.common_interval import (IntervalList,
                                             interval_list_intersect)


def get_valid_ephys_position_times_from_interval(
        interval_list_name,
        nwb_file_name
):

    interval_valid_times = (IntervalList &
                            {'nwb_file_name': nwb_file_name,
                             'interval_list_name': interval_list_name}
                            ).fetch1('valid_times')

    position_interval_names = (RawPosition & {
        'nwb_file_name': nwb_file_name,
    }).fetch('interval_list_name')
    position_interval_names = position_interval_names[np.argsort(
        [int(name.strip('pos valid time')) for name in position_interval_names])]
    valid_pos_times = [(IntervalList &
                       {'nwb_file_name': nwb_file_name,
                        'interval_list_name': pos_interval_name}
                        ).fetch1('valid_times')
                       for pos_interval_name in position_interval_names]

    valid_ephys_times = (IntervalList &
                         {'nwb_file_name': nwb_file_name,
                          'interval_list_name': 'raw data valid times'}
                         ).fetch1('valid_times')

    return interval_list_intersect(
        interval_list_intersect(interval_valid_times, valid_ephys_times),
        np.concatenate(valid_pos_times))
