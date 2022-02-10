import datajoint as dj
import numpy as np
from numpy.lib import emath
from functools import reduce

from .common_session import Session  # noqa: F401

schema = dj.schema('common_interval')

# TODO: ADD export to NWB function to save relevant intervals in an NWB file

@schema
class IntervalList(dj.Manual):
    definition = """
    # Time intervals used for analysis
    -> Session
    interval_list_name: varchar(200) # descriptive name of this interval list
    ---
    valid_times: longblob # numpy array with start and end times for each interval
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, *, nwb_file_name):
        """Add each entry in the NWB file epochs table to the IntervalList table.

        The interval list name for each epoch is set to the first tag for the epoch.
        If the epoch has no tags, then 'interval_x' will be used as the interval list name, where x is the index
        (0-indexed) of the epoch in the epochs table.
        The start time and stop time of the epoch are stored in the valid_times field as a numpy array of
        [start time, stop time] for each epoch.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        nwb_file_name : str
            The file name of the NWB file, used as a primary key to the Session table.
        """
        epochs = nwbf.epochs.to_dataframe()
        epoch_dict = dict()
        epoch_dict['nwb_file_name'] = nwb_file_name
        for e in epochs.iterrows():
            epoch_dict['interval_list_name'] = e[1].tags[0]
            epoch_dict['valid_times'] = np.asarray(
                [[e[1].start_time, e[1].stop_time]])
            cls.insert1(epoch_dict, skip_duplicates=True)

@schema
class SortInterval(dj.Manual):
    definition = """
    -> Session
    sort_interval_name: varchar(200) # name for this interval
    ---
    sort_interval: longblob # 1D numpy array with start and end time for a single interval to be used for spike sorting
    """


# TODO: make all of the functions below faster if possible
def intervals_by_length(interval_list, min_length=0.0, max_length=1e10):
    """Returns an interval list with only the intervals whose length is > min_length and < max_length

    Args:
        interval_list ((N,2) np.array): input interval list.
        min_length (float, optional): [minimum interval length in seconds]. Defaults to 0.0.
        max_length ([type], optional): [maximum interval length in seconds]. Defaults to 1e10.
    """
    # get the length of each interval
    lengths = np.ravel(np.diff(interval_list))
    # return only intervals of the appropriate lengths
    return interval_list[np.logical_and(lengths > min_length, lengths < max_length)]

def interval_list_contains_ind(valid_times, timestamps):
    """Returns the indices for the timestamps that are contained within the valid_times intervals

    :param valid_times: Array of [start, end] times
    :type valid_times: numpy array
    :param timestamps: list of timestamps
    :type timestamps: numpy array or list
    :return: indices of timestamps that are in one of the valid_times intervals
    """
    ind = []
    for valid_time in valid_times:
        ind += np.ravel(np.argwhere(np.logical_and(timestamps >= valid_time[0],
                                                   timestamps <= valid_time[1]))).tolist()
    return np.asarray(ind)


def interval_list_contains(valid_times, timestamps):
    """Returns the timestamps that are contained within the valid_times intervals

    :param valid_times: Array of [start, end] times
    :type valid_times: numpy array
    :param timestamps: list of timestamps
    :type timestamps: numpy array or list
    :return: numpy array of timestamps that are in one of the valid_times intervals
    """
    ind = []
    for valid_time in valid_times:
        ind += np.ravel(np.argwhere(np.logical_and(timestamps >= valid_time[0],
                                                   timestamps <= valid_time[1]))).tolist()
    return timestamps[ind]


def interval_list_excludes_ind(valid_times, timestamps):
    """Returns the indices of the timestamps that are excluded from the valid_times intervals

    :param valid_times: Array of [start, end] times
    :type valid_times: numpy array
    :param timestamps: list of timestamps
    :type timestamps: numpy array or list
    :return: numpy array of timestamps that are in one of the valid_times intervals
    """
    # add the first and last times to the list and creat a list of invalid intervals
    valid_times_list = np.ndarray.ravel(valid_times).tolist()
    valid_times_list.insert(0, timestamps[0] - 0.00001)
    valid_times_list.append(timestamps[-1] + 0.001)
    invalid_times = np.array(valid_times_list).reshape(-1, 2)
    # add the first and last timestamp indices
    ind = []
    for invalid_time in invalid_times:
        ind += np.ravel(np.argwhere(np.logical_and(timestamps > invalid_time[0],
                                                   timestamps < invalid_time[1]))).tolist()
    return np.asarray(ind)


def interval_list_excludes(valid_times, timestamps):
    """Returns the indices of the timestamps that are excluded from the valid_times intervals

    :param valid_times: Array of [start, end] times
    :type valid_times: numpy array
    :param timestamps: list of timestamps
    :type timestamps: numpy array or list
    :return: numpy array of timestamps that are in one of the valid_times intervals
    """
    # add the first and last times to the list and creat a list of invalid intervals
    valid_times_list = np.ravel(valid_times).tolist()
    valid_times_list.insert(0, timestamps[0] - 0.00001)
    valid_times_list.append(timestamps[-1] + 0.00001)
    invalid_times = np.array(valid_times_list).reshape(-1, 2)
    # add the first and last timestamp indices
    ind = []
    for invalid_time in invalid_times:
        ind += np.ravel(np.argwhere(np.logical_and(timestamps > invalid_time[0],
                                                   timestamps < invalid_time[1]))).tolist()
    return timestamps[ind]

def interval_list_intersect(interval_list1, interval_list2):
    """Finds the intersections between two interval lists

    Parameters
    ----------
    interval_list1 : np.array, (N,2) where N = number of intervals
    interval_list2 : np.array, (N,2) where N = number of intervals

    Each interval is (start time, stop time)
    
    Returns
    -------
    interval_list: np.array, (N,2)
    """
    
    if interval_list1.ndim==1:
        interval_list1 = np.expand_dims(interval_list1,0)
    else:
        interval_list1 = interval_list1[np.argsort(interval_list1[:,0])]
        interval_list1 = reduce(_union_concat, interval_list1)
        
    if interval_list2.ndim==1:
        interval_list2 = np.expand_dims(interval_list2,0)
    else:
        interval_list2 = interval_list2[np.argsort(interval_list2[:,0])]
        interval_list2 = reduce(_union_concat, interval_list2)

    intersecting_intervals = []
    for interval2 in interval_list2:
        for interval1 in interval_list1:
            if _intersection(interval2, interval1) is not None:
                intersecting_intervals.append(_intersection(interval1, interval2))   
                
    intersecting_intervals = np.asarray(intersecting_intervals)
    intersecting_intervals = intersecting_intervals[np.argsort(intersecting_intervals[:,0])]
    
    return intersecting_intervals

def _intersection(interval1, interval2):
    """Takes (set-theoretic) intersection of two intervals
    """
    intersection = np.array([max([interval1[0],interval2[0]]),
                             min([interval1[1],interval2[1]])])
    if intersection[1]>intersection[0]:
        return intersection
    else:
        return None
    
def _union(interval1, interval2):
    """Take (set-theoretic) union of two intervals
    """
    if _intersection(interval1, interval2) is None:
        return np.array([interval1, interval2])
    else:
        return np.array([min([interval1[0],interval2[0]]),
                         max([interval1[1],interval2[1]])])

def _union_concat(interval1, interval2):
    """Given two intervals, takes their union if overlapping;
    concatenates if not
    
    This is recursively called with `reduce`.
    Interval1 is usually a list of intervals; interval2 is a single interval
    """
    if interval1.ndim==1:
        interval1 = np.expand_dims(interval1, 0)
    if interval2.ndim==1:
        interval2 = np.expand_dims(interval2, 0)

    x = _union(interval1[-1], interval2[0])
    if x.ndim==1:
        x = np.expand_dims(x, 0)
    return np.concatenate((interval1[:-1], x), axis=0)

def union_adjacent_index(interval1, interval2):
    """unions two intervals that are adjacent in index
    e.g. [a,b] and [b+1, c] is converted to [a,c]
    if not adjacent, just concatenates interval2 at the end of interval1

    Parameters
    ----------
    interval1 : np.array
        [description]
    interval2 : np.array
        [description]
    """
    if interval1.ndim==1:
        interval1 = np.expand_dims(interval1, 0)
    if interval2.ndim==1:
        interval2 = np.expand_dims(interval2, 0)

    if interval1[-1][1]+1 == interval2[0][0] or interval2[0][1]+1 == interval1[-1][0]:
        x = np.array([[np.min([interval1[-1][0],interval2[0][0]]), 
                       np.max([interval1[-1][1],interval2[0][1]])]])
        return np.concatenate((interval1[:-1], x), axis=0)
    else:
        return np.concatenate((interval1, interval2),axis=0)

# TODO: test interval_list_union code
def interval_list_union(interval_list1, interval_list2, min_length=0.0, max_length=1e10):
    """Finds the union (all times in one or both) for two interval lists

    :param interval_list1: The first interval list
    :type interval_list1: numpy array of intervals [start, stop]
    :param interval_list2: The second interval list
    :type interval_list2: numpy array of intervals [start, stop]
    :param min_length: optional minimum length of interval for inclusion in output, default 0.0
    :type min_length: float
    :param max_length: optional maximum length of interval for inclusion in output, default 1e10
    :type max_length: float
    :return: interval_list
    :rtype:  numpy array of intervals [start, stop]
    """
    # return np.array([min(interval_list1[0],interval_list2[0]),
    #                  max(interval_list1[1],interval_list2[1])])
    interval_list1 = np.ravel(interval_list1)
    # create a parallel list where 1 indicates the start and -1 the end of an interval
    interval_list1_start_end = np.ones(interval_list1.shape)
    interval_list1_start_end[1::2] = -1

    interval_list2 = np.ravel(interval_list2)
    # create a parallel list for the second interval where 1 indicates the start and -1 the end of an interval
    interval_list2_start_end = np.ones(interval_list2.shape)
    interval_list2_start_end[1::2] = -1

    # concatenate the two lists so we can resort the intervals and apply the same sorting to the start-end arrays
    combined_intervals = np.concatenate((interval_list1, interval_list2))
    ss = np.concatenate((interval_list1_start_end, interval_list2_start_end))
    sort_ind = np.argsort(combined_intervals)
    combined_intervals = combined_intervals[sort_ind]
    # a cumulative sum of 1 indicates the beginning of a joint interval; a cumulative sum of 0 indicates the end
    union_starts = np.ravel(np.array(np.where(np.cumsum(ss[sort_ind]) == 1)))
    union_stops = np.ravel(np.array(np.where(np.cumsum(ss[sort_ind]) == 0)))
    union = []
    for start, stop in zip(union_starts, union_stops):
        union.append([combined_intervals[start], combined_intervals[stop]])
    return np.asarray(union)

def interval_list_censor(interval_list, timestamps):
    """returns a new interval list that starts and ends at the first and last timestamp

    Args:
        interval_list (numpy array of intervals [start, stop]): interval list from IntervalList valid times
        timestamps (numpy array or list): timestamp list

    Returns:
        interval_list (numpy array of intervals [start, stop])
    """
    # check that all timestamps are in the interval list
    assert len(interval_list_contains_ind(interval_list, timestamps)) == len(timestamps), 'interval_list must contain all timestamps' 
    
    timestamps_interval = np.asarray([[timestamps[0], timestamps[-1]]])
    return interval_list_intersect(interval_list, timestamps_interval)    
