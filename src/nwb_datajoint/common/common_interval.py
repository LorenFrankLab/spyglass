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
    # Time intervals with data
    -> Session
    interval_list_name: varchar(200) # descriptive name of this interval list
    ---
    valid_times: longblob # numpy array with start and end times for each interval
    """

    def insert_from_nwbfile(self, nwbf, *, nwb_file_name):
        '''
        :param nwbf:
        :param nwb_file_name:
        :return: None
        Adds each of the entries in the nwb epochs table to the Interval list
        '''
        epochs = nwbf.epochs.to_dataframe()
        epoch_dict = dict()
        epoch_dict['nwb_file_name'] = nwb_file_name
        for e in epochs.iterrows():
            epoch_dict['interval_list_name'] = e[1].tags[0]
            epoch_dict['valid_times'] = np.asarray(
                [[e[1].start_time, e[1].stop_time]])
            self.insert1(epoch_dict, skip_duplicates=True)


@schema
class SortInterval(dj.Manual):
    definition = """
    -> Session
    sort_interval_name: varchar(200) # name for this interval
    ---
    sort_interval: longblob # 1D numpy array with start and end time for a single interval to be used for spike sorting
    """


# TODO: make all of the functions below faster if possible
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
    """Finds the intersection (overlapping times) for two interval lists

    Parameters
    ----------
    interval_list1: np.array, (N,2) where N = number of intervals
    interval_list2: np.array, (N,2) where N = number of intervals

    Each interval is (start time, stop time)
    
    Returns
    -------
    interval_list: np.array, (N,2)
    """
    
    if interval_list1.ndim==1:
        interval_list1 = np.expand_dims(interval_list1,0)
    else:
        interval_list1 = interval_list1[np.argsort(interval_list1[:,0])]
        interval_list1 = reduce(_union_overlapping_intervals, interval_list1)
        
    if interval_list2.ndim==1:
        interval_list2 = np.expand_dims(interval_list2,0)
    else:
        interval_list2 = interval_list2[np.argsort(interval_list2[:,0])]
        interval_list2 = reduce(_union_overlapping_intervals, interval_list2)

    intersecting_intervals = []
    for interval2 in interval_list2:
        for interval1 in interval_list1:
            if _overlapping(interval2, interval1):
                intersecting_intervals.append(_intersection(interval1, interval2))   
                
    intersecting_intervals = np.array(intersecting_intervals)
    intersecting_intervals = intersecting_intervals[np.argsort(intersecting_intervals[:,0])]
    
    return intersecting_intervals

def _intersection(interval1, interval2):
    """Take intersection of the two intervals
    """
    return [np.max([interval1[0],interval2[0]]), np.min([interval1[1],interval2[1]])]
    
def _union(interval1, interval2):
    """Take union of the two intervals
    """
    return [np.min([interval1[0],interval2[0]]), np.max([interval1[1],interval2[1]])]

def _overlapping(interval1, interval2):
    """Checks if the two intervals overlap
    """
    x = _intersection(interval1, interval2)
    return x[1] > x[0]

def _union_overlapping_intervals(interval1, interval2):
    """Given two intervals, takes their union if overlapping;
    concatenates if not
    """
    if interval1.ndim==1:
        interval1 = np.expand_dims(interval1, 0)
    if interval2.ndim==1:
        interval2 = np.expand_dims(interval2, 0)

    if _overlapping(interval1[-1], interval2[0]):
        x = np.array([_union(interval1[-1], interval2[0])])
        return np.concatenate((interval1[:-1], x), axis=0)
    else:
        return np.concatenate((interval1, interval2),axis=0)

# def interval_list_intersect(interval_list1, interval_list2):
#     """
#     Finds the intersection (overlapping times) for two interval lists

#     Parameters
#     ----------
#     interval_list1: np.array, (N,2) where N = number of intervals
#         first element is start time; second element is stop time
#     interval_list2: np.array, (N,2) where N = number of intervals
#         first element is start time; second element is stop time

#     Returns
#     -------
#     interval_list: np.array, (2,)
#     """
 
#     interval_list1 = np.ravel(interval_list1)
#     # create a parallel list where 1 indicates the start and -1 the end of an interval
#     interval_list1_start_end = np.ones(interval_list1.shape)
#     interval_list1_start_end[1::2] = -1

#     interval_list2 = np.ravel(interval_list2)
#     # create a parallel list for the second interval where 2 indicates the start and -2 the end of an interval
#     interval_list2_start_end = np.ones(interval_list2.shape) * 2
#     interval_list2_start_end[1::2] = -2

#     # concatenate the two lists so we can resort the intervals and apply the same sorting to the start-stop arrays
#     combined_intervals = np.concatenate((interval_list1, interval_list2))
#     ss = np.concatenate((interval_list1_start_end, interval_list2_start_end))
#     sort_ind = np.argsort(combined_intervals)
#     combined_intervals = combined_intervals[sort_ind]
#     # a cumulative sum of 3 indicates the beginning of a joint interval, and the following element is the end
#     intersection_starts = np.ravel(
#         np.array(np.where(np.cumsum(ss[sort_ind]) == 3)))
#     intersection_stops = intersection_starts + 1
#     intersect = []
#     for start, stop in zip(intersection_starts, intersection_stops):
#         intersect.append([combined_intervals[start], combined_intervals[stop]])
#     return np.asarray(intersect)


# TODO: test interval_list_union code
def interval_list_union(interval_list1, interval_list2):
    """Finds the union (all times in one or both) for two interval lists

    :param interval_list1: The first interval list
    :type interval_list1: numpy array of intervals [start, stop]
    :param interval_list2: The second interval list
    :type interval_list2: numpy array of intervals [start, stop]
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