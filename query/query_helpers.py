from math import floor

import numpy as np
import matplotlib.ticker as mticker

import pynwb


# pip install python-intervals, *NOT* pip install intervals
import intervals as iv

def array_from_IntervalSeries(intervalSeries):
    '''Create m x 2 numpy array from the set of intervals in a pynwb.misc.IntervalSeries object'''
    if not isinstance(intervalSeries, pynwb.misc.IntervalSeries):
        raise TypeError("'intervalSeries' parameter must be of type pynwb.misc.IntervalSeries")
    assert np.all(np.abs(intervalSeries.data)==1), "Multiple interval types in an IntervalSeries not supported"
    return np.reshape(intervalSeries.timestamps, (-1,2))

def array_from_intervals(intervals):
    '''Create m x 2 numpy array from the set of intervals in an interval.Interval object'''
    if isinstance(intervals, iv.AtomicInterval): 
        intervals = iv.Interval(intervals)
    if not isinstance(intervals, iv.Interval):
        raise TypeError("'intervals' parameter must be of type intervals.Interval")
    return np.array([[ai.lower, ai.upper] for ai in intervals])

def intervals_from_array(arr):
    '''Create an interval.Interval from an m x 2 numpy array of interval start/end times'''
    if not (isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2):
        raise TypeError("'arr' must be an m x 2 numpy array")
    ivl = iv.empty()
    for i in arr:
        ivl = ivl | iv.closed(*i)
    return ivl

def interval_list_from_IntervalSeries(intervalSeries):
    '''Create a list of interval.Interval objects, 1 for each interval in a pynwb.misc.IntervalSeries'''
    if not isinstance(intervalSeries, pynwb.misc.IntervalSeries):
        raise TypeError("'intervalSeries' parameter must be of type pynwb.misc.IntervalSeries")
    if isinstance(intervals, iv.AtomicInterval): 
        intervals = iv.Interval(intervals)
    return [iv.closed(*i) for i in array_from_IntervalSeries(intervalSeries)]

def intervals_from_IntervalSeries(intervalSeries):
    '''Create an interval.Interval from the union of all intervals in an pynwb.misc.IntervalSeries'''
    if not isinstance(intervalSeries, pynwb.misc.IntervalSeries):
        raise TypeError("'intervalSeries' parameter must be of type pynwb.misc.IntervalSeries")
    return intervals_from_array(array_from_IntervalSeries(intervalSeries))

def times_in_intervals(times, intervals, return_indices=False):
    '''Return list of times that are contained by a list of intervals (interval.Interval object)'''
    if isinstance(intervals, iv.AtomicInterval): 
        intervals = iv.Interval(intervals)
    if not isinstance(intervals, iv.Interval):
        raise TypeError("'intervals' parameter must be of type intervals.Interval")
    if not return_indices:
        return [t for t in times if t in intervals]
    else:
        return [i for i, t in enumerate(times) if t in intervals]
        
def intervals_from_continuous(data, timestamps, fn):
    '''Get Intervals in a continous data series when a specified function of the data evaluates to true.
    Currently returns the first sample after a crossing, and the last sample before a crossing as the 
    bounds of the interval (NB this means if a signal goes high for one sample, you will get a singleton).
    '''
    
    # TODO: think about how to implement interpolation
    # TODO: flag for inner/outer
    if not (isinstance(data, np.ndarray) and data.ndim == 1):
        raise TypeError("'data' must be a 1-dimensonal array")

    if not (isinstance(timestamps, np.ndarray) and timestamps.shape == data.shape):
        raise TypeError("'timestamps' must be same size as data")

        
    if not np.size(data):
        return iv.empty()
        
    fn_data = fn(data)
    assert fn_data.dtype == np.bool_
    datxi = np.where(np.diff(fn_data)) # get indices of crossings

    # Normalize datxi so that there are an even number of entries, with the the first 
    # element (if any) being an interval start, and subsequent entries alternating between
    # starts/ends

    # if data begins while function is true, include this as an interval
    if fn_data[0]:
        datxi = np.insert(datxi,0,-1) # later we'll increment this to 0

    # if data ends while function is true, include this as an interval    
    if fn_data[-1]:
        datxi = np.append(datxi, np.size(fn_data)-1)

    intervals = np.reshape(timestamps[datxi],(-1,2)) 
    
    # we 
    intervals[:,0] = intervals[:,0]+1 # increment starts, since we want first sample *after* crossing

    return(intervals_from_array(intervals))

## Plotting

def plot_pointprocess(intervals, times, ypos=1, axis=None, interval_height=25, tick_height=10, color='b'):
    '''Plot a Point Process (events + their enclosing intervals)'''
    ivl_arr = array_from_intervals(intervals).T
    
    if not axis:
        axis = plt.axes()
    
    int_h = axis.plot(ivl_arr, np.full(ivl_arr.shape, ypos),
                     color=color,
                     linewidth=interval_height,
                     marker='',
                     alpha=0.1,
                     solid_capstyle='butt')
    times_h = axis.plot(times, np.full(times.shape, ypos),
                       color=color,
                       marker='|', 
                       markersize=tick_height, 
                       linestyle='')
    return int_h, times_h

# matplotlib ticklabel formatter
# e.g.:
# ax1.xaxis.set_major_formatter(fmt_truncate_posix)

def fmt_truncate_posix (x, pos):
    oom = 6
    #     offset_str = "%de%d + \n" % (x // 10 ** oom, oom)
    ellipsis = "\u2026"
#    offset_str = "%d" % (x // 10 ** oom)
    offset_str = format(floor(x // 10 ** oom), ",d")
    # oom zero-padded digits before decimal point and up to 6 digits past it 
    # (no trailing zeros)
    # NB no way to do this with single format string: %05.11g gives too many digits past 
    #the decimal when integer part is small.
#     remainder_str = ("%%0%dd" % oom) % (x % 10 ** oom) + \
#                     ("%0.6g" % (x % 1))[1:] # omit leading 0
    remainder_str = format(floor(x % 10 ** oom), "07,d") + \
                    ("%0.6g" % (x % 1))[1:] # omit leading 0

    if pos == 0: # first visible tick
        print (x)
#         return offset_str + ellipsis + "\n" + ellipsis + remainder_str
#         return offset_str + "," + remainder_str
        return offset_str + "," + remainder_str

    else:
        return "\u2026" + remainder_str