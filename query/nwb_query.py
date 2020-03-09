#NOTE 3-2020;  We should probably use the pyintervals package to represent
#intervals. 


# ------------------------------------------------
#   FRANK LAB NWB QUERY PROTOTYPES
# ------------------------------------------------
#
#  These classes prototype the kind of queries that the Frank Lab,
#  and likely many other labs, will frequently require when analyzing data.
#  In particular, we focus heavily on time queries. For examples of these classes being 
#  used on real experimental data, see place_field_with_queries.ipynb (Frank Lab data) and
#  frank_lab_queries_allen_data.ipynb (Allen Institute data).
#
#  Contents: 
#
#   Classes:
#    - TimeIntervals: a set of non-overlapping time intervals (start/stop times), along
#          with necessary operations on intervals, such as intersection and union
#    - TimeBasedData: an abstract class representing any time-based data. The only requirement we
#          have is that all time-based data contain "valid_intervals" over which the data are observed/valid
#    - PointData: a set of point times occuring within some valid intervals (e.g. unit spiking occuring within
#          the time intervals over which the cell was measured). Optionally there can also be data ("marks") 
#          associated with each of those timestamps 
#    - EventData: a set of time intervals occuring with some valid intervals (e.g. periods in which the animal
#           runs above a threshold speed occuring within the overall period of measuring animal behavior).
#    - ContinuousData: a set of sample times and sample values occuring within some valid intervals. This class is
#           designed to handle sample measurements of something that is, theoretically, a continuous functions of time. 
#           (e.g. samples of animal speed within the period that you are observing the animal)
#
#   Functions: Various plotting and helper functions for working with the above classes.
# ------------------------------------------------



import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
import intervals as iv  # backend for TimeIntervals
import bisect

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as patches
from math import floor


class TimeIntervals():
    """
    Class for representing a list of non-overlapping time intervals.
    
    Currently uses python-intervals as a backend for handling operations like intesection and union,
    abstracting this from other classes that use intervals. 
    To replace with a different backend, change the methods accordingly.
    """
    
    def __init__(self, bounds=None):
        self.intervals = self.__make_intervals(bounds)
        
    def __make_intervals(self, bounds):
        '''Create an interval.Interval from start/end times'''
        if bounds is None:
            return iv.empty()
        elif (isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[1] == 2):
            # input is a 2D (m x 2) numpy array
            intervals = iv.empty()
            for ivl in bounds:
                intervals = intervals | iv.closed(*ivl)
            return intervals
        elif (isinstance(bounds, np.ndarray) and bounds.ndim == 1 and bounds.size == 2):
            # input is a 1D (1 x 2) numpy vector
            return iv.closed(*bounds)
        else:
            raise TypeError("'bounds' must be an m x 2 numpy array (where m is the number of valid intervals) or " +
                            "a 1x2 numpy vector (for a single valid interval)")
        
    def to_array(self):
        '''Create m x 2 numpy array from the set of intervals in an TimeIntervals object'''
        if self.intervals.is_empty(): # iv.empty() has len 1; it contains a special empty interval (I.inf, -I.inf)
            return np.zeros((0,0))
        return np.array([[atomic_ivl.lower, atomic_ivl.upper] for atomic_ivl in self.intervals])
    
    def durations(self):
        '''Return duration of each valid_interval'''
        return np.diff(self.to_array(), axis=1)
        
    def __and__(self, time_intervals):
        '''Return the intersection of two TimeIntervals'''
        intersection = self.intervals & time_intervals.intervals  # intersect using python-intervals
        return TimeIntervals(np.array([[ivl.lower, ivl.upper] for ivl in intersection]))

    def intersect(self, time_intervals):
        '''Return the intersection of two TimeIntervals'''
        return self & time_intervals
    
    def __or__(self, time_intervals):
        '''Return the union of two TimeIntervals'''
        union = self.intervals | time_intervals.intervals  # union using python-intervals
        return TimeIntervals(np.array([[ivl.lower, ivl.upper] for ivl in union]))

    def union(self, time_intervals):
        '''Return the union of two TimeIntervals'''
        return self | time_intervals

    def contains_which(self, times):
        """Check which of the times are contained in the TimeIntervals.
        More efficient that using __contains__ on every point."""
        valid_indices = []
        valid_points = []
        # Use binary search to determine where each point times would go in the flattend
        # list of query edges. The parity of this determines whether or not the point was in the query.
        query_edges = self.to_array().flatten()
        for i, t in enumerate(times):
            bisect_idx = bisect.bisect(query_edges, t)
            if bisect_idx % 2 != 0:  # odd means the point was in the query
                valid_indices.append(i)
                valid_points.append(t)
        return valid_indices, valid_points
        
    
    def __contains__(self, t):
        """Check whether time t is in TimeIntervals. Supports the 'v in TimeIntervals' pattern."""
        return t in self.intervals  # python-intervals algo for contains seems slow (try binary search?)
    
    def __len__(self):
        """Return number of non-overlapping intervals (i.e. start/stop) in this TimeIntervals."""
        if self.intervals.is_empty(): # iv.empty() has len 1; it contains a special empty interval (I.inf, -I.inf)
            return 0
        return len(self.intervals)
    
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        # make TimeIntervals an iterator object
        if self.iter_idx >= self.__len__():
            raise StopIteration
        else:
            ivl = self.intervals[self.iter_idx]
            self.iter_idx += 1
            return np.array([ivl.lower, ivl.upper])

    def __getitem__(self, idx):
        # allow indexing of sub-intervals, returning a new TimeIntervals
        if idx >= len(self.intervals):
            raise IndexError("TimeIntervals index out of range.")
        return TimeIntervals(self.to_array()[idx])

    def __repr__(self):
        return (f'{self.__class__.__qualname__}('
           f'{self.intervals!r})')
    
class TimeBasedData(ABC):
    """Abstract base class for representing data observed over some time intervals."""
    
    # Abstract method. All subclasses must implement a time_query method.
    @abstractmethod     
    def time_query(self, query):
        raise NotImplementedError("Subclasses of abstract class Data must implement 'time_query' method.")
        
    # All subclasses will inherit a valid_intervals property. May be overridden.
    @property
    def valid_intervals(self):
        return self._valid_intervals
    
    @valid_intervals.setter
    def valid_intervals(self, new_valid_intervals):
        if not(isinstance(new_valid_intervals, TimeIntervals)):
            raise TypeError("'valid_intervals' must be of type nwb_query.TimeIntervals")
        self._valid_intervals = new_valid_intervals
        
        
        
class PointData(TimeBasedData):
    '''
    Represent events occurring at discrete points in time (i.e. with no duration), observed
    over some time intervals (i.e. a point process). Optionally with mark data associated
    with each event (i.e. a marked point process).
    '''    
    def __init__(self, point_times, valid_intervals, marks=None):
        self.typecheck(point_times, marks=marks) # valid_intervals is typechecked in TimeBasedData
        self.point_times = point_times
        self.valid_intervals = valid_intervals  # this uses the inherited setter
        self.marks = marks
    
    def typecheck(self, point_times, marks=None):
        if not isinstance(point_times, np.ndarray):
            raise TypeError("'point_times' must be a numpy.array")
        if not(point_times.ndim==1):
            raise ValueError("'point_times' must be a vector (1-dimensional array).")
        if marks and not isinstance(marks, np.ndarray):
            raise TypeError("'marks' must be a numpy.array")
        if marks and not(marks.shape[0]==self.point_times.shape[0]):
            raise ValueError("'marks' must have same # of entries (rows) as 'point_times'.")
    
    def time_query(self, query):
        '''
        Return PointData with data available during requested query.
        
        Arguments
        ---------
        query : EventData, TimeIntervals
            Time intervals to select from the PointData object.
        
        Returns
        -------
        query_result : PointData
            New PointData object with the resulting valid_intervals being the intersection of the
            original valid_intervals and the query intervals. The resulting point_data and marks are those
            found in the resulting valid_intervals (which will be the same as the marks found in the query
            intervals, if the marks all lie within their valid_intervals, as they should).
        '''
        
        # Find the resulting valid_intervals where the data have support (i.e. intersect with the selection intervals)
        if isinstance(query, EventData):
            result_valid_intervals = self.valid_intervals & query.event_intervals
        elif isinstance(query, TimeIntervals):
            result_valid_intervals = self.valid_intervals & query
        else:
            raise TypeError("PointData.query currently only supports queries of " +
                            "type nwb_query.EventData or nwb_query.TimeIntervals")
            
        # Collect the valid point_times in the query
        
        valid_indices, valid_points = query.contains_which(self.point_times)
        
#         valid_indices = [i for (i, t) in enumerate(self.point_times) if t in query]  
        
        
        result_point_times = self.point_times[valid_indices]
        if self.marks:
            result_marks = self.marks[valid_indices, :]
            return PointData(result_point_times, result_valid_intervals, result_marks)
        else:
            return PointData(result_point_times, result_valid_intervals)
        
        
    def mark_with_ContinuousData(self, continuous_data, merge_valid_intervals=True, interpolation='linear'):
        """
        Evaluate ContinuousData at each point_time and add result as a corresponding mark. 
        
        Arguments
        ---------
        continuous_data : ContinuousData
            The data to mark this PointProcess with. The timestamps need not overlap exactly, as we will
            interpolate the data value at each time in the PointProcess.
        merge_valid_intervals : True (default), False
            If merge_valid_intervals=True, the resulting PointData will have valid_intervals that 
            are the intersection of the valid_intervals of the inputs, and it may not contain all 
            of the original point_times. Otherwise, all point_times will be returned, but will be
            marked with 'None' at times when continuous_data is undefined (i.e. outside its 
            valid_intervals).
        interpolation : 'linear' (default), 'nearest', 'quadratic', 'cubic'
            The interpolation type to be passed into scipy.interp1d as 'kind'.
            
        Returns
        -------
        marked_point_process : PointProcess
            A new PointProcess with each point_time associated with marked data, the values of which
            are interpolated from the input ContinuousData object. Under the hood this is currently
            a Pandas DataFrame.
        """
        if continuous_data.samples.ndim > 1 and not (interpolation == 'nearest' or interpolation == 'linear'):
            raise NotImplementedError("For data > 1-D, only 'nearest' and 'linear' interpolation are currently suppported")
        
        
        # Make an interpolation function using the continuous data and sample_times, which we will use to
        # evaluate the ContinuousData at the point_times of this PointData. Note that while the interpolator will
        # accept a Pandas DataFrame as input, it will not build a function from sample_time -> DataFrame. Rather
        # it will build a function from sample_times -> numpy array. So we will use numpy arrays for the interpolator,
        # and then convert back to pandas after calling it.
        interpolator = interp1d(x=continuous_data.sample_times, 
                                y=continuous_data.samples.values,
                                kind=interpolation, 
                                axis=0)
        
        mark_shape = continuous_data.samples.shape[1:] # 1 row per time point, but mark data can be multi-d
        
        if merge_valid_intervals:
            # timequery on pp: intersects valid_intervals and discards point_times outside overlapping region
            result_pp = self.time_query(continuous_data.valid_intervals)
            # don't need to worry about time points outside of the continuous data, b/c we have already filtered point_times
            result_marks = interpolator(result_pp.point_times)
            # convert back to pandas (see comment when building the interpolator)
            result_pp.marks = pd.DataFrame(data=result_marks, columns=continuous_data.samples.columns)
            return result_pp
        else:
            result_marks = []
            for t in self.point_times:
                if t in continuous_data.valid_intervals:
                    result_marks.append(interpolator(t)) 
                else:
                     # set mark to None if the time point occurs outside the continuous data
                    result_marks.append(np.full(mark_shape, None)) 
            result_marks = np.concatenate(result_marks, axis=0)
            # convert back to pandas (see comment when building interpolator)
            result_marks = pd.DataFrame(data=result_marks, columns=continuous_data.samples.columns)
            return PointData(self.point_times, self.valid_intervals, marks=result_marks)


class ContinuousData(TimeBasedData):
    '''
    Represent continuously-sampled numerical data, observed over some time intervals.
    '''        
    def __init__(self, samples, sample_times, valid_intervals=None, find_gaps=False):
        if valid_intervals is None and find_gaps:
            # find the valid_intervals from gaps in the data (WARNING: UNTESTED)
            self.valid_intervals = self.__find_valid_intervals(self.sample_times)
        elif valid_intervals is None:
            # use first and last sample as the valid_intervals
            sample_times_range = np.array([[sample_times[0], sample_times[-1]]])
            self.valid_intervals = TimeIntervals(sample_times_range)
        else:
            if not isinstance(valid_intervals, TimeIntervals):
                self.valid_intervals = TimeIntervals(valid_intervals)
            else:
                self.valid_intervals = valid_intervals
        
        if not isinstance(samples, pd.core.frame.DataFrame):
            raise TypeError("'Samples' must be a Pandas DataFrame.")
            
        self.samples = samples
        self.sample_times = sample_times
        
        
    
    def __find_valid_intervals(self, sample_times, gap_threshold_samps=1.5):
        """Optionally build valid_intervals from any gaps in the data.
        
        WARNING: This is currently not tested.
        """
        import warnings
        warnings.warn("Deducing valid_interval is currently untested, may be bogus.")        
        stepsize = np.mean(np.diff(sample_times, 1)) # use first derivatives to estimate the stepsize
        diffs = np.diff(sample_times, 2)  # use second derivative to identify gaps
        epsilon = gap_threshold_samps * stepsize  # only count if the gap is big with respect to the stepsize
        ivl_end_indices = np.where(diffs > epsilon)[0] + 1  
        if ivl_end_indices.size == 0:  # no gaps in observation
            return TimeIntervals(np.array([[sample_times[0], sample_times[-1]]]))
        else:
            # append the last valid index of the array to the end indices
            np.append(ivl_end_indices, ivl_end_indices.size-1) 
            # build the valid_intervals
            bounds = []  
            for i, end_idx in enumerate(ivl_end_indices):
                if i == 0:   # handle the first interval
                    bounds.append([self.sample_times[0], self.sample_times[end_idx]])
                else:
                    previous_end_idx = ivl_end_indices[i-1]
                    new_start_idx = previous_end_idx + 1
                    bounds.append([self.sample_times[new_start_idx], self.sample_times[end_idx]])
            return TimeIntervals(np.array(bounds))
            

    def time_query(self, query):
        """
        Return ContinuousData in the specified time_intervals. 
        
        Arguments
        ---------
        query : EventData, TimeIntervals
            Time periods to select from the ContinousData
        
        Returns
        -------
        A new ContinuousData object, where the resulting valid_intervals are the
        intersection of the valid_intervals of this ContinuousData and the query's time_intevals.
        The resulting samples and sample_times are those occurring in the resulting valid_intervals
        (which will be the same as the marks found in the query intervals, if the marks all lie
        within their valid_intervals, as they should if the data is well-formed).
        """        
        
        # Constrain the resulting valid_intervals to where the data have support (i.e. intersect with selection intervals)
        if isinstance(query, EventData):
            result_valid_intervals = self.valid_intervals & query.event_intervals
        elif isinstance(query, TimeIntervals):
            result_valid_intervals = self.valid_intervals & query
        elif isinstance(query, np.ndarray):
            result_valid_intervals = self.valid_intervals & TimeIntervals(query)
        else:
            raise TypeError("'query' must be of type nwb_query.EventData, nwb_query.TimeIntervals, or numpy.ndarray")
        
        # Get index into samples and sample_times of interval starts/ends
        result_bounds = result_valid_intervals.to_array()
        result_lower_bounds = result_bounds[:,0]
        result_upper_bounds = result_bounds[:,1]
        # Intervals are closed; find first/last matching sample_times for lower/upper bounds
        result_lower_index = np.searchsorted(self.sample_times, result_lower_bounds, side='left')
        result_upper_index = np.searchsorted(self.sample_times, result_upper_bounds, side='right')

        # loop over valid interval start/stop indices, getting the valid samples and sample_times
        result_sample_times = []
        result_samples = pd.DataFrame(columns=self.samples.columns)       
        for idx_lower, idx_upper in zip(result_lower_index, result_upper_index):
            # append an array of times for this valid interval (we will concatenate them later)
            result_sample_times.append(self.sample_times[idx_lower:idx_upper])
            # concatenate samples for the next valid interval to the result_samples DataFrame
            samples_next_valid_interval = self.samples.iloc[idx_lower:idx_upper, :]
            result_samples = pd.concat([result_samples, samples_next_valid_interval], ignore_index=True)
        result_sample_times = np.concatenate(result_sample_times) # concatenate all the sample times 
        return ContinuousData(samples=result_samples,
                              sample_times=result_sample_times,
                              valid_intervals=result_valid_intervals)
    
    def data_query(self, query_columns):
        """
        Return a new ContinuousData with samples only including the specified columns.
        
        Arguments
        ---------
        query_columns : list, np.ndarray
            List or array of column names (strings) to select from the samples
            
        Returns
        -------
        query_result : ContinuousData
            Has the same sample_times and valid_intervals as the original, but 
            samples contains only the columns in query_columns.
        """
        if not (isinstance(query_columns, list) or isinstance(query_columns, np.ndarray)):
            raise TypeError("'query_columns' must be a list or numpy array of column names")
        for col in query_columns:
            if not isinstance(col, str):
                raise TypeError("All column names in 'query_columns' must be strings.")
            if col not in self.samples.columns:
                raise ValueError("{} not in the ContinuousData samples.".format(col))
                        
        return ContinuousData(samples=self.samples[query_columns],
                              sample_times=self.sample_times,
                              valid_intervals=self.valid_intervals)
        
    
    def filter_intervals(self, func):
        """
        Return a EventData where the ContinuousData fulfills a boolean lambda function ('func').
        The function will be applied separately to each row of ContinuousData.samples.
        
        Arguments
        ---------
        func : lambda function
            functions that takes n arguments as inputs, where n is the number of
            columns in ContinuousData.samples, and returns True or False.
        
        Returns
        -------
        intervals_where_true : EventData
            Each event corresponds to a time period in which the function returns True.
        """
        if self.samples.shape[0] == 0:
            return EventData(event_intervals=TimeIntervals(),
                             valid_intervals=self.valid_intervals)
                                 
        # apply the function to each row of the samples
        func_of_data = np.array([func(*row) for row in self.samples.values])
        
        # Get the up/down crossing indices, i.e. the first/last elements in each interval that fulfill 'func'
        assert func_of_data.dtype == 'bool'
        df = np.diff(func_of_data.astype(np.int8))
        up_crossings = np.where(df == 1)[0] + 1
        down_crossings = np.where(df == -1)[0]

        # if data begins while function is true, include this as an up-crossing
        if func_of_data[0]:
            up_crossings = np.insert(up_crossings, 0, 0)

        # if data ends while function is true, include this as a down-crossing    
        if func_of_data[-1]:
            down_crossings = np.append(down_crossings, func_of_data.shape[0]-1)
        
        # Create the time intervals
        up_times = self.sample_times[up_crossings]
        down_times = self.sample_times[down_crossings]
        interval_bounds = np.array((up_times, down_times)).T
        filtered_intervals = TimeIntervals(interval_bounds)
        
        # Return a EventData instance, with the same valid_intervals as this ContinuousData instance
        return EventData(event_intervals=filtered_intervals,
                         valid_intervals=self.valid_intervals)
        

    

class EventData(TimeBasedData):
    """Represent events (with a duration) occurring during a period of time.
    
    
    A EventData object can be used for querying additional datasets, while matinating the 
    correct valid intervals from the initial dataset. For example, selecting time intervals 
    where an animal was running faster than a threshold speed, and using the resulting 
    EventData object to query spiking data for a clustered unit. 
    """
    
    def __init__(self, event_intervals, valid_intervals):
        if not isinstance(event_intervals, TimeIntervals):
            raise TypeError("'event_intervals' must be a query.TimeIntervals instance")
        if not np.all([event_ivl in valid_intervals for event_ivl in event_intervals.intervals]):
            raise ValueError("'event_intervals' must be fully contained in 'valid_intervals'.")
        self.event_intervals = event_intervals
        self.valid_intervals = valid_intervals # uses the inherited setter
    
    
    def time_query(self, query):
        '''
        Return EventData with events and valid intervals available during the requested query.
        
        Arguments
        ---------
        query : EventData, TimeIntervals
            Time intervals to select from the EventData.
        
        Returns
        -------
        query_result : EventData
            New EventData object with events and valid_intervals corresponding to the intersection
            of the original EventData and the query intervals.
        '''
        
        # Find the resulting event_intervals and valid_intervals
        if isinstance(query, EventData):
            result_event_intervals = self.event_intervals & query.event_intervals
            result_valid_intervals = self.valid_intervals & query.event_intervals
        elif isinstance(query, TimeIntervals):
            result_event_intervals = self.event_intervals & query
            result_valid_intervals = self.valid_intervals & query
        else:
            raise TypeError("EventData.query currently only supports queries of " +
                            "type nwb_query.EventData or nwb_query.TimeIntervals")
        return EventData(event_intervals=result_event_intervals,
                         valid_intervals=result_valid_intervals)
     
    def to_array(self):
        '''Create m x 2 numpy array from the set of event intervals in this EventData object.
        The event intervals are, themselves, a TimeIntervals object, so we just call to_array() on that.'''
        return self.event_intervals.to_array()
    
    def __contains__(self, t):
        """Check whether time t is in the event intervals."""
        return t in self.event_intervals
    
    def contains_which(self, times):
        """Check which of the times are contained in the TimeIntervals.
        More efficient that using __contains__ on every point."""
        valid_indices = []
        valid_points = []
        # Use binary search to determine where each point times would go in the flattend
        # list of query edges. The parity of this determines whether or not the point was in the query.
        query_edges = self.to_array().flatten()
        for i, t in enumerate(times):
            bisect_idx = bisect.bisect(query_edges, t)
            if bisect_idx % 2 != 0:  # odd means the point was in the query
                valid_indices.append(i)
                valid_points.append(t)
        return valid_indices, valid_points
    
    def valid_contain(self, t):
        """Check whether time t is in the EventData's valid intervals."""
        return t in self.valid_intervals
    
    def durations(self):
        """Durations of the event intervals."""
        return self.event_intervals.durations()
    
    def valid_durations(self):
        """Durations of the valid intervals."""
        return self.valid_intervals.durations()
    
    def __and__(self, other):
        '''Return the intersection of two EventData objects'''
        if not isinstance(other, EventData):
            raise TypeError("'other' must be a nwb_query.EventData object")
        result_event_ivl = self.event_intervals & other.event_intervals
        result_valid_ivl = self.valid_intervals & other.valid_intervals
        return EventData(result_event_ivl, result_valid_ivl)

    def intersect(self, other):
        '''Return the intersection of two EventData objects'''
        return self & other
    
    def __or__(self, other):
        '''Return the union of two EventData objects'''
        if not isinstance(other, EventData):
            raise TypeError("'other' must be a nwb_query.EventData object")
        result_event_ivl = self.event_intervals | other.event_intervals
        result_valid_ivl = self.valid_intervals | other.valid_intervals
        return EventData(result_event_ivl, result_valid_ivl)

    def union(self, other):
        '''Return the union of two EventData objects'''
        return self | other

def plot_PointData_multiple(spikeplots, axis=None):
    if not axis:
        axis = plt.axes()
        
    for i, spikeplot in enumerate(spikeplots):
        plot_PointData(spikeplot[0], axis=axis, ypos=i)

    axis.set_yticks(range(i+1))
    axis.set_yticklabels( [spikeplot[1] for spikeplot in spikeplots])
    xl = axis.get_xlim()
    axis.set_ylim([-1,i+1])
    axis.invert_yaxis()


def plot_PointData(PointData, ypos=1, axis=None, interval_height=25, tick_height=10, color='b'):
    '''Plot a Point Process (events + their enclosing intervals)'''
    valid_ivl_arr = PointData.valid_intervals.to_array().T
    
    if not axis:
        axis = plt.axes()

    # TODO: use plot_TimeIntervals
    ivl_h = axis.plot(valid_ivl_arr, np.full(valid_ivl_arr.shape, ypos),
                     color=color,
                     linewidth=interval_height,
                     marker='',
                     alpha=0.1,
                     solid_capstyle='butt')

    points_h = axis.plot(PointData.point_times, np.full(PointData.point_times.shape, ypos),
                       color=color,
                       marker='|', 
                       markersize=tick_height, 
                       linestyle='')
    
    xtick_locator = mticker.AutoLocator()
    xtick_formatter = mticker.FuncFormatter(fmt_truncate_posix)

    axis.xaxis.set_major_locator(xtick_locator)
    axis.xaxis.set_major_formatter(xtick_formatter)

    axis.set_xlabel('Time (s)')


    return ivl_h, points_h

def fmt_truncate_posix (x, pos):
    oom = 6
    #     offset_str = "%de%d + \n" % (x // 10 ** oom, oom)
    ellipsis = "\u2026"
    offset_str = format(floor(x // 10 ** oom), ",d")
    # oom zero-padded digits before decimal point and up to 6 digits past it 
    # (no trailing zeros)
    # NB no way to do this with single format string: %05.11g gives too many digits past 
    # the decimal when integer part is small.
    remainder_str = format(floor(x % 10 ** oom), "07,d") + \
                    ("%0.6g" % (x % 1))[1:] # omit leading 0

    if pos == 0: # first visible tick
        return offset_str + "," + remainder_str

    else:
        return "\u2026" + remainder_str

    
def plot_ContinuousData(ContinuousData, axis=None, interval_pad_factor=1.1, ivl_color='b'):
    '''Plot Continuous Data and their enclosing intervals'''
    '''Interval_pad_factor--height of the interval box, as a multiple of data range'''

    if not axis:
        axis = plt.axes()
    
    datarange = [ContinuousData.samples.values.min(),  ContinuousData.samples.values.max()]
    ivl_pad = np.diff(datarange) * (interval_pad_factor-1) / 2
    ivl_y = [datarange[0] - ivl_pad, datarange[1] + ivl_pad]
    
    ivl_h = plot_TimeIntervals(ContinuousData.valid_intervals, ivl_y, axis=axis)
    
    data_h = []
    
#     # TODO: insert NaN's between Intervals then plot as a single line 
#     #(better for line format, legends, handles, etc)
  
#     # Plot discontinuous lines by inserting points where x=NaN, y=NaN between intervals
#     ivl_end_times = ContinuousData.valid_intervals.to_array()[:,1]
#     ivl_end_index = ContinuousData.sample_times.searchsorted(ivl_end_times, 'left')
#     sample_times_nan_split = np.insert(ContinuousData.sample_times, ivl_end_index, np.nan, axis=0)
#     samples_nan_split = np.insert(ContinuousData.samples, ivl_end_index, np.nan, axis=0)
#     ivl_h = axis.plot(sample_times_nan_split, samples_nan_split)
    
    for ivl in ContinuousData.valid_intervals:
        axis.set_prop_cycle(None) # reset line color cycle etc
        ivl_data_idx = [np.searchsorted(ContinuousData.sample_times, ivl[0], 'left'),
                        np.searchsorted(ContinuousData.sample_times, ivl[1], 'right')]
        data_h.append(
            axis.plot(ContinuousData.sample_times[ivl_data_idx[0]:ivl_data_idx[1]], 
                      ContinuousData.samples.iloc[ivl_data_idx[0]:ivl_data_idx[1], :]))

    axis.legend(ContinuousData.samples.columns, loc=2)
    _format_xaxis_posixtime(axis)
    return ivl_h, data_h


def plot_TimeIntervals(TimeIntervals, ivl_y, axis=None, color='b'):

    if not axis:
        axis = plt.axes()

    ivl_h = []
    for ivl in TimeIntervals:
        ivl_h.append(
            axis.add_patch(patches.Rectangle((ivl[0], ivl_y[0]),
                                         ivl[1] - ivl[0],
                                         ivl_y[1] - ivl_y[0],
                                         fill=True,
                                         color=color,
                                         alpha=0.1)))
    _format_xaxis_posixtime(axis)
    return ivl_h

def plot_EventData(EventData, ypos=1, axis=None, valid_color='b', color='k'):

    if not axis:
        axis = plt.axes()
    
    ivl_h = []
    ivl_h.append(plot_TimeIntervals(EventData.valid_intervals, axis=axis, ivl_y=(ypos-0.5, ypos+0.5), color=valid_color))
    ivl_h.append(plot_TimeIntervals(EventData.event_intervals, axis=axis, ivl_y=(ypos-0.25, ypos+0.25), color=color))

    axis.axis('tight')
    _format_xaxis_posixtime(axis)
    return ivl_h
    
def _format_xaxis_posixtime(axis):

    xtick_locator = mticker.AutoLocator()
    xtick_formatter = mticker.FuncFormatter(fmt_truncate_posix)

    axis.xaxis.set_major_locator(xtick_locator)
    axis.xaxis.set_major_formatter(xtick_formatter)
    axis.set_xlabel('Time (s)')


def point_to_line_dist(point, line):
    """Return distance of point to line segment and the nearest projection point.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the 
    distance to both endpoints and take the shortest distance.
    """
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
        np.linalg.norm(np.cross(unit_line, line[0] - point)) /
        np.linalg.norm(unit_line)
    )

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) + 
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    d_end0 = np.linalg.norm(line[0] - point)
    d_end1 = np.linalg.norm(line[1] - point)
    if d_end0 < d_end1:
        endpoint_dist = d_end0
        end_proj = line[0]
    else:
        endpoint_dist = d_end1
        end_proj = line[1]

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist, (x_seg, y_seg)
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist, end_proj
