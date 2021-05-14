import datajoint as dj
import numpy as np
import pynwb

from .common_ephys import Raw  # noqa: F401
from .common_interval import IntervalList, interval_list_contains
from .common_nwbfile import Nwbfile
from .common_session import Session  # noqa: F401
from .common_task import TaskEpoch
from .dj_helper_fn import fetch_nwb
from .nwb_helper_fn import get_data_interface, get_valid_intervals, estimate_sampling_rate, get_nwb_file

schema = dj.schema('common_behav')


@schema
class RawPosition(dj.Imported):
    definition = """
    -> Session
    ---
    raw_position_object_id: varchar(80)    # the object id of the data in the NWB file
    -> IntervalList                        # the list of intervals for this object
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        # Get the position data. FIX: change Position to position when name changed or fix helper function to allow
        # upper or lower case
        position = get_data_interface(nwbf, 'position', pynwb.behavior.Position)
        if position is None:
            print(f'No position data interface found in {nwb_file_name}\n')
            return

        for pos_epoch, spatial_series in enumerate(position.spatial_series.values()):
            pos_interval_name = f'pos {pos_epoch} valid times'
            # get the valid intervals for the position data
            timestamps = np.asarray(spatial_series.timestamps)

            # estimate the sampling rate
            sampling_rate = estimate_sampling_rate(timestamps, 1.75)
            if sampling_rate < 0:
                raise ValueError(f'Error adding position data for position epoch {pos_epoch}')
            print("Processing raw position data. Estimated sampling rate: {} Hz".format(sampling_rate))

            # add the valid intervals to the Interval list
            interval_dict = dict()
            interval_dict['nwb_file_name'] = key['nwb_file_name']
            interval_dict['interval_list_name'] = pos_interval_name
            # allow single skipped frames
            interval_dict['valid_times'] = get_valid_intervals(timestamps, sampling_rate, 2.5, 0)
            IntervalList().insert1(interval_dict, skip_duplicates=True)

            key['raw_position_object_id'] = position.object_id
            # TODO why is the SpatialSeries object_id not stored?
            # this is created when we populate the Task schema
            key['interval_list_name'] = pos_interval_name
            self.insert1(key, skip_duplicates=True)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, 'nwb_file_abs_path'), *attrs, **kwargs)


@schema
class StateScriptFile(dj.Imported):
    definition = """
    -> TaskEpoch
    ---
    file_object_id: varchar(40)  # the object id of the file object
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # TODO change to associated_files when NWB file changed.
        associated_files = nwbf.processing.get('associated files')
        if associated_files is not None:
            # TODO type checking that associated_file_obj is of type ndx_franklab_novela.AssociatedFiles
            for associated_file_obj in associated_files.data_interfaces.values():
                # parse the task_epochs string
                epoch_list = associated_file_obj.task_epochs.split(',')
                # find the file associated with this epoch
                if str(key['epoch']) in epoch_list:
                    key['file_object_id'] = associated_file_obj.object_id
                    self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, 'nwb_file_abs_path'), *attrs, **kwargs)


@schema
class VideoFile(dj.Imported):
    definition = """
    -> TaskEpoch
    video_file_num = 0: int
    ---
    video_file_object_id: varchar(40)  # the object id of the file object
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        video = get_data_interface(nwbf, 'video')

        # get the interval for the current TaskEpoch
        interval_list_name = (TaskEpoch() & key).fetch1('interval_list_name')
        valid_times = (IntervalList & {'nwb_file_name': key['nwb_file_name'],
                                       'interval_list_name': interval_list_name}).fetch1('valid_times')

        for video_obj in video.time_series.values():
            # check to see if the times for this video_object are largely overlapping with the task epoch times
            if len(interval_list_contains(valid_times, video_obj.timestamps) > .9 * len(video_obj.timestamps)):
                key['video_file_object_id'] = video_obj.object_id
                self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, 'nwb_file_abs_path'), *attrs, **kwargs)


@schema
class HeadDir(dj.Imported):
    definition = """
    -> Session
    ---
    nwb_object_id: int    # the object id of the data in the NWB file
    -> IntervalList       # the list of intervals for this object
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # position data is stored in the Behavior module
        try:
            behav_mod = nwbf.get_processing_module('Behavior')
            headdir = behav_mod.data_interfaces['Head Direction']  # noqa: F841 TODO: make use of headdir
        except Exception:  # TODO: use more precise error check
            print(f'Unable to import HeadDir: no Behavior module found in {nwb_file_name}')
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_list_name'] = 'task epochs'
        self.insert1(key)


@schema
class Speed(dj.Imported):
    definition = """
    -> Session
    ---
    nwb_object_id: int    # the object id of the data in the NWB file
    -> IntervalList       # the list of intervals for this object
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # position data is stored in the Behavior module
        try:
            behav_mod = nwbf.get_processing_module('Behavior')
            speed = behav_mod.data_interfaces['Speed']  # noqa: F841 TODO: make use of speed
        except Exception:  # TODO: use more precise error check
            print(f'Unable to import Speed: no Behavior module found in {nwb_file_name}')
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_list_name'] = 'task epochs'
        self.insert1(key)


@schema
class LinPos(dj.Imported):
    definition = """
    -> Session
    ---
    nwb_object_id: int    # the object id of the data in the NWB file
    -> IntervalList       # the list of intervals for this object
    """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # position data is stored in the Behavior module
        try:
            behav_mod = nwbf.get_processing_module("Behavior")
            linpos = behav_mod.data_interfaces['Linearized Position']  # noqa: F841 TODO: make use of speed
        except Exception:  # TODO: use more precise error check
            print(f'Unable to import LinPos: no Behavior module found in {nwb_file_name}')
            return
        key['nwb_object_id'] = -1
        # this is created when we populate the Task schema
        key['interval_list_name'] = 'task epochs'
        self.insert1(key)
