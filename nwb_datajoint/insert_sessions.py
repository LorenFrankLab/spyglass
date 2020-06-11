# pipeline to populate a schema from an NWB file

import os
import pynwb

from .common_lab import Lab, LabMember, Institution
from .common_subject import Subject
from .common_device import Device, Probe
from .common_task import Task
from .common_nwbfile import Nwbfile
from .common_session import Session
from .common_interval import IntervalList
from .common_ephys import ElectrodeConfig
# import nwb_datajoint.common_behav as common_behav
# import nwb_datajoint.common_interval as common_interval
# import nwb_datajoint.common_ephys as common_ephys
# import nwb_datajoint.common_session as common_session
# import nwb_datajoint.common_task as common_task
import datajoint as dj
import kachery as ka

conn = dj.conn()

def insert_sessions(nwb_file_names):
    assert os.getenv('KACHERY_STORAGE_DIR', None), 'You must set the KACHERY_STORAGE_DIR environment variable.'

    for nwb_file_name in nwb_file_names:
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        assert os.path.exists(nwb_file_abspath), f'File does not exist: {nwb_file_abspath}'
    
        Nwbfile().insert_from_file_name(nwb_file_name)
    
    Session().populate()

    # with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
    #     nwbf = io.read()
    #     print('Institution...')
    #     Institution().insert_from_nwbfile(nwbf)
    #     print('Lab...')
    #     Lab().insert_from_nwbfile(nwbf)
    #     print('LabMember...')
    #     LabMember().insert_from_nwbfile(nwbf)

    #     print('Subject...')
    #     Subject().insert_from_nwbfile(nwbf)

    #     print('Device...')
    #     Device().insert_from_nwbfile(nwbf)
    #     print('Probe...')
    #     Probe().insert_from_nwbfile(nwbf)

    #     """
    #     Task and Apparatus Information structures.
    #     These hold general information not specific to any one epoch. Specific information is added in task.TaskEpoch
    #     """
    #     print('Task...')
    #     Task().insert_from_nwbfile(nwbf)
    #     # common_task.Apparatus().insert_from_nwb(nwb_file_name)

    #     # now that those schema are updated, we can populate the Session
    #     # and Experimenter lists and then insert the rest of the schema

    # Session().populate()

        # print('IntervalList...')
        # IntervalList().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)
        
        # # populate the electrode configuration table for this session
        # print('ElectrodeConfig...')
        # ElectrodeConfig().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)
        
        # print('Skipping raw for now...')
        # # # common_ephys.Raw().insert_from_nwb(nwb_file_name)

        # # #common_task.TaskEpoch.insert_from_nwb(nwb_file_name)
        
        # # # ephys.Units.populate()

        # # # populate the behavioral variables. Note that this has to be done after task.TaskEpoch
        # # print('RawPosition...')
        # # common_behav.RawPosition.populate()
        # # # common_behav.HeadDir.populate()
        # # # common_behav.Speed.populate()
        # # # common_behav.LinPos.populate()





