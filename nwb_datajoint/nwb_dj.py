# pipeline to populate a schema from an NWB file

import pynwb

import nwb_datajoint.common_behav as common_behav
import nwb_datajoint.common_device as common_device
import nwb_datajoint.common_interval as common_interval
import nwb_datajoint.common_ephys as common_ephys
import nwb_datajoint.common_lab as common_lab
import nwb_datajoint.common_session as common_session
import nwb_datajoint.common_subject as common_subject
import nwb_datajoint.common_task as common_task
import datajoint as dj

import kachery as ka

# import ndx_franklab_novela.probe

conn = dj.conn()


def NWBPopulate(file_names):

    #print('file names:', file_names)

    # CHANGE per object when object_ids are implemented in the NWB file
    default_nwb_object_id = 0
    for nwb_file_name in file_names:
        # FIX: create insert_from_nwb method for Institution and Lab
        """
        Institution, Lab, and Experimenter
        """
        print('Institution...')
        common_lab.Institution().insert_from_nwb(nwb_file_name)
        print('Lab...')
        common_lab.Lab().insert_from_nwb(nwb_file_name)
        print('LabMember...')
        common_lab.LabMember().insert_from_nwb(nwb_file_name)

        """
        Subject
        """
        print('Subject...')
        common_subject.Subject().insert_from_nwb(nwb_file_name)

        """
        Device
        """
        print('Device...')
        common_device.Device().insert_from_nwb(nwb_file_name)
        print('Probe...')
        common_device.Probe().insert_from_nwb(nwb_file_name)

        """
        Task and Apparatus Information structures.
        These hold general information not specific to any one epoch. Specific information is added in task.TaskEpoch
        """
        print('Task...')
        common_task.Task().insert_from_nwb(nwb_file_name)
        # common_task.Apparatus().insert_from_nwb(nwb_file_name)


        # now that those schema are updated, we can populate the Session and Experimenter lists and then insert the
        # rest of the schema

        print('Computing SHA-1 and storing in kachery...')
        with ka.config(use_hard_links=True):
            kachery_path = ka.store_file(nwb_file_name)
            sha1 = ka.get_file_hash(kachery_path)

        print('Session...')
        common_session.Session().insert_from_nwb(nwb_file_name=nwb_file_name, nwb_file_sha1=sha1)

        # print('Session...')
        # common_session.Session.populate()
        print('ExperimenterList...')
        common_session.ExperimenterList.populate()

        print('Skipping IntervalList...')
        # common_interval.IntervalList().insert_from_nwb(nwb_file_name)
        # populate the electrode configuration table for this session
        print('Skipping ElectrodeConfig...')
        # common_ephys.ElectrodeConfig().insert_from_nwb(nwb_file_name)
        print('Skipping raw...')
        # common_ephys.Raw().insert_from_nwb(nwb_file_name)

        #common_task.TaskEpoch.insert_from_nwb(nwb_file_name)


        # ephys.Units.populate()

        # populate the behavioral variables. Note that this has to be done after task.TaskEpoch
        print('RawPosition...')
        common_behav.RawPosition.populate()
        # common_behav.HeadDir.populate()
        # common_behav.Speed.populate()
        # common_behav.LinPos.populate()





