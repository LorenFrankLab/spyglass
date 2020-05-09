# pipeline to populate a schema from an NWB file

import pynwb

import common_behav
import common_device
import common_ephys
import common_lab
import common_session
import common_subject
import common_task
import datajoint as dj
import ndx_fl_novela.probe

import franklab_nwb_extensions.fl_extension as fl_extension

conn = dj.conn()


def NWBPopulate(file_names):

    print('file names:', file_names)

    # CHANGE per object when object_ids are implemented in the NWB file
    default_nwb_object_id = 0
    for nwb_file_name in file_names:
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                nwb_file_name))
            print(io.read())
            continue

        print('Starting NWB file', nwb_file_name)

        """
        NWBFile
        """
        common_session.Nwbfile.insert1(dict(nwb_file_name=nwb_file_name), skip_duplicates=True)

        """
        Subject

        ADD date_of_birth when NWB bug fixed
        """

        common_subject.Subject().insert_from_nwb(nwb_file_name)

        """
        Device
        """
        common_device.Device().insert_from_nwb(nwb_file_name)
        common_device.Probe().insert_from_nwb(nwb_file_name)

        """
        Institution
        """
        common_lab.Institution.insert1(
                dict(institution_name=nwbf.institution), skip_duplicates=True)

        """
        Lab
        """
        common_lab.Lab.insert1(dict(lab_name=nwbf.lab), skip_duplicates=True)

        """
        Task and Apparatus Information structures.
        These hold general information not specific to any one epoch. Specific information is added in task.Task
        """
        common_task.Task().insert_from_nwb(nwb_file_name)
        common_task.Apparatus().insert_from_nwb(nwb_file_name)

        # now that those schema are updated, we can call the populate method for the rest of the schema
        common_session.Session.populate()
        common_session.ExperimenterList.populate()
        common_task.TaskEpoch.populate()

        # populate the electrode configuration table for this session
        common_ephys.ElectrodeConfig.populate()
        # ephys.Units.populate()

        # populate the behavioral variables. Note that this has to be done after task.TaskEpoch
        common_behav.Position.populate()
        common_behav.HeadDir.populate()
        common_behav.Speed.populate()
        common_behav.LinPos.populate()

        io.close()
