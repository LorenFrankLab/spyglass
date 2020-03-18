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
import franklabnwb

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

        # start a database transaction ensure all of this happens correctly
        # If we need to exit on error, remember to conn.cancel_transaction()
        # not clear to me if this is helpful / necessary
        conn.start_transaction()

        """
        NWBFile
        """
        if {'nwb_file_name': nwb_file_name} not in common_session.Nwbfile():
            common_session.Nwbfile.insert1(dict(nwb_file_name=nwb_file_name))

        """
        Subject

        ADD date_of_birth when NWB bug fixed
        """
        sub = nwbf.subject
        # check to see if the subject has been added to the table aready
        if {'subject_id': sub.subject_id} not in common_subject.Subject():
            subject_dict = dict()
            subject_dict['subject_id'] = sub.subject_id
            if sub.age is None:
                subject_dict['age'] = 'unknown'

            subject_dict['description'] = sub.description
            subject_dict['genotype'] = sub.genotype
            subject_dict['sex'] = sub.sex
            subject_dict['species'] = sub.species
            print(subject_dict)
            common_subject.Subject.insert1(subject_dict)

        """
        Device
        """
        devices = list(nwbf.devices.keys())
        device_dict = dict()
        for d in devices:
            if {'device_name': d} not in common_device.Device():
                # FIX: we need to get the right fields in the NWB device for this schema
                # device.Device.insert1(dict(device_name=d))
                device_dict['device_name'] = d
                device_dict['system'] = 'SpikeGadgets'
                device_dict['amplifier'] = 'Other'
                device_dict['adc_circuit'] = ''
                common_device.Device.insert1(device_dict)

        """
        Institution
        """
        if {'institution_name': nwbf.institution} not in common_lab.Institution():
            common_lab.Institution.insert1(
                dict(institution_name=nwbf.institution))

        """
        Lab
        """
        if {'lab_name': nwbf.lab} not in common_lab.Lab():
            common_lab.Lab.insert1(dict(lab_name=nwbf.lab))

        """
        Task and Apparatus Information structures.
        These hold general information not specific to any one epoch. Specific information is added in task.Task
        """
        task_dict = dict()
        task_mod = []
        try:
            task_mod = nwbf.get_processing_module("Task")
        except:
            print('No Task module found in {}\n'.format(nwb_file_name))
        if task_mod != []:
            for d in task_mod.data_interfaces:
                if type(task_mod[d]) == franklabnwb.fl_extension.Task:
                    # see this task if is already in the database
                    if {'task_name': d} not in common_task.TaskInfo():
                        # FIX task type and subtype would need to be in the NWB file
                        task_dict['task_name'] = d
                        task_dict['task_type'] = ''
                        task_dict['task_subtype'] = ''
                        common_task.TaskInfo.insert1(task_dict)
                    else:
                        print('Skipping task {}; already in schema\n'.format(d))

        apparatus_dict = dict()
        apparatus_mod = []
        try:
            apparatus_mod = nwbf.get_processing_module("Apparatus")
        except:
            print('No Apparatus module found in {}\n'.format(nwb_file_name))
        if apparatus_mod != []:
            for d in apparatus_mod.data_interfaces:
                if type(apparatus_mod[d]) == franklabnwb.fl_extension.Apparatus:
                    # see this Apparaus if is already in the database
                    if {'apparatus_name': d} not in common_task.ApparatusInfo():
                        apparatus_dict['apparatus_name'] = d
                        common_task.ApparatusInfo.insert1(apparatus_dict)
                    else:
                        print('Skipping apparatus {}; already in schema\n'.format(d))

        conn.commit_transaction()

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
