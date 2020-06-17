import common_device as device
import datajoint as dj

schema = dj.schema('franklab')


def add_probes():
    probetype_dict = dict()
    # add a tetrode probe type
    probetype_dict['probe_type'] = 'tetrode'
    if {'probe_type': 'tetrode'} not in device.Probe():
        probetype_dict['num_shanks'] = 1
        probetype_dict['probe_description'] = 'tetrode made from 4 12.5 um nichrome wires'
        probetype_dict['contact_side_numbering'] = 1
        # check to see if this probe is in the schema

        device.Probe.insert1(
            probetype_dict, allow_direct_insert=True, replace=True)
        device.Probe.Shank.insert1(['tetrode', 0])
        for i in range(4):
            device.Probe.Electrode.insert1(
                [['tetrode'], 0, i, 12.5, 0, 0, 0], replace=True)

        # add a 32 channel polymer probe type. FIX the x,y,z coordinates for each channel
    # check this name
    probetype_dict['probe_type'] = '32c-2s8mm6cm-20um-40um-dl'
    if {'probe_type': probetype_dict['probe_type']} not in device.Probe():
        probetype_dict['num_shanks'] = 2
        probetype_dict['probe_description'] = '32 channel polymide probe'
        probetype_dict['contact_side_numbering'] = 1
        device.Probe.insert1(
            probetype_dict, allow_direct_insert=True, replace=True)
        device.Probe.Shank.insert1(['32c-2s8mm6cm-20um-40um-dl', 0])
        device.Probe.Shank.insert1(['32c-2s8mm6cm-20um-40um-dl', 1])
        for i in range(16):
            device.Probe.Electrode.insert1(
                [['32c-2s8mm6cm-20um-40um-dl'], 0, i, 20, 0, 0, 0], replace=True)
        for i in range(16, 32):
            device.Probe.Electrode.insert1(
                [['32c-2s8mm6cm-20um-40um-dl'], 1, i, 20, 0, 0, 0], replace=True)
