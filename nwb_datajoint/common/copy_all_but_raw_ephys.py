import pynwb
import ndx_franklab_novela
import shutil
from .nwb_helper_fn import get_data_interface


def copy_all_but_raw_ephys(nwb_file_abspath, out_nwb_file_abspath):
    with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
        nwbf = io.read()

        # pop off the unnecessary elements to save space
        nwbf.acquisition.pop('e-series')
        # test removal of analog data
        """
        analog_found = False
        if len(get_data_interface(nwbf, 'analog')):
            analog_found = True
            nwbf.processing['analog'].data_interfaces
        nwbf.
        """

        # export the new NWB file
        with pynwb.NWBHDF5IO(path=out_nwb_file_abspath, mode='w') as export_io:
            export_io.export(io, nwbf)

    # add link from new file back to raw ephys data in raw file
    manager = pynwb.get_manager()
    with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r', manager=manager) as io:
        nwbf_raw = io.read()
        raw_ephys = nwbf_raw.acquisition['e-series']

        with pynwb.NWBHDF5IO(path=out_nwb_file_abspath, mode='a', manager=manager) as export_io:
            nwbf_export = export_io.read()

            # add link to raw ephys ElectricalSeries in raw data file
            nwbf_export.add_acquisition(raw_ephys)

            export_io.write(nwbf_export)

    return out_nwb_file_abspath


def test_read(out_nwb_file_abspath):
    with pynwb.NWBHDF5IO(path=out_nwb_file_abspath, mode='r') as io:
        nwbf = io.read()  # will raise BrokenLinkWarning after move
        print(nwbf.acquisition['e-series'])  # will raise KeyError after move


def main():
    nwb_file_abspath = '/mnt/c/Users/Ryan/Documents/NWB_Data/Frank Lab Data/beans20190718-trim.nwb'
    out_nwb_file_abspath = '/mnt/c/Users/Ryan/Documents/NWB_Data/Frank Lab Data/beans20190718-trim-no_raw.nwb'
    copy_all_but_raw_ephys(nwb_file_abspath, out_nwb_file_abspath)
    test_read(out_nwb_file_abspath)

    # test read after move
    out_nwb_file_abspath_new = '/mnt/c/Users/Ryan/Documents/NWB_Data/Frank Lab Data/tmp/beans20190718-trim-no_raw.nwb'
    shutil.move(out_nwb_file_abspath, out_nwb_file_abspath_new)
    test_read(out_nwb_file_abspath_new)


if __name__ == '__main__':
    main()
