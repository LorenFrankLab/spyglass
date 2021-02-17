import datajoint as dj
import kachery as ka
import os
import pandas as pd
import pathlib
import pynwb

from .nwb_helper_fn import get_electrode_indices

schema = dj.schema("common_nwbfile")

# define the fields that should be kept in AnalysisNWBFiles
nwb_keep_fields = ('devices', 'electrode_groups', 'electrodes', 'experiment_description',
                   'experimenter', 'file_create_date', 'identifier', 'intervals',
                   'institution', 'lab', 'session_description', 'session_id',
                   'session_start_time', 'subject', 'timestamps_reference_time')


@schema
class Nwbfile(dj.Manual):
    definition = """
    # Table for holding the Nwb files.
    nwb_file_name: varchar(255) # name of the NWB file
    ---
    nwb_file_abs_path: filepath@raw
    """

    def insert_from_relative_file_name(self, nwb_file_name):
        """
        Insert a new session from an existing nwb file.

        Parameters
        ----------
        nwb_file_name : str
            Relative path to the nwb file
        """
        nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
        assert os.path.exists(nwb_file_abs_path), f'File does not exist: {nwb_file_abs_path}'

        self.insert1(dict(
            nwb_file_name=nwb_file_name,
            nwb_file_abs_path=nwb_file_abs_path,
        ), skip_duplicates=True)

    @staticmethod
    def get_abs_path(nwb_file_name):
        base_dir = pathlib.Path(os.getenv('NWB_DATAJOINT_BASE_DIR', None))
        assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR or provide the base_dir argument'

        nwb_file_abspath = base_dir / 'raw' / nwb_file_name
        return str(nwb_file_abspath)

    @staticmethod
    def add_to_lock(nwb_file_name):
        """
        Adds the specified NWB file to the file with the list of nwb files to be
        locked

        :param nwb_file_name: the name of an nwb file that has been inserted into the Nwbfile() schema
        :type nwb_file_name: string
        :return: None
        """
        key = {'nwb_file_name': nwb_file_name}
        # check to make sure the file exists
        assert len((Nwbfile() & key).fetch()) > 0, f'Error adding {nwb_file_name} to lock file, not in Nwbfile() schema'

        lock_file = open(os.getenv('NWB_LOCK_FILE'), 'a+')
        lock_file.write(f'{nwb_file_name}\n')
        lock_file.close()

    def cleanup(self, delete_files=False):
        """ Removes the filepath entries for nwb files that are not in use. Does not delete the files themselves.
        Run this after deleting the Nwbfile() entries themselves."""
        self.external['raw'].delete(delete_external_files=delete_files)


# TODO: add_to_kachery will not work because we can't update the entry after it's been used in another table.
# We therefore need another way to keep track of the
@schema
class AnalysisNwbfile(dj.Manual):
    definition = """
    # Table for holding the NWB files that contain results of analysis, such as spike sorting
    analysis_file_name : varchar(255) # name of the file
    ---
    -> Nwbfile                                    # name of the parent NWB file. Used for naming and metadata copy
    analysis_file_abs_path: filepath@analysis     # the full path to the file
    analysis_file_description = '': varchar(255)  # an optional description of this analysis
    analysis_parameters = NULL: blob              # additional relevant parmeters. Currently used only for analyses
                                                  # that span multiple NWB files
    """

    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def create(self, nwb_file_name):
        """
        Opens the NWB file that ends with _, creates a copy, writes out the
        copy to disk and return the name of the new file.
        Note that this does NOT add the file to the schema; that needs to be
        done after data are written to it.

        Parameters
        ----------
        nwb_file_name : string

        Returns
        -------
        analysis_file_name : string
        """

        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)

        io = pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r')
        nwbf = io.read()

        # pop off the unnecessary elements to save space
        nwb_fields = nwbf.fields
        for field in nwb_fields:
            if field not in nwb_keep_fields:
                nwb_object = getattr(nwbf, field)
                if type(nwb_object) is pynwb.core.LabelledDict:
                    for module in list(nwb_object.keys()):
                        nwb_object.pop(module)

        # key = dict()
        # key['nwb_file_name'] = nwb_file_name
        # get the current number of analysis files related to this nwb file
        n_analysis_files = len((AnalysisNwbfile() & {'nwb_file_name': nwb_file_name}).fetch())
        # name the file, adding the number of files with preceeding zeros
        analysis_file_name = os.path.splitext(nwb_file_name)[0] + str(n_analysis_files).zfill(6) + '.nwb'
        # key['analysis_file_name'] = analysis_file_name
        # key['analysis_file_description'] = ''
        # write the new file
        print(f'Writing new NWB file {analysis_file_name}')
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        # key['analysis_file_abs_path'] = analysis_file_abs_path
        # export the new NWB file
        with pynwb.NWBHDF5IO(path=analysis_file_abs_path, mode='w') as export_io:
            export_io.export(io, nwbf)

        io.close()

        # insert the new file
        # self.insert1(key)
        return analysis_file_name

    @staticmethod
    def copy(nwb_file_name):
        """
        Makes a copy of an analysis NWB file.
        Note that this does NOT add the file to the schema; that needs to be
        done after data are written to it.

        Parameters
        ----------
        nwb_file_name : string
            name of analysis nwb file to be copied

        Returns
        -------
        analysis_file_name : string
        """

        nwb_file_abspath = AnalysisNwbfile.get_abs_path(nwb_file_name)

        io = pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r')
        nwbf = io.read()

        # get the current number of analysis files related to this nwb file
        original_nwb_file_name = (AnalysisNwbfile &
                                  {'analysis_file_name': nwb_file_name}).fetch('nwb_file_name')[0]
        n_analysis_files = len((AnalysisNwbfile & {'nwb_file_name': original_nwb_file_name}).fetch())
        # name the file, adding the number of files with preceeding zeros
        analysis_file_name = os.path.splitext(original_nwb_file_name)[0] + str(n_analysis_files).zfill(6) + '.nwb'
        # write the new file
        print(f'Writing new NWB file {analysis_file_name}...')
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        # export the new NWB file
        with pynwb.NWBHDF5IO(path=analysis_file_abs_path, mode='w') as export_io:
            export_io.export(io, nwbf)

        io.close()

        return analysis_file_name

    def add(self, nwb_file_name, analysis_file_name):
        """
        Adds the specified file to AnalysisNWBfile table

        Parameters
        ----------
        nwb_file_name : string
            the name of the parent NWB file
        analysis_file_name : string
            the name of the analysis nwb file that was created
        """
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        key['analysis_file_name'] = analysis_file_name
        key['analysis_file_description'] = ''
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        key['analysis_file_abs_path'] = analysis_file_abs_path
        self.insert1(key)

    @staticmethod
    def get_abs_path(analysis_nwb_file_name):
        base_dir = pathlib.Path(os.getenv('NWB_DATAJOINT_BASE_DIR', None))
        assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR or provide the base_dir argument'

        analysis_nwb_file_abspath = base_dir / 'analysis' / analysis_nwb_file_name
        return str(analysis_nwb_file_abspath)

    @staticmethod
    def add_to_lock(analysis_file_name):
        """ Adds the specified analysis nwbfile to the file with the list of nwb files to be locked

        :param analysis_file_name: the name of an nwb file that has been inserted into the Nwbfile() schema
        :type nwb_file_name: string
        :return: None
        """
        key = {'analysis_file_name': analysis_file_name}
        # check to make sure the file exists
        assert len((AnalysisNwbfile() & key).fetch()) > 0, \
            f'Error adding {analysis_file_name} to lock file, not in AnalysisNwbfile() schema'
        lock_file = open(os.getenv('ANALYSIS_LOCK_FILE'), 'a+')
        lock_file.write(f'{analysis_file_name}\n')
        lock_file.close()

    def add_nwb_object(self, analysis_file_name, nwb_object):
        # TODO: change to add_object with checks for object type and a name parameter, which should be specified if
        # it is not an NWB container
        """
        Adds an nwb object to the analysis file in the scratch area and returns the nwb object id

        :param analysis_file_name: the name of the analysis nwb file
        :type analysis_file_name: str
        :param nwb_object: the nwb object created by pynwb
        :type nwb_object: NWBDataInterface
        :param processing_module: the name of the processing module to create, defaults to 'analysis'
        :type processing_module: str, optional
        :return: the nwb object id of the added object
        :rtype: str
        """
        # open the file, write the new object and return the object id
        with pynwb.NWBHDF5IO(path=self.get_abs_path(analysis_file_name), mode="a") as io:
            nwbf = io.read()
            nwbf.add_scratch(nwb_object)
            io.write(nwbf)
            return nwb_object.object_id

    def add_units(self, analysis_file_name, units, units_valid_times,
                  units_sort_interval, metrics=None, units_waveforms=None):
        """
        Given a units dictionary where each entry is (unit id, spike times)

        Parameters
        ----------
        analysis_file_name: str
            the name of the analysis nwb file
        units: dict
            dictionary of units and times with unit ids as keys
        units_valid_times: dict
            dictionary of units and valid times with unit ids as keys
        units_sort_interval: dict
            dictionary of units and sort_interval with unit ids as keys
        units_waveforms: dataframe
            optional dictionary of unit waveforms with unit ids as keys (optional)
        metrics: dict
            optional cluster metrics

        Returns
        -------
        the nwb object id of the Units object and the object id of the waveforms object ('' if None)
        """
        with pynwb.NWBHDF5IO(path=self.get_abs_path(analysis_file_name), mode="a") as io:
            nwbf = io.read()
            sort_intervals = list()
            if len(units.keys()):
                # Add spike times and valid time range for the sort
                for id in units.keys():
                    nwbf.add_unit(spike_times=units[id], id=id,
                                  # waveform_mean = units_templates[id],
                                  obs_intervals=units_valid_times[id])
                    sort_intervals.append(units_sort_interval[id])
                # Add a column for the sort interval (subset of valid time)
                nwbf.add_unit_column(name='sort_interval',
                                     description='the interval used for spike sorting',
                                     data=sort_intervals)
                # If metrics were specified, add one column per metric
                if metrics is not None:
                    for metric in list(metrics):
                        metric_data = metrics[metric].to_list()
                        print(f'Adding metric {metric} : {metric_data}')
                        nwbf.add_unit_column(name=metric,
                                             description=f'{metric} sorting metric',
                                             data=metric_data)
                # If the waveforms were specified, add them as a dataframe
                waveforms_object_id = ''
                if units_waveforms is not None:
                    waveforms_df = pd.DataFrame.from_dict(units_waveforms,
                                                          orient='index')
                    waveforms_df.columns = ['waveforms']
                    nwbf.add_scratch(waveforms_df, name='units_waveforms', notes='spike waveforms for each unit')
                    waveforms_object_id = nwbf.scratch['units_waveforms'].object_id

                io.write(nwbf)
                return nwbf.units.object_id, waveforms_object_id
            else:
                return ''

    def get_electrode_indices(self, analysis_file_name, electrode_ids):
        """
        Given an analysis NWB file name, returns the indices of the specified
        electrode_ids.

        :param analysis_file_name: analysis NWB file name
        :type analysi_file_name: str
        :param electrode_ids: array or list of electrode_ids
        :type electrode_ids: numpy array or list
        :return: electrode_indices (numpy array of indices)
        """
        with pynwb.NWBHDF5IO(path=self.get_abs_path(analysis_file_name), mode="a") as io:
            nwbf = io.read()
            return get_electrode_indices(nwbf.electrodes, electrode_ids)

    def cleanup(self, delete_files=False):
        """
        Removes the filepath entries for nwb files that are not in use.
        Does not delete the files themselves unless delete_files = True is
        specified. Run this after deleting the Nwbfile() entries themselves.

        Parameters
        ----------
        delete_files : bool
            True if original files be deleted (default False
        """
        self.external['analysis'].delete(delete_external_files=delete_files)


@schema
class NwbfileKachery(dj.Computed):
    definition = """
    -> Nwbfile
    ---
    nwb_file_sha1: varchar(40)  # the sha1 hash of the NWB file for kachery
    """

    def make(self, key):
        print('Computing SHA-1 and storing in kachery...')
        nwb_file_abs_path = Nwbfile.get_abs_path(key['nwb_file_name'])
        with ka.config(use_hard_links=True):
            kachery_path = ka.store_file(nwb_file_abs_path)
            key['nwb_file_sha1'] = ka.get_file_hash(kachery_path)
        self.insert1(key)


@schema
class AnalysisNwbfileKachery(dj.Computed):
    definition = """
    -> AnalysisNwbfile
    ---
    analysis_file_sha1: varchar(40)  # the sha1 hash of the file
    """

    def make(self, key):
        print('Computing SHA-1 and storing in kachery...')
        analysis_file_abs_path = AnalysisNwbfile().get_abs_path(key['analysis_file_name'])
        with ka.config(use_hard_links=True):
            kachery_path = ka.store_file(analysis_file_abs_path)
            key['analysis_file_sha1'] = ka.get_file_hash(kachery_path)
        self.insert1(key)

    # TODO: load from kachery and fetch_nwb
