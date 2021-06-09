import os
import pathlib

import datajoint as dj
import kachery as ka
import pandas as pd
import pynwb

from .dj_helper_fn import get_child_tables
from .nwb_helper_fn import get_electrode_indices, get_nwb_file

schema = dj.schema("common_nwbfile")

# define the fields that should be kept in AnalysisNWBFiles
NWB_KEEP_FIELDS = ('devices', 'electrode_groups', 'electrodes', 'experiment_description',
                   'experimenter', 'file_create_date', 'identifier', 'intervals',
                   'institution', 'lab', 'session_description', 'session_id',
                   'session_start_time', 'subject', 'timestamps_reference_time')


@schema
class Nwbfile(dj.Manual):
    definition = """
    # Table for holding the NWB files.
    nwb_file_name: varchar(255) # name of the NWB file
    ---
    nwb_file_abs_path: filepath@raw
    """

    def insert_from_relative_file_name(self, nwb_file_name):
        """Insert a new session from an existing NWB file.

        Parameters
        ----------
        nwb_file_name : str
            The relative path to the NWB file.
        """
        nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
        assert os.path.exists(
            nwb_file_abs_path), f'File does not exist: {nwb_file_abs_path}'

        self.insert1(dict(
            nwb_file_name=nwb_file_name,
            nwb_file_abs_path=nwb_file_abs_path,
        ), skip_duplicates=True)

    @staticmethod
    def get_abs_path(nwb_file_name):
        """Return the absolute path for a stored raw NWB file given just the file name.

        The NWB_DATAJOINT_BASE_DIR environment variable must be set.

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file that has been inserted into the Nwbfile() schema.

        Returns
        -------
        nwb_file_abspath : str
            The absolute path for the given file name.
        """
        base_dir = pathlib.Path(os.getenv('NWB_DATAJOINT_BASE_DIR', None))
        assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR or provide the base_dir argument'

        nwb_file_abspath = base_dir / 'raw' / nwb_file_name
        return str(nwb_file_abspath)

    @staticmethod
    def add_to_lock(nwb_file_name):
        """Add the specified NWB file to the file with the list of NWB files to be locked.

        The NWB_LOCK_FILE environment variable must be set.

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file that has been inserted into the Nwbfile() schema.
        """
        key = {'nwb_file_name': nwb_file_name}
        # check to make sure the file exists
        assert len((Nwbfile() & key).fetch()) > 0, \
            f'Error adding {nwb_file_name} to lock file, not in Nwbfile() schema'

        lock_file = open(os.getenv('NWB_LOCK_FILE'), 'a+')
        lock_file.write(f'{nwb_file_name}\n')
        lock_file.close()

    def cleanup(self, delete_files=False):
        """Remove the filepath entries for NWB files that are not in use.

        This does not delete the files themselves unless delete_files=True is specified
        Run this after deleting the Nwbfile() entries themselves.
        """
        self.external['raw'].delete(delete_external_files=delete_files)


# TODO: add_to_kachery will not work because we can't update the entry after it's been used in another table.
# We therefore need another way to keep track of the
@schema
class AnalysisNwbfile(dj.Manual):
    definition = """
    # Table for holding the NWB files that contain results of analysis, such as spike sorting.
    analysis_file_name : varchar(255)             # name of the file
    ---
    -> Nwbfile                                    # name of the parent NWB file. Used for naming and metadata copy
    analysis_file_abs_path: filepath@analysis     # the full path to the file
    analysis_file_description = '': varchar(255)  # an optional description of this analysis
    analysis_parameters = NULL: blob              # additional relevant parmeters. Currently used only for analyses
                                                  # that span multiple NWB files
    """

    def create(self, nwb_file_name):
        """Open the NWB file, create a copy, write the copy to disk and return the name of the new file.

        Note that this does NOT add the file to the schema; that needs to be done after data are written to it.

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file to be copied.

        Returns
        -------
        analysis_file_name : str
            The name of the new NWB file.
        """
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r', load_namespaces=True) as io:
            nwbf = io.read()
            # pop off the unnecessary elements to save space
            nwb_fields = nwbf.fields
            for field in nwb_fields:
                if field not in NWB_KEEP_FIELDS:
                    nwb_object = getattr(nwbf, field)
                    if isinstance(nwb_object, pynwb.core.LabelledDict):
                        for module in list(nwb_object.keys()):
                            nwb_object.pop(module)

            analysis_file_name = self.__get_new_file_name(nwb_file_name)
            # write the new file
            print(f'Writing new NWB file {analysis_file_name}')
            analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
                analysis_file_name)
            # export the new NWB file
            with pynwb.NWBHDF5IO(path=analysis_file_abs_path, mode='w', manager=io.manager) as export_io:
                export_io.export(io, nwbf)

        return analysis_file_name

    @classmethod
    def __get_new_file_name(cls, nwb_file_name):
        # get the current number of analysis files related to this nwb file
        n_analysis_files = len(
            (AnalysisNwbfile() & {'nwb_file_name': nwb_file_name}).fetch())
        # name the file, adding the number of files with preceeding zeros
        analysis_file_name = os.path.splitext(
            nwb_file_name)[0] + str(n_analysis_files).zfill(6) + '.nwb'
        return analysis_file_name

        # # get the list of names of analysis files related to this nwb file
        # names = (AnalysisNwbfile() & {'nwb_file_name': nwb_file_name}).fetch('analysis_file_name')
        # n1 = [str.replace(name, os.path.splitext(nwb_file_name)[0], '') for name in names]
        # max_analysis_file_num = max([int(str.replace(ext, '.nwb', '')) for ext in n1])
        # # name the file, adding the number of files with preceeding zeros
        # analysis_file_name = os.path.splitext(nwb_file_name)[0] + str(max_analysis_file_num+1).zfill(6) + '.nwb'
        # print(analysis_file_name)
        # return analysis_file_name

    @classmethod
    def copy(cls, nwb_file_name):
        """Make a copy of an analysis NWB file.

        Note that this does NOT add the file to the schema; that needs to be done after data are written to it.

        Parameters
        ----------
        nwb_file_name : str
            The name of the analysis NWB file to be copied.

        Returns
        -------
        analysis_file_name : str
            The name of the new NWB file.
        """
        nwb_file_abspath = AnalysisNwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r', load_namespaces=True) as io:
            nwbf = io.read()
            # get the current number of analysis files related to this nwb file
            original_nwb_file_name = (AnalysisNwbfile &
                                      {'analysis_file_name': nwb_file_name}).fetch('nwb_file_name')[0]
            analysis_file_name = cls.__get_new_file_name(
                original_nwb_file_name)
            # write the new file
            print(f'Writing new NWB file {analysis_file_name}...')
            analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
                analysis_file_name)
            # export the new NWB file
            with pynwb.NWBHDF5IO(path=analysis_file_abs_path, mode='w', manager=io.manager) as export_io:
                export_io.export(io, nwbf)

        return analysis_file_name

    def add(self, nwb_file_name, analysis_file_name):
        """Add the specified file to AnalysisNWBfile table.

        Parameters
        ----------
        nwb_file_name : str
            The name of the parent NWB file.
        analysis_file_name : str
            The name of the analysis NWB file that was created.
        """
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        key['analysis_file_name'] = analysis_file_name
        key['analysis_file_description'] = ''
        key['analysis_file_abs_path'] = AnalysisNwbfile.get_abs_path(
            analysis_file_name)
        self.insert1(key)

    @staticmethod
    def get_abs_path(analysis_nwb_file_name):
        """Return the absolute path for a stored analysis NWB file given just the file name.

        The NWB_DATAJOINT_BASE_DIR environment variable must be set.

        Parameters
        ----------
        analysis_nwb_file_name : str
            The name of the NWB file that has been inserted into the AnalysisNwbfile() schema

        Returns
        -------
        analysis_nwb_file_abspath : str
            The absolute path for the given file name.
        """
        base_dir = pathlib.Path(os.getenv('NWB_DATAJOINT_BASE_DIR', None))
        assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR environment variable.'

        analysis_nwb_file_abspath = str(
            base_dir / 'analysis' / analysis_nwb_file_name)
        return analysis_nwb_file_abspath

    @staticmethod
    def add_to_lock(analysis_file_name):
        """Add the specified analysis NWB file to the file with the list of nwb files to be locked.

        The ANALYSIS_LOCK_FILE environment variable must be set.

        Parameters
        ----------
        analysis_file_name : str
            The name of the NWB file that has been inserted into the AnalysisNwbfile() schema
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
        """Add an NWB object to the analysis file in the scratch area and returns the NWB object ID

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        nwb_object : pynwb.core.NWBDataInterface
            The NWB object created by PyNWB.

        Returns
        -------
        nwb_object_id : str
            The NWB object ID of the added object.
        """
        with pynwb.NWBHDF5IO(path=self.get_abs_path(analysis_file_name), mode="a", load_namespaces=True) as io:
            nwbf = io.read()
            nwbf.add_scratch(nwb_object)
            io.write(nwbf)
            return nwb_object.object_id

    def add_units(self, analysis_file_name, units, units_valid_times,
                  units_sort_interval, metrics=None, units_waveforms=None):
        """Add units, given a units dictionary where each entry is (unit id, spike times).

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        units : dict
            Dictionary of units and times with unit ids as keys.
        units_valid_times : dict
            Dictionary of units and valid times with unit ids as keys.
        units_sort_interval : dict
            Dictionary of units and sort_interval with unit ids as keys.
        units_waveforms : dataframe, optional
            Dictionary of unit waveforms with unit ids as keys.
        metrics : dict, optional
            Cluster metrics.

        Returns
        -------
        units_object_id, waveforms_object_id : str, str
            The NWB object id of the Units object and the object id of the waveforms object ('' if None)
        """
        with pynwb.NWBHDF5IO(path=self.get_abs_path(analysis_file_name), mode="a", load_namespaces=True) as io:
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
                # If the waveforms were specified, add them as a dataframe to scratch
                waveforms_object_id = ''
                if units_waveforms is not None:
                    waveforms_df = pd.DataFrame.from_dict(units_waveforms,
                                                          orient='index')
                    waveforms_df.columns = ['waveforms']
                    nwbf.add_scratch(
                        waveforms_df, name='units_waveforms', notes='spike waveforms for each unit')
                    waveforms_object_id = nwbf.scratch['units_waveforms'].object_id

                io.write(nwbf)
                return nwbf.units.object_id, waveforms_object_id
            else:
                return ''

    @classmethod
    def get_electrode_indices(cls, analysis_file_name, electrode_ids):
        """Given an analysis NWB file name, returns the indices of the specified electrode_ids.

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        electrode_ids : numpy array or list
            Array or list of electrode IDs.

        Returns
        -------
        electrode_indices : numpy array
            Array of indices in the electrodes table for the given electrode IDs.
        """
        nwbf = get_nwb_file(cls.get_abs_path(analysis_file_name))
        return get_electrode_indices(nwbf.electrodes, electrode_ids)

    def cleanup(self, delete_files=False):
        """Remove the filepath entries for NWB files that are not in use.

        Does not delete the files themselves unless delete_files=True is specified.
        Run this after deleting the Nwbfile() entries themselves.

        Parameters
        ----------
        delete_files : bool, optional
            Whether the original files should be deleted (default False).
        """
        self.external['analysis'].delete(delete_external_files=delete_files)

        # the usage of the above function to clean up AnalysisNwbfile table is as follows:
    @staticmethod
    def nightly_cleanup():
        from nwb_datajoint.common import common_nwbfile
        child_tables = get_child_tables(common_nwbfile.AnalysisNwbfile)
        (common_nwbfile.AnalysisNwbfile - child_tables).delete_quick()

        # a separate external files clean up required - this is to be done during times when no other transactions are in progress.
        common_nwbfile.schema.external['analysis'].delete(
            delete_external_files=True)


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
        analysis_file_abs_path = AnalysisNwbfile(
        ).get_abs_path(key['analysis_file_name'])
        with ka.config(use_hard_links=True):
            kachery_path = ka.store_file(analysis_file_abs_path)
            key['analysis_file_sha1'] = ka.get_file_hash(kachery_path)
        self.insert1(key)

    # TODO: load from kachery and fetch_nwb
