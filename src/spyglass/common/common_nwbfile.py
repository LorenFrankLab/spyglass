import os
import pathlib
import random
import stat
import string

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si
from hdmf.common import DynamicTable

from ..utils.dj_helper_fn import get_child_tables
from ..utils.nwb_helper_fn import get_electrode_indices, get_nwb_file

schema = dj.schema("common_nwbfile")

# define the fields that should be kept in AnalysisNWBFiles
NWB_KEEP_FIELDS = (
    "devices",
    "electrode_groups",
    "electrodes",
    "experiment_description",
    "experimenter",
    "file_create_date",
    "identifier",
    "intervals",
    "institution",
    "lab",
    "session_description",
    "session_id",
    "session_start_time",
    "subject",
    "timestamps_reference_time",
)


@schema
class Nwbfile(dj.Manual):
    definition = """
    # Table for holding the NWB files.
    nwb_file_name: varchar(255)   # name of the NWB file
    ---
    nwb_file_abs_path: filepath@raw
    INDEX (nwb_file_abs_path)
    """
    # NOTE the INDEX above is implicit from filepath@... above but needs to be explicit
    # so that alter() can work

    @classmethod
    def insert_from_relative_file_name(cls, nwb_file_name):
        """Insert a new session from an existing NWB file.

        Parameters
        ----------
        nwb_file_name : str
            The relative path to the NWB file.
        """
        nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
        assert os.path.exists(
            nwb_file_abs_path
        ), f"File does not exist: {nwb_file_abs_path}"

        key = dict()
        key["nwb_file_name"] = nwb_file_name
        key["nwb_file_abs_path"] = nwb_file_abs_path
        cls.insert1(key, skip_duplicates=True)

    @staticmethod
    def get_abs_path(nwb_file_name):
        """Return the absolute path for a stored raw NWB file given just the file name.

        The SPYGLASS_BASE_DIR environment variable must be set.

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file that has been inserted into the Nwbfile() schema.

        Returns
        -------
        nwb_file_abspath : str
            The absolute path for the given file name.
        """
        base_dir = pathlib.Path(os.getenv("SPYGLASS_BASE_DIR", None))
        assert (
            base_dir is not None
        ), "You must set SPYGLASS_BASE_DIR or provide the base_dir argument"

        nwb_file_abspath = base_dir / "raw" / nwb_file_name
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
        key = {"nwb_file_name": nwb_file_name}
        # check to make sure the file exists
        assert (
            len((Nwbfile() & key).fetch()) > 0
        ), f"Error adding {nwb_file_name} to lock file, not in Nwbfile() schema"

        lock_file = open(os.getenv("NWB_LOCK_FILE"), "a+")
        lock_file.write(f"{nwb_file_name}\n")
        lock_file.close()

    @staticmethod
    def cleanup(delete_files=False):
        """Remove the filepath entries for NWB files that are not in use.

        This does not delete the files themselves unless delete_files=True is specified
        Run this after deleting the Nwbfile() entries themselves.
        """
        schema.external["raw"].delete(delete_external_files=delete_files)


# TODO: add_to_kachery will not work because we can't update the entry after it's been used in another table.
# We therefore need another way to keep track of the
@schema
class AnalysisNwbfile(dj.Manual):
    definition = """
    # Table for holding the NWB files that contain results of analysis, such as spike sorting.
    analysis_file_name: varchar(255)               # name of the file
    ---
    -> Nwbfile                                     # name of the parent NWB file. Used for naming and metadata copy
    analysis_file_abs_path: filepath@analysis      # the full path to the file
    analysis_file_description = "": varchar(2000)  # an optional description of this analysis
    analysis_parameters = NULL: blob               # additional relevant parmeters. Currently used only for analyses
                                                   # that span multiple NWB files
    INDEX (analysis_file_abs_path)
    """
    # NOTE the INDEX above is implicit from filepath@... above but needs to be explicit
    # so that alter() can work

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
        with pynwb.NWBHDF5IO(
            path=nwb_file_abspath, mode="r", load_namespaces=True
        ) as io:
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
            print(f"Writing new NWB file {analysis_file_name}")
            analysis_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            # export the new NWB file
            with pynwb.NWBHDF5IO(
                path=analysis_file_abs_path, mode="w", manager=io.manager
            ) as export_io:
                export_io.export(io, nwbf)

        # change the permissions to only allow owner to write
        permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        os.chmod(analysis_file_abs_path, permissions)

        return analysis_file_name

    @classmethod
    def __get_new_file_name(cls, nwb_file_name):
        # each file ends with a random string of 10 digits, so we generate that string and redo if by some miracle
        # it's already there
        file_in_table = True
        while file_in_table:
            analysis_file_name = (
                os.path.splitext(nwb_file_name)[0]
                + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
                + ".nwb"
            )
            file_in_table = AnalysisNwbfile & {"analysis_file_name": analysis_file_name}

        return analysis_file_name

    @classmethod
    def __get_analysis_file_dir(cls, analysis_file_name: str):
        # strip off everything after and including the final underscore and return the result
        return analysis_file_name[0 : analysis_file_name.rfind("_")]

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
        with pynwb.NWBHDF5IO(
            path=nwb_file_abspath, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # get the current number of analysis files related to this nwb file
            query = AnalysisNwbfile & {"analysis_file_name": nwb_file_name}
            original_nwb_file_name = query.fetch("nwb_file_name")[0]
            analysis_file_name = cls.__get_new_file_name(original_nwb_file_name)
            # write the new file
            print(f"Writing new NWB file {analysis_file_name}...")
            analysis_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            # export the new NWB file
            with pynwb.NWBHDF5IO(
                path=analysis_file_abs_path, mode="w", manager=io.manager
            ) as export_io:
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
        key["nwb_file_name"] = nwb_file_name
        key["analysis_file_name"] = analysis_file_name
        key["analysis_file_description"] = ""
        key["analysis_file_abs_path"] = AnalysisNwbfile.get_abs_path(analysis_file_name)
        self.insert1(key)

    @staticmethod
    def get_abs_path(analysis_nwb_file_name):
        """Return the absolute path for a stored analysis NWB file given just the file name.

        The SPYGLASS_BASE_DIR environment variable must be set.

        Parameters
        ----------
        analysis_nwb_file_name : str
            The name of the NWB file that has been inserted into the AnalysisNwbfile() schema

        Returns
        -------
        analysis_nwb_file_abspath : str
            The absolute path for the given file name.
        """
        base_dir = pathlib.Path(os.getenv("SPYGLASS_BASE_DIR", None))
        assert (
            base_dir is not None
        ), "You must set SPYGLASS_BASE_DIR environment variable."

        # see if the file exists and is stored in the base analysis dir
        test_path = str(base_dir / "analysis" / analysis_nwb_file_name)

        if os.path.exists(test_path):
            return test_path
        else:
            # use the new path
            analysis_file_base_path = (
                base_dir
                / "analysis"
                / AnalysisNwbfile.__get_analysis_file_dir(analysis_nwb_file_name)
            )
            if not analysis_file_base_path.exists():
                os.mkdir(str(analysis_file_base_path))
            return str(analysis_file_base_path / analysis_nwb_file_name)

    def add_nwb_object(self, analysis_file_name, nwb_object, table_name="pandas_table"):
        # TODO: change to add_object with checks for object type and a name parameter, which should be specified if
        # it is not an NWB container
        """Add an NWB object to the analysis file in the scratch area and returns the NWB object ID

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        nwb_object : pynwb.core.NWBDataInterface
            The NWB object created by PyNWB.
        table_name : str (optional, defaults to 'pandas_table')
            The name of the DynamicTable made from a dataframe.

        Returns
        -------
        nwb_object_id : str
            The NWB object ID of the added object.
        """
        with pynwb.NWBHDF5IO(
            path=self.get_abs_path(analysis_file_name), mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            if isinstance(nwb_object, pd.DataFrame):
                dt_object = DynamicTable.from_dataframe(name=table_name, df=nwb_object)
                nwbf.add_scratch(dt_object)
                io.write(nwbf)
                return dt_object.object_id
            else:
                nwbf.add_scratch(nwb_object)
                io.write(nwbf)
                return nwb_object.object_id

    def add_units(
        self,
        analysis_file_name,
        units,
        units_valid_times,
        units_sort_interval,
        metrics=None,
        units_waveforms=None,
        labels=None,
    ):
        """Add units to analysis NWB file

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        units : dict
            keys are unit ids, values are spike times
        units_valid_times : dict
            Dictionary of units and valid times with unit ids as keys.
        units_sort_interval : dict
            Dictionary of units and sort_interval with unit ids as keys.
        units_waveforms : dict, optional
            Dictionary of unit waveforms with unit ids as keys.
        metrics : dict, optional
            Cluster metrics.
        labels : dict, optional
            Curation labels for clusters

        Returns
        -------
        units_object_id, waveforms_object_id : str, str
            The NWB object id of the Units object and the object id of the waveforms object ('' if None)
        """
        with pynwb.NWBHDF5IO(
            path=self.get_abs_path(analysis_file_name), mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            sort_intervals = list()
            if len(units.keys()):
                # Add spike times and valid time range for the sort
                for id in units.keys():
                    nwbf.add_unit(
                        spike_times=units[id],
                        id=id,
                        # waveform_mean = units_templates[id],
                        obs_intervals=units_valid_times[id],
                    )
                    sort_intervals.append(units_sort_interval[id])
                # Add a column for the sort interval (subset of valid time)
                nwbf.add_unit_column(
                    name="sort_interval",
                    description="the interval used for spike sorting",
                    data=sort_intervals,
                )
                # If metrics were specified, add one column per metric
                if metrics is not None:
                    for metric in metrics:
                        if metrics[metric]:
                            unit_ids = np.array(list(metrics[metric].keys()))
                            metric_values = np.array(list(metrics[metric].values()))
                            # sort by unit_ids and apply that sorting to values to ensure that things go in the right order
                            metric_values = metric_values[np.argsort(unit_ids)]
                            print(f"Adding metric {metric} : {metric_values}")
                            nwbf.add_unit_column(
                                name=metric,
                                description=f"{metric} metric",
                                data=metric_values,
                            )
                if labels is not None:
                    unit_ids = np.array(list(units.keys()))
                    for unit in unit_ids:
                        if unit not in labels:
                            labels[unit] = ""
                    label_values = np.array(list(labels.values()))
                    label_values = label_values[np.argsort(unit_ids)].tolist()
                    nwbf.add_unit_column(
                        name="label",
                        description="label given during curation",
                        data=label_values,
                    )
                # If the waveforms were specified, add them as a dataframe to scratch
                waveforms_object_id = ""
                if units_waveforms is not None:
                    waveforms_df = pd.DataFrame.from_dict(
                        units_waveforms, orient="index"
                    )
                    waveforms_df.columns = ["waveforms"]
                    nwbf.add_scratch(
                        waveforms_df,
                        name="units_waveforms",
                        notes="spike waveforms for each unit",
                    )
                    waveforms_object_id = nwbf.scratch["units_waveforms"].object_id

                io.write(nwbf)
                return nwbf.units.object_id, waveforms_object_id
            else:
                return ""

    def add_units_waveforms(
        self,
        analysis_file_name,
        waveform_extractor: si.WaveformExtractor,
        metrics=None,
        labels=None,
    ):
        """Add units to analysis NWB file along with the waveforms

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        waveform_extractor : si.WaveformExtractor object
        metrics : dict, optional
            Cluster metrics.
        labels : dict, optional
            Curation labels for clusters

        Returns
        -------
        units_object_id : str
            The NWB object id of the Units object
        """

        with pynwb.NWBHDF5IO(
            path=self.get_abs_path(analysis_file_name), mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            for id in waveform_extractor.sorting.get_unit_ids():
                # (spikes, samples, channels)
                waveforms = waveform_extractor.get_waveforms(unit_id=id)
                # (channels, spikes, samples)
                waveforms = np.moveaxis(waveforms, source=2, destination=0)
                nwbf.add_unit(
                    spike_times=waveform_extractor.sorting.get_unit_spike_train(
                        unit_id=id
                    ),
                    id=id,
                    electrodes=waveform_extractor.recording.get_channel_ids(),
                    waveforms=waveforms,
                )

            # The following is a rough sketch of AnalysisNwbfile().add_waveforms
            # analysis_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
            # or
            # nwbfile = pynwb.NWBFile(...)
            # (channels, spikes, samples)
            # wfs = [
            #         [     # elec 1
            #             [1, 2, 3],  # spike 1, [sample 1, sample 2, sample 3]
            #             [1, 2, 3],  # spike 2
            #             [1, 2, 3],  # spike 3
            #             [1, 2, 3]   # spike 4
            #         ], [  # elec 2
            #             [1, 2, 3],  # spike 1
            #             [1, 2, 3],  # spike 2
            #             [1, 2, 3],  # spike 3
            #             [1, 2, 3]   # spike 4
            #         ], [  # elec 3
            #             [1, 2, 3],  # spike 1
            #             [1, 2, 3],  # spike 2
            #             [1, 2, 3],  # spike 3
            #             [1, 2, 3]   # spike 4
            #         ]
            # ]
            # elecs = ... # DynamicTableRegion referring to three electrodes (rows) of the electrodes table
            # nwbfile.add_unit(spike_times=[1, 2, 3], electrodes=elecs, waveforms=wfs)

            # If metrics were specified, add one column per metric
            if metrics is not None:
                for metric_name, metric_dict in metrics.items():
                    print(f"Adding metric {metric_name} : {metric_dict}")
                    metric_data = metric_dict.values().to_list()
                    nwbf.add_unit_column(
                        name=metric_name, description=metric_name, data=metric_data
                    )
            if labels is not None:
                nwbf.add_unit_column(
                    name="label", description="label given during curation", data=labels
                )

            io.write(nwbf)
            return nwbf.units.object_id

    def add_units_metrics(self, analysis_file_name, metrics):
        """Add units to analysis NWB file along with the waveforms

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        waveform_extractor : si.WaveformExtractor object
        metrics : dict, optional
            Cluster metrics.
        labels : dict, optional
            Curation labels for clusters

        Returns
        -------
        units_object_id : str
            The NWB object id of the Units object
        """
        metric_names = list(metrics.keys())
        unit_ids = list(metrics[metric_names[0]].keys())
        with pynwb.NWBHDF5IO(
            path=self.get_abs_path(analysis_file_name), mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            for id in unit_ids:
                nwbf.add_unit(id=id)

            for metric_name, metric_dict in metrics.items():
                print(f"Adding metric {metric_name} : {metric_dict}")
                metric_data = list(metric_dict.values())
                nwbf.add_unit_column(
                    name=metric_name, description=metric_name, data=metric_data
                )

            io.write(nwbf)
            return nwbf.units.object_id

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

    @staticmethod
    def cleanup(delete_files=False):
        """Remove the filepath entries for NWB files that are not in use.

        Does not delete the files themselves unless delete_files=True is specified.
        Run this after deleting the Nwbfile() entries themselves.

        Parameters
        ----------
        delete_files : bool, optional
            Whether the original files should be deleted (default False).
        """
        schema.external["analysis"].delete(delete_external_files=delete_files)

    @staticmethod
    def nightly_cleanup():
        child_tables = get_child_tables(AnalysisNwbfile)
        (AnalysisNwbfile - child_tables).delete_quick()

        # a separate external files clean up required - this is to be done
        # during times when no other transactions are in progress.
        AnalysisNwbfile.cleanup(True)

        # also check to see whether there are directories in the spikesorting folder with this


@schema
class NwbfileKachery(dj.Computed):
    definition = """
    -> Nwbfile
    ---
    nwb_file_uri: varchar(200)  # the uri the NWB file for kachery
    """

    def make(self, key):
        print(f'Linking {key["nwb_file_name"]} and storing in kachery...')
        key["nwb_file_uri"] = kc.link_file(Nwbfile().get_abs_path(key["nwb_file_name"]))
        self.insert1(key)


@schema
class AnalysisNwbfileKachery(dj.Computed):
    definition = """
    -> AnalysisNwbfile
    ---
    analysis_file_uri: varchar(200)  # the uri of the file
    """

    def make(self, key):
        print(f'Linking {key["analysis_file_name"]} and storing in kachery...')
        key["analysis_file_uri"] = kc.link_file(
            AnalysisNwbfile().get_abs_path(key["analysis_file_name"])
        )
        self.insert1(key)
