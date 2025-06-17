import os
import random
import string
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import uuid4

import datajoint as dj
import h5py
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si
from hdmf.common import DynamicTable
from pynwb.core import ScratchData

from spyglass import __version__ as sg_version
from spyglass.settings import analysis_dir, raw_dir
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import get_child_tables
from spyglass.utils.nwb_hash import NwbfileHasher
from spyglass.utils.nwb_helper_fn import get_electrode_indices, get_nwb_file

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
class Nwbfile(SpyglassMixin, dj.Manual):
    definition = """
    # Table for holding the NWB files.
    nwb_file_name: varchar(64)   # name of the NWB file
    ---
    nwb_file_abs_path: filepath@raw
    INDEX (nwb_file_abs_path)
    """
    # NOTE the INDEX above is implicit from filepath@... above but needs to be
    # explicit so that alter() can work

    # NOTE: See #630, #664. Excessive key length.

    @classmethod
    def insert_from_relative_file_name(cls, nwb_file_name: str) -> None:
        """Insert a new session from an existing NWB file.

        Parameters
        ----------
        nwb_file_name : str
            The relative path to the NWB file.
        """
        nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name, new_file=True)

        assert os.path.exists(
            nwb_file_abs_path
        ), f"File does not exist: {nwb_file_abs_path}"

        key = dict()
        key["nwb_file_name"] = nwb_file_name
        key["nwb_file_abs_path"] = nwb_file_abs_path
        cls.insert1(key, skip_duplicates=True)

    def fetch_nwb(self):
        return [
            get_nwb_file(self.get_abs_path(file))
            for file in self.fetch("nwb_file_name")
        ]

    @classmethod
    def _get_file_name(cls, nwb_file_name: str) -> str:
        """Get valid nwb file name given substring."""
        query = cls & f'nwb_file_name LIKE "%{nwb_file_name}%"'

        if len(query) == 1:
            return query.fetch1("nwb_file_name")

        raise ValueError(
            f"Found {len(query)} matches for {nwb_file_name} in Nwbfile table:"
            + f" \n{query}"
        )

    @classmethod
    def get_file_key(cls, nwb_file_name: str) -> dict:
        """Return primary key using nwb_file_name substring."""
        return {"nwb_file_name": cls._get_file_name(nwb_file_name)}

    @classmethod
    def get_abs_path(
        cls, nwb_file_name: str, new_file: bool = False, **kwargs
    ) -> str:
        """Return absolute path for a stored raw NWB file given file name.

        The SPYGLASS_BASE_DIR must be set, either as an environment or part of
        dj.config['custom']. See spyglass.settings.load_config

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file that has been inserted into the Nwbfile()
            table. May be file substring. May include % wildcard(s).
        new_file : bool, optional
            Adding a new file to Nwbfile table. Defaults to False.

        Returns
        -------
        nwb_file_abspath : str
            The absolute path for the given file name.
        """
        if new_file:
            return raw_dir + "/" + nwb_file_name

        return raw_dir + "/" + cls._get_file_name(nwb_file_name)

    @staticmethod
    def add_to_lock(nwb_file_name: str) -> None:
        """Add the specified NWB file to the list of locked items.

        The NWB_LOCK_FILE environment variable must be set to the path of the
        lock file, listing locked NWB files.

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file in the Nwbfile table.
        """
        if not (Nwbfile() & {"nwb_file_name": nwb_file_name}):
            raise FileNotFoundError(
                f"File not found in Nwbfile table. Cannot lock {nwb_file_name}"
            )

        with open(os.getenv("NWB_LOCK_FILE"), "a+") as lock_file:
            lock_file.write(f"{nwb_file_name}\n")

    @staticmethod
    def cleanup(delete_files: bool = False) -> None:
        """Remove the filepath entries for NWB files that are not in use.

        This does not delete the files themselves unless delete_files=True is
        specified. Run this after deleting the Nwbfile() entries themselves.
        """
        schema.external["raw"].delete(delete_external_files=delete_files)


@schema
class AnalysisNwbfile(SpyglassMixin, dj.Manual):
    definition = """
    # Table for NWB files that contain results of analysis.
    analysis_file_name: varchar(64)                # name of the file
    ---
    -> Nwbfile                                     # name of the parent NWB file. Used for naming and metadata copy
    analysis_file_abs_path: filepath@analysis      # the full path to the file
    analysis_file_description = "": varchar(2000)  # an optional description of this analysis
    analysis_parameters = NULL: blob               # additional relevant parameters. Currently used only for analyses
                                                   # that span multiple NWB files
    INDEX (analysis_file_abs_path)
    """
    # NOTE the INDEX above is implicit from filepath@...
    # above but needs to be explicit so that alter() can work

    # See #630, #664. Excessive key length.

    _creation_times = {}

    def create(
        self,
        nwb_file_name: str,
        recompute_file_name: Optional[str] = None,
        alternate_dir: Optional[Union[str, Path]] = None,
        restrict_permission: Optional[bool] = False,
    ) -> str:
        """Open the NWB file, create copy, write to disk and return new name.

        Note that this does NOT add the file to the schema; that needs to be
        done after data are written to it.

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file to be copied.
        recompute_file_name : str, optional
            The name of the file to be regenerated. Defaults to None.
        alternate_dir : Union[str, Path], Optional
            An alternate directory to store the file. Defaults to analysis_dir.
        restrict_permission : bool, optional
            Default False, no permission restriction (666). If True, restrict
            write permissions to owner only.

        Returns
        -------
        analysis_file_name : str
            The name of the new NWB file.
        """
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        alter_source_script = False
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
            # add the version of spyglass that created this file
            if nwbf.source_script is None:
                nwbf.source_script = f"spyglass={sg_version}"
            else:
                alter_source_script = True

            analysis_file_name = (
                recompute_file_name or self.__get_new_file_name(nwb_file_name)
            )

            # write the new file
            if not recompute_file_name:
                logger.info(f"Writing new NWB file {analysis_file_name}")

            analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
                analysis_file_name, from_schema=bool(recompute_file_name)
            )

            if alternate_dir:  # override the default analysis_dir for recompute
                relative = Path(analysis_file_abs_path).relative_to(
                    analysis_dir
                )
                analysis_file_abs_path = Path(alternate_dir) / relative

            # export the new NWB file
            parent_path = Path(analysis_file_abs_path).parent
            if not parent_path.exists():
                parent_path.mkdir(parents=True)
            with pynwb.NWBHDF5IO(
                path=analysis_file_abs_path, mode="w", manager=io.manager
            ) as export_io:
                export_io.export(io, nwbf)

        if alter_source_script:
            self._alter_spyglass_version(analysis_file_abs_path)

        # create a new object id for the file
        with h5py.File(analysis_file_abs_path, "a") as f:
            f.attrs["object_id"] = str(uuid4())

        # permissions: 0o644 (only owner write), 0o666 (open)
        permissions = 0o644 if restrict_permission else 0o666
        os.chmod(analysis_file_abs_path, permissions)

        return analysis_file_name

    @staticmethod
    def _alter_spyglass_version(nwb_file_path: str) -> None:
        """Change the source script to the current version of spyglass"""
        with h5py.File(nwb_file_path, "a") as f:
            f["/general/source_script"][()] = f"spyglass={sg_version}"

    @classmethod
    def __get_new_file_name(cls, nwb_file_name: str) -> str:
        # each file ends with a random string of 10 digits, so we generate that
        # string and redo if by some miracle it's already there
        file_in_table = True
        while file_in_table:
            analysis_file_name = (
                os.path.splitext(nwb_file_name)[0]
                + "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=10)
                )
                + ".nwb"
            )
            file_in_table = AnalysisNwbfile & {
                "analysis_file_name": analysis_file_name
            }

        return analysis_file_name

    @classmethod
    def __get_analysis_file_dir(cls, analysis_file_name: str) -> str:
        """Strip off final underscore and remaining chars, return the result."""
        return analysis_file_name[0 : analysis_file_name.rfind("_")]

    @classmethod
    def copy(cls, nwb_file_name: str):
        """Make a copy of an analysis NWB file.

        Note that this does NOT add the file to the schema; that needs to be
        done after data are written to it.

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
            logger.info(f"Writing new NWB file {analysis_file_name}...")
            analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
                analysis_file_name
            )
            # export the new NWB file
            with pynwb.NWBHDF5IO(
                path=analysis_file_abs_path, mode="w", manager=io.manager
            ) as export_io:
                export_io.export(io, nwbf)

        return analysis_file_name

    def add(self, nwb_file_name: str, analysis_file_name: str) -> None:
        """Add the specified file to AnalysisNWBfile table.

        Parameters
        ----------
        nwb_file_name : str
            The name of the parent NWB file.
        analysis_file_name : str
            The name of the analysis NWB file that was created.
        """
        key = {
            "nwb_file_name": nwb_file_name,
            "analysis_file_name": analysis_file_name,
            "analysis_file_description": "",
            "analysis_file_abs_path": AnalysisNwbfile.get_abs_path(
                analysis_file_name
            ),
        }
        self.insert1(key)

    @classmethod
    def get_abs_path(
        cls, analysis_nwb_file_name: str, from_schema: Optional[bool] = False
    ) -> str:
        """Return the absolute path for an analysis NWB file given the name.

        The spyglass config from settings.py must be set.

        Parameters
        ----------
        analysis_nwb_file_name : str
            The name of the NWB file in AnalysisNwbfile.
        from_schema : bool, optional
            If true, get the file path from the schema externals table, skipping
            checksum and file existence checks. Defaults to False.

        Returns
        -------
        analysis_nwb_file_abspath : str
            The absolute path for the given file name.
        """
        if from_schema:  # Skips checksum check
            query = (
                schema.external["analysis"]
                & f"filepath LIKE '%{analysis_nwb_file_name}'"
            )
            if len(query) == 1:  # Else try the standard way
                return Path(analysis_dir) / query.fetch1("filepath")
            logger.warning(
                f"Found {len(query)} files for: {analysis_nwb_file_name}"
            )

        # If an entry exists in the database get the stored datajoint filepath
        file_key = {"analysis_file_name": analysis_nwb_file_name}
        if cls & file_key:
            try:
                # runs if file exists locally
                return (cls & file_key).fetch1(
                    "analysis_file_abs_path", log_export=False
                )
            except FileNotFoundError as e:
                # file exists in database but not locally
                # parse the intended path from the error message
                return str(e).split(": ")[1].replace("'", "")

        # File not in database, define what it should be
        # see if the file exists and is stored in the base analysis dir
        test_path = f"{analysis_dir}/{analysis_nwb_file_name}"

        if os.path.exists(test_path):
            return test_path
        else:
            # use the new path
            analysis_file_base_path = Path(
                analysis_dir
            ) / AnalysisNwbfile.__get_analysis_file_dir(analysis_nwb_file_name)
            if not analysis_file_base_path.exists():
                os.mkdir(str(analysis_file_base_path))
            return str(analysis_file_base_path / analysis_nwb_file_name)

    def add_nwb_object(
        self,
        analysis_file_name: str,
        nwb_object: pynwb.core.NWBDataInterface,
        table_name: Optional[str] = "pandas_table",
    ):
        """Add an NWB object to the analysis file and return the NWB object ID

        Adds object to the scratch space of the NWB file.

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        nwb_object : pynwb.core.NWBDataInterface
            The NWB object created by PyNWB.
        table_name : str, optional
            The name of the pynwb object made from a passed dataframe or array.
            Defaults to "pandas_table" or "numpy_array" for dataframes and arrays
            respectively.

        Returns
        -------
        nwb_object_id : str
            The NWB object ID of the added object.
        """
        with pynwb.NWBHDF5IO(
            path=self.get_abs_path(analysis_file_name),
            mode="a",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            # convert to pynwb object if it is a dataframe or array
            if isinstance(nwb_object, pd.DataFrame):
                nwb_object = DynamicTable.from_dataframe(
                    name=table_name or "pandas_table", df=nwb_object
                )
            elif isinstance(nwb_object, np.ndarray):
                nwb_object = ScratchData(
                    name=table_name or "numpy_array",
                    data=nwb_object,
                )
            if nwb_object.name in nwbf.scratch:
                raise ValueError(
                    f"Object with name '{nwb_object.name}' already exists in "
                    + f"{analysis_file_name}. Please pass a different name argument "
                    + "to AnalysisNwbfile.add_nwb_object()."
                )
            nwbf.add_scratch(nwb_object)
            io.write(nwbf)
            return nwb_object.object_id

    def get_hash(
        self,
        analysis_file_name: str,
        from_schema: Optional[bool] = False,
        precision_lookup: Optional[Dict[str, int]] = None,
        return_hasher: Optional[bool] = False,
    ) -> Union[str, NwbfileHasher]:
        """Return the hash of the file contents.

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        from_schema : bool, Optional
            If true, get the file path from the schema externals table, skipping
            checksum and file existence checks. Defaults to False.
        precision_lookup : dict, Optional
            A dictionary of object names and rounding precisions, dictating the
            level of precision to which the data should be rounded before
            hashing. Defaults to None, no rounding.
        return_hasher: bool, Optional
            If true, return the hasher object instead of the hash. Defaults to
            False.

        Returns
        -------
        hash : [str, NwbfileHasher]
            The hash of the file contents or the hasher object itself.
        """
        hasher = NwbfileHasher(
            self.get_abs_path(analysis_file_name, from_schema=from_schema),
            precision_lookup=precision_lookup,
        )
        return hasher if return_hasher else hasher.hash

    def _update_external(self, analysis_file_name: str, hash: str):
        """Update the external contents checksum for an analysis file.

        Ensures that the file contents match the hash. If not, raise an error.

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        hash : str
            The hash of the file contents as calculated by NwbfileHasher.
            If the hash does not match the file contents, the file and
            downstream entries are deleted.

        Raises
        ------
        ValueError
            If the hash does not match the file contents, the file is deleted
            and a ValueError is raised.
        """
        file_path = self.get_abs_path(analysis_file_name, from_schema=True)
        new_hash = self.get_hash(analysis_file_name, from_schema=True)

        if hash != new_hash:
            Path(file_path).unlink()  # remove mismatched file
            raise ValueError(
                f"Failed to recompute {analysis_file_name}.",
                "Could not exactly replicate file content. Please check ",
                "UserEnvironment table for mismatched dependencies.",
            )

        external_tbl = schema.external["analysis"]
        file_path = (
            Path(self.__get_analysis_file_dir(analysis_file_name))
            / analysis_file_name
        )
        key = (external_tbl & f"filepath = '{file_path}'").fetch1()
        abs_path = Path(analysis_dir) / file_path
        key.update(
            {
                "contents_hash": dj.hash.uuid_from_file(abs_path),
                "size": abs_path.stat().st_size,
            }
        )

        external_tbl.update1(key)

    def add_units(
        self,
        analysis_file_name: str,
        units: dict,
        units_valid_times: dict,
        units_sort_interval: dict,
        metrics: Optional[dict] = None,
        units_waveforms: Optional[dict] = None,
        labels: Optional[dict] = None,
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
            The NWB object id of the Units object and the object id of the
            waveforms object ('' if None)
        """
        with pynwb.NWBHDF5IO(
            path=self.get_abs_path(analysis_file_name),
            mode="a",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            sort_intervals = list()

            if not len(units.keys()):
                return ""

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
            metrics = metrics or []  # do nothing if metrics is None
            for metric in metrics:
                if not metrics.get(metric):
                    continue

                unit_ids = np.array(list(metrics[metric].keys()))
                metric_values = np.array(list(metrics[metric].values()))

                # sort by unit_ids and apply that sorting to values
                # to ensure that things go in the right order

                metric_values = metric_values[np.argsort(unit_ids)]
                logger.info(f"Adding metric {metric} : {metric_values}")
                nwbf.add_unit_column(
                    name=metric,
                    description=f"{metric} metric",
                    data=metric_values,
                )

            if labels is not None:
                unit_ids = np.array(list(units.keys()))
                labels.update(
                    {unit: "" for unit in unit_ids if unit not in labels}
                )
                label_values = np.array(list(labels.values()))
                label_values = label_values[np.argsort(unit_ids)].tolist()
                nwbf.add_unit_column(
                    name="label",
                    description="label given during curation",
                    data=label_values,
                )

            # If the waveforms were specified, add them as a df to scratch
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

    def add_units_waveforms(
        self,
        analysis_file_name: str,
        waveform_extractor: si.WaveformExtractor,
        metrics: Optional[dict] = None,
        labels: Optional[dict] = None,
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
            path=self.get_abs_path(analysis_file_name),
            mode="a",
            load_namespaces=True,
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
            # analysis_file_name =
            #   AnalysisNwbfile().create(key['nwb_file_name'])
            #   or
            #   nwbfile = pynwb.NWBFile(...)
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
            # elecs = ... # DynamicTableRegion referring to three electrodes
            #               (rows) of the electrodes table
            # nwbfile.add_unit(spike_times=[1, 2, 3], electrodes=elecs,
            #                  waveforms=wfs)

            # If metrics were specified, add one column per metric
            if metrics is not None:
                for metric_name, metric_dict in metrics.items():
                    logger.info(f"Adding metric {metric_name} : {metric_dict}")
                    metric_data = metric_dict.values().to_list()
                    nwbf.add_unit_column(
                        name=metric_name,
                        description=metric_name,
                        data=metric_data,
                    )
            if labels is not None:
                nwbf.add_unit_column(
                    name="label",
                    description="label given during curation",
                    data=labels,
                )

            io.write(nwbf)
            return nwbf.units.object_id

    def add_units_metrics(self, analysis_file_name: str, metrics: dict):
        """Add units to analysis NWB file along with the waveforms

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        metrics : dict
            Cluster metrics.

        Returns
        -------
        units_object_id : str
            The NWB object id of the Units object
        """
        metric_names = list(metrics.keys())
        unit_ids = list(metrics[metric_names[0]].keys())
        with pynwb.NWBHDF5IO(
            path=self.get_abs_path(analysis_file_name),
            mode="a",
            load_namespaces=True,
        ) as io:
            nwbf = io.read()
            for id in unit_ids:
                nwbf.add_unit(id=id)

            for metric_name, metric_dict in metrics.items():
                logger.info(f"Adding metric {metric_name} : {metric_dict}")
                metric_data = list(metric_dict.values())
                nwbf.add_unit_column(
                    name=metric_name, description=metric_name, data=metric_data
                )

            io.write(nwbf)
            return nwbf.units.object_id

    @classmethod
    def get_electrode_indices(
        cls, analysis_file_name: str, electrode_ids: np.array
    ):
        """Returns indices of the specified electrode_ids for an analysis file.

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        electrode_ids : numpy array or list
            Array or list of electrode IDs.

        Returns
        -------
        electrode_indices : numpy array
            Array of indices in the electrodes table for the given electrode
            IDs.
        """
        nwbf = get_nwb_file(cls.get_abs_path(analysis_file_name))
        return get_electrode_indices(nwbf.electrodes, electrode_ids)

    @staticmethod
    def cleanup_external(delete_files=False):
        """Remove the filepath entries for NWB files that are not in use.

        Does not delete the files themselves unless delete_files=True is
        specified. Run this after deleting the Nwbfile() entries themselves.

        Parameters
        ----------
        delete_files : bool, optional
            Whether the original files should be deleted (default False).
        """
        schema.external["analysis"].delete(delete_external_files=delete_files)

    @staticmethod
    def cleanup():
        """Clean up orphaned AnalysisNwbfile entries and external files."""
        child_tables = get_child_tables(AnalysisNwbfile)
        (AnalysisNwbfile - child_tables).delete_quick()

        # a separate external files clean up required - this is to be done
        # during times when no other transactions are in progress.
        AnalysisNwbfile.cleanup_external(delete_files=True)

    def log(self, *args, **kwargs):
        """Null log method. Revert to _disabled_log to turn back on."""
        logger.debug("Logging disabled.")
