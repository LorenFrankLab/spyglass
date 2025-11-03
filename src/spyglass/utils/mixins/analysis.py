import os
import random
import string
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import uuid4

import datajoint as dj
import h5py
import numpy as np
import pandas as pd
import pynwb
import scipy.io
import spikeinterface as si
from datajoint.table import Table
from hdmf.common import DynamicTable
from pynwb.core import ScratchData

from spyglass.utils.dj_helper_fn import get_child_tables
from spyglass.utils.mixins.base import BaseMixin
from spyglass.utils.nwb_hash import NwbfileHasher
from spyglass.utils.nwb_helper_fn import get_electrode_indices, get_nwb_file

# Only differs from the common AnalysisNwbfile in adding 'Custom' to heading
ENFORCED_DEFINITION = """
# Custom table for NWB files that contain results of analysis.
analysis_file_name: varchar(64)                # name of the file
---
-> Nwbfile                                     # name of the parent NWB file. Used for naming and metadata copy
analysis_file_abs_path: filepath@analysis      # the full path to the file
analysis_file_description = "": varchar(2000)  # an optional description of this analysis
analysis_parameters = NULL: blob               # additional relevant parameters. Currently used only for analyses
                                               # that span multiple NWB files
INDEX (analysis_file_abs_path)
"""

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

HASH_ERROR_MSG = (
    "WHAT: The recomputed file contents don't match the expected hash.\n"
    "Expected hash: {hash}\n"
    "Actual hash:   {new_hash}\n\n"
    "WHY: This usually happens when:\n"
    "- Analysis code changed between runs (non-deterministic behavior)\n"
    "- Package versions differ (check environment with UserEnvironment table)\n"
    "- Random seeds not set consistently\n"
    "- Floating point precision varies across machines\n\n"
    "HOW TO FIX:\n"
    "1. Ensure analysis code is deterministic (set random seeds)\n"
    "2. Check environment matches: (UserEnvironment & 'user=YOU').fetch()\n"
    "3. If legitimate change, delete old entry and recompute\n\n"
    "NOTE: Mismatched file at '{file_path}' was deleted automatically.\n\n"
    "See: docs/troubleshooting.md#checksum-mismatch"
)


class AnalysisMixin(BaseMixin):
    """Provides analysis file management for AnalysisNwbfile tables.

    This mixin provides core functionality for both common and custom
    AnalysisNwbfile tables including file creation, NWB object management,
    cleanup/orphan detection, and export integration (copy-to-common).

    Key Methods:
        build() - RECOMMENDED: Create builder for safe file creation (context manager)
        create() - Legacy: Create new analysis file with unique suffix
        add() - Legacy: Register analysis file in table
        get_file_path() - Get absolute path to analysis file
        cleanup() - Remove orphaned files across all custom tables
        _copy_to_common() - Copy entries to common table during export

    The build() method returns an AnalysisFileBuilder context manager that
    enforces the CREATE → POPULATE → REGISTER lifecycle, preventing common
    errors like forgetting registration or modifying registered files.

    This mixin is used by SpyglassAnalysis for custom tables and directly
    inherited by the common AnalysisNwbfile table.
    """

    _creation_times = {}
    _cached_analysis_dir = None

    # ---------------------------- Table management ----------------------------
    @property
    def _enforced_definition(self) -> str:
        """Replace definition win enforced definition."""
        return ENFORCED_DEFINITION

    def _register_table(self) -> None:
        from spyglass.common.common_nwbfile import AnalysisRegistry

        AnalysisRegistry().insert1(self.full_table_name)

    def _copy_to_common(self, file_names: list = None) -> None:
        """Copy entries from this custom table to the common AnalysisNwbfile."""
        from spyglass.common.common_nwbfile import AnalysisNwbfile

        # Build restriction based on file_names
        if file_names is None:
            restr_table = self
        else:
            # Normalize to list
            if isinstance(file_names, str):
                file_names = [file_names]

            # Build appropriate restriction for single or multiple files
            if len(file_names) == 1:
                restr = f'analysis_file_name = "{file_names[0]}"'
            else:
                restr = f"analysis_file_name in {tuple(file_names)}"

            restr_table = self & restr

        entries = restr_table.fetch(as_dict=True)

        if not entries:
            return  # nothing to copy

        AnalysisNwbfile().insert(entries, skip_duplicates=True)

        self._logger.debug(
            f"Copied {len(entries)} entries to common: "
            f"{[e['analysis_file_name'] for e in entries]}"
        )

    # --------------------------- NWB file management -------------------------

    @cached_property
    def _analysis_dir(self) -> str:
        """Analysis directory from settings (cached at class level)."""
        if self.__class__._cached_analysis_dir is None:
            from spyglass.settings import analysis_dir

            self.__class__._cached_analysis_dir = analysis_dir
        return self.__class__._cached_analysis_dir

    @cached_property
    def _nwb_table(self) -> Table:
        from spyglass.common import Nwbfile

        return Nwbfile

    @cached_property
    def _ext_tbl(self) -> Table:
        """Return the external table for this schema."""
        context = self.heading.table_info.get("context")
        # Fallback for FreeTable null context
        schema = context.get("schema") if context else dj.Schema(self.database)
        return schema.external["analysis"]

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
        nwb_file_abspath = self._nwb_table.get_abs_path(nwb_file_name)
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

            # pop off optogenetic_epochs if it exists
            if (
                "intervals" in nwb_fields
                and "optogenetic_epochs" in nwbf.intervals
            ):
                nwbf.intervals.pop("optogenetic_epochs")

            # add the version of spyglass that created this file
            if nwbf.source_script is None:
                nwbf.source_script = self._logged_env_info()
            else:
                alter_source_script = True

            analysis_file_name = (
                recompute_file_name or self.__get_new_file_name(nwb_file_name)
            )

            # write the new file
            if not recompute_file_name:
                self._logger.info(f"Writing new NWB file {analysis_file_name}")

            analysis_file_abs_path = self.get_abs_path(
                analysis_file_name, from_schema=bool(recompute_file_name)
            )

            if alternate_dir:  # override the default analysis_dir for recompute
                relative = Path(analysis_file_abs_path).relative_to(
                    self._analysis_dir
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

    def _alter_spyglass_version(self, nwb_file_path: str) -> None:
        """Change the source script to the current version of spyglass"""
        with h5py.File(nwb_file_path, "a") as f:
            f["/general/source_script"][()] = self._logged_env_info()

    def _logged_env_info(self) -> str:
        """Get the environment information for logging."""
        sg_version = self._spyglass_version
        env_info = f"spyglass={sg_version} \n\n"
        env_info += "Python Environment:\n"
        python_env = subprocess.check_output(
            ["conda", "env", "export"], text=True
        )
        env_info += python_env
        return env_info

    @classmethod
    def __get_new_file_name(cls, nwb_file_name: str) -> str:
        """Generate a new unique file name based on the original NWB file name.

        Adds a random string of 10 uppercase letters and digits to the base
        name of the original NWB file. Ensures that the new file name is not
        already in this table and that the file does not already exist in the
        analysis directory.

        Parameters
        ----------
        nwb_file_name : str
            The name of the original NWB file.

        Returns
        -------
        analysis_file_name : str
            The name of the new NWB file.
        """
        str_options = string.ascii_uppercase + string.digits

        file_in_table = True  # file exists, may not be on disk
        file_exist = True  # file exists on disk, may be in different table

        while file_in_table or file_exist:
            rand_str = "".join(random.choices(str_options, k=10))
            fname = os.path.splitext(nwb_file_name)[0] + rand_str + ".nwb"
            file_dict = dict(analysis_file_name=fname)
            file_in_table = bool(cls() & file_dict)
            file_exist = cls.__get_analysis_path(fname).exists()

        # Create the empty file to reserve the name before returning
        # Avoids conflicts from multiple AnalysisNwbfile instances
        reservation_path = cls.__get_analysis_path(fname)
        reservation_path.parent.mkdir(parents=True, exist_ok=True)
        reservation_path.touch()

        return fname

    @classmethod
    def __get_analysis_file_dir(cls, fname: str) -> str:
        """Strip off final underscore and remaining chars, return the result."""
        return fname[0 : fname.rfind("_")]

    @classmethod
    def __get_file_parent(cls, fname: str, relative=True) -> Path:
        """Get the parent dir name for an analysis NWB file."""
        return Path(cls()._analysis_dir) / cls.__get_analysis_file_dir(fname)

    @classmethod
    def __get_analysis_path(cls, fname: str, relative: bool = False) -> Path:
        """Get the path for an analysis NWB file.

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis NWB file.
        relative : bool, optional
            If true, return the path relative to analysis_dir. Defaults False.


        Returns
        -------
        path : str
            The path for the analysis NWB file.
        """
        abs_path = cls.__get_file_parent(fname) / fname
        return (
            abs_path.relative_to(cls()._analysis_dir) if relative else abs_path
        )

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
        nwb_file_abspath = cls().get_abs_path(nwb_file_name)

        with pynwb.NWBHDF5IO(
            path=nwb_file_abspath, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # get the current number of analysis files related to this nwb file
            query = cls & {"analysis_file_name": nwb_file_name}
            original_nwb_file_name = query.fetch("nwb_file_name")[0]
            analysis_file_name = cls.__get_new_file_name(original_nwb_file_name)
            # write the new file
            cls()._logger.info(f"Writing new NWB file {analysis_file_name}...")
            analysis_file_abs_path = cls().get_abs_path(analysis_file_name)
            # export the new NWB file
            with pynwb.NWBHDF5IO(
                path=analysis_file_abs_path, mode="w", manager=io.manager
            ) as export_io:
                export_io.export(io, nwbf)

        return analysis_file_name

    def add(self, nwb_file_name: str, analysis_file_name: str) -> None:
        """Add the specified file to table.

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
            "analysis_file_abs_path": self.get_abs_path(analysis_file_name),
        }
        self.insert1(key)

    def build(self, nwb_file_name: str):
        """Create a builder for safe analysis file creation.

        Returns context manager that handles CREATE → POPULATE → REGISTER
        lifecycle with automatic state tracking and error prevention.

        This is the recommended way to create analysis files. The builder:
        - Automatically registers files on successful exit
        - Prevents modification of registered files (state checks)
        - Logs failed files for cleanup on exceptions
        - Provides clear error messages for invalid operations

        Parameters
        ----------
        nwb_file_name : str
            Parent NWB file name

        Returns
        -------
        builder : AnalysisFileBuilder
            Context manager for file lifecycle

        Examples
        --------
        Basic usage:
        >>> with AnalysisNwbfile().build(nwb_file_name) as builder:
        ...     builder.add_nwb_object(my_data, "results")
        ...     file = builder.analysis_file_name
        # File automatically registered on exit!

        Multiple operations:
        >>> with AnalysisNwbfile().build(nwb_file_name) as builder:
        ...     builder.add_nwb_object(position, "position")
        ...     builder.add_nwb_object(velocity, "velocity")
        ...     file = builder.analysis_file_name

        Direct I/O for complex cases:
        >>> with AnalysisNwbfile().build(nwb_file_name) as builder:
        ...     with builder.open_for_write() as io:
        ...         nwbf = io.read()
        ...         nwbf.add_unit(spike_times=times, id=unit_id)
        ...         io.write(nwbf)

        See Also
        --------
        AnalysisFileBuilder : Full API documentation
        create : Legacy method for file creation
        add : Legacy method for registration
        """
        from spyglass.utils.mixins.analysis_builder import AnalysisFileBuilder

        return AnalysisFileBuilder(self, nwb_file_name)

    @classmethod
    def get_abs_path(
        cls, analysis_nwb_file_name: str, from_schema: Optional[bool] = False
    ) -> str:
        """Return the absolute path for an analysis NWB file given the name.

        The spyglass config from settings.py must be set.

        Parameters
        ----------
        analysis_nwb_file_name : str
            The name of the NWB file in this table.
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
                cls()._ext_tbl & f"filepath LIKE '%{analysis_nwb_file_name}'"
            )
            if len(query) == 1:  # Else try the standard way
                return Path(cls()._analysis_dir) / query.fetch1("filepath")
            cls()._logger.warning(
                f"Found {len(query)} files for: {analysis_nwb_file_name}"
            )

        # If an entry exists in the database get the stored datajoint filepath
        file_key = {"analysis_file_name": analysis_nwb_file_name}
        query = cls() & file_key
        if bool(query):
            try:  # runs if file exists locally
                return query.fetch1("analysis_file_abs_path", log_export=False)
            except FileNotFoundError as e:
                # file exists in database but not locally
                # parse the intended path from the error message
                return str(e).split(": ")[1].replace("'", "")

        # File not in database, define what it should be
        # see if the file exists and is stored in the base analysis dir
        test_path = f"{cls()._analysis_dir}/{analysis_nwb_file_name}"

        if Path(test_path).exists():
            return test_path
        else:
            # use the new path
            analysis_file_base_path = Path(
                cls()._analysis_dir
            ) / cls.__get_analysis_file_dir(analysis_nwb_file_name)
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
                    + f"{analysis_file_name}. Please pass a different name "
                    + "argument to AnalysisNwbfile.add_nwb_object()."
                )
            nwbf.add_scratch(nwb_object)
            io.write(nwbf)
            return nwb_object.object_id

    # -------------------------------- Hashing --------------------------------

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
                f"Checksum mismatch for analysis file '{file_path}'."
                + HASH_ERROR_MSG.format(
                    hash=hash, new_hash=new_hash, file_path=file_path
                )
            )

        file_path = self.__get_analysis_path(analysis_file_name, relative=True)
        key = (self._ext_tbl & f"filepath = '{str(file_path)}'").fetch1()
        abs_path = Path(self._analysis_dir) / file_path
        key.update(
            {
                "contents_hash": dj.hash.uuid_from_file(abs_path),
                "size": abs_path.stat().st_size,
            }
        )

        self._ext_tbl.update1(key)

    # ------------------------------ Ephys Data ------------------------------

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
                self._logger.info(f"Adding metric {metric} : {metric_values}")
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

            # If metrics were specified, add one column per metric
            if metrics is not None:
                for metric_name, metric_dict in metrics.items():
                    self._logger.info(
                        f"Adding metric {metric_name} : {metric_dict}"
                    )
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
                self._logger.info(
                    f"Adding metric {metric_name} : {metric_dict}"
                )
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

    # ------------------------------ Maintenance ------------------------------

    def cleanup_external(self):
        """Remove the filepath entries for NWB files that are not in use.

        Because an unused file in the common may be in use in a custom table,
        we never want to delete external files. Instead, the common handles
        orphan detection and deletion.
        """
        self._ext_tbl.delete(delete_external_files=False)

    def get_orphans(self):
        """Clean up orphaned entries and external files."""
        return self - get_child_tables(self)

    def log(self, *args, **kwargs):
        """Null log method. Revert to _disabled_log to turn back on."""
        self._logger.debug("Logging disabled.")
