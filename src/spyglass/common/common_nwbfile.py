import os
import random
import string
import subprocess
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
        file_path = raw_dir + "/" + nwb_file_name
        if new_file:
            return file_path

        query = cls & {"nwb_file_name": nwb_file_name}
        if len(query) != 1:
            raise ValueError(
                f"Could not find 1 entry for {nwb_file_name}:\n{query}"
            )

        return file_path

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


@schema
class AnalysisRegistry(dj.Manual):
    definition = """
    full_table_name: varchar(128)  # full table name of the analysis
    ---
    created_at = CURRENT_TIMESTAMP: timestamp  # when registered
    created_by : varchar(32)                   # who registered
    """
