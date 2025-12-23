"""Mixin class for fetching NWB files and pynapple objects."""

import os
from functools import cached_property
from typing import Tuple

import numpy as np
from datajoint import Table

from spyglass.utils.dj_helper_fn import instance_table
from spyglass.utils.mixins.base import BaseMixin
from spyglass.utils.nwb_helper_fn import file_from_dandi, get_nwb_file

try:
    import pynapple  # noqa F401
except (ImportError, ModuleNotFoundError):
    pynapple = None


class FetchMixin(BaseMixin):

    @cached_property
    def _nwb_table_tuple(self) -> tuple:
        """NWB file table class and attribute name.

        Automatically detects the appropriate NWB table type by inspecting
        foreign keys and parent relationships. Supports:
        - Raw data files (Nwbfile)
        - Common analysis files (AnalysisNwbfile)
        - Custom analysis files (team-specific tables)

        Used to determine fetch_nwb behavior. Also used in Merge.fetch_nwb.
        Implemented as a cached_property to avoid circular imports.

        Returns
        -------
        tuple
            (table_class, attribute_name) for NWB file fetching
        """
        from spyglass.common.common_nwbfile import (
            AnalysisNwbfile,
            AnalysisRegistry,
            Nwbfile,
        )

        # Helper function to extract prefix from parent name
        # Check for custom AnalysisNwbfile parent
        analysis_parents = [
            p for p in self.parents() if p.endswith(".`analysis_nwbfile`")
        ]

        if len(analysis_parents) > 1:
            raise ValueError(
                f"{self.__class__.__name__} has multiple AnalysisNwbfile "
                "parents. This is not permitted."
            )

        if len(analysis_parents) == 1:
            # Common AnalysisNwbfile
            if "common_nwbfile" in analysis_parents[0]:
                return (AnalysisNwbfile, "analysis_file_abs_path")

            # Custom AnalysisNwbfile
            custom_class = AnalysisRegistry().get_class(analysis_parents[0])
            if custom_class is None:
                raise ValueError(
                    f"Custom analysis table '{analysis_parents[0]}' "
                    "not found in registry"
                )
            return (custom_class, "analysis_file_abs_path")

        # Check for explicit _nwb_table attribute
        resolved = getattr(self, "_nwb_table", None)

        # Fallback: Check definition for Nwbfile foreign key
        if not resolved and "-> Nwbfile" in self.definition:
            resolved = Nwbfile

        # If still not resolved, raise error
        if not resolved:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have a "
                "(Analysis)Nwbfile foreign key or _nwb_table attribute."
            )

        # Determine attribute name based on table type
        table_dict = {
            AnalysisNwbfile: "analysis_file_abs_path",
            Nwbfile: "nwb_file_abs_path",
        }

        attr_name = table_dict.get(resolved, "analysis_file_abs_path")

        return (resolved, attr_name)

    def _get_nwb_files_and_path_fn(
        self, tbl: Table, attr_name: str, *attrs, **kwargs
    ) -> Tuple[list, callable]:
        """Get NWB file names and path resolution function.

        Parameters
        ----------
        tbl : table or class
            DataJoint table/class to fetch from. Can be a table class that
            was already resolved by _nwb_table_tuple.
        attr_name : str
            Attribute name to fetch from the table.

        Returns
        -------
        nwb_files : list
            List of NWB file names.
        file_path_fn : function
            Function to get the absolute path to the NWB file.
        """
        from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
        from spyglass.utils.dj_mixin import SpyglassMixin

        which = "analysis" if "analysis" in attr_name else "nwb"
        file_name_str = (
            "analysis_file_name" if which == "analysis" else "nwb_file_name"
        )

        # tbl is already the resolved class from _nwb_table_tuple
        # Get the get_abs_path method from it directly
        file_path_fn = getattr(tbl, "get_abs_path", None)

        # Instance the table for the join query
        tbl_inst = instance_table(tbl)
        tbl_name = getattr(tbl_inst, "camel_name", tbl_inst.table_name)

        if file_path_fn is None:  # use prev approach as fallback
            file_path_fn = (
                AnalysisNwbfile.get_abs_path
                if which == "analysis"
                else Nwbfile.get_abs_path
            )
        if not callable(file_path_fn):
            raise ValueError(
                f"Table {tbl_name} does not have a valid get_abs_path method."
            )

        # logging arg only if instanced table inherits Mixin
        inst = instance_table(self)
        log_exp = (
            dict(log_export=False)
            if isinstance(inst, SpyglassMixin)
            else dict()
        )

        nwb_files = (
            self.join(tbl_inst.proj(nwb2load_filepath=attr_name), **log_exp)
        ).fetch(file_name_str)

        return nwb_files, file_path_fn

    def _download_missing_files(self, nwb_files, file_path_fn):
        """Download missing NWB files from kachery/dandi or recompute.

        Parameters
        ----------
        nwb_files : list
            List of NWB file names.
        file_path_fn : function
            Function to get the absolute path to the NWB file.
        """
        for file_name in nwb_files:
            file_path = file_path_fn(file_name, from_schema=True)
            if not os.path.exists(file_path):
                # get from kachery/dandi or recompute, store in cache
                get_nwb_file(file_path, self)

    def _execute_nwb_query(self, tbl, attr_name, *attrs, **kwargs):
        """Execute join query and fetch records with NWB filepaths.

        Parameters
        ----------
        tbl : table
            DataJoint table to fetch from.
        attr_name : str
            Attribute name to fetch from the table.
        *attrs : list
            Attributes from normal DataJoint fetch call.
        **kwargs : dict
            Keyword arguments from normal DataJoint fetch call.

        Returns
        -------
        rec_dicts : list
            List of dictionaries with fetch results and nwb2load_filepath.
        """
        from spyglass.utils.dj_mixin import SpyglassMixin

        file_name_attr = (
            "analysis_file_name" if "analysis" in attr_name else "nwb_file_name"
        )

        # Get file names and path function
        nwb_files, file_path_fn = self._get_nwb_files_and_path_fn(
            tbl, attr_name, *attrs, **kwargs
        )

        # logging arg only if instanced table inherits Mixin
        inst = instance_table(self)
        log_exp = (
            dict(log_export=False)
            if isinstance(inst, SpyglassMixin)
            else dict()
        )
        tbl_inst = instance_table(tbl)
        query_table = self.join(
            tbl_inst.proj(nwb2load_filepath=attr_name), **log_exp
        )

        kwargs["as_dict"] = True  # force return as dictionary
        attrs = attrs or self.heading.names  # if not specified, fetch all
        rec_dicts = query_table.fetch(*attrs, **kwargs)

        # get filepath for each. Use datajoint for checksum if local
        for rec_dict in rec_dicts:
            file_path = file_path_fn(rec_dict[file_name_attr])
            if file_from_dandi(file_path):
                # skip the filepath checksum if streamed from Dandi
                rec_dict["nwb2load_filepath"] = file_path
                continue

            # Drop secondary blob attrs that can't be part of restrictions
            rec_only_pk = {
                k: v
                for k, v in rec_dict.items()
                if k in query_table.heading.primary_key
            }
            rec_dict["nwb2load_filepath"] = (query_table & rec_only_pk).fetch1(
                "nwb2load_filepath"
            )

        return rec_dicts

    def _process_object_ids(self, rec_dicts, *attrs):
        """Process object_id fields and convert to NWB objects.

        Parameters
        ----------
        rec_dicts : list
            List of dictionaries with fetch results.
        *attrs : list
            Attributes from fetch call.

        Returns
        -------
        list
            List of dicts with object_id fields converted to NWB objects.
        """
        if not rec_dicts or not np.any(
            ["object_id" in key for key in rec_dicts[0]]
        ):
            return rec_dicts

        ret = []
        for rec_dict in rec_dicts:
            nwbf = get_nwb_file(rec_dict.pop("nwb2load_filepath"))
            # for each attr that contains substring 'object_id', store key-value:
            # attr name to NWB object
            # remove '_object_id' from attr name
            nwb_objs = {
                id_attr.replace("_object_id", ""): self._get_nwb_object(
                    nwbf.objects, rec_dict[id_attr]
                )
                for id_attr in attrs
                if "object_id" in id_attr and rec_dict[id_attr] != ""
            }
            ret.append({**rec_dict, **nwb_objs})

        return ret

    @staticmethod
    def _get_nwb_object(objects, object_id):
        """Retrieve NWB object and try to convert to dataframe if possible."""
        try:
            return objects[object_id].to_dataframe()
        except AttributeError:
            return objects[object_id]

    def fetch_nwb(self, *attrs, **kwargs):
        """Fetch NWBFile object from relevant table.

        Implementing class must have a foreign key reference to Nwbfile or
        AnalysisNwbfile (i.e., "-> (Analysis)Nwbfile" in definition)
        or a _nwb_table attribute. If both are present, the attribute takes
        precedence.

        Parameters
        ----------
        *attrs : list
            Attributes from normal DataJoint fetch call.
        **kwargs : dict
            Keyword arguments from normal DataJoint fetch call.

        Returns
        -------
        nwb_objects : list
            List of dicts containing fetch results and NWB objects.
        """
        table, tbl_attr = self._nwb_table_tuple

        # Handle export logging
        log_export = kwargs.pop("log_export", True)
        is_export = log_export and self.export_id

        if is_export and ("analysis" in tbl_attr):
            self._log_fetch_nwb(table, tbl_attr)

        is_analysis_table = self.full_table_name.endswith(
            "_nwbfile`.`analysis_nwbfile`"
        )
        if is_export and is_analysis_table:
            self._copy_to_common()

        # Set defaults for fetch
        kwargs["as_dict"] = True  # force return as dictionary
        if not attrs:
            attrs = self.heading.names

        # Get file names and path resolution function
        nwb_files, file_path_fn = self._get_nwb_files_and_path_fn(
            table, tbl_attr, *attrs, **kwargs
        )

        # Download any missing files
        self._download_missing_files(nwb_files, file_path_fn)

        # Execute query and get records
        rec_dicts = self._execute_nwb_query(table, tbl_attr, *attrs, **kwargs)

        # Process object_id fields if present
        return self._process_object_ids(rec_dicts, *attrs)

    def fetch_pynapple(self, *attrs, **kwargs):
        """Get a pynapple object from the given DataJoint query.

        Parameters
        ----------
        *attrs : list
            Attributes from normal DataJoint fetch call.
        **kwargs : dict
            Keyword arguments from normal DataJoint fetch call.

        Returns
        -------
        pynapple_objects : list of pynapple objects
            List of dicts containing pynapple objects.

        Raises
        ------
        ImportError
            If pynapple is not installed.

        """
        if pynapple is None:
            raise ImportError("Pynapple is not installed.")

        nwb_files, file_path_fn = self._get_nwb_files_and_path_fn(
            self._nwb_table_tuple[0],
            self._nwb_table_tuple[1],
            *attrs,
            **kwargs,
        )

        return [
            pynapple.load_file(file_path_fn(file_name))
            for file_name in nwb_files
        ]
