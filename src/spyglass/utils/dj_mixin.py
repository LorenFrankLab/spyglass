import inspect
import os
import sys
from abc import abstractmethod
from contextlib import nullcontext
from functools import cached_property
from os import environ as os_environ
from time import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeAlias, Union

import datajoint as dj
from datajoint.errors import DataJointError
from datajoint.expression import QueryExpression
from datajoint.table import Table
from datajoint.utils import to_camel_case
from packaging.version import parse as version_parse
from pandas import DataFrame
from pymysql.err import DataError
from pynwb import NWBHDF5IO, NWBFile

from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_helper_fn import (
    NonDaemonPool,
    _quick_get_analysis_path,
    accept_divergence,
    bytes_to_human_readable,
    ensure_names,
    fetch_nwb,
    get_child_tables,
    get_nwb_table,
    populate_pass_function,
)
from spyglass.utils.dj_merge_tables import Merge, is_merge_table
from spyglass.utils.logging import logger
from spyglass.utils.mixins import (
    AnalysisMixin,
    CautiousDeleteMixin,
    ExportMixin,
    HelperMixin,
    PopulateMixin,
    RestrictByMixin,
    IngestionMixin,
)


class SpyglassMixin(
    CautiousDeleteMixin,
    ExportMixin,  # -> FetchMixin -> BaseMixin
    HelperMixin,
    PopulateMixin,
    RestrictByMixin,
):
    """Mixin for Spyglass DataJoint tables.

    Provides methods for fetching NWBFile objects and checking user permission
    prior to deleting. As a mixin class, all Spyglass tables can inherit custom
    methods from a central location.

    Methods
    -------
    fetch_nwb(*attrs, **kwargs)
        Fetch NWBFile object from relevant table. Uses either a foreign key to
        a NWBFile table (including AnalysisNwbfile) or a _nwb_table attribute to
        determine which table to use.
    cautious_delete(force_permission=False, *args, **kwargs)
        Check user permissions before deleting table rows. Permission is granted
        to users listed as admin in LabMember table or to users on a team with
        with the Session experimenter(s). If the table where the delete is
        executed cannot be linked to a Session, a warning is logged and the
        delete continues. If the Session has no experimenter, or if the user is
        not on a team with the Session experimenter(s), a PermissionError is
        raised. `force_permission` can be set to True to bypass permission check.
    """

    # Class-level set to track validated tables by full_table_name
    _fk_validated = set()

    def __init__(self, *args, **kwargs):
        """Initialize SpyglassMixin.

        Checks that schema prefix is in SHARED_MODULES.
        Validates that table doesn't have multiple AnalysisNwbfile foreign keys.
        """
        # Uncomment to force Spyglass version check. See #439
        # _ = self._has_updated_sg_version

        # Check for multiple AnalysisNwbfile FKs on first instantiation only
        # Cannot use parents check during table declaration
        # Use class-level set to ensure validation happens once per table
        if (
            self.is_declared
            and self.full_table_name not in SpyglassMixin._fk_validated
            and hasattr(self, "parents")
        ):
            self._validate_analysis_nwbfile_fks()

        if self.is_declared:
            return  # Skip further checks after declaration

        if self.database and self.database.split("_")[0] not in [
            *SHARED_MODULES,
            dj.config["database.user"],
            dj.config.get("custom", dict()).get("database.prefix"),
            "temp",
            "test",
        ]:
            logger.error(
                f"Schema prefix not in SHARED_MODULES: {self.database}"
            )
        if is_merge_table(self) and not isinstance(self, Merge):
            raise TypeError(
                "Table definition matches Merge but does not inherit class: "
                + self.full_table_name
            )

    def _validate_analysis_nwbfile_fks(self):
        """Ensure table doesn't reference multiple AnalysisNwbfile tables.

        Tables should only have one foreign key to an AnalysisNwbfile table
        (either the central common.AnalysisNwbfile or a custom one).
        Having multiple references creates ambiguity for fetch_nwb() and export.

        Raises
        ------
        ValueError
            If table has more than one AnalysisNwbfile foreign key reference.
        """
        # Find all parents that are AnalysisNwbfile tables, reserved suffix.
        analysis_fks = [
            p
            for p in self.parents()
            if p.endswith("_nwbfile`.`analysis_nwbfile`")
        ]

        if len(analysis_fks) > 1:
            raise ValueError(
                "Tables cannot have multiple AnalysisNwbfiles: "
                f"\n{self.full_table_name} - {analysis_fks}"
                "\nThis table must be dropped and re-created without one"
            )

        SpyglassMixin._fk_validated.add(self.full_table_name)

    def get_params_blob_from_key(self, key: dict, default="default") -> dict:
        """Get params blob from table using key, assuming 1 primary key.

        Defaults to 'default' if no entry is found.

        TODO: Split SpyglassMixin to SpyglassParamsMixin.
        """
        pk = self.primary_key[0]
        blob_fields = [
            k.name for k in self.heading.attributes.values() if k.is_blob
        ]
        if len(blob_fields) != 1:
            raise ValueError(
                f"Table must have only 1 blob field, found {len(blob_fields)}"
            )
        blob_attr = blob_fields[0]

        if isinstance(key, str):
            key = {pk: key}
        if not isinstance(key, dict):
            raise ValueError("key must be a dictionary")
        passed_key = key.get(pk, None)
        if not passed_key:
            logger.warning("No key passed, using default")
        return (self & {pk: passed_key or default}).fetch1(blob_attr)


class SpyglassMixinPart(SpyglassMixin, dj.Part):
    """
    A part table for Spyglass Group tables. Assists in propagating
    delete calls from upstream tables to downstream tables.
    """

    def delete(self, *args, **kwargs):
        """Delete master and part entries."""
        restriction = self.restriction or True  # for (tbl & restr).delete()

        try:  # try restriction on master
            restricted = self.master & restriction
        except DataJointError:  # if error, assume restr of self
            restricted = self & restriction

        restricted.delete(*args, **kwargs)


class SpyglassAnalysis(SpyglassMixin, AnalysisMixin):
    """Mixin for custom AnalysisNwbfile tables.

    Provides automatic definition enforcement, schema validation, and registry
    integration for team-specific AnalysisNwbfile tables. This mixin enables
    transaction lock isolation by allowing teams to create their own analysis
    file tables in separate schemas.

    Usage:
        Users should import the pre-configured table from custom_nwbfile.
        By default, it uses database.user as the prefix:

        from spyglass.common.custom_nwbfile import AnalysisNwbfile
        # Schema "{username}_nwbfile" is created automatically
        # Uses database.user as prefix (or custom database.prefix if set)

    See Also:
        spyglass.common.custom_nwbfile for the pre-configured table.
        docs/src/ForDevelopers/CustomAnalysisFiles.md for comprehensive guide.
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyglassAnalysis.

        Enforces ...
        - Database conforms to `{prefix}_nwbfile` naming convention, one '_'
        - Table conforms to AnalysisNwbfile class name
        - Exact definition match for common AnalysisNwbfile table
        - Not a part table (part tables cannot be AnalysisNwbfile)
        - Inserts into AnalysisRegistry on declaration
        """

        if self.is_declared:
            return

        user_prefix = dj.config.get("custom", dict()).get("database.prefix")

        if not self.database:
            raise ValueError("Database must be set for Analysis tables")

        if self.database.count("_") != 1:
            raise ValueError(
                f"Must be exactly 1 '_' in schema name, found: {self.database}"
            )

        prefix, suffix = self.database.split("_", 1)

        if prefix == "common":
            self._logger.debug(
                f"Skipping prefix check for common schema: {self.database}"
            )
        elif prefix != user_prefix:
            raise ValueError(
                f"Schema prefix {prefix} does not match "
                + f"configured prefix: {user_prefix}"
            )
        if suffix != "nwbfile":
            raise ValueError(
                "Analysis requires {prefix}_nwbfile schema, "
                + f"found: {self.database}"
            )

        # Check if this is a part table (part tables cannot be AnalysisNwbfile)
        full_name = self.full_table_name
        if dj.utils.get_master(full_name) != "":
            raise ValueError(
                f"AnalysisNwbfile cannot be a part table: {full_name}. "
                "Part tables are not allowed to inherit from SpyglassAnalysis."
            )

        self.definition = self._enforced_definition
        self._register_table()


class SpyglassIngestion(SpyglassMixin, IngestionMixin):
    """Mixin for Spyglass ingestion tables.

    Provides additional methods and properties to automate population of table
    entries from raw NWB files.
    """
