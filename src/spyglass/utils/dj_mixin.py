import os
import sys
from contextlib import nullcontext
from functools import cached_property
from os import environ as os_environ
from time import time
from typing import List, Union

import datajoint as dj
from datajoint.errors import DataJointError
from datajoint.expression import QueryExpression
from datajoint.table import Table
from datajoint.utils import to_camel_case
from packaging.version import parse as version_parse
from pandas import DataFrame
from pymysql.err import DataError

from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_helper_fn import (
    NonDaemonPool,
    _quick_get_analysis_path,
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
    CautiousDeleteMixin,
    ExportMixin,
    FetchMixin,
    HelperMixin,
    PopulateMixin,
)


class SpyglassMixin(
    ExportMixin,
    CautiousDeleteMixin,
    HelperMixin,
    FetchMixin,
    PopulateMixin,
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

    # _nwb_table = None # NWBFile table class, defined at the table level

    def __init__(self, *args, **kwargs):
        """Initialize SpyglassMixin.

        Checks that schema prefix is in SHARED_MODULES.
        """
        # Uncomment to force Spyglass version check. See #439
        # _ = self._has_updated_sg_version

        if self.is_declared:
            return
        if self.database and self.database.split("_")[0] not in [
            *SHARED_MODULES,
            dj.config["database.user"],
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
