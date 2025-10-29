"""Helper methods for DataJoint tables, inherited as SpyglassHelpers."""

import os
import sys
from contextlib import nullcontext
from typing import Union

import datajoint as dj
from datajoint.expression import QueryExpression
from datajoint.utils import to_camel_case
from pandas import DataFrame

from spyglass.utils.dj_helper_fn import (
    _quick_get_analysis_path,
    bytes_to_human_readable,
    get_child_tables,
)
from spyglass.utils.mixins.base import BaseMixin


class HelperMixin(BaseMixin):
    """Helper methods for DataJoint tables."""

    def dict_to_pk(self, key):
        """Return primary key from dictionary."""
        return {k: v for k, v in key.items() if k in self.primary_key}

    def dict_to_full_key(self, key):
        """Return full key from dictionary."""
        return {k: v for k, v in key.items() if k in self.heading.names}

    @property
    def camel_name(self):
        """Return table name in camel case."""
        return to_camel_case(self.table_name)

    def _auto_increment(self, key, pk, *args, **kwargs):
        """Auto-increment primary key."""
        if not key.get(pk):
            key[pk] = (dj.U().aggr(self, n=f"max({pk})").fetch1("n") or 0) + 1
        return key

    def file_like(self, name=None, **kwargs):
        """Convenience method for wildcard search on file name fields."""
        if not name:
            return self
        attr = None
        for field in self.heading.names:
            if "file" in field:
                attr = field
                break
        if not attr:
            self._logger.error(
                f"No file_like field found in {self.full_table_name}"
            )
            return
        return self & f"{attr} LIKE '%{name}%'"

    def restrict_by_list(
        self, field: str, values: list, return_restr=False
    ) -> QueryExpression:
        """Restrict a field by list of values."""
        if field not in self.heading.attributes:
            raise KeyError(f"Field '{field}' not in {self.camel_name}.")
        quoted_vals = '"' + '","'.join(map(str, values)) + '"'
        restr = self & f"{field} IN ({quoted_vals})"
        return restr if return_restr else self & restr

    def find_insert_fail(self, key):
        """Find which parent table is causing an IntergrityError on insert."""
        rets = []
        for parent in self.parents(as_objects=True):
            parent_key = {
                k: v for k, v in key.items() if k in parent.heading.names
            }
            parent_name = to_camel_case(parent.table_name)
            if query := parent & parent_key:
                rets.append(f"{parent_name}:\n{query}")
            else:
                rets.append(f"{parent_name}: MISSING")
        self._logger.info("\n".join(rets))

    @classmethod
    def _safe_context(cls):
        """Return transaction if not already in one."""
        return (
            cls.connection.transaction
            if not cls.connection.in_transaction
            else nullcontext()
        )

    @classmethod
    def get_fully_defined_key(
        cls, key: dict = None, required_fields: list[str] = None
    ) -> dict:
        if key is None:
            key = dict()

        required_fields = required_fields or cls.primary_key
        if isinstance(key, (str, dict)):  # check is either keys or substrings
            if all(field in key for field in required_fields):
                return key  # return if all required fields are present

            if not len(query := cls() & key) == 1:  # check if key is unique
                raise KeyError(
                    "Key is neither fully specified nor a unique entry in"
                    + f"table.\n\tTable: {cls.full_table_name}\n\tKey: {key}"
                    + f"Required fields: {required_fields}\n\tResult: {query}"
                )
            key = query.fetch1("KEY")

        return key

    def ensure_single_entry(self, key: dict = True):
        """Ensure that the key corresponds to a single entry in the table.

        Parameters
        ----------
        key : dict
            The key to check. Default to True, no further restriction of `self`.
        """
        if len(self & key) != 1:
            raise KeyError(
                f"Please restrict {self.full_table_name} to 1 entry when calling "
                f"{sys._getframe(1).f_code.co_name}(). "
                f"Found {len(self & key)} entries"
            )

    def load_shared_schemas(self, additional_prefixes: list = None) -> None:
        """Load shared schemas to include in graph traversal.

        Parameters
        ----------
        additional_prefixes : list, optional
            Additional prefixes to load. Default None.
        """
        from spyglass.utils.database_settings import SHARED_MODULES

        all_shared = [
            *SHARED_MODULES,
            dj.config["database.user"],
            "file",
            "sharing",
        ]

        if additional_prefixes:
            all_shared.extend(additional_prefixes)

        # Get a list of all shared schemas in spyglass
        schemas = dj.conn().query(
            "SELECT DISTINCT table_schema "  # Unique schemas
            + "FROM information_schema.key_column_usage "
            + "WHERE"
            + '    table_name not LIKE "~%%"'  # Exclude hidden
            + "    AND constraint_name='PRIMARY'"  # Only primary keys
            + "AND ("  # Only shared schemas
            + " OR ".join([f"table_schema LIKE '{s}_%%'" for s in all_shared])
            + ") "
            + "ORDER BY table_schema;"
        )

        # Load the dependencies for all shared schemas
        for schema in schemas:
            dj.schema(schema[0]).connection.dependencies.load()

    # -------------------------------- Orphans --------------------------------

    def delete_orphans(
        self, dry_run: bool = True, **kwargs
    ) -> Union[QueryExpression, None]:
        """Get entries in the table without any child table entries.

        Parameters
        ----------
        dry_run : bool, optional
            If True, return the orphaned entries without deleting them.
            Default True.
        **kwargs : dict
            Passed to datajoint.table.Table.delete if dry_run is False.

        Returns
        -------
        QueryExpression, optional
            If dry_run, a query expression containing the orphaned entries.
        """
        orphans = self - get_child_tables(self)
        if dry_run:
            return orphans
        orphans.super_delete(warn=False, **kwargs)
        return None

    # ------------------------------ Check locks ------------------------------

    def exec_sql_fetchall(self, query):
        """
        Execute the given query and fetch the results.    Parameters
        ----------
        query : str
            The SQL query to execute.    Returns
        -------
        list of tuples
            The results of the query.
        """
        results = dj.conn().query(query).fetchall()
        return results  # Check if performance schema is enabled

    def check_threads(self, detailed=False, all_threads=False) -> DataFrame:
        """Check for locked threads in the database.

        Parameters
        ----------
        detailed : bool, optional
            Show all columns in the metadata_locks table. Default False, show
            summary.
        all_threads : bool, optional
            Show all threads, not just those related to this table.
            Default False.


        Returns
        -------
        DataFrame
            A DataFrame containing the metadata locks.
        """
        performance__status = self.exec_sql_fetchall(
            "SHOW VARIABLES LIKE 'performance_schema';"
        )
        if performance__status[0][1] == "OFF":
            raise RuntimeError(
                "Database does not monitor threads. "
                + "Please ask you administrator to enable performance schema."
            )

        metadata_locks_query = """
        SELECT
            ml.OBJECT_SCHEMA, -- Table schema
            ml.OBJECT_NAME, -- Table name
            ml.OBJECT_TYPE, -- What is locked
            ml.LOCK_TYPE, -- Type of lock
            ml.LOCK_STATUS, -- Lock status
            ml.OWNER_THREAD_ID, -- Thread ID of the lock owner
            t.PROCESSLIST_ID, -- User connection ID
            t.PROCESSLIST_USER, -- User
            t.PROCESSLIST_HOST, -- User machine
            t.PROCESSLIST_TIME, -- Time in seconds
            t.PROCESSLIST_DB, -- Thread database
            t.PROCESSLIST_COMMAND, -- Likely Query
            t.PROCESSLIST_STATE, -- Waiting for lock, sending data, or locked
            t.PROCESSLIST_INFO -- Actual query
        FROM performance_schema.metadata_locks AS ml
        JOIN performance_schema.threads AS t
        ON ml.OWNER_THREAD_ID = t.THREAD_ID
        """

        where_clause = (
            f"WHERE ml.OBJECT_SCHEMA = '{self.database}' "
            + f"AND ml.OBJECT_NAME = '{self.table_name}'"
        )
        metadata_locks_query += ";" if all_threads else where_clause

        df = DataFrame(
            self.exec_sql_fetchall(metadata_locks_query),
            columns=[
                "Schema",  # ml.OBJECT_SCHEMA -- Table schema
                "Table Name",  # ml.OBJECT_NAME -- Table name
                "Locked",  # ml.OBJECT_TYPE -- What is locked
                "Lock Type",  # ml.LOCK_TYPE -- Type of lock
                "Lock Status",  # ml.LOCK_STATUS -- Lock status
                "Thread ID",  # ml.OWNER_THREAD_ID -- Thread ID of the lock owner
                "Connection ID",  # t.PROCESSLIST_ID -- User connection ID
                "User",  # t.PROCESSLIST_USER -- User
                "Host",  # t.PROCESSLIST_HOST -- User machine
                "Time (s)",  # t.PROCESSLIST_TIME -- Time in seconds
                "Process Database",  # t.PROCESSLIST_DB -- Thread database
                "Process",  # t.PROCESSLIST_COMMAND -- Likely Query
                "State",  # t.PROCESSLIST_STATE
                "Query",  # t.PROCESSLIST_INFO -- Actual query
            ],
        )

        df["Name"] = df["User"].apply(self._delete_deps[0]().get_djuser_name)

        keep_cols = []
        if all_threads:
            keep_cols.append("Table")
            df["Table"] = df["Schema"] + "." + df["Table Name"]
        df = df.drop(columns=["Schema", "Table Name"])

        if not detailed:
            keep_cols.extend(["Locked", "Name", "Time (s)", "Process", "State"])
            df = df[keep_cols]

        return df

    # --------------------------- Check disc usage ------------------------------

    def get_table_storage_usage(self, human_readable=False):
        """Total size of all analysis files in the table.
        Uses the analysis_file_name field to find the file paths and sum their
        sizes.
        Parameters
        ----------
        human_readable : bool, optional
            If True, return a human-readable string of the total size.
            Default False, returns total size in bytes.

        Returns
        -------
        Union[str, int]
            Total size of all analysis files in the table. If human_readable is
            True, returns a string with the size in bytes, KiB, MiB, GiB, TiB,
            or PiB. If human_readable is False, returns the total size in bytes.

        """
        if "analysis_file_name" not in self.heading.names:
            self._logger.warning(
                f"{self.full_table_name} does not have an analysis_file_name field."
            )
            return "0 Mib" if human_readable else 0
        file_names = self.fetch("analysis_file_name")
        file_paths = [
            _quick_get_analysis_path(file_name) for file_name in file_names
        ]
        file_paths = [path for path in file_paths if path is not None]
        file_sizes = [os.stat(path).st_size for path in file_paths]
        total_size = sum(file_sizes)
        if not human_readable:
            return total_size
        human_size = bytes_to_human_readable(total_size)
        return human_size
