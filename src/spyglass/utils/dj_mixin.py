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


IngestionEntries: TypeAlias = Dict["SpyglassIngestion", List[dict]]
# How SpyglassIngestion handles generated entries from NWB objects
# Dict keys are SpyglassIngestion table classes, or FreeTable Table objects
# Values are lists of dicts to insert into those tables


class SpyglassIngestion(SpyglassMixin):
    """A mixin for Spyglass tables that ingest data from NWB files.

    Attributes
    ----------
    _expected_duplicates : bool
        If true, checks that pre-existing entries are consistent in secondary
        keys with inserted, entries and allows for skipping duplicates on insert
    _prompt_insert : bool
        If true, prompts user before inserting new table entries from NWB file.
    _only_ingest_first : bool
        If true, only ingests the first matching NWB object from the file.
    _source_nwb_object_name : str, optional
        If set, only ingests NWB objects with this name. Useful for
        distinguishing between multiple objects of the same type. E.g.
        BehavioralEvents named 'behavioral_events' vs 'analog' or 'video'
        objects of the same type ingested by DIOEvents table.
    table_key_to_obj_attr : Dict[str, Dict[str, Union[str, Callable]]]
        A dict of dicts mapping table keys to NWB object attributes.
    _source_nwb_object_type : Type
        The type of NWB object to import from the NWB file. If None, the table
        must implement get_nwb_objects.

    """

    _expected_duplicates = False
    _prompt_insert = False
    _only_ingest_first = False
    _source_nwb_object_name = None  # Optional filter on object name

    @property
    def table_key_to_obj_attr(
        self,
    ) -> Dict[str, Dict[str, Union[str, Callable]]]:
        """A dict of dicts mapping table keys to NWB object attributes.

        First level keys are the nwb object. The reserved key "self" refers to
        the original object. Additional keys can be added to access data from
        other nwb objects that are attributes of the object (e.g.
        device.model).

        Second level keys are the table keys to map to the nwb object
        attributes. If the values of this dictionary are strings, they are
        interpreted as attribute names of the nwb object. If the values are
        callables, they are called with the nwb object as the only argument.
        """
        # Dev note: cannot use abstractmethod because DataJoint creates an
        # instance with @schema decorator, yielding errors even when the
        # method is implemented in the subclass.
        raise NotImplementedError(
            "SpyglassIngestion tables need to implement table_key_to_obj_attr."
        )

    @property
    def _source_nwb_object_type(self) -> Type:
        """The type of NWB object to import from the NWB file."""
        raise NotImplementedError(
            "SpyglassIngestion tables need to implement _source_nwb_object_type."
        )

    def _config_entries(self, tbl, base_key, entries) -> List[dict]:
        """Generate entries for a given table and base key."""
        return {tbl: [dict(base_key, **entry) for entry in entries]}

    def generate_entries_from_config(
        self, config: dict, base_key=None
    ) -> IngestionEntries:
        """Generates a list of table entries from a config dictionary."""

        base_key = base_key or dict()
        self_entries = config.get(self.camel_name, [])
        entries = self._config_entries(self, base_key, self_entries)

        for part in self.parts(as_objects=True):
            camel_part = to_camel_case(part.full_table_name.split("__")[-1])
            part_entries = config.get(camel_part, [])
            if len(part_entries) == 0:
                continue
            entries[part] = self._config_entries(part, base_key, part_entries)

        return entries

    def generate_entries_from_nwb_object(
        self, nwb_obj, base_key=None
    ) -> IngestionEntries:
        """Generates a list of table entries from an NWB object.

        If generating entries for multiple tables, ensure the parent entry is
        returned before the child in the IngestionEntries dict.
        """
        base_key = base_key or dict()
        base_key = base_key.copy()  # avoid modifying original

        # For table objects, generate entry(s) for each row
        if hasattr(nwb_obj, "to_dataframe"):
            obj_df = nwb_obj.to_dataframe()
            entries = sum(
                [
                    self.generate_entries_from_nwb_object(row, base_key)[self]
                    for row in obj_df.itertuples()
                ],
                [],
            )
            return {self: entries}

        obj_ = None
        for object_name, mapping in self.table_key_to_obj_attr.items():
            obj_ = (
                nwb_obj
                if object_name == "self"
                else getattr(nwb_obj, object_name)
            )

            if obj_ is None:
                raise ValueError(
                    f"NWB object {object_name} not found in {nwb_obj}."
                )

            base_key.update(
                {
                    k: (getattr(obj_, v) if isinstance(v, str) else v(obj_))
                    for k, v in mapping.items()
                }
            )
        return {self: [base_key]}

    def get_nwb_objects(
        self,
        nwb_file: NWBFile,
        nwb_file_name: str = None,
    ) -> List:
        """Returns a list of NWB objects to be imported.

        By default, returns a list with the root nwb_file object.
        Can be overridden to return a list of other nwb objects (e.g. all devices).
        """
        matching_objects = [
            obj
            for obj in nwb_file.objects.values()
            if isinstance(obj, self._source_nwb_object_type)
        ]

        if self._source_nwb_object_name:
            matching_objects = [
                obj
                for obj in matching_objects
                if getattr(obj, "name", None) == self._source_nwb_object_name
            ]

        return matching_objects

    def _insert_logline(self, nwb_file_name=None, n_entries=0, table=None):
        """Log line for insert_from_nwbfile."""

        # String formatting permits either SpyglassMixin or FreeTable objects
        def _camel(tbl=None):
            if tbl is None:
                return ""
            s = getattr(tbl, "full_table_name", str(tbl))
            return to_camel_case(s.split(".")[-1].replace("__", ".")).strip("`")

        this_tbl, self_tbl = _camel(table), _camel(self)

        suffix = "" if this_tbl == self_tbl else f" via {self_tbl}"
        logger.info(
            f"{nwb_file_name} inserts {n_entries} into {this_tbl}{suffix}"
        )

    def insert_from_nwbfile(
        self,
        nwb_file_name: str,
        config: dict = None,
        dry_run: bool = False,
    ):
        """Insert entries into the table from an NWB file.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file to import from.
        config : dict, optional
            A configuration dictionary to supplement NWB data. Default None.
        dry_run : bool, optional
            If True, do not insert into the database, just return the entries
            that would be inserted. Default False.
        """
        from spyglass.common.common_nwbfile import Nwbfile

        nwb_key = {"nwb_file_name": nwb_file_name}
        if not (query := Nwbfile & nwb_key):
            raise ValueError(f"NWB file {nwb_file_name} not found in database.")

        nwb_file = query.fetch_nwb()[0]
        base_entry = nwb_key if "nwb_file_name" in self.primary_key else dict()

        # compile list of table entries from all objects in this file
        fetched_objs = self.get_nwb_objects(nwb_file, nwb_file_name)
        entries = (
            self.generate_entries_from_nwb_object(
                nwb_obj=fetched_objs[0],
                base_key=base_entry.copy(),
            )
            if fetched_objs
            else dict()
        )
        if not self._only_ingest_first:
            next_objs = fetched_objs[1:] if len(fetched_objs) > 1 else []
            for nwb_obj in next_objs:
                obj_entries = self.generate_entries_from_nwb_object(
                    nwb_obj,
                    base_entry.copy(),
                )
                for table, table_entries in obj_entries.items():
                    entries[table].extend(table_entries)

        if config:
            config_entries = self.generate_entries_from_config(config)
            for table, table_entries in config_entries.items():
                if table in entries:
                    entries[table].extend(table_entries)
                else:
                    entries[table] = table_entries

        # Remove tables with no entries - if all entries 'None', skip table
        # Motivated by nwb with no Institution, results in nulled fk subj ref
        debug_backup = entries.copy()
        _ = debug_backup  # Intentionally kept for debugging
        entries = self._adjust_entries(entries, nwb_file_name=nwb_file_name)
        if entries is None or len(entries) == 0:
            return dict()

        # validate that new entries are consistent with existing entries
        if self._expected_duplicates:
            self.validate_duplicates(entries)

        # run insertions
        if not dry_run:
            self._run_nwbfile_insert(entries, nwb_file_name=nwb_file_name)

        return entries

    def _run_nwbfile_insert(
        self, entries: IngestionEntries, nwb_file_name: str = None
    ) -> None:
        """Run insert on compiled Dict[TableObject, inserts]."""

        def expect_dupes(tbl):
            """Allow table to override self._expected_duplicates."""
            # Implemented to default to self for FreeTable instances
            return (
                getattr(table, "_expected_duplicates", None)
                or self._expected_duplicates
            )

        # An integrity here probably means a parallel insert was dropped
        # check debug_backup in parent func for entries that were dropped
        for table, table_entries in entries.items():
            table.insert(
                table_entries,
                skip_duplicates=expect_dupes(table),
                allow_direct_insert=True,
            )
            self._insert_logline(nwb_file_name, len(table_entries), table)

    def _key_has_required_attrs(self, key):
        """Check that all non-nullable attributes are present in the key."""
        for attr in self.heading.attributes.values():
            if attr.nullable or attr.autoincrement or attr.default is not None:
                continue  # skip nullable, autoincrement, or default val attrs
            if attr.name not in key or key.get(attr.name) is None:
                return False
        return True

    def _adjust_keys_for_entry(self, keys: List[dict]) -> List[dict]:
        """Passthrough. Allows children to adjust keys before comparing."""
        # Motivated by Subject.sex: comparing None to "U" should be equal
        # Without this step, reinsert triggers accept_divergence prompt
        # By default, checks that all non-nullable keys present
        return [key for key in keys if self._key_has_required_attrs(key)]

    def _remove_null_from_dicts(self, d: dict) -> dict:
        """Remove keys with None values from a dictionary."""
        # Fallback if FreeTable does not implement _adjust_keys_for_entry
        # May error on part table with nullable or default keys
        return {k: v for k, v in d.items() if v not in [None, ""]}

    def _adjust_entries(
        self, entries: IngestionEntries, nwb_file_name: str = None
    ) -> Optional[IngestionEntries]:
        """Run _adjust_key for each table in planned entries.

        Given a Dict[TableObject, List[dict]], with planned entries values,
        run each table's _adjust_keys_for_entry function on the list of dicts.
        Removes invalid/null entries and tables with no valid entries.
        """

        null_keys = []

        for table, table_entries in entries.items():
            # ensure instanced
            tbl = table() if inspect.isclass(table) else table

            # Allow children to adjust keys before comparing
            # Provide backup for FreeTable instances
            adjust_func = getattr(
                tbl, "_adjust_keys_for_entry", self._remove_null_from_dicts
            )
            adjusted_entries = adjust_func(table_entries)

            if not any(adjusted_entries):
                null_keys.append(table)  # mark for removal from dict
            else:
                entries[table] = adjusted_entries

        for table in null_keys:
            self._insert_logline(nwb_file_name, 0, table)
            _ = entries.pop(table)

        return entries if len(entries) > 0 else None

    def validate_duplicates(self, entry_dict: Dict[dj.Table, List[dict]]):
        """Validate new entries against existing entries in the database.

        Parameters
        ----------
        entry_dict : dict or Dict[dj.Table, List[dict]]
            The new entry or dict of table entries to validate against existing
            entries in the database.
        """
        for table, table_entries in entry_dict.items():
            for entry in self._adjust_keys_for_entry(table_entries):
                self.validate1_duplicate(table, entry)

    def validate1_duplicate(self, tbl, new_key):
        """If matching primary key, check for consistency in secondary keys.

        If divergence, prompt user whether to accept existing value

        Parameters
        ----------
        tbl : dj.Table
            The table to validate against.
        new_key : dict
            The new key to validate against existing entries in the database.
        """
        # NOTE: switching from `self` to `tbl` to allow validation from
        # FreeTable instances captured with `parts(as_objects=True)`

        # If novel entry, nothing to validate
        if not (query := tbl & new_key):
            return

        existing = query.fetch1()

        for key in set(new_key).union(existing):
            if not self._unequal_vals(key, new_key, existing):
                continue  # skip if values are equal
            if not accept_divergence(
                key,
                new_key.get(key),
                existing.get(key),
                self._test_mode,
                to_camel_case(tbl.full_table_name.split(".")[-1]).strip("`"),
            ):
                # If the user does not accept the divergence,
                # raise an error to prevent data inconsistency
                raise dj.errors.DuplicateError(
                    f"Attempted entry in {self.camel_name} already exists "
                    + f"with different values for {key}: "
                    + f"{new_key.get(key)} != {existing.get(key)}"
                )

    @staticmethod
    def _unequal_vals(key, a, b):
        a, b = a.get(key) or "", b.get(key, "") or ""
        if isinstance(a, str) and isinstance(b, str):
            return a.lower() != b.lower()
        return a != b  # prevent false positive on None != ""
