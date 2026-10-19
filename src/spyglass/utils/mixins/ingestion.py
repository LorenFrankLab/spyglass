import inspect
from typing import Callable, Dict, List, Optional, Type, Union

import datajoint as dj
import numpy as np
from packaging.version import Version
from pynwb import NWBFile

from spyglass.utils.dj_helper_fn import accept_divergence
from spyglass.utils.logging import logger
from spyglass.utils.mixins.base import BaseMixin
from spyglass.utils.nwb_hash import get_file_namespaces
from spyglass.utils.nwb_helper_fn import is_nwb_obj_type

# typing alias compatible with Python 3.9
IngestionEntries = dict["IngestionMixin", list[dict]]
# How IngestionMixin handles generated entries from NWB objects
# Dict keys are Spyglass table classes or instances, so that every table in a
# plan carries the mixin's properties. Values are lists of dicts to insert.


class IngestionMixin(BaseMixin):
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
    _source_nwb_object_description : str or tuple of str, optional
        If set, only ingests NWB objects whose description contains one of
        these markers. Useful when the type and name are shared across
        objects that differ only by what they describe. E.g. AssociatedFiles
        describing a state script vs any other associated file.
    table_key_to_obj_attr : Dict[str, Dict[str, Union[str, Callable]]]
        A dict of dicts mapping table keys to NWB object attributes.
    _source_nwb_object_type : Type
        The type of NWB object to import from the NWB file. If None, the table
        must implement get_nwb_objects.

    """

    _expected_duplicates = False  # If True, rows to be shared across sessions
    _prompt_insert = False
    _only_ingest_first = False
    _source_nwb_object_name = None  # Optional filter on object name
    _source_nwb_object_description = None  # Optional filter on description
    _single_entry_per_table = False  # If False, DynamicTables 1:1 per row
    _extension_requirements = dict()  # Opt: {ext_name: min_version} to check

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
            "IngestionMixin tables need to implement table_key_to_obj_attr."
        )

    @property
    def _source_nwb_object_type(self) -> Type:
        """The type of NWB object to import from the NWB file."""
        raise NotImplementedError(
            "IngestionMixin tables need to implement _source_nwb_object_type."
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

        for part_name, part in inspect.getmembers(
            type(self),
            lambda member: inspect.isclass(member)
            and issubclass(member, dj.Part),
        ):
            part_entries = config.get(part_name, [])
            if len(part_entries) == 0:
                continue
            entries.update(self._config_entries(part(), base_key, part_entries))

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
        if (
            hasattr(nwb_obj, "to_dataframe")
            and not self._single_entry_per_table
        ):
            obj_df = nwb_obj.to_dataframe()
            entries = dict()
            for row in obj_df.itertuples():
                # Keep every table a row generates, not just this one: a row
                # may produce part entries or a parent's entries alongside
                # its own. First-seen order is preserved, so a subclass that
                # yields a parent before self keeps that ordering.
                for (
                    table,
                    table_entries,
                ) in self.generate_entries_from_nwb_object(
                    row, base_key
                ).items():
                    entries.setdefault(table, []).extend(table_entries)
            return entries

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

            for k, v in mapping.items():
                # attribute name as string
                if isinstance(v, str):
                    base_key[k] = getattr(obj_, v)
                # attribute with default value as tuple (attr_name, default_val)
                elif (
                    isinstance(v, tuple)
                    and len(v) == 2
                    and isinstance(v[0], str)
                ):
                    base_key[k] = getattr(obj_, v[0], v[1])
                # callable function
                elif callable(v):
                    base_key[k] = v(obj_)
                else:
                    raise ValueError(
                        f"Invalid mapping for {k}: {v}. Must be str, "
                        + "tuple of (str, default), or callable."
                    )
        return {self: [base_key]}

    def populate(self, *restrictions, **kwargs):
        """Ingest whole NWB files rather than running `make` per key.

        Ingestion tables are filled by `insert_from_nwbfile`, which parses a
        file once and inserts every entry it yields, so `make` on these tables
        is a deprecation shim. Callers reaching for the DataJoint idiom are
        routed to the ingestion path, once per file that has no entries here
        yet -- otherwise `populate()` would call the shim and raise for any
        session whose rows this table happens to be missing.

        Tables not keyed by `nwb_file_name` fall through to DataJoint's
        `populate`, since a file cannot be identified for them.

        Parameters
        ----------
        *restrictions
            Restrictions on the key source, as for DataJoint's populate.
        **kwargs
            Accepted and ignored; ingestion takes no populate options.
        """
        from spyglass.common.common_nwbfile import Nwbfile

        if "nwb_file_name" not in self.primary_key:
            return super().populate(*restrictions, **kwargs)

        source = getattr(self, "key_source", None)
        if source is None:
            source = Nwbfile()
        if restrictions:
            source = source & dj.AndList(restrictions)

        files = {
            key["nwb_file_name"]
            for key in source.fetch("KEY", as_dict=True)
            if "nwb_file_name" in key
        }

        for nwb_file_name in sorted(files):
            if self & {"nwb_file_name": nwb_file_name}:
                continue  # already ingested for this file
            self.insert_from_nwbfile(nwb_file_name)

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
            if is_nwb_obj_type(obj, self._source_nwb_object_type)
        ]

        if self._source_nwb_object_name:
            if isinstance(self._source_nwb_object_name, str):
                self._source_nwb_object_name = [self._source_nwb_object_name]
            matching_objects = [
                obj
                for obj in matching_objects
                if self.sanitize_nwb_object_name(getattr(obj, "name", None))
                in [
                    self.sanitize_nwb_object_name(name)
                    for name in self._source_nwb_object_name
                ]
            ]

        if self._source_nwb_object_description:
            matching_objects = [
                obj
                for obj in matching_objects
                if self._matches_description(obj)
            ]

        return matching_objects

    def _matches_description(self, nwb_obj) -> bool:
        """Whether an object's description holds a declared marker.

        Parameters
        ----------
        nwb_obj : object
            A candidate NWB object.

        Returns
        -------
        bool
            True if no markers are declared, or if any marker appears in the
            object's description. Matched case- and space-insensitively, as
            for `_source_nwb_object_name`.
        """
        markers = self._source_nwb_object_description
        if not markers:
            return True
        if isinstance(markers, str):
            markers = [markers]

        description = self.sanitize_nwb_object_name(
            getattr(nwb_obj, "description", None)
        )
        if not description:
            return False

        return any(
            self.sanitize_nwb_object_name(marker) in description
            for marker in markers
        )

    @staticmethod
    def sanitize_nwb_object_name(name: Optional[str]) -> Optional[str]:
        """Sanitize NWB object name for case- and space-insensitive matching."""
        return name.lower().replace(" ", "") if name else None

    def _insert_logline(self, nwb_file_name=None, n_entries=0, table=None):
        """Log line for insert_from_nwbfile. Expects an instanced table."""
        this_tbl = table.camel_name if table is not None else ""
        self_tbl = self.camel_name

        suffix = "" if this_tbl == self_tbl else f" via {self_tbl}"
        self._info_msg(
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

        # fetch relevant NWB objects from file
        fetched_objs = self.get_nwb_objects(nwb_file, nwb_file_name)
        if len(fetched_objs) == 0 and not config:
            return dict()  # config may still supply entries on its own

        # check extension requirements (if any). Logs warning if objects found and
        # requirements not met
        if not self.check_extension_requirements(nwb_file_name):
            return dict()

        # compile list of table entries from all objects in this file
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
                    # setdefault, not indexing: a later object may generate
                    # entries for a table the first object did not touch.
                    entries.setdefault(table, []).extend(table_entries)

        if config:
            # Pass the base key: config entries need the file they belong to
            config_entries = self.generate_entries_from_config(
                config, base_entry.copy()
            )
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
        entries_to_insert = self.validate_duplicates(entries)

        # run insertions
        if not dry_run:
            self._run_nwbfile_insert(
                entries_to_insert, nwb_file_name=nwb_file_name
            )

        return entries

    def _run_nwbfile_insert(
        self, entries: IngestionEntries, nwb_file_name: str = None
    ) -> None:
        """Run insert on compiled Dict[TableObject, inserts]."""
        # An integrity here probably means a parallel insert was dropped
        # check debug_backup in parent func for entries that were dropped
        # One transaction for the whole plan
        with self._safe_context():
            for table, table_entries in entries.items():
                table.insert(
                    table_entries,
                    skip_duplicates=False,
                    allow_direct_insert=True,
                )
                self._insert_logline(nwb_file_name, len(table_entries), table)

    def _key_has_required_attrs(self, key):
        """Check that all non-nullable attributes are present in the key."""
        for attr in self.heading.attributes.values():
            if attr.nullable or attr.autoincrement or attr.default is not None:
                continue  # skip nullable, autoincrement, or default val attrs
            if attr.name not in key or key.get(attr.name) is None:
                self._info_msg(
                    f"Key {key} missing required attribute {attr.name}."
                )
                return False
        return True

    def _adjust_keys_for_entry(self, keys: List[dict]) -> List[dict]:
        """Passthrough. Allows children to adjust keys before comparing."""
        # Motivated by Subject.sex: comparing None to "U" should be equal
        # Without this step, reinsert triggers accept_divergence prompt
        # By default, checks that all non-nullable keys present
        return [key for key in keys if self._key_has_required_attrs(key)]

    def _remove_null_from_dicts(self, keys: List[dict]) -> List[dict]:
        """Remove null-valued items from each key in a list.

        Fallback for tables that do not implement `_adjust_keys_for_entry` --
        a table declared with SpyglassMixin rather than SpyglassIngestion,
        such as a parent generated alongside this one. Takes the same
        list-in/list-out shape as the method it stands in for; it previously
        took a single dict and so raised AttributeError for every caller.

        Parameters
        ----------
        keys : list of dict
            Planned entries for one table.

        Returns
        -------
        list of dict
            The same entries, without null or empty values.
        """
        return [
            {k: v for k, v in key.items() if not self._is_null(v)}
            for key in keys
        ]

    @staticmethod
    def _is_null(value) -> bool:
        """Whether a planned value is null, for tables with no adjustment."""
        if isinstance(value, np.ndarray):
            return value.size == 0
        return value is None or value == ""

    def _adjust_entries(
        self, entries: IngestionEntries, nwb_file_name: str = None
    ) -> Optional[IngestionEntries]:
        """Run _adjust_key for each table in planned entries.

        Given a Dict[TableObject, List[dict]], with planned entries values,
        run each table's _adjust_keys_for_entry function on the list of dicts.
        Removes invalid/null entries and tables with no valid entries.
        """

        null_keys = dict()  # key as emitted -> instanced, for the log line

        for table, table_entries in entries.items():
            # ensure instanced
            tbl = table() if inspect.isclass(table) else table

            # Allow children to adjust keys before comparing. A parent
            # generated alongside this table is a plain SpyglassMixin, with no
            # adjustment of its own.
            adjust_func = getattr(
                tbl, "_adjust_keys_for_entry", self._remove_null_from_dicts
            )
            adjusted_entries = adjust_func(table_entries)

            if not any(adjusted_entries):
                null_keys[table] = tbl  # mark for removal from dict
            else:
                entries[table] = adjusted_entries

        for table, tbl in null_keys.items():
            self._insert_logline(nwb_file_name, 0, tbl)
            _ = entries.pop(table)

        return entries if len(entries) > 0 else None

    def _expects_duplicates(self, tbl) -> bool:
        """Whether pre-existing entries in `tbl` are validated, not re-raised.

        Each table an ingestion emits answers for itself, so a table that
        legitimately recurs across files (Task, say) can be validated while
        the table driving the ingestion is not. Tables without the flag -- a
        plain SpyglassMixin parent generated alongside this one -- inherit
        this table's setting.

        Parameters
        ----------
        tbl : dj.Table
            A table appearing in the planned entries.

        Returns
        -------
        bool
            True if existing entries should be validated and skipped.
        """
        return getattr(tbl, "_expected_duplicates", self._expected_duplicates)

    def _dedup_within_batch(self, tbl, table_entries: List[dict]) -> List[dict]:
        """Collapse planned entries that share a primary key.

        The database check in `validate1_duplicate` compares each entry to what
        is already stored, not to its siblings in the same plan. Two objects in
        one file can name the same novel parent and `_run_nwbfile_insert`
        inserts with `skip_duplicates=False`, so the pair would raise and abort
        the file.

        Parameters
        ----------
        tbl : dj.Table
            The table the entries are planned for.
        table_entries : list of dict
            Planned entries for that table, in emission order.

        Returns
        -------
        list of dict
            The entries with later same-primary-key repeats removed.

        Raises
        ------
        dj.errors.DuplicateError
            If two planned entries share a primary key but disagree on a
            secondary value. Neither is stored yet, so there is no existing
            value to defer to.
        """
        seen = dict()
        deduped = []

        for entry in table_entries:
            pk = tuple(entry.get(attr) for attr in tbl.primary_key)
            if (first := seen.get(pk)) is None:
                seen[pk] = entry
                deduped.append(entry)
                continue
            for key in set(first).union(entry):
                if self._unequal_vals(key, first, entry):
                    raise dj.errors.DuplicateError(
                        f"{self.camel_name} generated conflicting entries "
                        + f"for {tbl.camel_name} key "
                        + f"{dict(zip(tbl.primary_key, pk))}: {key} is "
                        + f"{first.get(key)} in one and {entry.get(key)} "
                        + "in another."
                    )

        return deduped

    def validate_duplicates(self, entry_dict: Dict[dj.Table, List[dict]]):
        """Validate new entries against existing entries in the database.

        Entries are first de-duplicated against their siblings in the same
        plan. Only tables that expect duplicates are then validated against
        the database; the rest are passed through, so an unexpected duplicate
        still raises on insert.

        Parameters
        ----------
        entry_dict : dict or Dict[dj.Table, List[dict]]
            The new entry or dict of table entries to validate against existing
            entries in the database.

        Returns
        -------
        dict or Dict[dj.Table, List[dict]]
            The new entries to insert after validation. Avoids need to flag
            skip_duplicates
        """
        entries_to_insert = dict()
        for table, table_entries in entry_dict.items():
            if isinstance(table, type):
                table = table()  # instantiate table object if class provided

            table_entries = self._dedup_within_batch(table, table_entries)

            if not self._expects_duplicates(table):
                entries_to_insert[table] = table_entries
                continue

            entries_to_insert[table] = []
            for table_entry in table_entries:
                if entry := self.validate1_duplicate(table, table_entry):
                    entries_to_insert[table].append(entry)

        return entries_to_insert

    def validate1_duplicate(self, tbl, new_key):
        """Validate a single new entry against existing entries in the database.

        If divergence, prompt user whether to accept existing value

        Parameters
        ----------
        tbl : dj.Table
            The table to validate against.
        new_key : dict
            The new key to validate against existing entries in the database.

        Returns
        -------
        dict or None
            The new entry to insert after validation, or None if the entry
            already exists and is consistent.
        """
        # NOTE: `tbl` rather than `self` so that a table generated alongside
        # this one is validated against itself, not against this table.
        # Same fallback as _adjust_entries: tbl may be a plain SpyglassMixin
        # table generated alongside this one, with no adjustment of its own.
        adjust_func = getattr(
            tbl, "_adjust_keys_for_entry", self._remove_null_from_dicts
        )
        adjusted_entries = adjust_func([new_key])
        if not adjusted_entries:
            return  # entry filtered out by adjustment

        adj_new_key = adjusted_entries[0]
        primary_key = {
            k: v for k, v in adj_new_key.items() if k in tbl.primary_key
        }
        if not (query := (tbl & primary_key)):
            return new_key  # If novel primary key, nothing to validate

        existing = query.fetch1()

        for key in set(adj_new_key).union(existing):
            if not self._unequal_vals(key, adj_new_key, existing):
                continue  # skip if values are equal
            if not accept_divergence(
                key,
                adj_new_key.get(key),
                existing.get(key),
                self._test_mode,
                tbl.camel_name,
            ):
                # If the user does not accept the divergence,
                # raise an error to prevent data inconsistency
                raise dj.errors.DuplicateError(
                    f"Attempted entry in {self.camel_name} already exists "
                    + f"with different values for {key}: "
                    + f"{adj_new_key.get(key)} != {existing.get(key)}"
                )

        return  # validated existing entry, nothing to insert

    @staticmethod
    def _unequal_vals(key, a, b):
        a_val, b_val = a.get(key), b.get(key)

        # Arrays first: both `array or ""` and `array != other` yield an
        # array, whose truth value is ambiguous. Blob attributes reach here
        # from tables that emit a parent's entries alongside their own --
        # IntervalList.valid_times, say.
        if isinstance(a_val, np.ndarray) or isinstance(b_val, np.ndarray):
            return not np.array_equal(a_val, b_val)

        # Only None collapses to "": the point is to avoid a false positive on
        # None vs "". Coalescing every false value would treat a stored 0,
        # 0.0 or False as missing and hide a genuine divergence.
        a_val = "" if a_val is None else a_val
        b_val = "" if b_val is None else b_val

        if isinstance(a_val, str) and isinstance(b_val, str):
            return a_val.lower() != b_val.lower()
        return a_val != b_val

    def check_extension_requirements(self, nwb_file_name: str) -> bool:
        """Check that the NWB file meets the extension requirements (if any).

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file to check.

        Returns
        -------
        bool
            True if the NWB file meets the extension requirements.
        """
        # early exit if no extension requirements specified
        if not self._extension_requirements:
            return True

        from spyglass.common.common_nwbfile import Nwbfile

        nwb_file_path = Nwbfile().get_abs_path(nwb_file_name)
        file_namespaces = get_file_namespaces(nwb_file_path)

        for extension, min_version in self._extension_requirements.items():
            if (extension not in file_namespaces) or (
                Version(file_namespaces.get(extension)) < Version(min_version)
            ):
                logger.warning(
                    f"NWB file {nwb_file_name} can not be ingested into "
                    + f"{self.camel_name} due to unmet extension requirement:"
                    + f"{extension} >= {min_version} \n"
                    + "Please submit feature request or contact the Spyglass "
                    + "team for assistance."
                )
                return False
        return True
