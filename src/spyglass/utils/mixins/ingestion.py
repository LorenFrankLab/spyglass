import inspect
from datetime import datetime
from traceback import format_exc
from typing import Callable, Dict, List, Optional, Type, Union

import datajoint as dj
import numpy as np
from packaging.version import Version
from pynwb import NWBFile

from spyglass.data_import.ingestion_plan import (
    FileContext,
    PlannedEntries,
    Problem,
    TablePlan,
    row_key,
)
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

    Notes
    -----
    Ingestion runs once per file, and a table may cache file-level state on
    itself while it runs -- a camera map, an epoch lookup, an enumerator.
    Such state is the table's own, and **the table is responsible for
    resetting it when a new file is passed**, at the top of its
    `insert_from_nwbfile` (or wherever it first sees the new file). Two
    reasons this is not optional: a class-level `dict()` or counter is shared
    by every instance, so mutating it writes through to the class; and
    `populate` loops files on a single instance, so nothing else will clear
    it between files. Names cannot collide across tables -- each is a separate
    class -- so a table need only answer for its own.
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

    def _entries_from_mapping(self, nwb_obj, base_key=None) -> IngestionEntries:
        """Apply `table_key_to_obj_attr` to one object, yielding one entry.

        Parameters
        ----------
        nwb_obj : object
            The NWB object, or one row of one, to read attributes from.
        base_key : dict, optional
            Key fields the entry inherits.

        Returns
        -------
        IngestionEntries
            `{self: [entry]}` for this table alone.
        """
        base_key = dict(base_key or dict())  # avoid modifying original

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

    def plan_from_nwbfile(
        self,
        nwb_file_name: str,
        config: dict = None,
        nwb_file=None,
    ) -> TablePlan:
        """Parse an NWB file into the entries this table would insert.

        Writes nothing. Every failure becomes a `Problem` on the returned
        plan rather than an exception out of it, so one bad table does not
        hide what is wrong with the rest of the file.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file to parse.
        config : dict, optional
            A configuration dictionary to supplement NWB data. Default None.
        nwb_file : pynwb.NWBFile, optional
            An already-open file, so a caller planning many tables opens it
            once. Default None, fetching it here.

        Returns
        -------
        TablePlan
            The entries this table would insert, its status, and any problems.
        """
        from spyglass.common.common_nwbfile import Nwbfile

        empty = PlannedEntries().freeze()
        nwb_key = {"nwb_file_name": nwb_file_name}

        if nwb_file is None:
            if not (query := Nwbfile & nwb_key):
                return TablePlan(
                    table_name=self.full_table_name,
                    entries=empty,
                    status="failed",
                    problems=(
                        Problem(
                            severity="fatal",
                            code="file_not_registered",
                            message=f"NWB file {nwb_file_name} not in Nwbfile",
                            table=self.full_table_name,
                        ),
                    ),
                )
            nwb_file = query.fetch_nwb()[0]

        base_entry = nwb_key if "nwb_file_name" in self.primary_key else dict()
        ctx = FileContext(
            nwb_file_name=nwb_file_name,
            nwb_file=nwb_file,
            config=config or dict(),
            base_key=base_entry,
        )

        try:
            planned = self._parse(ctx)
        except Exception as err:  # a raise here is this table's failure alone
            return TablePlan(
                table_name=self.full_table_name,
                entries=empty,
                status="failed",
                problems=tuple(ctx.problems)
                + (
                    Problem(
                        severity="hard",
                        code="parse_error",
                        message=str(err),
                        table=self.full_table_name,
                        exc_type=type(err).__name__,
                        traceback=format_exc()[-2000:],
                    ),
                ),
                reads=tuple(ctx.reads),
            )

        return TablePlan(
            table_name=self.full_table_name,
            entries=planned.freeze(),
            status="ok" if planned else "skipped",
            problems=tuple(ctx.problems),
            reads=tuple(ctx.reads),
        )

    def _parse(self, ctx) -> PlannedEntries:
        """Run the parse contract for one file, returning its entries.

        Shared by planning and inserting; performs no writes.

        Parameters
        ----------
        ctx : FileContext
            Context for this file.

        Returns
        -------
        PlannedEntries
            Entries for this table and any other it feeds.
        """
        planned = PlannedEntries()

        self.before_parse(ctx)

        sources = self.find_sources(ctx)
        if len(sources) == 0 and not ctx.config:
            return planned  # config may still supply entries on its own

        # check extension requirements (if any). Logs warning if objects found
        # and requirements not met
        if not self.check_extension_requirements(ctx.nwb_file_name):
            ctx.problem(
                "soft",
                "extension_unmet",
                f"{ctx.nwb_file_name} does not meet this table's extension "
                + "requirements",
                table=self.full_table_name,
            )
            return planned

        if self._only_ingest_first:
            sources = sources[:1]

        for nwb_obj in sources:
            ctx.record_read(nwb_obj)
            planned.extend(self.entries_for_source(nwb_obj, ctx))

        planned.extend(self.entries_for_config(ctx))

        return planned

    # ------------------------- parse contract --------------------------

    def before_parse(self, ctx) -> None:
        """Prepare for parsing one file. Override to resolve file-level data.

        Runs before any source object is found, so lookups the mapping needs
        -- a config, a camera map, the session's intervals -- belong here,
        cached on `ctx`, rather than on the table object.

        Parameters
        ----------
        ctx : FileContext
            Context for this file.
        """

    def find_sources(self, ctx) -> List:
        """Return the NWB objects this table ingests from.

        Default: every object matching `_source_nwb_object_type`, filtered by
        `_source_nwb_object_name` if set.

        Parameters
        ----------
        ctx : FileContext
            Context for this file.

        Returns
        -------
        list
            Source objects, each passed to `entries_for_source`.
        """
        return self.get_nwb_objects(ctx.nwb_file, ctx.nwb_file_name)

    def entries_for_source(self, source, ctx) -> PlannedEntries:
        """Return the entries one source object generates.

        Default: a source that expands into rows -- a DynamicTable -- is
        handed row by row to `entries_for_row`; anything else goes there
        whole. Override this to work at the level of the container; override
        `entries_for_row` to work at the level of its rows. Neither override
        needs to test which it was given.

        Parameters
        ----------
        source : object
            One NWB object from `find_sources`.
        ctx : FileContext
            Context for this file. `ctx.base_key` holds the key fields every
            entry inherits, e.g. the file name.

        Returns
        -------
        PlannedEntries
            Entries for this table, and any other table it feeds.
        """
        # Legacy path: a table overriding generate_entries_from_nwb_object
        # still drives ingestion through it, row expansion included.
        if self._overrides_legacy_generate:
            return PlannedEntries.from_dict(
                self.generate_entries_from_nwb_object(
                    source, dict(ctx.base_key)
                )
            )

        if hasattr(source, "to_dataframe") and not self._single_entry_per_table:
            entries = PlannedEntries()
            for row in source.to_dataframe().itertuples():
                entries.extend(self.entries_for_row(row, ctx))
            return entries

        return self.entries_for_row(source, ctx)

    def entries_for_row(self, row, ctx) -> PlannedEntries:
        """Return the entries one row generates.

        Default: apply `table_key_to_obj_attr` to the row.

        Parameters
        ----------
        row : object
            One row of a source object, or the object itself when it does not
            expand into rows.
        ctx : FileContext
            Context for this file.

        Returns
        -------
        PlannedEntries
            Entries for this table, and any other table it feeds.
        """
        return PlannedEntries.from_dict(
            self._entries_from_mapping(row, dict(ctx.base_key))
        )

    def entries_for_config(self, ctx) -> PlannedEntries:
        """Return the entries this table's config declares.

        Default: the generic handling, which reads entries shaped as table
        keys. Override for a config that names its data some other way.

        Parameters
        ----------
        ctx : FileContext
            Context for this file.

        Returns
        -------
        PlannedEntries
            Entries the config supplies, empty if it supplies none.
        """
        if not ctx.config:
            return PlannedEntries()
        return PlannedEntries.from_dict(
            self.generate_entries_from_config(ctx.config, ctx.base_key)
        )

    def after_insert(self, ctx, inserted) -> None:
        """Run after entries are inserted. Override for follow-on work.

        The place for side effects that must follow persistence, which
        parsing itself must not perform.

        Parameters
        ----------
        ctx : FileContext
            Context for this file.
        inserted : dict
            The entries that were inserted, keyed by table.
        """

    @property
    def _overrides_legacy_generate(self) -> bool:
        """Whether this table still defines generate_entries_from_nwb_object."""
        return (
            type(self).generate_entries_from_nwb_object
            is not IngestionMixin.generate_entries_from_nwb_object
        )

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

        # One parse, shared with plan_from_nwbfile: the entries inserted here
        # are the entries a plan would have reported. Merging across source
        # objects is PlannedEntries' job, so a later object introducing a
        # table the first did not is no longer a KeyError.
        ctx = FileContext(
            nwb_file_name=nwb_file_name,
            nwb_file=nwb_file,
            config=config or dict(),
            base_key=base_entry,
        )
        entries = self._parse(ctx).as_dict()

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
            self.after_insert(ctx, entries_to_insert)

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

        # Datetimes are stored at second resolution and without a timezone,
        # so a value read back never matches the one parsed from the file
        # exactly. Compare what the database can actually hold.
        if isinstance(a_val, datetime) and isinstance(b_val, datetime):
            naive = [value.replace(tzinfo=None) for value in (a_val, b_val)]
            return abs((naive[0] - naive[1]).total_seconds()) >= 1

        # Only None collapses to "": the point is to avoid a false positive on
        # None vs "". Coalescing every false value would treat a stored 0,
        # 0.0 or False as missing and hide a genuine divergence.
        a_val = "" if a_val is None else a_val
        b_val = "" if b_val is None else b_val

        if isinstance(a_val, str) and isinstance(b_val, str):
            return a_val.lower() != b_val.lower()
        return a_val != b_val

    def check_planned_rows(self, rows, key_space, table=None) -> tuple:
        """Report what would go wrong inserting rows this table planned.

        Each failure becomes a `Problem` rather than an exception, so one bad
        table does not hide the rest of the file. A table answers for the
        entries it plans -- including those destined for a table it feeds,
        which need not itself ingest anything. Whether the file as a whole
        can be ingested is the planner's question, not this one.

        Parameters
        ----------
        rows : iterable of dict
            The planned entries.
        key_space : VirtualKeySpace
            The keys that will exist once the whole plan is inserted --
            what the database holds now, plus what the plan intends to add.
            Foreign keys are checked against it rather than against the
            database, so a table that emits its parent's entries alongside
            its own does not appear broken.
        table : dj.Table, optional
            The table the rows are destined for, as an instance. Default
            None, this table.

        Returns
        -------
        tuple of Problem
        """
        table = self if table is None else table
        problems: List[Problem] = []
        seen: set = set()

        for row in rows:
            key = row_key(table, row)

            if key in seen:
                problems.append(
                    Problem(
                        severity="hard",
                        code="duplicate_key",
                        message=f"Two planned entries share the key {dict(key)}",
                        table=table.full_table_name,
                    )
                )
            seen.add(key)

            problems.extend(self._required_attribute_problems(table, row))
            problems.extend(self._attribute_fit_problems(table, row))
            problems.extend(self._foreign_key_problems(table, row, key_space))
            problems.extend(self._divergence_problems(table, row))

        return tuple(problems)

    def _required_attribute_problems(self, table, row) -> List[Problem]:
        """Report attributes the table requires and the row does not supply."""
        return [
            Problem(
                severity="hard",
                code="missing_attribute",
                message=f"{attr.name} is required and absent",
                table=table.full_table_name,
            )
            for attr in table.heading.attributes.values()
            if not (
                attr.nullable or attr.autoincrement or attr.default is not None
            )
            and row.get(attr.name) is None
        ]

    def _attribute_fit_problems(self, table, row) -> List[Problem]:
        """Report values that do not fit the columns they are destined for."""
        problems = []

        for name, value in row.items():
            attr = table.heading.attributes.get(name)
            if attr is None or value is None:
                continue

            if attr.type.startswith("varchar") and isinstance(value, str):
                limit = int(attr.type[len("varchar(") : -1])
                if len(value) > limit:
                    problems.append(
                        Problem(
                            severity="hard",
                            code="value_too_long",
                            message=(
                                f"{name} is {len(value)} characters, "
                                + f"{attr.type} holds {limit}"
                            ),
                            table=table.full_table_name,
                        )
                    )

        return problems

    def _foreign_key_problems(self, table, row, key_space) -> List[Problem]:
        """Report parents this row points at that nothing will supply."""
        problems = []

        for parent, props in table.parents(
            as_objects=True, foreign_key_info=True
        ):
            # attr_map maps parent attribute -> child attribute
            attr_map = props.get("attr_map") or {}

            parent_key = {}
            for parent_attr in parent.primary_key:
                child_attr = attr_map.get(parent_attr, parent_attr)
                if child_attr not in row:
                    parent_key = None
                    break
                parent_key[parent_attr] = row[child_attr]

            if parent_key is None:  # nothing in this row addresses that parent
                continue
            if any(value is None for value in parent_key.values()):
                continue  # nullable foreign key, left empty

            if not key_space.holds(parent, row_key(parent, parent_key)):
                problems.append(
                    Problem(
                        severity="hard",
                        code="missing_parent",
                        message=(
                            f"{parent.full_table_name} has no entry for "
                            + f"{parent_key}, in the database or in this plan"
                        ),
                        table=table.full_table_name,
                    )
                )

        return problems

    def _divergence_problems(self, table, row) -> List[Problem]:
        """Report a planned entry that exists already with different values.

        A divergence is not novelty: the entry is present, and the plan
        disagrees with it. Recorded with the revision that would align the
        two, never resolved here -- a dry run does not prompt.
        """
        primary = {k: v for k, v in row.items() if k in table.primary_key}
        if len(primary) != len(table.primary_key):
            return []
        if not (query := (table & primary)):
            return []

        existing = query.fetch1()
        # Only what the plan actually specifies: an attribute the plan leaves
        # unset is not a disagreement with the row already stored, it is a
        # column this table does not populate.
        differing = {
            key: existing.get(key)
            for key in row
            if key not in table.primary_key
            and key in existing
            and self._unequal_vals(key, row, existing)
        }
        if not differing:
            return []

        return [
            Problem(
                severity="hard",
                code="divergence",
                message=(
                    f"{primary} exists with different values for "
                    + f"{sorted(differing)}"
                ),
                table=table.full_table_name,
                suggested_revision=differing,
            )
        ]

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
