import re
from inspect import getmodule
from itertools import chain as iter_chain
from pprint import pprint
from time import time
from typing import List, Union

import datajoint as dj
import numpy as np
from datajoint.condition import make_condition
from datajoint.errors import DataJointError
from datajoint.preview import repr_html
from datajoint.utils import from_camel_case, get_master, to_camel_case
from IPython.core.display import HTML

from spyglass.utils.logging import logger
from spyglass.utils.mixins.base import BaseMixin
from spyglass.utils.mixins.export import ExportMixin

RESERVED_PRIMARY_KEY = "merge_id"
RESERVED_SECONDARY_KEY = "source"
RESERVED_SK_LENGTH = 32
MERGE_DEFINITION = (
    f"\n    {RESERVED_PRIMARY_KEY}: uuid\n    ---\n"
    + f"    {RESERVED_SECONDARY_KEY}: varchar({RESERVED_SK_LENGTH})\n    "
)


def is_merge_table(table):
    """Return True if table fields exactly match Merge table."""

    def trim_def(definition):  # ignore full-line comments
        no_comment = re.sub(r"^\s*#.*\n", "\n", definition, flags=re.MULTILINE)
        no_blanks = re.sub(r"\n\s*\n", "\n", no_comment.strip())
        return no_blanks.replace(" ", "")

    if isinstance(table, str):
        table = dj.FreeTable(dj.conn(), table)
    if not isinstance(table, dj.Table):
        return False
    if get_master(table.full_table_name):
        return False  # Part tables are not merge tables
    if not table.is_declared:
        if tbl_def := getattr(table, "definition", None):
            return trim_def(MERGE_DEFINITION) == trim_def(tbl_def)
        logger.warning(
            f"Cannot determine merge table status for {table.table_name}"
        )
        return True
    return table.primary_key == [
        RESERVED_PRIMARY_KEY
    ] and table.heading.secondary_attributes == [RESERVED_SECONDARY_KEY]


class Merge(ExportMixin, dj.Manual):
    """Mixin for Merge tables: master table with one part per upstream pipeline.

    Preferred usage (instance methods on a restricted table)::

        result = MergeTable & {"nwb_file_name": "session.nwb"}
        result.fetch("lfp_sampling_rate")   # walks parts automatically
        result.get_part_table()             # Merge Part table for restriction
        result.get_parent_table()           # upstream source table
        result.view()                       # print merged union
        result.delete()                     # delete master + part entries
        result.delete_upstream(dry_run=True)# delete upstream source rows

    The ``&`` operator resolves part-table fields automatically: both dict
    restrictions (``{"nwb_file_name": "session.nwb"}``) and string
    restrictions (``'nwb_file_name LIKE "mini%"'``) that reference part-table
    field names are resolved to matching ``merge_id`` sets before the master
    is restricted.

    Use ``super_restrict()`` / ``super_fetch()`` to bypass part-resolution and
    operate on the master table directly.

    Deprecated class methods (``merge_X``) still work but emit a warning and
    will be removed in Spyglass 0.7.0. See the migration table in the
    Merge Tables documentation.
    """

    def __init__(self):
        super().__init__()
        self._reserved_pk = RESERVED_PRIMARY_KEY
        self._reserved_sk = RESERVED_SECONDARY_KEY
        if not self.is_declared:
            if not is_merge_table(self):  # Check definition
                self._warn_msg(
                    "Merge table with non-default definition\n"
                    + f"Expected:\n{MERGE_DEFINITION.strip()}\n"
                    + f"Actual  :\n{self.definition.strip()}"
                )
            for part in self.parts(as_objects=True):
                if part.primary_key != self.primary_key:
                    self._warn_msg(  # PK is only 'merge_id' in parts, no others
                        f"Unexpected primary key in {part.table_name}"
                        + f"\n\tExpected: {self.primary_key}"
                        + f"\n\tActual  : {part.primary_key}"
                    )

        self._source_class_dict = {}

    # ----------------------- Part-field restriction helpers ------------------

    _MASTER_ONLY_ATTRS = frozenset(
        [RESERVED_PRIMARY_KEY, RESERVED_SECONDARY_KEY]
    )
    _DJ_SPECIAL_ATTRS = frozenset(["KEY"])

    def _has_non_master_fields(self, restriction) -> bool:
        """True if restriction dict contains keys absent from master heading."""
        if not isinstance(restriction, dict):
            return False
        return any(k not in self.heading.names for k in restriction)

    def _resolve_restriction_to_merge_ids(self, restriction) -> list:
        """Return merge_ids matching a part-table-field restriction."""
        parts = self._merge_restrict_parts(
            restriction=restriction,
            as_objects=True,
            return_empties=False,
            add_invalid_restrict=False,
        )
        return [
            mid
            for part in parts
            for mid in part.fetch(RESERVED_PRIMARY_KEY, as_dict=False).tolist()
        ]

    def _get_part_only_fields(self) -> frozenset:
        """Field names that exist in parts but not in the master heading."""
        self._ensure_dependencies_loaded()
        master_names = frozenset(self.heading.names)
        return (
            frozenset(
                name
                for part in self.parts(as_objects=True)
                for name in part.heading.names
            )
            - master_names
        )

    def _string_has_part_field(self, restriction: str) -> bool:
        """True if a SQL string restriction references a part-table-only field."""
        return any(
            re.search(r"\b" + re.escape(f) + r"\b", restriction, re.IGNORECASE)
            for f in self._get_part_only_fields()
        )

    def _resolve_top_restriction(self):
        """Return restriction for part-walking, materializing _top if set.

        When _top is set (via ``T & dj.Top(limit=n)``), DataJoint applies the
        LIMIT/ORDER BY at the SQL level in super().fetch().  We materialize the
        resulting merge_ids so part-walking honours the same limit.
        """
        top = getattr(self, "_top", None)
        if top is None:
            return self.restriction or True
        limited_ids = super().fetch(RESERVED_PRIMARY_KEY, log_export=False)
        return (
            [{RESERVED_PRIMARY_KEY: mid} for mid in limited_ids]
            if len(limited_ids)
            else "FALSE"
        )

    def restrict(self, restriction, *args, **kwargs):
        """Resolve part-table fields before restricting master.

        Dict restrictions whose keys are absent from the master heading and
        string restrictions referencing part-table field names are resolved to
        matching ``merge_id`` sets before the master is restricted.
        ``dj.Top`` restrictions are delegated to DataJoint core unchanged.
        """
        should_resolve = self._has_non_master_fields(restriction) or (
            isinstance(restriction, str)
            and self._string_has_part_field(restriction)
        )
        if should_resolve:
            merge_ids = self._resolve_restriction_to_merge_ids(restriction)
            restriction = (
                [{RESERVED_PRIMARY_KEY: mid} for mid in merge_ids]
                if merge_ids
                else "FALSE"
            )
        return super().restrict(restriction, *args, **kwargs)

    def super_restrict(self, restriction):
        """Raw DataJoint restrict on master only, bypasses part resolution.

        Use when you intentionally want to restrict on master-only fields
        (``merge_id``, ``source``) without triggering part-table resolution.
        """
        return super().restrict(restriction)

    def fetch(self, *attrs, log_export=True, **kwargs):
        """Walk parts unless request is master-only (merge_id, source, KEY).

        Overrides ExportMixin.fetch.  Use ``super_fetch()`` for a raw
        master-only fetch when part-walking is unwanted.

        When the table was restricted with ``dj.Top``, the LIMIT is
        materialised into a merge_id set before part-walking so that the
        row-count contract is preserved.
        """
        if attrs and all(
            a in self._MASTER_ONLY_ATTRS | self._DJ_SPECIAL_ATTRS for a in attrs
        ):
            return super().fetch(*attrs, log_export=log_export, **kwargs)
        if not attrs and kwargs.get("format") == "array":
            return super().fetch(*attrs, log_export=log_export, **kwargs)
        return self._merge_fetch_impl(
            *attrs,
            restriction=self._resolve_top_restriction(),
            log_export=log_export,
            **kwargs,
        )

    def fetch1(self, *attrs, log_export=True, **kwargs):
        """Fetch exactly one entry, walking parts like fetch().

        Follows DataJoint convention: returns a dict (no attrs), a scalar
        (one attr), or a tuple (multiple attrs).  Raises DataJointError when
        the restricted table contains ≠ 1 row.
        """
        if attrs and all(
            a in self._MASTER_ONLY_ATTRS | self._DJ_SPECIAL_ATTRS for a in attrs
        ):
            return super().fetch1(*attrs, log_export=log_export, **kwargs)

        n = len(self)
        if n != 1:
            raise DataJointError(
                f"fetch1 expected exactly 1 entry, but found {n}."
            )

        rows = self.fetch(
            *attrs,
            log_export=log_export,
            as_dict=True,
            **{k: v for k, v in kwargs.items() if k != "as_dict"},
        )
        if not rows:
            raise DataJointError(
                f"fetch1 matched {n} row(s) by count but could not retrieve "
                f"the requested attributes {list(attrs)} from any part table. "
                "Verify that the attributes exist in the relevant part table."
            )
        row = rows[0]
        if not attrs:
            return row
        if len(attrs) == 1:
            return row[attrs[0]]
        return tuple(row[a] for a in attrs)

    def super_fetch(self, *args, **kwargs):
        """Raw fetch on master table only (merge_id, source columns).

        Use when part-walking is unwanted, e.g. listing all merge_ids without
        resolving part tables.
        """
        return super().fetch(*args, **kwargs)

    @staticmethod
    def _part_name(part=None):
        """Return the CamelCase name of a part table"""
        if not isinstance(part, str):
            part = part.table_name
        return to_camel_case(part.split("__")[-1].strip("`"))

    def get_source_from_key(self, key: dict) -> str:
        """Return the source of a given key"""
        return self._normalize_source(key)

    def parts(self, camel_case=False, *args, **kwargs) -> list:
        """Return a list of part tables, add option for CamelCase names.

        See DataJoint `parts` for additional arguments. If camel_case is True,
        forces return of strings rather than objects.
        """
        self._ensure_dependencies_loaded()

        if camel_case and kwargs.get("as_objects"):
            self._warn_msg(
                "Overriding as_objects=True to return CamelCase part names."
            )
            kwargs["as_objects"] = False

        parts = super().parts(*args, **kwargs)

        if camel_case:
            parts = [self._part_name(part) for part in parts]

        return parts

    @classmethod
    def _merge_restrict_parts(
        cls,
        restriction: str = True,
        as_objects: bool = True,
        return_empties: bool = True,
        add_invalid_restrict: bool = True,
    ) -> list:
        """Returns a list of parts with restrictions applied.

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the parts. Default True, no restrictions.
        as_objects: bool, optional
            Default True. Return part tables as objects
        return_empties: bool, optional
            Default True. Return empty part tables
        add_invalid_restrict: bool, optional
            Default True. Include part for which the restriction is invalid.

        Returns
        ------
        list
            list of datajoint tables, parts of Merge Table
        """

        cls._ensure_dependencies_loaded()

        # Normalize restriction to sql string
        restr_str = make_condition(cls(), restriction, set())

        parts_all = cls.parts(as_objects=True)
        # If the restriction makes ref to a source, we only want that part
        if (
            not return_empties
            and isinstance(restr_str, str)
            and f"`{cls()._reserved_sk}`" in restr_str
        ):
            parts_all = [
                part
                for part in parts_all
                if from_camel_case(
                    restr_str.split(f'`{cls()._reserved_sk}`="')[-1].split('"')[
                        0
                    ]
                )  # Only look at source part table
                in part.full_table_name
            ]
        if isinstance(restriction, dict):  # restr by source already done above
            _ = restriction.pop(cls()._reserved_sk, None)  # won't work for str
            # If a dict restriction has all invalid keys, it is treated as True
            if not add_invalid_restrict:
                parts_all = [  # so exclude tables w/ nonmatching attrs
                    p
                    for p in parts_all
                    if all([k in p.heading.names for k in restriction.keys()])
                ]

        parts = []
        for part in parts_all:
            try:
                parts.append(part.restrict(restriction))
            except DataJointError:  # If restriction not valid on given part
                if add_invalid_restrict:
                    parts.append(part)

        if not return_empties:
            parts = [p for p in parts if len(p)]
        if not as_objects:
            parts = [p.full_table_name for p in parts]

        return parts

    @classmethod
    def _merge_restrict_parents(
        cls,
        restriction: str = True,
        parent_name: str = None,
        as_objects: bool = True,
        return_empties: bool = True,
        add_invalid_restrict: bool = True,
    ) -> list:
        """Returns a list of part parents with restrictions applied.

        Rather than part tables, we look at parents of those parts, the source
        of the data.

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the returned parent. Default True, no
            restrictions.
        parent_name: str, optional
            CamelCase name of the parent.
        as_objects: bool, optional
            Default True. Return part tables as objects
        return_empties: bool, optional
            Default True. Return empty part tables
        add_invalid_restrict: bool, optional
            Default True. Include part for which the restriction is invalid.

        Returns
        ------
        list
            list of datajoint tables, parents of parts of Merge Table
        """
        # .restrict(restriction) does not work on returned part FreeTable
        # & part.fetch below restricts parent to entries in merge table
        part_parents = [
            parent
            & part.fetch(*part.heading.secondary_attributes, as_dict=True)
            for part in cls()._merge_restrict_parts(
                restriction=restriction,
                return_empties=return_empties,
                add_invalid_restrict=add_invalid_restrict,
            )
            for parent in part.parents(as_objects=True)  # ID respective parents
            if cls().table_name not in parent.full_table_name  # Not merge table
        ]
        if parent_name:
            part_parents = [
                p
                for p in part_parents
                if from_camel_case(parent_name) in p.full_table_name
            ]
        if not as_objects:
            part_parents = [p.full_table_name for p in part_parents]

        return part_parents

    @classmethod
    def _merge_repr(
        cls, restriction: str = True, include_empties=False
    ) -> dj.expression.Union:
        """Merged view, including null entries for columns unique to one part.

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the merged view
        include_empties: bool, optional
            Default False. Add columns for empty parts.

        Returns
        ------
        datajoint.expression.Union
        """

        parts = [  # join with master to include sec key (i.e., 'source')
            cls().join(p, log_export=False)
            for p in cls._merge_restrict_parts(
                restriction=restriction,
                add_invalid_restrict=False,
                return_empties=include_empties,
            )
        ]
        if not parts:
            cls()._warn_msg("No parts found. Try adjusting restriction.")
            return

        attr_dict = {  # NULL for non-numeric, 0 for numeric
            attr.name: "0" if attr.numeric else "NULL"
            for attr in iter_chain.from_iterable(
                part.heading.attributes.values() for part in parts
            )
        }

        def _proj_part(part):
            """Project part, adding NULL/0 for missing attributes"""
            return dj.U(*attr_dict.keys()) * part.proj(
                ...,  # include all attributes from part
                **{
                    k: v
                    for k, v in attr_dict.items()
                    if k not in part.heading.names
                },
            )

        query = _proj_part(parts[0])  # start with first part
        for part in parts[1:]:  # add remaining parts
            query += _proj_part(part)

        return query

    @classmethod
    def _merge_insert(cls, rows: list, part_name: str = None, **kwargs) -> None:
        """Insert rows into merge, ensuring data exists in part parent(s).

        Parameters
        ---------
        rows: List[dict]
            An iterable where an element is a dictionary.
        part: str, optional
            CamelCase name of the part table

        Raises
        ------
        TypeError
            If rows is not a list of dicts
        ValueError
            If data doesn't exist in part parents, integrity error
        """
        cls._ensure_dependencies_loaded()

        type_err_msg = "Input `rows` must be a list of dictionaries"
        try:
            for r in iter(rows):
                if not isinstance(r, dict):
                    raise TypeError(type_err_msg)
        except TypeError:
            raise TypeError(type_err_msg)

        parts = cls._merge_restrict_parts(as_objects=True)
        if part_name:
            parts = [
                p
                for p in parts
                if from_camel_case(part_name) in p.full_table_name
            ]

        master_entries = []
        parts_entries = {p: [] for p in parts}
        for row in rows:
            keys = []  # empty to-be-inserted keys
            for part in parts:  # check each part
                part_name = cls._part_name(part)
                part_parent = part.parents(as_objects=True)[-1]
                if part_parent & row:  # if row is in part parent
                    keys = (part_parent & row).fetch("KEY")  # get pk
                    if len(keys) > 1:
                        raise ValueError(
                            "Ambiguous entry. Data has mult rows in "
                            + f"{part_name}:\n\tData:{row}\n\t{keys}"
                        )
                    key = keys[0]
                    if part & key:
                        logger.info(f"Key already in part {part_name}: {key}")
                        continue
                    master_sk = {cls()._reserved_sk: part_name}
                    uuid = dj.hash.key_hash(key | master_sk)
                    master_pk = {cls()._reserved_pk: uuid}

                    master_entries.append({**master_pk, **master_sk})
                    parts_entries[part].append({**master_pk, **key})

            if not keys:
                raise ValueError(
                    "Non-existing entry in any of the parent tables - Entry: "
                    + f"{row}"
                )

        with cls._safe_context():
            super().insert(cls(), master_entries, **kwargs)
            for part, part_entries in parts_entries.items():
                part.insert(part_entries, **kwargs)

    @classmethod
    def _ensure_dependencies_loaded(cls) -> None:
        """Ensure connection dependencies loaded.

        Otherwise parts returns none
        """
        if not dj.conn.connection.dependencies._loaded:
            dj.conn.connection.dependencies.load()

    def insert(self, rows: list, **kwargs):
        """Merges table specific insert, ensuring data exists in part parents.

        Parameters
        ---------
        rows: List[dict]
            An iterable where an element is a dictionary.

        Raises
        ------
        TypeError
            If rows is not a list of dicts
        ValueError
            If data doesn't exist in part parents, integrity error
        """
        self._merge_insert(rows, **kwargs)

    # -------------------- Deprecation helper --------------------------------

    @staticmethod
    def _deprecate(name: str, alt: str) -> None:
        """Log deprecation warning for a merge method (v0.7.0 removal)."""
        from spyglass.common.common_usage import ActivityLog

        ActivityLog().deprecate_log(name=name, alt=alt, version="0.7.0")

    # ------------- Private implementation methods (no deprecation) -----------

    @classmethod
    def _merge_restrict_parts_to_tables(
        cls,
        restriction: str = True,
        join_master: bool = False,
        restrict_part: bool = True,
        multi_source: bool = False,
        return_empties: bool = False,
    ):
        """Core logic shared by merge_get_part and get_part_table."""
        sources = [
            cls._part_name(part)
            for part in cls._merge_restrict_parts(
                restriction=restriction,
                as_objects=False,
                return_empties=return_empties,
                add_invalid_restrict=False,
            )
        ]

        if not multi_source and len(sources) != 1:
            if len(sources) == 0:
                raise ValueError(
                    "Found 0 matching parts. The upstream source may not have "
                    "been inserted into this merge table yet.\n\t"
                    + "Checked parts: "
                    + str(
                        [p.full_table_name for p in cls.parts(as_objects=True)]
                    )
                )
            raise ValueError(
                f"Found {len(sources)} potential parts: {sources}\n\t"
                + "Try adding a restriction before invoking `get_part_table`.\n\t"
                + "Or permitting multiple sources with `multi_source=True`."
            )
        if len(sources) == 0:
            return None

        parts = [
            (
                getattr(cls, source)().restrict(restriction)
                if restrict_part
                else getattr(cls, source)()
            )
            for source in sources
        ]
        if join_master:
            parts = [cls * part for part in parts]

        return parts if multi_source else parts[0]

    @classmethod
    def _merge_restrict_parents_to_tables(
        cls,
        restriction: str = True,
        join_master: bool = False,
        multi_source: bool = False,
        return_empties: bool = False,
        add_invalid_restrict: bool = True,
    ):
        """Core logic shared by merge_get_parent and get_parent_table."""
        part_parents = cls._merge_restrict_parents(
            restriction=restriction,
            as_objects=True,
            return_empties=return_empties,
            add_invalid_restrict=add_invalid_restrict,
        )

        if not multi_source and len(part_parents) != 1:
            raise ValueError(
                f"Found {len(part_parents)} potential parents: {part_parents}"
                + "\n\tTry adding a string restriction when invoking "
                + "`get_parent_table()`. Or permitting multiple sources with "
                + "`multi_source=True`."
            )

        if join_master:
            part_parents = [cls * part for part in part_parents]

        return part_parents if multi_source else part_parents[0]

    @classmethod
    def _merge_delete_parent_restricted(
        cls,
        restriction: str = True,
        dry_run: bool = True,
        **kwargs,
    ):
        """Core logic shared by merge_delete_parent and delete_upstream."""
        part_parents = cls._merge_restrict_parents(
            restriction=restriction, as_objects=True, return_empties=False
        )

        if dry_run:
            return part_parents

        merge_ids = cls._merge_repr(restriction=restriction).fetch(
            RESERVED_PRIMARY_KEY, as_dict=True
        )

        super().delete((cls & merge_ids), **kwargs)

        if cls & merge_ids:
            return

        for part_parent in part_parents:
            super().delete(part_parent, **kwargs)

    # -------------------- Deprecated merge_X classmethods -------------------

    @classmethod
    def merge_view(cls, restriction: str = True):
        """Print merged view, including null entries for unique columns.

        .. deprecated:: 0.7.0
            Use ``(T & restriction).view()`` instead.

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the merged view
        """

        # If we overwrite `preview`, we then encounter issues with operators
        # getting passed a `Union`, which doesn't have a method we can
        # intercept to manage master/parts
        cls._deprecate("merge_view", "(T & restriction).view()")
        return pprint(cls._merge_repr(restriction=restriction))

    @classmethod
    def merge_html(cls, restriction: str = True):
        """Displays HTML in notebooks."""
        cls._deprecate("merge_html", "(T & restriction).html()")
        return HTML(repr_html(cls._merge_repr(restriction=restriction)))

    @classmethod
    def merge_restrict(cls, restriction: str = True) -> dj.U:
        """Return a merged Union view with restriction applied.

        .. deprecated:: 0.7.0
            Use ``MergeTable & restriction`` instead. The ``&`` operator now
            resolves part-table fields automatically.

        Parameters
        ----------
        restriction: str
            Restriction to apply to the merged view.

        Returns
        -------
        datajoint.Union
            Merged view with restriction applied.
        """
        cls._deprecate("merge_restrict", "T & restriction")
        return cls._merge_repr(restriction=restriction)

    @classmethod
    def merge_delete(cls, restriction: str = True, **kwargs):
        """Delete entries matching restriction from master and part tables.

        .. deprecated:: 0.7.0
            Use ``(T & restriction).delete()`` instead.

        Parameters
        ----------
        restriction: str
            Optional restriction. If omitted, deletes all entries.
        kwargs: dict
            Additional keyword arguments forwarded to DataJoint ``delete``.

        Example
        -------
            >>> (MergeTable & "field = 1").delete()
        """
        cls._deprecate("merge_delete", "(T & restriction).delete()")
        query = cls._merge_repr(restriction=restriction)
        if query is None:
            return  # No parts, nothing to delete
        uuids = [
            {k: v}
            for entry in query.fetch("KEY")
            for k, v in entry.items()
            if k == RESERVED_PRIMARY_KEY
        ]
        if not uuids:
            return  # Nothing to delete
        (cls() & uuids).delete(**kwargs)

    @classmethod
    def merge_delete_parent(
        cls, restriction: str = True, dry_run=True, **kwargs
    ) -> list:
        """Delete entries from merge master, part, and respective part parents

        Note: Clears merge entries from their respective parents.

        Parameters
        ----------
        restriction: str
            Optional restriction to apply before deletion from parents. If not
            provided, delete all entries present in Merge Table.
        dry_run: bool
            Default True. If true, return list of tables with entries that would
            be deleted. Otherwise, table entries.
        kwargs: dict
            Additional keyword arguments for DataJoint delete.
        """
        cls._deprecate(
            "merge_delete_parent",
            "(T & restriction).delete_upstream(dry_run=...)",
        )
        return cls._merge_delete_parent_restricted(
            restriction=restriction, dry_run=dry_run, **kwargs
        )

    def fetch_nwb(
        self,
        restriction: str = None,
        multi_source=False,
        disable_warning=False,
        return_merge_ids=False,
        log_export=True,
        *attrs,
        **kwargs,
    ):
        """Return the (Analysis)Nwbfile file linked in the source.

        Relies on SpyglassMixin._nwb_table_tuple to determine the table to
        fetch from and the appropriate path attribute to return.

        Parameters
        ----------
        restriction: str, optional
            Restriction to apply to parents before running fetch. Default True.
        multi_source: bool
            Return from multiple parents. Default False.
        return_merge_ids: bool
            Default False. Return merge_ids with nwb files.
        log_export: bool
            Default True. During export, log this fetch an export event.

        Notes
        -----
        Nwb files not strictly returned in same order as self
        """
        if isinstance(self, dict):
            raise ValueError("Try replacing Merge.method with Merge().method")
        restriction = restriction or self.restriction or True
        merge_restriction = self.extract_merge_id(restriction)

        sources = set(
            (self & merge_restriction).fetch(
                self._reserved_sk, log_export=False
            )
        )
        nwb_list = []
        merge_ids = []
        for source in sources:
            source_restr = (
                self
                & dj.AndList([{self._reserved_sk: source}, merge_restriction])
            ).fetch("KEY", log_export=False)
            nwb_list.extend(
                (self & source_restr)
                .merge_restrict_class(
                    restriction,
                    permit_multiple_rows=True,
                    add_invalid_restrict=False,
                )
                .fetch_nwb()
            )
            if return_merge_ids:
                merge_ids.extend(
                    [
                        (
                            self
                            & dj.AndList(
                                [self._merge_restrict_parts(file), source_restr]
                            )
                        ).fetch1(self._reserved_pk)
                        for file in nwb_list
                    ]
                )
        if return_merge_ids:
            return nwb_list, merge_ids
        return nwb_list

    @classmethod
    def merge_get_part(
        cls,
        restriction: str = True,
        join_master: bool = False,
        restrict_part=True,
        multi_source=False,
        return_empties=False,
    ) -> dj.Table:
        """Retrieve part table from a restricted Merge table.

        Note: unlike other Merge Table methods, returns the native table, not
        a FreeTable

        Parameters
        ----------
        restriction: str
            Optional restriction to apply before determining part to return.
            Default True.
        join_master: bool
            Join part with Merge master to show source field. Default False.
        restrict_part: bool
            Apply restriction to part. Default True. If False, return the
            native part table.
        multi_source: bool
            Return multiple parts. Default False.
        return_empties: bool
            Default False. Return empty part tables.

        Returns
        ------
        Union[dj.Table, List[dj.Table]]
            Native part table(s) of Merge. If `multi_source`, returns list.

        Example
        -------
            >>> (MergeTable & restriction).get_part_table()
            >>> MergeTable().merge_get_part(restriction, join_master=True)

        Raises
        ------
        ValueError
            If multiple sources are found, but not expected lists and suggests
            restricting
        """
        cls._deprecate(
            "merge_get_part", "(T & restriction).get_part_table(...)"
        )
        return cls._merge_restrict_parts_to_tables(
            restriction=restriction,
            join_master=join_master,
            restrict_part=restrict_part,
            multi_source=multi_source,
            return_empties=return_empties,
        )

    @classmethod
    def merge_get_parent(
        cls,
        restriction: str = True,
        join_master: bool = False,
        multi_source: bool = False,
        return_empties: bool = False,
        add_invalid_restrict: bool = True,
    ) -> dj.FreeTable:
        """Returns a list of part parents with restrictions applied.

        Rather than part tables, we look at parents of those parts, the source
        of the data, and only the rows that have keys inserted in the merge
        table.

        Parameters
        ----------
        restriction: str
            Optional restriction to apply before determining parent to return.
            Default True.
        join_master: bool
            Default False. Join part with Merge master to show uuid and source
        multi_source: bool
            Return multiple parents. Default False.
        return_empties: bool
            Default False. Return empty parent tables.
        add_invalid_restrict: bool
            Default True. Include parent for which the restriction is invalid.

        Returns
        ------
        dj.FreeTable
            Parent of parts of Merge Table as FreeTable.
        """
        cls._deprecate(
            "merge_get_parent", "(T & restriction).get_parent_table(...)"
        )
        return cls._merge_restrict_parents_to_tables(
            restriction=restriction,
            join_master=join_master,
            multi_source=multi_source,
            return_empties=return_empties,
            add_invalid_restrict=add_invalid_restrict,
        )

    @property
    def source_class_dict(self) -> dict:
        """Dictionary of part names and their respective classes."""
        # NOTE: fails if table is aliased in dj.Part but not merge script
        # i.e., must import aliased table as part name
        if not self._source_class_dict:
            module = getmodule(self)
            self._source_class_dict = {
                part_name: getattr(module, part_name)
                for part_name in self.parts(camel_case=True)
                if hasattr(module, part_name)
            }
        for part_name in self.parts(camel_case=True):
            if part_name not in self._source_class_dict:
                logger.warning(f"Missing code for {part_name}")
        return self._source_class_dict

    def _normalize_source(
        self, source: Union[str, dj.Table, dj.condition.AndList, dict]
    ) -> str:
        fetched_source = None
        if isinstance(source, (Merge, dj.condition.AndList)):
            try:
                fetched_source = (self & source).fetch(self._reserved_sk)
            except DataJointError:
                raise ValueError(f"Unable to find source for {source}")
            source = fetched_source[0]
            if len(fetched_source) > 1:
                logger.warning(f"Multiple sources. Selecting first: {source}.")
        if isinstance(source, dj.Table):
            source = self._part_name(source)
        if isinstance(source, dict):
            source = self._part_name(
                self.__class__._merge_restrict_parents_to_tables(source)
            )

        return source

    def merge_get_parent_class(self, source: str) -> dj.Table:
        """Return the class of the parent table for a given CamelCase source.

        Parameters
        ----------
        source: Union[str, dict, dj.Table]
            Accepts a CamelCase name of the source, or key as a dict, or a part
            table.

        Returns
        -------
        dj.Table
            Class instance of the parent table, including class methods.
        """

        ret = self.source_class_dict.get(self._normalize_source(source))
        if not ret:
            logger.error(
                f"No source class found for {source}: \n\t"
                + f"{self.parts(camel_case=True)}"
            )
        return ret

    def merge_restrict_class(
        self,
        key: dict,
        permit_multiple_rows: bool = False,
        add_invalid_restrict=True,
    ) -> dj.Table:
        """Returns native parent class, restricted with key."""
        parent = self.__class__._merge_restrict_parents_to_tables(
            key, add_invalid_restrict=add_invalid_restrict
        )
        parent_key = parent.fetch("KEY", as_dict=True)

        if not permit_multiple_rows and len(parent_key) > 1:
            raise ValueError(
                f"Ambiguous entry. Data has mult rows in parent:\n\tData:{key}"
                + f"\n\t{parent_key}"
            )

        parent_class = self.merge_get_parent_class(parent)
        return parent_class & parent_key

    @classmethod
    def merge_fetch(
        cls, restriction: str = True, *attrs, log_export=True, **kwargs
    ) -> list:
        """Perform a fetch across all parts. If >1 result, return as a list.

        Parameters
        ----------
        restriction: str
            Optional restriction to apply before determining parent to return.
            Default True.
        log_export: bool
            Default True. During export, log this fetch an export event.
        attrs, kwargs
            arguments passed to DataJoint `fetch` call

        Returns
        -------
        Union[ List[np.array], List[dict], List[pd.DataFrame] ]
            Table contents, with type determined by kwargs
        """
        cls._deprecate("merge_fetch", "(T & restriction).fetch(*attrs)")
        return cls()._merge_fetch_impl(
            *attrs,
            restriction=restriction,
            log_export=log_export,
            **kwargs,
        )

    def _merge_fetch_impl(
        self, *attrs, restriction: str = True, log_export=True, **kwargs
    ) -> list:
        """Core part-walking fetch logic, shared by fetch() and merge_fetch()."""
        if log_export and self.export_id:
            self._log_fetch(  # Transforming restriction to merge_id
                restriction=self._merge_repr(restriction=restriction).fetch(
                    RESERVED_PRIMARY_KEY, as_dict=True
                )
            )

        results = []
        parts = self._merge_restrict_parts(
            restriction=restriction,
            as_objects=True,
            return_empties=False,
            add_invalid_restrict=False,
        )

        for part in parts:
            try:
                results.append(part.fetch(*attrs, **kwargs))
            except DataJointError as e:
                logger.warning(
                    f"{e.args[0]} Skipping "
                    + to_camel_case(part.table_name.split("__")[-1])
                )

        if not results:
            logger.info(
                "No merge_fetch results.\n\t"
                + "If not restricting, try: `M.fetch(True,'attr')\n\t"
                + "If restricting by source, use dict: "
                + "`M.fetch({'source':'X'}"
            )
            return []

        if len(results) == 1:
            return results[0]

        # Multiple parts — merge preserving the return type of each fetch
        first = results[0]
        if isinstance(first, np.ndarray):
            # Single-attr fetch (no as_dict): list of arrays → concatenate
            return np.concatenate(results)
        if isinstance(first, list):
            if first and isinstance(first[0], np.ndarray):
                # Multi-attr fetch (no as_dict): each result is [arr_a, arr_b, ...]
                return [
                    np.concatenate([r[i] for r in results])
                    for i in range(len(first))
                ]
            # as_dict=True: each result is a list of dicts — flatten
            return [item for sub in results for item in sub]
        try:  # format='frame' returns DataFrame
            import pandas as pd

            if isinstance(first, pd.DataFrame):
                return pd.concat(results, ignore_index=True)
        except ImportError:
            pass
        return results

    def merge_populate(self, source: str, keys=None, **kwargs):
        """Populate source table and insert successes into merge.

        Deprecated — use ``(T & restriction).populate(source, keys)``.
        """
        self._deprecate(
            "merge_populate", "(T & restriction).populate(source, keys)"
        )
        parent_class = self.merge_get_parent_class(source)
        if parent_class is None:
            raise ValueError(f"No parent class found for source: {source}")
        if keys is None and hasattr(parent_class, "key_source"):
            keys = parent_class.key_source
        if keys is not None:
            parent_class.populate(keys, **kwargs)
            successes = (parent_class & keys).fetch("KEY", as_dict=True)
        else:
            parent_class.populate(**kwargs)
            successes = parent_class().fetch("KEY", as_dict=True)
        self.insert(successes, skip_duplicates=True)

    # -------------------- Instance-method replacements ----------------------

    def get_part_table(
        self,
        join_master: bool = False,
        restrict_part: bool = True,
        multi_source: bool = False,
        return_empties: bool = False,
    ):
        """Return part table(s) for self.restriction.

        Instance-method replacement for ``merge_get_part``. Use as
        ``(T & restriction).get_part_table()``.
        """
        return self.__class__._merge_restrict_parts_to_tables(
            restriction=self.restriction or True,
            join_master=join_master,
            restrict_part=restrict_part,
            multi_source=multi_source,
            return_empties=return_empties,
        )

    def get_parent_table(
        self,
        join_master: bool = False,
        multi_source: bool = False,
        return_empties: bool = False,
        add_invalid_restrict: bool = True,
    ):
        """Return parent table(s) for self.restriction.

        Instance-method replacement for ``merge_get_parent``. Use as
        ``(T & restriction).get_parent_table()``.
        """
        return self.__class__._merge_restrict_parents_to_tables(
            restriction=self.restriction or True,
            join_master=join_master,
            multi_source=multi_source,
            return_empties=return_empties,
            add_invalid_restrict=add_invalid_restrict,
        )

    def delete_upstream(self, dry_run: bool = True, **kwargs):
        """Delete merge entry and its upstream part-parent rows.

        Instance-method replacement for ``merge_delete_parent``. Use as
        ``(T & restriction).delete_upstream(dry_run=True)``.
        """
        return self.__class__._merge_delete_parent_restricted(
            restriction=self.restriction or True,
            dry_run=dry_run,
            **kwargs,
        )

    def view(self, include_empties: bool = False):
        """Return merged preview string for self.restriction.

        Respects ``dj.Top`` limits set via ``T & dj.Top(limit=n)``.
        Returns None when the restricted view is empty.
        """
        query = self._merge_repr(
            restriction=self._resolve_top_restriction(),
            include_empties=include_empties,
        )
        if query is None:
            return None
        return query.preview()

    def __repr__(self):
        """Show merged, part-walked preview by default."""
        out = self.view()
        if out is not None:
            return out
        return super().__repr__()

    def super_view(self):
        """Show master-only preview (no part-walking)."""
        return super().__repr__()

    def _repr_html_(self):
        """HTML repr for notebooks, showing merged view as HTML string."""
        query = self._merge_repr(
            restriction=self._resolve_top_restriction(),
            include_empties=False,
        )
        if query is None:
            return "<i>&lt;empty&gt;</i>"
        return repr_html(query)

    def html(self, include_empties: bool = False):
        """Return HTML merged view for self.restriction (notebooks).

        Instance-method replacement for ``merge_html``. Use as
        ``(T & restriction).html()``.
        Respects ``dj.Top`` limits set via ``T & dj.Top(limit=n)``.
        """
        return HTML(
            repr_html(
                self._merge_repr(
                    restriction=self._resolve_top_restriction(),
                    include_empties=include_empties,
                )
            )
        )

    def populate(self, source: str, keys=None, **kwargs):
        """Populate source table and insert successes into merge.

        Instance-method replacement for ``merge_populate``. Use as
        ``T.populate(source, keys)``.

        Parameters
        ----------
        source : str
            CamelCase name of the source table.
        keys : list or dj.expression, optional
            Keys to populate. If None and source is a Computed table,
            uses ``key_source``.
        **kwargs
            Forwarded to ``parent_class.populate``.
        """
        parent_class = self.merge_get_parent_class(source)
        if parent_class is None:
            raise ValueError(f"No parent class found for source: {source}")
        if keys is None and hasattr(parent_class, "key_source"):
            keys = parent_class.key_source
        if keys is not None:
            parent_class.populate(keys, **kwargs)
            successes = (parent_class & keys).fetch("KEY", as_dict=True)
        else:
            parent_class.populate(**kwargs)
            successes = parent_class().fetch("KEY", as_dict=True)
        self.insert(successes, skip_duplicates=True)

    def delete(self, force_permission=False, *args, **kwargs):
        """Delete master and all part entries for current restriction.

        DataJoint cascades master→part deletions automatically when deleting
        from the master table, so no need to delete Part rows separately.
        `force_permission` is handled upstream by cautious_delete.
        """
        if not len(self):
            return
        super().delete(*args, **kwargs)

    def super_delete(self, warn=True, *args, **kwargs):
        """Alias for datajoint.table.Table.delete.

        Added to support MRO of SpyglassMixin
        """
        if warn:
            self._warn_msg("!! Bypassing cautious_delete !!")
            self._log_delete(start=time(), super_delete=True)
        super().delete(*args, **kwargs)

    @classmethod
    def extract_merge_id(cls, restriction) -> Union[dict, list]:
        """Utility function to extract merge_id from a restriction

        Removes all other restricted attributes, and defaults to a
        universal set (either empty dict or True) when there is no
        merge_id present in the input, relying on parent func to
        restrict on secondary or part-parent key(s).

        Assumes that a valid set of merge_id keys should have OR logic
        to allow selection of an entries.

        Parameters
        ----------
        restriction : str, dict, or dj.condition.AndList
            A datajoint restriction

        Returns
        -------
        restriction
            A restriction containing only the merge_id key
        """
        if restriction is None:
            return None
        if isinstance(restriction, dict):
            if merge_id := restriction.get("merge_id"):
                return {"merge_id": merge_id}
            else:
                return {}
        merge_restr = []
        if isinstance(restriction, dj.condition.AndList) or isinstance(
            restriction, List
        ):
            merge_id_list = [cls.extract_merge_id(r) for r in restriction]
            merge_restr = [x for x in merge_id_list if x is not None]
        elif isinstance(restriction, str):
            parsed = [x.split(")")[0] for x in restriction.split("(") if x]
            merge_restr = [x for x in parsed if "merge_id" in x]

        if len(merge_restr) == 0:
            return True
        return merge_restr


_Merge = Merge

# Underscore as class name avoids errors when this included in a Diagram
# Aliased because underscore otherwise excludes from API docs.


def delete_downstream_merge(
    table: dj.Table,
    **kwargs,
) -> list:
    """Given a table/restriction, id or delete relevant downstream merge entries

    Passthrough to SpyglassMixin.delete_downstream_parts
    """
    from spyglass.common.common_usage import ActivityLog
    from spyglass.utils.dj_mixin import SpyglassMixin

    ActivityLog().deprecate_log(
        name="delete_downstream_merge",
        alt="Table.delete",
    )

    if not isinstance(table, SpyglassMixin):
        raise ValueError("Input must be a Spyglass Table.")

    table = table if isinstance(table, dj.Table) else table()

    return table.delete(**kwargs)
