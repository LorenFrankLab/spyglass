import re
from contextlib import nullcontext
from inspect import getmodule
from itertools import chain as iter_chain
from pprint import pprint
from typing import Union

import datajoint as dj
from datajoint.condition import make_condition
from datajoint.errors import DataJointError
from datajoint.preview import repr_html
from datajoint.utils import from_camel_case, get_master, to_camel_case
from IPython.core.display import HTML

from spyglass.utils.logging import logger

RESERVED_PRIMARY_KEY = "merge_id"
RESERVED_SECONDARY_KEY = "source"
RESERVED_SK_LENGTH = 32


class Merge(dj.Manual):
    """Adds funcs to support standard Merge table operations.

    Many methods have the @classmethod decorator to permit MergeTable.method()
    symtax. This makes access to instance attributes (e.g., (MergeTable &
    "example='restriction'").restriction) harder, but these attributes have
    limited utility when the user wants to, for example, restrict the merged
    view rather than the master table itself.
    """

    def __init__(self):
        super().__init__()
        self._reserved_pk = RESERVED_PRIMARY_KEY
        self._reserved_sk = RESERVED_SECONDARY_KEY
        merge_def = (
            f"\n    {self._reserved_pk}: uuid\n    ---\n"
            + f"    {self._reserved_sk}: varchar({RESERVED_SK_LENGTH})\n    "
        )
        if not self.is_declared:
            # remove comments after # from each line of definition
            if self._remove_comments(self.definition) != merge_def:
                logger.warn(
                    "Merge table with non-default definition\n\t"
                    + f"Expected: {merge_def.strip()}\n\t"
                    + f"Actual  : {self.definition.strip()}"
                )
            for part in self.parts(as_objects=True):
                if part.primary_key != self.primary_key:
                    logger.warn(
                        f"Unexpected primary key in {part.table_name}"
                        + f"\n\tExpected: {self.primary_key}"
                        + f"\n\tActual  : {part.primary_key}"
                    )
        self._analysis_nwbfile = None
        self._source_class_dict = {}

    @property
    def source_class_dict(self) -> dict:
        if not self._source_class_dict:
            module = getmodule(self)
            self._source_class_dict = {
                part_name: getattr(module, part_name)
                for part_name in self.parts(camel_case=True)
            }
        return self._source_class_dict

    @property  # CB: This is a property to avoid circular import
    def analysis_nwbfile(self):
        if self._analysis_nwbfile is None:
            from spyglass.common import AnalysisNwbfile  # noqa F401

            self._analysis_nwbfile = AnalysisNwbfile
        return self._analysis_nwbfile

    def _remove_comments(self, definition):
        """Use regular expressions to remove comments and blank lines"""
        return re.sub(  # First remove comments, then blank lines
            r"\n\s*\n", "\n", re.sub(r"#.*\n", "\n", definition)
        )

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
            logger.warning(
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

        if not restriction:
            restriction = True

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
    def _merge_repr(cls, restriction: str = True) -> dj.expression.Union:
        """Merged view, including null entries for columns unique to one part.

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the merged view

        Returns
        ------
        datajoint.expression.Union
        """

        parts = [
            cls() * p  # join with master to include sec key (i.e., 'source')
            for p in cls._merge_restrict_parts(
                restriction=restriction,
                add_invalid_restrict=False,
                return_empties=False,  # motivated by SpikeSortingOutput.Import
            )
        ]

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
    def _merge_insert(
        cls, rows: list, part_name: str = None, mutual_exclusvity=True, **kwargs
    ) -> None:
        """Insert rows into merge, ensuring db integrity and mutual exclusivity

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
            If entry already exists, mutual exclusivity errors
            If data doesn't exist in part parents, integrity error
        """
        cls._ensure_dependencies_loaded()

        try:
            for r in iter(rows):
                assert isinstance(
                    r, dict
                ), 'Input "rows" must be a list of dictionaries'
        except TypeError:
            raise TypeError('Input "rows" must be a list of dictionaries')

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
            keys = []  # empty to-be-inserted key
            for part in parts:  # check each part
                part_parent = part.parents(as_objects=True)[-1]
                part_name = cls._part_name(part)
                if part_parent & row:  # if row is in part parent
                    if keys and mutual_exclusvity:  # if key from other part
                        raise ValueError(
                            "Mutual Exclusivity Error! Entry exists in more "
                            + f"than one table - Entry: {row}"
                        )

                    keys = (part_parent & row).fetch("KEY")  # get pk
                    if len(keys) > 1:
                        raise ValueError(
                            "Ambiguous entry. Data has mult rows in "
                            + f"{part_name}:\n\tData:{row}\n\t{keys}"
                        )
                    master_pk = {  # make uuid
                        cls()._reserved_pk: dj.hash.key_hash(keys[0]),
                    }
                    parts_entries[part].append({**master_pk, **keys[0]})
                    master_entries.append(
                        {**master_pk, cls()._reserved_sk: part_name}
                    )

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
    def _safe_context(cls):
        """Return transaction if not already in one."""
        return (
            cls.connection.transaction
            if not cls.connection.in_transaction
            else nullcontext()
        )

    @classmethod
    def _ensure_dependencies_loaded(cls) -> None:
        """Ensure connection dependencies loaded.

        Otherwise parts returns none
        """
        if not dj.conn.connection.dependencies._loaded:
            dj.conn.connection.dependencies.load()

    def insert(self, rows: list, mutual_exclusvity=True, **kwargs):
        """Merges table specific insert

        Ensuring db integrity and mutual exclusivity

        Parameters
        ---------
        rows: List[dict]
            An iterable where an element is a dictionary.
        mutual_exclusvity: bool
            Check for mutual exclusivity before insert. Default True.

        Raises
        ------
        TypeError
            If rows is not a list of dicts
        ValueError
            If entry already exists, mutual exclusivity errors
            If data doesn't exist in part parents, integrity error
        """
        self._merge_insert(rows, mutual_exclusvity=mutual_exclusvity, **kwargs)

    @classmethod
    def merge_view(cls, restriction: str = True):
        """Prints merged view, including null entries for unique columns.

        Note: To handle this Union as a table-like object, use `merge_resrict`

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the merged view
        """

        # If we overwrite `preview`, we then encounter issues with operators
        # getting passed a `Union`, which doesn't have a method we can
        # intercept to manage master/parts

        return pprint(cls._merge_repr(restriction=restriction))

    @classmethod
    def merge_html(cls, restriction: str = True):
        """Displays HTML in notebooks."""

        return HTML(repr_html(cls._merge_repr(restriction=restriction)))

    @classmethod
    def merge_restrict(cls, restriction: str = True) -> dj.U:
        """Given a restriction, return a merged view with restriction applied.

        Example
        -------
            >>> MergeTable.merge_restrict("field = 1")

        Parameters
        ----------
        restriction: str
            Restriction one would apply if `merge_view` was a real table.

        Returns
        -------
        datajoint.Union
            Merged view with restriction applied.
        """
        return cls._merge_repr(restriction=restriction)

    @classmethod
    def merge_delete(cls, restriction: str = True, **kwargs):
        """Given a restriction string, delete corresponding entries.

        Parameters
        ----------
        restriction: str
            Optional restriction to apply before deletion from master/part
            tables. If not provided, delete all entries.
        kwargs: dict
            Additional keyword arguments for DataJoint delete.

        Example
        -------
            >>> MergeTable.merge_delete("field = 1")
        """
        uuids = [
            {k: v}
            for entry in cls.merge_restrict(restriction).fetch("KEY")
            for k, v in entry.items()
            if k == cls()._reserved_pk
        ]
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
        part_parents = cls._merge_restrict_parents(
            restriction=restriction, as_objects=True, return_empties=False
        )

        if dry_run:
            return part_parents

        merge_ids = cls.merge_restrict(restriction).fetch(
            RESERVED_PRIMARY_KEY, as_dict=True
        )

        # CB: Removed transaction protection here bc 'no' confirmation resp
        # still resulted in deletes. If re-add, consider transaction=False
        super().delete((cls & merge_ids), **kwargs)

        if cls & merge_ids:  # If 'no' on del prompt from above, skip below
            return  # User can still abort del below, but yes/no is unlikly

        for part_parent in part_parents:
            super().delete(part_parent, **kwargs)

    def fetch_nwb(
        self,
        restriction: str = True,
        multi_source=False,
        disable_warning=False,
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
        """
        if isinstance(self, dict):
            raise ValueError("Try replacing Merge.method with Merge().method")
        if restriction is True and self.restriction:
            if not disable_warning:
                _warn_on_restriction(self, restriction)
            restriction = self.restriction

        return self.merge_restrict_class(restriction).fetch_nwb()

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
        sources = [
            cls._part_name(part)  # friendly part name
            for part in cls._merge_restrict_parts(
                restriction=restriction,
                as_objects=False,
                return_empties=return_empties,
                add_invalid_restrict=False,
            )
        ]

        if not multi_source and len(sources) != 1:
            raise ValueError(
                f"Found {len(sources)} potential parts: {sources}\n\t"
                + "Try adding a restriction before invoking `get_part`.\n\t"
                + "Or permitting multiple sources with `multi_source=True`."
            )

        parts = [
            getattr(cls, source)().restrict(restriction)
            if restrict_part  # Re-apply restriction or don't
            else getattr(cls, source)()
            for source in sources
        ]
        if join_master:
            parts = [cls * part for part in parts]

        return parts if multi_source else parts[0]

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

        part_parents = cls._merge_restrict_parents(
            restriction=restriction,
            as_objects=True,
            return_empties=return_empties,
            add_invalid_restrict=add_invalid_restrict,
        )

        if not multi_source and len(part_parents) != 1:
            __import__("pdb").set_trace()
            raise ValueError(
                f"Found  {len(part_parents)} potential parents: {part_parents}"
                + "\n\tTry adding a string restriction when invoking "
                + "`get_parent`. Or permitting multiple sources with "
                + "`multi_source=True`."
            )

        if join_master:
            part_parents = [cls * part for part in part_parents]

        return part_parents if multi_source else part_parents[0]

    @property
    def source_class_dict(self) -> dict:
        if not self._source_class_dict:
            module = getmodule(self)
            self._source_class_dict = {
                part_name: getattr(module, part_name)
                for part_name in self.parts(camel_case=True)
            }
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
                logger.warn(f"Multiple sources. Selecting first: {source}.")
        if isinstance(source, dj.Table):
            source = self._part_name(source)
        if isinstance(source, dict):
            source = self._part_name(self.merge_get_parent(source))

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

    def merge_restrict_class(self, key: dict) -> dj.Table:
        """Returns native parent class, restricted with key."""
        parent_key = self.merge_get_parent(key).fetch("KEY", as_dict=True)

        if len(parent_key) > 1:
            raise ValueError(
                f"Ambiguous entry. Data has mult rows in parent:\n\tData:{key}"
                + f"\n\t{parent_key}"
            )

        parent_class = self.merge_get_parent_class(key)
        return parent_class & parent_key

    @classmethod
    def merge_fetch(self, restriction: str = True, *attrs, **kwargs) -> list:
        """Perform a fetch across all parts. If >1 result, return as a list.

        Parameters
        ----------
        restriction: str
            Optional restriction to apply before determining parent to return.
            Default True.
        attrs, kwargs
            arguments passed to DataJoint `fetch` call

        Returns
        -------
        Union[ List[np.array], List[dict], List[pd.DataFrame] ]
            Table contents, with type determined by kwargs
        """
        results = []
        parts = self()._merge_restrict_parts(
            restriction=restriction,
            as_objects=True,
            return_empties=False,
            add_invalid_restrict=False,
        )

        for part in parts:
            try:
                results.extend(part.fetch(*attrs, **kwargs))
            except DataJointError as e:
                logger.warn(
                    f"{e.args[0]} Skipping "
                    + to_camel_case(part.table_name.split("__")[-1])
                )

        # Note: this could collapse results like merge_view, but user may call
        # for recarray, pd.DataFrame, or dict, and fetched contents differ if
        # attrs or "KEY" called. Intercept format, merge, and then transform?

        if not results:
            logger.info(
                "No merge_fetch results.\n\t"
                + "If not restricting, try: `M.merge_fetch(True,'attr')\n\t"
                + "If restricting by source, use dict: "
                + "`M.merge_fetch({'source':'X'})"
            )
        return results[0] if len(results) == 1 else results

    @classmethod
    def merge_populate(source: str, key=None):
        raise NotImplementedError(
            "CBroz: In the future, this command will support executing "
            + "part_parent `make` and then inserting all entries into Merge"
        )


_Merge = Merge

# Underscore as class name avoids errors when this included in a Diagram
# Aliased because underscore otherwise excludes from API docs.


def delete_downstream_merge(
    table: dj.Table,
    restriction: str = None,
    dry_run=True,
    recurse_level=2,
    disable_warning=False,
    **kwargs,
) -> list:
    """Given a table/restriction, id or delete relevant downstream merge entries

    Parameters
    ----------
    table: dj.Table
        DataJoint table or restriction thereof
    restriction: str
        Optional restriction to apply before deletion from merge/part
        tables. If not provided, delete all downstream entries.
    dry_run: bool
        Default True. If true, return list of tuples, merge/part tables
        downstream of table input. Otherwise, delete merge/part table entries.
    recurse_level: int
        Default 2. Depth to recurse into table descendants.
    disable_warning: bool
        Default False. If True, don't warn about restrictions on table object.
    kwargs: dict
        Additional keyword arguments for DataJoint delete.

    Returns
    -------
    List[Tuple[dj.Table, dj.Table]]
        Entries in merge/part tables downstream of table input.
    """
    if not disable_warning:
        _warn_on_restriction(table, restriction)

    if not restriction:
        restriction = True

    descendants = _unique_descendants(table, recurse_level)
    merge_table_dicts = _search_descendants(
        table_list=descendants,
        parent_table=table,
        restriction=restriction,
    )

    if dry_run:
        return [d["view"] for d in merge_table_dicts]

    for merge_dict in merge_table_dicts:
        (merge_dict["master"] & merge_dict["keys"]).delete(**kwargs)


def _warn_on_restriction(table: dj.Table, restriction: str = None):
    """Warn if restriction on table object differs from input restriction"""
    if restriction is None and table.restriction:
        logger.warn(
            f"Warning: ignoring table restriction: {table().restriction}.\n\t"
            + "Please pass restrictions as an arg"
        )


def _unique_descendants(
    table: dj.Table,
    recurse_level: int = 2,
    return_names: bool = False,
    attribute=None,
) -> list:
    """Recurisively find unique descendants of a given table

    Parameters
    ----------
    table: dj.Table
        The node in the tree from which to find descendants.
    recurse_level: int
        The maximum level of descendants to find.
    return_names: bool
        If True, return names of descendants found. Else return Table objects.
    attribute: str, optional
        If provided, only return descendants that have this attribute.

    Returns
    -------
    List[dj.Table, str]
        List descendants found when recurisively called to recurse_level
    """

    if recurse_level == 0:
        return {}

    skip_attr_check = True if attribute is None else False

    descendants = {}

    def recurse_descendants(sub_table, level):
        for descendant in sub_table.descendants(as_objects=True):
            if descendant.full_table_name not in descendants and (
                skip_attr_check or attribute in descendant.heading.attributes
            ):
                descendants[descendant.full_table_name] = descendant
                if level > 1:
                    recurse_descendants(descendant, level - 1)

    recurse_descendants(table, recurse_level)

    return (
        list(descendants.keys()) if return_names else list(descendants.values())
    )


def _search_descendants(
    table_list: list,
    parent_table: dj.Table,
    restriction: str = True,
    connection: dj.connection.Connection = None,
) -> list:
    """
    Given list of descendant tables, find merge tables linked to parent table.

    Returns a list of tuples, with master and part. Part will have restriction
    applied. If restriction yield empty list, skip.

    Parameters
    ----------
    table_list : List[dj.Table]
        A list of datajoint tables.
    parent_table : dj.Table
        The parent table of the tables in table_list.
    restiction : str, optional
        Restriction applies to parent table. Default True, no restriction.
    connection : datajoint.connection.Connection
        A database connection. Default None, use connection from first table.

    Returns
    -------
    List[dict]
        A list of dictionaries, each with keys: master, view, and keys.
        These are the master table, the restricted view of the master table,
        and the primary keys of the restricted view.
    """
    conn = connection or table_list[0].connection

    ret = []
    unique_parts = []

    # Adapted from Spyglass PR 535
    for table in table_list:
        table_name = table.full_table_name
        if table_name in unique_parts:  # then repeat in list
            continue

        master_name = get_master(table_name)
        if not master_name:  # then not a part table
            continue

        master = dj.FreeTable(conn, master_name)
        # if table has no master, or master is not MergeTable, skip
        if RESERVED_PRIMARY_KEY not in master.heading.names:
            continue

        try:
            restricted_master = (
                master & restriction
            )  # .merge_restrict(restriction)
        except DataJointError:
            continue

        if not restricted_master:  # No entries relevant to restriction in part
            continue

        unique_parts.append(table_name)
        ret.append(
            dict(
                master=master,
                view=restricted_master,
                keys=restricted_master.fetch(
                    RESERVED_PRIMARY_KEY, as_dict=True
                ),
            )
        )

    return ret
