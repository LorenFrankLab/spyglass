from contextlib import nullcontext
from itertools import chain as iter_chain
from pprint import pprint

import datajoint as dj
from datajoint.condition import make_condition
from datajoint.errors import DataJointError
from datajoint.preview import repr_html
from datajoint.utils import from_camel_case, get_master, to_camel_case
from IPython.core.display import HTML

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb

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
        # TODO: Change warnings to logger. Throw error? - CBroz1
        if not self.is_declared:
            if self.definition != merge_def:
                print(
                    "WARNING: merge table with non-default definition\n\t"
                    + f"Expected: {merge_def.strip()}\n\t"
                    + f"Actual  : {self.definition.strip()}"
                )
            for part in self.parts(as_objects=True):
                if part.primary_key != self.primary_key:
                    print(
                        f"WARNING: unexpected primary key in {part.table_name}"
                        + f"\n\tExpected: {self.primary_key}"
                        + f"\n\tActual  : {part.primary_key}"
                    )

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
                return_empties=False,
            )
        ]

        primary_attrs = list(
            dict.fromkeys(  # get all columns from parts
                iter_chain.from_iterable([p.heading.names for p in parts])
            )
        )
        # primary_attrs.append(cls()._reserved_sk)
        query = dj.U(*primary_attrs) * parts[0].proj(  # declare query
            ...,  # include all attributes from part 0
            **{
                a: "NULL"  # add null value where part has no column
                for a in primary_attrs
                if a not in parts[0].heading.names
            },
        )
        for part in parts[1:]:  # add to declared query for each part
            query += dj.U(*primary_attrs) * part.proj(
                ...,
                **{
                    a: "NULL"
                    for a in primary_attrs
                    if a not in part.heading.names
                },
            )
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
                part_name = to_camel_case(part.table_name.split("__")[-1])
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
            super().delete(part_parent, **kwargs)  # add safemode=False?

    @classmethod
    def fetch_nwb(
        cls, restriction: str = True, multi_source=False, *attrs, **kwargs
    ):
        """Return the AnalysisNwbfile file linked in the source.

        Parameters
        ----------
        restriction: str, optional
            Restriction to apply to parents before running fetch. Default none.
        multi_source: bool
            Return from multiple parents. Default False.
        """
        part_parents = cls._merge_restrict_parents(
            restriction=restriction,
            return_empties=False,
            add_invalid_restrict=False,
        )

        if not multi_source and len(part_parents) != 1:
            raise ValueError(
                f"{len(part_parents)} possible sources found in Merge Table:"
                + " and ".join([p.full_table_name for p in part_parents])
            )

        nwbs = []
        for part_parent in part_parents:
            nwbs.extend(
                fetch_nwb(
                    part_parent,
                    (AnalysisNwbfile, "analysis_file_abs_path"),
                    *attrs,
                    **kwargs,
                )
            )
        return nwbs

    @classmethod
    def merge_get_part(
        cls,
        restriction: str = True,
        join_master: bool = False,
        restrict_part=True,
        multi_source=False,
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
            to_camel_case(n.split("__")[-1].strip("`"))  # friendly part name
            for n in cls._merge_restrict_parts(
                restriction=restriction,
                as_objects=False,
                return_empties=False,
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
        multi_source=False,
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

        Returns
        ------
        dj.FreeTable
            Parent of parts of Merge Table as FreeTable.
        """

        part_parents = cls._merge_restrict_parents(
            restriction=restriction,
            as_objects=True,
            return_empties=False,
            add_invalid_restrict=False,
        )

        if not multi_source and len(part_parents) != 1:
            raise ValueError(
                f"Found  {len(part_parents)} potential parents: {part_parents}"
                + "\n\tTry adding a string restriction when invoking "
                + "`get_parent`. Or permitting multiple sources with "
                + "`multi_source=True`."
            )

        if join_master:
            part_parents = [cls * part for part in part_parents]

        return part_parents if multi_source else part_parents[0]

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
                print(
                    f"WARNING: {e.args[0]} Skipping "
                    + to_camel_case(part.table_name.split("__")[-1])
                )

        # Note: this could collapse results like merge_view, but user may call
        # for recarray, pd.DataFrame, or dict, and fetched contents differ if
        # attrs or "KEY" called. Intercept format, merge, and then transform?

        if not results:
            print(
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
    restriction: str = True,
    dry_run=True,
    recurse_level=2,
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
    kwargs: dict
        Additional keyword arguments for DataJoint delete.

    Returns
    -------
    List[Tuple[dj.Table, dj.Table]]
        Entries in merge/part tables downstream of table input.
    """
    if table.restriction:
        print(
            f"Warning: ignoring table restriction: {table.restriction}.\n\t"
            + "Please pass restrictions as an arg"
        )

    descendants = _unique_descendants(table, recurse_level)
    merge_table_pairs = _master_table_pairs(descendants, restriction)

    # restrict the merge table based on uuids in part
    merge_pairs = [
        (merge & uuids, part)  # don't need part for del, but show on dry_run
        for merge, part in merge_table_pairs
        for uuids in part.fetch(RESERVED_PRIMARY_KEY, as_dict=True)
    ]

    if dry_run:
        return merge_pairs

    for merge_table, _ in merge_pairs:
        merge_table.delete(**kwargs)


def _unique_descendants(
    table: dj.Table, recurse_level: int, return_names: bool = False
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

    Returns
    -------
    List[dj.Table, str]
        List descendants found when recurisively called to recurse_level
    """

    if recurse_level == 0:
        return []

    descendants = {}

    def recurse_descendants(sub_table, level):
        for descendant in sub_table.descendants(as_objects=True):
            if descendant.full_table_name not in descendants:
                descendants[descendant.full_table_name] = descendant
                if level > 1:
                    recurse_descendants(descendant, level - 1)

    recurse_descendants(table, recurse_level)

    return (
        list(descendants.keys()) if return_names else list(descendants.values())
    )


def _master_table_pairs(
    table_list: list,
    restriction: str = True,
    connection: dj.connection.Connection = None,
) -> list:
    """
    Given list of tables, return a list of master table pairs.

    Returns a list of tuples, with master and part. Part will have restriction
    applied. If restriction yield empty list, skip.

    Parameters
    ----------
    table_list : List[dj.Table]
        A list of datajoint tables.
    restriction : str
        A restriction string. Default True, no restriction.
    connection : datajoint.connection.Connection
        A database connection. Default None, use connection from first table.

    Returns
    -------
    List[Tuple[dj.Table, dj.Table]]
        A list of master table pairs.
    """
    conn = connection or table_list[0].connection

    master_table_pairs = []
    # Adapted from Spyglass PR 535
    for table in table_list:
        master_name = get_master(table.full_table_name)
        if not master_name:  # then it's not a part table
            continue

        master = dj.FreeTable(conn, master_name)

        if RESERVED_PRIMARY_KEY not in master.heading.attributes.keys():
            continue

        restricted_table = table.restrict(restriction)

        if not restricted_table:
            continue

        master_table_pairs.append((master, restricted_table))

    return master_table_pairs
