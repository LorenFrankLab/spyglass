from contextlib import nullcontext
from itertools import chain as iter_chain
from pprint import pprint

import datajoint as dj
from datajoint.condition import make_condition
from datajoint.errors import DataJointError
from datajoint.preview import repr_html
from datajoint.utils import from_camel_case, to_camel_case
from IPython.core.display import HTML

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb

RESERVED_PRIMARY_KEY = "merge_id"
RESERVED_SECONDARY_KEY = "source"
RESERVED_SK_LENGTH = 32


class Merge(dj.Manual):
    """Adds funcs to support insert/delete/view of Merge tables."""

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
                    "WARNING: merge table declared with non-default definition\n\t"
                    + f"Expected: {merge_def.strip()}\n\t"
                    + f"Actual  : {self.definition.strip()}"
                )
            for part in self.parts(as_objects=True):
                if part.primary_key != self.primary_key:
                    print(
                        f"WARNING: unexpected primary key for {part.table_name}\n\t"
                        + f"Expected: {self.primary_key}\n\t"
                        + f"Actual  : {part.primary_key}"
                    )

    @classmethod
    def _merge_restrict_parts(
        cls,
        restriction: str = True,
        as_objects: bool = True,
        return_empties: bool = True,
    ) -> list:
        """Returns a list of parts with restrictions applied.

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the merged view. Default True, no restrictions.
        as_objects: bool, optional
            Default True. Return part tables as objects
        return_empties: bool, optional
            Default True. Return empty part tables

        Returns
        ------
        list
            list of datajoint tables, parts of Merge Table
        """
        if not dj.conn.connection.dependencies._loaded:
            dj.conn.connection.dependencies.load()  # Otherwise parts returns none

        if not restriction:
            restriction = True

        # Normalize restriction to sql string
        restriction = make_condition(cls(), restriction, set())

        parts_all = cls.parts(as_objects=True)
        # If the restriction makes ref to a source, we only want that part
        if isinstance(restriction, str) and cls()._reserved_sk in restriction:
            parts_all = [
                part
                for part in parts_all
                if from_camel_case(
                    restriction.split(f'`{cls()._reserved_sk}`="')[-1].split(
                        '"'
                    )[0]
                )  # Only look at source part table
                in part.full_table_name
            ]

        parts = []
        for part in parts_all:
            try:
                parts.append(part.restrict(restriction))
            except DataJointError:  # If restriction not valid on given part
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
        as_objects: bool = True,
        return_empties: bool = True,
    ) -> list:
        """Returns a list of part parents with restrictions applied.

        Rather than part tables, we look at parents of those parts, the source
        of the data.

        Parameters
        ---------
        restriction: str, optional
            Restriction to apply to the merged view. Default True, no restrictions.
        as_objects: bool, optional
            Default True. Return part tables as objects
        return_empties: bool, optional
            Default True. Return empty part tables

        Returns
        ------
        list
            list of datajoint tables, parents of parts of Merge Table
        """
        part_parents = [
            parent  # Below, restricting parent to info from restricted part
            & part.fetch(*part.heading.secondary_attributes, as_dict=True)
            for part in cls()._merge_restrict_parts(
                restriction=restriction, return_empties=return_empties
            )
            for parent in part.parents(as_objects=True)  # ID respective parents
            if cls().table_name not in parent.full_table_name  # Not merge table
        ]
        if not as_objects:
            part_parents = [p.full_table_name for p in part_parents]

        return part_parents

    @classmethod
    def _merge_repr(
        cls, restriction: str = True, **kwargs
    ) -> dj.expression.Union:
        """Merged view, including null entries for columns unique to one part table.

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
            for p in cls._merge_restrict_parts(restriction=restriction)
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
    def _merge_insert(cls, rows: list, **kwargs) -> None:
        """Insert rows into merge table, ensuring db integrity and mutual exclusivity

        Parameters
        ---------
        rows: List[dict]
            An iterable where an element is a dictionary.

        Raises
        ------
        TypeError
            If rows is not a list of dicts
        ValueError
            If entry already exists, mutual exclusivity errors
            If data doesn't exist in part parents, integrity error
        """

        try:
            for r in iter(rows):
                assert isinstance(
                    r, dict
                ), 'Input "rows" must be a list of dictionaries'
        except TypeError:
            raise TypeError('Input "rows" must be a list of dictionaries')

        parts = cls._merge_restrict_parts(as_objects=True)
        master_entries = []
        parts_entries = {p: [] for p in parts}
        for row in rows:
            key = {}
            for part in parts:
                master = part.parents(as_objects=True)[-1]
                part_name = to_camel_case(part.table_name.split("__")[-1])
                if master & row:
                    if not key:
                        key = (master & row).fetch1("KEY")
                        master_pk = {
                            cls()._reserved_pk: dj.hash.key_hash(key),
                        }
                        parts_entries[part].append({**master_pk, **key})
                        master_entries.append(
                            {**master_pk, cls()._reserved_sk: part_name}
                        )
                    else:
                        raise ValueError(
                            "Mutual Exclusivity Error! Entry exists in more "
                            + f"than one table - Entry: {row}"
                        )

            if not key:
                raise ValueError(
                    "Non-existing entry in any of the parent tables - Entry: "
                    + f"{row}"
                )

        # 1. nullcontext() allows use within `make` but decreases reliability
        # 2. cls.connection.transaction is more relaiable but throws errors if
        # used within another transaction, i.e. in `make`

        with nullcontext():  # TODO: ensure this block within transaction
            super().insert(cls(), master_entries, **kwargs)
            for part, part_entries in parts_entries.items():
                part.insert(part_entries, **kwargs)

    @classmethod
    def insert(cls, rows: list, **kwargs):
        """Insert rows into merge table

        Ensuring db integrity and mutual exclusivity

        Parameters
        ---------
        rows: List[dict]
            An iterable where an element is a dictionary.

        Raises
        ------
        TypeError
            If rows is not a list of dicts
        ValueError
            If entry already exists, mutual exclusivity errors
            If data doesn't exist in part parents, integrity error
        """
        cls._merge_insert(rows, **kwargs)

    @classmethod
    def merge_view(cls, restriction: str = True):
        """Prints merged view, including null entries for unique columns.

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
        """Given a restriction string, return a merged view with restriction applied.

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
        """Delete enties from merge master, part, and respective part parents

        Note: Clears merge entries from their respective parents.

        Parameters
        ----------
        restriction: str
            Optional restriction to apply before deletion from parents. If not
            provided, delete all entries present in Merge Table.
        dry_run: bool
            Default True. If true, return list of tables with entries that would be
            deleted. Otherwise, table entries.
        kwargs: dict
            Additional keyword arguments for DataJoint delete.
        """

        part_parents = cls._merge_restrict_parents(
            restriction=restriction, as_objects=True, return_empties=False
        )

        if dry_run:
            return part_parents

        super().delete(cls(), **kwargs)
        for part_parent in part_parents:
            super().delete(part_parent, **kwargs)

    def fetch_nwb(self, *attrs, **kwargs):
        part_parents = self._merge_restrict_parents(
            restriction=self.restriction, return_empties=False
        )

        if len(part_parents) == 1:
            return fetch_nwb(
                part_parents[0],
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )
        else:
            raise ValueError(
                f"{len(part_parents)} possible sources found in Merge Table"
                + part_parents
            )

    def get_part_table(self) -> dj.Table:
        """Hacky way to retrieve the part table from a merge table with a single entry"""
        return getattr(self, self.fetch1("source")) & self


def delete_downstream_merge(
    table: dj.Table, restriction: str = True, dry_run=True, **kwargs
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
    kwargs: dict
        Additional keyword arguments for DataJoint delete.

    Returns
    -------
    List[Tuple[dj.Table, dj.Table]]
        Entries in merge/part tables downstream of table input.
    """
    if not restriction:
        restriction = True

    # Adapted from Spyglass PR 535
    # dj.utils.get_master could maybe help here, but it uses names, not objs
    merge_pairs = [  # get each merge/part table
        (master, descendant.restrict(restriction))
        for descendant in table.descendants(as_objects=True)  # given tbl desc
        for master in descendant.parents(as_objects=True)  # and those parents
        # if is a part table (using a dunder not immediately after schema name)
        if "__" in descendant.full_table_name.replace("`.`__", "")
        # and it is not in not in direct descendants
        and master.full_table_name not in table.descendants(as_objects=False)
        # and it uses our reserved primary key in attributes
        and RESERVED_PRIMARY_KEY in master.heading.attributes.keys()
    ]

    # restrict the merge table based on uuids in part
    merge_pairs = [
        (merge & uuids, part)  # don't need part for del, but show on dry_run
        for merge, part in merge_pairs
        for uuids in part.fetch(RESERVED_PRIMARY_KEY, as_dict=True)
    ]

    if dry_run:
        return merge_pairs

    for merge_table, _ in merge_pairs:
        merge_table.delete(**kwargs)
