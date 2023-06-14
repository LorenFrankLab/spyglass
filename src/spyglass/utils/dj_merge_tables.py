from itertools import chain as iter_chain
from pprint import pprint

import datajoint as dj
from datajoint.errors import DataJointError
from datajoint.preview import repr_html


class Merge(dj.Manual):
    """Adds funcs to support insert/delete/view of Merge tables."""

    def __init__(self):
        super().__init__()
        self._reserved_pk = "merge_id"  # reserved primary key
        merge_def = f"\n    {self._reserved_pk}: uuid\n    "
        # TODO: Change warnings to logger. Throw error?
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
        parts = []
        for part in cls.parts(as_objects=True):
            try:
                parts.append(part.restrict(restriction))
            except DataJointError:  # If restriction not valid on given part
                parts.append(part)
        primary_attrs = list(
            dict.fromkeys(  # get all columns from parts
                iter_chain.from_iterable([p.heading.names for p in parts])
            )
        )
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

        parts = cls.parts(as_objects=True)
        master_entries = []
        parts_entries = {p: [] for p in parts}
        for row in rows:
            key = {}
            for part in parts:
                master = part.parents(as_objects=True)[-1]
                if master & row:
                    if not key:
                        key = (master & row).fetch1("KEY")
                        master_key = {cls.primary_key[0]: dj.hash.key_hash(key)}
                        parts_entries[part].append({**master_key, **key})
                        master_entries.append(master_key)
                    else:
                        raise ValueError(
                            "Mutual Exclusivity Error! Entry exists in more than one "
                            + f"table - Entry: {row}"
                        )

            if not key:
                raise ValueError(
                    f"Non-existing entry in any of the parent tables - Entry: {row}"
                )

        with cls.connection.transaction:
            super().insert(cls(), master_entries, **kwargs)
            for part, part_entries in parts_entries.items():
                part.insert(part_entries, **kwargs)

    @classmethod
    def insert(cls, rows, **kwargs):
        """Insert rows into merge table, ensuring db integrity and mutual exclusivity"""
        cls._merge_insert(rows, **kwargs)

    @classmethod
    def merge_view(cls, limit=None, width=None):
        """Merged view, including null entries for columns unique to one part table."""

        # If we overwrite `preview`, we then encounter issues with operators
        # getting passed a `Union`, which doesn't have a method we can
        # intercept to manage master/parts

        return pprint(cls._merge_repr())

    @classmethod
    def merge_html(cls):
        """Returns HTML to display table in Jupyter notebook."""
        print("WARNING: This method is untested")
        return repr_html(cls._merge_repr())

    @classmethod
    def merge_restrict(cls, restriction):
        """Given a restriction string, return a merged view with the restrict'n applied.

        Example
        -------
            >>> MergeTable.merge_restrict("field = 1")
        """
        return cls._merge_repr(restriction=restriction)

    @classmethod
    def merge_delete_parent(cls, **kwargs) -> None:
        """Delete enties from merge master, part, and respective part parents"""
        part_parents = [
            parent
            for part in cls().parts(as_objects=True)  # For each part table
            for parent in part.parents(as_objects=True)  # ID respective parents
            if cls().table_name not in parent.full_table_name  # Not merge table
        ]
        super().delete(cls(), **kwargs)
        for part_parent in part_parents:
            super().delete(part_parent, **kwargs)

    @classmethod
    def merge_delete(cls, restriction: str = True, **kwargs):
        """Given a restriction string, delete corresponding entries.

        Example
        -------
            >>> MergeTable.merge_delete("field = 1")
        """
        uuids = [
            {k: v}
            for entry in cls.merge_restrict(restriction).fetch("KEY")
            for k, v in entry.items()
            if k == cls._reserved_pk
        ]
        (cls() & uuids).delete(**kwargs)

    # TODO: test fetch nwb


def delete_downstream_merge(table: dj.Table, **kwargs) -> None:
    """Given a table (or restriction), delete the relevant downstream merge entries"""
    # Adapted from Spyglass PR 535
    merge_pairs = [
        (master, descendant)  # get each merge/part table
        for descendant in table.descendants(as_objects=True)  # given tbl desc
        for master in descendant.parents(as_objects=True)  # and those parents
        # if it is a part table (using a dunder not immediately after schema name)
        if "__" in descendant.full_table_name.replace("`.`__", "")
        # and it is not in not in direct descendants
        and master.full_table_name not in table.descendants(as_objects=False)
    ]

    for merge_table, part_table in merge_pairs:
        keys = ((merge_table * part_table) & table.fetch()).fetch("KEY")
        for entry in keys:
            (merge_table & entry).delete(**kwargs)
