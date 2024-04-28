from abc import ABC, abstractmethod
from typing import Dict, List, Union

from datajoint import FreeTable, logger
from datajoint.condition import make_condition
from datajoint.table import Table
from networkx import NetworkXNoPath, shortest_path

from spyglass.utils.dj_helper_fn import unique_dicts


class AbstractGraph(ABC):
    def __init__(self, seed_table: Table, verbose: bool = False, **kwargs):
        """Abstract class for graph traversal and restriction application.

        Parameters
        ----------
        seed_table : Table
            Table to use to establish connection and graph
        verbose : bool, optional
            Whether to print verbose output. Default False
        """
        self.connection = seed_table.connection
        self.graph = seed_table.connection.dependencies
        self.graph.load()

        self.verbose = verbose
        self.leaves = set()
        self.visited = set()
        self.to_visit = set()
        self.no_visit = set()
        self.cascaded = False

    def _log_truncate(self, log_str, max_len=80):
        """Truncate log lines to max_len and print if verbose."""
        if not self.verbose:
            return
        logger.info(
            log_str[:max_len] + "..." if len(log_str) > max_len else log_str
        )

    @abstractmethod
    def cascade(self):
        """Cascade restrictions through graph."""
        raise NotImplementedError("Child class mut implement `cascade` method")

    def _get_node(self, table):
        """Get node from graph."""
        if not isinstance(table, str):
            table = table.full_table_name
        if not (node := self.graph.nodes.get(table)):
            raise ValueError(
                f"Table {table} not found in graph."
                + "\n\tPlease import this table and rerun"
            )
        return node

    def _set_node(self, table, attr="ft", value=None):
        """Set attribute on node. General helper for various attributes."""
        _ = self._get_node(table)  # Ensure node exists
        self.graph.nodes[table][attr] = value

    def _get_attr_dict(
        self, attr, default_factory=lambda: None
    ) -> List[Dict[str, str]]:
        """Get given attr for each table in self.visited

        Uses factory to create default value for missing attributes.
        """
        return {
            t: self._get_node(t).get(attr, default_factory())
            for t in self.visited
        }

    def _get_attr_map_btwn(self, child, parent):
        """Get attribute map between child and parent.

        Currently used for debugging."""
        child = child if isinstance(child, str) else child.full_table_name
        parent = parent if isinstance(parent, str) else parent.full_table_name

        reverse = False
        try:
            path = shortest_path(self.graph, child, parent)
        except NetworkXNoPath:
            reverse, child, parent = True, parent, child
            path = shortest_path(self.graph, child, parent)

        if len(path) != 2 and not path[1].isnumeric():
            raise ValueError(f"{child} -> {parent} not direct path: {path}")

        try:
            attr_map = self.graph[child][path[1]]["attr_map"]
        except KeyError:
            attr_map = self.graph[path[1]][child]["attr_map"]

        return self._parse_attr_map(attr_map, reverse=reverse)

    def _parse_attr_map(self, attr_map, reverse=False):
        """Parse attribute map. Remove self-references."""
        if not attr_map:
            return {}
        if reverse:
            return {v: k for k, v in attr_map.items() if k != v}
        return {k: v for k, v in attr_map.items() if k != v}

    def _get_restr(self, table):
        """Get restriction from graph node.

        Defaults to False if no restriction is set so that it doesn't appear
        in attrs like `all_ft`.
        """
        table = table if isinstance(table, str) else table.full_table_name
        return self._get_node(table).get("restr", "False")

    def _set_restr(self, table, restriction, replace=False):
        """Add restriction to graph node. If one exists, merge with new."""
        ft = self._get_ft(table)
        restriction = (  # Convert to condition if list or dict
            make_condition(ft, restriction, set())
            if not isinstance(restriction, str)
            else restriction
        )
        existing = self._get_restr(table)
        if not replace and existing != "False":  # False is default
            if existing == restriction:
                return
            join = ft & [existing, restriction]
            if len(join) == len(ft & existing):
                return  # restriction is a subset of existing
            restriction = make_condition(
                ft, unique_dicts(join.fetch("KEY", as_dict=True)), set()
            )

        self._set_node(table, "restr", restriction)

    def _get_ft(self, table, with_restr=False):
        """Get FreeTable from graph node. If one doesn't exist, create it."""
        table = table if isinstance(table, str) else table.full_table_name
        restr = self._get_restr(table) if with_restr else True
        if ft := self._get_node(table).get("ft"):
            return ft & restr
        ft = FreeTable(self.connection, table)
        self._set_node(table, "ft", ft)
        return ft & restr

    @property
    def all_ft(self):
        """Get restricted FreeTables from all visited nodes."""
        self.cascade()
        return [
            self._get_ft(table, with_restr=True)
            for table in self.visited
            if not table.isnumeric()
        ]

    def get_restr_ft(self, table: Union[int, str]):
        """Get restricted FreeTable from graph node.

        Currently used for testing.

        Parameters
        ----------
        table : Union[int, str]
            Table name or index in visited set
        """
        if isinstance(table, int):
            table = list(self.visited)[table]
        return self._get_ft(table, with_restr=True)

    @property
    def as_dict(self) -> List[Dict[str, str]]:
        """Return as a list of dictionaries of table_name: restriction"""
        self.cascade()
        return [
            {"table_name": table, "restriction": self._get_restr(table)}
            for table in self.visited
            if self._get_restr(table)
        ]

    def _bridge_restr(
        self,
        table1: str,
        table2: str,
        restr: str,
        attr_map: dict = None,
        primary: bool = True,
        **kwargs,
    ):
        """Given two tables and a restriction, return restriction for table2.

        Similar to ((table1 & restr) * table2).fetch(*table2.primary_key)
        but with the ability to resolve aliases across tables. One table should
        be the parent of the other. Replaces previous _child_to_parent.

        Parameters
        ----------
        table1 : str
            Table name. Restriction always applied to this table.
        table2 : str
            Table name. Restriction pulled from this table.
        restr : str
            Restriction to apply to table1.
        attr_map : dict, optional
            dictionary mapping aliases across tables, as pulled from
            DataJoint-assembled graph. Default None.
        primary : bool, optional
            Is parent in child's primary key? Default True. Also derived from
            DataJoint-assembled graph. If True, project only primary key fields
            to avoid secondary key collisions.

        Returns
        -------
        List[Dict[str, str]]
            List of dicts containing primary key fields for restricted table2.
        """
        ft1 = self._get_ft(table1) & restr
        ft2 = self._get_ft(table2)

        if len(ft1) == 0:
            return ["False"]

        attr_map = self._parse_attr_map(attr_map)

        if table1 in ft2.parents():
            flip = False
            child, parent = ft2, ft1
        else:  # table2 in ft1.children()
            flip = True
            child, parent = ft1, ft2

        if primary:
            join = (parent.proj(**attr_map) * child).proj()
        else:
            join = (parent.proj(..., **attr_map) * child).proj()

        if set(ft2.primary_key).isdisjoint(set(join.heading.names)):
            join = join.proj(**self._parse_attr_map(attr_map, reverse=True))

        ret = unique_dicts(join.fetch(*ft2.primary_key, as_dict=True))

        if self.verbose and len(ft2) and len(ret) == len(ft2):
            self._log_truncate(f"NULL restr {table2}")
        if self.verbose and attr_map:
            self._log_truncate(f"attr_map {table1} -> {table2}: {flip}")

        return ret

    def cascade1(self, table, restriction, direction="up", replace=False):
        """Cascade a restriction up the graph, recursively on parents.

        Parameters
        ----------
        table : str
            table name
        restriction : str
            restriction to apply
        """

        self._set_restr(table, restriction, replace=replace)
        self.visited.add(table)

        next_func = (
            self.graph.parents if direction == "up" else self.graph.children
        )

        for next_table, data in next_func(table).items():
            if next_table.isnumeric():
                next_table, data = next_func(next_table).popitem()

            if (
                next_table in self.visited
                or next_table in self.no_visit
                or table == next_table
            ):
                continue

            next_restr = self._bridge_restr(
                table1=table,
                table2=next_table,
                restr=restriction,
                **data,
            )

            self.cascade1(
                table=next_table,
                restriction=next_restr,
                direction=direction,
                replace=replace,
            )
