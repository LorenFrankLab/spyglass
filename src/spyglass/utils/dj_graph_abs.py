from abc import ABC, abstractmethod
from itertools import chain as iter_chain
from typing import Dict, List, Tuple, Union

from datajoint import FreeTable, logger
from datajoint.condition import make_condition
from datajoint.dependencies import unite_master_parts
from datajoint.table import Table
from datajoint.utils import to_camel_case
from networkx import NetworkXNoPath, all_simple_paths, shortest_path
from networkx.algorithms.dag import topological_sort

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

    def _get_edge(self, child, parent) -> Tuple[bool, Dict[str, str]]:
        """Get edge data between child and parent.

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            Tuple of boolean indicating direction and edge data. True if child
            is child of parent.
        """
        child = child if isinstance(child, str) else child.full_table_name
        parent = parent if isinstance(parent, str) else parent.full_table_name

        if edge := self.graph.get_edge_data(parent, child):
            return False, edge
        elif edge := self.graph.get_edge_data(child, parent):
            return True, edge

        # Handle alias nodes. `shortest_path` doesn't work with aliases
        p1 = all_simple_paths(self.graph, child, parent)
        p2 = all_simple_paths(self.graph, parent, child)
        paths = [p for p in iter_chain(p1, p2)]  # list for error handling
        for path in paths:
            if len(path) > 3 or (len(path) > 2 and not path[1].isnumeric()):
                continue
            return self._get_edge(path[0], path[1])

        raise ValueError(f"{child} -> {parent} not direct path: {paths}")

    def _rev_attrs(self, attr_map):
        """Parse attribute map. Remove self-references."""
        return {v: k for k, v in attr_map.items()}

    def _get_restr(self, table):
        """Get restriction from graph node.

        Defaults to False if no restriction is set so that it doesn't appear
        in attrs like `all_ft`.
        """
        table = table if isinstance(table, str) else table.full_table_name
        return self._get_node(table).get("restr")

    def _set_restr(self, table, restriction, replace=False):
        """Add restriction to graph node. If one exists, merge with new."""
        ft = self._get_ft(table)
        restriction = (  # Convert to condition if list or dict
            make_condition(ft, restriction, set())
            if not isinstance(restriction, str)
            else restriction
        )
        existing = self._get_restr(table)
        if not replace and existing:
            if restriction == existing:
                return
            join = ft & [existing, restriction]
            if len(join) == len(ft & existing):
                return  # restriction is a subset of existing
            restriction = make_condition(
                ft, unique_dicts(join.fetch("KEY", as_dict=True)), set()
            )

        self._log_truncate(f"Set {table.split('.')[-1]} {restriction}")
        # if "#pk_node" in table:
        #     __import__("pdb").set_trace()
        self._set_node(table, "restr", restriction)

    def _get_ft(self, table, with_restr=False):
        """Get FreeTable from graph node. If one doesn't exist, create it."""
        table = table if isinstance(table, str) else table.full_table_name

        if with_restr:
            restr = self._get_restr(table)
            if not restr:
                logger.warning(f"No restriction for {table}")
                restr = False
        else:
            restr = True

        if ft := self._get_node(table).get("ft"):
            return ft & restr
        ft = FreeTable(self.connection, table)
        self._set_node(table, "ft", ft)
        return ft & restr

    def topological_sort(self, nodes=None) -> List[str]:
        """Get topological sort of visited nodes. From datajoint.diagram"""
        nodes = nodes or self.visited
        nodes = [n for n in nodes if not n.isnumeric()]
        return unite_master_parts(
            list(topological_sort(self.graph.subgraph(nodes)))
        )

    @property
    def all_ft(self):
        """Get restricted FreeTables from all visited nodes."""
        self.cascade()
        return [
            self._get_ft(table, with_restr=True)
            for table in self.topological_sort()
        ]

    def _print_restr(self, leaves=False):
        """Print restrictions for each table in visited set."""
        mylist = self.leaves if leaves else self.visited
        for table in mylist:
            self._log_truncate(
                f"{table.split('.')[-1]:>35} {self._get_restr(table)}"
            )

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
        direction: str = None,
        attr_map: dict = None,
        primary: bool = None,
        aliased: bool = None,
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
        # Direction UP: table1 -> table2, parent -> child
        if not all([direction, attr_map, primary, aliased]):
            dir_bool, edge = self._get_edge(table1, table2)
            direction = "up" if dir_bool else "down"
            attr_map = edge.get("attr_map")
            primary = edge.get("primary")
            aliased = edge.get("aliased")

        ft1 = self._get_ft(table1)
        rt1 = ft1 & restr
        ft2 = self._get_ft(table2)

        if len(ft1) == 0:
            return ["False"]

        adjust = bool(set(attr_map.values()) - set(ft1.heading.names))
        if adjust:
            attr_map = self._rev_attrs(attr_map)

        join = rt1.proj(**attr_map) * ft2

        ret = unique_dicts(join.fetch(*ft2.primary_key, as_dict=True))

        null = None
        if self.verbose:
            dir = "Up" if direction == "up" else "Dn"
            prim = "Pri" if primary else "Sec"
            adjust = "Flip" if adjust else "NoFp"
            aliaa = "Alias" if aliased else "NoAli"
            null = (
                "NULL"
                if len(ret) == 0
                else "FULL" if len(ft2) == len(ret) else "part"
            )
            strt = f"{to_camel_case(ft1.table_name)}"
            endp = f"{to_camel_case(ft2.table_name)}"
            self._log_truncate(
                f"{dir} {prim} {aliaa} {adjust}: {null} {strt} -> {endp}"
            )
        if null and null != "part":
            pass
        # __import__("pdb").set_trace()

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
                direction=direction,
                **data,
            )

            self.cascade1(
                table=next_table,
                restriction=next_restr,
                direction=direction,
                replace=replace,
            )
