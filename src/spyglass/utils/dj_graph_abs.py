from abc import ABC, abstractmethod
from typing import Dict, List, Union

from datajoint import FreeTable, logger
from datajoint.condition import make_condition
from datajoint.table import Table
from networkx import NetworkXNoPath, NodeNotFound, shortest_path


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
        """Get attribute map between child and parent."""
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
        return attr_map if not reverse else {v: k for k, v in attr_map.items()}

    def _get_restr(self, table):
        """Get restriction from graph node."""
        table = table if isinstance(table, str) else table.full_table_name
        return self._get_node(table).get("restr", "False")

    def _set_restr(self, table, restriction):
        """Add restriction to graph node. If one exists, merge with new."""
        ft = self._get_ft(table)
        restriction = (  # Convert to condition if list or dict
            make_condition(ft, restriction, set())
            if not isinstance(restriction, str)
            else restriction
        )
        if existing := self._get_restr(table):
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
        restr1: str,
        attr_map: dict = None,
        primary: bool = True,
    ):
        ft1 = self._get_ft(table1)
        ft2 = self._get_ft(table2)
        if attr_map is None:
            attr_map = self._get_attr_map_btwn(table1, table2)

        if table1 in ft2.children():
            table1, table2 = table2, table1
            ft1, ft2 = ft2, ft1

        if primary:
            join = ft1.proj(**attr_map) * ft2
        else:
            join = ft1.proj(..., **attr_map) * ft2

        return unique_dicts(join.fetch(*ft2.primary_key, as_dict=True))

    def _child_to_parent(
        self,
        child,
        parent,
        restriction,
        attr_map=None,
        primary=True,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Given a child, child's restr, and parent, get parent's restr.

        Parameters
        ----------
        child : str
            child table name
        parent : str
            parent table name
        restriction : str
            restriction to apply to child
        attr_map : dict, optional
            dictionary mapping aliases across parend/child, as pulled from
            DataJoint-assembled graph. Default None. Func will flip this dict
            to convert from child to parent fields.
        primary : bool, optional
            Is parent in child's primary key? Default True. Also derived from
            DataJoint-assembled graph. If True, project only primary key fields
            to avoid secondary key collisions.

        Returns
        -------
        List[Dict[str, str]]
            List of dicts containing primary key fields for restricted parent
            table.
        """

        # Need to flip attr_map to respect parent's fields
        attr_reverse = (
            {v: k for k, v in attr_map.items() if k != v} if attr_map else {}
        )
        child_ft = self._get_ft(child)
        parent_ft = self._get_ft(parent).proj()
        restr = restriction or self._get_restr(child_ft) or True
        restr_child = child_ft & restr

        if primary:  # Project only primary key fields to avoid collisions
            join = restr_child.proj(**attr_reverse) * parent_ft
        else:  # Include all fields
            join = restr_child.proj(..., **attr_reverse) * parent_ft

        ret = unique_dicts(join.fetch(*parent_ft.primary_key, as_dict=True))

        if len(ret) == len(parent_ft):
            self._log_truncate(f"NULL restr {parent}")

        return ret

    def cascade1(self, table, restriction, direction="up"):
        """Cascade a restriction up the graph, recursively on parents.

        Parameters
        ----------
        table : str
            table name
        restriction : str
            restriction to apply
        """
        self._set_restr(table, restriction)
        self.visited.add(table)

        next_nodes = (
            self.graph.parents(table)
            if direction == "up"
            else self.graph.children(table)
        )

        for next_table, data in next_nodes.items():
            if next_table in self.visited or next_table in self.no_visit:
                continue

            if next_table.isnumeric():
                next_table, data = self.graph.parents(next_table).popitem()

            parent_restr = self._child_to_parent(
                child=table,
                parent=next_table,
                restriction=restriction,
                **data,
            )

            self.cascade1(
                table=next_table,
                restriction=parent_restr,
                direction=direction,
            )


def unique_dicts(list_of_dict):
    """Remove duplicate dictionaries from a list."""
    return [dict(t) for t in {tuple(d.items()) for d in list_of_dict}]


def _fuzzy_get(index: Union[int, str], names: List[str], sources: List[str]):
    """Given lists of items/names, return item at index or by substring."""
    if isinstance(index, int):
        return sources[index]
    for i, part in enumerate(names):
        if index in part:
            return sources[i]
    return None
