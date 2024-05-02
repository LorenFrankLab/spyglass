"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from abc import ABC, abstractmethod
from collections.abc import KeysView
from enum import Enum
from itertools import chain as iter_chain
from typing import Any, Dict, List, Set, Tuple, Union

from datajoint import FreeTable, Table
from datajoint.condition import make_condition
from datajoint.dependencies import unite_master_parts
from datajoint.utils import get_master, to_camel_case
from networkx import all_simple_paths
from networkx.algorithms.dag import topological_sort
from tqdm import tqdm

from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import PERIPHERAL_TABLES, unique_dicts


class Direction(Enum):
    """Cascade direction enum."""

    UP = "up"
    DOWN = "down"


class AbstractGraph(ABC):
    """Abstract class for graph traversal and restriction application.

    Inherited by...
    - RestrGraph: Cascade restriction(s) through a graph
    - FindKeyGraph: Iherits from RestrGraph. Cascades through the graph to
        find where a restriction works, and cascades back across visited
        nodes.
    - TableChain: Takes parent and child nodes, finds the shortest path,
        and applies a restriction across the path.

    Methods
    -------
    cascade: Abstract method implemented by child classes
    cascade1: Cascade a restriction up/down the graph, recursively

    Properties
    ----------
    all_ft: Get all FreeTables for visited nodes with restrictions applied.
    as_dict: Get visited nodes as a list of dictionaries of
        {table_name: restriction}
    """

    def __init__(self, seed_table: Table, verbose: bool = False, **kwargs):
        """Initialize graph and connection.

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

    @abstractmethod
    def cascade(self):
        """Cascade restrictions through graph."""
        raise NotImplementedError("Child class mut implement `cascade` method")

    def _log_truncate(self, log_str: str, max_len: int = 80):
        """Truncate log lines to max_len and print if verbose."""
        if not self.verbose:
            return
        logger.info(
            log_str[:max_len] + "..." if len(log_str) > max_len else log_str
        )

    def _ensure_name(self, table: Union[str, Table]) -> str:
        """Ensure table is a string."""
        return table if isinstance(table, str) else table.full_table_name

    def _get_node(self, table: Union[str, Table]):
        """Get node from graph."""
        table = self._ensure_name(table)
        if not (node := self.graph.nodes.get(table)):
            raise ValueError(
                f"Table {table} not found in graph."
                + "\n\tPlease import this table and rerun"
            )
        return node

    def _set_node(self, table, attr: str = "ft", value: Any = None):
        """Set attribute on node. General helper for various attributes."""
        _ = self._get_node(table)  # Ensure node exists
        self.graph.nodes[table][attr] = value

    def _get_edge(self, child: str, parent: str) -> Tuple[bool, Dict[str, str]]:
        """Get edge data between child and parent.

        Used as a fallback for _bridge_restr. Not required in typical use.

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            Tuple of boolean indicating direction and edge data. True if child
            is child of parent.
        """
        child = self._ensure_name(child)
        parent = self._ensure_name(parent)

        if edge := self.graph.get_edge_data(parent, child):
            return False, edge
        elif edge := self.graph.get_edge_data(child, parent):
            return True, edge

        # Handle alias nodes. `shortest_path` doesn't work with aliases
        p1 = all_simple_paths(self.graph, child, parent)
        p2 = all_simple_paths(self.graph, parent, child)
        paths = [p for p in iter_chain(p1, p2)]  # list for error handling
        for path in paths:  # Ignore long and non-alias paths
            if len(path) > 3 or (len(path) > 2 and not path[1].isnumeric()):
                continue
            return self._get_edge(path[0], path[1])

        raise ValueError(f"{child} -> {parent} not direct path: {paths}")

    def _get_restr(self, table):
        """Get restriction from graph node."""
        return self._get_node(self._ensure_name(table)).get("restr")

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

        self._set_node(table, "restr", restriction)

    def _get_ft(self, table, with_restr=False):
        """Get FreeTable from graph node. If one doesn't exist, create it."""
        table = self._ensure_name(table)
        if with_restr:
            if not (restr := self._get_restr(table) or False):
                self._log_truncate(f"No restriction for {table}")
        else:
            restr = True

        if not (ft := self._get_node(table).get("ft")):
            ft = FreeTable(self.connection, table)
            self._set_node(table, "ft", ft)

        return ft & restr

    @property
    def all_ft(self):
        """Get restricted FreeTables from all visited nodes.

        Topological sort logic adopted from datajoint.diagram.
        """
        self.cascade()
        nodes = [n for n in self.visited if not n.isnumeric()]
        sorted_nodes = unite_master_parts(
            list(topological_sort(self.graph.subgraph(nodes)))
        )
        all_ft = [
            self._get_ft(table, with_restr=True) for table in sorted_nodes
        ]
        return [ft for ft in all_ft if len(ft) > 0]

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
        direction: Direction = None,
        attr_map: dict = None,
        aliased: bool = None,
        **kwargs,
    ):
        """Given two tables and a restriction, return restriction for table2.

        Similar to ((table1 & restr) * table2).fetch(*table2.primary_key)
        but with the ability to resolve aliases across tables. One table should
        be the parent of the other. If direction or attr_map are not provided,
        they will be inferred from the graph.

        Parameters
        ----------
        table1 : str
            Table name. Restriction always applied to this table.
        table2 : str
            Table name. Restriction pulled from this table.
        restr : str
            Restriction to apply to table1.
        direction : Direction, optional
            Direction to cascade. Default None.
        attr_map : dict, optional
            dictionary mapping aliases across tables, as pulled from
            DataJoint-assembled graph. Default None.


        Returns
        -------
        List[Dict[str, str]]
            List of dicts containing primary key fields for restricted table2.
        """
        if not all([direction, attr_map]):
            dir_bool, edge = self._get_edge(table1, table2)
            direction = "up" if dir_bool else "down"
            attr_map = edge.get("attr_map")

        ft1 = self._get_ft(table1) & restr
        ft2 = self._get_ft(table2)

        if len(ft1) == 0:
            return ["False"]

        if rev_attr := bool(set(attr_map.values()) - set(ft1.heading.names)):
            attr_map = {v: k for k, v in attr_map.items()}  # reverse

        join = ft1.proj(**attr_map) * ft2
        ret = unique_dicts(join.fetch(*ft2.primary_key, as_dict=True))

        if self.verbose:  # For debugging. Not required for typical use.
            partial = (
                "NULL"
                if len(ret) == 0
                else "FULL" if len(ft2) == len(ret) else "part"
            )
            flipped = "Fliped" if rev_attr else "NoFlip"
            dir = "Up" if direction == "up" else "Dn"
            strt = f"{to_camel_case(ft1.table_name)}"
            endp = f"{to_camel_case(ft2.table_name)}"
            self._log_truncate(
                f"{partial} {dir} {flipped}: {strt} -> {endp}, {len(ret)}"
            )

        return ret

    def _camel(self, table):
        if isinstance(table, KeysView):
            table = list(table)
        if not isinstance(table, list):
            table = [table]
        ret = [to_camel_case(t.split(".")[-1].strip("`")) for t in table]
        return ret[0] if len(ret) == 1 else ret

    def cascade1(
        self,
        table: str,
        restriction: str,
        direction: Direction = "up",
        replace=False,
        count=0,
        **kwargs,
    ):
        """Cascade a restriction up the graph, recursively on parents/children.

        Parameters
        ----------
        table : str
            Table name
        restriction : str
            Restriction to apply
        direction : Direction, optional
            Direction to cascade. Default 'up'
        replace : bool, optional
            Replace existing restriction. Default False
        """
        if count > 100:
            raise RecursionError("Cascade1: Recursion limit reached.")

        self._set_restr(table, restriction, replace=replace)
        self.visited.add(table)

        G = self.graph
        next_func = G.parents if direction == "up" else G.children
        dir_dict = {"direction": direction}
        dir_name = "Parents" if direction == "up" else "Children"

        # Master/Parts added will go in opposite direction for one link.
        # Direction is intentionally not passed to _bridge_restr in this case.
        if direction == "up":
            next_tables = {
                k: {**v, **dir_dict} for k, v in next_func(table).items()
            }
            next_tables.update(
                {part: {} for part in self._get_ft(table).parts()}
            )
        else:
            next_tables = {
                k: {**v, **dir_dict} for k, v in next_func(table).items()
            }
            if (master_name := get_master(table)) != "":
                next_tables[master_name] = {}

        log_dict = {
            "Table   ": self._camel(table),
            f"{dir_name}": self._camel(next_func(table).keys()),
            "Parts   ": self._camel(self._get_ft(table).parts()),
            "Master  ": self._camel(get_master(table)),
        }
        logger.info(
            f"Cascade1: {count}\n\t\t\t   "
            + "\n\t\t\t   ".join(f"{k}: {v}" for k, v in log_dict.items())
        )
        for next_table, data in next_tables.items():
            if next_table.isnumeric():  # Skip alias nodes
                next_table, data = next_func(next_table).popitem()

            if (
                next_table in self.visited
                or next_table in self.no_visit  # Subclasses can set this
                or table == next_table
            ):
                path = f"{self._camel(table)} -> {self._camel(next_table)}"
                if next_table in self.visited:
                    self._log_truncate(f"SkipVist: {path}")
                if next_table in self.no_visit:
                    self._log_truncate(f"NoVisit : {path}")
                if table == next_table:
                    self._log_truncate(f"Self    : {path}")

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
                count=count + 1,
            )


class RestrGraph(AbstractGraph):
    def __init__(
        self,
        seed_table: Table,
        table_name: str = None,
        restriction: str = None,
        leaves: List[Dict[str, str]] = None,
        direction: Direction = "up",
        cascade: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """Use graph to cascade restrictions up from leaves to all ancestors.

        'Leaves' are nodes with restrictions applied. Restrictions are cascaded
        up/down the graph to all ancestors/descendants. If cascade is desired
        in both direction, leaves/cascades should be added and run separately.
        Future development could allow for direction setting on a per-leaf
        basis.

        Parameters
        ----------
        seed_table : Table
            Table to use to establish connection and graph
        table_name : str, optional
            Table name of single leaf, default None
        restriction : str, optional
            Restriction to apply to leaf. default None
        leaves : Dict[str, str], optional
            List of dictionaries with keys table_name and restriction. One
            entry per leaf node. Default None.
        direction : Direction, optional
            Direction to cascade. Default 'up'
        cascade : bool, optional
            Whether to cascade restrictions up the graph on initialization.
            Default False
        verbose : bool, optional
            Whether to print verbose output. Default False
        """
        super().__init__(seed_table, verbose=verbose)

        self.add_leaf(
            table_name=table_name, restriction=restriction, direction=direction
        )
        self.add_leaves(leaves)

        if cascade:
            self.cascade(direction=direction)

    def __repr__(self):
        l_str = ",\n\t".join(self.leaves) + "\n" if self.leaves else ""
        processed = "Cascaded" if self.cascaded else "Uncascaded"
        return f"{processed} {self.__class__.__name__}(\n\t{l_str})"

    @property
    def leaf_ft(self):
        """Get restricted FreeTables from graph leaves."""
        return [self._get_ft(table, with_restr=True) for table in self.leaves]

    def add_leaf(
        self, table_name=None, restriction=True, cascade=False, direction="up"
    ) -> None:
        """Add leaf to graph and cascade if requested.

        Parameters
        ----------
        table_name : str, optional
            table name of leaf. Default None, do nothing.
        restriction : str, optional
            restriction to apply to leaf. Default True, no restriction.
        cascade : bool, optional
            Whether to cascade the restrictions up the graph. Default False.
        """
        if not table_name:
            return

        self.cascaded = False

        new_visits = (
            set(self._get_ft(table_name).ancestors())
            if direction == "up"
            else set(self._get_ft(table_name).descendants())
        )

        self.to_visit |= new_visits  # Add to total ancestors
        self.visited -= new_visits  # Remove from visited to revisit

        self.leaves.add(table_name)
        self._set_restr(table_name, restriction)  # Redundant if cascaded

        if cascade:
            self.cascade1(table_name, restriction)
            self.cascade_files()
            self.cascaded = True

    def _process_leaves(self, leaves=None, default_restriction=True):
        """Process leaves to ensure they are unique and have required keys."""
        if not leaves:
            return []
        if not isinstance(leaves, list):
            leaves = [leaves]
        if all(isinstance(leaf, str) for leaf in leaves):
            leaves = [
                {"table_name": leaf, "restriction": default_restriction}
                for leaf in leaves
            ]
        if all(isinstance(leaf, dict) for leaf in leaves) and not all(
            leaf.get("table_name") for leaf in leaves
        ):
            raise ValueError(f"All leaves must have table_name: {leaves}")

        return unique_dicts(leaves)

    def add_leaves(
        self,
        leaves: Union[str, List, List[Dict[str, str]]] = None,
        default_restriction: str = None,
        cascade=False,
    ) -> None:
        """Add leaves to graph and cascade if requested.

        Parameters
        ----------
        leaves : Union[str, List, List[Dict[str, str]]], optional
            Table names of leaves, either as a list of strings or a list of
            dictionaries with keys table_name and restriction. One entry per
            leaf node. Default None, do nothing.
        default_restriction : str, optional
            Default restriction to apply to each leaf. Default True, no
            restriction. Only used if leaf missing restriction.
        cascade : bool, optional
            Whether to cascade the restrictions up the graph. Default False
        """
        leaves = self._process_leaves(
            leaves=leaves, default_restriction=default_restriction
        )
        for leaf in leaves:
            self.add_leaf(
                leaf.get("table_name"),
                leaf.get("restriction"),
                cascade=False,
            )
        if cascade:
            self.cascade()

    def cascade(self, show_progress=None, direction="up") -> None:
        """Cascade all restrictions up the graph.

        Parameters
        ----------
        show_progress : bool, optional
            Show tqdm progress bar. Default to verbose setting.
        """
        if self.cascaded:
            return

        to_visit = self.leaves - self.visited

        for table in tqdm(
            to_visit,
            desc="RestrGraph: cascading restrictions",
            total=len(to_visit),
            disable=not (show_progress or self.verbose),
        ):
            restr = self._get_restr(table)
            self._log_truncate(f"Start     {table}: {restr}")
            self.cascade1(table, restr, direction=direction)
        if not self.visited == self.to_visit:
            raise RuntimeError(
                "Cascade: FAIL - incomplete cascade. Please post issue."
            )

        self.cascade_files()
        self.cascaded = True

    # ----------------------------- File Handling -----------------------------

    def _get_files(self, table):
        """Get analysis files from graph node."""
        return self._get_node(table).get("files", [])

    def cascade_files(self):
        """Set node attribute for analysis files."""
        for table in self.visited:
            ft = self._get_ft(table, with_restr=True)
            if not set(self.analysis_pk).issubset(ft.heading.names):
                continue
            files = list(ft.fetch(*self.analysis_pk))
            self._set_node(table, "files", files)

    @property
    def analysis_file_tbl(self) -> Table:
        """Return the analysis file table. Avoids circular import."""
        from spyglass.common import AnalysisNwbfile

        return AnalysisNwbfile()

    @property
    def analysis_pk(self) -> List[str]:
        """Return primary key fields from analysis file table."""
        return self.analysis_file_tbl.primary_key

    @property
    def file_dict(self) -> Dict[str, List[str]]:
        """Return dictionary of analysis files from all visited nodes.

        Included for debugging, to associate files with tables.
        """
        self.cascade()
        return {t: self._get_node(t).get("files", []) for t in self.visited}

    @property
    def file_paths(self) -> List[str]:
        """Return list of unique analysis files from all visited nodes.

        This covers intermediate analysis files that may not have been fetched
        directly by the user.
        """
        self.cascade()
        return [
            {"file_path": self.analysis_file_tbl.get_abs_path(file)}
            for file in set(
                [f for files in self.file_dict.values() for f in files]
            )
            if file is not None
        ]


class FindKeyGraph(RestrGraph):
    def __init__(
        self,
        seed_table: Table,
        table_name: str = None,
        restriction: str = None,
        leaves: List[Dict[str, str]] = None,
        direction: Direction = "up",
        cascade: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """Graph to restrict leaf by upstream keys.

        Parameters
        ----------
        seed_table : Table
            Table to use to establish connection and graph
        table_name : str, optional
            Table name of single leaf, default seed_table.full_table_name
        restriction : str, optional
            Restriction to apply to leaf. default None, True
        verbose : bool, optional
            Whether to print verbose output. Default False
        """

        super().__init__(seed_table, verbose=verbose)

        self.direction = direction
        self.searched = set()
        self.found = False

        if restriction and table_name:
            self._set_find_restr(table_name, restriction)
            self.add_leaf(table_name, True, cascade=False, direction=direction)
        self.add_leaves(
            leaves,
            default_restriction=restriction,
            cascade=False,
            direction=direction,
        )

        self.no_visit.update(PERIPHERAL_TABLES)

        if cascade and restriction:
            self.cascade()
            self.cascaded = True

    def _set_find_restr(self, table_name, restriction):
        """Set restr to look for from leaf node."""
        if isinstance(restriction, dict):
            logger.warning("Using `>>` or `<<`: DICT unreliable, use STR.")

        restr_attrs = set()  # modified by make_condition
        table_ft = self._get_ft(table_name)
        restr_string = make_condition(table_ft, restriction, restr_attrs)

        self._set_node(table_name, "find_restr", restr_string)
        self._set_node(table_name, "restr_attrs", restr_attrs)

    def _get_find_restr(self, table) -> Tuple[str, Set[str]]:
        """Get restr and restr_attrs from leaf node."""
        node = self._get_node(table)
        return node.get("find_restr", False), node.get("restr_attrs", set())

    def add_leaves(
        self,
        leaves=None,
        default_restriction=None,
        cascade=False,
        direction=None,
    ):
        leaves = self._process_leaves(
            leaves=leaves, default_restriction=default_restriction
        )
        for leaf in leaves:  # Multiple leaves
            self._set_find_restr(**leaf)
            self.add_leaf(
                leaf["table_name"],
                True,
                cascade=False,
                direction=direction,
            )

    def cascade(self, direction=None, show_progress=None) -> None:
        direction = direction or self.direction
        if self.cascaded:
            return
        for table in self.leaves:
            restriction, restr_attrs = self._get_find_restr(table)
            self.cascade1_search(
                table=table,
                restriction=restriction,
                restr_attrs=restr_attrs,
                replace=True,
            )
        self.cascaded = True
        if not self.found:
            searched = "parents" if direction == "up" else "children"
            logger.warning(
                f"Restriction could not be applied to any {searched}.\n\t"
                + f"From: {self.leaves}\n\t"
                + f"Restr: {restriction}"
            )

    def _ban_unsearched(self):
        """After found match, ignore others for cascade back to leaf."""
        all_tables = set([n for n in self.graph.nodes])
        unsearched = all_tables - self.searched
        camel_searched = self._camel(list(self.searched))
        logger.info(f"Searched: {camel_searched}")
        self.no_visit.update(unsearched)

    def cascade1_search(
        self,
        table: str = None,
        restriction: str = True,
        restr_attrs: Set[str] = None,
        direction: Direction = None,
        replace: bool = True,
        limit: int = 100,
    ):
        if self.found or not table or limit < 1 or table in self.searched:
            return

        self.searched.add(table)

        direction = direction or self.direction
        next_func = (
            self.graph.parents if direction == "up" else self.graph.children
        )

        next_searches = set()
        for next_table, data in next_func(table).items():
            self._log_truncate(
                f"Search: {self._camel(table)} -> {self._camel(next_table)}"
            )
            if next_table.isnumeric():
                next_table, data = next_func(next_table).popitem()

            if next_table in self.no_visit or table == next_table:
                continue

            next_ft = self._get_ft(next_table)
            if restr_attrs.issubset(set(next_ft.heading.names)):
                self.searched.add(next_table)
                # self.searched.add(get_master(next_table))
                self.searched.update(next_ft.parts())
                self.found = True
                self._ban_unsearched()
                self.cascade1(
                    table=next_table,
                    restriction=restriction,
                    direction="down" if direction == "up" else "up",
                    replace=replace,
                    **data,
                )
                return

            next_searches.update(
                set([*next_ft.parts(), get_master(next_table), next_table])
            )

        for next_table in next_searches:
            if not next_table:
                continue  # Skip None from get_master
            self.cascade1_search(
                table=next_table,
                restriction=restriction,
                restr_attrs=restr_attrs,
                direction=direction,
                replace=replace,
                limit=limit - 1,
            )
            if self.found:
                return
