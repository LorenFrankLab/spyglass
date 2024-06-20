"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from abc import ABC, abstractmethod
from collections.abc import KeysView
from enum import Enum
from functools import cached_property
from itertools import chain as iter_chain
from typing import Any, Dict, List, Set, Tuple, Union

import datajoint as dj
from datajoint import FreeTable, Table
from datajoint.condition import make_condition
from datajoint.dependencies import unite_master_parts
from datajoint.utils import get_master, to_camel_case
from networkx import (
    NetworkXNoPath,
    NodeNotFound,
    all_simple_paths,
    shortest_path,
)
from networkx.algorithms.dag import topological_sort
from tqdm import tqdm

from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import (
    PERIPHERAL_TABLES,
    fuzzy_get,
    unique_dicts,
)
from spyglass.utils.dj_merge_tables import is_merge_table


class Direction(Enum):
    """Cascade direction enum. Calling Up returns True. Inverting flips."""

    UP = "up"
    DOWN = "down"
    NONE = None

    def __str__(self):
        return self.value

    def __invert__(self) -> "Direction":
        """Invert the direction."""
        if self.value is None:
            logger.warning("Inverting NONE direction")
            return Direction.NONE
        return Direction.UP if self.value == "down" else Direction.DOWN

    def __bool__(self) -> bool:
        """Return True if direction is not None."""
        return self.value is not None


class AbstractGraph(ABC):
    """Abstract class for graph traversal and restriction application.

    Inherited by...
    - RestrGraph: Cascade restriction(s) through a graph
    - TableChain: Takes parent and child nodes, finds the shortest path,
        and applies a restriction across the path. If either parent or child
        is a merge table, use TableChains instead. If either parent or child
        are not provided, search_restr is required to find the path to the
        missing table.

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
        self.seed_table = seed_table
        self.connection = seed_table.connection

        # Undirected graph may not be needed, but adding FT to the graph
        # prevents `to_undirected` from working. If using undirected, remove
        # PERIPHERAL_TABLES from the graph.
        self.graph = seed_table.connection.dependencies
        self.graph.load()

        self.verbose = verbose
        self.leaves = set()
        self.visited = set()
        self.to_visit = set()
        self.no_visit = set()
        self.cascaded = False

    # --------------------------- Abstract Methods ---------------------------

    @abstractmethod
    def cascade(self):
        """Cascade restrictions through graph."""
        raise NotImplementedError("Child class mut implement `cascade` method")

    # ---------------------------- Logging Helpers ----------------------------

    def _log_truncate(self, log_str: str, max_len: int = 80):
        """Truncate log lines to max_len and print if verbose."""
        if not self.verbose:
            return
        logger.info(
            log_str[:max_len] + "..." if len(log_str) > max_len else log_str
        )

    def _camel(self, table):
        """Convert table name(s) to camel case."""
        if isinstance(table, KeysView):
            table = list(table)
        if not isinstance(table, list):
            table = [table]
        ret = [to_camel_case(t.split(".")[-1].strip("`")) for t in table]
        return ret[0] if len(ret) == 1 else ret

    def _print_restr(self):
        """Print restrictions for debugging."""
        for table in self.visited:
            if restr := self._get_restr(table):
                logger.info(f"{table}: {restr}")

    # ------------------------------ Graph Nodes ------------------------------

    def _ensure_name(self, table: Union[str, Table] = None) -> str:
        """Ensure table is a string."""
        if table is None:
            return None
        if isinstance(table, str):
            return table
        if isinstance(table, list):
            return [self._ensure_name(t) for t in table]
        return getattr(table, "full_table_name", None)

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

        Used as a fallback for _bridge_restr. Required for Maser/Part links to
        temporarily flip direction.

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

    def _and_parts(self, table):
        """Return table, its master and parts."""
        ret = [table]
        if master := get_master(table):
            ret.append(master)
        if parts := self._get_ft(table).parts():
            ret.extend(parts)
        return ret

    # ---------------------------- Graph Traversal -----------------------------

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

        if bool(set(attr_map.values()) - set(ft1.heading.names)):
            attr_map = {v: k for k, v in attr_map.items()}  # reverse

        join = ft1.proj(**attr_map) * ft2
        ret = unique_dicts(join.fetch(*ft2.primary_key, as_dict=True))

        if self.verbose:  # For debugging. Not required for typical use.
            result = (
                "EMPTY"
                if len(ret) == 0
                else "FULL" if len(ft2) == len(ret) else "partial"
            )
            path = f"{self._camel(table1)} -> {self._camel(table2)}"
            self._log_truncate(f"Bridge Link: {path}: result {result}")

        return ret

    def _get_next_tables(self, table: str, direction: Direction) -> Tuple:
        """Get next tables/func based on direction.

        Used in cascade1 and cascade1_search to add master and parts. Direction
        is intentionally omitted to force _get_edge to determine the edge for
        this gap before resuming desired direction. Nextfunc is used to get
        relevant parent/child tables after aliast node.

        Parameters
        ----------
        table : str
            Table name
        direction : Direction
            Direction to cascade

        Returns
        -------
        Tuple[Dict[str, Dict[str, str]], Callable
            Tuple of next tables and next function to get parent/child tables.
        """
        G = self.graph
        dir_dict = {"direction": direction}

        bonus = {}
        direction = Direction(direction)
        if direction == Direction.UP:
            next_func = G.parents
            bonus.update({part: {} for part in self._get_ft(table).parts()})
        elif direction == Direction.DOWN:
            next_func = G.children
            if (master_name := get_master(table)) != "":
                bonus = {master_name: {}}
        else:
            raise ValueError(f"Invalid direction: {direction}")

        next_tables = {
            k: {**v, **dir_dict} for k, v in next_func(table).items()
        }
        next_tables.update(bonus)

        return next_tables, next_func

    def cascade1(
        self,
        table: str,
        restriction: str,
        direction: Direction = Direction.UP,
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

        next_tables, next_func = self._get_next_tables(table, direction)

        self._log_truncate(
            f"Checking {count:>2}: {self._camel(next_tables.keys())}"
        )
        for next_table, data in next_tables.items():
            if next_table.isnumeric():  # Skip alias nodes
                next_table, data = next_func(next_table).popitem()

            if (
                next_table in self.visited
                or next_table in self.no_visit  # Subclasses can set this
                or table == next_table
            ):
                reason = (
                    "Already saw"
                    if next_table in self.visited
                    else "Banned Tbl "
                )
                self._log_truncate(f"{reason}: {self._camel(next_table)}")
                continue

            next_restr = self._bridge_restr(
                table1=table,
                table2=next_table,
                restr=restriction,
                **data,
            )

            if next_restr == ["False"]:  # Stop cascade if empty restriction
                continue

            self.cascade1(
                table=next_table,
                restriction=next_restr,
                direction=direction,
                replace=replace,
                count=count + 1,
            )

    # ---------------------------- Graph Properties ----------------------------

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

    # --------------------------- Dunder Properties ---------------------------

    def __repr__(self):
        l_str = ",\n\t".join(self.leaves) + "\n" if self.leaves else ""
        processed = "Cascaded" if self.cascaded else "Uncascaded"
        return f"{processed} {self.__class__.__name__}(\n\t{l_str})"

    def __getitem__(self, index: Union[int, str]):
        all_ft_names = [t.full_table_name for t in self.all_ft]
        return fuzzy_get(index, all_ft_names, self.all_ft)

    def __len__(self):
        return len(self.all_ft)

    # ---------------------------- Public Properties --------------------------

    @property
    def leaf_ft(self):
        """Get restricted FreeTables from graph leaves."""
        return [self._get_ft(table, with_restr=True) for table in self.leaves]

    # ------------------------------- Add Nodes -------------------------------

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

    # ------------------------------ Graph Traversal --------------------------

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


class TableChains:
    """Class for representing chains from parent to Merge table via parts.

    Functions as a plural version of TableChain, allowing a single `cascade`
    call across all chains from parent -> Merge table.

    Attributes
    ----------
    parent : Table
        Parent or origin of chains.
    child : Table
        Merge table or destination of chains.
    connection : datajoint.Connection, optional
        Connection to database used to create FreeTable objects. Defaults to
        parent.connection.
    part_names : List[str]
        List of full table names of child parts.
    chains : List[TableChain]
        List of TableChain objects for each part in child.
    has_link : bool
        Cached attribute to store whether parent is linked to child via any of
        child parts. False if (a) child is not in parent.descendants or (b)
        nx.NetworkXNoPath is raised by nx.shortest_path for all chains.

    Methods
    -------
    __init__(parent, child, connection=None)
        Initialize TableChains with parent and child tables.
    __repr__()
        Return full representation of chains.
        Multiline parent -> child for each chain.
    __len__()
        Return number of chains with links.
    __getitem__(index: Union[int, str])
        Return TableChain object at index, or use substring of table name.
    cascade(restriction: str = None)
        Return list of cascade for each chain in self.chains.
    """

    def __init__(self, parent, child, direction=Direction.DOWN, verbose=False):
        self.parent = parent
        self.child = child
        self.connection = parent.connection
        self.part_names = child.parts()
        self.chains = [
            TableChain(parent, part, direction=direction, verbose=verbose)
            for part in self.part_names
        ]
        self.has_link = any([chain.has_link for chain in self.chains])

    # --------------------------- Dunder Properties ---------------------------

    def __repr__(self):
        l_str = ",\n\t".join([str(c) for c in self.chains]) + "\n"
        return f"{self.__class__.__name__}(\n\t{l_str})"

    def __len__(self):
        return len([c for c in self.chains if c.has_link])

    def __getitem__(self, index: Union[int, str]):
        """Return FreeTable object at index."""
        return fuzzy_get(index, self.part_names, self.chains)

    # ---------------------------- Public Properties --------------------------

    @property
    def max_len(self):
        """Return length of longest chain."""
        return max([len(chain) for chain in self.chains])

    # ------------------------------ Graph Traversal --------------------------

    def cascade(
        self, restriction: str = None, direction: Direction = Direction.DOWN
    ):
        """Return list of cascades for each chain in self.chains."""
        restriction = restriction or self.parent.restriction or True
        cascades = []
        for chain in self.chains:
            if joined := chain.cascade(restriction, direction):
                cascades.append(joined)
        return cascades


class TableChain(RestrGraph):
    """Class for representing a chain of tables.

    A chain is a sequence of tables from parent to child identified by
    networkx.shortest_path. Parent -> Merge should use TableChains instead to
    handle multiple paths to the respective parts of the Merge table.

    Attributes
    ----------
    parent : str
        Parent or origin of chain.
    child : str
        Child or destination of chain.
    has_link : bool
        Cached attribute to store whether parent is linked to child.
    path : List[str]
        Names of tables along the path from parent to child.
    all_ft : List[dj.FreeTable]
        List of FreeTable objects for each table in chain with restriction
        applied.

    Methods
    -------
    find_path(directed=True)
        Returns path OrderedDict of full table names in chain. If directed is
        True, uses directed graph. If False, uses undirected graph. Undirected
        excludes PERIPHERAL_TABLES like interval_list, nwbfile, etc. to maintain
        valid joins.
    cascade(restriction: str = None, direction: str = "up")
        Given a restriction at the beginning, return a restricted FreeTable
        object at the end of the chain. If direction is 'up', start at the child
        and move up to the parent. If direction is 'down', start at the parent.
    """

    def __init__(
        self,
        parent: Table = None,
        child: Table = None,
        direction: Direction = Direction.NONE,
        search_restr: str = None,
        cascade: bool = False,
        verbose: bool = False,
        allow_merge: bool = False,
        banned_tables: List[str] = None,
        **kwargs,
    ):
        if not allow_merge and child is not None and is_merge_table(child):
            raise TypeError("Child is a merge table. Use TableChains instead.")

        self.parent = self._ensure_name(parent)
        self.child = self._ensure_name(child)

        if not self.parent and not self.child:
            raise ValueError("Parent or child table required.")
        if not search_restr and not (self.parent and self.child):
            raise ValueError("Search restriction required to find path.")

        seed_table = parent if isinstance(parent, Table) else child
        super().__init__(seed_table=seed_table, verbose=verbose)

        self.no_visit.update(PERIPHERAL_TABLES)
        self.no_visit.update(self._ensure_name(banned_tables) or [])
        self.no_visit.difference_update([self.parent, self.child])
        self.searched_tables = set()
        self.found_restr = False
        self.link_type = None
        self.searched_path = False
        self._link_symbol = " -> "

        self.search_restr = search_restr
        self.direction = Direction(direction)

        self.leaf = None
        if search_restr and not parent:
            self.direction = Direction.UP
            self.leaf = self.child
        if search_restr and not child:
            self.direction = Direction.DOWN
            self.leaf = self.parent
        if self.leaf:
            self._set_find_restr(self.leaf, search_restr)
            self.add_leaf(self.leaf, True, cascade=False, direction=direction)

        if cascade and search_restr:
            self.cascade_search()
            self.cascade(restriction=search_restr)
            self.cascaded = True

    # --------------------------- Dunder Properties ---------------------------

    def __str__(self):
        """Return string representation of chain: parent -> child."""
        if not self.has_link:
            return "No link"
        return (
            self._camel(self.parent)
            + self._link_symbol
            + self._camel(self.child)
        )

    def __repr__(self):
        """Return full representation of chain: parent -> {links} -> child."""
        if not self.has_link:
            return "No link"
        return "Chain: " + self.path_str

    def __len__(self):
        """Return number of tables in chain."""
        if not self.has_link:
            return 0
        return len(self.path)

    def __getitem__(self, index: Union[int, str]):
        return fuzzy_get(index, self.path, self.all_ft)

    # ---------------------------- Public Properties --------------------------

    @property
    def has_link(self) -> bool:
        """Return True if parent is linked to child.

        If not searched, search for path. If searched and no link is found,
        return False. If searched and link is found, return True.
        """
        if not self.searched_path:
            _ = self.path
        return self.link_type is not None

    @cached_property
    def all_ft(self) -> List[dj.FreeTable]:
        """Return list of FreeTable objects for each table in chain.

        Unused. Preserved for future debugging.
        """
        if not self.has_link:
            return None
        return [
            self._get_ft(table, with_restr=False)
            for table in self.path
            if not table.isnumeric()
        ]

    @property
    def path_str(self) -> str:
        if not self.path:
            return "No link"
        return self._link_symbol.join([self._camel(t) for t in self.path])

    # ------------------------------ Graph Nodes ------------------------------

    def _set_find_restr(self, table_name, restriction):
        """Set restr to look for from leaf node."""
        if isinstance(restriction, dict):
            restriction = [restriction]

        if isinstance(restriction, list) and all(
            [isinstance(r, dict) for r in restriction]
        ):
            restr_attrs = set(key for restr in restriction for key in restr)
            find_restr = restriction
        elif isinstance(restriction, str):
            restr_attrs = set()  # modified by make_condition
            table_ft = self._get_ft(table_name)
            find_restr = make_condition(table_ft, restriction, restr_attrs)
        else:
            raise ValueError(
                f"Invalid restriction type, use STR: {restriction}"
            )

        self._set_node(table_name, "restr_attrs", restr_attrs)
        self._set_node(table_name, "find_restr", find_restr)

    def _get_find_restr(self, table) -> Tuple[str, Set[str]]:
        """Get restr and restr_attrs from leaf node."""
        node = self._get_node(table)
        return node.get("find_restr", False), node.get("restr_attrs", set())

    # ---------------------------- Graph Traversal ----------------------------

    def cascade_search(self) -> None:
        if self.cascaded:
            return
        restriction, restr_attrs = self._get_find_restr(self.leaf)
        self.cascade1_search(
            table=self.leaf,
            restriction=restriction,
            restr_attrs=restr_attrs,
            replace=True,
        )
        if not self.found_restr:
            searched = (
                "parents" if self.direction == Direction.UP else "children"
            )
            logger.warning(
                f"Restriction could not be applied to any {searched}.\n\t"
                + f"From: {self.leaves}\n\t"
                + f"Restr: {restriction}"
            )

    def _set_found_vars(self, table):
        """Set found_restr and searched_tables."""
        self._set_restr(table, self.search_restr, replace=True)
        self.found_restr = True
        self.searched_tables.update(set(self._and_parts(table)))

        if self.direction == Direction.UP:
            self.parent = table
        elif self.direction == Direction.DOWN:
            self.child = table

        self._log_truncate(f"FVars: {self._camel(table)}")

        self.direction = ~self.direction
        _ = self.path  # Reset path

    def cascade1_search(
        self,
        table: str = None,
        restriction: str = True,
        restr_attrs: Set[str] = None,
        replace: bool = True,
        limit: int = 100,
        **kwargs,
    ):
        if (
            self.found_restr
            or not table
            or limit < 1
            or table in self.searched_tables
        ):
            return

        self.searched_tables.add(table)
        next_tables, next_func = self._get_next_tables(table, self.direction)

        for next_table, data in next_tables.items():
            if next_table.isnumeric():
                next_table, data = next_func(next_table).popitem()
            self._log_truncate(
                f"Search Link: {self._camel(table)} -> {self._camel(next_table)}"
            )

            if next_table in self.no_visit or table == next_table:
                reason = "Already Saw" if next_table == table else "Banned Tbl "
                self._log_truncate(f"{reason}: {self._camel(next_table)}")
                continue

            next_ft = self._get_ft(next_table)
            if restr_attrs.issubset(set(next_ft.heading.names)):
                self._log_truncate(f"Found: {self._camel(next_table)}")
                self._set_found_vars(next_table)
                return

            self.cascade1_search(
                table=next_table,
                restriction=restriction,
                restr_attrs=restr_attrs,
                replace=replace,
                limit=limit - 1,
                **data,
            )
            if self.found_restr:
                return

    # ------------------------------ Path Finding ------------------------------

    def find_path(self, directed=True) -> List[str]:
        """Return list of full table names in chain.

        Parameters
        ----------
        directed : bool, optional
            If True, use directed graph. If False, use undirected graph.
            Defaults to True. Undirected permits paths to traverse from merge
            part-parent -> merge part -> merge table. Undirected excludes
            PERIPHERAL_TABLES like interval_list, nwbfile, etc.

        Returns
        -------
        List[str]
            List of names in the path.
        """
        source, target = self.parent, self.child
        search_graph = self.graph

        if not directed:
            self.connection.dependencies.load()
            self.undirect_graph = self.connection.dependencies.to_undirected()
            search_graph = self.undirect_graph

        search_graph.remove_nodes_from(self.no_visit)

        try:
            path = shortest_path(search_graph, source, target)
        except NetworkXNoPath:
            return None  # No path found, parent func may do undirected search
        except NodeNotFound:
            self.searched_path = True  # No path found, don't search again
            return None

        self._log_truncate(f"Path Found : {path}")

        ignore_nodes = self.graph.nodes - set(path)
        self.no_visit.update(ignore_nodes)

        self._log_truncate(f"Ignore     : {ignore_nodes}")
        return path

    @cached_property
    def path(self) -> list:
        """Return list of full table names in chain."""
        if self.searched_path and not self.has_link:
            return None

        path = None
        if path := self.find_path(directed=True):
            self.link_type = "directed"
        elif path := self.find_path(directed=False):
            self.link_type = "undirected"
        self.searched_path = True

        return path

    def cascade(self, restriction: str = None, direction: Direction = None):
        if not self.has_link:
            return

        _ = self.path

        direction = Direction(direction) or self.direction
        if direction == Direction.UP:
            start, end = self.child, self.parent
        elif direction == Direction.DOWN:
            start, end = self.parent, self.child
        else:
            raise ValueError(f"Invalid direction: {direction}")

        self.cascade1(
            table=start,
            restriction=restriction or self._get_restr(start),
            direction=direction,
            replace=True,
        )

        return self._get_ft(end, with_restr=True)

    def restrict_by(self, *args, **kwargs) -> None:
        """Cascade passthrough."""
        return self.cascade(*args, **kwargs)
