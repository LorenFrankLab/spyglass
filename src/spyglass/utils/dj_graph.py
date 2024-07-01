"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from functools import cached_property
from itertools import chain as iter_chain
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

from datajoint import FreeTable, Table
from datajoint.condition import make_condition
from datajoint.dependencies import unite_master_parts
from datajoint.user_tables import TableMeta
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
from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_helper_fn import (
    PERIPHERAL_TABLES,
    fuzzy_get,
    unique_dicts,
)


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
    ft_from_list: Return non-empty FreeTable objects from list of table names

    Properties
    ----------
    all_ft: Get all FreeTables for visited nodes with restrictions applied.
    restr_ft: Get non-empty FreeTables for visited nodes with restrictions.
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

        # Deepcopy graph to avoid seed `load()` resetting custom attributes
        seed_table.connection.dependencies.load()
        graph = seed_table.connection.dependencies
        orig_conn = graph._conn  # Cannot deepcopy connection
        graph._conn = None
        self.graph = deepcopy(graph)
        graph._conn = orig_conn

        # undirect not needed in all cases but need to do before adding ft nodes
        self.undirect_graph = self.graph.to_undirected()

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

    # --------------------------- Dunder Properties ---------------------------

    def __repr__(self):
        l_str = (
            ",\n\t".join(self._camel(self.leaves)) + "\n"
            if self.leaves
            else "Seed: " + self._camel(self.seed_table) + "\n"
        )
        casc_str = "Cascaded" if self.cascaded else "Uncascaded"
        return f"{casc_str} {self.__class__.__name__}(\n\t{l_str})"

    def __getitem__(self, index: Union[int, str]):
        names = [t.full_table_name for t in self.restr_ft]
        return fuzzy_get(index, names, self.restr_ft)

    def __len__(self):
        return len(self.restr_ft)

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
        table = self._ensure_names(table)
        if isinstance(table, str):
            return to_camel_case(table.split(".")[-1].strip("`"))
        if isinstance(table, Iterable) and not isinstance(
            table, (Table, TableMeta)
        ):
            return [self._camel(t) for t in table]

    # ------------------------------ Graph Nodes ------------------------------

    def _ensure_names(
        self, table: Union[str, Table] = None
    ) -> Union[str, List[str]]:
        """Ensure table is a string."""
        if table is None:
            return None
        if isinstance(table, str):
            return table
        if isinstance(table, Iterable) and not isinstance(
            table, (Table, TableMeta)
        ):
            return [self._ensure_names(t) for t in table]
        return getattr(table, "full_table_name", None)

    def _get_node(self, table: Union[str, Table]):
        """Get node from graph."""
        table = self._ensure_names(table)
        if not (node := self.graph.nodes.get(table)):
            raise ValueError(
                f"Table {table} not found in graph."
                + "\n\tPlease import this table and rerun"
            )
        return node

    def _set_node(self, table, attr: str = "ft", value: Any = None):
        """Set attribute on node. General helper for various attributes."""
        table = self._ensure_names(table)
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
        child = self._ensure_names(child)
        parent = self._ensure_names(parent)

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
        return self._get_node(self._ensure_names(table)).get("restr")

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

    def _get_ft(self, table, with_restr=False, warn=True):
        """Get FreeTable from graph node. If one doesn't exist, create it."""
        table = self._ensure_names(table)
        if with_restr:
            if not (restr := self._get_restr(table) or False):
                if warn:
                    self._log_truncate(f"No restr for {self._camel(table)}")
        else:
            restr = True

        if not (ft := self._get_node(table).get("ft")):
            ft = FreeTable(self.connection, table)
            self._set_node(table, "ft", ft)

        return ft & restr

    def _is_out(self, table, warn=True):
        """Check if table is outside of spyglass."""
        table = self._ensure_names(table)
        if self.graph.nodes.get(table):
            return False
        ret = table.split(".")[0].split("_")[0].strip("`") not in SHARED_MODULES
        if warn and ret:  # Log warning if outside
            logger.warning(f"Skipping unimported: {table}")
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
        if self._is_out(table2) or self._is_out(table1):  # 2 more likely
            return ["False"]  # Stop cascade if outside, see #1002

        if not all([direction, attr_map]):
            dir_bool, edge = self._get_edge(table1, table2)
            direction = "up" if dir_bool else "down"
            attr_map = edge.get("attr_map")

        # May return empty table if outside imported and outside spyglass
        ft1 = self._get_ft(table1) & restr
        ft2 = self._get_ft(table2)

        if len(ft1) == 0 or len(ft2) == 0:
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

        bonus = {}  # Add master and parts to next tables
        direction = Direction(direction)
        if direction == Direction.UP:
            next_func = G.parents
            table_ft = self._get_ft(table)
            for part in table_ft.parts():  # Assumes parts do not alias master
                bonus[part] = {
                    "attr_map": {k: k for k in table_ft.primary_key},
                    **dir_dict,
                }
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

        if next_list := next_tables.keys():
            self._log_truncate(
                f"Checking {count:>2}: {self._camel(table)}"
                + f" -> {self._camel(next_list)}"
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

    def _topo_sort(
        self, nodes: List[str], subgraph: bool = True, reverse: bool = False
    ) -> List[str]:
        """Return topologically sorted list of nodes.

        Parameters
        ----------
        nodes : List[str]
            List of table names
        subgraph : bool, optional
            Whether to use subgraph. Default True
        reverse : bool, optional
            Whether to reverse the order. Default False. If true, bottom-up.
            If None, return nodes as is.
        """
        if reverse is None:
            return nodes
        nodes = [
            node
            for node in self._ensure_names(nodes)
            if not self._is_out(node, warn=False)
        ]
        graph = self.graph.subgraph(nodes) if subgraph else self.graph
        ordered = unite_master_parts(list(topological_sort(graph)))
        if reverse:
            ordered.reverse()
        return [n for n in ordered if n in nodes]

    @property
    def all_ft(self):
        """Get restricted FreeTables from all visited nodes.

        Topological sort logic adopted from datajoint.diagram.
        """
        self.cascade(warn=False)
        nodes = [n for n in self.visited if not n.isnumeric()]
        return [
            self._get_ft(table, with_restr=True, warn=False)
            for table in self._topo_sort(nodes, subgraph=True, reverse=False)
        ]

    @property
    def restr_ft(self):
        """Get non-empty restricted FreeTables from all visited nodes."""
        return [ft for ft in self.all_ft if len(ft) > 0]

    def ft_from_list(
        self,
        tables: List[str],
        with_restr: bool = True,
        sort_reverse: bool = None,
        return_empty: bool = False,
    ) -> List[FreeTable]:
        """Return non-empty FreeTable objects from list of table names.

        Parameters
        ----------
        tables : List[str]
            List of table names
        with_restr : bool, optional
            Restrict FreeTable to restriction. Default True.
        sort_reverse : bool, optional
            Sort reverse topologically. Default True. If None, no sort.
        """

        self.cascade(warn=False)

        fts = [
            self._get_ft(table, with_restr=with_restr, warn=False)
            for table in self._topo_sort(
                tables, subgraph=False, reverse=sort_reverse
            )
        ]

        return fts if return_empty else [ft for ft in fts if len(ft) > 0]

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
        leaves: List[Dict[str, str]] = None,
        destinations: List[str] = None,
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
        leaves : Dict[str, str], optional
            List of dictionaries with keys table_name and restriction. One
            entry per leaf node. Default None.
        destinations : List[str], optional
            List of endpoints of interest in the graph. Default None. Used to
            ignore nodes not in the path(s) to the destination(s).
        direction : Direction, optional
            Direction to cascade. Default 'up'
        cascade : bool, optional
            Whether to cascade restrictions up the graph on initialization.
            Default False
        verbose : bool, optional
            Whether to print verbose output. Default False
        """
        super().__init__(seed_table, verbose=verbose)

        self.add_leaves(leaves)

        dir_list = ["up", "down"] if direction == "both" else [direction]

        if cascade:
            for dir in dir_list:
                self._log_truncate(f"Start {dir:<4} : {self.leaves}")
                self.cascade(direction=dir)
                self.cascaded = False
                self.visited -= self.leaves
            self.cascaded = True
            self.visited |= self.leaves

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
        """Process leaves to ensure they are unique and have required keys.

        Accepts ...
        - [str]: table names, use default_restriction
        - [{'table_name': str, 'restriction': str}]: used for export
        - [{table_name: restriction}]: userd for distance restriction
        """
        if not leaves:
            return []
        if not isinstance(leaves, list):
            leaves = [leaves]
        if all(isinstance(leaf, str) for leaf in leaves):
            leaves = [
                {"table_name": leaf, "restriction": default_restriction}
                for leaf in leaves
            ]
        hashable = True
        if all(isinstance(leaf, dict) for leaf in leaves):
            new_leaves = []
            for leaf in leaves:
                if "table_name" in leaf and "restriction" in leaf:
                    new_leaves.append(leaf)
                    continue
                for table, restr in leaf.items():
                    if not isinstance(restr, (str, dict)):
                        hashable = False  # likely a dj.AndList
                    new_leaves.append(
                        {"table_name": table, "restriction": restr}
                    )
            if not hashable:
                return new_leaves
            leaves = new_leaves

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

    def cascade(self, show_progress=None, direction="up", warn=True) -> None:
        """Cascade all restrictions up the graph.

        Parameters
        ----------
        show_progress : bool, optional
            Show tqdm progress bar. Default to verbose setting.
        """
        if self.cascaded:
            if warn:
                self._log_truncate("Already cascaded")
            return

        to_visit = self.leaves - self.visited

        for table in tqdm(
            to_visit,
            desc="RestrGraph: cascading restrictions",
            total=len(to_visit),
            disable=not (show_progress or self.verbose),
        ):
            restr = self._get_restr(table)
            self._log_truncate(
                f"Start  {direction:<4}: {self._camel(table)}, {restr}"
            )
            self.cascade1(table, restr, direction=direction)

        self.cascaded = True  # Mark here so next step can use `restr_ft`
        self.cascade_files()  # Otherwise attempts to re-cascade, recursively

    # ----------------------------- File Handling -----------------------------

    @property
    def analysis_file_tbl(self) -> Table:
        """Return the analysis file table. Avoids circular import."""
        from spyglass.common import AnalysisNwbfile

        return AnalysisNwbfile()

    def cascade_files(self):
        """Set node attribute for analysis files."""
        analysis_pk = self.analysis_file_tbl.primary_key
        for ft in self.restr_ft:
            if not set(analysis_pk).issubset(ft.heading.names):
                continue
            files = list(ft.fetch(*analysis_pk))
            self._set_node(ft, "files", files)

    @property
    def file_dict(self) -> Dict[str, List[str]]:
        """Return dictionary of analysis files from all visited nodes.

        Included for debugging, to associate files with tables.
        """
        self.cascade(warn=False)
        return {t: self._get_node(t).get("files", []) for t in self.restr_ft}

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


class TableChain(RestrGraph):
    """Class for representing a chain of tables.

    A chain is a sequence of tables from parent to child identified by
    networkx.shortest_path from parent to child. To avoid issues with merge
    tables, use the Merge table as the child, not the part table.

    Either the parent or child can be omitted if a search_restr is provided.
    The missing table will be found by searching for where the restriction
    can be applied.

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
    cascade_search()
        Search from the leaf node to find where a restriction can be applied.
    """

    def __init__(
        self,
        parent: Table = None,
        child: Table = None,
        direction: Direction = Direction.NONE,
        search_restr: str = None,
        cascade: bool = False,
        verbose: bool = False,
        banned_tables: List[str] = None,
        **kwargs,
    ):
        self.parent = self._ensure_names(parent)
        self.child = self._ensure_names(child)

        if not self.parent and not self.child:
            raise ValueError("Parent or child table required.")

        seed_table = parent if isinstance(parent, Table) else child
        super().__init__(seed_table=seed_table, verbose=verbose)

        self._ignore_peripheral(except_tables=[self.parent, self.child])
        self.no_visit.update(self._ensure_names(banned_tables) or [])
        self.no_visit.difference_update(set([self.parent, self.child]))
        self.searched_tables = set()
        self.found_restr = False
        self.link_type = None
        self.searched_path = False
        self._link_symbol = " -> "

        self.search_restr = search_restr
        self.direction = Direction(direction)
        if self.parent and self.child and not self.direction:
            self.direction = Direction.DOWN

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
            self.cascade_search()  # only cascade if found or not looking
            if (search_restr and self.found_restr) or not search_restr:
                self.cascade(restriction=search_restr)
            self.cascaded = True

    # ------------------------------ Ignore Nodes ------------------------------

    def _ignore_peripheral(self, except_tables: List[str] = None):
        """Ignore peripheral tables in graph traversal."""
        except_tables = self._ensure_names(except_tables)
        ignore_tables = set(PERIPHERAL_TABLES) - set(except_tables or [])
        self.no_visit.update(ignore_tables)
        self.undirect_graph.remove_nodes_from(ignore_tables)

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

    @property
    def path_str(self) -> str:
        if not self.path:
            return "No link"
        return self._link_symbol.join([self._camel(t) for t in self.path])

    @property
    def path_ft(self) -> List[FreeTable]:
        """Return FreeTables along the path."""
        path_with_ends = set([self.parent, self.child]) | set(self.path)
        return self.ft_from_list(path_with_ends, with_restr=True)

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
            self.link_type = None
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

        and_parts = set([table])
        if master := get_master(table):
            and_parts.add(master)
        if parts := self._get_ft(table).parts():
            and_parts.update(parts)

        self.searched_tables.update(and_parts)

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
        search_graph = self.graph if directed else self.undirect_graph

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

    def cascade(
        self, restriction: str = None, direction: Direction = None, **kwargs
    ):
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
            restriction=restriction or self._get_restr(start) or True,
            direction=direction,
            replace=True,
        )

        # Cascade will stop if any restriction is empty, so set rest to None
        # This would cause issues if we want a table partway through the chain
        # but that's not a typical use case, were the start and end are desired
        non_numeric = [t for t in self.path if not t.isnumeric()]
        if any(self._get_restr(t) is None for t in non_numeric):
            for table in non_numeric:
                if table is not start:
                    self._set_restr(table, False, replace=True)

        return self._get_ft(end, with_restr=True)

    def restrict_by(self, *args, **kwargs) -> None:
        """Cascade passthrough."""
        return self.cascade(*args, **kwargs)
