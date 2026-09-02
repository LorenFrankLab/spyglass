"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property, lru_cache
from hashlib import md5 as hash_md5
from itertools import chain as iter_chain
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

import datajoint as dj
from datajoint import FreeTable, Table, VirtualModule
from datajoint.condition import make_condition
from datajoint.expression import QueryExpression
from datajoint.hash import key_hash
from datajoint.user_tables import TableMeta
from datajoint.utils import get_master, to_camel_case
from networkx import (
    DiGraph,
    Graph,
    NetworkXNoPath,
    NodeNotFound,
    all_simple_paths,
    shortest_path,
)
from tqdm import tqdm

from spyglass.utils import logger
from spyglass.utils.database_settings import table_is_shared
from spyglass.utils.dj_helper_fn import PERIPHERAL_TABLES  # is_nonempty,
from spyglass.utils.dj_helper_fn import (
    ensure_names,
    fuzzy_get,
    is_trivially_true,
    unique_dicts,
)


def dj_topo_sort(graph: DiGraph) -> List[str]:
    """Topologically sort graph.

    Uses datajoint's topo_sort if available, otherwise uses networkx's
    topological_sort, combined with datajoint's unite_master_parts.

    NOTE: This ordering will impact _hash_upstream, but usage should be
    consistent before/after a no-transaction populate.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph to sort

    Returns
    -------
    List[str]
        List of table names in topological order
    """
    try:  # Datajoint 0.14.2+ uses topo_sort instead of unite_master_parts
        from datajoint.dependencies import topo_sort

        return topo_sort(graph)
    except ImportError:
        from datajoint.dependencies import unite_master_parts
        from networkx.algorithms.dag import topological_sort

        return unite_master_parts(list(topological_sort(graph)))


# Node attributes this module attaches while cascading. Presence of any one
# marks a node as reached by the cascade, see `AbstractGraph.included_tables`.
CASCADE_NODE_KEYS = frozenset({"restr", "restr_list", "files"})


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
    path: List of table names to traverse in the graph, optionally set by
        child classes. Used in TableChain.
    """

    # When False, `_bridge_restr` always derives the next restriction by
    # joining, skipping any short-circuit. Tests flip this to assert that the
    # optimized and unoptimized cascades yield identical restrictions.
    _fast_bridge = True

    def __init__(
        self,
        seed_table: Table,
        verbose: bool = False,
        graph: DiGraph = None,
        **kwargs,
    ):
        """Initialize graph and connection.

        Parameters
        ----------
        seed_table : Table
            Table to use to establish connection and graph
        verbose : bool, optional
            Whether to print verbose output. Default False
        graph : DiGraph, optional
            Dependency graph to build from instead of loading a fresh one. Used
            to share one load across the per-leaf graphs of a multi-leaf
            cascade. Copied, and any cascade data on it is discarded, so the
            new graph's restrictions remain its own.
        """
        self.seed_table = seed_table
        self.connection = seed_table.connection

        if graph is None:
            # `force=False`: registering a schema clears the dependency graph,
            # so a reload still happens whenever one is genuinely needed.
            seed_table.connection.dependencies.load(force=False)
            graph = seed_table.connection.dependencies

        # `copy` rather than `deepcopy`: it gives fresh node attribute dicts,
        # which is all that is needed to keep this graph's restrictions its
        # own, without duplicating every attribute value. It also drops the
        # connection, which cannot be deep-copied.
        self.graph = graph.copy()
        # `copy` carries no custom attributes, so the copy reports itself as
        # unloaded. `ancestors`/`descendants` call `load(force=False)`, which
        # would then try to query through the connection the copy does not
        # have. The copied data is loaded, so say so.
        self.graph._loaded = True
        for _, node in self.graph.nodes(data=True):
            for key in CASCADE_NODE_KEYS:  # Inherit structure, not restrictions
                node.pop(key, None)

        self.verbose = verbose
        self.debug_bridge = False  # see `_bridge_result`, costly when set
        self.max_flat_rows = 5_000  # see `_flatten_restr`
        self.skip_external = True
        self.spawned_schemas = set()
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
        table = ensure_names(table)
        if isinstance(table, str):
            return to_camel_case(table.split(".")[-1].strip("`"))
        if isinstance(table, Iterable) and not isinstance(
            table, (Table, TableMeta)
        ):
            return [self._camel(t) for t in table]

    # ------------------------------ Graph Nodes ------------------------------

    @cached_property
    def undirect_graph(self) -> Graph:
        """Get an undirected copy of the dependency graph, structure only.

        Only `TableChain` needs this, and building it eagerly copied every
        node's data for every graph. Once a cascade has run that data includes
        FreeTable objects, which hold a connection and must not be copied.
        Callers only ever use names and edges.
        """
        undirected = Graph()
        undirected.add_nodes_from(self.graph.nodes())
        undirected.add_edges_from(self.graph.edges())
        return undirected

    def _get_node(self, table: Union[str, Table]):
        """Get node from graph, spawning unimported schemas as needed.

        Nodes of tables outside spyglass are either absent from the graph or
        present without data (i.e., children of imported tables). Either case
        means the schema was never imported, so attempt to spawn it before
        giving up. Relevant when `skip_external` is False, where the cascade is
        expected to reach non-spyglass tables.
        """
        table = ensure_names(table)
        if node := self.graph.nodes.get(table):
            return node

        try:
            self._spawn_virtual_module(table)
        except dj.errors.DataJointError:
            pass  # Schema does not exist, raise below

        if not (node := self.graph.nodes.get(table)):
            raise ValueError(
                f"Table {table} not found in graph."
                + "\n\tPlease import this table and rerun"
            )
        return node

    def _set_node(self, table, attr: str = "ft", value: Any = None):
        """Set attribute on node. General helper for various attributes."""
        table = ensure_names(table)
        _ = self._get_node(table)  # Ensure node exists
        self.graph.nodes[table][attr] = value

    def _get_edge(self, child: str, parent: str) -> Tuple[bool, Dict[str, str]]:
        """Get edge data between child and parent.

        Used as a fallback for _bridge_restr. Required for Maser/Part links to
        temporarily flip direction.

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            Tuple of a direction flag and the edge data. The flag is False when
            the arguments are ordered as named, meaning the graph holds an edge
            parent -> child, and True when they are swapped. `_bridge_restr`
            maps this flag directly onto a cascade direction.
        """
        child = ensure_names(child)
        parent = ensure_names(parent)

        if edge := self.graph.get_edge_data(parent, child):
            return False, edge
        elif edge := self.graph.get_edge_data(child, parent):
            return True, edge

        # Handle alias nodes. `shortest_path` doesn't work with aliases
        # cutoff=2 bounds an otherwise exponential walk: only paths of up to 3
        # nodes are accepted below, so longer ones need not be enumerated.
        p1 = all_simple_paths(self.graph, child, parent, cutoff=2)
        p2 = all_simple_paths(self.graph, parent, child, cutoff=2)
        paths = [p for p in iter_chain(p1, p2)]  # list for error handling
        for path in paths:  # Ignore long and non-alias paths
            if len(path) > 3 or (len(path) > 2 and not path[1].isnumeric()):
                continue
            return self._get_edge(path[0], path[1])

        raise ValueError(f"{child} -> {parent} not direct path: {paths}")

    def _get_restr(self, table):
        """Get restriction from graph node."""
        if self._get_node(ensure_names(table)).get("restr") is None:
            restr_list = self._get_restr_list(table)
            if not restr_list:
                return None
            ft = self._get_ft(table)
            self._set_node(table, "restr", self._combine_restr(ft, restr_list))
        return self._get_node(ensure_names(table)).get("restr")

    @staticmethod
    def _combine_restr(
        ft: FreeTable, restr_list: List
    ) -> Union[str, QueryExpression]:
        """OR together the restrictions accumulated on one node.

        When every item is a plain condition, `make_condition` combines them
        into a single condition string. Restricting the table by the list
        instead would express the same union as a derived table, which then
        nests into every later use of it.

        Parameters
        ----------
        ft : FreeTable
            The table the restrictions apply to.
        restr_list : List
            Restrictions to combine, in the order they were added.

        Returns
        -------
        str | QueryExpression
            The union of the given restrictions.
        """
        if all(isinstance(r, str) for r in restr_list):
            return make_condition(ft, list(restr_list), set())
        return AbstractGraph._coerce_to_condition(ft, ft & restr_list)

    def _get_restr_list(self, table):
        """Get restriction list from graph node."""
        return self._get_node(ensure_names(table)).get("restr_list", [])

    @staticmethod
    def _coerce_to_condition(
        ft: FreeTable, r: Any
    ) -> Union[str, QueryExpression]:
        """Coerce restriction to a valid condition.

        If r is a QueryExpression, project to primary key to keep relational.
        This saves on database requests while propagating restrictions.
        Otherwise, returns a valid restriction string or condition.

        Parameters
        ----------
        ft : FreeTable
            The FreeTable to apply the restriction to.
        r : Any
            The restriction to apply. Can be a string, dict, list, or
            QueryExpression.

        Returns
        -------
        str | QueryExpression
            The restriction as a string or QueryExpression.
        """

        if isinstance(r, str):
            return r
        if isinstance(r, QueryExpression):
            return r.proj(*ft.primary_key)  # keep relational

        # dict/list → condition (fallback)
        return make_condition(ft, r, set())

    def _warn_if_trivially_true(self, table, restriction) -> None:
        """Warn when a shared table is about to be restricted to everything.

        Restrictions here are combined with OR, so one that matches every row
        makes its table's restriction the whole table, and the cascade then
        carries that breadth outward. Arriving at a shared table it means
        either a whole-table restriction was logged upstream or a bridge
        produced one, and in both cases the result reaches data the caller
        never asked for.

        A table under a user's own prefix is left alone: exporting all of it
        can be the intent.

        Parameters
        ----------
        table : str
            Table the restriction is being set on.
        restriction : str | QueryExpression
            The restriction, already coerced to a condition.
        """
        if not is_trivially_true(restriction) or not table_is_shared(table):
            return
        self._log_truncate(  # verbose graphs say where it came from
            f"Whole-table restriction on {self._camel(table)}"
        )
        logger.warning(
            f"Restriction on shared table {ensure_names(table)} matches every "
            "row. Whatever depends on this graph -- an export, a delete -- "
            "will cover the whole table."
        )

    def _set_restr(
        self, table, restriction, replace=False
    ) -> Union[str, QueryExpression]:
        """
        Add restriction to graph node. If one exists, merge with new.

        Parameters
        ----------
        table : str
            Table name
        restriction : str | QueryExpression
            Restriction to log to node
        replace : bool, optional
            Whether to replace existing restriction. Default False will combine
            with OR logic.

        Returns
        -------
        str | QueryExpression
            The resulting restriction after addition/merging.
        """
        ft = self._get_ft(table)
        restriction = self._coerce_to_condition(ft, restriction)
        self._warn_if_trivially_true(table, restriction)
        existing = self._get_restr(table)

        if (not existing) or replace:
            self._set_node(table, "restr_list", [restriction])
            self._set_node(table, "restr", restriction)
            return restriction

        # Merge restrictions. Convergent paths deliver the same condition to a
        # shared ancestor repeatedly; duplicates widen the OR-list without
        # widening what it selects.
        restr_list = self._get_restr_list(table)
        if restriction not in restr_list:
            restr_list = restr_list + [restriction]
        self._set_node(table, "restr_list", restr_list)
        # restriction = self._coerce_to_condition(ft, ft & restr_list)
        self._set_node(
            table, "restr", None
        )  # Placeholder to avoid redundant coercion
        return restriction

    @lru_cache(maxsize=128)
    def _get_ft_with_restr(self, table, restr):
        """Get FreeTable from graph node with restriction applied.

        This helper method is cached to avoid redundant FreeTable creation while
        ensuring that any updated restrictions are applied correctly.
        """
        if not (ft := self._get_node(table).get("ft")):
            ft = FreeTable(self.connection, table)
            self._set_node(table, "ft", ft)

        return ft & self._coerce_to_condition(ft, restr)

    def _get_ft(self, table, with_restr=False, warn=True):
        """Get FreeTable from graph node. If one doesn't exist, create it."""
        table = ensure_names(table)
        if with_restr:
            if not (restr := self._get_restr(table) or False):
                if warn:
                    self._log_truncate(f"No restr for {self._camel(table)}")
        else:
            restr = True

        return self._get_ft_with_restr(table, restr)

    @lru_cache(maxsize=1024)
    def _table_is_nonempty(self, table) -> bool:
        """Whether an unrestricted table has any rows.

        Cached because the answer is invariant for the life of the graph and
        would otherwise be re-queried once per edge arriving at the table.
        """
        return bool(self._get_ft(table))

    def _has_out_prefix(self, table):
        return not table_is_shared(table)

    def _spawn_virtual_module(self, table):
        """Add the tables of a table's schema to the graph, if not imported.

        Spawning registers the schema on the connection, which is a
        process-wide effect that outlives this graph. A cascade over many
        leaves builds one graph per leaf, so without checking the connection
        first, every one of them would repeat the spawn and its log line for
        the same schema.

        Parameters
        ----------
        table : str
            Full table name, used to determine the schema to spawn.

        Raises
        ------
        DataJointError
            If the schema does not exist on the database server.
        """
        schema = table.split(".")[0].strip("`")
        if schema in self.spawned_schemas:  # Already merged into this graph
            return
        self.spawned_schemas.add(schema)

        if schema in self.connection.schemas:
            # Registered already, by an import or an earlier graph's spawn, so
            # its tables are in the connection's dependency graph. This graph's
            # copy may predate that, so still merge -- but quietly.
            v_graph = self.connection.dependencies
            v_graph.load(force=False)
        else:
            logger.warning(f"Spawning tables for {schema}")
            vm = VirtualModule(f"RestrGraph_{schema}", schema)
            v_graph = vm.schema.connection.dependencies
            # Registering the spawned schema clears the dependency graph, so a
            # reload is only skipped when the graph already reflects this
            # schema.
            v_graph.load(force=False)

        self.graph.add_nodes_from(v_graph.nodes(data=True))
        self.graph.add_edges_from(v_graph.edges(data=True))

    @lru_cache(maxsize=1024)
    def _is_out(self, table, warn=True):
        """Check if table is outside of spyglass."""
        table = ensure_names(table)
        if table.isnumeric():  # if alias node, determine status from child
            children = list(self.graph.children(table))
            if len(children) > 1:
                raise ValueError(f"Alias has multiple connections: {table}")
            if children[0].isnumeric():
                raise ValueError(f"Alias of alias, should not happen: {table}")
            return self._is_out(children[0])

        # If already in imported, return
        # Reverts #1356: was `table in self.graph.nodes`, now `get`
        #   - Present nodes may be children of imported, with no data
        #   - Only imported tables have data retrieved by `get`
        if self.graph.nodes.get(table):
            return False

        # If within spyglass, attempt spawn
        try:
            self._spawn_virtual_module(table)
        except dj.errors.DataJointError:
            if warn:
                logger.warning(f"Skipping unimported: {table}")
            return True

        # If spawn successful, return
        if self.graph.nodes.get(table):
            return False

        if warn:
            logger.warning(f"Skipping unimported: {table}")  # pragma: no cover
        return True

    def enforce_restr_strings(self):
        """Ensure all restrictions are strings.

        Converts any non-string restrictions to string conditions.
        """
        for table in self._restricted_nodes():
            restr = self._get_restr(table)
            if not restr or isinstance(restr, str):
                continue
            ft = self._get_ft(table)
            new_restr = make_condition(ft, (ft & restr).fetch("KEY"), set())
            self._set_node(table, "restr", new_restr)

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
        if self.skip_external and (
            self._is_out(table2) or self._is_out(table1)
        ):
            return ["False"]  # Stop cascade if outside, see #1002

        if not all([direction, attr_map]):
            dir_bool, edge = self._get_edge(table1, table2)
            direction = "up" if dir_bool else "down"
            attr_map = edge.get("attr_map")

        # May return empty table if outside imported and outside spyglass
        ft1 = self._get_ft(table1) & restr
        ft2 = self._get_ft(table2)

        path = f"{self._camel(table1)} -> {self._camel(table2)}"

        # `ft1` emptiness is checked once per node by the caller, rather than
        # here, where it would be repeated for every outgoing edge.
        if not self._table_is_nonempty(table2):
            self._log_truncate(f"Bridge Link: {path}: result EMPTY INPUT")
            return ["False"]

        if bool(set(attr_map.values()) - set(ft1.heading.names)):
            attr_map = {v: k for k, v in attr_map.items()}  # reverse

        if self._fast_bridge and self._can_copy_restr(
            table1, table2, restr, attr_map, ft1, ft2
        ):
            self._log_truncate(f"Bridge Copy: {path}")
            return restr

        ret = ft2 & (ft1.proj(**attr_map))

        if self.verbose:  # For debugging. Not required for typical use.
            self._log_truncate(
                f"Bridge Link: {path}{self._bridge_result(ft2, ret)}"
            )
            logger.debug(ret)

        return ret

    def _flatten_restr(self, table, restriction) -> Tuple[Any, bool]:
        """Replace a derived restriction with the literal keys it selects.

        Each bridge restricts a table by a projection of the previous one, so
        an un-flattened cascade carries one nested subquery per hop. Nothing
        executes while the expression is only being built, but every evaluation
        of it -- an emptiness check, a fetch, the next bridge -- pays for the
        whole nested chain, and that cost climbs steeply with depth. Fetching
        the keys once and restating them as a condition keeps each hop flat.

        The fetch is not extra work: the caller has to know whether the
        restriction selects anything, which means evaluating it either way.

        Parameters
        ----------
        table : str
            Table the restriction applies to.
        restriction : Any
            Restriction to flatten.

        Returns
        -------
        Tuple[Any, bool]
            The restriction to propagate, flattened where worthwhile, and
            whether it selects any rows.
        """
        ft_base = self._get_ft(table)
        # A bridge result carries the target's secondary attributes too, and
        # restricting by those is not a valid join. Project as `_set_restr`
        # would, but leave `restriction` itself untouched so that declining to
        # flatten returns exactly what the caller passed in.
        coerced = self._coerce_to_condition(ft_base, restriction)
        ft = ft_base & coerced

        if not (self._fast_bridge and isinstance(coerced, QueryExpression)):
            return restriction, bool(ft)

        # One row past the cap distinguishes "at the cap" from "over" it
        keys = ft.fetch("KEY", limit=self.max_flat_rows + 1)

        if len(keys) > self.max_flat_rows:
            # Restating this many keys would trade a nested query for an
            # unwieldy literal one
            return restriction, True
        if not len(keys):
            return restriction, False

        return make_condition(ft_base, list(keys), set()), True

    def _can_copy_restr(
        self, table1, table2, restr, attr_map, ft1, ft2
    ) -> bool:
        """Whether table2's restriction is table1's verbatim, with no join.

        Adapted from the 'copy the restriction' rule datajoint 2.0 applies when
        an edge renames nothing and the restriction only touches the linking
        columns. Each hop that takes this path avoids nesting another subquery
        inside the restriction, which is what makes deep cascades expensive.

        Only valid from parent to child. Every child row references an existing
        parent row, so a child row satisfies the condition on the linking
        columns exactly when its parent does. The reverse does not hold: a
        parent row satisfying the condition need not have any child row, so
        copying upward would reach rows the join excludes.

        Parameters
        ----------
        table1, table2 : str
            Source and target table names. The restriction is derived for
            table2 from table1.
        restr : Any
            Restriction applied to table1. Only plain string conditions can be
            copied; anything else may carry its own join semantics.
        attr_map : dict
            Mapping of table2 attribute names to table1 attribute names, as
            oriented by the caller.
        ft1, ft2 : FreeTable
            Free tables for table1 (restricted) and table2 (unrestricted).

        Returns
        -------
        bool
            True when the semijoin reduces to the condition itself.
        """
        if not self._fast_bridge or not isinstance(restr, str) or not attr_map:
            return False

        if any(k != v for k, v in attr_map.items()):
            return False  # renamed across the edge

        if not self.graph.get_edge_data(table1, table2):
            return False  # table2 is not table1's child

        # The semijoin matches on every attribute the projection and the target
        # share, not only the mapped ones. If the projection carries extra
        # names, it is stricter than the condition alone.
        projected = set(ft1.primary_key) | set(attr_map)
        if projected & set(ft2.heading.names) != set(attr_map):
            return False

        restr_attrs = set()
        make_condition(ft1, restr, restr_attrs)

        return bool(restr_attrs) and restr_attrs <= set(attr_map)

    def _bridge_result(self, ft2, ret) -> str:
        """Describe a bridge result for logging.

        Says nothing unless `debug_bridge` is set. Every description costs at
        least one evaluation of the derived restriction, and distinguishing a
        full match from a partial one additionally needs an anti-join over the
        whole unrestricted target -- per edge, for a log line. The caller
        evaluates the restriction once per node regardless, so a quiet bridge
        log costs nothing.

        Parameters
        ----------
        ft2 : FreeTable
            The unrestricted target of the bridge.
        ret : QueryExpression
            The restriction derived for that target.

        Returns
        -------
        str
            Empty, or a description prefixed with `: result `.
        """
        if not self.debug_bridge:
            return ""
        if not bool(ret):
            return ": result EMPTY"
        return ": result " + (
            "FULL" if not bool(ft2 - ret.proj()) else "partial"
        )

    def _get_adjacent_path_item(
        self, table: str, direction: Direction = Direction.UP
    ) -> str:
        """Get adjacent path item in the graph.

        Used to get the next table in the path for a given direction.

        Parameters
        ----------
        table : str
            Table name
        direction : Direction, optional
            Direction to cascade. Default 'up'

        Returns
        -------
        str
            Name of the next table in the path or empty string if not found.
        """
        null_return = {table: dict()}  # parent func treats as dead end

        path = getattr(self, "path", [])
        if table not in path:  # if path is empty or table not in path
            return null_return  # pragma: no cover

        idx = path.index(table)
        is_up = direction == Direction.UP
        next_idx = idx - 1 if is_up else idx + 1

        if next_idx in [-1, len(path)]:  # Out of bounds
            return null_return

        next_tbl = path[next_idx]

        if next_tbl.isnumeric():  # Skip alias nodes
            next_next = next_idx - 1 if is_up else next_idx + 1
            table = next_tbl  # for alias, want edge from alias to subsequent
            next_tbl = path[next_next]
        if next_tbl.isnumeric():
            raise ValueError(  # pragma: no cover
                f"Multiple sequential alias nodes found in path {path}. "
                + "This should not happen. Please report this issue."
            )

        try:
            edge = self.graph.edges[table, next_tbl]
        except KeyError:  # if shortest path is not direct
            edge = self.graph.edges[next_tbl, table]

        return {next_tbl: edge}

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

        # Evaluated once per node here rather than once per outgoing edge
        # inside `_bridge_restr`. Doubles as the emptiness check: a restriction
        # selecting nothing yields an empty bridge on every edge.
        restriction, nonempty = self._flatten_restr(table, restriction)

        restriction = self._set_restr(table, restriction, replace=replace)
        self.visited.add(table)

        if not nonempty:
            self._log_truncate(f"Empty restr : {self._camel(table)}")
            return

        if getattr(self, "found_path", None):  # * Avoid refactor #1356
            # * Ideally, would only grab path once
            # Workaround to avoid a class-inheritance refactor
            next_tables = self._get_adjacent_path_item(table, direction)
            next_func = None  # Won't be called bc numeric in path raises
        else:
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
                next_table in self.no_visit  # Subclasses can set this
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

            if next_table in self.visited:
                # check if new restriction contains entries not in existing restriction
                # if not, skip cascade to avoid redundant work
                if not bool(
                    self._get_ft_with_restr(next_table, next_restr)
                    - self._get_restr_list(next_table)
                ):
                    self._log_truncate(
                        f"Already cascaded: {self._camel(next_table)}"
                    )
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
            for node in ensure_names(nodes)
            if not self._is_out(node, warn=False)
        ]
        graph = self.graph.subgraph(nodes) if subgraph else self.graph
        ordered = dj_topo_sort(graph)
        if reverse:
            ordered.reverse()
        return [n for n in ordered if n in nodes]

    @property
    def all_ft(self):
        """Get restricted FreeTables from all visited nodes.

        Topological sort logic adopted from datajoint.diagram.
        """
        self.cascade(warn=False)
        nodes = [n for n in self.included_tables if not n.isnumeric()]
        return [
            self._get_ft(table, with_restr=True, warn=False)
            for table in self._topo_sort(nodes, subgraph=True, reverse=False)
        ]

    @property
    def restr_analysis_file_linked_ft(self):
        self.cascade(warn=False)
        valid_tables = self.analysis_file_tbl.children()
        nodes = [
            n
            for n in self.included_tables
            if not n.isnumeric() and n in valid_tables
        ]
        return [
            self._get_ft(table, with_restr=True, warn=False)
            for table in self._topo_sort(nodes, subgraph=True, reverse=False)
        ]

    @property
    def restr_ft(self):
        """Get non-empty restricted FreeTables from all visited nodes."""
        return [ft for ft in self.all_ft if bool(ft)]

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

        return fts if return_empty else [ft for ft in fts if bool(ft)]

    @property
    def as_dict(self) -> List[Dict[str, str]]:
        """Return as a list of dictionaries of table_name: restriction"""
        # `warn=False` as with the other accessors: reading a cascaded graph is
        # routine, and callers read this repeatedly.
        self.cascade(warn=False)
        return [
            {"table_name": table, "restriction": self._get_restr(table)}
            for table in self.included_tables
            if self._get_restr(table)
        ]

    def _restricted_nodes(self) -> Set[str]:
        """Get the tables carrying restrictions or files set by this module.

        Unlike `included_tables`, does not require the graph to have cascaded,
        so it is usable while restrictions are still being assembled.

        Returns
        -------
        Set[str]
            Names of tables this module has attached cascade data to.
        """
        return {
            table
            for table, node in self.graph.nodes.items()
            if not CASCADE_NODE_KEYS.isdisjoint(node)
        }

    @property
    def included_tables(self) -> Set[str]:
        """Get the tables a cascade reached, as those carrying cascade data.

        Membership cannot be inferred from node truthiness. DataJoint's
        `Dependencies.load` gives every table of every loaded schema a
        `primary_key` node attribute, so a truthiness check returns the whole
        database rather than the cascaded subset -- and callers then build a
        FreeTable and run an existence query per table.

        Keyed on the attributes this module writes rather than on `visited`, so
        that tables restricted by graph addition (`_graph_union`) are included
        despite never having been traversed.
        """
        if not self.cascaded:
            return set()
        return {
            table
            for table, node in self.graph.nodes.items()
            if not CASCADE_NODE_KEYS.isdisjoint(node)
        }


class RestrGraph(AbstractGraph):
    def __init__(
        self,
        seed_table: Table,
        leaves: List[Dict[str, str]] = None,
        destinations: List[str] = None,
        direction: Direction = "up",
        include_files: bool = False,
        cascade: bool = False,
        verbose: bool = False,
        skip_external: bool = True,
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
        include_files : bool, optional
            Default False. If True, add 'files' list to nodes in graph, add
            externals tables. For use in export, not database-state hashing, or
            long-distance restrictions.
        cascade : bool, optional
            Whether to cascade restrictions up the graph on initialization.
            Default False
        verbose : bool, optional
            Whether to print verbose output. Default False
        skip_external : bool, optional
            Whether to skip tables outside of spyglass during cascade. Default
            True. Set to False to continue the cascade into non-spyglass tables.
        **kwargs : dict
            Passed to `AbstractGraph`, notably `graph` to build from an already
            loaded dependency graph rather than reloading one.
        """
        super().__init__(seed_table, verbose=verbose, **kwargs)
        self.include_files = include_files
        self.skip_external = skip_external

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

    @property
    def hash(self):
        """Return hash of all visited nodes."""
        initial = hash_md5(b"")
        for table in self.all_ft:
            # for row in table.fetch(as_dict=True):
            for row in table:
                initial.update(key_hash(row).encode("utf-8"))
        return initial.hexdigest()

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

        if len(to_visit) == 0:
            if warn:
                self._log_truncate("No new leaves to cascade")
            self.cascaded = True
            return

        if len(to_visit) == 1:
            table = to_visit.pop()
            restr = self._get_restr(table)
            self._log_truncate(
                f"Start  {direction:<4}: {self._camel(table)}, {restr}"
            )
            self.cascade1(table, restr, direction=direction, replace=False)

        else:
            # Run the cascade of each leaf separately to avoid order dependence
            # Then combine results with __add__
            self.cascaded = True  # set now so can be added with (+)
            cascaded_leaves = []
            # Chained forward so each leaf inherits the FreeTables built by the
            # ones before it, saving a heading query per table per leaf.
            # Restrictions do not carry: they are stripped on construction.
            shared_graph = self.graph
            for table in tqdm(
                to_visit,
                desc="RestrGraph: cascading restrictions",
                total=len(to_visit),
                disable=not (show_progress or self.verbose),
            ):
                leaf_graph = RestrGraph(
                    seed_table=self.seed_table,
                    graph=shared_graph,  # Share one load across the leaves
                    leaves=[
                        {
                            "table_name": table,
                            "restriction": self._get_restr(table),
                        }
                    ],
                    include_files=False,  # only need after merging the leaf graphs
                    direction=direction,
                    verbose=self.verbose,
                    cascade=True,
                    skip_external=self.skip_external,
                )
                cascaded_leaves.append(leaf_graph)
                shared_graph = leaf_graph.graph
            logger.debug("adding cascaded leaves")
            self = self + cascaded_leaves

        self.cascaded = True  # Mark here so next step can use `restr_ft`
        self.cascade_files()  # Otherwise attempts to re-cascade, recursively

    # ---------------------------- Graph Intersection ---------------------------
    def _graph_intersect(self, other: "RestrGraph") -> "RestrGraph":
        """Returns intersection of two RestrGraphs.

        Only tables present in both graphs are retained, with restrictions
        combined with AND logic.

        Parameters
        ----------
        other : RestrGraph
            Another RestrGraph to intersect with self.

        Returns
        -------
        RestrGraph

            New un-cascaded RestrGraph representing the intersection of self and other.
        """
        if self.cascaded:
            graph_1_tables = [
                tbl for tbl in self.included_tables if self._get_restr(tbl)
            ]
        else:
            # If not cascaded, apply intersection to leaves only
            graph_1_tables = self.leaves

        other.enforce_restr_strings()
        graph_2_tables = [
            tbl for tbl in other.included_tables if other._get_restr(tbl)
        ]
        if not graph_1_tables or not graph_2_tables:
            # If either graph has no tables with restrictions, return empty RestrGraph
            return RestrGraph(
                seed_table=self.seed_table,
                leaves=[],
                include_files=self.include_files,
                cascade=False,
                verbose=self.verbose,
                skip_external=self.skip_external,
            )

        table_dicts = []

        for table in graph_1_tables:
            if table not in graph_2_tables:
                continue
            ft = self._get_ft(table)
            intersect_restriction = ft & dj.AndList(
                [
                    self._get_restr(table),
                    other._get_restr(table),
                ]
            )

            table_dicts.append(
                {
                    "table_name": table,
                    "restriction": make_condition(
                        ft, intersect_restriction.fetch("KEY"), set()
                    ),
                }
            )
        return RestrGraph(
            seed_table=self.seed_table,
            leaves=table_dicts,
            include_files=self.include_files,
            cascade=False,
            verbose=self.verbose,
            # Carried over: an intersection that silently reverted to the
            # default would stop at non-spyglass tables the inputs reached.
            skip_external=self.skip_external,
        )

    def __and__(self, other: "RestrGraph") -> "RestrGraph":
        """Return intersection of two RestrGraphs."""
        if not isinstance(other, RestrGraph):
            raise TypeError(f"Cannot AND RestrGraph with {type(other)}")
        return self._graph_intersect(other)

    def whitelist(self, other: "RestrGraph") -> "RestrGraph":
        """
        Return a new RestrGraph restricted to the intersect of both graphs.

        This is a convenience alias for the bitwise AND operator
        (``self & other``) and has identical semantics and return value.

        Parameters
        ----------
        other : RestrGraph
            Another RestrGraph whose restrictions will be intersected with
            this one.
        Returns
        -------
        RestrGraph
            A new RestrGraph representing the intersection of ``self`` and
            ``other``.
        """
        return self & other

    # ----------------------------- Graph Addition -----------------------------

    def _graph_union(self, other: "RestrGraph") -> "RestrGraph":
        if not (self.cascaded and other.cascaded):
            raise ValueError("Both RestrGraphs must be cascaded before union.")
        other_dicts = other.as_dict

        for dict in other_dicts:
            table = dict["table_name"]
            new_restr_list = self._get_restr_list(
                table
            ) + other._get_restr_list(table)

            self._set_node(table, "restr_list", new_restr_list)
            ft = self._get_ft(table)
            restriction = self._coerce_to_condition(ft, ft & new_restr_list)
            self._set_node(table, "restr", restriction)
        return self

    def _graph_union_list(self, other: "List") -> "RestrGraph":
        if not all(isinstance(x, RestrGraph) for x in other):
            raise TypeError("All items in list must be RestrGraph objects.")
        if not self.cascaded and all([x.cascaded for x in other]):
            raise ValueError("All RestrGraphs must be cascaded before union.")

        other_dicts_list = [x.as_dict for x in other]
        all_table_names = []
        for other_dicts in other_dicts_list:
            all_table_names += [d["table_name"] for d in other_dicts]
        all_table_names = set(all_table_names)
        for table in all_table_names:
            new_restr_list = self._get_restr_list(table)
            for other_graph in other:
                new_restr_list += other_graph._get_restr_list(table)
            self._set_node(table, "restr_list", new_restr_list)
            ft = self._get_ft(table)
            restriction = self._coerce_to_condition(ft, ft & new_restr_list)
            self._set_node(table, "restr", restriction)
        return self

    def __add__(self, other: "RestrGraph") -> "RestrGraph":
        """Return union of two RestrGraphs."""
        if isinstance(other, RestrGraph):
            return self._graph_union(other)

        elif isinstance(other, List):
            return self._graph_union_list(other)

        raise TypeError(f"Cannot OR RestrGraph with {type(other)}")

    # ----------------------------- File Handling -----------------------------

    @property
    def analysis_file_tbl(self) -> Table:
        """Return the analysis file table. Avoids circular import."""
        from spyglass.common import AnalysisNwbfile

        return AnalysisNwbfile()

    @property
    def file_externals(self):
        from spyglass.common.common_nwbfile import schema

        return schema.external

    def cascade_files(self):
        """Add file lists as to nodes in graph.

        1. For any table fk'ing AnalysisNwbfile, add files to node.
        2. For both raw and analysis files, add restrictions to externals tables
            Paths come from the externals tables themselves, see
            `_external_paths`, so no file is touched on disk.
        """
        if not self.include_files:  # Skip if not needed
            return  # if _hash_upstream, may cause 'missing node' error

        analysis_pk = self.analysis_file_tbl.primary_key
        for ft in self.restr_analysis_file_linked_ft:
            if not set(analysis_pk).issubset(ft.heading.names):
                continue
            files = list(ft.fetch(*analysis_pk))
            if len(files):
                self._set_node(ft, "files", files)

        raw_ext = self.file_externals["raw"].full_table_name
        analysis_ext = self.file_externals["analysis"].full_table_name

        if not {raw_ext, analysis_ext}.issubset(self.graph.nodes):
            return  # Skip if externals not in graph

        def set_external(external, file_list=None):
            """Set restriction on external table."""
            if not file_list:
                return
            restr = (
                f"filepath in {tuple(file_list)}"
                if len(file_list) > 1
                else f"filepath = '{file_list[0]}'"
            )
            tbl = raw_ext if external == "raw" else analysis_ext
            self._set_restr(tbl, restr)

        set_external(
            "analysis",
            self._external_paths(
                self.analysis_file_tbl.full_table_name,
                "analysis_file_abs_path",
                "analysis",
            ),
        )
        set_external(
            "raw",
            self._external_paths(
                "`common_nwbfile`.`nwbfile`", "nwb_file_abs_path", "raw"
            ),
        )

    def _external_paths(
        self, table_name: str, filepath_attr: str, store: str
    ) -> List[str]:
        """Get store-relative paths of the files a restricted table references.

        Reads the paths from the external table, joining on the hash the file
        table stores, rather than fetching the filepath attribute from the file
        table itself. Fetching it makes DataJoint resolve every row to an
        absolute path -- stat-ing, checksumming, and re-staging each file --
        only for the caller to strip the store root back off. The relative path
        wanted here is the value the external table already holds, so none of
        that work is needed, and a file whose contents have drifted from its
        recorded checksum no longer fails the cascade.

        Parameters
        ----------
        table_name : str
            Name of the file table, e.g. `common_nwbfile`.`nwbfile`.
        filepath_attr : str
            Name of that table's filepath attribute, which stores the hash
            keying the external table.
        store : str
            Store name, `raw` or `analysis`.

        Returns
        -------
        List[str]
            Paths relative to the store's root, for the restricted rows.
        """
        ft = self._get_ft(table_name, with_restr=True)
        external = self.file_externals[store]
        return list((external & ft.proj(hash=filepath_attr)).fetch("filepath"))

    @property
    def file_dict(self) -> Dict[str, List[str]]:
        """Return dictionary of analysis files from all visited nodes.

        Included for debugging, to associate files with tables.
        """
        self.cascade(warn=False)
        return {t: self._get_node(t).get("files", []) for t in self.restr_ft}

    def _stored_files(self, as_dict=False):
        """Return dictionary of table names and files.

        Dictionary format is used for debugging and testing. Set format is used
        for hashing and typical use.
        """
        self.cascade(warn=False)

        pairs = [
            (table, file)
            for table in self.included_tables
            for file in self._get_node(table).get("files", [])
        ]
        return dict(pairs) if as_dict else {file for _, file in pairs}

    @property
    def file_paths(self) -> List[str]:
        """Return list of unique analysis files from all visited nodes.

        This covers intermediate analysis files that may not have been fetched
        directly by the user.
        """
        self.cascade(warn=False)

        return [
            self.analysis_file_tbl.get_abs_path(file)
            for file in self._stored_files()
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
        valid joins by default. If no path is found, another search is attempted
        with PERIPHERAL_TABLES included.
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
        banned_tables: List[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize a TableChain object.

        Parameters
        ----------
        parent : Table, optional
            Parent table of the chain. Default None.
        child : Table, optional
            Child table of the chain. Default None.
        direction : Direction, optional
            Direction of the chain. Default 'none'. If both parent and child
            are provided, direction is inferred from the link type.
        search_restr : str, optional
            Restriction to search for in the chain. If provided, the chain will
            search for where this restriction can be applied. Default None,
            expecting this restriction to be passed when invoking `cascade`.
        cascade : bool, optional
            Whether to cascade the restrictions through the chain on
            initialization. Default False.
        banned_tables : List[str], optional
            List of table names to ignore in the graph traversal. Default None.
            If provided, these tables will not be visited during the search.
            Useful for excluding peripheral tables or other unwanted nodes.
        verbose : bool, optional
            Whether to print verbose output. Default False.
        """
        self.parent = ensure_names(parent)
        self.child = ensure_names(child)

        if not self.parent and not self.child:
            raise ValueError("Parent or child table required.")

        seed_table = parent if isinstance(parent, Table) else child
        super().__init__(seed_table=seed_table, verbose=verbose)

        self._ignore_peripheral(except_tables=[self.parent, self.child])
        self._ignore_outside_spy(except_tables=[self.parent, self.child])

        self.no_visit.update(ensure_names(banned_tables) or [])

        self.no_visit.difference_update(set([self.parent, self.child]))

        self.searched_tables = set()
        self.found_path = False
        self.found_restr = False
        self.link_type = None
        self.searched_path = False
        self._link_symbol = " -> "

        self.search_restr = search_restr
        self.direction = Direction(direction)
        if self.parent and self.child and not self.direction:
            self.direction = Direction.DOWN

        self.leaf = None
        if search_restr and not self.parent:  # using `parent` fails on empty
            self.direction = Direction.UP
            self.leaf = self.child
        if search_restr and not self.child:
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
        except_tables = ensure_names(except_tables)
        ignore_tables = set(PERIPHERAL_TABLES) - set(except_tables or [])
        self.no_visit.update(ignore_tables)

    def _ignore_outside_spy(self, except_tables: List[str] = None):
        """Ignore tables not shared on shared prefixes."""
        except_tables = ensure_names(except_tables)
        ignore_tables = set(  # Ignore tables not in shared modules
            [
                t
                for t in self.undirect_graph.nodes
                if t not in except_tables and self._is_out(t, warn=False)
            ]
        )
        self.no_visit.update(ignore_tables)

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
        """Return string representation of path: parent -> {links} -> child."""
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
        """Cascade restriction through graph to search for applicable table."""
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
        """Search parents/children for a match of the provided restriction."""
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

            link = f"{self._camel(table)} -> {self._camel(next_table)}"
            self._log_truncate(f"Search Link: {link}")

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
        search_graph = (  # Copy to ensure orig not modified by no_visit
            self.graph.copy() if directed else self.undirect_graph.copy()
        )

        # Ignore nodes that should not be visited #1353
        search_graph.remove_nodes_from(self.no_visit)

        try:
            path = shortest_path(search_graph, source, target)
        except NetworkXNoPath:
            return None  # No path found, parent func may do undirected search
        except NodeNotFound:
            self.searched_path = True  # No path found, don't search again
            return None  # pragma: no cover

        self._log_truncate(f"Path Found : {path}")
        self.found_path = True

        ignore_nodes = self.graph.nodes - set(path)
        self.no_visit.update(ignore_nodes)

        return path

    @cached_property
    def path(self) -> list:
        """Return list of full table names in chain."""
        if self.searched_path and not self.has_link:
            self._log_truncate("No path found, already searched")
            return None  # pragma: no cover
        if not (self.parent and self.child):
            self._log_truncate("No parent or child set, cannot find path.")
            return None  # pragma: no cover

        path = None
        if path := self.find_path(directed=True):
            self.link_type = "directed"
        elif path := self.find_path(directed=False):
            self.link_type = "undirected"
        else:  # Search with peripheral
            self.no_visit.difference_update(PERIPHERAL_TABLES)
            if path := self.find_path(directed=True):
                self.link_type = "directed w/peripheral"  # pragma: no cover
            elif path := self.find_path(directed=False):
                self.link_type = "undirected w/peripheral"  # pragma: no cover

        if path is None:
            self._log_truncate("No path found")

        self.searched_path = True

        return path

    def cascade(
        self, restriction: str = None, direction: Direction = None, **kwargs
    ):
        """Cascade restriction up or down the chain."""
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
        safe_tbls = [
            t for t in self.path if not t.isnumeric() and not self._is_out(t)
        ]
        if any(self._get_restr(t) is None for t in safe_tbls):
            for table in safe_tbls:
                if table is not start:
                    self._set_restr(table, False, replace=True)

        self.cascaded = True
        return self._get_ft(end, with_restr=True)

    def restrict_by(self, *args, **kwargs) -> None:
        """Cascade passthrough."""
        return self.cascade(*args, **kwargs)
