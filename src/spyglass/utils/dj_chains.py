from collections import OrderedDict
from functools import cached_property
from typing import List, Union

import datajoint as dj
import networkx as nx
from datajoint.expression import QueryExpression
from datajoint.table import Table
from datajoint.utils import get_master, to_camel_case

from spyglass.utils.dj_merge_tables import RESERVED_PRIMARY_KEY as MERGE_PK
from spyglass.utils.logging import logger

# Tables that should be excluded from the undirected graph when finding paths
# to maintain valid joins.
PERIPHERAL_TABLES = [
    "`common_interval`.`interval_list`",
    "`common_nwbfile`.`__analysis_nwbfile_kachery`",
    "`common_nwbfile`.`__nwbfile_kachery`",
    "`common_nwbfile`.`analysis_nwbfile_kachery_selection`",
    "`common_nwbfile`.`analysis_nwbfile_kachery`",
    "`common_nwbfile`.`analysis_nwbfile`",
    "`common_nwbfile`.`kachery_channel`",
    "`common_nwbfile`.`nwbfile_kachery_selection`",
    "`common_nwbfile`.`nwbfile_kachery`",
    "`common_nwbfile`.`nwbfile`",
]


class TableChains:
    """Class for representing chains from parent to Merge table via parts.

    Functions as a plural version of TableChain, allowing a single `join`
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
    join(restriction: str = None)
        Return list of joins for each chain in self.chains.
    """

    def __init__(self, parent, child, connection=None):
        self.parent = parent
        self.child = child
        self.connection = connection or parent.connection
        parts = child.parts(as_objects=True)
        self.part_names = [part.full_table_name for part in parts]
        self.chains = [TableChain(parent, part) for part in parts]
        self.has_link = any([chain.has_link for chain in self.chains])

    def __repr__(self):
        return "\n".join([str(chain) for chain in self.chains])

    def __len__(self):
        return len([c for c in self.chains if c.has_link])

    @property
    def max_len(self):
        """Return length of longest chain."""
        return max([len(chain) for chain in self.chains])

    def __getitem__(self, index: Union[int, str]):
        """Return FreeTable object at index."""
        if isinstance(index, str):
            for i, part in enumerate(self.part_names):
                if index in part:
                    return self.chains[i]
        return self.chains[index]

    def join(self, restriction=None) -> List[QueryExpression]:
        """Return list of joins for each chain in self.chains."""
        restriction = restriction or self.parent.restriction or True
        joins = []
        for chain in self.chains:
            if joined := chain.join(restriction):
                joins.append(joined)
        return joins


class TableChain:
    """Class for representing a chain of tables.

    A chain is a sequence of tables from parent to child identified by
    networkx.shortest_path. Parent -> Merge should use TableChains instead to
    handle multiple paths to the respective parts of the Merge table.

    Attributes
    ----------
    parent : Table
        Parent or origin of chain.
    child : Table
        Child or destination of chain.
    _connection : datajoint.Connection, optional
        Connection to database used to create FreeTable objects. Defaults to
        parent.connection.
    _link_symbol : str
        Symbol used to represent the link between parent and child. Hardcoded
        to " -> ".
    has_link : bool
        Cached attribute to store whether parent is linked to child. False if
        child is not in parent.descendants or nx.NetworkXNoPath is raised by
        nx.shortest_path.
    link_type : str
        'directed' or 'undirected' based on whether path is found with directed
        or undirected graph. None if no path is found.
    graph : nx.DiGraph
        Directed graph of parent's dependencies from datajoint.connection.
    names : List[str]
        List of full table names in chain.
    objects : List[dj.FreeTable]
        List of FreeTable objects for each table in chain.
    attr_maps : List[dict]
        List of attribute maps for each link in chain.
    path : OrderedDict[str, Dict[str, Union[dj.FreeTable,dict]]]
        Dictionary of full table names in chain. Keys are self.names
        Values are a dict of free_table (self.objects) and
        attr_map (dict of new_name: old_name, self.attr_map).

    Methods
    -------
    __str__()
        Return string representation of chain: parent -> child.
    __repr__()
        Return full representation of chain: parent -> {links} -> child.
    __len__()
        Return number of tables in chain.
    __getitem__(index: Union[int, str])
        Return FreeTable object at index, or use substring of table name.
    find_path(directed=True)
        Returns path OrderedDict of full table names in chain. If directed is
        True, uses directed graph. If False, uses undirected graph. Undirected
        excludes PERIPHERAL_TABLES like interval_list, nwbfile, etc. to maintain
        valid joins.
    join(restriction: str = None)
        Return join of tables in chain with restriction applied to parent.
    """

    def __init__(self, parent: Table, child: Table, connection=None):
        self._connection = connection or parent.connection
        self.graph = self._connection.dependencies
        self.graph.load()

        if (  # if child is a merge table
            get_master(child.full_table_name) == ""
            and MERGE_PK in child.heading.names
        ):
            raise TypeError("Child is a merge table. Use TableChains instead.")

        self._link_symbol = " -> "
        self.parent = parent
        self.child = child
        self.link_type = None
        self._searched = False

        if child.full_table_name not in self.graph.nodes:
            logger.warning(
                "Can't find item in graph. Try importing: "
                + f"{child.full_table_name}"
            )
            self._searched = True

    def __str__(self):
        """Return string representation of chain: parent -> child."""
        if not self.has_link:
            return "No link"
        return (
            to_camel_case(self.parent.table_name)
            + self._link_symbol
            + to_camel_case(self.child.table_name)
        )

    def __repr__(self):
        """Return full representation of chain: parent -> {links} -> child."""
        if not self.has_link:
            return "No link"
        return "Chain: " + self._link_symbol.join(
            [t.table_name for t in self.objects]
        )

    def __len__(self):
        """Return number of tables in chain."""
        if not self.has_link:
            return 0
        return len(self.names)

    def __getitem__(self, index: Union[int, str]) -> dj.FreeTable:
        """Return FreeTable object at index."""
        if not self.has_link:
            return None
        if isinstance(index, str):
            for i, name in enumerate(self.names):
                if index in name:
                    return self.objects[i]
        return self.objects[index]

    @property
    def has_link(self) -> bool:
        """Return True if parent is linked to child.

        If not searched, search for path. If searched and no link is found,
        return False. If searched and link is found, return True.
        """
        if not self._searched:
            _ = self.path
        return self.link_type is not None

    def pk_link(self, src, trg, data) -> float:
        """Return 1 if data["primary"] else float("inf").

        Currently unused. Preserved for future debugging. shortest_path accepts
        an option weight callable parameter.
        nx.shortest_path(G, source, target,weight=pk_link)
        """
        return 1 if data["primary"] else float("inf")

    def find_path(self, directed=True) -> OrderedDict:
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
        OrderedDict
            Dictionary of full table names in chain. Keys are full table names.
            Values are free_table (dj.FreeTable representation) and attr_map
            (dict of new_name: old_name). Attribute maps on the table upstream
            of an alias node that can be used in .proj(). Returns None if no
            path is found.

        Ignores numeric table names in paths, which are
        'gaps' or alias nodes in the graph. See datajoint.Diagram._make_graph
        source code for comments on alias nodes.
        """
        source, target = self.parent.full_table_name, self.child.full_table_name
        if not directed:
            self.graph = self.graph.to_undirected()
            self.graph.remove_nodes_from(PERIPHERAL_TABLES)
        try:
            path = nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            self._searched = True
            return None

        ret = OrderedDict()
        prev_table = None
        for i, table in enumerate(path):
            if table.isnumeric():  # get proj() attribute map for alias node
                if not prev_table:
                    raise ValueError("Alias node found without prev table.")
                try:
                    attr_map = self.graph[table][prev_table]["attr_map"]
                except KeyError:  # Why is this only DLCCentroid??
                    attr_map = self.graph[prev_table][table]["attr_map"]
                ret[prev_table]["attr_map"] = attr_map
            else:
                free_table = dj.FreeTable(self._connection, table)
                ret[table] = {"free_table": free_table, "attr_map": {}}
                prev_table = table
        return ret

    @cached_property
    def path(self) -> OrderedDict:
        """Return list of full table names in chain."""
        if self._searched and not self.has_link:
            return None

        link = None
        if link := self.find_path(directed=True):
            self.link_type = "directed"
        elif link := self.find_path(directed=False):
            self.link_type = "undirected"
        self._searched = True

        return link

    @cached_property
    def names(self) -> List[str]:
        """Return list of full table names in chain."""
        if not self.has_link:
            return None
        return list(self.path.keys())

    @cached_property
    def objects(self) -> List[dj.FreeTable]:
        """Return list of FreeTable objects for each table in chain.

        Unused. Preserved for future debugging.
        """
        if not self.has_link:
            return None
        return [v["free_table"] for v in self.path.values()]

    @cached_property
    def attr_maps(self) -> List[dict]:
        """Return list of attribute maps for each table in chain.

        Unused. Preserved for future debugging.
        """
        if not self.has_link:
            return None
        return [v["attr_map"] for v in self.path.values()]

    def join(
        self, restriction: str = None, reverse_order: bool = False
    ) -> dj.expression.QueryExpression:
        """Return join of tables in chain with restriction applied to parent.

        Parameters
        ----------
        restriction : str, optional
            Restriction to apply to first table in the order.
            Defaults to self.parent.restriction.
        reverse_order : bool, optional
            If True, join tables in reverse order. Defaults to False.
        """
        if not self.has_link:
            return None

        restriction = restriction or self.parent.restriction or True
        path = (
            OrderedDict(reversed(self.path.items()))
            if reverse_order
            else self.path
        ).copy()

        _, first_val = path.popitem(last=False)
        join = first_val["free_table"] & restriction
        for i, val in enumerate(path.values()):
            attr_map, free_table = val["attr_map"], val["free_table"]
            try:
                join = (join.proj() * free_table).proj(**attr_map)
            except dj.DataJointError as e:
                attribute = str(e).split("attribute ")[-1]
                logger.error(
                    f"{str(self)} at {free_table.table_name} with {attribute}"
                )
                return None
        return join
