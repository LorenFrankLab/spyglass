from collections import OrderedDict
from functools import cached_property
from typing import List, Union

import datajoint as dj
import networkx as nx
from datajoint.expression import QueryExpression
from datajoint.table import Table
from datajoint.utils import to_camel_case

from spyglass.utils.dj_graph_abs import AbstractGraph, _fuzzy_get
from spyglass.utils.dj_merge_tables import RESERVED_PRIMARY_KEY as MERGE_PK
from spyglass.utils.dj_merge_tables import is_merge_table
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
        return _fuzzy_get(index, self.part_names, self.chains)

    def join(
        self, restriction=None, reverse_order=False
    ) -> List[QueryExpression]:
        """Return list of joins for each chain in self.chains."""
        restriction = restriction or self.parent.restriction or True
        joins = []
        for chain in self.chains:
            if joined := chain.join(restriction, reverse_order=reverse_order):
                joins.append(joined)
        return joins

    def cascade(self, restriction: str = None, direction: str = "down"):
        """Return list of cascades for each chain in self.chains."""
        restriction = restriction or self.parent.restriction or True
        cascades = []
        for chain in self.chains:
            if joined := chain.cascade(restriction, direction):
                cascades.append(joined)
        return cascades


class TableChain(AbstractGraph):
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

    def __init__(
        self,
        parent: Table,
        child: Table,
        verbose: bool = False,
    ):
        if is_merge_table(child):
            raise TypeError("Child is a merge table. Use TableChains instead.")

        super().__init__(seed_table=parent, verbose=verbose)
        _ = self._get_node(child.full_table_name)  # ensure child is in graph

        self._link_symbol = " -> "
        self.parent = parent
        self.child = child
        self.link_type = None
        self._searched = False
        self.undirect_graph = None

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
        return "Chain: " + self._link_symbol.join(self.path)

    def __len__(self):
        """Return number of tables in chain."""
        if not self.has_link:
            return 0
        return len(self.names)

    def __getitem__(self, index: Union[int, str]):
        return _fuzzy_get(index, self.names, self.objects)

    @property
    def has_link(self) -> bool:
        """Return True if parent is linked to child.

        If not searched, search for path. If searched and no link is found,
        return False. If searched and link is found, return True.
        """
        if not self._searched:
            _ = self.path
        return self.link_type is not None

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
            self.undirect_graph = self.graph.to_undirected()
            self.undirect_graph.remove_nodes_from(PERIPHERAL_TABLES)

        search_graph = self.graph if directed else self.undirect_graph

        try:
            path = nx.shortest_path(search_graph, source, target)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            self._searched = True
            return None

        ignore_nodes = self.graph.nodes - set(path)
        self.no_visit.update(ignore_nodes)

        return path

    @cached_property
    def path(self) -> list:
        """Return list of full table names in chain."""
        if self._searched and not self.has_link:
            return None

        path = None
        if path := self.find_path(directed=True):
            self.link_type = "directed"
        elif path := self.find_path(directed=False):
            self.link_type = "undirected"
        self._searched = True

        return path

    @cached_property
    def objects(self) -> List[dj.FreeTable]:
        """Return list of FreeTable objects for each table in chain.

        Unused. Preserved for future debugging.
        """
        if not self.has_link:
            return None
        return [self._get_ft(table, with_restr=False) for table in self.path]

    def cascade(self, restriction: str = None, direction: str = "up"):
        _ = self.path
        if not self.has_link:
            return None
        if direction == "up":
            start, end = self.child, self.parent
        else:
            start, end = self.parent, self.child
        if not self.cascaded:
            self.cascade1(
                table=start.full_table_name,
                restriction=restriction,
                direction=direction,
            )
            self.cascaded = True
        return self._get_ft(end.full_table_name, with_restr=True)

    def join(
        self, restriction: str = None, reverse_order: bool = False
    ) -> dj.expression.QueryExpression:
        if not self.has_link:
            return None

        direction = "down" if reverse_order else "up"
        return self.cascade(restriction, direction)

    def old_join(
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
