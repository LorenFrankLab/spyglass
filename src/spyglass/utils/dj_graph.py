"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from typing import Dict, List, Union

from datajoint import FreeTable
from datajoint.condition import make_condition
from datajoint.table import Table
from tqdm import tqdm

from spyglass.common import AnalysisNwbfile
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import unique_dicts


class RestrGraph:
    def __init__(
        self,
        seed_table: Table,
        table_name: str = None,
        restriction: str = None,
        leaves: List[Dict[str, str]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Use graph to cascade restrictions up from leaves to all ancestors.

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
        verbose : bool, optional
            Whether to print verbose output. Default False
        """

        self.connection = seed_table.connection
        self.graph = seed_table.connection.dependencies
        self.graph.load()

        self.verbose = verbose
        self.cascaded = False
        self.ancestors = set()
        self.visited = set()
        self.leaves = set()
        self.analysis_pk = AnalysisNwbfile().primary_key

        if table_name and restriction:
            self.add_leaf(table_name, restriction)
        if leaves:
            self.add_leaves(leaves, show_progress=verbose)

    def __repr__(self):
        l_str = ",\n\t".join(self.leaves) + "\n" if self.leaves else ""
        processed = "Cascaded" if self.cascaded else "Uncascaded"
        return f"{processed} RestrictionGraph(\n\t{l_str})"

    @property
    def all_ft(self):
        """Get restricted FreeTables from all visited nodes."""
        self.cascade()
        return [self._get_ft(table, with_restr=True) for table in self.visited]

    @property
    def leaf_ft(self):
        """Get restricted FreeTables from graph leaves."""
        return [self._get_ft(table, with_restr=True) for table in self.leaves]

    def _get_node(self, table):
        """Get node from graph."""
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

    def _get_ft(self, table, with_restr=False):
        """Get FreeTable from graph node. If one doesn't exist, create it."""
        table = table if isinstance(table, str) else table.full_table_name
        restr = self._get_restr(table) if with_restr else True
        if ft := self._get_node(table).get("ft"):
            return ft & restr
        ft = FreeTable(self.connection, table)
        self._set_node(table, "ft", ft)
        return ft & restr

    def _get_restr(self, table):
        """Get restriction from graph node."""
        table = table if isinstance(table, str) else table.full_table_name
        return self._get_node(table).get("restr", "False")

    def _get_files(self, table):
        """Get analysis files from graph node."""
        return self._get_node(table).get("files", [])

    def _set_restr(self, table, restriction):
        """Add restriction to graph node. If one exists, merge with new."""
        ft = self._get_ft(table)
        restriction = (  # Convert to condition if list or dict
            make_condition(ft, restriction, set())
            if not isinstance(restriction, str)
            else restriction
        )
        # orig_restr = restriction
        if existing := self._get_restr(table):
            if existing == restriction:
                return
            join = ft & [existing, restriction]
            if len(join) == len(ft & existing):
                return  # restriction is a subset of existing
            restriction = make_condition(
                ft, unique_dicts(join.fetch("KEY", as_dict=True)), set()
            )

        # if table == "`spikesorting_merge`.`spike_sorting_output`":
        #     __import__("pdb").set_trace()

        self._set_node(table, "restr", restriction)

    def get_restr_ft(self, table: Union[int, str]):
        """Get restricted FreeTable from graph node.

        Currently used. May be useful for debugging.

        Parameters
        ----------
        table : Union[int, str]
            Table name or index in visited set
        """
        if isinstance(table, int):
            table = list(self.visited)[table]
        return self._get_ft(table, with_restr=True)

    def _log_truncate(self, log_str, max_len=80):
        """Truncate log lines to max_len and print if verbose."""
        if not self.verbose:
            return
        logger.info(
            log_str[:max_len] + "..." if len(log_str) > max_len else log_str
        )

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

    def cascade_files(self):
        """Set node attribute for analysis files."""
        for table in self.visited:
            ft = self._get_ft(table)
            if not set(self.analysis_pk).issubset(ft.heading.names):
                continue
            files = (ft & self._get_restr(table)).fetch(*self.analysis_pk)
            self._set_node(table, "files", files)

    def cascade1(self, table, restriction):
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

        for parent, data in self.graph.parents(table).items():
            if parent in self.visited:
                continue

            if parent.isnumeric():
                parent, data = self.graph.parents(parent).popitem()

            parent_restr = self._child_to_parent(
                child=table,
                parent=parent,
                restriction=restriction,
                **data,
            )

            self.cascade1(parent, parent_restr)  # Parent set on recursion

    def cascade(self, show_progress=None) -> None:
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
            self.cascade1(table, restr)
        if not self.visited == self.ancestors:
            raise RuntimeError(
                "Cascade: FAIL - incomplete cascade. Please post issue."
            )

        self.cascade_files()
        self.cascaded = True

    def add_leaf(self, table_name, restriction, cascade=False) -> None:
        """Add leaf to graph and cascade if requested.

        Parameters
        ----------
        table_name : str
            table name of leaf
        restriction : str
            restriction to apply to leaf
        """
        new_ancestors = set(self._get_ft(table_name).ancestors())
        self.ancestors |= new_ancestors  # Add to total ancestors
        self.visited -= new_ancestors  # Remove from visited to revisit

        self.leaves.add(table_name)
        self._set_restr(table_name, restriction)  # Redundant if cascaded

        if cascade:
            self.cascade1(table_name, restriction)
            self.cascade_files()
            self.cascaded = True

    def add_leaves(
        self, leaves: List[Dict[str, str]], cascade=False, show_progress=None
    ) -> None:
        """Add leaves to graph and cascade if requested.

        Parameters
        ----------
        leaves : List[Dict[str, str]]
            list of dictionaries containing table_name and restriction
        cascade : bool, optional
            Whether to cascade the restrictions up the graph. Default False
        show_progress : bool, optional
            Show tqdm progress bar. Default to verbose setting.
        """

        if not leaves:
            return
        if not isinstance(leaves, list):
            leaves = [leaves]
        leaves = unique_dicts(leaves)
        for leaf in tqdm(
            leaves,
            desc="RestrGraph: adding leaves",
            total=len(leaves),
            disable=not (show_progress or self.verbose),
        ):
            if not (
                (table_name := leaf.get("table_name"))
                and (restriction := leaf.get("restriction"))
            ):
                raise ValueError(
                    f"Leaf must have table_name and restriction: {leaf}"
                )
            self.add_leaf(table_name, restriction, cascade=False)
        if cascade:
            self.cascade()
            self.cascade_files()

    @property
    def as_dict(self) -> List[Dict[str, str]]:
        """Return as a list of dictionaries of table_name: restriction"""
        self.cascade()
        return [
            {"table_name": table, "restriction": self._get_restr(table)}
            for table in self.ancestors
            if self._get_restr(table)
        ]

    @property
    def file_dict(self) -> Dict[str, List[str]]:
        """Return dictionary of analysis files from all visited nodes.

        Currently unused, but could be useful for debugging.
        """
        if not self.cascaded:
            logger.warning("Uncascaded graph. Using leaves only.")
            table_list = self.leaves
        else:
            table_list = self.visited

        return {
            table: self._get_files(table)
            for table in table_list
            if any(self._get_files(table))
        }

    @property
    def file_paths(self) -> List[str]:
        """Return list of unique analysis files from all visited nodes.

        This covers intermediate analysis files that may not have been fetched
        directly by the user.
        """
        self.cascade()
        unique_files = set(
            [file for table in self.visited for file in self._get_files(table)]
        )
        return [
            {"file_path": AnalysisNwbfile().get_abs_path(file)}
            for file in unique_files
            if file is not None
        ]
