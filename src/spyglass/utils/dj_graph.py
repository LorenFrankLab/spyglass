"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from typing import Dict, List

from datajoint.table import Table
from tqdm import tqdm

from spyglass.common import AnalysisNwbfile
from spyglass.utils.dj_graph_abs import AbstractGraph
from spyglass.utils.dj_helper_fn import unique_dicts


class RestrGraph(AbstractGraph):
    def __init__(
        self,
        seed_table: Table,
        table_name: str = None,
        restriction: str = None,
        leaves: List[Dict[str, str]] = None,
        cascade: bool = False,
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
        cascade : bool, optional
            Whether to cascade restrictions up the graph on initialization.
            Default False
        verbose : bool, optional
            Whether to print verbose output. Default False
        """
        super().__init__(seed_table, verbose=verbose)

        self.leaves = set()
        self.analysis_pk = AnalysisNwbfile().primary_key

        if table_name and restriction:
            self.add_leaf(table_name, restriction)
        if leaves:
            self.add_leaves(leaves, show_progress=verbose)

        if cascade:
            self.cascade()

    def __repr__(self):
        l_str = ",\n\t".join(self.leaves) + "\n" if self.leaves else ""
        processed = "Cascaded" if self.cascaded else "Uncascaded"
        return f"{processed} RestrictionGraph(\n\t{l_str})"

    @property
    def leaf_ft(self):
        """Get restricted FreeTables from graph leaves."""
        return [self._get_ft(table, with_restr=True) for table in self.leaves]

    def old_cascade1(self, table, restriction, direction="up"):
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

        for parent, data in next_nodes.items():
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

            self.cascade1(
                table=parent,
                restriction=parent_restr,
                direction=direction,
            )

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
        if not self.visited == self.to_visit:
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
        self.cascaded = False

        new_ancestors = set(self._get_ft(table_name).ancestors())
        self.to_visit |= new_ancestors  # Add to total ancestors
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

    # ----------------------------- File Handling -----------------------------

    def _get_files(self, table):
        """Get analysis files from graph node."""
        return self._get_node(table).get("files", [])

    def cascade_files(self):
        """Set node attribute for analysis files."""
        for table in self.visited:
            ft = self._get_ft(table)
            if not set(self.analysis_pk).issubset(ft.heading.names):
                continue
            files = list((ft & self._get_restr(table)).fetch(*self.analysis_pk))
            self._set_node(table, "files", files)

    @property
    def file_dict(self) -> Dict[str, List[str]]:
        """Return dictionary of analysis files from all visited nodes.

        Currently unused, but could be useful for debugging.
        """
        self.cascade()
        return self._get_attr_dict("files", default_factory=lambda: [])

    @property
    def file_paths(self) -> List[str]:
        """Return list of unique analysis files from all visited nodes.

        This covers intermediate analysis files that may not have been fetched
        directly by the user.
        """
        self.cascade()
        return [
            {"file_path": AnalysisNwbfile().get_abs_path(file)}
            for file in set(
                [f for files in self.file_dict.values() for f in files]
            )
            if file is not None
        ]
