"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from typing import Dict, List, Set, Tuple, Union

from datajoint.condition import make_condition
from datajoint.table import Table
from tqdm import tqdm

from spyglass.common import AnalysisNwbfile
from spyglass.utils import logger
from spyglass.utils.dj_graph_abs import AbstractGraph
from spyglass.utils.dj_helper_fn import PERIPHERAL_TABLES, unique_dicts


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

        self.analysis_pk = AnalysisNwbfile().primary_key

        self.add_leaf(table_name=table_name, restriction=restriction)
        self.add_leaves(leaves)

        if cascade:
            self.cascade()

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

        if direction == "up":
            new_visits = set(self._get_ft(table_name).ancestors())
        else:
            new_visits = set(self._get_ft(table_name).descendants())

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


class FindKeyGraph(RestrGraph):
    def __init__(
        self,
        seed_table: Table,
        table_name: str = None,
        restriction: str = None,
        leaves: List[Dict[str, str]] = None,
        direction: str = "up",
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

        if restriction and table_name:
            self._set_find_restr(table_name, restriction)
            self.add_leaf(table_name, True, cascade=False, direction=direction)
        self.add_leaves(
            leaves,
            default_restriction=restriction,
            cascade=False,
            direction=direction,
        )

        all_nodes = set([n for n in self.graph.nodes if not n.isnumeric()])
        self.no_visit.update(all_nodes - self.to_visit)  # Skip non-ancestors
        self.no_visit.update(PERIPHERAL_TABLES)

        if cascade and restriction:
            self.cascade()
            self.cascaded = True

    def _set_find_restr(self, table_name, restriction):
        """Set restr to look for from leaf node."""
        if isinstance(restriction, dict):
            logger.warning("key_from_upstream: DICT unreliable, use STR.")

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
            self._log_truncate(f"Start  {table}: {self._get_restr(table)}")
            restriction, restr_attrs = self._get_find_restr(table)
            self.cascade1_search(
                table=table,
                restriction=restriction,
                restr_attrs=restr_attrs,
                replace=True,
            )
        self.cascaded = True

    def cascade1_search(
        self,
        table: str,
        restriction: str,
        restr_attrs: Set[str] = None,
        direction: str = None,
        replace: bool = True,
    ):
        self._log_truncate(f"Search {table}: {restriction}")
        if self.cascaded:
            return

        direction = direction or self.direction
        next_func = (
            self.graph.parents if direction == "up" else self.graph.children
        )

        for next_table, data in next_func(table).items():
            if next_table.isnumeric():
                next_table, data = next_func(next_table).popitem()

            if next_table in self.no_visit or table == next_table:
                self._log_truncate(f"Skip   {next_table}: {restriction}")
                reason = "no_visit" if next_table in self.no_visit else "same"
                self._log_truncate(f"B/C    {next_table}: {reason}")
                continue

            next_ft = self._get_ft(next_table)
            if restr_attrs.issubset(set(next_ft.heading.names)):
                self._log_truncate(f"Found  {next_table}: {restriction}")
                self.cascade1(
                    table=next_table,
                    restriction=restriction,
                    direction="down" if direction == "up" else "up",
                    replace=replace,
                )
                self.cascaded = True

            self.cascade1_search(
                table=next_table,
                restriction=restriction,
                restr_attrs=restr_attrs,
                direction=direction,
                replace=replace,
            )
