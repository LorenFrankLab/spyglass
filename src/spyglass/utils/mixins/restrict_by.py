"""Mixin to allow restriction by ancestor or descendant fields."""

from typing import List

from datajoint import DataJointError
from datajoint.expression import QueryExpression

from spyglass.utils.dj_helper_fn import ensure_names
from spyglass.utils.mixins.base import BaseMixin


class RestrictByMixin(BaseMixin):
    """Mixin to allow restriction by ancestor or descendant fields."""

    _banned_search_tables = set()  # Tables to avoid in restrict_by

    def ban_search_table(self, table):
        """Ban table from search in restrict_by."""
        self._banned_search_tables.update(ensure_names(table, force_list=True))

    def unban_search_table(self, table):
        """Unban table from search in restrict_by."""
        self._banned_search_tables.difference_update(
            ensure_names(table, force_list=True)
        )

    def see_banned_tables(self):
        """Print banned tables."""
        self._logger.info(f"Banned tables: {self._banned_search_tables}")

    def restrict_by(
        self,
        restriction: str = True,
        direction: str = "up",
        return_graph: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> QueryExpression:
        """Restrict self based on up/downstream table.

        If fails to restrict table, the shortest path may not have been correct.
        If there's a different path that should be taken, ban unwanted tables.

        >>> my_table = MyTable() # must be instantced
        >>> my_table.ban_search_table(UnwantedTable1)
        >>> my_table.ban_search_table([UnwantedTable2, UnwantedTable3])
        >>> my_table.unban_search_table(UnwantedTable3)
        >>> my_table.see_banned_tables()
        >>>
        >>> my_table << my_restriction

        Parameters
        ----------
        restriction : str
            Restriction to apply to the some table up/downstream of self.
        direction : str, optional
            Direction to search for valid restriction. Default 'up'.
        return_graph : bool, optional
            If True, return FindKeyGraph object. Default False, returns
            restricted version of present table.
        verbose : bool, optional
            If True, print verbose output. Default False.

        Returns
        -------
        Union[QueryExpression, TableChain]
            Restricted version of present table or TableChain object. If
            return_graph, use all_ft attribute to see all tables in cascade.
        """
        TableChain = self._graph_deps[0]

        if restriction is True:
            return self

        try:
            ret = self.restrict(restriction)  # Save time trying first
            if len(ret) < len(self):
                # If it actually restricts, if not it might by a dict that
                # is not a valid restriction, returned as True
                self._logger.warning(
                    "Restriction valid for this table. Using as is."
                )
                return ret
        except DataJointError:  # need assert_join_compatible return bool
            self._logger.debug("Restriction not valid. Attempting to cascade.")

        if direction == "up":
            parent, child = None, self
        elif direction == "down":
            parent, child = self, None
        else:
            raise ValueError("Direction must be 'up' or 'down'.")

        graph = TableChain(
            parent=parent,
            child=child,
            direction=direction,
            search_restr=restriction,
            banned_tables=list(self._banned_search_tables),
            cascade=True,
            verbose=verbose,
            **kwargs,
        )

        if not graph.found_restr:
            return None

        if return_graph:
            return graph

        ret = self & graph._get_restr(self.full_table_name)
        warn_text = (
            f" after restrict with path: {graph.path_str}\n\t "
            + "See `help(YourTable.restrict_by)`"
        )
        if len(ret) == len(self):
            self._logger.warning("Same length" + warn_text)
        elif len(ret) == 0:
            self._logger.warning("No entries" + warn_text)

        return ret

    def __lshift__(self, restriction) -> QueryExpression:
        """Restriction by upstream operator e.g. ``q1 << q2``.

        Returns
        -------
        QueryExpression
            A restricted copy of the query expression using the nearest upstream
            table for which the restriction is valid.
        """
        return self.restrict_by(restriction, direction="up")

    def __rshift__(self, restriction) -> QueryExpression:
        """Restriction by downstream operator e.g. ``q1 >> q2``.

        Returns
        -------
        QueryExpression
            A restricted copy of the query expression using the nearest upstream
            table for which the restriction is valid.
        """
        return self.restrict_by(restriction, direction="down")

    def restrict_all(
        self,
        restriction: str = True,
        direction: str = "down",
        return_graph: bool = False,
        verbose: bool = False,
    ) -> List:
        RestrGraph = self._graph_deps[1]
        rg = RestrGraph(
            seed_table=self,
            leaves=dict(
                table_name=self.full_table_name,
                restriction=self.restriction or restriction,
            ),
            direction=direction,
            banned_tables=list(self._banned_search_tables),
            cascade=True,
            verbose=verbose,
        )
        if return_graph:
            return rg
        self._logger.info(rg.restr_ft)
        return None
