"""Mixin for tables with custom populate behavior."""

from spyglass.utils.mixins.base import BaseMixin


class PopulateMixin(BaseMixin):

    _parallel_make = False  # Tables that use parallel processing in make
    _use_transaction = True  # Use transaction in populate.

    # -------------------------------- populate --------------------------------

    def _hash_upstream(self, keys):
        """Hash upstream table keys for no transaction populate.

        Uses a RestrGraph to capture all upstream tables, restrict them to
        relevant entries, and hash the results. This is used to check if
        upstream tables have changed during a no-transaction populate and avoid
        the following data-integrity error:

        1. User A starts no-transaction populate.
        2. User B deletes and repopulates an upstream table, changing contents.
        3. User A finishes populate, inserting data that is now invalid.

        Parameters
        ----------
        keys : list
            List of keys for populating table.
        """
        RestrGraph = self._graph_deps[1]
        if not (parents := self.parents(as_objects=True, primary=True)):
            # Should not happen, as this is only called from populated tables
            raise RuntimeError("No upstream tables found for upstream hash.")

        if isinstance(keys, dict):
            keys = [keys]  # case for single population key
        leaves = {  # Restriction on each primary parent
            p.full_table_name: [
                {k: v for k, v in key.items() if k in p.heading.names}
                for key in keys
            ]
            for p in parents
        }

        return RestrGraph(seed_table=self, leaves=leaves, cascade=True).hash

    def populate(self, *restrictions, **kwargs):
        """Populate table in parallel, with or without transaction protection.

        Supersedes datajoint.table.Table.populate for classes with that
        spawn processes in their make function and always use transactions.

        `_use_transaction` class attribute can be set to False to disable
        transaction protection for a table. This is not recommended for tables
        with short processing times. A before-and-after hash check is performed
        to ensure upstream tables have not changed during populate, and may
        be a more time-consuming process. To permit the `make` to insert without
        populate, set `_allow_insert` to True.
        """
        processes = kwargs.pop("processes", 1)

        # Decide if using transaction protection
        use_transact = kwargs.pop("use_transaction", None)
        if use_transact is None:  # if user does not specify, use class default
            use_transact = self._use_transaction
            if self._use_transaction is False:  # If class default is off, warn
                self._logger.warning(
                    "Turning off transaction protection this table by default. "
                    + "Use use_transation=True to re-enable.\n"
                    + "Read more about transactions:\n"
                    + "https://docs.datajoint.io/python/definition/05-Transactions.html\n"
                    + "https://github.com/LorenFrankLab/spyglass/issues/1030"
                )
        if use_transact is False and processes > 1:
            raise RuntimeError(
                "Must use transaction protection with parallel processing.\n"
                + "Call with use_transation=True.\n"
                + f"Table default transaction use: {self._use_transaction}"
            )

        # Get keys, needed for no-transact or multi-process w/_parallel_make
        keys = [True]
        if use_transact is False or (processes > 1 and self._parallel_make):
            keys = (self._jobs_to_do(restrictions) - self.target).fetch(
                "KEY", limit=kwargs.get("limit", None)
            )

        if use_transact is False:
            upstream_hash = self._hash_upstream(keys)
            if kwargs:  # Warn of ignoring populate kwargs, bc using `make`
                self._logger.warning(
                    "Ignoring kwargs when not using transaction protection."
                )

        if processes == 1 or not self._parallel_make:
            if use_transact:  # Pass single-process populate to super
                kwargs["processes"] = processes
                return super().populate(*restrictions, **kwargs)
            else:  # No transaction protection, use bare make
                for key in keys:
                    self.make(key)
                if upstream_hash != self._hash_upstream(keys):
                    (self & keys).delete(safemode=False)
                    self._logger.error(
                        "Upstream tables changed during non-transaction "
                        + "populate. Please try again."
                    )
                return None

        # If parallel in both make and populate, use non-daemon processes
        # package the call list
        call_list = [(type(self), key, kwargs) for key in keys]

        # Create a pool of non-daemon processes to populate a single entry each
        pool = NonDaemonPool(processes=processes)
        try:
            pool.map(populate_pass_function, call_list)
        except Exception as e:
            raise e
        finally:
            pool.close()
            pool.terminate()
