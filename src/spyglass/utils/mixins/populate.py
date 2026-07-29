"""Mixin for tables with custom populate behavior."""

from spyglass.utils.dj_helper_fn import NonDaemonPool, populate_pass_function
from spyglass.utils.mixins.base import BaseMixin


class PopulateMixin(BaseMixin):

    _parallel_make = False  # Tables that use parallel processing in make
    _use_transaction = True  # Use transaction in populate.

    # -------------------------------- populate --------------------------------

    def populate(self, *restrictions, **kwargs):
        """Populate table in parallel, using transaction protection.

        Supersedes datajoint.table.Table.populate for classes that spawn
        processes in their make function and always use transactions.
        """
        processes = kwargs.pop("processes", 1)

        # Non-transaction populate is no longer supported.
        use_transact = kwargs.pop("use_transaction", None)
        if use_transact is None:
            use_transact = getattr(self, "_use_transaction", None)
        if use_transact is False:
            raise NotImplementedError(
                "Non-transaction populate no longer supported. Please remove "
                + "_use_transaction = False from the class and use tri-part "
                + "make instead. See DataJoint documentation for details."
            )

        # Get keys, needed for no-transact or multi-process w/_parallel_make
        keys = [True]
        if processes > 1 and self._parallel_make:
            keys = (self._jobs_to_do(restrictions) - self.target).fetch(
                "KEY", limit=kwargs.get("limit", None)
            )

        if processes == 1 or not self._parallel_make:
            kwargs["processes"] = processes
            return super().populate(*restrictions, **kwargs)

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
