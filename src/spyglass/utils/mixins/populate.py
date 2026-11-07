"""Mixin for tables with custom populate behavior."""

import functools
import shutil
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, NamedTuple, Tuple, Union

from spyglass.utils.dj_helper_fn import NonDaemonPool, populate_pass_function
from spyglass.utils.logging import logger
from spyglass.utils.mixins.base import BaseMixin

# Instance attribute: one list of recorded outputs per active ``_populate1``.
_SCOPES_ATTR = "_staged_output_scopes"

# Longer than any single populate; finite so a stuck holder cannot wedge others.
POPULATE_LOCK_TIMEOUT_S = 8 * 3600


class StagedOutputs(NamedTuple):
    """Outputs a ``make_compute`` wrote for its ``make_insert`` to register.

    Attributes
    ----------
    analysis_file_names : tuple of str
        ``AnalysisNwbfile`` names that have no table row yet.
    paths : tuple of str or Path
        Files or folders, unique to this attempt.
    owners : tuple
        Objects whose ``close()`` discards their staged output. ``close`` must
        be safe to call more than once.
    """

    analysis_file_names: Tuple[str, ...] = ()
    paths: Tuple[Union[str, Path], ...] = ()
    owners: Tuple[Any, ...] = ()


def unlink_staged_analysis_file(analysis_file_name: str, *, context: str):
    """Remove an unregistered analysis file; log, never raise, on failure.

    Parameters
    ----------
    analysis_file_name : str
        Name of the ``AnalysisNwbfile`` to remove.
    context : str
        Caller name used in the log message.
    """
    try:
        from spyglass.common.common_nwbfile import AnalysisNwbfile

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        Path(abs_path).unlink(missing_ok=True)
    except Exception as exc:  # keep the populate's own error
        logger.error(
            f"{context}: failed to remove staged analysis file "
            f"{analysis_file_name!r}: {exc!r}"
        )


def _remove_staged_path(path: Union[str, Path], *, context: str):
    """Remove a staged file or folder; log, never raise, on failure."""
    path = Path(path)
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.error(f"{context}: failed to remove staged {path}: {exc!r}")


def _recording_compute(compute):
    """Wrap ``make_compute`` to record the outputs its result staged."""

    @functools.wraps(compute)
    def make_compute(self, key, *args, **kwargs):
        computed = compute(self, key, *args, **kwargs)
        scopes = self.__dict__.get(_SCOPES_ATTR)
        if scopes:
            scopes[-1].append(self._staged_outputs(computed))
        return computed

    return make_compute


def _releasing_insert(insert):
    """Wrap ``make_insert`` to forget the outputs it registered."""

    @functools.wraps(insert)
    def make_insert(self, key, *args, **kwargs):
        insert(self, key, *args, **kwargs)
        scopes = self.__dict__.get(_SCOPES_ATTR)
        if scopes:
            scopes[-1].clear()

    return make_insert


def _get_lock(connection, name: str, timeout_s: int) -> bool:
    """Return whether ``GET_LOCK`` took ``name`` within ``timeout_s``."""
    result = connection.query(
        "SELECT GET_LOCK(%s, %s)", args=(name, int(timeout_s))
    ).fetchone()
    return bool(result and result[0] == 1)


@contextmanager
def populate_key_lock(table, key: dict):
    """Serialize populates of one ``(table, key)`` with a MySQL named lock.

    Best effort: yields whether the lock was taken, and the caller proceeds
    either way, so the worst case is a repeated compute. The lock lives on the
    database server, so it coordinates jobs on different hosts, and a dropped
    session releases it.

    Parameters
    ----------
    table : dj.Table
        The table being populated.
    key : dict
        The primary key being populated.
    """
    from datajoint.hash import key_hash

    # MySQL caps lock names at 64 characters.
    name = "sgpop:" + key_hash({"__table__": table.full_table_name, **key})
    connection = table.connection
    acquired = False
    try:
        try:
            acquired = _get_lock(connection, name, 0)
            if not acquired:
                logger.info(
                    f"{type(table).__name__}: waiting for another job "
                    f"populating {key}"
                )
                acquired = _get_lock(connection, name, POPULATE_LOCK_TIMEOUT_S)
        except Exception as exc:  # a real DB error resurfaces in populate
            logger.warning(f"Populate lock for {key} unavailable: {exc!r}")
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                connection.query("SELECT RELEASE_LOCK(%s)", args=(name,))
            except Exception:  # closing the session releases it anyway
                pass


class PopulateMixin(BaseMixin):

    _parallel_make = False  # Tables that use parallel processing in make
    _use_transaction = True  # Use transaction in populate.

    # Tri-part tables whose make_compute writes files define
    # _staged_outputs(self, computed) -> StagedOutputs to opt in to cleanup.
    _staged_outputs = None
    # True makes a job wait for another job's populate of the same key, then
    # skip it, instead of computing it twice.
    _populate_key_lock = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._staged_outputs is None:
            return
        for name, wrap in (
            ("make_compute", _recording_compute),
            ("make_insert", _releasing_insert),
        ):
            method = cls.__dict__.get(name)
            if method is not None:
                setattr(cls, name, wrap(method))

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

    # ----------------------------- staged outputs -----------------------------

    def _populate1(self, key, jobs, *args, **kwargs):
        """Run DataJoint's ``_populate1``, under a per-key lock if opted in.

        DataJoint skips a key that is already populated before computing it,
        so a job that waited on the lock skips the key another job finished.
        """
        lock = (
            populate_key_lock(self, key)
            if self._populate_key_lock
            else nullcontext()
        )
        with lock:
            return self._populate1_with_cleanup(key, jobs, *args, **kwargs)

    def _populate1_with_cleanup(self, key, jobs, *args, **kwargs):
        """Run DataJoint's ``_populate1``; remove outputs left unregistered.

        DataJoint runs a tri-part ``make_compute`` outside the transaction, so
        files it writes outlive a populate that fails before ``make_insert``
        commits. For tables that define ``_staged_outputs``, the outputs of
        each ``make_compute`` are recorded, forgotten once ``make_insert``
        returns, and removed on a raise or a non-``True`` status (``False`` for
        a skipped key, ``(key, error)`` under ``suppress_errors``). A failed
        commit after ``make_insert`` or a killed process leaves them on disk.
        """
        if self._staged_outputs is None:
            return super()._populate1(key, jobs, *args, **kwargs)

        scopes = self.__dict__.setdefault(_SCOPES_ATTR, [])
        pending = []
        scopes.append(pending)
        try:
            status = super()._populate1(key, jobs, *args, **kwargs)
        except BaseException:
            self._discard_staged_outputs(pending)
            raise
        finally:
            scopes.pop()
        if status is not True:
            self._discard_staged_outputs(pending)
        return status

    def _discard_staged_outputs(self, pending):
        """Remove every recorded output; log, never raise, on failure."""
        context = f"{type(self).__name__}.populate"
        for outputs in pending:
            for name in outputs.analysis_file_names:
                unlink_staged_analysis_file(name, context=context)
            for owner in outputs.owners:
                try:
                    owner.close()
                except Exception as exc:  # keep the populate's own error
                    logger.error(
                        f"{context}: failed to discard staged output "
                        f"{owner!r}: {exc!r}"
                    )
            for path in outputs.paths:
                _remove_staged_path(path, context=context)
