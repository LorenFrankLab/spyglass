"""Remove what a tri-part ``make_compute`` staged when its populate fails.

A v2 table whose ``make_compute`` writes an output (an ``AnalysisNwbfile``, a
private analyzer build, a figure bundle) leaves registering it to
``make_insert``. DataJoint 0.14.9 runs the two in separate steps of
``AutoPopulate._populate1`` (``datajoint/autopopulate.py:362-453``):

1. ``make_fetch`` then ``make_compute``, outside any transaction (lines
   404-407). The generator running them is dropped right after
   ``make_compute`` returns, even on success, so nothing can be cleaned up
   in it.
2. A transaction starts, ``make`` restarts and ``make_fetch`` runs again
   (lines 409-411). If that fetch raises, or its hash differs from the first
   one, DataJoint raises (``Referential integrity failed``, lines 412-420)
   before the computed result ever reaches ``make_insert`` (line 421).
3. Any exception cancels the transaction (lines 423-427) and is re-raised,
   or returned as ``(key, error)`` when ``suppress_errors`` (lines 443-447).

A table's own ``make_compute`` / ``make_insert`` cleanup covers failures inside
those two methods, not step 2. The :class:`StagedOutputCleanupMixin` covers
everything from ``make_compute`` returning until ``make_insert`` returns: it
records the staged outputs of each computed result when ``make_compute``
returns, forgets them once ``make_insert`` returns (they are registered from
then on), and removes whatever is still recorded when ``_populate1`` ends
without success.

Not covered: a failed COMMIT after ``make_insert`` has returned (DataJoint's
``else`` clause, line 449). The registration rolls back but the outputs were
already forgotten, so they stay on disk -- including what ``make_insert``
moved into place before the commit (``Sorting``'s analyzer publish,
``FigPackCuration``'s bundle install). Forgetting at that point is what keeps
a failure after a successful commit (``jobs.complete``, line 452) from
removing a committed row's files. Nor is a process killed between
``make_compute`` and ``make_insert``; its staging stays until swept.

This module is DB-free; removing an ``AnalysisNwbfile`` imports its helper
lazily.
"""

from __future__ import annotations

import functools
import shutil
from typing import Any, NamedTuple

from spyglass.utils import logger

#: Instance attribute holding one list of recorded outputs per active
#: ``_populate1`` call (a stack, so a nested populate keeps its own list).
_SCOPES_ATTR = "_staged_output_scopes"


class StagedOutputs(NamedTuple):
    """Outputs a ``make_compute`` wrote for its ``make_insert`` to register.

    Attributes
    ----------
    analysis_file_names : tuple of str
        Staged ``AnalysisNwbfile`` names that have no table row yet.
    owners : tuple
        Staging owners whose ``close()`` discards the private staged output
        and never a published one (e.g. ``StagedAnalyzer``). ``close`` must be
        safe to call more than once.
    folders : tuple of str
        Private staging directories, unique to this attempt, that
        ``make_insert`` moves into place.
    """

    analysis_file_names: tuple[str, ...] = ()
    owners: tuple[Any, ...] = ()
    folders: tuple[str, ...] = ()


def _recording_compute(compute):
    """Wrap ``make_compute`` to record its result's staged outputs."""

    @functools.wraps(compute)
    def make_compute(self, key, *args, **kwargs):
        computed = compute(self, key, *args, **kwargs)
        scopes = self.__dict__.get(_SCOPES_ATTR)
        if scopes:
            scopes[-1].append(computed.staged_outputs())
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


class StagedOutputCleanupMixin:
    """Remove a populate attempt's staged outputs unless its insert succeeds.

    For a tri-part table whose ``make_compute`` returns a carrier with a
    ``staged_outputs() -> StagedOutputs`` method. ``make_compute`` and
    ``make_insert`` defined on the class are wrapped when the class is
    created; the inherited generator ``make`` is untouched, so DataJoint's
    tri-part dispatch still applies. Put the mixin first in the bases so its
    ``_populate1`` runs around DataJoint's.

    Outputs are recorded only inside ``_populate1``, on the table instance
    running it: a direct ``make_compute`` call records nothing, and
    ``populate(processes>1)`` records in each worker's own instance.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, wrap in (
            ("make_compute", _recording_compute),
            ("make_insert", _releasing_insert),
        ):
            method = cls.__dict__.get(name)
            if method is not None:
                setattr(cls, name, wrap(method))

    def _populate1(self, key, jobs, *args, **kwargs):
        """Run DataJoint's ``_populate1``; remove outputs left unregistered.

        DataJoint returns ``True`` after a committed insert, ``False`` for a
        key it skipped (already populated or reserved elsewhere; nothing was
        computed), and ``(key, error)`` for a failure under
        ``suppress_errors``; any other failure raises. Recorded outputs are
        removed on a raise or a ``(key, error)`` return, after DataJoint has
        cancelled the transaction, and the error is passed on unchanged.
        """
        scopes = self.__dict__.setdefault(_SCOPES_ATTR, [])
        pending: list[StagedOutputs] = []
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

    def _discard_staged_outputs(self, pending: list[StagedOutputs]) -> None:
        """Remove every recorded output; log, never raise, on failure."""
        context = f"{type(self).__name__}.populate"
        for outputs in pending:
            for name in outputs.analysis_file_names:
                self._unlink_analysis_file(name, context=context)
            for owner in outputs.owners:
                try:
                    owner.close()
                except Exception as exc:  # noqa: BLE001 - keep the cause
                    logger.error(
                        f"{context}: failed to discard staged output "
                        f"{owner!r}: {exc!r}"
                    )
            for folder in outputs.folders:
                try:
                    shutil.rmtree(folder)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.error(
                        f"{context}: failed to remove staged folder "
                        f"{folder!r}: {exc!r}"
                    )

    @staticmethod
    def _unlink_analysis_file(analysis_file_name: str, *, context: str):
        """Remove a staged, unregistered ``AnalysisNwbfile`` (best effort)."""
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        _unlink_staged_analysis_file(analysis_file_name, context=context)
