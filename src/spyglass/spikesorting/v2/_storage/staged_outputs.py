"""Remove what a tri-part ``make_compute`` staged when its populate fails.

DataJoint runs ``make_compute`` outside the transaction and can fail before
``make_insert`` registers what it wrote. Outputs are recorded when
``make_compute`` returns, forgotten when ``make_insert`` returns, and removed
if ``_populate1`` ends without success. A failed commit after ``make_insert``
or a killed process leaves staging on disk until swept.
"""

from __future__ import annotations

import functools
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from spyglass.spikesorting.v2._core.dj_compat import (
    require_staged_populate_support,
)
from spyglass.utils import logger

#: Stack of recorded-output lists, one per active ``_populate1`` call.
_SCOPES_ATTR = "_staged_output_scopes"


def unlink_staged_analysis_file(
    analysis_file_name: str,
    *,
    context: str,
    existing_analysis_file_name: str | None = None,
) -> None:
    """Discard a staged analysis file without masking the caller's failure.

    A canonical artifact named by ``existing_analysis_file_name`` is never
    removed. Failures are logged with ``context``, never raised.
    """
    if existing_analysis_file_name:
        logger.error(
            f"{context}: in-place rebuild of canonical artifact "
            f"{analysis_file_name!r} failed; leaving it in place "
            "(refusing to unlink a canonical artifact on failure)."
        )
        return
    try:
        from spyglass.common.common_nwbfile import AnalysisNwbfile

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        Path(abs_path).unlink(missing_ok=True)
    except Exception as cleanup_exc:  # noqa: BLE001 - preserve original failure
        logger.error(
            f"{context}: failed to clean up staged analysis file "
            f"{analysis_file_name!r}: {cleanup_exc!r}"
        )


@contextmanager
def staged_analysis_file(
    nwb_file_name: str,
    *,
    context: str,
) -> Iterator[str]:
    """Create an unregistered ``AnalysisNwbfile``; remove it if the body fails.

    ``AnalysisNwbfile().create`` writes a stub to disk immediately, so a
    failed write would otherwise orphan it.

    Parameters
    ----------
    nwb_file_name : str
        Parent NWB the analysis file is created from.
    context : str
        Caller name used in cleanup log messages.

    Yields
    ------
    str
        The staged analysis file name.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    analysis_file_name = AnalysisNwbfile().create(
        nwb_file_name=nwb_file_name,
        restrict_permission=True,  # 0o644, not world-writable 0o666
    )
    try:
        yield analysis_file_name
    except Exception:
        unlink_staged_analysis_file(analysis_file_name, context=context)
        raise


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

    For tri-part tables whose ``make_compute`` result has
    ``staged_outputs() -> StagedOutputs``. Put the mixin first in the bases.
    Outputs are recorded only inside ``_populate1``, so a direct
    ``make_compute`` call records nothing.
    """

    def __init_subclass__(cls, **kwargs):
        require_staged_populate_support()
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

        Removes outputs on a raise or a non-``True`` status (``False`` for a
        skipped key, ``(key, error)`` under ``suppress_errors``).
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
        unlink_staged_analysis_file(analysis_file_name, context=context)
