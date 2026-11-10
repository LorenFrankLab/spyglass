"""Remove what a tri-part ``make_compute`` staged when its populate fails.

The cleanup itself is ``PopulateMixin``'s ``_staged_outputs`` hook (see
``spyglass.utils.mixins.populate``). v2 carriers name their own outputs with
``staged_outputs()``; ``StagedOutputCleanupMixin`` connects the two.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from spyglass.utils.mixins.populate import (  # re-exported
    StagedOutputs,
    unlink_staged_analysis_file,
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


class StagedOutputCleanupMixin:
    """Opt a v2 tri-part table in to ``PopulateMixin``'s staged-output cleanup.

    For tables whose ``make_compute`` result has
    ``staged_outputs() -> StagedOutputs``. Put the mixin ahead of
    ``SpyglassMixin`` so this hook overrides ``PopulateMixin``'s ``None``.
    """

    def _staged_outputs(self, computed) -> StagedOutputs:
        return computed.staged_outputs()
