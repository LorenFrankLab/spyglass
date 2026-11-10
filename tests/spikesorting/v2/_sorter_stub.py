"""Substitute the spike sorter that ``Sorting.populate`` runs.

Replaces only the sorter step of ``Sorting``'s compute path, keeping the
real compute/NWB/insert path, so tests get known units or can observe what
the sorter receives. Tests use :func:`plant_sorter` and :func:`active_sorter`
so the private dispatch method is named only here (and in
``test_sorting_dispatch.py``, which tests the dispatch itself).

The stand-in is called exactly as the real sorter step is, with every
argument passed by keyword::

    run_sorter(
        sorter=..., sorter_params=..., recording=..., sorting_id=...,
        job_kwargs=..., execution_params=..., statistics_spans=...,
    )

and must return a ``spikeinterface`` ``BaseSorting`` whose spike frames index
``recording``.

Neither function imports Spyglass at module import time, so importing this
module opens no database connection.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest


def plant_sorter(
    monkeypatch: pytest.MonkeyPatch, run_sorter: Callable[..., Any]
) -> None:
    """Make ``Sorting.populate`` call ``run_sorter`` instead of the sorter.

    Installed as a static method on ``Sorting`` through ``monkeypatch``, so
    it is undone when ``monkeypatch`` is.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patch object that owns the substitution and its undo.
    run_sorter : Callable[..., spikeinterface.core.BaseSorting]
        Stand-in for the sorter step; see the module docstring for the
        call signature. A plain function: it receives no ``self``/``cls``.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    monkeypatch.setattr(Sorting, "_run_sorter", staticmethod(run_sorter))


def active_sorter() -> Callable[..., Any]:
    """Return the sorter step ``Sorting.populate`` currently calls.

    Read this before :func:`plant_sorter` to get the real sorter step, so a
    stand-in can record its inputs and then delegate to it.

    Returns
    -------
    Callable[..., spikeinterface.core.BaseSorting]
        The sorter step as a plain function (no ``self``/``cls``), with the
        call signature given in the module docstring.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    return Sorting._run_sorter
