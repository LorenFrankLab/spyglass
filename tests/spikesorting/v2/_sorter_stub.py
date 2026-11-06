"""Substitute the spike sorter that ``Sorting.populate`` runs.

Many v2 tests need a populated ``Sorting`` row with known units (a planted
merge group, a spike at an interval boundary, a sort that yields no units)
or need to observe exactly what the sorter receives. They get there by
replacing the one step of ``Sorting``'s compute path that calls into the
sorter, and keep the rest of the real compute/NWB/insert path.

Every test that substitutes the sorter goes through :func:`plant_sorter` and
:func:`active_sorter` rather than patching ``Sorting`` itself, so the
substitution names the private dispatch method only in this module.
``test_sorting_dispatch.py`` calls that method directly, because it tests the
dispatch itself.

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

    The stand-in is installed on the ``Sorting`` class as a static method
    through ``monkeypatch``, so it is undone with ``monkeypatch`` -- at the
    end of the test for the ``monkeypatch`` fixture, on leaving
    ``monkeypatch.context()``, or on ``MonkeyPatch.undo()`` for a
    ``pytest.MonkeyPatch()`` created inside a wider-scoped fixture.

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
