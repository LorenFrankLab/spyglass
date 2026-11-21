"""The ``IntervalList`` row a ``Recording`` owns for its kept times.

Names are ``f"recording_{recording_id}"``; ``Recording.ValidInterval`` owns
the row. ``recording`` imports this module, so ``IntervalList`` is imported
inside the functions.
"""

from __future__ import annotations

RECORDING_VALID_INTERVAL_PIPELINE = "spikesorting_recording_v2"

_RECORDING_INTERVAL_LIST_PREFIX = "recording_"


def recording_interval_list_name(recording_id) -> str:
    """Return the ``IntervalList.interval_list_name`` for a ``Recording``."""
    return f"{_RECORDING_INTERVAL_LIST_PREFIX}{recording_id}"


def valid_interval_row(nwb_file_name: str, recording_id, valid_times) -> dict:
    """Build the ``IntervalList`` row for a recording's kept times.

    Parameters
    ----------
    nwb_file_name : str
        The recording's session.
    recording_id : uuid.UUID or str
        The ``Recording`` the row belongs to.
    valid_times : np.ndarray
        ``(n_intervals, 2)`` session seconds from
        :func:`._recording.restriction.intended_save_intervals`.

    Returns
    -------
    dict
    """
    return {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": recording_interval_list_name(recording_id),
        "valid_times": valid_times,
        "pipeline": RECORDING_VALID_INTERVAL_PIPELINE,
    }


def insert_valid_interval_row(row: dict) -> None:
    """Insert ``row`` into ``IntervalList``, adopting an identical row.

    An upstream cascade deletes a ``Recording`` without its interval row, and
    a repopulate reuses the name.

    Raises
    ------
    ValueError
        If a row of that name exists with other times or another pipeline.
    """
    import numpy as np

    from spyglass.common import IntervalList

    key = {k: row[k] for k in ("nwb_file_name", "interval_list_name")}
    existing = (IntervalList & key).fetch(as_dict=True)
    if not existing:
        IntervalList.insert1(row)
        return
    same = existing[0]["pipeline"] == row["pipeline"] and np.array_equal(
        existing[0]["valid_times"], row["valid_times"]
    )
    if not same:
        raise ValueError(
            f"IntervalList {key} already exists with other times or from "
            f"pipeline {existing[0]['pipeline']!r}; refusing to adopt it. "
            "Delete that row if nothing uses it, then repopulate."
        )


def refuse_used_valid_intervals(restrictions, owner_part, *, context) -> None:
    """Raise if a table other than ``owner_part`` uses these intervals.

    A recording's interval can be chosen as another pipeline's interval;
    deleting it would cascade there.

    Parameters
    ----------
    restrictions : list of dict
        ``{nwb_file_name, interval_list_name}`` keys of the owned rows.
    owner_part : dj.Part
        ``Recording.ValidInterval``; its rows are skipped.
    context : str
        Caller name for the message.

    Raises
    ------
    ValueError
        Naming each referencing table and its row count.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2._core.table_integrity import (
        foreign_key_referrers,
    )

    referrers = foreign_key_referrers(
        IntervalList(), list(restrictions), skip={owner_part.full_table_name}
    )
    if referrers:
        names = sorted({row["interval_list_name"] for row in restrictions})
        raise ValueError(
            f"{context}: refusing to delete. Other tables use the recording "
            f"interval(s) {names}: {referrers} (rows per table). Delete those "
            "rows first, then delete the recording."
        )
