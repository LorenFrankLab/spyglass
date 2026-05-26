"""Shared helpers for ingesting NWB files into the isolated test database.

Modern-spike-sorting fixture generation, the v1 baseline-capture script, and
the fixture round-trip test all need the same three-step pattern: copy the
NWB into the isolated raw directory, run ``insert_sessions``, and look up the
``Nwbfile`` row Spyglass created. Centralising it here keeps the deterministic
copy naming and the ``reinsert=True`` choice in one place.

This helper is intentionally version-tolerant on the ``reinsert`` kwarg
because ``baseline_capture.py`` is designed to run under master spyglass
(SI 0.99) where ``insert_sessions`` predates the ``reinsert`` parameter;
the v2 parity test then consumes the captured baseline under the current
spyglass + SI 0.104.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path


def copy_and_insert_nwb(nwb_source: Path | str) -> str:
    """Copy an NWB file into the test raw directory and ingest it.

    Parameters
    ----------
    nwb_source : pathlib.Path or str
        Source NWB file. Copied (not linked) into ``$SPYGLASS_RAW_DIR``.

    Returns
    -------
    str
        The Spyglass ``nwb_file_name``: the basename with the trailing-
        underscore copy suffix Spyglass uses
        (``get_nwb_copy_filename``).
    """
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwb_source = Path(nwb_source)
    raw_target = Path(raw_dir) / nwb_source.name
    if not raw_target.exists():
        shutil.copy(nwb_source, raw_target)
    kwargs = {"raise_err": True}
    if "reinsert" in inspect.signature(insert_sessions).parameters:
        kwargs["reinsert"] = True
    else:
        # Master spyglass's ``insert_sessions`` lacks the ``reinsert``
        # parameter and silently no-ops on a duplicate Nwbfile row.
        # Emulate the v2-branch reinsert path: drop any existing row
        # for the target name (cascades through downstream tables) so
        # ``populate_all_common`` actually runs.
        from spyglass.common.common_nwbfile import Nwbfile
        import datajoint as dj

        target_copy = get_nwb_copy_filename(nwb_source.name)
        existing = Nwbfile() & {"nwb_file_name": target_copy}
        if existing:
            prior_safemode = dj.config.get("safemode", True)
            dj.config["safemode"] = False
            try:
                existing.delete()
            finally:
                dj.config["safemode"] = prior_safemode
    insert_sessions(nwb_source.name, **kwargs)
    return get_nwb_copy_filename(nwb_source.name)
