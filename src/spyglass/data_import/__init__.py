"""Reading an NWB file into the database: plan it, then insert it.

A plain eager package, like every other Spyglass subpackage. It can be, because
nothing under `spyglass.utils` imports from here: the plan vocabulary the
ingestion mixin speaks lives in `spyglass.utils.ingestion_plan`, so importing
this package is never part of `utils` initializing itself. Keep it that way --
a `utils` module that imports from `data_import` puts this package's import
inside `spyglass.utils`'s own, at which point `settings`, `utils` and `common`
are all half-built and nothing here can be imported at all.
"""

from spyglass.data_import.insert_tools import (
    copy_nwb_link_raw_ephys,
    insert_sessions,
)

__all__ = ["copy_nwb_link_raw_ephys", "insert_sessions"]
