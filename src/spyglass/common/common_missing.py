"""Tables recording which database-tracked paths are absent from disk.

TEMPORARY: these live on a personal prefix while the design settles, so this
module is deliberately absent from `spyglass.common.__init__`. Importing it
declares the schema, which only its owner can create. Move to
`common_file_tracking` before merge.
"""

from datetime import datetime
from typing import Optional

import datajoint as dj

schema = dj.Schema("claude_missing")


@schema
class CheckMissingRun(dj.Manual):
    """One `check_missing` pass over the tables that declare tracked paths."""

    definition = """
    run_id: int auto_increment
    ---
    started=CURRENT_TIMESTAMP: timestamp
    finished=NULL: datetime
    """

    class Coverage(dj.Part):
        definition = """
        # What this run reached. Absence of a row means the scan did not run.
        -> master
        full_table_name: varchar(128)
        attr: varchar(64)
        ---
        n_checked=0: int
        n_missing=0: int
        n_excluded=0: int  # intentionally absent, so never stat'd
        completed=0: tinyint
        """

    def open(self) -> int:
        """Record the start of a run.

        Returns
        -------
        run_id : int
            Key of the new run.
        """
        # `started` is supplied rather than left to its default: DataJoint
        # rejects a row with no named attributes
        self.insert1(dict(started=datetime.now()))
        # LAST_INSERT_ID is per connection, so it survives concurrent runs
        return dj.conn().query("SELECT LAST_INSERT_ID()").fetchone()[0]

    def close(self, run_id: int) -> None:
        """Stamp a run as finished.

        Parameters
        ----------
        run_id : int
            Run to stamp.
        """
        self.update1(dict(run_id=run_id, finished=datetime.now()))


@schema
class MissingPath(dj.Manual):
    """A tracked path that was absent or unreadable when a run looked.

    Holds only paths that were not whole, so a row is a finding. `state`
    separates a loss from a path the scan could not read.

    One row per path. Where several entries share a path, `source_key` names
    one of them, and `run_id` is the run that first recorded the path, not
    every run that found it absent. So `MissingPath & {'run_id': n}` is what
    run `n` discovered, while `CheckMissingRun.Coverage.n_missing` is what it
    found absent, including paths an earlier run had already recorded.

    A restored path is removed by `refresh_missing`, so a row always
    describes a current absence. Dropping the row rather than flagging it
    resolved keeps `run_id` tied to the absence it reports: a path lost,
    restored and lost again gets a fresh row naming the second loss, and
    `last_covered_run` bounds that loss rather than the first one.
    """

    definition = """
    full_table_name: varchar(128)  # table that tracks the path
    attr: varchar(64)              # attribute holding the path
    path_hash: char(32)            # md5 of path, which is too long to key
    ---
    -> CheckMissingRun             # run that first recorded it
    path: varchar(1000)
    source_key: blob               # primary key of the row that tracks it
    state: varchar(16)             # see check_missing.STATES
    """

    @property
    def lost(self):
        """Paths that are gone: the restore list.

        Excludes unreadable paths, where presence is unknown.
        """
        from spyglass.utils.mixins.check_missing import LOST_STATES

        return self & [{"state": state} for state in LOST_STATES]

    @property
    def unreadable(self):
        """Findings where the scan could not determine presence."""
        return self & "state='denied'"

    def last_covered_run(self, key: dict) -> Optional[int]:
        """Return the latest completed run of this attribute before the find.

        The loss falls between that run and the one named by the finding.
        None when no earlier run completed coverage of the attribute, which
        is every finding of a first run.

        Named for what it measures. Coverage is per attribute, not per path,
        so a path whose row was created after that run resolves to it as
        well. The interval still contains the loss, but the run did not
        observe this path.

        Parameters
        ----------
        key : dict
            One `MissingPath` entry.

        Returns
        -------
        run_id : int or None
            Latest run that covered the attribute before this finding.
        """
        entry = (self & key).fetch1()
        covered = CheckMissingRun.Coverage & {
            "full_table_name": entry["full_table_name"],
            "attr": entry["attr"],
            "completed": 1,
        }
        earlier = [r for r in covered.fetch("run_id") if r < entry["run_id"]]

        return max(earlier) if earlier else None
