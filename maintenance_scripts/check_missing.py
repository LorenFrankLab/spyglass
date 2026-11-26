#!/usr/bin/env python
"""Record database-tracked paths that are absent from disk.

Separate from cleanup.py on purpose. `cleanup.py` runs five cleanups in one
function and aborts the cron on the first failure, so a step placed after it
is skipped whenever cleanup breaks. A check meant to notice loss must not
depend on cleanup succeeding.

Writes a summary to the path named by CHECK_MISSING_OUT, and only when the
run recorded something new. A run that finds the same losses as the last one
records nothing, so writing unconditionally would mail an unchanging report
every night and train the reader to ignore it.
"""

import os
import sys

from spyglass.common.common_file_tracking import CheckMissingRun, MissingPath


def format_summary(run_id: int, newly: int, outstanding: int) -> str:
    """Return the report body for one run.

    Parameters
    ----------
    run_id : int
        Run to describe.
    newly : int
        Paths this run recorded for the first time.
    outstanding : int
        Paths currently absent, across all runs.

    Returns
    -------
    body : str
        Plain text, one line per attribute scanned.
    """
    frame = CheckMissingRun().summary(run_id)

    lines = [
        f"check_missing run {run_id}",
        f"  newly recorded : {newly}",
        f"  outstanding    : {outstanding}",
        "",
    ]

    if frame.empty:
        lines.append("No attribute was covered. The run reached nothing.")
        return "\n".join(lines)

    lines.append(
        f"{'table':<40} {'attr':<28} {'checked':>9} "
        f"{'missing':>8} {'excluded':>9}"
    )
    for row in frame.to_dict("records"):
        table = row["full_table_name"].replace("`", "").split(".")[-1]
        lines.append(
            f"{table:<40} {row['attr']:<28} {row['n_checked']:>9} "
            f"{row['n_missing']:>8} {row['n_excluded']:>9}"
        )

    lines += [
        "",
        "An attribute absent above was not covered, which is not the same",
        "as having nothing missing. Inspect with MissingPath().lost",
    ]

    return "\n".join(lines)


def main() -> int:
    """Scan every declaring table and report what the run recorded."""
    run_id = CheckMissingRun().run_all(verbose=False)

    if run_id is None:
        print("check_missing: no declaring table found", file=sys.stderr)
        return 1

    newly = len(MissingPath() & {"run_id": run_id})
    outstanding = len(MissingPath().lost)
    body = format_summary(run_id, newly, outstanding)
    print(body)

    out_path = os.environ.get("CHECK_MISSING_OUT")
    if out_path and newly:
        with open(out_path, "w") as out_file:
            out_file.write(body + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
