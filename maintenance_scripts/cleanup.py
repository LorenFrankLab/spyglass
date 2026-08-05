#!/usr/bin/env python
"""Cleanup script for tables, external files, and temp directory

This script is intended to be run periodically to clean up the database tables
(pruning orphan entries), external files (deleting unreferenced files), and
temporary directory (deleting old files).
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

from spyglass.common import AnalysisNwbfile, Nwbfile
from spyglass.common.common_version import SpyglassVersions
from spyglass.decoding.decoding_merge import DecodingOutput
from spyglass.decoding.v1.clusterless import schema as clusterless_schema
from spyglass.decoding.v1.sorted_spikes import schema as spikes_schema
from spyglass.settings import temp_dir
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)
from spyglass.spikesorting.v0.spikesorting_sorting import SpikeSorting

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)


def run_table_cleanups() -> tuple:
    """Run each table cleanup independently.

    AnalysisNwbfile.cleanup() can refuse a plan on safety grounds. It is
    second of five, so without per-step isolation a refusal would skip the
    three later table cleanups, the temp sweep, and the issue report (the
    first cleanup has already run by then).

    Returns
    -------
    tuple of (list of str, bool)
        Labeled failure messages (empty when everything succeeded), and
        whether an analysis-storage phase failed.
    """
    steps = [  # (label, callable, touches analysis storage)
        ("Nwbfile", lambda: Nwbfile().cleanup(), False),
        ("AnalysisNwbfile", lambda: AnalysisNwbfile().cleanup(), True),
        ("SpikeSorting", lambda: SpikeSorting().cleanup(verbose=False), False),
        ("DecodingOutput", lambda: DecodingOutput().cleanup(), True),
        (
            "SpikeSortingRecording",
            lambda: SpikeSortingRecording().cleanup(verbose=False),
            False,
        ),
    ]
    errors = []
    analysis_storage_failed = False
    for name, func, touches_analysis in steps:
        # Once analysis cleanup has failed, the state of that storage is
        # unknown, so later phases that also delete from it are skipped.
        # Unrelated stores (raw, sorting, recording) proceed regardless.
        if touches_analysis and analysis_storage_failed:
            msg = f"{name}.cleanup() skipped: analysis storage state unknown"
            print(msg)
            errors.append(msg)
            continue
        try:
            func()
        except Exception as err:  # noqa: BLE001 - reported, not swallowed
            if touches_analysis:
                analysis_storage_failed = True
            msg = f"{name}.cleanup() failed: {err}"
            print(msg)
            errors.append(msg)
    return errors, analysis_storage_failed


def cleanup_external_files():
    """Delete unreferenced external files"""
    spikes_schema.external["analysis"].delete(delete_external_files=True)
    clusterless_schema.external["analysis"].delete(delete_external_files=True)


def cleanup_temp_dir(days_old: int = 7, dry_run: bool = True):
    """Delete files in temp_dir that are older than days_old

    As a precaution, this function only deletes files if temp_dir is named
    "tmp" or "temp".

    Parameters
    ----------
    days_old : int, optional
        Number of days old files should be before deletion (default is 7)
    dry_run : bool, optional
        If True, only print the command that would be run (default is True)
    """
    dir_path = Path(temp_dir)
    if not dir_path.is_dir() or dir_path.name not in ["tmp", "temp"]:
        raise RuntimeError(
            f"Invalid temp_dir: {temp_dir!r} is not a directory named "
            "'tmp' or 'temp'; refusing to sweep it"
        )

    if dry_run:
        print(f"Dry run of delete files in {temp_dir} older than {days_old}d")
        return

    # Argument lists, not shell=True: a temp_dir containing spaces or shell
    # metacharacters would otherwise be split or interpreted.
    delete_cmd = [
        "find",
        str(dir_path),
        "-type",
        "f",
        "-mtime",
        f"+{days_old}",
        "-delete",
    ]
    # -mindepth 1 so the sweep cannot delete the configured temp root
    # itself once it is empty.
    empty_dirs = [
        "find",
        str(dir_path),
        "-mindepth",
        "1",
        "-type",
        "d",
        "-empty",
        "-delete",
    ]
    try:
        subprocess.run(delete_cmd, check=True)
        subprocess.run(empty_dirs, check=True)
    except subprocess.CalledProcessError as e:
        # Raise so main() can record it; printing hid the failure from the
        # caller and from the cron job's exit status.
        raise RuntimeError(f"Error cleaning temp_dir: {e}") from e


def main():
    errors = []
    print("Updating Spyglass versions table...")
    try:
        SpyglassVersions().fetch_from_pypi()
    except Exception as err:  # noqa: BLE001
        msg = f"SpyglassVersions().fetch_from_pypi() failed: {err}"
        print(msg)
        errors.append(msg)

    print("Running table cleanups...")
    table_errors, analysis_storage_failed = run_table_cleanups()
    errors.extend(table_errors)

    if analysis_storage_failed:
        msg = (
            "External analysis cleanup skipped: analysis storage state unknown"
        )
        print(msg)
        errors.append(msg)
    else:
        print("Cleaning up external files...")
        try:
            cleanup_external_files()
        except Exception as err:  # noqa: BLE001
            msg = f"cleanup_external_files() failed: {err}"
            print(msg)
            errors.append(msg)

    print("Cleaning up temporary directory...")
    try:
        cleanup_temp_dir(dry_run=False)
    except Exception as err:  # noqa: BLE001
        msg = f"cleanup_temp_dir() failed: {err}"
        print(msg)
        errors.append(msg)

    # This monitoring pass may populate AnalysisFileIssues, but it does not
    # delete analysis files, so it still runs after earlier cleanup failures.
    # Keep it wrapped so a monitoring or report-write failure reaches the
    # final summary and exit status.
    print("Checking for AnalysisFile Issues...")
    try:
        results = AnalysisNwbfile().check_all_files()
        out_path = os.environ.get("FILE_ISSUES_OUT")
        if out_path:
            issues = {tbl: cnt for tbl, cnt in results.items() if cnt > 0}
            with open(out_path, "w") as f:
                for tbl, cnt in issues.items():
                    f.write(f"{tbl}: {cnt}\n")
    except Exception as err:  # noqa: BLE001
        msg = f"check_all_files() failed: {err}"
        print(msg)
        errors.append(msg)

    # When the monitoring pass succeeds, its issue report has been written
    # before this nonzero exit so the cron job can send it. If the monitoring
    # pass or report write itself failed, that failure is summarized here and
    # the report may be absent or incomplete.
    if errors:
        print("Cleanup completed with failures:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
