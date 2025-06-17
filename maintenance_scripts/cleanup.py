#!/usr/bin/env python
"""Cleanup script for tables, external files, and temp directory

This script is intended to be run periodically to clean up the database tables
(pruning orphan entries), external files (deleting unreferenced files), and
temporary directory (deleting old files).
"""

import subprocess
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


def run_table_cleanups():
    """Run respective table cleanups"""
    Nwbfile().cleanup()  # cleanup 'raw' externals
    AnalysisNwbfile().cleanup()  # delete orphans, cleanup 'analysis' externals
    SpikeSorting().cleanup(verbose=False)  # remove unreferenced sorting_dir
    DecodingOutput().cleanup()  # remove `.nc` and `.pkl` files
    SpikeSortingRecording().cleanup(verbose=False)  # remove untracked folders


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
        print(f"Invalid temp_dir: {temp_dir}")
        return

    if dry_run:
        print(f"Dry run of delete files in {temp_dir} older than {days_old}d")
        return

    delete_cmd = f"find {temp_dir} -type f -mtime +{days_old} -delete"
    empty_dirs = f"find {temp_dir} -type d -empty -delete"
    subprocess_kwargs = dict(shell=True, check=True, executable="/bin/bash")
    try:
        subprocess.run(delete_cmd, **subprocess_kwargs)
        subprocess.run(empty_dirs, **subprocess_kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error cleaning temp_dir: {e}")


def main():
    print("Updating Spyglass versions table...")
    SpyglassVersions().fetch_from_pypi()
    print("Running table cleanups...")
    run_table_cleanups()
    print("Cleaning up external files...")
    cleanup_external_files()
    print("Cleaning up temporary directory...")
    cleanup_temp_dir(dry_run=False)


if __name__ == "__main__":
    main()
