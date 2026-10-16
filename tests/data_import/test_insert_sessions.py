import shutil
import warnings
from pathlib import Path

import pynwb
import pytest
from hdmf.backends.warnings import BrokenLinkWarning


@pytest.fixture(scope="session")
def copy_nwb_link_raw_ephys(data_import):
    from spyglass.data_import.insert_tools import (
        copy_nwb_link_raw_ephys,
    )  # noqa: E402

    return copy_nwb_link_raw_ephys


def test_open_path(mini_path, mini_open):
    this_acq = mini_open.acquisition
    assert "e-series" in this_acq, "Ephys link no longer exists"
    assert (
        str(mini_path) == this_acq["e-series"].data.file.filename
    ), "Path of ephys link is incorrect"


@pytest.mark.slow
def test_copy_link(mini_path, settings, mini_closed, copy_nwb_link_raw_ephys):
    """Test readability after moving the linking raw file, breaking link"""
    new_path = Path(settings.raw_dir) / "no_ephys.nwb"
    new_moved = Path(settings.temp_dir) / "no_ephys_moved.nwb"

    copy_nwb_link_raw_ephys(mini_path.name, new_path.name)
    shutil.move(new_path, new_moved)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with pynwb.NWBHDF5IO(path=str(new_moved), mode="r") as io:
            with pytest.warns(BrokenLinkWarning):
                nwb_acq = io.read().acquisition
    assert "e-series" not in nwb_acq, "Ephys link still exists after move"


@pytest.fixture
def two_raw_files(common, data_import):
    """Two minimal ingestible raw files.

    Built with pynwb's mocks rather than copied from the shared test file:
    this test only needs each file to reach Session, and a mock is kilobytes
    against the test file's tens of megabytes.
    """
    from pynwb.testing.mock.file import mock_NWBFile, mock_Subject

    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    names = ["multi_insert_a.nwb", "multi_insert_b.nwb"]
    paths = [Path(raw_dir) / name for name in names]

    for index, path in enumerate(paths):
        nwbfile = mock_NWBFile(
            identifier=f"multi_insert_{index}",
            session_description="Mock file for multi-file insert_sessions",
        )
        mock_Subject(nwbfile=nwbfile)
        path.unlink(missing_ok=True)
        with pynwb.NWBHDF5IO(path, mode="w") as io:
            io.write(nwbfile)

    copy_names = [get_nwb_copy_filename(name) for name in names]

    yield names, copy_names

    # Always clean up: a leftover Nwbfile entry gives populate() phantom work
    # on the next run. See tests/README.md on fixture teardown.
    for copy_name in copy_names:
        (common.Nwbfile & {"nwb_file_name": copy_name}).delete(safemode=False)
    for path in paths:
        path.unlink(missing_ok=True)


def test_insert_sessions_processes_every_file(two_raw_files, common):
    """A list argument ingests every file, not only the first.

    `insert_sessions` used to return from inside its loop, so files after the
    first were silently skipped.
    """
    from spyglass.data_import import insert_sessions

    names, copy_names = two_raw_files

    results = insert_sessions(names, raise_err=True)

    assert isinstance(results, list), "Expected one result per file"
    assert len(results) == len(names), (
        f"Processed {len(results)} of {len(names)} files -- a file after the "
        + "first was skipped"
    )

    for copy_name in copy_names:
        assert common.Nwbfile & {
            "nwb_file_name": copy_name
        }, f"{copy_name} was never registered"
        assert common.Session & {
            "nwb_file_name": copy_name
        }, f"{copy_name} registered but not ingested"


def test_raise_err_raises_once_every_file_has_been_planned(
    two_raw_files, common, monkeypatch, quiet_logs
):
    """`raise_err` is the caller's, and it waits for the whole list.

    `insert_sessions` is the only place that raises; everything below it
    reports through the plan. Raising at the first blocked file would make the
    second file's report depend on the first file's luck, so both are named.
    """
    from spyglass.data_import import insert_sessions
    from spyglass.utils.mixins.ingestion import IngestionMixin

    names, copy_names = two_raw_files
    original = IngestionMixin._parse

    def _parse(self, ctx):
        if self.camel_name == "Session":
            raise ValueError("induced failure")
        return original(self, ctx)

    monkeypatch.setattr(IngestionMixin, "_parse", _parse)

    with pytest.raises(ValueError) as err:
        insert_sessions(names, raise_err=True)

    message = str(err.value)
    for copy_name in copy_names:
        assert copy_name in message, (
            f"{copy_name} is missing from the exception; it raised before "
            + f"planning every file:\n{message}"
        )


def test_a_dry_run_reports_without_raising(
    two_raw_files, monkeypatch, quiet_logs
):
    """`dry_run` never raises, even asked to.

    Reporting without stopping is the whole job; an exception would deny the
    caller the report it asked for.
    """
    from spyglass.data_import import insert_sessions
    from spyglass.utils.mixins.ingestion import IngestionMixin

    names, _ = two_raw_files
    original = IngestionMixin._parse

    def _parse(self, ctx):
        if self.camel_name == "Session":
            raise ValueError("induced failure")
        return original(self, ctx)

    monkeypatch.setattr(IngestionMixin, "_parse", _parse)

    results = insert_sessions(names, raise_err=True, dry_run=True)

    assert len(results) == len(names), "One report per file, none skipped"
    assert all(results), "Each file blocked, so each result is truthy"


def test_a_skipped_file_still_returns_a_result(
    common, mini_copy_name, mini_insert
):
    """One result per file given, in the order given.

    A file already in `Nwbfile` is skipped, and used to contribute nothing to
    the returned list. With several files that left the list shorter than the
    input, so a caller could neither tell which file had been skipped nor line
    results up with what it asked for.
    """
    from spyglass.data_import import insert_sessions

    raw_name = mini_copy_name.replace("_.nwb", ".nwb")

    with pytest.warns(UserWarning, match="already in Nwbfile"):
        results = insert_sessions(raw_name)

    assert len(results) == 1, "One file in, one result out"
    (skipped,) = results
    assert skipped.nwb_file_name == mini_copy_name, "named by its copy"
    assert not skipped, "Skipping is not a failure, so the result is falsy"
    assert "already ingested" in skipped.report(), (
        "and the headline says so, in one line; a clean plan does not pad "
        "itself with empty sections"
    )

    # The override lives on the problem, not in the default report: an `info`
    # problem is a note that something was resolved, and those are opt-in.
    (note,) = [
        problem
        for problem in skipped.problems
        if problem.code == "file_already_registered"
    ]
    assert "reinsert" in note.message, "which says how to override it"
    assert "reinsert" in skipped.report(verbose=True)
